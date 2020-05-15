
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fc168159fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 06:12:30.314546
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 06:12:30.317229
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 06:12:30.319746
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 06:12:30.322497
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fc174171470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 352867.4688
Epoch 2/10

1/1 [==============================] - 0s 91ms/step - loss: 234440.1094
Epoch 3/10

1/1 [==============================] - 0s 86ms/step - loss: 126270.5625
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 57853.3672
Epoch 5/10

1/1 [==============================] - 0s 85ms/step - loss: 29181.9160
Epoch 6/10

1/1 [==============================] - 0s 81ms/step - loss: 16639.4121
Epoch 7/10

1/1 [==============================] - 0s 83ms/step - loss: 10544.9785
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 7281.4297
Epoch 9/10

1/1 [==============================] - 0s 84ms/step - loss: 5394.0171
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 4237.5859

  #### Inference Need return ypred, ytrue ######################### 
[[-0.1879951  10.999642    8.102377   11.636628   10.762582   10.156994
  11.83316    12.749531    9.366148   11.918738    7.2793045  10.845903
   7.9440827   9.719712   10.789444   12.590488   10.204909   10.278701
  12.861094   11.052843    9.884805   11.320904    9.427224   12.298735
  10.485814   12.424072    8.477196   10.068238    7.890599   10.993758
  10.675747   10.197408   10.548426   11.327742   12.131545   11.320211
  11.145818    8.895183   10.943463    8.945371   11.942681   11.312675
   9.402243   11.636043   10.046403   12.822814    9.6304865  10.95183
  10.446426   10.49183    11.241236   11.156486   11.415762    9.160095
  10.984309   11.230154    9.687305   11.073462    9.711      12.123531
  -0.36651355  1.3030739   0.2572791   1.7360218   0.1543898   0.24980474
   1.1887386  -1.8701762  -1.296829    0.5689399  -0.09517765  1.4462756
   1.4325833   1.5492644   1.5587795  -1.6554854  -1.9509778  -0.5584816
  -0.6634304  -0.18783826 -0.09787837  1.4697073   1.7525685  -0.92090374
   1.0043232   2.4504726   1.901064   -0.25879943 -1.0359372   0.9291572
   0.8816297   0.22563893 -0.28544313 -0.813404   -2.7574027  -1.5767198
   0.13957193 -1.0085816   0.5214361  -1.9406269  -2.4339943  -0.21213192
  -2.2168963   0.8952172   1.0739572   1.0150962  -1.2290436  -2.7680848
  -1.0111644  -0.25511894 -0.5566724   3.4263425   2.1472096   2.5090694
  -0.82254064 -1.3665091  -0.91741717  0.6599181   0.08007303 -0.72407603
   0.6004036  -1.5596006   0.04905176 -0.23600096  0.21841314 -0.9400319
   0.13512747 -0.11328272  0.16508284 -0.35186416 -0.23023134 -1.2836958
   0.4732768  -0.2762096   2.2628675  -0.7116973  -1.2942019   0.33417523
  -1.1282362   1.0287435  -1.2751229   0.6629107  -2.0957186   0.6266537
   1.785281   -0.6108073  -1.6083777  -1.9784634  -1.2571685   0.19411454
  -0.4680899   0.2728694  -1.2409196  -0.17516476 -2.6222475   2.3434064
  -0.8811583  -0.22361918 -1.6400983   0.8052664  -2.333661    0.6044619
  -0.09175126 -2.0455391   1.5831696   0.43188092 -0.2919506  -1.6681352
   1.1057692   0.21043685  0.41797763 -0.7370438   1.601979   -1.173501
  -2.4238353   1.3271449   1.3144557  -2.4879715  -0.845842    1.0101218
   0.4063567  10.07119    10.498689    9.229364   12.128452   11.140284
  11.074068   12.027119   10.854509    9.60934    12.241385   10.132318
   9.069134    9.503147   10.735887   11.658426   11.057268    9.957141
  10.173113   11.674231    9.784892   12.259665    9.69209    10.450113
  11.222039   11.137449    9.543987   13.092516    8.388094   10.4727955
  12.039532   11.749405   10.208578    8.965057   11.061396   12.315506
  12.74946    10.945912   11.155034    9.556831   10.211415    8.665091
  11.328655   10.522538    8.981574   11.3107195  12.031666   11.700059
  10.425908   10.594899   10.725508   11.261457    9.272797    8.163199
  12.245684    7.7816944  12.580566   11.201568   10.806969   11.233347
   0.5265974   2.3476758   0.8187284   2.1414914   1.1512743   0.13755661
   1.1521865   0.87185174  0.10084051  1.6486179   0.4795171   3.126
   0.6047403   2.1127193   1.1670957   1.6424465   3.1426601   2.20338
   1.6359315   0.48996508  3.0232658   0.41200697  0.5296821   1.4048997
   2.4811323   1.4576007   1.4735262   2.4651194   1.5514158   2.6990027
   0.29661864  2.4665823   2.7471418   0.62241     3.2657957   2.0503495
   1.0777493   0.27365553  0.5192811   2.4213648   0.30829746  0.2419442
   2.217028    0.9890689   0.74079823  0.24737424  0.07331431  0.7462927
   1.6674869   0.6141031   0.12644082  0.47604954  0.4904449   0.26958913
   2.2106156   1.7753513   0.18762583  3.164554    0.4425913   0.51441145
   3.4159012   0.12787038  0.0820179   0.21406353  1.1034882   0.47751266
   0.071392    1.4988778   0.12070644  0.16036975  0.25613356  0.8144846
   0.74098825  1.972708    1.6412656   2.166166    1.1356509   2.5299191
   3.528234    0.42786872  1.2706585   1.4784099   1.1750423   0.17725688
   0.45857394  0.20081043  0.71034217  0.8276503   1.4595877   0.1656608
   1.0869007   1.6308393   0.42067492  0.17906022  0.47713518  1.5979544
   1.8702415   0.32732052  0.18277752  0.31412446  4.2361393   1.0366052
   0.20535111  0.19785517  2.4588552   2.1721158   0.5073882   3.1550384
   1.8727256   0.7763429   1.9459577   0.28330958  0.11198974  2.3592157
   1.0790313   1.4541413   0.40009826  3.568708    0.70259637  0.24587679
   7.0096235  -8.301276   -9.069649  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 06:12:37.586071
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.4454
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 06:12:37.590156
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8384.92
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 06:12:37.593161
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.3656
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 06:12:37.595812
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -749.929
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140468312160128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140465782362744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140465782363248
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140465782363752
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140465782364256
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140465782364760

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fc16fffc278> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.504820
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.451016
grad_step = 000002, loss = 0.401386
grad_step = 000003, loss = 0.350687
grad_step = 000004, loss = 0.297392
grad_step = 000005, loss = 0.257799
grad_step = 000006, loss = 0.252809
grad_step = 000007, loss = 0.246770
grad_step = 000008, loss = 0.219316
grad_step = 000009, loss = 0.200888
grad_step = 000010, loss = 0.193075
grad_step = 000011, loss = 0.188064
grad_step = 000012, loss = 0.181667
grad_step = 000013, loss = 0.173342
grad_step = 000014, loss = 0.164309
grad_step = 000015, loss = 0.155485
grad_step = 000016, loss = 0.146902
grad_step = 000017, loss = 0.139929
grad_step = 000018, loss = 0.135239
grad_step = 000019, loss = 0.130732
grad_step = 000020, loss = 0.124413
grad_step = 000021, loss = 0.116709
grad_step = 000022, loss = 0.109263
grad_step = 000023, loss = 0.103220
grad_step = 000024, loss = 0.098716
grad_step = 000025, loss = 0.095055
grad_step = 000026, loss = 0.091104
grad_step = 000027, loss = 0.086340
grad_step = 000028, loss = 0.081334
grad_step = 000029, loss = 0.076843
grad_step = 000030, loss = 0.073094
grad_step = 000031, loss = 0.069835
grad_step = 000032, loss = 0.066632
grad_step = 000033, loss = 0.063195
grad_step = 000034, loss = 0.059628
grad_step = 000035, loss = 0.056263
grad_step = 000036, loss = 0.053251
grad_step = 000037, loss = 0.050471
grad_step = 000038, loss = 0.047800
grad_step = 000039, loss = 0.045236
grad_step = 000040, loss = 0.042799
grad_step = 000041, loss = 0.040481
grad_step = 000042, loss = 0.038303
grad_step = 000043, loss = 0.036287
grad_step = 000044, loss = 0.034359
grad_step = 000045, loss = 0.032424
grad_step = 000046, loss = 0.030521
grad_step = 000047, loss = 0.028777
grad_step = 000048, loss = 0.027212
grad_step = 000049, loss = 0.025720
grad_step = 000050, loss = 0.024232
grad_step = 000051, loss = 0.022798
grad_step = 000052, loss = 0.021483
grad_step = 000053, loss = 0.020277
grad_step = 000054, loss = 0.019128
grad_step = 000055, loss = 0.018027
grad_step = 000056, loss = 0.016983
grad_step = 000057, loss = 0.016001
grad_step = 000058, loss = 0.015072
grad_step = 000059, loss = 0.014184
grad_step = 000060, loss = 0.013332
grad_step = 000061, loss = 0.012530
grad_step = 000062, loss = 0.011805
grad_step = 000063, loss = 0.011142
grad_step = 000064, loss = 0.010490
grad_step = 000065, loss = 0.009852
grad_step = 000066, loss = 0.009269
grad_step = 000067, loss = 0.008745
grad_step = 000068, loss = 0.008247
grad_step = 000069, loss = 0.007768
grad_step = 000070, loss = 0.007309
grad_step = 000071, loss = 0.006882
grad_step = 000072, loss = 0.006494
grad_step = 000073, loss = 0.006132
grad_step = 000074, loss = 0.005787
grad_step = 000075, loss = 0.005468
grad_step = 000076, loss = 0.005177
grad_step = 000077, loss = 0.004901
grad_step = 000078, loss = 0.004640
grad_step = 000079, loss = 0.004402
grad_step = 000080, loss = 0.004184
grad_step = 000081, loss = 0.003980
grad_step = 000082, loss = 0.003789
grad_step = 000083, loss = 0.003612
grad_step = 000084, loss = 0.003457
grad_step = 000085, loss = 0.003315
grad_step = 000086, loss = 0.003182
grad_step = 000087, loss = 0.003062
grad_step = 000088, loss = 0.002953
grad_step = 000089, loss = 0.002854
grad_step = 000090, loss = 0.002767
grad_step = 000091, loss = 0.002687
grad_step = 000092, loss = 0.002616
grad_step = 000093, loss = 0.002552
grad_step = 000094, loss = 0.002495
grad_step = 000095, loss = 0.002445
grad_step = 000096, loss = 0.002399
grad_step = 000097, loss = 0.002360
grad_step = 000098, loss = 0.002325
grad_step = 000099, loss = 0.002295
grad_step = 000100, loss = 0.002268
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002245
grad_step = 000102, loss = 0.002225
grad_step = 000103, loss = 0.002207
grad_step = 000104, loss = 0.002191
grad_step = 000105, loss = 0.002178
grad_step = 000106, loss = 0.002167
grad_step = 000107, loss = 0.002157
grad_step = 000108, loss = 0.002148
grad_step = 000109, loss = 0.002141
grad_step = 000110, loss = 0.002134
grad_step = 000111, loss = 0.002129
grad_step = 000112, loss = 0.002124
grad_step = 000113, loss = 0.002119
grad_step = 000114, loss = 0.002116
grad_step = 000115, loss = 0.002112
grad_step = 000116, loss = 0.002109
grad_step = 000117, loss = 0.002106
grad_step = 000118, loss = 0.002103
grad_step = 000119, loss = 0.002100
grad_step = 000120, loss = 0.002097
grad_step = 000121, loss = 0.002095
grad_step = 000122, loss = 0.002092
grad_step = 000123, loss = 0.002090
grad_step = 000124, loss = 0.002087
grad_step = 000125, loss = 0.002084
grad_step = 000126, loss = 0.002082
grad_step = 000127, loss = 0.002079
grad_step = 000128, loss = 0.002076
grad_step = 000129, loss = 0.002073
grad_step = 000130, loss = 0.002070
grad_step = 000131, loss = 0.002067
grad_step = 000132, loss = 0.002064
grad_step = 000133, loss = 0.002061
grad_step = 000134, loss = 0.002057
grad_step = 000135, loss = 0.002054
grad_step = 000136, loss = 0.002051
grad_step = 000137, loss = 0.002047
grad_step = 000138, loss = 0.002044
grad_step = 000139, loss = 0.002040
grad_step = 000140, loss = 0.002037
grad_step = 000141, loss = 0.002033
grad_step = 000142, loss = 0.002029
grad_step = 000143, loss = 0.002026
grad_step = 000144, loss = 0.002022
grad_step = 000145, loss = 0.002018
grad_step = 000146, loss = 0.002014
grad_step = 000147, loss = 0.002010
grad_step = 000148, loss = 0.002006
grad_step = 000149, loss = 0.002003
grad_step = 000150, loss = 0.001999
grad_step = 000151, loss = 0.001994
grad_step = 000152, loss = 0.001990
grad_step = 000153, loss = 0.001986
grad_step = 000154, loss = 0.001982
grad_step = 000155, loss = 0.001978
grad_step = 000156, loss = 0.001974
grad_step = 000157, loss = 0.001970
grad_step = 000158, loss = 0.001966
grad_step = 000159, loss = 0.001961
grad_step = 000160, loss = 0.001957
grad_step = 000161, loss = 0.001951
grad_step = 000162, loss = 0.001946
grad_step = 000163, loss = 0.001942
grad_step = 000164, loss = 0.001938
grad_step = 000165, loss = 0.001934
grad_step = 000166, loss = 0.001930
grad_step = 000167, loss = 0.001927
grad_step = 000168, loss = 0.001922
grad_step = 000169, loss = 0.001917
grad_step = 000170, loss = 0.001911
grad_step = 000171, loss = 0.001905
grad_step = 000172, loss = 0.001899
grad_step = 000173, loss = 0.001894
grad_step = 000174, loss = 0.001888
grad_step = 000175, loss = 0.001883
grad_step = 000176, loss = 0.001878
grad_step = 000177, loss = 0.001872
grad_step = 000178, loss = 0.001867
grad_step = 000179, loss = 0.001863
grad_step = 000180, loss = 0.001866
grad_step = 000181, loss = 0.001897
grad_step = 000182, loss = 0.002000
grad_step = 000183, loss = 0.002150
grad_step = 000184, loss = 0.001985
grad_step = 000185, loss = 0.001834
grad_step = 000186, loss = 0.001942
grad_step = 000187, loss = 0.001932
grad_step = 000188, loss = 0.001826
grad_step = 000189, loss = 0.001880
grad_step = 000190, loss = 0.001892
grad_step = 000191, loss = 0.001815
grad_step = 000192, loss = 0.001838
grad_step = 000193, loss = 0.001862
grad_step = 000194, loss = 0.001808
grad_step = 000195, loss = 0.001806
grad_step = 000196, loss = 0.001836
grad_step = 000197, loss = 0.001805
grad_step = 000198, loss = 0.001783
grad_step = 000199, loss = 0.001808
grad_step = 000200, loss = 0.001801
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001769
grad_step = 000202, loss = 0.001778
grad_step = 000203, loss = 0.001789
grad_step = 000204, loss = 0.001764
grad_step = 000205, loss = 0.001752
grad_step = 000206, loss = 0.001764
grad_step = 000207, loss = 0.001760
grad_step = 000208, loss = 0.001740
grad_step = 000209, loss = 0.001734
grad_step = 000210, loss = 0.001740
grad_step = 000211, loss = 0.001736
grad_step = 000212, loss = 0.001721
grad_step = 000213, loss = 0.001711
grad_step = 000214, loss = 0.001712
grad_step = 000215, loss = 0.001711
grad_step = 000216, loss = 0.001704
grad_step = 000217, loss = 0.001691
grad_step = 000218, loss = 0.001682
grad_step = 000219, loss = 0.001678
grad_step = 000220, loss = 0.001675
grad_step = 000221, loss = 0.001673
grad_step = 000222, loss = 0.001668
grad_step = 000223, loss = 0.001665
grad_step = 000224, loss = 0.001659
grad_step = 000225, loss = 0.001653
grad_step = 000226, loss = 0.001649
grad_step = 000227, loss = 0.001650
grad_step = 000228, loss = 0.001654
grad_step = 000229, loss = 0.001669
grad_step = 000230, loss = 0.001700
grad_step = 000231, loss = 0.001768
grad_step = 000232, loss = 0.001793
grad_step = 000233, loss = 0.001788
grad_step = 000234, loss = 0.001681
grad_step = 000235, loss = 0.001595
grad_step = 000236, loss = 0.001606
grad_step = 000237, loss = 0.001677
grad_step = 000238, loss = 0.001711
grad_step = 000239, loss = 0.001644
grad_step = 000240, loss = 0.001585
grad_step = 000241, loss = 0.001576
grad_step = 000242, loss = 0.001593
grad_step = 000243, loss = 0.001621
grad_step = 000244, loss = 0.001632
grad_step = 000245, loss = 0.001609
grad_step = 000246, loss = 0.001556
grad_step = 000247, loss = 0.001543
grad_step = 000248, loss = 0.001559
grad_step = 000249, loss = 0.001572
grad_step = 000250, loss = 0.001591
grad_step = 000251, loss = 0.001591
grad_step = 000252, loss = 0.001566
grad_step = 000253, loss = 0.001543
grad_step = 000254, loss = 0.001533
grad_step = 000255, loss = 0.001517
grad_step = 000256, loss = 0.001508
grad_step = 000257, loss = 0.001513
grad_step = 000258, loss = 0.001520
grad_step = 000259, loss = 0.001529
grad_step = 000260, loss = 0.001557
grad_step = 000261, loss = 0.001616
grad_step = 000262, loss = 0.001691
grad_step = 000263, loss = 0.001819
grad_step = 000264, loss = 0.001779
grad_step = 000265, loss = 0.001644
grad_step = 000266, loss = 0.001493
grad_step = 000267, loss = 0.001529
grad_step = 000268, loss = 0.001647
grad_step = 000269, loss = 0.001613
grad_step = 000270, loss = 0.001512
grad_step = 000271, loss = 0.001482
grad_step = 000272, loss = 0.001550
grad_step = 000273, loss = 0.001568
grad_step = 000274, loss = 0.001487
grad_step = 000275, loss = 0.001470
grad_step = 000276, loss = 0.001522
grad_step = 000277, loss = 0.001519
grad_step = 000278, loss = 0.001473
grad_step = 000279, loss = 0.001456
grad_step = 000280, loss = 0.001480
grad_step = 000281, loss = 0.001495
grad_step = 000282, loss = 0.001469
grad_step = 000283, loss = 0.001443
grad_step = 000284, loss = 0.001449
grad_step = 000285, loss = 0.001466
grad_step = 000286, loss = 0.001461
grad_step = 000287, loss = 0.001439
grad_step = 000288, loss = 0.001431
grad_step = 000289, loss = 0.001437
grad_step = 000290, loss = 0.001443
grad_step = 000291, loss = 0.001440
grad_step = 000292, loss = 0.001426
grad_step = 000293, loss = 0.001416
grad_step = 000294, loss = 0.001417
grad_step = 000295, loss = 0.001421
grad_step = 000296, loss = 0.001424
grad_step = 000297, loss = 0.001426
grad_step = 000298, loss = 0.001426
grad_step = 000299, loss = 0.001420
grad_step = 000300, loss = 0.001414
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001407
grad_step = 000302, loss = 0.001400
grad_step = 000303, loss = 0.001394
grad_step = 000304, loss = 0.001391
grad_step = 000305, loss = 0.001388
grad_step = 000306, loss = 0.001386
grad_step = 000307, loss = 0.001385
grad_step = 000308, loss = 0.001385
grad_step = 000309, loss = 0.001387
grad_step = 000310, loss = 0.001393
grad_step = 000311, loss = 0.001414
grad_step = 000312, loss = 0.001463
grad_step = 000313, loss = 0.001594
grad_step = 000314, loss = 0.001773
grad_step = 000315, loss = 0.002091
grad_step = 000316, loss = 0.001872
grad_step = 000317, loss = 0.001523
grad_step = 000318, loss = 0.001393
grad_step = 000319, loss = 0.001629
grad_step = 000320, loss = 0.001707
grad_step = 000321, loss = 0.001409
grad_step = 000322, loss = 0.001468
grad_step = 000323, loss = 0.001664
grad_step = 000324, loss = 0.001444
grad_step = 000325, loss = 0.001405
grad_step = 000326, loss = 0.001550
grad_step = 000327, loss = 0.001424
grad_step = 000328, loss = 0.001383
grad_step = 000329, loss = 0.001485
grad_step = 000330, loss = 0.001408
grad_step = 000331, loss = 0.001366
grad_step = 000332, loss = 0.001436
grad_step = 000333, loss = 0.001396
grad_step = 000334, loss = 0.001354
grad_step = 000335, loss = 0.001399
grad_step = 000336, loss = 0.001385
grad_step = 000337, loss = 0.001345
grad_step = 000338, loss = 0.001371
grad_step = 000339, loss = 0.001377
grad_step = 000340, loss = 0.001343
grad_step = 000341, loss = 0.001347
grad_step = 000342, loss = 0.001364
grad_step = 000343, loss = 0.001348
grad_step = 000344, loss = 0.001334
grad_step = 000345, loss = 0.001343
grad_step = 000346, loss = 0.001343
grad_step = 000347, loss = 0.001332
grad_step = 000348, loss = 0.001331
grad_step = 000349, loss = 0.001332
grad_step = 000350, loss = 0.001329
grad_step = 000351, loss = 0.001327
grad_step = 000352, loss = 0.001323
grad_step = 000353, loss = 0.001319
grad_step = 000354, loss = 0.001320
grad_step = 000355, loss = 0.001320
grad_step = 000356, loss = 0.001314
grad_step = 000357, loss = 0.001309
grad_step = 000358, loss = 0.001311
grad_step = 000359, loss = 0.001311
grad_step = 000360, loss = 0.001306
grad_step = 000361, loss = 0.001302
grad_step = 000362, loss = 0.001302
grad_step = 000363, loss = 0.001301
grad_step = 000364, loss = 0.001298
grad_step = 000365, loss = 0.001296
grad_step = 000366, loss = 0.001295
grad_step = 000367, loss = 0.001292
grad_step = 000368, loss = 0.001289
grad_step = 000369, loss = 0.001288
grad_step = 000370, loss = 0.001288
grad_step = 000371, loss = 0.001286
grad_step = 000372, loss = 0.001283
grad_step = 000373, loss = 0.001282
grad_step = 000374, loss = 0.001280
grad_step = 000375, loss = 0.001278
grad_step = 000376, loss = 0.001275
grad_step = 000377, loss = 0.001273
grad_step = 000378, loss = 0.001272
grad_step = 000379, loss = 0.001270
grad_step = 000380, loss = 0.001268
grad_step = 000381, loss = 0.001266
grad_step = 000382, loss = 0.001264
grad_step = 000383, loss = 0.001263
grad_step = 000384, loss = 0.001260
grad_step = 000385, loss = 0.001259
grad_step = 000386, loss = 0.001257
grad_step = 000387, loss = 0.001257
grad_step = 000388, loss = 0.001259
grad_step = 000389, loss = 0.001266
grad_step = 000390, loss = 0.001285
grad_step = 000391, loss = 0.001334
grad_step = 000392, loss = 0.001409
grad_step = 000393, loss = 0.001561
grad_step = 000394, loss = 0.001620
grad_step = 000395, loss = 0.001668
grad_step = 000396, loss = 0.001439
grad_step = 000397, loss = 0.001269
grad_step = 000398, loss = 0.001263
grad_step = 000399, loss = 0.001379
grad_step = 000400, loss = 0.001466
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001336
grad_step = 000402, loss = 0.001236
grad_step = 000403, loss = 0.001275
grad_step = 000404, loss = 0.001335
grad_step = 000405, loss = 0.001308
grad_step = 000406, loss = 0.001228
grad_step = 000407, loss = 0.001247
grad_step = 000408, loss = 0.001301
grad_step = 000409, loss = 0.001278
grad_step = 000410, loss = 0.001227
grad_step = 000411, loss = 0.001225
grad_step = 000412, loss = 0.001248
grad_step = 000413, loss = 0.001249
grad_step = 000414, loss = 0.001226
grad_step = 000415, loss = 0.001218
grad_step = 000416, loss = 0.001218
grad_step = 000417, loss = 0.001222
grad_step = 000418, loss = 0.001225
grad_step = 000419, loss = 0.001217
grad_step = 000420, loss = 0.001198
grad_step = 000421, loss = 0.001196
grad_step = 000422, loss = 0.001209
grad_step = 000423, loss = 0.001212
grad_step = 000424, loss = 0.001201
grad_step = 000425, loss = 0.001194
grad_step = 000426, loss = 0.001191
grad_step = 000427, loss = 0.001187
grad_step = 000428, loss = 0.001184
grad_step = 000429, loss = 0.001187
grad_step = 000430, loss = 0.001192
grad_step = 000431, loss = 0.001188
grad_step = 000432, loss = 0.001184
grad_step = 000433, loss = 0.001182
grad_step = 000434, loss = 0.001180
grad_step = 000435, loss = 0.001175
grad_step = 000436, loss = 0.001171
grad_step = 000437, loss = 0.001170
grad_step = 000438, loss = 0.001169
grad_step = 000439, loss = 0.001166
grad_step = 000440, loss = 0.001164
grad_step = 000441, loss = 0.001163
grad_step = 000442, loss = 0.001162
grad_step = 000443, loss = 0.001161
grad_step = 000444, loss = 0.001159
grad_step = 000445, loss = 0.001157
grad_step = 000446, loss = 0.001157
grad_step = 000447, loss = 0.001158
grad_step = 000448, loss = 0.001159
grad_step = 000449, loss = 0.001164
grad_step = 000450, loss = 0.001178
grad_step = 000451, loss = 0.001207
grad_step = 000452, loss = 0.001270
grad_step = 000453, loss = 0.001352
grad_step = 000454, loss = 0.001480
grad_step = 000455, loss = 0.001475
grad_step = 000456, loss = 0.001420
grad_step = 000457, loss = 0.001231
grad_step = 000458, loss = 0.001147
grad_step = 000459, loss = 0.001207
grad_step = 000460, loss = 0.001289
grad_step = 000461, loss = 0.001309
grad_step = 000462, loss = 0.001201
grad_step = 000463, loss = 0.001147
grad_step = 000464, loss = 0.001192
grad_step = 000465, loss = 0.001223
grad_step = 000466, loss = 0.001183
grad_step = 000467, loss = 0.001137
grad_step = 000468, loss = 0.001166
grad_step = 000469, loss = 0.001206
grad_step = 000470, loss = 0.001175
grad_step = 000471, loss = 0.001134
grad_step = 000472, loss = 0.001137
grad_step = 000473, loss = 0.001161
grad_step = 000474, loss = 0.001161
grad_step = 000475, loss = 0.001145
grad_step = 000476, loss = 0.001139
grad_step = 000477, loss = 0.001141
grad_step = 000478, loss = 0.001132
grad_step = 000479, loss = 0.001126
grad_step = 000480, loss = 0.001134
grad_step = 000481, loss = 0.001139
grad_step = 000482, loss = 0.001131
grad_step = 000483, loss = 0.001116
grad_step = 000484, loss = 0.001113
grad_step = 000485, loss = 0.001119
grad_step = 000486, loss = 0.001117
grad_step = 000487, loss = 0.001111
grad_step = 000488, loss = 0.001111
grad_step = 000489, loss = 0.001114
grad_step = 000490, loss = 0.001112
grad_step = 000491, loss = 0.001105
grad_step = 000492, loss = 0.001102
grad_step = 000493, loss = 0.001104
grad_step = 000494, loss = 0.001102
grad_step = 000495, loss = 0.001098
grad_step = 000496, loss = 0.001097
grad_step = 000497, loss = 0.001099
grad_step = 000498, loss = 0.001099
grad_step = 000499, loss = 0.001097
grad_step = 000500, loss = 0.001096
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001097
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

  date_run                              2020-05-15 06:12:53.071659
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.204534
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 06:12:53.077139
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.101226
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 06:12:53.082872
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.127766
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 06:12:53.087302
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.538159
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
0   2020-05-15 06:12:30.314546  ...    mean_absolute_error
1   2020-05-15 06:12:30.317229  ...     mean_squared_error
2   2020-05-15 06:12:30.319746  ...  median_absolute_error
3   2020-05-15 06:12:30.322497  ...               r2_score
4   2020-05-15 06:12:37.586071  ...    mean_absolute_error
5   2020-05-15 06:12:37.590156  ...     mean_squared_error
6   2020-05-15 06:12:37.593161  ...  median_absolute_error
7   2020-05-15 06:12:37.595812  ...               r2_score
8   2020-05-15 06:12:53.071659  ...    mean_absolute_error
9   2020-05-15 06:12:53.077139  ...     mean_squared_error
10  2020-05-15 06:12:53.082872  ...  median_absolute_error
11  2020-05-15 06:12:53.087302  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5bd19070b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3629056/9912422 [00:00<00:00, 36204362.92it/s]9920512it [00:00, 30036080.54it/s]                             
0it [00:00, ?it/s]32768it [00:00, 554151.34it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475641.92it/s]1654784it [00:00, 11995062.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 159892.31it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b831d2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b828010b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b831d2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b827190f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b7ff954e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b7ff7e748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b831d2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b826d7710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5b7ff954e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5bd07dceb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2df5c1c1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f37f8c6d4cee16fd63cd479e17ca0cb0fa9f92e33fbf1995bb5a046b7c6ef5a2
  Stored in directory: /tmp/pip-ephem-wheel-cache-6yi5yrna/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2d8da176d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  720896/17464789 [>.............................] - ETA: 1s
 1671168/17464789 [=>............................] - ETA: 0s
 2686976/17464789 [===>..........................] - ETA: 0s
 3760128/17464789 [=====>........................] - ETA: 0s
 4825088/17464789 [=======>......................] - ETA: 0s
 6029312/17464789 [=========>....................] - ETA: 0s
 7282688/17464789 [===========>..................] - ETA: 0s
 8536064/17464789 [=============>................] - ETA: 0s
 9846784/17464789 [===============>..............] - ETA: 0s
11214848/17464789 [==================>...........] - ETA: 0s
12640256/17464789 [====================>.........] - ETA: 0s
14139392/17464789 [=======================>......] - ETA: 0s
15720448/17464789 [==========================>...] - ETA: 0s
17358848/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 06:14:16.485172: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 06:14:16.488923: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 06:14:16.489082: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55afc9c10460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 06:14:16.489096: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4673 - accuracy: 0.5130 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 3s - loss: 7.7140 - accuracy: 0.4969
12000/25000 [=============>................] - ETA: 2s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6863 - accuracy: 0.4987
15000/25000 [=================>............] - ETA: 2s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 1s - loss: 7.7050 - accuracy: 0.4975
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6421 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6449 - accuracy: 0.5014
25000/25000 [==============================] - 6s 244us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 06:14:28.236389
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 06:14:28.236389  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:21:02, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:34:38, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:39:27, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:10:04, 29.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<5:42:11, 41.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.65M/862M [00:01<3:58:12, 59.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:46:04, 85.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:55:37, 122kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:20:44, 174kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:01<56:14, 248kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<39:21, 353kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.9M/862M [00:02<27:27, 502kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<19:16, 713kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<13:28, 1.01MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<09:31, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<06:46, 1.99MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<05:08, 2.62MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<12:20:46, 18.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<8:39:05, 25.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<6:02:32, 37.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<4:18:59, 51.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<3:04:32, 72.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<2:09:54, 103kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:06<1:30:50, 147kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:11:55, 185kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<51:44, 257kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<36:30, 364kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<28:32, 464kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<22:39, 584kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<16:30, 801kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:10<11:40, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<12:46:19, 17.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<8:57:32, 24.5kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<6:15:53, 35.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<4:25:26, 49.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<3:08:32, 69.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<2:12:26, 98.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<1:32:50, 141kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<1:08:19, 191kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<49:10, 265kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<34:42, 375kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:18<27:12, 477kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<20:23, 636kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<14:31, 890kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<13:10, 979kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<11:57, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<09:03, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:20<06:27, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<35:30, 361kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<26:11, 490kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<18:35, 689kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<15:53, 803kB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<12:29, 1.02MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<09:04, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:10, 1.38MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:45, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:45, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:52, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:32, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<04:17, 2.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<07:05, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:18, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:43, 2.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:10, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:59, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:26, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:00, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:08, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:18, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:44, 2.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:08, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:35, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:13, 2.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:48, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:22, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:04, 3.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:39, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:26, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:02, 2.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:41, 3.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:56, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:49, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:05, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:19, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:42, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<04:18, 2.79MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:45, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:17, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<04:00, 2.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:07, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<03:53, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:25, 3.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<8:24:30, 23.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<5:53:32, 33.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<4:06:42, 47.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<3:00:05, 65.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<2:08:33, 91.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<1:30:28, 130kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<1:03:12, 186kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<55:55, 210kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<40:21, 291kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<28:30, 411kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<22:34, 517kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<18:16, 638kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<13:19, 875kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<09:27, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<11:45, 986kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:27, 1.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<06:54, 1.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:28, 1.54MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:40, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:59, 1.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:19, 2.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<37:24, 306kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<27:23, 418kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<19:25, 588kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<16:10, 704kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<12:28, 913kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<09:00, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<08:55, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:30, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:26, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:40, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:15, 2.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:34, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:16, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:53, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:33, 3.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<08:37, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:11, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:18, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:15, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:43, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:46, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:23, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:42, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:40, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:28, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:40, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:14, 2.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:27, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:51, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:30, 3.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<11:11, 963kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<08:57, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:29, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<07:01, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:11, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:35, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<04:01, 2.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<27:19, 389kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<20:14, 525kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<14:24, 736kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<12:29, 846kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<11:00, 960kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<08:15, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:53, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<37:23, 281kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<27:15, 385kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<19:18, 542kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<14:12, 735kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<7:04:59, 24.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<4:57:10, 35.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<3:29:07, 49.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<2:28:45, 69.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:44:37, 99.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<1:13:03, 141kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<59:46, 172kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<42:54, 240kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<30:12, 340kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<23:22, 438kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<18:26, 555kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<13:21, 765kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<09:26, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<13:15, 767kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<10:20, 983kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:29, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:32, 1.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<07:25, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:39, 1.78MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<04:04, 2.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<08:57, 1.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<07:20, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:23, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:02, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:21, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:54, 2.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:33, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<07:41, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:26, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:45, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:34, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:55, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:41, 2.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:50, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:23, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:18, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:16, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:11, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:01, 3.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<36:51, 261kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<26:46, 359kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<18:54, 507kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<15:24, 619kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<11:47, 809kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<08:26, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:04, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:39, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:53, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:35, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:55, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:38, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<03:21, 2.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<31:58, 292kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<23:10, 403kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<16:24, 568kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<11:33, 802kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<31:24, 295kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<22:46, 407kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<16:06, 574kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<11:23, 808kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<25:39, 359kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<19:54, 462kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<14:20, 641kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<10:08, 903kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<11:04, 825kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:41, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:18, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:30, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:29, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:04, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:54, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:20, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:00, 2.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<20:12, 442kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<6:02:24, 24.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<4:13:19, 35.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<2:58:06, 49.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<2:06:31, 70.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<1:28:50, 99.7kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<1:02:05, 142kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<46:52, 188kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<33:43, 261kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<23:46, 369kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<18:34, 470kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<14:51, 588kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<10:46, 809kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<07:38, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:53, 974kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<07:07, 1.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:11, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:36, 1.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:44, 1.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:28, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:12, 2.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<23:40, 360kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<17:28, 488kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<12:25, 684kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<10:36, 798kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:16, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:58, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:07, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:59, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:35, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:17, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:55, 932kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:06, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:09, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<03:42, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<35:16, 234kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<26:21, 313kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<18:47, 438kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<13:13, 620kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<12:34, 651kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:39, 847kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:01, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:37, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<07:17, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:40, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:11, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:01, 2.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<14:38, 550kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<12:52, 625kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<09:34, 839kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:55, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:55, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<17:52, 446kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<14:56, 534kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<10:58, 726kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<07:49, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<05:34, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<59:22, 133kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<43:58, 180kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<31:15, 253kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<21:59, 358kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<17:08, 457kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<14:07, 555kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<10:26, 750kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<07:25, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:19, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<07:14, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:31, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:03, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:29, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:14, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:10, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:01, 2.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:35, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:07, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:58, 2.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:09, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:40, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:27, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:21, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:40, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:26, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:16, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:37, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:42, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:41, 2.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:34, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:04, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:06, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:07, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:17, 3.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:53, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:07, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:42, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:29, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:54, 1.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:30, 1.60MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:31, 2.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:37, 2.73MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:34, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:26, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:34, 1.99MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:36, 2.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:26, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:28, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:34, 2.73MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:37, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:14, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:19, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:28, 2.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:28, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:09, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:15, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:28, 2.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:12, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:55, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:05, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:26, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<01:47, 3.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<11:20, 599kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<09:31, 714kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:04, 958kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:04, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:15, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:14, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:04, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:57, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:48, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<4:03:37, 27.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<2:50:35, 39.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<1:59:04, 55.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<1:24:48, 77.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<1:02:55, 105kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<44:49, 147kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<31:27, 209kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<22:01, 297kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<19:50, 329kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<15:29, 421kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<11:10, 583kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<07:54, 820kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<07:16, 888kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:42, 963kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:03, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<03:36, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:11, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:13, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:01, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:54, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:21, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:18, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:05, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:55, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:28, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:31, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:04, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:14, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:24, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:24, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:28, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:37, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:57, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:07, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:16, 2.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:24, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:33, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:30, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:32, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:47, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:07, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:12, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:18, 2.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:58, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:14, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:18, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:23, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:20, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:27, 1.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:27, 1.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:29, 2.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:06, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:27, 1.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:30, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:31, 2.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:52, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:02, 1.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:06, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:16, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:11, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:51, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:15, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:45, 2.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:26, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:42, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:00, 2.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:46, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:10, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:31, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:50, 2.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:29, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:21, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:18, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:24, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:25, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:55, 2.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:30, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:37, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:00, 2.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:26, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:30, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:42, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:57, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:46, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:25, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:22, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:25, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:34, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:44, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:55, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:06, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:15, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:26, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:41, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:57, 2.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:03, 1.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:48, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:54, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:04, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:23, 902kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:41, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:28, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:31, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:12, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:24, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:38, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:57, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:24, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:34, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:58, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:26, 3.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:47, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:31, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:40, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:55, 2.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<17:47, 258kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<13:16, 345kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<09:27, 484kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<06:37, 684kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<07:11, 629kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:07, 737kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:33, 990kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:12, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:54, 907kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:09, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:04, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:54, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:01, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:21, 1.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:42, 2.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:30, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:42, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:08, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:32, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:34, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:44, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:07, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:33, 2.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:13, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:11, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:39, 2.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:13, 3.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:44, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:30, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:03, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:44, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:15, 3.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:39, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:38, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:00, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:28, 2.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:24, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<2:21:15, 27.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<1:38:43, 39.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<1:08:10, 56.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<2:08:17, 29.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<1:31:12, 42.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<1:04:03, 59.7kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<44:41, 85.2kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<31:42, 119kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<22:53, 164kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<16:08, 232kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<11:45, 315kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<08:56, 413kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:24, 574kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<04:28, 813kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<14:17, 254kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<10:42, 339kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<07:37, 474kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<05:19, 671kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<06:02, 589kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:56, 721kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<03:36, 980kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:04, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:49, 1.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:08, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:05, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:36, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:09, 2.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:14, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:13, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:42, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:12, 2.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<16:50, 195kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<12:25, 264kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<08:49, 371kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<06:07, 526kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<08:16, 389kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<06:24, 501kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:37, 691kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:42, 849kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:11, 986kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:22, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:40, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:15, 586kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:16, 719kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:06, 984kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:12, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:46, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:32, 1.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:54, 1.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:20, 2.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<25:29, 115kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<18:24, 160kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<12:58, 226kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<08:57, 321kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<15:05, 190kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<11:07, 258kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<07:52, 363kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<05:30, 514kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<04:51, 578kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<03:56, 710kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:52, 972kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<02:01, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:00, 911kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:38, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:58, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:46, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:21, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:58, 2.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:50, 677kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:12, 811kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:20, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:39, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:01, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:53, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:26, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:23, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:27, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:48, 2.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:07, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<1:28:11, 27.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<1:01:29, 38.8kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<42:13, 55.4kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<31:46, 73.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<23:06, 101kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<16:22, 142kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<11:22, 202kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<08:22, 270kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<06:17, 359kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:28, 503kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<03:08, 707kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<02:13, 987kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:15, 516kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:52, 566kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:54, 753kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:04, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:29, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:51, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:10, 976kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:42, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:15, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:54, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:28, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:52, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:30, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:05, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:47, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:38, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:54, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:30, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:06, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:47, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:25, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:42, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:22, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:59, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:42, 2.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:28, 531kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:07, 590kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:19, 790kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:40, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:36<01:11, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:41, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:51, 957kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:27, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:02, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:45, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:42, 633kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:32, 673kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:53, 897kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:21, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:58, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:32, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:39, 990kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:16, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:55, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:40, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:16, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:29, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:10, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:50, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:56, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:10, 1.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:57, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:41, 2.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:49, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:06, 1.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:54, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:39, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:46, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:01, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:49, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:43, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:59, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:48, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:34, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:54, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:32, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:38, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:51, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:30, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:22, 2.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<02:18, 473kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:12, 493kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:40, 646kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:11, 896kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:49, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:09, 876kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:18, 776kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:02, 973kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:44, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:31, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:01, 925kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:08, 835kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:54, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:38, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:27, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:01, 855kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:08, 766kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:54, 971kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:38, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:44, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:52, 927kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:41, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:29, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:21, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:28, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:40, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:32, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:23, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:16, 2.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:55, 735kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:56, 715kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:43, 922kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:30, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:21, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:35, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:39, 914kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:31, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:22, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:15, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:50, 643kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:47, 669kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:36, 867kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:25, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:17, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:42, 664kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:40, 683kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:30, 896kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:21, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:14, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:21, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:25, 936kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:19, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:09, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:33, 592kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 615kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:23, 804kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:16, 1.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:10, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:26, 580kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:25, 605kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:19, 791kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:12, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:07, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:18, 606kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:18, 625kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:13, 816kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:04, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:15, 484kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:13, 540kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 712kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 988kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<02:00, 29.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<01:15, 42.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:30, 60.2kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 957/400000 [00:00<00:41, 9567.41it/s]  0%|          | 1888/400000 [00:00<00:41, 9488.51it/s]  1%|          | 2833/400000 [00:00<00:41, 9476.54it/s]  1%|          | 3845/400000 [00:00<00:41, 9659.09it/s]  1%|          | 4864/400000 [00:00<00:40, 9810.87it/s]  1%|▏         | 5790/400000 [00:00<00:40, 9638.66it/s]  2%|▏         | 6770/400000 [00:00<00:40, 9683.86it/s]  2%|▏         | 7702/400000 [00:00<00:40, 9570.40it/s]  2%|▏         | 8653/400000 [00:00<00:40, 9551.60it/s]  2%|▏         | 9571/400000 [00:01<00:41, 9381.79it/s]  3%|▎         | 10499/400000 [00:01<00:41, 9348.36it/s]  3%|▎         | 11457/400000 [00:01<00:41, 9414.30it/s]  3%|▎         | 12401/400000 [00:01<00:41, 9421.31it/s]  3%|▎         | 13354/400000 [00:01<00:40, 9450.63it/s]  4%|▎         | 14305/400000 [00:01<00:40, 9467.79it/s]  4%|▍         | 15272/400000 [00:01<00:40, 9525.98it/s]  4%|▍         | 16250/400000 [00:01<00:39, 9599.56it/s]  4%|▍         | 17209/400000 [00:01<00:40, 9520.66it/s]  5%|▍         | 18160/400000 [00:01<00:40, 9469.15it/s]  5%|▍         | 19107/400000 [00:02<00:40, 9386.38it/s]  5%|▌         | 20046/400000 [00:02<00:41, 9094.30it/s]  5%|▌         | 20958/400000 [00:02<00:42, 8838.87it/s]  5%|▌         | 21845/400000 [00:02<00:43, 8784.86it/s]  6%|▌         | 22726/400000 [00:02<00:43, 8660.09it/s]  6%|▌         | 23694/400000 [00:02<00:42, 8942.12it/s]  6%|▌         | 24592/400000 [00:02<00:42, 8829.52it/s]  6%|▋         | 25521/400000 [00:02<00:41, 8960.06it/s]  7%|▋         | 26450/400000 [00:02<00:41, 9053.93it/s]  7%|▋         | 27369/400000 [00:02<00:40, 9091.97it/s]  7%|▋         | 28322/400000 [00:03<00:40, 9216.86it/s]  7%|▋         | 29251/400000 [00:03<00:40, 9237.47it/s]  8%|▊         | 30193/400000 [00:03<00:39, 9289.25it/s]  8%|▊         | 31199/400000 [00:03<00:38, 9507.12it/s]  8%|▊         | 32152/400000 [00:03<00:38, 9498.30it/s]  8%|▊         | 33150/400000 [00:03<00:38, 9637.20it/s]  9%|▊         | 34116/400000 [00:03<00:38, 9544.10it/s]  9%|▉         | 35077/400000 [00:03<00:38, 9560.46it/s]  9%|▉         | 36034/400000 [00:03<00:38, 9470.51it/s]  9%|▉         | 36992/400000 [00:03<00:38, 9502.05it/s]  9%|▉         | 37943/400000 [00:04<00:38, 9306.48it/s] 10%|▉         | 38875/400000 [00:04<00:39, 9229.14it/s] 10%|▉         | 39799/400000 [00:04<00:39, 9119.73it/s] 10%|█         | 40713/400000 [00:04<00:39, 9023.60it/s] 10%|█         | 41617/400000 [00:04<00:41, 8706.82it/s] 11%|█         | 42573/400000 [00:04<00:39, 8943.93it/s] 11%|█         | 43488/400000 [00:04<00:39, 9004.16it/s] 11%|█         | 44392/400000 [00:04<00:39, 8974.67it/s] 11%|█▏        | 45293/400000 [00:04<00:39, 8985.09it/s] 12%|█▏        | 46233/400000 [00:04<00:38, 9101.20it/s] 12%|█▏        | 47145/400000 [00:05<00:38, 9071.26it/s] 12%|█▏        | 48063/400000 [00:05<00:38, 9102.27it/s] 12%|█▏        | 48974/400000 [00:05<00:39, 8986.46it/s] 12%|█▏        | 49874/400000 [00:05<00:39, 8933.57it/s] 13%|█▎        | 50768/400000 [00:05<00:39, 8843.86it/s] 13%|█▎        | 51681/400000 [00:05<00:39, 8924.92it/s] 13%|█▎        | 52598/400000 [00:05<00:38, 8995.57it/s] 13%|█▎        | 53601/400000 [00:05<00:37, 9277.23it/s] 14%|█▎        | 54532/400000 [00:05<00:37, 9283.88it/s] 14%|█▍        | 55463/400000 [00:05<00:37, 9160.97it/s] 14%|█▍        | 56400/400000 [00:06<00:37, 9221.18it/s] 14%|█▍        | 57346/400000 [00:06<00:36, 9290.52it/s] 15%|█▍        | 58279/400000 [00:06<00:36, 9301.55it/s] 15%|█▍        | 59210/400000 [00:06<00:37, 9106.94it/s] 15%|█▌        | 60123/400000 [00:06<00:37, 9073.76it/s] 15%|█▌        | 61053/400000 [00:06<00:37, 9138.76it/s] 15%|█▌        | 61968/400000 [00:06<00:37, 9126.46it/s] 16%|█▌        | 62882/400000 [00:06<00:37, 8970.44it/s] 16%|█▌        | 63781/400000 [00:06<00:37, 8950.67it/s] 16%|█▌        | 64774/400000 [00:07<00:36, 9222.04it/s] 16%|█▋        | 65716/400000 [00:07<00:36, 9280.45it/s] 17%|█▋        | 66683/400000 [00:07<00:35, 9393.68it/s] 17%|█▋        | 67683/400000 [00:07<00:34, 9566.28it/s] 17%|█▋        | 68642/400000 [00:07<00:34, 9559.72it/s] 17%|█▋        | 69600/400000 [00:07<00:34, 9500.97it/s] 18%|█▊        | 70552/400000 [00:07<00:35, 9314.04it/s] 18%|█▊        | 71552/400000 [00:07<00:34, 9509.12it/s] 18%|█▊        | 72540/400000 [00:07<00:34, 9616.83it/s] 18%|█▊        | 73504/400000 [00:07<00:34, 9502.72it/s] 19%|█▊        | 74481/400000 [00:08<00:33, 9579.44it/s] 19%|█▉        | 75441/400000 [00:08<00:33, 9560.94it/s] 19%|█▉        | 76398/400000 [00:08<00:34, 9472.99it/s] 19%|█▉        | 77355/400000 [00:08<00:33, 9499.77it/s] 20%|█▉        | 78323/400000 [00:08<00:33, 9552.58it/s] 20%|█▉        | 79334/400000 [00:08<00:33, 9712.70it/s] 20%|██        | 80307/400000 [00:08<00:33, 9669.29it/s] 20%|██        | 81326/400000 [00:08<00:32, 9818.44it/s] 21%|██        | 82333/400000 [00:08<00:32, 9890.30it/s] 21%|██        | 83323/400000 [00:08<00:32, 9833.06it/s] 21%|██        | 84330/400000 [00:09<00:31, 9902.24it/s] 21%|██▏       | 85321/400000 [00:09<00:32, 9744.40it/s] 22%|██▏       | 86297/400000 [00:09<00:32, 9732.55it/s] 22%|██▏       | 87294/400000 [00:09<00:31, 9801.83it/s] 22%|██▏       | 88275/400000 [00:09<00:32, 9519.13it/s] 22%|██▏       | 89230/400000 [00:09<00:32, 9443.06it/s] 23%|██▎       | 90176/400000 [00:09<00:32, 9421.13it/s] 23%|██▎       | 91170/400000 [00:09<00:32, 9569.25it/s] 23%|██▎       | 92161/400000 [00:09<00:31, 9668.95it/s] 23%|██▎       | 93130/400000 [00:09<00:32, 9444.91it/s] 24%|██▎       | 94077/400000 [00:10<00:32, 9389.41it/s] 24%|██▍       | 95018/400000 [00:10<00:32, 9276.73it/s] 24%|██▍       | 95948/400000 [00:10<00:33, 9188.62it/s] 24%|██▍       | 96872/400000 [00:10<00:32, 9202.01it/s] 24%|██▍       | 97793/400000 [00:10<00:33, 9104.37it/s] 25%|██▍       | 98745/400000 [00:10<00:32, 9223.19it/s] 25%|██▍       | 99669/400000 [00:10<00:32, 9153.35it/s] 25%|██▌       | 100586/400000 [00:10<00:32, 9139.88it/s] 25%|██▌       | 101517/400000 [00:10<00:32, 9189.18it/s] 26%|██▌       | 102437/400000 [00:10<00:32, 9155.30it/s] 26%|██▌       | 103417/400000 [00:11<00:31, 9338.06it/s] 26%|██▌       | 104355/400000 [00:11<00:31, 9343.16it/s] 26%|██▋       | 105291/400000 [00:11<00:31, 9272.05it/s] 27%|██▋       | 106246/400000 [00:11<00:31, 9350.09it/s] 27%|██▋       | 107182/400000 [00:11<00:31, 9177.15it/s] 27%|██▋       | 108149/400000 [00:11<00:31, 9317.16it/s] 27%|██▋       | 109097/400000 [00:11<00:31, 9364.70it/s] 28%|██▊       | 110035/400000 [00:11<00:31, 9319.67it/s] 28%|██▊       | 110968/400000 [00:11<00:31, 9278.47it/s] 28%|██▊       | 111897/400000 [00:12<00:31, 9138.06it/s] 28%|██▊       | 112812/400000 [00:12<00:31, 9106.75it/s] 28%|██▊       | 113738/400000 [00:12<00:31, 9150.40it/s] 29%|██▊       | 114654/400000 [00:12<00:31, 8995.44it/s] 29%|██▉       | 115564/400000 [00:12<00:31, 9024.74it/s] 29%|██▉       | 116468/400000 [00:12<00:31, 8924.54it/s] 29%|██▉       | 117362/400000 [00:12<00:31, 8926.03it/s] 30%|██▉       | 118256/400000 [00:12<00:31, 8918.02it/s] 30%|██▉       | 119149/400000 [00:12<00:31, 8915.43it/s] 30%|███       | 120068/400000 [00:12<00:31, 8994.34it/s] 30%|███       | 120968/400000 [00:13<00:31, 8951.30it/s] 30%|███       | 121872/400000 [00:13<00:30, 8976.29it/s] 31%|███       | 122774/400000 [00:13<00:30, 8988.88it/s] 31%|███       | 123674/400000 [00:13<00:30, 8974.50it/s] 31%|███       | 124585/400000 [00:13<00:30, 9013.85it/s] 31%|███▏      | 125487/400000 [00:13<00:31, 8594.99it/s] 32%|███▏      | 126364/400000 [00:13<00:31, 8643.99it/s] 32%|███▏      | 127232/400000 [00:13<00:31, 8617.64it/s] 32%|███▏      | 128111/400000 [00:13<00:31, 8666.33it/s] 32%|███▏      | 128996/400000 [00:13<00:31, 8717.94it/s] 32%|███▏      | 129869/400000 [00:14<00:31, 8673.66it/s] 33%|███▎      | 130771/400000 [00:14<00:30, 8773.54it/s] 33%|███▎      | 131655/400000 [00:14<00:30, 8792.69it/s] 33%|███▎      | 132535/400000 [00:14<00:30, 8681.75it/s] 33%|███▎      | 133404/400000 [00:14<00:30, 8644.48it/s] 34%|███▎      | 134286/400000 [00:14<00:30, 8695.74it/s] 34%|███▍      | 135178/400000 [00:14<00:30, 8761.40it/s] 34%|███▍      | 136063/400000 [00:14<00:30, 8785.42it/s] 34%|███▍      | 136950/400000 [00:14<00:29, 8810.50it/s] 34%|███▍      | 137853/400000 [00:14<00:29, 8873.90it/s] 35%|███▍      | 138741/400000 [00:15<00:29, 8777.02it/s] 35%|███▍      | 139620/400000 [00:15<00:29, 8747.52it/s] 35%|███▌      | 140509/400000 [00:15<00:29, 8788.15it/s] 35%|███▌      | 141396/400000 [00:15<00:29, 8810.11it/s] 36%|███▌      | 142278/400000 [00:15<00:29, 8744.89it/s] 36%|███▌      | 143183/400000 [00:15<00:29, 8832.87it/s] 36%|███▌      | 144100/400000 [00:15<00:28, 8930.69it/s] 36%|███▋      | 145002/400000 [00:15<00:28, 8955.15it/s] 36%|███▋      | 145900/400000 [00:15<00:28, 8962.19it/s] 37%|███▋      | 146814/400000 [00:15<00:28, 9013.40it/s] 37%|███▋      | 147716/400000 [00:16<00:28, 8940.52it/s] 37%|███▋      | 148617/400000 [00:16<00:28, 8960.83it/s] 37%|███▋      | 149543/400000 [00:16<00:27, 9048.26it/s] 38%|███▊      | 150463/400000 [00:16<00:27, 9091.02it/s] 38%|███▊      | 151373/400000 [00:16<00:27, 9052.49it/s] 38%|███▊      | 152298/400000 [00:16<00:27, 9108.17it/s] 38%|███▊      | 153254/400000 [00:16<00:26, 9237.66it/s] 39%|███▊      | 154179/400000 [00:16<00:26, 9239.20it/s] 39%|███▉      | 155104/400000 [00:16<00:26, 9129.40it/s] 39%|███▉      | 156043/400000 [00:16<00:26, 9203.76it/s] 39%|███▉      | 156964/400000 [00:17<00:26, 9129.30it/s] 39%|███▉      | 157878/400000 [00:17<00:26, 9051.59it/s] 40%|███▉      | 158784/400000 [00:17<00:27, 8724.85it/s] 40%|███▉      | 159745/400000 [00:17<00:26, 8971.11it/s] 40%|████      | 160684/400000 [00:17<00:26, 9092.17it/s] 40%|████      | 161605/400000 [00:17<00:26, 9127.01it/s] 41%|████      | 162577/400000 [00:17<00:25, 9296.78it/s] 41%|████      | 163534/400000 [00:17<00:25, 9376.60it/s] 41%|████      | 164474/400000 [00:17<00:26, 9003.05it/s] 41%|████▏     | 165397/400000 [00:17<00:25, 9068.22it/s] 42%|████▏     | 166308/400000 [00:18<00:25, 8999.43it/s] 42%|████▏     | 167214/400000 [00:18<00:25, 9015.40it/s] 42%|████▏     | 168118/400000 [00:18<00:26, 8758.39it/s] 42%|████▏     | 168997/400000 [00:18<00:28, 7969.40it/s] 42%|████▏     | 169905/400000 [00:18<00:27, 8271.51it/s] 43%|████▎     | 170850/400000 [00:18<00:26, 8590.85it/s] 43%|████▎     | 171785/400000 [00:18<00:25, 8804.27it/s] 43%|████▎     | 172676/400000 [00:18<00:27, 8230.16it/s] 43%|████▎     | 173591/400000 [00:18<00:26, 8486.04it/s] 44%|████▎     | 174478/400000 [00:19<00:26, 8595.82it/s] 44%|████▍     | 175392/400000 [00:19<00:25, 8750.97it/s] 44%|████▍     | 176307/400000 [00:19<00:25, 8864.17it/s] 44%|████▍     | 177271/400000 [00:19<00:24, 9081.79it/s] 45%|████▍     | 178185/400000 [00:19<00:24, 9056.19it/s] 45%|████▍     | 179095/400000 [00:19<00:24, 9040.21it/s] 45%|████▌     | 180044/400000 [00:19<00:23, 9168.67it/s] 45%|████▌     | 181043/400000 [00:19<00:23, 9400.44it/s] 45%|████▌     | 181987/400000 [00:19<00:23, 9186.70it/s] 46%|████▌     | 182909/400000 [00:19<00:23, 9185.75it/s] 46%|████▌     | 183852/400000 [00:20<00:23, 9257.50it/s] 46%|████▌     | 184819/400000 [00:20<00:22, 9375.59it/s] 46%|████▋     | 185768/400000 [00:20<00:22, 9408.15it/s] 47%|████▋     | 186768/400000 [00:20<00:22, 9576.58it/s] 47%|████▋     | 187728/400000 [00:20<00:22, 9478.06it/s] 47%|████▋     | 188678/400000 [00:20<00:22, 9370.42it/s] 47%|████▋     | 189621/400000 [00:20<00:22, 9387.90it/s] 48%|████▊     | 190569/400000 [00:20<00:22, 9414.41it/s] 48%|████▊     | 191545/400000 [00:20<00:21, 9512.58it/s] 48%|████▊     | 192497/400000 [00:20<00:21, 9492.04it/s] 48%|████▊     | 193447/400000 [00:21<00:21, 9448.24it/s] 49%|████▊     | 194393/400000 [00:21<00:22, 9343.46it/s] 49%|████▉     | 195332/400000 [00:21<00:21, 9356.82it/s] 49%|████▉     | 196352/400000 [00:21<00:21, 9594.25it/s] 49%|████▉     | 197317/400000 [00:21<00:21, 9608.88it/s] 50%|████▉     | 198280/400000 [00:21<00:21, 9591.56it/s] 50%|████▉     | 199299/400000 [00:21<00:20, 9761.58it/s] 50%|█████     | 200302/400000 [00:21<00:20, 9838.82it/s] 50%|█████     | 201287/400000 [00:21<00:20, 9659.62it/s] 51%|█████     | 202255/400000 [00:22<00:20, 9522.94it/s] 51%|█████     | 203236/400000 [00:22<00:20, 9607.16it/s] 51%|█████     | 204212/400000 [00:22<00:20, 9652.15it/s] 51%|█████▏    | 205204/400000 [00:22<00:20, 9729.96it/s] 52%|█████▏    | 206178/400000 [00:22<00:20, 9634.04it/s] 52%|█████▏    | 207149/400000 [00:22<00:19, 9655.19it/s] 52%|█████▏    | 208116/400000 [00:22<00:20, 9557.08it/s] 52%|█████▏    | 209116/400000 [00:22<00:19, 9685.39it/s] 53%|█████▎    | 210121/400000 [00:22<00:19, 9791.57it/s] 53%|█████▎    | 211102/400000 [00:22<00:19, 9681.29it/s] 53%|█████▎    | 212075/400000 [00:23<00:19, 9695.29it/s] 53%|█████▎    | 213046/400000 [00:23<00:19, 9639.44it/s] 54%|█████▎    | 214011/400000 [00:23<00:19, 9569.27it/s] 54%|█████▍    | 215032/400000 [00:23<00:18, 9750.19it/s] 54%|█████▍    | 216009/400000 [00:23<00:18, 9712.09it/s] 54%|█████▍    | 216982/400000 [00:23<00:18, 9645.94it/s] 54%|█████▍    | 217963/400000 [00:23<00:18, 9693.33it/s] 55%|█████▍    | 218933/400000 [00:23<00:18, 9650.65it/s] 55%|█████▍    | 219899/400000 [00:23<00:18, 9527.56it/s] 55%|█████▌    | 220853/400000 [00:23<00:19, 9328.34it/s] 55%|█████▌    | 221788/400000 [00:24<00:19, 9330.75it/s] 56%|█████▌    | 222723/400000 [00:24<00:19, 9056.01it/s] 56%|█████▌    | 223632/400000 [00:24<00:19, 9042.21it/s] 56%|█████▌    | 224568/400000 [00:24<00:19, 9133.70it/s] 56%|█████▋    | 225529/400000 [00:24<00:18, 9269.23it/s] 57%|█████▋    | 226458/400000 [00:24<00:18, 9254.64it/s] 57%|█████▋    | 227385/400000 [00:24<00:18, 9251.59it/s] 57%|█████▋    | 228359/400000 [00:24<00:18, 9390.90it/s] 57%|█████▋    | 229300/400000 [00:24<00:18, 9240.96it/s] 58%|█████▊    | 230307/400000 [00:24<00:17, 9474.92it/s] 58%|█████▊    | 231257/400000 [00:25<00:17, 9389.06it/s] 58%|█████▊    | 232198/400000 [00:25<00:18, 9008.80it/s] 58%|█████▊    | 233104/400000 [00:25<00:18, 8985.83it/s] 59%|█████▊    | 234074/400000 [00:25<00:18, 9187.41it/s] 59%|█████▉    | 235000/400000 [00:25<00:17, 9205.48it/s] 59%|█████▉    | 235923/400000 [00:25<00:18, 9070.00it/s] 59%|█████▉    | 236869/400000 [00:25<00:17, 9182.15it/s] 59%|█████▉    | 237839/400000 [00:25<00:17, 9329.75it/s] 60%|█████▉    | 238872/400000 [00:25<00:16, 9607.52it/s] 60%|█████▉    | 239899/400000 [00:25<00:16, 9795.57it/s] 60%|██████    | 240882/400000 [00:26<00:16, 9734.85it/s] 60%|██████    | 241858/400000 [00:26<00:16, 9512.70it/s] 61%|██████    | 242813/400000 [00:26<00:17, 9167.50it/s] 61%|██████    | 243735/400000 [00:26<00:17, 9165.62it/s] 61%|██████    | 244673/400000 [00:26<00:16, 9227.55it/s] 61%|██████▏   | 245599/400000 [00:26<00:16, 9220.45it/s] 62%|██████▏   | 246523/400000 [00:26<00:16, 9168.73it/s] 62%|██████▏   | 247486/400000 [00:26<00:16, 9300.79it/s] 62%|██████▏   | 248439/400000 [00:26<00:16, 9367.97it/s] 62%|██████▏   | 249399/400000 [00:27<00:15, 9435.29it/s] 63%|██████▎   | 250392/400000 [00:27<00:15, 9575.90it/s] 63%|██████▎   | 251351/400000 [00:27<00:15, 9460.89it/s] 63%|██████▎   | 252299/400000 [00:27<00:15, 9298.31it/s] 63%|██████▎   | 253231/400000 [00:27<00:15, 9252.53it/s] 64%|██████▎   | 254193/400000 [00:27<00:15, 9357.70it/s] 64%|██████▍   | 255172/400000 [00:27<00:15, 9481.69it/s] 64%|██████▍   | 256122/400000 [00:27<00:15, 9456.43it/s] 64%|██████▍   | 257069/400000 [00:27<00:15, 9452.90it/s] 65%|██████▍   | 258015/400000 [00:27<00:15, 9328.00it/s] 65%|██████▍   | 258949/400000 [00:28<00:15, 9292.06it/s] 65%|██████▍   | 259879/400000 [00:28<00:15, 9051.91it/s] 65%|██████▌   | 260786/400000 [00:28<00:15, 9051.09it/s] 65%|██████▌   | 261699/400000 [00:28<00:15, 9072.38it/s] 66%|██████▌   | 262624/400000 [00:28<00:15, 9124.54it/s] 66%|██████▌   | 263632/400000 [00:28<00:14, 9389.78it/s] 66%|██████▌   | 264652/400000 [00:28<00:14, 9617.12it/s] 66%|██████▋   | 265623/400000 [00:28<00:13, 9644.19it/s] 67%|██████▋   | 266590/400000 [00:28<00:13, 9610.36it/s] 67%|██████▋   | 267553/400000 [00:28<00:13, 9595.14it/s] 67%|██████▋   | 268524/400000 [00:29<00:13, 9626.70it/s] 67%|██████▋   | 269491/400000 [00:29<00:13, 9636.37it/s] 68%|██████▊   | 270456/400000 [00:29<00:13, 9491.93it/s] 68%|██████▊   | 271461/400000 [00:29<00:13, 9650.56it/s] 68%|██████▊   | 272428/400000 [00:29<00:13, 9643.56it/s] 68%|██████▊   | 273394/400000 [00:29<00:13, 9515.74it/s] 69%|██████▊   | 274347/400000 [00:29<00:13, 9455.71it/s] 69%|██████▉   | 275294/400000 [00:29<00:13, 9257.32it/s] 69%|██████▉   | 276235/400000 [00:29<00:13, 9300.77it/s] 69%|██████▉   | 277167/400000 [00:29<00:13, 9173.81it/s] 70%|██████▉   | 278091/400000 [00:30<00:13, 9191.66it/s] 70%|██████▉   | 279064/400000 [00:30<00:12, 9346.05it/s] 70%|███████   | 280000/400000 [00:30<00:13, 9230.54it/s] 70%|███████   | 280925/400000 [00:30<00:13, 9155.98it/s] 70%|███████   | 281842/400000 [00:30<00:13, 9008.75it/s] 71%|███████   | 282745/400000 [00:30<00:13, 8994.60it/s] 71%|███████   | 283646/400000 [00:30<00:13, 8938.07it/s] 71%|███████   | 284541/400000 [00:30<00:12, 8941.22it/s] 71%|███████▏  | 285479/400000 [00:30<00:12, 9066.51it/s] 72%|███████▏  | 286387/400000 [00:30<00:12, 9065.44it/s] 72%|███████▏  | 287335/400000 [00:31<00:12, 9184.14it/s] 72%|███████▏  | 288266/400000 [00:31<00:12, 9218.83it/s] 72%|███████▏  | 289189/400000 [00:31<00:12, 9067.64it/s] 73%|███████▎  | 290141/400000 [00:31<00:11, 9197.27it/s] 73%|███████▎  | 291062/400000 [00:31<00:11, 9155.44it/s] 73%|███████▎  | 291979/400000 [00:31<00:11, 9148.84it/s] 73%|███████▎  | 292895/400000 [00:31<00:11, 9131.30it/s] 73%|███████▎  | 293809/400000 [00:31<00:11, 9021.44it/s] 74%|███████▎  | 294712/400000 [00:31<00:11, 8940.78it/s] 74%|███████▍  | 295613/400000 [00:31<00:11, 8959.61it/s] 74%|███████▍  | 296510/400000 [00:32<00:11, 8708.36it/s] 74%|███████▍  | 297383/400000 [00:32<00:12, 8474.14it/s] 75%|███████▍  | 298234/400000 [00:32<00:12, 7964.86it/s] 75%|███████▍  | 299124/400000 [00:32<00:12, 8223.35it/s] 75%|███████▍  | 299996/400000 [00:32<00:11, 8365.68it/s] 75%|███████▌  | 300914/400000 [00:32<00:11, 8593.41it/s] 75%|███████▌  | 301850/400000 [00:32<00:11, 8808.03it/s] 76%|███████▌  | 302743/400000 [00:32<00:10, 8841.86it/s] 76%|███████▌  | 303639/400000 [00:32<00:10, 8874.77it/s] 76%|███████▌  | 304546/400000 [00:33<00:10, 8932.32it/s] 76%|███████▋  | 305445/400000 [00:33<00:10, 8947.16it/s] 77%|███████▋  | 306366/400000 [00:33<00:10, 9021.68it/s] 77%|███████▋  | 307303/400000 [00:33<00:10, 9122.56it/s] 77%|███████▋  | 308219/400000 [00:33<00:10, 9131.83it/s] 77%|███████▋  | 309133/400000 [00:33<00:09, 9089.25it/s] 78%|███████▊  | 310057/400000 [00:33<00:09, 9132.51it/s] 78%|███████▊  | 310977/400000 [00:33<00:09, 9151.94it/s] 78%|███████▊  | 311893/400000 [00:33<00:09, 9065.01it/s] 78%|███████▊  | 312823/400000 [00:33<00:09, 9132.39it/s] 78%|███████▊  | 313778/400000 [00:34<00:09, 9251.39it/s] 79%|███████▊  | 314761/400000 [00:34<00:09, 9416.84it/s] 79%|███████▉  | 315708/400000 [00:34<00:08, 9432.51it/s] 79%|███████▉  | 316654/400000 [00:34<00:08, 9438.33it/s] 79%|███████▉  | 317599/400000 [00:34<00:08, 9272.31it/s] 80%|███████▉  | 318528/400000 [00:34<00:08, 9244.03it/s] 80%|███████▉  | 319454/400000 [00:34<00:08, 9119.18it/s] 80%|████████  | 320421/400000 [00:34<00:08, 9276.42it/s] 80%|████████  | 321354/400000 [00:34<00:08, 9290.66it/s] 81%|████████  | 322299/400000 [00:34<00:08, 9336.93it/s] 81%|████████  | 323302/400000 [00:35<00:08, 9532.25it/s] 81%|████████  | 324257/400000 [00:35<00:07, 9501.34it/s] 81%|████████▏ | 325209/400000 [00:35<00:08, 9310.80it/s] 82%|████████▏ | 326142/400000 [00:35<00:08, 9225.02it/s] 82%|████████▏ | 327066/400000 [00:35<00:08, 9048.40it/s] 82%|████████▏ | 327977/400000 [00:35<00:07, 9066.46it/s] 82%|████████▏ | 328885/400000 [00:35<00:07, 9039.06it/s] 82%|████████▏ | 329790/400000 [00:35<00:07, 8889.03it/s] 83%|████████▎ | 330681/400000 [00:35<00:07, 8830.56it/s] 83%|████████▎ | 331600/400000 [00:35<00:07, 8932.99it/s] 83%|████████▎ | 332520/400000 [00:36<00:07, 9010.38it/s] 83%|████████▎ | 333422/400000 [00:36<00:07, 8977.15it/s] 84%|████████▎ | 334321/400000 [00:36<00:07, 8969.12it/s] 84%|████████▍ | 335221/400000 [00:36<00:07, 8975.77it/s] 84%|████████▍ | 336119/400000 [00:36<00:07, 8832.15it/s] 84%|████████▍ | 337021/400000 [00:36<00:07, 8885.86it/s] 84%|████████▍ | 337952/400000 [00:36<00:06, 8994.49it/s] 85%|████████▍ | 338863/400000 [00:36<00:06, 9026.66it/s] 85%|████████▍ | 339767/400000 [00:36<00:06, 8970.03it/s] 85%|████████▌ | 340722/400000 [00:36<00:06, 9135.33it/s] 85%|████████▌ | 341722/400000 [00:37<00:06, 9375.79it/s] 86%|████████▌ | 342738/400000 [00:37<00:05, 9597.80it/s] 86%|████████▌ | 343729/400000 [00:37<00:05, 9688.15it/s] 86%|████████▌ | 344701/400000 [00:37<00:05, 9425.36it/s] 86%|████████▋ | 345647/400000 [00:37<00:05, 9335.26it/s] 87%|████████▋ | 346583/400000 [00:37<00:05, 9329.46it/s] 87%|████████▋ | 347518/400000 [00:37<00:05, 9334.28it/s] 87%|████████▋ | 348453/400000 [00:37<00:05, 9298.39it/s] 87%|████████▋ | 349384/400000 [00:37<00:05, 9204.70it/s] 88%|████████▊ | 350328/400000 [00:38<00:05, 9272.81it/s] 88%|████████▊ | 351309/400000 [00:38<00:05, 9424.76it/s] 88%|████████▊ | 352253/400000 [00:38<00:05, 9372.79it/s] 88%|████████▊ | 353213/400000 [00:38<00:04, 9437.69it/s] 89%|████████▊ | 354158/400000 [00:38<00:04, 9182.10it/s] 89%|████████▉ | 355116/400000 [00:38<00:04, 9295.72it/s] 89%|████████▉ | 356048/400000 [00:38<00:04, 9152.64it/s] 89%|████████▉ | 356965/400000 [00:38<00:04, 9134.75it/s] 89%|████████▉ | 357880/400000 [00:38<00:04, 9116.28it/s] 90%|████████▉ | 358793/400000 [00:38<00:04, 9054.63it/s] 90%|████████▉ | 359781/400000 [00:39<00:04, 9286.16it/s] 90%|█████████ | 360714/400000 [00:39<00:04, 9298.32it/s] 90%|█████████ | 361646/400000 [00:39<00:04, 9248.82it/s] 91%|█████████ | 362615/400000 [00:39<00:03, 9375.93it/s] 91%|█████████ | 363554/400000 [00:39<00:03, 9214.80it/s] 91%|█████████ | 364477/400000 [00:39<00:03, 9174.65it/s] 91%|█████████▏| 365396/400000 [00:39<00:03, 9041.21it/s] 92%|█████████▏| 366302/400000 [00:39<00:03, 8955.57it/s] 92%|█████████▏| 367209/400000 [00:39<00:03, 8988.65it/s] 92%|█████████▏| 368170/400000 [00:39<00:03, 9165.12it/s] 92%|█████████▏| 369149/400000 [00:40<00:03, 9344.00it/s] 93%|█████████▎| 370129/400000 [00:40<00:03, 9475.17it/s] 93%|█████████▎| 371079/400000 [00:40<00:03, 9376.44it/s] 93%|█████████▎| 372019/400000 [00:40<00:03, 9260.35it/s] 93%|█████████▎| 372959/400000 [00:40<00:02, 9299.71it/s] 93%|█████████▎| 373890/400000 [00:40<00:02, 9182.20it/s] 94%|█████████▎| 374810/400000 [00:40<00:02, 9034.94it/s] 94%|█████████▍| 375715/400000 [00:40<00:02, 8988.95it/s] 94%|█████████▍| 376615/400000 [00:40<00:02, 8611.13it/s] 94%|█████████▍| 377494/400000 [00:40<00:02, 8661.50it/s] 95%|█████████▍| 378364/400000 [00:41<00:02, 8500.83it/s] 95%|█████████▍| 379217/400000 [00:41<00:02, 8212.44it/s] 95%|█████████▌| 380096/400000 [00:41<00:02, 8377.47it/s] 95%|█████████▌| 380956/400000 [00:41<00:02, 8442.19it/s] 95%|█████████▌| 381803/400000 [00:41<00:02, 8243.47it/s] 96%|█████████▌| 382631/400000 [00:41<00:02, 8196.59it/s] 96%|█████████▌| 383498/400000 [00:41<00:01, 8330.77it/s] 96%|█████████▌| 384378/400000 [00:41<00:01, 8463.50it/s] 96%|█████████▋| 385227/400000 [00:41<00:01, 8184.91it/s] 97%|█████████▋| 386049/400000 [00:42<00:01, 8113.85it/s] 97%|█████████▋| 386903/400000 [00:42<00:01, 8235.12it/s] 97%|█████████▋| 387780/400000 [00:42<00:01, 8386.99it/s] 97%|█████████▋| 388621/400000 [00:42<00:01, 8357.68it/s] 97%|█████████▋| 389459/400000 [00:42<00:01, 8301.06it/s] 98%|█████████▊| 390326/400000 [00:42<00:01, 8405.94it/s] 98%|█████████▊| 391168/400000 [00:42<00:01, 7629.70it/s] 98%|█████████▊| 392050/400000 [00:42<00:01, 7949.31it/s] 98%|█████████▊| 392933/400000 [00:42<00:00, 8193.74it/s] 98%|█████████▊| 393831/400000 [00:42<00:00, 8412.04it/s] 99%|█████████▊| 394737/400000 [00:43<00:00, 8595.57it/s] 99%|█████████▉| 395649/400000 [00:43<00:00, 8743.51it/s] 99%|█████████▉| 396562/400000 [00:43<00:00, 8855.70it/s] 99%|█████████▉| 397490/400000 [00:43<00:00, 8976.84it/s]100%|█████████▉| 398392/400000 [00:43<00:00, 8962.28it/s]100%|█████████▉| 399291/400000 [00:43<00:00, 8943.30it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9157.38it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f51ed9205c0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01080761634570164 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.011578025825844959 	 Accuracy: 49

  model saves at 49% accuracy 

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
2020-05-15 06:23:28.489559: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 06:23:28.494868: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 06:23:28.495122: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55807a578e20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 06:23:28.495177: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5187349be0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.3753 - accuracy: 0.5190
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4826 - accuracy: 0.5120
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5532 - accuracy: 0.5074
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5772 - accuracy: 0.5058
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5615 - accuracy: 0.5069
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6015 - accuracy: 0.5042
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6189 - accuracy: 0.5031
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
11000/25000 [============>.................] - ETA: 3s - loss: 7.6680 - accuracy: 0.4999
12000/25000 [=============>................] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6702 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6707 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f51429bc668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f515da97358> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3467 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.3448 - val_crf_viterbi_accuracy: 0.6533

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f42d4356f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 07:13:02.037160
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 07:13:02.039938
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 07:13:02.042203
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 07:13:02.045260
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f42e0120438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355987.0938
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 304714.5938
Epoch 3/10

1/1 [==============================] - 0s 88ms/step - loss: 217962.1562
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 136363.3906
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 82467.2109
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 50057.3906
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 31976.6191
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 21452.9258
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 15100.8984
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 11258.3125

  #### Inference Need return ypred, ytrue ######################### 
[[-1.2775661e-01 -8.3037961e-01  2.4677572e-01 -9.7881484e-01
   7.0073187e-02  5.8665115e-01  2.5763515e-01  8.1780958e-01
   2.8303802e-01  1.6015244e+00 -8.5112321e-01  1.5693953e+00
   9.7243840e-01  6.7217427e-01 -1.3699939e+00 -9.6432492e-02
   5.8475542e-01  1.4221702e+00  2.6689130e-01  1.5260191e+00
  -3.7220076e-01 -1.6342759e-02 -2.7369681e-01  1.0066508e+00
   3.0190542e-02  1.6627881e-01 -9.5048279e-02 -4.0528917e-01
   5.1024145e-01 -8.3184588e-01 -5.0633699e-01  8.0344009e-01
  -1.2146381e+00 -9.3381524e-02  9.8948061e-01 -2.5480789e-01
   5.1021761e-01  6.4965630e-01  2.5741255e-01  7.3223859e-01
  -3.5800004e-01 -1.0954069e+00  4.1024554e-01 -1.0923017e+00
   1.5570134e-01 -4.4325367e-01  8.8103110e-01  1.8454647e-01
   6.9850683e-03 -1.0492361e+00 -3.2678407e-01 -6.3117588e-01
   2.7984002e-01  1.2935022e-01 -2.3692435e-01 -5.3899360e-01
   6.2907493e-01 -9.1907543e-01 -9.2591983e-01 -8.8322389e-01
   9.6398950e-01 -2.7841526e-01  1.4683196e-01 -9.3999660e-01
  -1.3973945e-01 -4.0664384e-01 -1.7397317e-01 -2.1609783e-02
   6.3235641e-01  7.7885258e-01  2.3293909e-01  5.9816682e-01
   3.0730510e-01 -8.0483836e-01  3.9375469e-01  3.4130612e-01
   5.9731925e-01 -4.3204203e-03  2.4562272e-01 -5.1638246e-02
  -3.1330189e-01  4.5111960e-01  3.5369354e-01 -8.3180833e-01
   2.2114761e-02 -2.8170907e-01 -1.0767397e+00 -1.7139813e-01
  -1.0702976e+00  1.2965860e+00 -3.5399577e-01  3.7065852e-01
   3.8149267e-01 -1.1685991e-01 -4.9339727e-01 -4.6327627e-01
   6.7971921e-01 -2.3364267e-01 -1.6476440e+00 -5.0814968e-01
   5.9634256e-01  4.3865734e-01 -1.0945164e+00 -4.6479028e-01
   6.8996894e-01  1.2617554e-01 -3.7340879e-01 -2.8924835e-01
   9.9563110e-01 -1.4920239e-01 -3.9069343e-01 -6.1057073e-01
   8.3302945e-01 -3.9379466e-01 -1.6506845e-01  6.4613390e-01
   5.4538012e-01 -1.4109191e-01  2.7660072e-01  8.8419527e-01
  -9.0527095e-02  4.5438166e+00  4.7821126e+00  7.1413937e+00
   5.1340261e+00  6.4216452e+00  4.7461710e+00  6.6713123e+00
   5.6312704e+00  6.1441851e+00  5.1335158e+00  5.9022784e+00
   6.1926079e+00  6.9167657e+00  5.6728969e+00  5.4898424e+00
   6.1214895e+00  6.6761937e+00  5.8471198e+00  4.6235595e+00
   5.3229165e+00  5.9931660e+00  5.7505369e+00  4.5021968e+00
   5.4697762e+00  6.2995071e+00  4.6735353e+00  6.3299565e+00
   5.2783928e+00  7.0531769e+00  5.8805289e+00  5.6353951e+00
   4.3854980e+00  5.1521187e+00  5.2421651e+00  5.7115831e+00
   5.4540567e+00  4.2759385e+00  5.4382911e+00  4.6624465e+00
   5.7968574e+00  6.3790622e+00  5.3071012e+00  5.7226291e+00
   6.1937900e+00  5.4156723e+00  4.4809141e+00  4.5828123e+00
   4.8622208e+00  6.3836908e+00  5.7351713e+00  6.3454742e+00
   5.1368046e+00  5.4457722e+00  5.4621615e+00  4.5524015e+00
   6.1325412e+00  6.7545204e+00  4.8971682e+00  5.8188167e+00
   2.4639649e+00  5.0965798e-01  6.1152864e-01  7.6611388e-01
   1.8125449e+00  6.6802049e-01  1.6334012e+00  1.9014249e+00
   1.5849767e+00  1.5428276e+00  3.5799301e-01  1.0348561e+00
   2.3773046e+00  2.7599978e-01  6.3640285e-01  8.4392238e-01
   1.2140957e+00  8.0618048e-01  1.6538370e+00  7.8517896e-01
   1.4228587e+00  1.3813622e+00  2.9228830e+00  9.1541016e-01
   1.6570470e+00  8.3008838e-01  6.1424726e-01  4.3261343e-01
   3.2169020e-01  7.9985368e-01  9.2374599e-01  8.8819015e-01
   4.0786290e-01  2.1888483e+00  4.5601690e-01  8.1782645e-01
   1.7703940e+00  1.0588857e+00  1.2569947e+00  8.9014143e-01
   8.8181317e-01  2.0301542e+00  2.5859327e+00  2.6294985e+00
   1.8452185e+00  1.6792901e+00  4.6938765e-01  4.5742786e-01
   1.2193496e+00  9.3511879e-01  7.8335178e-01  1.1091820e+00
   6.4898384e-01  1.4056447e+00  8.8727242e-01  1.9257767e+00
   2.6558113e+00  8.6804760e-01  7.4532914e-01  6.3291574e-01
   4.4676340e-01  1.1630274e+00  1.2149620e+00  1.5745085e+00
   1.8722432e+00  6.6386420e-01  9.5228791e-01  7.8448051e-01
   1.6354405e+00  1.2105798e+00  9.9103296e-01  1.3100418e+00
   9.7270805e-01  4.8047203e-01  2.3412833e+00  1.1259243e+00
   8.3058488e-01  8.2318729e-01  3.8210297e-01  1.8458320e+00
   2.2641380e+00  1.4798534e+00  1.5347826e+00  1.5978411e+00
   2.4194834e+00  6.7134523e-01  2.3675547e+00  1.6738174e+00
   1.3724725e+00  6.6919518e-01  5.5598545e-01  2.3993716e+00
   1.9189554e-01  1.3499738e+00  1.0841005e+00  1.2430753e+00
   7.0875275e-01  4.5135498e-01  3.4969163e-01  1.9182110e-01
   1.6882527e+00  9.8765332e-01  8.4099025e-01  5.1583827e-01
   6.2336409e-01  1.9849713e+00  5.6225741e-01  1.2491460e+00
   3.7174416e-01  9.2007565e-01  1.0810699e+00  4.8536944e-01
   9.0319550e-01  7.5899029e-01  5.5883169e-01  2.6978803e-01
   4.2640853e-01  1.1224550e+00  7.9544926e-01  2.0021921e-01
   4.9618244e-02  6.8902593e+00  5.9498253e+00  7.0846682e+00
   7.3935857e+00  6.1179070e+00  6.3261323e+00  6.6306405e+00
   6.2578435e+00  6.7043700e+00  6.0366130e+00  7.0304313e+00
   5.1647158e+00  6.5611954e+00  6.3764315e+00  7.1293192e+00
   6.2104797e+00  6.1961527e+00  7.7294145e+00  7.0388083e+00
   6.1880279e+00  6.2984757e+00  6.7059078e+00  6.8873186e+00
   6.2655592e+00  6.9777269e+00  6.0812907e+00  7.4458098e+00
   6.2945724e+00  6.9972191e+00  5.9427156e+00  6.0209279e+00
   5.7059369e+00  6.7397270e+00  7.0852127e+00  5.4800453e+00
   5.5439553e+00  5.9926066e+00  6.5147147e+00  7.2228951e+00
   7.7438703e+00  5.7132359e+00  6.8644371e+00  6.8596559e+00
   5.3023500e+00  5.1348329e+00  7.1472297e+00  6.2021112e+00
   6.8420367e+00  6.4433284e+00  6.8617721e+00  6.1554170e+00
   5.6592226e+00  5.2419353e+00  6.0533934e+00  5.7188568e+00
   6.2193351e+00  6.8040986e+00  6.7154946e+00  6.2206297e+00
  -6.7575054e+00 -5.8835917e+00  7.6411443e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 07:13:12.718395
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.702
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 07:13:12.722220
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9372.81
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 07:13:12.726761
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.0025
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 07:13:12.729329
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -838.402
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139924645660040
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139922133201360
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139922133201864
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139922133202368
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139922133202872
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139922133203376

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f42bfd30dd8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.660825
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.620177
grad_step = 000002, loss = 0.594484
grad_step = 000003, loss = 0.565884
grad_step = 000004, loss = 0.538243
grad_step = 000005, loss = 0.518242
grad_step = 000006, loss = 0.504149
grad_step = 000007, loss = 0.486717
grad_step = 000008, loss = 0.469111
grad_step = 000009, loss = 0.453608
grad_step = 000010, loss = 0.440260
grad_step = 000011, loss = 0.429628
grad_step = 000012, loss = 0.420842
grad_step = 000013, loss = 0.412161
grad_step = 000014, loss = 0.401703
grad_step = 000015, loss = 0.389591
grad_step = 000016, loss = 0.377290
grad_step = 000017, loss = 0.365813
grad_step = 000018, loss = 0.355160
grad_step = 000019, loss = 0.344969
grad_step = 000020, loss = 0.335101
grad_step = 000021, loss = 0.325532
grad_step = 000022, loss = 0.315937
grad_step = 000023, loss = 0.305900
grad_step = 000024, loss = 0.295711
grad_step = 000025, loss = 0.285928
grad_step = 000026, loss = 0.276522
grad_step = 000027, loss = 0.267220
grad_step = 000028, loss = 0.258051
grad_step = 000029, loss = 0.249236
grad_step = 000030, loss = 0.240676
grad_step = 000031, loss = 0.232023
grad_step = 000032, loss = 0.223399
grad_step = 000033, loss = 0.215079
grad_step = 000034, loss = 0.206966
grad_step = 000035, loss = 0.199025
grad_step = 000036, loss = 0.191395
grad_step = 000037, loss = 0.183996
grad_step = 000038, loss = 0.176672
grad_step = 000039, loss = 0.169488
grad_step = 000040, loss = 0.162510
grad_step = 000041, loss = 0.155708
grad_step = 000042, loss = 0.149126
grad_step = 000043, loss = 0.142807
grad_step = 000044, loss = 0.136698
grad_step = 000045, loss = 0.130744
grad_step = 000046, loss = 0.124950
grad_step = 000047, loss = 0.119305
grad_step = 000048, loss = 0.113808
grad_step = 000049, loss = 0.108511
grad_step = 000050, loss = 0.103425
grad_step = 000051, loss = 0.098502
grad_step = 000052, loss = 0.093761
grad_step = 000053, loss = 0.089181
grad_step = 000054, loss = 0.084724
grad_step = 000055, loss = 0.080436
grad_step = 000056, loss = 0.076291
grad_step = 000057, loss = 0.072296
grad_step = 000058, loss = 0.068485
grad_step = 000059, loss = 0.064802
grad_step = 000060, loss = 0.061267
grad_step = 000061, loss = 0.057860
grad_step = 000062, loss = 0.054576
grad_step = 000063, loss = 0.051445
grad_step = 000064, loss = 0.048446
grad_step = 000065, loss = 0.045587
grad_step = 000066, loss = 0.042864
grad_step = 000067, loss = 0.040261
grad_step = 000068, loss = 0.037784
grad_step = 000069, loss = 0.035423
grad_step = 000070, loss = 0.033190
grad_step = 000071, loss = 0.031076
grad_step = 000072, loss = 0.029079
grad_step = 000073, loss = 0.027193
grad_step = 000074, loss = 0.025406
grad_step = 000075, loss = 0.023722
grad_step = 000076, loss = 0.022135
grad_step = 000077, loss = 0.020646
grad_step = 000078, loss = 0.019247
grad_step = 000079, loss = 0.017938
grad_step = 000080, loss = 0.016709
grad_step = 000081, loss = 0.015623
grad_step = 000082, loss = 0.014508
grad_step = 000083, loss = 0.013490
grad_step = 000084, loss = 0.012571
grad_step = 000085, loss = 0.011691
grad_step = 000086, loss = 0.010893
grad_step = 000087, loss = 0.010140
grad_step = 000088, loss = 0.009451
grad_step = 000089, loss = 0.008809
grad_step = 000090, loss = 0.008214
grad_step = 000091, loss = 0.007669
grad_step = 000092, loss = 0.007159
grad_step = 000093, loss = 0.006701
grad_step = 000094, loss = 0.006267
grad_step = 000095, loss = 0.005878
grad_step = 000096, loss = 0.005513
grad_step = 000097, loss = 0.005182
grad_step = 000098, loss = 0.004880
grad_step = 000099, loss = 0.004601
grad_step = 000100, loss = 0.004351
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004117
grad_step = 000102, loss = 0.003907
grad_step = 000103, loss = 0.003716
grad_step = 000104, loss = 0.003541
grad_step = 000105, loss = 0.003386
grad_step = 000106, loss = 0.003240
grad_step = 000107, loss = 0.003111
grad_step = 000108, loss = 0.002995
grad_step = 000109, loss = 0.002887
grad_step = 000110, loss = 0.002793
grad_step = 000111, loss = 0.002706
grad_step = 000112, loss = 0.002627
grad_step = 000113, loss = 0.002558
grad_step = 000114, loss = 0.002495
grad_step = 000115, loss = 0.002438
grad_step = 000116, loss = 0.002387
grad_step = 000117, loss = 0.002341
grad_step = 000118, loss = 0.002299
grad_step = 000119, loss = 0.002262
grad_step = 000120, loss = 0.002229
grad_step = 000121, loss = 0.002199
grad_step = 000122, loss = 0.002171
grad_step = 000123, loss = 0.002146
grad_step = 000124, loss = 0.002124
grad_step = 000125, loss = 0.002104
grad_step = 000126, loss = 0.002086
grad_step = 000127, loss = 0.002069
grad_step = 000128, loss = 0.002054
grad_step = 000129, loss = 0.002040
grad_step = 000130, loss = 0.002028
grad_step = 000131, loss = 0.002017
grad_step = 000132, loss = 0.002007
grad_step = 000133, loss = 0.001997
grad_step = 000134, loss = 0.001988
grad_step = 000135, loss = 0.001979
grad_step = 000136, loss = 0.001970
grad_step = 000137, loss = 0.001962
grad_step = 000138, loss = 0.001955
grad_step = 000139, loss = 0.001949
grad_step = 000140, loss = 0.001943
grad_step = 000141, loss = 0.001937
grad_step = 000142, loss = 0.001933
grad_step = 000143, loss = 0.001931
grad_step = 000144, loss = 0.001932
grad_step = 000145, loss = 0.001934
grad_step = 000146, loss = 0.001932
grad_step = 000147, loss = 0.001923
grad_step = 000148, loss = 0.001909
grad_step = 000149, loss = 0.001899
grad_step = 000150, loss = 0.001894
grad_step = 000151, loss = 0.001896
grad_step = 000152, loss = 0.001899
grad_step = 000153, loss = 0.001900
grad_step = 000154, loss = 0.001895
grad_step = 000155, loss = 0.001886
grad_step = 000156, loss = 0.001876
grad_step = 000157, loss = 0.001868
grad_step = 000158, loss = 0.001864
grad_step = 000159, loss = 0.001863
grad_step = 000160, loss = 0.001862
grad_step = 000161, loss = 0.001861
grad_step = 000162, loss = 0.001859
grad_step = 000163, loss = 0.001857
grad_step = 000164, loss = 0.001862
grad_step = 000165, loss = 0.001881
grad_step = 000166, loss = 0.001927
grad_step = 000167, loss = 0.001981
grad_step = 000168, loss = 0.002012
grad_step = 000169, loss = 0.001932
grad_step = 000170, loss = 0.001846
grad_step = 000171, loss = 0.001846
grad_step = 000172, loss = 0.001902
grad_step = 000173, loss = 0.001919
grad_step = 000174, loss = 0.001857
grad_step = 000175, loss = 0.001825
grad_step = 000176, loss = 0.001859
grad_step = 000177, loss = 0.001882
grad_step = 000178, loss = 0.001854
grad_step = 000179, loss = 0.001818
grad_step = 000180, loss = 0.001828
grad_step = 000181, loss = 0.001854
grad_step = 000182, loss = 0.001843
grad_step = 000183, loss = 0.001815
grad_step = 000184, loss = 0.001809
grad_step = 000185, loss = 0.001826
grad_step = 000186, loss = 0.001834
grad_step = 000187, loss = 0.001819
grad_step = 000188, loss = 0.001804
grad_step = 000189, loss = 0.001808
grad_step = 000190, loss = 0.001821
grad_step = 000191, loss = 0.001827
grad_step = 000192, loss = 0.001822
grad_step = 000193, loss = 0.001821
grad_step = 000194, loss = 0.001835
grad_step = 000195, loss = 0.001850
grad_step = 000196, loss = 0.001857
grad_step = 000197, loss = 0.001832
grad_step = 000198, loss = 0.001801
grad_step = 000199, loss = 0.001784
grad_step = 000200, loss = 0.001791
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001808
grad_step = 000202, loss = 0.001812
grad_step = 000203, loss = 0.001800
grad_step = 000204, loss = 0.001782
grad_step = 000205, loss = 0.001775
grad_step = 000206, loss = 0.001780
grad_step = 000207, loss = 0.001789
grad_step = 000208, loss = 0.001791
grad_step = 000209, loss = 0.001784
grad_step = 000210, loss = 0.001775
grad_step = 000211, loss = 0.001769
grad_step = 000212, loss = 0.001768
grad_step = 000213, loss = 0.001770
grad_step = 000214, loss = 0.001771
grad_step = 000215, loss = 0.001768
grad_step = 000216, loss = 0.001763
grad_step = 000217, loss = 0.001758
grad_step = 000218, loss = 0.001756
grad_step = 000219, loss = 0.001757
grad_step = 000220, loss = 0.001759
grad_step = 000221, loss = 0.001763
grad_step = 000222, loss = 0.001769
grad_step = 000223, loss = 0.001782
grad_step = 000224, loss = 0.001812
grad_step = 000225, loss = 0.001884
grad_step = 000226, loss = 0.001990
grad_step = 000227, loss = 0.002135
grad_step = 000228, loss = 0.002046
grad_step = 000229, loss = 0.001848
grad_step = 000230, loss = 0.001747
grad_step = 000231, loss = 0.001868
grad_step = 000232, loss = 0.001958
grad_step = 000233, loss = 0.001815
grad_step = 000234, loss = 0.001738
grad_step = 000235, loss = 0.001826
grad_step = 000236, loss = 0.001859
grad_step = 000237, loss = 0.001779
grad_step = 000238, loss = 0.001735
grad_step = 000239, loss = 0.001796
grad_step = 000240, loss = 0.001818
grad_step = 000241, loss = 0.001750
grad_step = 000242, loss = 0.001736
grad_step = 000243, loss = 0.001784
grad_step = 000244, loss = 0.001775
grad_step = 000245, loss = 0.001733
grad_step = 000246, loss = 0.001737
grad_step = 000247, loss = 0.001766
grad_step = 000248, loss = 0.001758
grad_step = 000249, loss = 0.001732
grad_step = 000250, loss = 0.001744
grad_step = 000251, loss = 0.001771
grad_step = 000252, loss = 0.001769
grad_step = 000253, loss = 0.001767
grad_step = 000254, loss = 0.001786
grad_step = 000255, loss = 0.001814
grad_step = 000256, loss = 0.001800
grad_step = 000257, loss = 0.001772
grad_step = 000258, loss = 0.001742
grad_step = 000259, loss = 0.001726
grad_step = 000260, loss = 0.001716
grad_step = 000261, loss = 0.001717
grad_step = 000262, loss = 0.001731
grad_step = 000263, loss = 0.001743
grad_step = 000264, loss = 0.001741
grad_step = 000265, loss = 0.001719
grad_step = 000266, loss = 0.001701
grad_step = 000267, loss = 0.001698
grad_step = 000268, loss = 0.001707
grad_step = 000269, loss = 0.001714
grad_step = 000270, loss = 0.001713
grad_step = 000271, loss = 0.001706
grad_step = 000272, loss = 0.001698
grad_step = 000273, loss = 0.001693
grad_step = 000274, loss = 0.001692
grad_step = 000275, loss = 0.001692
grad_step = 000276, loss = 0.001692
grad_step = 000277, loss = 0.001694
grad_step = 000278, loss = 0.001698
grad_step = 000279, loss = 0.001699
grad_step = 000280, loss = 0.001697
grad_step = 000281, loss = 0.001690
grad_step = 000282, loss = 0.001682
grad_step = 000283, loss = 0.001677
grad_step = 000284, loss = 0.001675
grad_step = 000285, loss = 0.001676
grad_step = 000286, loss = 0.001677
grad_step = 000287, loss = 0.001677
grad_step = 000288, loss = 0.001677
grad_step = 000289, loss = 0.001675
grad_step = 000290, loss = 0.001673
grad_step = 000291, loss = 0.001672
grad_step = 000292, loss = 0.001672
grad_step = 000293, loss = 0.001673
grad_step = 000294, loss = 0.001677
grad_step = 000295, loss = 0.001684
grad_step = 000296, loss = 0.001697
grad_step = 000297, loss = 0.001715
grad_step = 000298, loss = 0.001747
grad_step = 000299, loss = 0.001782
grad_step = 000300, loss = 0.001830
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001839
grad_step = 000302, loss = 0.001825
grad_step = 000303, loss = 0.001752
grad_step = 000304, loss = 0.001698
grad_step = 000305, loss = 0.001704
grad_step = 000306, loss = 0.001755
grad_step = 000307, loss = 0.001818
grad_step = 000308, loss = 0.001810
grad_step = 000309, loss = 0.001763
grad_step = 000310, loss = 0.001687
grad_step = 000311, loss = 0.001652
grad_step = 000312, loss = 0.001665
grad_step = 000313, loss = 0.001694
grad_step = 000314, loss = 0.001710
grad_step = 000315, loss = 0.001689
grad_step = 000316, loss = 0.001662
grad_step = 000317, loss = 0.001646
grad_step = 000318, loss = 0.001651
grad_step = 000319, loss = 0.001665
grad_step = 000320, loss = 0.001667
grad_step = 000321, loss = 0.001655
grad_step = 000322, loss = 0.001635
grad_step = 000323, loss = 0.001625
grad_step = 000324, loss = 0.001628
grad_step = 000325, loss = 0.001640
grad_step = 000326, loss = 0.001649
grad_step = 000327, loss = 0.001646
grad_step = 000328, loss = 0.001636
grad_step = 000329, loss = 0.001622
grad_step = 000330, loss = 0.001614
grad_step = 000331, loss = 0.001612
grad_step = 000332, loss = 0.001615
grad_step = 000333, loss = 0.001619
grad_step = 000334, loss = 0.001620
grad_step = 000335, loss = 0.001617
grad_step = 000336, loss = 0.001611
grad_step = 000337, loss = 0.001605
grad_step = 000338, loss = 0.001601
grad_step = 000339, loss = 0.001599
grad_step = 000340, loss = 0.001601
grad_step = 000341, loss = 0.001610
grad_step = 000342, loss = 0.001630
grad_step = 000343, loss = 0.001673
grad_step = 000344, loss = 0.001756
grad_step = 000345, loss = 0.001892
grad_step = 000346, loss = 0.002020
grad_step = 000347, loss = 0.002054
grad_step = 000348, loss = 0.001929
grad_step = 000349, loss = 0.001697
grad_step = 000350, loss = 0.001613
grad_step = 000351, loss = 0.001708
grad_step = 000352, loss = 0.001794
grad_step = 000353, loss = 0.001743
grad_step = 000354, loss = 0.001624
grad_step = 000355, loss = 0.001607
grad_step = 000356, loss = 0.001660
grad_step = 000357, loss = 0.001692
grad_step = 000358, loss = 0.001670
grad_step = 000359, loss = 0.001613
grad_step = 000360, loss = 0.001587
grad_step = 000361, loss = 0.001607
grad_step = 000362, loss = 0.001627
grad_step = 000363, loss = 0.001616
grad_step = 000364, loss = 0.001603
grad_step = 000365, loss = 0.001593
grad_step = 000366, loss = 0.001575
grad_step = 000367, loss = 0.001571
grad_step = 000368, loss = 0.001583
grad_step = 000369, loss = 0.001588
grad_step = 000370, loss = 0.001581
grad_step = 000371, loss = 0.001565
grad_step = 000372, loss = 0.001549
grad_step = 000373, loss = 0.001550
grad_step = 000374, loss = 0.001563
grad_step = 000375, loss = 0.001569
grad_step = 000376, loss = 0.001557
grad_step = 000377, loss = 0.001542
grad_step = 000378, loss = 0.001538
grad_step = 000379, loss = 0.001540
grad_step = 000380, loss = 0.001540
grad_step = 000381, loss = 0.001539
grad_step = 000382, loss = 0.001539
grad_step = 000383, loss = 0.001537
grad_step = 000384, loss = 0.001535
grad_step = 000385, loss = 0.001534
grad_step = 000386, loss = 0.001534
grad_step = 000387, loss = 0.001534
grad_step = 000388, loss = 0.001529
grad_step = 000389, loss = 0.001525
grad_step = 000390, loss = 0.001523
grad_step = 000391, loss = 0.001525
grad_step = 000392, loss = 0.001529
grad_step = 000393, loss = 0.001534
grad_step = 000394, loss = 0.001546
grad_step = 000395, loss = 0.001570
grad_step = 000396, loss = 0.001613
grad_step = 000397, loss = 0.001689
grad_step = 000398, loss = 0.001754
grad_step = 000399, loss = 0.001809
grad_step = 000400, loss = 0.001815
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001723
grad_step = 000402, loss = 0.001623
grad_step = 000403, loss = 0.001536
grad_step = 000404, loss = 0.001545
grad_step = 000405, loss = 0.001602
grad_step = 000406, loss = 0.001610
grad_step = 000407, loss = 0.001588
grad_step = 000408, loss = 0.001563
grad_step = 000409, loss = 0.001536
grad_step = 000410, loss = 0.001511
grad_step = 000411, loss = 0.001510
grad_step = 000412, loss = 0.001543
grad_step = 000413, loss = 0.001569
grad_step = 000414, loss = 0.001544
grad_step = 000415, loss = 0.001514
grad_step = 000416, loss = 0.001514
grad_step = 000417, loss = 0.001532
grad_step = 000418, loss = 0.001523
grad_step = 000419, loss = 0.001495
grad_step = 000420, loss = 0.001479
grad_step = 000421, loss = 0.001486
grad_step = 000422, loss = 0.001492
grad_step = 000423, loss = 0.001489
grad_step = 000424, loss = 0.001488
grad_step = 000425, loss = 0.001493
grad_step = 000426, loss = 0.001494
grad_step = 000427, loss = 0.001484
grad_step = 000428, loss = 0.001477
grad_step = 000429, loss = 0.001480
grad_step = 000430, loss = 0.001487
grad_step = 000431, loss = 0.001486
grad_step = 000432, loss = 0.001477
grad_step = 000433, loss = 0.001471
grad_step = 000434, loss = 0.001472
grad_step = 000435, loss = 0.001475
grad_step = 000436, loss = 0.001477
grad_step = 000437, loss = 0.001478
grad_step = 000438, loss = 0.001484
grad_step = 000439, loss = 0.001491
grad_step = 000440, loss = 0.001500
grad_step = 000441, loss = 0.001512
grad_step = 000442, loss = 0.001532
grad_step = 000443, loss = 0.001565
grad_step = 000444, loss = 0.001599
grad_step = 000445, loss = 0.001630
grad_step = 000446, loss = 0.001630
grad_step = 000447, loss = 0.001608
grad_step = 000448, loss = 0.001542
grad_step = 000449, loss = 0.001479
grad_step = 000450, loss = 0.001436
grad_step = 000451, loss = 0.001433
grad_step = 000452, loss = 0.001459
grad_step = 000453, loss = 0.001491
grad_step = 000454, loss = 0.001514
grad_step = 000455, loss = 0.001515
grad_step = 000456, loss = 0.001502
grad_step = 000457, loss = 0.001477
grad_step = 000458, loss = 0.001450
grad_step = 000459, loss = 0.001433
grad_step = 000460, loss = 0.001428
grad_step = 000461, loss = 0.001433
grad_step = 000462, loss = 0.001439
grad_step = 000463, loss = 0.001442
grad_step = 000464, loss = 0.001443
grad_step = 000465, loss = 0.001442
grad_step = 000466, loss = 0.001437
grad_step = 000467, loss = 0.001429
grad_step = 000468, loss = 0.001420
grad_step = 000469, loss = 0.001413
grad_step = 000470, loss = 0.001408
grad_step = 000471, loss = 0.001405
grad_step = 000472, loss = 0.001405
grad_step = 000473, loss = 0.001408
grad_step = 000474, loss = 0.001412
grad_step = 000475, loss = 0.001417
grad_step = 000476, loss = 0.001422
grad_step = 000477, loss = 0.001423
grad_step = 000478, loss = 0.001425
grad_step = 000479, loss = 0.001423
grad_step = 000480, loss = 0.001423
grad_step = 000481, loss = 0.001420
grad_step = 000482, loss = 0.001420
grad_step = 000483, loss = 0.001420
grad_step = 000484, loss = 0.001427
grad_step = 000485, loss = 0.001440
grad_step = 000486, loss = 0.001462
grad_step = 000487, loss = 0.001490
grad_step = 000488, loss = 0.001528
grad_step = 000489, loss = 0.001559
grad_step = 000490, loss = 0.001589
grad_step = 000491, loss = 0.001578
grad_step = 000492, loss = 0.001553
grad_step = 000493, loss = 0.001497
grad_step = 000494, loss = 0.001451
grad_step = 000495, loss = 0.001435
grad_step = 000496, loss = 0.001451
grad_step = 000497, loss = 0.001488
grad_step = 000498, loss = 0.001471
grad_step = 000499, loss = 0.001429
grad_step = 000500, loss = 0.001378
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001376
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

  date_run                              2020-05-15 07:13:30.745044
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.198322
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 07:13:30.750724
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0777321
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 07:13:30.757763
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139012
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 07:13:30.762682
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.181167
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
0   2020-05-15 07:13:02.037160  ...    mean_absolute_error
1   2020-05-15 07:13:02.039938  ...     mean_squared_error
2   2020-05-15 07:13:02.042203  ...  median_absolute_error
3   2020-05-15 07:13:02.045260  ...               r2_score
4   2020-05-15 07:13:12.718395  ...    mean_absolute_error
5   2020-05-15 07:13:12.722220  ...     mean_squared_error
6   2020-05-15 07:13:12.726761  ...  median_absolute_error
7   2020-05-15 07:13:12.729329  ...               r2_score
8   2020-05-15 07:13:30.745044  ...    mean_absolute_error
9   2020-05-15 07:13:30.750724  ...     mean_squared_error
10  2020-05-15 07:13:30.757763  ...  median_absolute_error
11  2020-05-15 07:13:30.762682  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6be1a68be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 316054.30it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 408600.14it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 564215.26it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 797562.49it/s] 77%|███████▋  | 7610368/9912422 [00:00<00:02, 1127916.60it/s]9920512it [00:00, 10359037.50it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 151381.83it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 313349.73it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 407091.34it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 563550.23it/s]1654784it [00:00, 2865474.82it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52686.94it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b94421e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b93a520b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b94421e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b939a80b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b911e34a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b911cfc18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b94421e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b939676d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b911e34a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6be1a2beb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb3b60f01d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c159d73d339386cb4a5f729e08011222440119acd17992524daed3b786f59f1b
  Stored in directory: /tmp/pip-ephem-wheel-cache-tiwm5u5w/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb34deeb710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   49152/17464789 [..............................] - ETA: 46s
   81920/17464789 [..............................] - ETA: 41s
  188416/17464789 [..............................] - ETA: 23s
  376832/17464789 [..............................] - ETA: 14s
  761856/17464789 [>.............................] - ETA: 8s 
 1531904/17464789 [=>............................] - ETA: 4s
 3055616/17464789 [====>.........................] - ETA: 2s
 5906432/17464789 [=========>....................] - ETA: 1s
 8888320/17464789 [==============>...............] - ETA: 0s
11919360/17464789 [===================>..........] - ETA: 0s
14901248/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 07:14:59.650650: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 07:14:59.658048: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-15 07:14:59.658205: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558149735ef0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 07:14:59.658220: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7433 - accuracy: 0.4950
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7318 - accuracy: 0.4958
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5702 - accuracy: 0.5063
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6036 - accuracy: 0.5041
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6406 - accuracy: 0.5017
11000/25000 [============>.................] - ETA: 3s - loss: 7.6429 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 3s - loss: 7.6398 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6407 - accuracy: 0.5017
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6162 - accuracy: 0.5033
15000/25000 [=================>............] - ETA: 2s - loss: 7.6257 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6019 - accuracy: 0.5042
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6037 - accuracy: 0.5041
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6191 - accuracy: 0.5031
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6301 - accuracy: 0.5024
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6436 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 07:15:12.508342
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 07:15:12.508342  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<7:00:32, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<4:55:23, 48.6kB/s] .vector_cache/glove.6B.zip:   1%|          | 6.52M/862M [00:00<3:25:23, 69.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:00<2:22:27, 99.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:00<1:38:41, 142kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:00<1:08:34, 202kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:00<47:34, 288kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:00<33:02, 411kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:01<23:02, 586kB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.2M/862M [00:02<18:49, 715kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:02<13:19, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:02<10:17, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<10:09:48, 22.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:04<7:06:05, 31.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:05<5:00:42, 44.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:06<3:31:26, 63.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:07<2:29:19, 88.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:08<1:45:08, 126kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:09<1:15:23, 175kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:09<53:19, 247kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:11<39:16, 334kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:12<28:08, 466kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:13<21:43, 602kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:13<15:50, 824kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:15<13:06, 992kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:15<09:49, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:17<08:56, 1.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:17<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:19<06:51, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:19<05:25, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:21<05:51, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:21<04:44, 2.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<03:55, 3.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<6:09:44, 34.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:23<4:17:31, 49.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<6:36:12, 32.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<4:38:34, 45.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<3:15:51, 64.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<2:17:35, 91.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<1:37:55, 128kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<1:09:06, 181kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<50:10, 249kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<35:41, 349kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<26:50, 462kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:32<19:23, 639kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<15:56, 776kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<13:01, 947kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<09:50, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<08:45, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<06:43, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<05:22, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<5:42:43, 35.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<4:00:17, 50.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<2:49:06, 71.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<1:59:35, 101kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:41<1:24:15, 143kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<1:00:38, 198kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:43<43:02, 279kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<31:54, 375kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<22:56, 520kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<17:52, 664kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<13:06, 906kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<11:02, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<08:19, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<06:27, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<5:21:01, 36.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:51<3:43:51, 52.3kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<2:42:48, 71.9kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<1:54:49, 102kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<1:21:42, 142kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:54<57:43, 201kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<42:04, 275kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<30:00, 385kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<22:46, 505kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<16:30, 696kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:00<13:20, 856kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<09:55, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<08:44, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:02<06:41, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<05:12, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<5:27:13, 34.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:04<3:48:00, 49.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<2:48:15, 66.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<1:58:12, 95.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<1:24:06, 133kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<59:39, 187kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<43:12, 257kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<31:09, 356kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<23:19, 473kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<17:14, 640kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<13:37, 805kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<10:32, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<08:56, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:15<06:57, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<05:25, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<4:53:40, 37.0kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<3:25:48, 52.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<2:24:49, 74.5kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<1:42:25, 105kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:20<1:12:09, 148kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<51:57, 205kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:22<36:52, 289kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<27:22, 387kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:24<19:52, 532kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<15:25, 682kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<11:20, 926kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<09:34, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<07:14, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<05:37, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<4:44:01, 36.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<3:18:59, 51.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<2:20:00, 73.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:32<1:37:33, 105kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<1:27:36, 117kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:33<1:02:07, 165kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<44:45, 228kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:35<32:08, 317kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:36<22:27, 451kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:37<40:38, 249kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<29:15, 346kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:38<20:26, 492kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<1:03:59, 157kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<45:34, 221kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:39<31:49, 314kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<32:24, 308kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<23:29, 425kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<16:49, 591kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<4:51:38, 34.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:43<3:23:27, 48.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<2:25:12, 68.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<1:42:22, 96.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<1:12:41, 135kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:46<51:37, 190kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:47<36:00, 271kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<50:08, 194kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<35:51, 272kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:49<25:01, 387kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<1:01:45, 157kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<43:58, 220kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:51<30:39, 313kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:29:11, 29.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<3:51:03, 41.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<2:41:36, 59.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<6:14:51, 25.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:54<4:21:21, 36.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<3:05:47, 51.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<2:10:44, 72.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<1:32:20, 102kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<1:05:20, 144kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<46:51, 200kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<33:33, 279kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<24:43, 376kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<18:06, 513kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<13:57, 660kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<10:30, 876kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<07:45, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<4:13:45, 36.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:05<2:57:00, 51.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<2:06:21, 72.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<1:29:07, 102kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<1:03:18, 143kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<44:59, 201kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:09<31:21, 286kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<46:08, 195kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<33:00, 272kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<24:17, 367kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<17:39, 504kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:13<12:21, 715kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<1:19:38, 111kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<56:25, 156kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:14<39:18, 223kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<1:37:43, 89.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<1:09:04, 127kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<48:36, 180kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<6:03:20, 24.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<4:13:22, 34.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<2:57:36, 48.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<2:04:51, 68.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<1:28:00, 97.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<1:02:27, 136kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<44:21, 192kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<32:06, 263kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<23:08, 365kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<17:20, 484kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:27<12:47, 655kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<10:08, 820kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<07:45, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<05:46, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<3:56:58, 34.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<2:45:13, 49.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<1:57:43, 69.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<1:22:59, 98.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:33<57:43, 141kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<9:21:24, 14.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<6:33:15, 20.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<4:34:25, 29.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:36<3:12:34, 41.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<2:14:58, 59.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<1:35:03, 84.1kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<1:07:12, 118kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<47:38, 166kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<33:34, 235kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<3:57:09, 33.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<2:45:17, 47.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<1:57:41, 66.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<1:22:56, 94.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<58:45, 132kB/s]   .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:45<41:42, 186kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<30:07, 255kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:47<21:41, 354kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<16:12, 470kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<11:57, 636kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<09:26, 800kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<07:13, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<06:07, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<04:55, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<03:46, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<3:54:47, 31.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:55<2:43:27, 45.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<1:56:50, 63.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<1:22:17, 89.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<58:13, 125kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<41:18, 176kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<29:45, 243kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<21:23, 338kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<15:55, 450kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<11:42, 611kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<09:11, 771kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<06:59, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<05:54, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<04:42, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<04:18, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<03:34, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<03:30, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<03:01, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<02:23, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:28:24, 32.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<2:25:01, 46.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<1:43:41, 65.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<1:13:03, 92.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<51:41, 130kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<36:39, 183kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<26:25, 251kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<19:00, 349kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<14:10, 463kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<10:25, 629kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<08:12, 791kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<06:15, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<05:18, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<04:13, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<03:53, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<04:54, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:25<03:39, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:26<02:40, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<04:27, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<02:48, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<3:04:20, 33.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<2:08:16, 48.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<1:31:26, 67.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<1:04:25, 95.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:32<45:34, 134kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<32:23, 188kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<23:20, 259kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<16:48, 359kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:35<11:42, 511kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<14:01, 426kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<10:17, 579kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<08:01, 736kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:38<06:04, 970kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<05:05, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<04:02, 1.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<03:39, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<03:01, 1.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:57, 1.93MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<02:31, 2.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<01:59, 2.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<2:53:28, 32.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<2:00:37, 46.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<1:25:50, 65.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<1:00:27, 92.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<42:40, 129kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<30:16, 182kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<21:47, 250kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<15:36, 348kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:52<10:51, 495kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<18:28, 291kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<13:20, 402kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<10:02, 529kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<07:23, 717kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:56<05:10, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<09:27, 554kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<07:02, 744kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<05:38, 916kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<04:21, 1.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<03:47, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<03:02, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<02:19, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<2:34:16, 32.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<1:47:23, 46.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<1:15:55, 65.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<53:28, 93.2kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<37:42, 130kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<26:45, 184kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<19:13, 252kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<13:49, 350kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:09<09:40, 494kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<4:53:24, 16.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<3:25:26, 23.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<2:22:40, 33.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<1:40:05, 47.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<1:09:50, 66.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<49:12, 94.2kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<34:40, 132kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<24:35, 186kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<17:39, 255kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<12:42, 354kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<09:26, 470kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<06:57, 637kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<05:27, 800kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<04:10, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<03:31, 1.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<02:49, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<02:34, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<02:08, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<01:39, 2.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<2:08:34, 32.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:28<1:29:03, 46.5kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<1:03:41, 64.6kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<44:49, 91.6kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<31:31, 128kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<22:21, 181kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<16:00, 248kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<11:28, 346kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:34<07:59, 491kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<07:34, 516kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<05:35, 698kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<03:57, 979kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<03:42, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<02:55, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<02:03, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<04:31, 834kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<03:31, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:40<02:29, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<03:07, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<02:40, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:41<01:54, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<01:56, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<1:44:12, 35.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<1:12:39, 50.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:43<50:17, 71.5kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<40:24, 88.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<28:31, 126kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:45<19:46, 179kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<15:27, 228kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<11:05, 317kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:47<07:42, 451kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<07:09, 482kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<05:14, 658kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:49<03:40, 926kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<03:38, 930kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:50<02:47, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:51<01:58, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<02:34, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<02:02, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<01:27, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<02:23, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<01:55, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<01:22, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<02:38, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<01:37, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<1:26:00, 36.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<59:43, 52.0kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:58<41:22, 74.1kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<3:23:31, 15.1kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<2:22:24, 21.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:00<1:38:23, 30.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:09:39, 43.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<48:53, 61.1kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:02<33:45, 87.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<24:55, 117kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<17:39, 165kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<12:13, 236kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<09:42, 294kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<07:01, 406kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:06<04:51, 577kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<05:04, 551kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<03:46, 739kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:08<02:37, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<03:26, 792kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<02:37, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:10<01:50, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<02:41, 986kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<02:05, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<01:28, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<02:29, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<02:03, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:13<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<01:53, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<01:31, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<1:10:00, 35.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<48:36, 50.4kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<33:53, 70.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<23:50, 100kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:19<16:27, 143kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<12:14, 191kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<08:46, 265kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<06:04, 377kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<05:00, 452kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:22<03:41, 613kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<02:34, 864kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<02:25, 903kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:24<02:00, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<01:25, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<01:35, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<01:25, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<01:00, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<01:17, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<01:03, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:29<00:45, 2.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:23, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:08, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:52, 2.24MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<57:12, 34.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<39:24, 48.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<27:31, 68.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<19:21, 96.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:34<13:13, 138kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<10:14, 177kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<07:17, 247kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:36<04:58, 352kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<04:50, 360kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<03:30, 493kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:38<02:24, 699kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<02:31, 661kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<01:54, 874kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:40<01:18, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<02:04, 772kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<01:34, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:42<01:05, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<02:09, 711kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<01:37, 939kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:44<01:06, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<02:25, 605kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<01:48, 807kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:45<01:14, 1.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<01:38, 848kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<01:15, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:55, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<40:50, 33.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<28:08, 47.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<19:16, 66.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<13:31, 94.4kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:51<09:08, 135kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<06:57, 175kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<04:56, 245kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:53<03:20, 348kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<02:56, 389kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<02:08, 532kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:55<01:26, 754kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<01:51, 582kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<01:22, 777kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<00:56, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<01:04, 934kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<00:50, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:35, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [05:59<00:24, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<01:47, 523kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:00<01:19, 702kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:01<00:53, 991kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<01:03, 824kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:48, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:34, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<23:48, 34.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<16:10, 49.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:04<10:44, 70.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<11:59, 63.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<08:24, 89.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<05:37, 128kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<03:59, 173kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<02:49, 242kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<01:52, 344kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<01:28, 421kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<01:04, 572kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:10<00:42, 810kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:49, 671kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:36, 887kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:12<00:24, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:32, 906kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:13<00:24, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<00:15, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:21, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:16, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<00:10, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:19, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<00:14, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<00:10, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<08:44, 34.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<05:31, 49.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<03:22, 69.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<02:19, 98.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:21<01:19, 141kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:53, 186kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:36, 260kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:23<00:17, 370kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:15, 378kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:24<00:10, 512kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:04, 724kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:02, 773kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:01, 1.01MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 903/400000 [00:00<00:44, 9028.78it/s]  0%|          | 1839/400000 [00:00<00:43, 9124.52it/s]  1%|          | 2738/400000 [00:00<00:43, 9083.08it/s]  1%|          | 3598/400000 [00:00<00:44, 8931.63it/s]  1%|          | 4487/400000 [00:00<00:44, 8916.02it/s]  1%|▏         | 5355/400000 [00:00<00:44, 8841.46it/s]  2%|▏         | 6296/400000 [00:00<00:43, 9003.71it/s]  2%|▏         | 7181/400000 [00:00<00:43, 8955.18it/s]  2%|▏         | 8122/400000 [00:00<00:43, 9086.90it/s]  2%|▏         | 9031/400000 [00:01<00:43, 9086.53it/s]  2%|▏         | 9988/400000 [00:01<00:42, 9224.57it/s]  3%|▎         | 10925/400000 [00:01<00:41, 9265.20it/s]  3%|▎         | 11852/400000 [00:01<00:41, 9265.15it/s]  3%|▎         | 12828/400000 [00:01<00:41, 9406.47it/s]  3%|▎         | 13840/400000 [00:01<00:40, 9608.11it/s]  4%|▎         | 14799/400000 [00:01<00:40, 9552.85it/s]  4%|▍         | 15789/400000 [00:01<00:39, 9651.45it/s]  4%|▍         | 16773/400000 [00:01<00:39, 9705.91it/s]  4%|▍         | 17743/400000 [00:01<00:39, 9681.48it/s]  5%|▍         | 18711/400000 [00:02<00:40, 9524.02it/s]  5%|▍         | 19664/400000 [00:02<00:40, 9410.03it/s]  5%|▌         | 20683/400000 [00:02<00:39, 9628.86it/s]  5%|▌         | 21648/400000 [00:02<00:40, 9444.97it/s]  6%|▌         | 22605/400000 [00:02<00:39, 9479.12it/s]  6%|▌         | 23555/400000 [00:02<00:40, 9312.24it/s]  6%|▌         | 24500/400000 [00:02<00:40, 9352.52it/s]  6%|▋         | 25437/400000 [00:02<00:40, 9139.24it/s]  7%|▋         | 26353/400000 [00:02<00:41, 9016.93it/s]  7%|▋         | 27292/400000 [00:02<00:40, 9124.44it/s]  7%|▋         | 28206/400000 [00:03<00:41, 9010.07it/s]  7%|▋         | 29109/400000 [00:03<00:41, 8830.94it/s]  7%|▋         | 29994/400000 [00:03<00:43, 8523.25it/s]  8%|▊         | 30850/400000 [00:03<00:44, 8334.95it/s]  8%|▊         | 31728/400000 [00:03<00:43, 8461.09it/s]  8%|▊         | 32611/400000 [00:03<00:42, 8567.52it/s]  8%|▊         | 33495/400000 [00:03<00:42, 8644.84it/s]  9%|▊         | 34362/400000 [00:03<00:43, 8478.96it/s]  9%|▉         | 35212/400000 [00:03<00:43, 8431.64it/s]  9%|▉         | 36110/400000 [00:03<00:42, 8588.80it/s]  9%|▉         | 36971/400000 [00:04<00:42, 8494.49it/s]  9%|▉         | 37833/400000 [00:04<00:42, 8528.84it/s] 10%|▉         | 38711/400000 [00:04<00:42, 8601.62it/s] 10%|▉         | 39596/400000 [00:04<00:41, 8672.91it/s] 10%|█         | 40465/400000 [00:04<00:42, 8382.88it/s] 10%|█         | 41319/400000 [00:04<00:42, 8428.77it/s] 11%|█         | 42177/400000 [00:04<00:42, 8473.25it/s] 11%|█         | 43026/400000 [00:04<00:42, 8462.90it/s] 11%|█         | 43884/400000 [00:04<00:41, 8496.64it/s] 11%|█         | 44749/400000 [00:04<00:41, 8541.94it/s] 11%|█▏        | 45623/400000 [00:05<00:41, 8600.16it/s] 12%|█▏        | 46500/400000 [00:05<00:40, 8648.49it/s] 12%|█▏        | 47422/400000 [00:05<00:40, 8810.26it/s] 12%|█▏        | 48305/400000 [00:05<00:40, 8638.98it/s] 12%|█▏        | 49193/400000 [00:05<00:40, 8709.30it/s] 13%|█▎        | 50068/400000 [00:05<00:40, 8720.78it/s] 13%|█▎        | 50941/400000 [00:05<00:40, 8702.94it/s] 13%|█▎        | 51847/400000 [00:05<00:39, 8806.15it/s] 13%|█▎        | 52758/400000 [00:05<00:39, 8894.11it/s] 13%|█▎        | 53660/400000 [00:06<00:38, 8931.16it/s] 14%|█▎        | 54582/400000 [00:06<00:38, 9012.95it/s] 14%|█▍        | 55507/400000 [00:06<00:37, 9079.53it/s] 14%|█▍        | 56416/400000 [00:06<00:38, 9027.83it/s] 14%|█▍        | 57362/400000 [00:06<00:37, 9152.68it/s] 15%|█▍        | 58321/400000 [00:06<00:36, 9278.28it/s] 15%|█▍        | 59250/400000 [00:06<00:37, 9058.18it/s] 15%|█▌        | 60158/400000 [00:06<00:37, 9037.55it/s] 15%|█▌        | 61091/400000 [00:06<00:37, 9122.35it/s] 16%|█▌        | 62005/400000 [00:06<00:37, 9082.25it/s] 16%|█▌        | 62915/400000 [00:07<00:37, 8959.64it/s] 16%|█▌        | 63863/400000 [00:07<00:36, 9107.25it/s] 16%|█▌        | 64775/400000 [00:07<00:36, 9088.41it/s] 16%|█▋        | 65705/400000 [00:07<00:36, 9149.98it/s] 17%|█▋        | 66621/400000 [00:07<00:36, 9114.44it/s] 17%|█▋        | 67533/400000 [00:07<00:36, 9076.02it/s] 17%|█▋        | 68505/400000 [00:07<00:35, 9257.03it/s] 17%|█▋        | 69476/400000 [00:07<00:35, 9387.97it/s] 18%|█▊        | 70465/400000 [00:07<00:34, 9531.79it/s] 18%|█▊        | 71473/400000 [00:07<00:33, 9688.59it/s] 18%|█▊        | 72444/400000 [00:08<00:34, 9531.75it/s] 18%|█▊        | 73399/400000 [00:08<00:34, 9400.31it/s] 19%|█▊        | 74367/400000 [00:08<00:34, 9481.97it/s] 19%|█▉        | 75345/400000 [00:08<00:33, 9567.51it/s] 19%|█▉        | 76319/400000 [00:08<00:33, 9618.19it/s] 19%|█▉        | 77303/400000 [00:08<00:33, 9682.14it/s] 20%|█▉        | 78275/400000 [00:08<00:33, 9691.34it/s] 20%|█▉        | 79245/400000 [00:08<00:34, 9420.43it/s] 20%|██        | 80190/400000 [00:08<00:34, 9285.03it/s] 20%|██        | 81121/400000 [00:08<00:34, 9187.12it/s] 21%|██        | 82042/400000 [00:09<00:34, 9174.65it/s] 21%|██        | 82961/400000 [00:09<00:35, 9020.28it/s] 21%|██        | 83933/400000 [00:09<00:34, 9217.05it/s] 21%|██▏       | 85012/400000 [00:09<00:32, 9637.48it/s] 21%|██▏       | 85997/400000 [00:09<00:32, 9700.17it/s] 22%|██▏       | 86972/400000 [00:09<00:32, 9606.01it/s] 22%|██▏       | 87964/400000 [00:09<00:32, 9695.85it/s] 22%|██▏       | 88937/400000 [00:09<00:32, 9635.69it/s] 22%|██▏       | 89903/400000 [00:09<00:32, 9449.18it/s] 23%|██▎       | 90851/400000 [00:09<00:34, 9085.28it/s] 23%|██▎       | 91765/400000 [00:10<00:34, 9036.93it/s] 23%|██▎       | 92672/400000 [00:10<00:34, 8862.93it/s] 23%|██▎       | 93570/400000 [00:10<00:34, 8895.02it/s] 24%|██▎       | 94528/400000 [00:10<00:33, 9087.77it/s] 24%|██▍       | 95440/400000 [00:10<00:34, 8929.93it/s] 24%|██▍       | 96336/400000 [00:10<00:34, 8921.17it/s] 24%|██▍       | 97236/400000 [00:10<00:33, 8944.32it/s] 25%|██▍       | 98142/400000 [00:10<00:33, 8978.46it/s] 25%|██▍       | 99041/400000 [00:10<00:33, 8962.96it/s] 25%|██▍       | 99938/400000 [00:11<00:33, 8927.82it/s] 25%|██▌       | 100849/400000 [00:11<00:33, 8978.11it/s] 25%|██▌       | 101748/400000 [00:11<00:33, 8952.39it/s] 26%|██▌       | 102644/400000 [00:11<00:33, 8858.22it/s] 26%|██▌       | 103545/400000 [00:11<00:33, 8903.05it/s] 26%|██▌       | 104436/400000 [00:11<00:33, 8765.97it/s] 26%|██▋       | 105366/400000 [00:11<00:33, 8918.56it/s] 27%|██▋       | 106260/400000 [00:11<00:33, 8766.79it/s] 27%|██▋       | 107148/400000 [00:11<00:33, 8799.58it/s] 27%|██▋       | 108034/400000 [00:11<00:33, 8817.00it/s] 27%|██▋       | 108983/400000 [00:12<00:32, 9006.73it/s] 27%|██▋       | 109886/400000 [00:12<00:32, 8994.18it/s] 28%|██▊       | 110829/400000 [00:12<00:31, 9119.35it/s] 28%|██▊       | 111758/400000 [00:12<00:31, 9167.59it/s] 28%|██▊       | 112722/400000 [00:12<00:30, 9302.50it/s] 28%|██▊       | 113654/400000 [00:12<00:30, 9276.81it/s] 29%|██▊       | 114583/400000 [00:12<00:31, 9197.61it/s] 29%|██▉       | 115504/400000 [00:12<00:30, 9201.16it/s] 29%|██▉       | 116425/400000 [00:12<00:31, 9139.72it/s] 29%|██▉       | 117389/400000 [00:12<00:30, 9283.83it/s] 30%|██▉       | 118319/400000 [00:13<00:31, 8968.94it/s] 30%|██▉       | 119219/400000 [00:13<00:31, 8831.83it/s] 30%|███       | 120105/400000 [00:13<00:31, 8825.91it/s] 30%|███       | 120990/400000 [00:13<00:32, 8695.38it/s] 30%|███       | 121862/400000 [00:13<00:32, 8684.71it/s] 31%|███       | 122750/400000 [00:13<00:31, 8740.49it/s] 31%|███       | 123682/400000 [00:13<00:31, 8906.24it/s] 31%|███       | 124635/400000 [00:13<00:30, 9084.55it/s] 31%|███▏      | 125604/400000 [00:13<00:29, 9256.93it/s] 32%|███▏      | 126532/400000 [00:13<00:29, 9174.61it/s] 32%|███▏      | 127452/400000 [00:14<00:29, 9124.55it/s] 32%|███▏      | 128366/400000 [00:14<00:30, 9047.04it/s] 32%|███▏      | 129298/400000 [00:14<00:29, 9124.43it/s] 33%|███▎      | 130212/400000 [00:14<00:29, 9106.78it/s] 33%|███▎      | 131166/400000 [00:14<00:29, 9231.26it/s] 33%|███▎      | 132103/400000 [00:14<00:28, 9271.72it/s] 33%|███▎      | 133144/400000 [00:14<00:27, 9584.89it/s] 34%|███▎      | 134204/400000 [00:14<00:26, 9868.23it/s] 34%|███▍      | 135196/400000 [00:14<00:27, 9458.19it/s] 34%|███▍      | 136149/400000 [00:15<00:29, 8940.74it/s] 34%|███▍      | 137054/400000 [00:15<00:29, 8833.80it/s] 34%|███▍      | 137946/400000 [00:15<00:29, 8810.26it/s] 35%|███▍      | 138868/400000 [00:15<00:29, 8927.81it/s] 35%|███▍      | 139765/400000 [00:15<00:29, 8916.71it/s] 35%|███▌      | 140669/400000 [00:15<00:28, 8951.65it/s] 35%|███▌      | 141567/400000 [00:15<00:28, 8926.75it/s] 36%|███▌      | 142473/400000 [00:15<00:28, 8963.68it/s] 36%|███▌      | 143371/400000 [00:15<00:29, 8845.29it/s] 36%|███▌      | 144393/400000 [00:15<00:27, 9216.25it/s] 36%|███▋      | 145325/400000 [00:16<00:27, 9245.68it/s] 37%|███▋      | 146320/400000 [00:16<00:26, 9443.49it/s] 37%|███▋      | 147290/400000 [00:16<00:26, 9516.17it/s] 37%|███▋      | 148245/400000 [00:16<00:26, 9450.34it/s] 37%|███▋      | 149192/400000 [00:16<00:26, 9423.59it/s] 38%|███▊      | 150254/400000 [00:16<00:25, 9751.86it/s] 38%|███▊      | 151234/400000 [00:16<00:25, 9714.05it/s] 38%|███▊      | 152209/400000 [00:16<00:25, 9571.92it/s] 38%|███▊      | 153198/400000 [00:16<00:25, 9662.62it/s] 39%|███▊      | 154205/400000 [00:16<00:25, 9779.91it/s] 39%|███▉      | 155185/400000 [00:17<00:25, 9711.32it/s] 39%|███▉      | 156182/400000 [00:17<00:24, 9785.05it/s] 39%|███▉      | 157194/400000 [00:17<00:24, 9880.96it/s] 40%|███▉      | 158328/400000 [00:17<00:23, 10274.90it/s] 40%|███▉      | 159361/400000 [00:17<00:24, 10025.33it/s] 40%|████      | 160369/400000 [00:17<00:24, 9615.96it/s]  40%|████      | 161347/400000 [00:17<00:24, 9663.09it/s] 41%|████      | 162348/400000 [00:17<00:24, 9764.45it/s] 41%|████      | 163410/400000 [00:17<00:23, 10004.65it/s] 41%|████      | 164415/400000 [00:17<00:24, 9736.93it/s]  41%|████▏     | 165418/400000 [00:18<00:23, 9820.68it/s] 42%|████▏     | 166404/400000 [00:18<00:24, 9656.76it/s] 42%|████▏     | 167373/400000 [00:18<00:24, 9662.89it/s] 42%|████▏     | 168342/400000 [00:18<00:24, 9510.72it/s] 42%|████▏     | 169296/400000 [00:18<00:24, 9375.38it/s] 43%|████▎     | 170236/400000 [00:18<00:24, 9287.61it/s] 43%|████▎     | 171219/400000 [00:18<00:24, 9441.75it/s] 43%|████▎     | 172165/400000 [00:18<00:24, 9435.53it/s] 43%|████▎     | 173230/400000 [00:18<00:23, 9768.32it/s] 44%|████▎     | 174211/400000 [00:18<00:23, 9778.31it/s] 44%|████▍     | 175192/400000 [00:19<00:23, 9632.77it/s] 44%|████▍     | 176170/400000 [00:19<00:23, 9676.47it/s] 44%|████▍     | 177147/400000 [00:19<00:22, 9701.84it/s] 45%|████▍     | 178128/400000 [00:19<00:22, 9731.97it/s] 45%|████▍     | 179103/400000 [00:19<00:23, 9457.60it/s] 45%|████▌     | 180052/400000 [00:19<00:23, 9460.55it/s] 45%|████▌     | 181047/400000 [00:19<00:22, 9601.05it/s] 46%|████▌     | 182026/400000 [00:19<00:22, 9656.40it/s] 46%|████▌     | 183067/400000 [00:19<00:21, 9869.12it/s] 46%|████▌     | 184056/400000 [00:20<00:22, 9528.06it/s] 46%|████▋     | 185013/400000 [00:20<00:23, 9324.01it/s] 46%|████▋     | 185995/400000 [00:20<00:22, 9463.94it/s] 47%|████▋     | 186954/400000 [00:20<00:22, 9500.28it/s] 47%|████▋     | 187932/400000 [00:20<00:22, 9580.56it/s] 47%|████▋     | 188892/400000 [00:20<00:22, 9380.24it/s] 47%|████▋     | 189883/400000 [00:20<00:22, 9529.18it/s] 48%|████▊     | 190851/400000 [00:20<00:21, 9571.11it/s] 48%|████▊     | 191810/400000 [00:20<00:22, 9348.45it/s] 48%|████▊     | 192748/400000 [00:20<00:22, 9280.26it/s] 48%|████▊     | 193678/400000 [00:21<00:22, 9100.73it/s] 49%|████▊     | 194662/400000 [00:21<00:22, 9309.53it/s] 49%|████▉     | 195596/400000 [00:21<00:22, 9231.39it/s] 49%|████▉     | 196565/400000 [00:21<00:21, 9362.39it/s] 49%|████▉     | 197504/400000 [00:21<00:21, 9354.66it/s] 50%|████▉     | 198441/400000 [00:21<00:21, 9235.01it/s] 50%|████▉     | 199383/400000 [00:21<00:21, 9287.31it/s] 50%|█████     | 200323/400000 [00:21<00:21, 9317.99it/s] 50%|█████     | 201256/400000 [00:21<00:21, 9176.80it/s] 51%|█████     | 202203/400000 [00:21<00:21, 9261.13it/s] 51%|█████     | 203130/400000 [00:22<00:21, 9111.92it/s] 51%|█████     | 204043/400000 [00:22<00:21, 9094.37it/s] 51%|█████     | 204974/400000 [00:22<00:21, 9155.33it/s] 51%|█████▏    | 205948/400000 [00:22<00:20, 9320.94it/s] 52%|█████▏    | 206912/400000 [00:22<00:20, 9411.79it/s] 52%|█████▏    | 207855/400000 [00:22<00:20, 9353.37it/s] 52%|█████▏    | 208871/400000 [00:22<00:19, 9580.09it/s] 52%|█████▏    | 209912/400000 [00:22<00:19, 9813.02it/s] 53%|█████▎    | 210904/400000 [00:22<00:19, 9843.30it/s] 53%|█████▎    | 211930/400000 [00:22<00:18, 9962.84it/s] 53%|█████▎    | 212928/400000 [00:23<00:18, 9941.92it/s] 53%|█████▎    | 213924/400000 [00:23<00:18, 9857.10it/s] 54%|█████▎    | 214930/400000 [00:23<00:18, 9916.07it/s] 54%|█████▍    | 215959/400000 [00:23<00:18, 10024.17it/s] 54%|█████▍    | 216986/400000 [00:23<00:18, 10093.91it/s] 54%|█████▍    | 217997/400000 [00:23<00:18, 9977.51it/s]  55%|█████▍    | 219028/400000 [00:23<00:17, 10071.11it/s] 55%|█████▌    | 220036/400000 [00:23<00:18, 9639.78it/s]  55%|█████▌    | 221005/400000 [00:23<00:18, 9499.38it/s] 55%|█████▌    | 221982/400000 [00:23<00:18, 9577.07it/s] 56%|█████▌    | 223001/400000 [00:24<00:18, 9751.04it/s] 56%|█████▌    | 223979/400000 [00:24<00:18, 9706.99it/s] 56%|█████▌    | 224952/400000 [00:24<00:18, 9710.30it/s] 56%|█████▋    | 225975/400000 [00:24<00:17, 9859.94it/s] 57%|█████▋    | 226963/400000 [00:24<00:17, 9700.64it/s] 57%|█████▋    | 227935/400000 [00:24<00:17, 9660.26it/s] 57%|█████▋    | 228903/400000 [00:24<00:17, 9622.71it/s] 57%|█████▋    | 229924/400000 [00:24<00:17, 9790.83it/s] 58%|█████▊    | 230952/400000 [00:24<00:17, 9930.71it/s] 58%|█████▊    | 231947/400000 [00:25<00:16, 9905.72it/s] 58%|█████▊    | 232939/400000 [00:25<00:17, 9552.52it/s] 58%|█████▊    | 233922/400000 [00:25<00:17, 9631.58it/s] 59%|█████▊    | 234888/400000 [00:25<00:17, 9517.79it/s] 59%|█████▉    | 235842/400000 [00:25<00:17, 9438.70it/s] 59%|█████▉    | 236822/400000 [00:25<00:17, 9543.42it/s] 59%|█████▉    | 237824/400000 [00:25<00:16, 9680.05it/s] 60%|█████▉    | 238821/400000 [00:25<00:16, 9764.53it/s] 60%|█████▉    | 239849/400000 [00:25<00:16, 9912.84it/s] 60%|██████    | 240858/400000 [00:25<00:15, 9962.56it/s] 60%|██████    | 241856/400000 [00:26<00:16, 9693.17it/s] 61%|██████    | 242828/400000 [00:26<00:16, 9566.83it/s] 61%|██████    | 243787/400000 [00:26<00:16, 9551.90it/s] 61%|██████    | 244744/400000 [00:26<00:16, 9449.35it/s] 61%|██████▏   | 245720/400000 [00:26<00:16, 9539.55it/s] 62%|██████▏   | 246680/400000 [00:26<00:16, 9556.67it/s] 62%|██████▏   | 247659/400000 [00:26<00:15, 9624.67it/s] 62%|██████▏   | 248728/400000 [00:26<00:15, 9918.34it/s] 62%|██████▏   | 249725/400000 [00:26<00:15, 9931.92it/s] 63%|██████▎   | 250748/400000 [00:26<00:14, 10018.24it/s] 63%|██████▎   | 251765/400000 [00:27<00:14, 10062.00it/s] 63%|██████▎   | 252773/400000 [00:27<00:14, 9919.77it/s]  63%|██████▎   | 253773/400000 [00:27<00:14, 9940.17it/s] 64%|██████▎   | 254768/400000 [00:27<00:14, 9855.16it/s] 64%|██████▍   | 255779/400000 [00:27<00:14, 9928.70it/s] 64%|██████▍   | 256773/400000 [00:27<00:14, 9909.94it/s] 64%|██████▍   | 257765/400000 [00:27<00:14, 9776.85it/s] 65%|██████▍   | 258744/400000 [00:27<00:14, 9492.33it/s] 65%|██████▍   | 259761/400000 [00:27<00:14, 9685.60it/s] 65%|██████▌   | 260733/400000 [00:27<00:14, 9635.36it/s] 65%|██████▌   | 261733/400000 [00:28<00:14, 9740.17it/s] 66%|██████▌   | 262709/400000 [00:28<00:14, 9520.50it/s] 66%|██████▌   | 263683/400000 [00:28<00:14, 9582.69it/s] 66%|██████▌   | 264643/400000 [00:28<00:14, 9442.14it/s] 66%|██████▋   | 265646/400000 [00:28<00:13, 9608.22it/s] 67%|██████▋   | 266609/400000 [00:28<00:13, 9565.72it/s] 67%|██████▋   | 267567/400000 [00:28<00:14, 9317.74it/s] 67%|██████▋   | 268502/400000 [00:28<00:14, 9288.40it/s] 67%|██████▋   | 269550/400000 [00:28<00:13, 9612.19it/s] 68%|██████▊   | 270516/400000 [00:29<00:13, 9346.01it/s] 68%|██████▊   | 271456/400000 [00:29<00:13, 9296.08it/s] 68%|██████▊   | 272414/400000 [00:29<00:13, 9378.76it/s] 68%|██████▊   | 273361/400000 [00:29<00:13, 9405.01it/s] 69%|██████▊   | 274360/400000 [00:29<00:13, 9571.88it/s] 69%|██████▉   | 275349/400000 [00:29<00:12, 9664.06it/s] 69%|██████▉   | 276317/400000 [00:29<00:12, 9556.94it/s] 69%|██████▉   | 277291/400000 [00:29<00:12, 9609.48it/s] 70%|██████▉   | 278303/400000 [00:29<00:12, 9756.94it/s] 70%|██████▉   | 279352/400000 [00:29<00:12, 9965.77it/s] 70%|███████   | 280351/400000 [00:30<00:12, 9740.30it/s] 70%|███████   | 281328/400000 [00:30<00:12, 9531.03it/s] 71%|███████   | 282284/400000 [00:30<00:12, 9529.92it/s] 71%|███████   | 283239/400000 [00:30<00:12, 9390.35it/s] 71%|███████   | 284180/400000 [00:30<00:12, 9382.95it/s] 71%|███████▏  | 285126/400000 [00:30<00:12, 9405.70it/s] 72%|███████▏  | 286085/400000 [00:30<00:12, 9459.41it/s] 72%|███████▏  | 287032/400000 [00:30<00:11, 9438.48it/s] 72%|███████▏  | 287977/400000 [00:30<00:12, 9165.95it/s] 72%|███████▏  | 288996/400000 [00:30<00:11, 9449.74it/s] 72%|███████▏  | 289982/400000 [00:31<00:11, 9566.64it/s] 73%|███████▎  | 290972/400000 [00:31<00:11, 9662.20it/s] 73%|███████▎  | 291941/400000 [00:31<00:11, 9645.08it/s] 73%|███████▎  | 292908/400000 [00:31<00:11, 9539.83it/s] 73%|███████▎  | 293880/400000 [00:31<00:11, 9590.32it/s] 74%|███████▎  | 294854/400000 [00:31<00:10, 9633.99it/s] 74%|███████▍  | 295819/400000 [00:31<00:11, 9286.16it/s] 74%|███████▍  | 296751/400000 [00:31<00:11, 9120.05it/s] 74%|███████▍  | 297701/400000 [00:31<00:11, 9228.96it/s] 75%|███████▍  | 298658/400000 [00:31<00:10, 9327.78it/s] 75%|███████▍  | 299593/400000 [00:32<00:10, 9182.71it/s] 75%|███████▌  | 300565/400000 [00:32<00:10, 9334.57it/s] 75%|███████▌  | 301501/400000 [00:32<00:10, 9116.46it/s] 76%|███████▌  | 302502/400000 [00:32<00:10, 9367.15it/s] 76%|███████▌  | 303443/400000 [00:32<00:10, 9243.62it/s] 76%|███████▌  | 304484/400000 [00:32<00:09, 9564.36it/s] 76%|███████▋  | 305540/400000 [00:32<00:09, 9841.33it/s] 77%|███████▋  | 306592/400000 [00:32<00:09, 10034.53it/s] 77%|███████▋  | 307630/400000 [00:32<00:09, 10134.54it/s] 77%|███████▋  | 308730/400000 [00:32<00:08, 10378.25it/s] 77%|███████▋  | 309772/400000 [00:33<00:08, 10344.25it/s] 78%|███████▊  | 310810/400000 [00:33<00:08, 10278.66it/s] 78%|███████▊  | 311840/400000 [00:33<00:08, 9952.66it/s]  78%|███████▊  | 312839/400000 [00:33<00:09, 9547.18it/s] 78%|███████▊  | 313800/400000 [00:33<00:09, 9399.22it/s] 79%|███████▊  | 314745/400000 [00:33<00:09, 9378.26it/s] 79%|███████▉  | 315687/400000 [00:33<00:09, 9284.62it/s] 79%|███████▉  | 316619/400000 [00:33<00:09, 9148.94it/s] 79%|███████▉  | 317542/400000 [00:33<00:08, 9171.17it/s] 80%|███████▉  | 318480/400000 [00:34<00:08, 9232.47it/s] 80%|███████▉  | 319405/400000 [00:34<00:08, 9052.78it/s] 80%|████████  | 320344/400000 [00:34<00:08, 9150.14it/s] 80%|████████  | 321329/400000 [00:34<00:08, 9347.95it/s] 81%|████████  | 322266/400000 [00:34<00:08, 9349.23it/s] 81%|████████  | 323269/400000 [00:34<00:08, 9542.31it/s] 81%|████████  | 324256/400000 [00:34<00:07, 9636.38it/s] 81%|████████▏ | 325222/400000 [00:34<00:07, 9616.20it/s] 82%|████████▏ | 326242/400000 [00:34<00:07, 9782.87it/s] 82%|████████▏ | 327263/400000 [00:34<00:07, 9906.41it/s] 82%|████████▏ | 328256/400000 [00:35<00:07, 9493.34it/s] 82%|████████▏ | 329210/400000 [00:35<00:07, 9484.63it/s] 83%|████████▎ | 330162/400000 [00:35<00:07, 9480.45it/s] 83%|████████▎ | 331118/400000 [00:35<00:07, 9503.78it/s] 83%|████████▎ | 332070/400000 [00:35<00:07, 9382.24it/s] 83%|████████▎ | 333020/400000 [00:35<00:07, 9414.61it/s] 83%|████████▎ | 333963/400000 [00:35<00:07, 9055.16it/s] 84%|████████▎ | 334881/400000 [00:35<00:07, 9090.22it/s] 84%|████████▍ | 335869/400000 [00:35<00:06, 9311.54it/s] 84%|████████▍ | 336857/400000 [00:35<00:06, 9473.60it/s] 84%|████████▍ | 337808/400000 [00:36<00:06, 9432.80it/s] 85%|████████▍ | 338804/400000 [00:36<00:06, 9584.16it/s] 85%|████████▍ | 339838/400000 [00:36<00:06, 9797.24it/s] 85%|████████▌ | 340821/400000 [00:36<00:06, 9787.39it/s] 85%|████████▌ | 341802/400000 [00:36<00:05, 9769.74it/s] 86%|████████▌ | 342791/400000 [00:36<00:05, 9805.46it/s] 86%|████████▌ | 343773/400000 [00:36<00:05, 9547.54it/s] 86%|████████▌ | 344730/400000 [00:36<00:05, 9378.80it/s] 86%|████████▋ | 345671/400000 [00:36<00:05, 9203.39it/s] 87%|████████▋ | 346688/400000 [00:37<00:05, 9472.29it/s] 87%|████████▋ | 347639/400000 [00:37<00:05, 9322.05it/s] 87%|████████▋ | 348585/400000 [00:37<00:05, 9360.89it/s] 87%|████████▋ | 349524/400000 [00:37<00:05, 9224.25it/s] 88%|████████▊ | 350449/400000 [00:37<00:05, 9228.97it/s] 88%|████████▊ | 351403/400000 [00:37<00:05, 9319.96it/s] 88%|████████▊ | 352363/400000 [00:37<00:05, 9399.28it/s] 88%|████████▊ | 353304/400000 [00:37<00:05, 9183.04it/s] 89%|████████▊ | 354225/400000 [00:37<00:04, 9171.23it/s] 89%|████████▉ | 355144/400000 [00:37<00:04, 9075.04it/s] 89%|████████▉ | 356140/400000 [00:38<00:04, 9321.33it/s] 89%|████████▉ | 357075/400000 [00:38<00:04, 9136.44it/s] 90%|████████▉ | 358032/400000 [00:38<00:04, 9259.93it/s] 90%|████████▉ | 358985/400000 [00:38<00:04, 9336.97it/s] 90%|████████▉ | 359921/400000 [00:38<00:04, 9325.11it/s] 90%|█████████ | 360855/400000 [00:38<00:04, 9225.90it/s] 90%|█████████ | 361800/400000 [00:38<00:04, 9291.11it/s] 91%|█████████ | 362730/400000 [00:38<00:04, 9248.77it/s] 91%|█████████ | 363656/400000 [00:38<00:04, 9074.77it/s] 91%|█████████ | 364565/400000 [00:38<00:03, 9033.51it/s] 91%|█████████▏| 365540/400000 [00:39<00:03, 9236.68it/s] 92%|█████████▏| 366562/400000 [00:39<00:03, 9509.41it/s] 92%|█████████▏| 367608/400000 [00:39<00:03, 9773.94it/s] 92%|█████████▏| 368590/400000 [00:39<00:03, 9688.14it/s] 92%|█████████▏| 369562/400000 [00:39<00:03, 9639.40it/s] 93%|█████████▎| 370529/400000 [00:39<00:03, 9424.14it/s] 93%|█████████▎| 371529/400000 [00:39<00:02, 9589.49it/s] 93%|█████████▎| 372549/400000 [00:39<00:02, 9763.28it/s] 93%|█████████▎| 373528/400000 [00:39<00:02, 9718.54it/s] 94%|█████████▎| 374522/400000 [00:39<00:02, 9782.06it/s] 94%|█████████▍| 375541/400000 [00:40<00:02, 9900.74it/s] 94%|█████████▍| 376533/400000 [00:40<00:02, 9819.14it/s] 94%|█████████▍| 377540/400000 [00:40<00:02, 9890.86it/s] 95%|█████████▍| 378530/400000 [00:40<00:02, 9603.48it/s] 95%|█████████▍| 379493/400000 [00:40<00:02, 9470.94it/s] 95%|█████████▌| 380443/400000 [00:40<00:02, 9215.38it/s] 95%|█████████▌| 381368/400000 [00:40<00:02, 9078.04it/s] 96%|█████████▌| 382279/400000 [00:40<00:01, 8864.60it/s] 96%|█████████▌| 383169/400000 [00:40<00:01, 8580.05it/s] 96%|█████████▌| 384080/400000 [00:41<00:01, 8731.39it/s] 96%|█████████▋| 385018/400000 [00:41<00:01, 8916.19it/s] 96%|█████████▋| 385931/400000 [00:41<00:01, 8977.55it/s] 97%|█████████▋| 386846/400000 [00:41<00:01, 9027.36it/s] 97%|█████████▋| 387751/400000 [00:41<00:01, 9033.42it/s] 97%|█████████▋| 388656/400000 [00:41<00:01, 8858.73it/s] 97%|█████████▋| 389640/400000 [00:41<00:01, 9129.81it/s] 98%|█████████▊| 390591/400000 [00:41<00:01, 9240.08it/s] 98%|█████████▊| 391595/400000 [00:41<00:00, 9464.90it/s] 98%|█████████▊| 392545/400000 [00:41<00:00, 9236.29it/s] 98%|█████████▊| 393473/400000 [00:42<00:00, 9102.08it/s] 99%|█████████▊| 394453/400000 [00:42<00:00, 9298.88it/s] 99%|█████████▉| 395493/400000 [00:42<00:00, 9603.70it/s] 99%|█████████▉| 396459/400000 [00:42<00:00, 9482.01it/s] 99%|█████████▉| 397411/400000 [00:42<00:00, 9400.75it/s]100%|█████████▉| 398354/400000 [00:42<00:00, 8841.92it/s]100%|█████████▉| 399247/400000 [00:42<00:00, 8536.53it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9352.52it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f522dfb6940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011030330848966716 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010946048542009947 	 Accuracy: 68

  model saves at 68% accuracy 

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
2020-05-15 07:24:03.484704: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 07:24:03.488635: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-15 07:24:03.489354: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56366a457040 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 07:24:03.489371: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5230eab588> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6871 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6223 - accuracy: 0.5029
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
11000/25000 [============>.................] - ETA: 3s - loss: 7.6220 - accuracy: 0.5029
12000/25000 [=============>................] - ETA: 3s - loss: 7.6385 - accuracy: 0.5018
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5923 - accuracy: 0.5048
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6173 - accuracy: 0.5032
15000/25000 [=================>............] - ETA: 2s - loss: 7.6227 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6149 - accuracy: 0.5034
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6326 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6739 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6794 - accuracy: 0.4992
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f5192e3c6a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f51d3ac3e80> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9115 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.9011 - val_crf_viterbi_accuracy: 0.2533

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

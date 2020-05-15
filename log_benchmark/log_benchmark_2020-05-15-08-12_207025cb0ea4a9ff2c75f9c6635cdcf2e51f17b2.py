
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f24473edf98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 08:13:10.924120
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 08:13:10.927880
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 08:13:10.930902
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 08:13:10.933943
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f24531b7400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356186.8125
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 316797.8125
Epoch 3/10

1/1 [==============================] - 0s 105ms/step - loss: 246380.0781
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 168806.2188
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 105961.1250
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 63927.0547
Epoch 7/10

1/1 [==============================] - 0s 96ms/step - loss: 39466.6992
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 25710.2617
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 17656.4043
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 12715.1816

  #### Inference Need return ypred, ytrue ######################### 
[[-7.98461556e-01  8.64452124e-03  6.92812979e-01  1.29590487e+00
   1.03928900e+00  4.56491113e-03  4.65645548e-03  9.09575880e-01
  -8.86396766e-01  5.02008915e-01  8.34403992e-01 -6.08483016e-01
   1.24852583e-01 -7.95845985e-01  4.40926552e-01  9.93569493e-01
  -1.87276989e-01 -3.54780018e-01  1.14411139e+00 -2.08518952e-02
  -5.77989340e-01  1.19618803e-01 -4.02813852e-01 -3.34126204e-01
   1.20310426e+00 -4.43649411e-01  4.54631388e-01 -1.52166113e-01
  -5.80618501e-01 -3.71898174e-01  7.59087086e-01 -5.77666283e-01
  -1.71518371e-01  7.51125813e-01 -4.66132402e-01 -6.07912123e-01
   7.02191114e-01 -5.56087852e-01 -6.79215550e-01 -3.72288316e-01
   8.33095193e-01 -7.49139965e-01  3.70626092e-01  3.22271615e-01
  -1.36064184e+00 -5.97645283e-01  8.16077828e-01 -1.94043726e-01
  -1.16483939e+00  1.01768470e+00  3.14725637e-02 -5.39647937e-01
   4.06849414e-01 -3.47431302e-01  7.33164191e-01 -3.95774007e-01
  -2.46162444e-01 -8.36778760e-01  6.22260273e-02  5.08033395e-01
  -1.21878386e-02  6.85954213e-01 -5.47665060e-01  3.91541123e-01
  -4.53866273e-01  3.15000385e-01  5.16215444e-01 -5.43251336e-01
  -1.00898492e+00  3.44983041e-01 -2.28868261e-01  4.12425697e-01
  -7.42079377e-01 -1.36047292e+00  4.56557155e-01 -6.40230179e-01
   8.23392272e-01 -1.06886804e-01 -6.97950721e-01 -3.93447369e-01
   1.74404100e-01  5.44078350e-02 -5.71454644e-01 -4.79386270e-01
  -3.68566990e-01  1.13040519e+00 -9.26277339e-02 -4.88468200e-01
   1.38591856e-01 -6.44562125e-01 -4.72422540e-01  5.93522012e-01
   1.90529436e-01 -2.86172807e-01  8.26245546e-02  4.55494016e-01
   3.57226610e-01  5.08453667e-01  1.33998656e+00 -7.35682905e-01
   9.84759569e-01  1.21346667e-01 -1.59096432e+00 -3.78801465e-01
   1.79255903e-01 -4.38279033e-01 -9.44725499e-02 -1.97196603e-01
  -1.85164559e+00  1.30734336e+00 -3.65983933e-01 -1.35385084e+00
   1.52249980e+00 -1.05183840e+00  5.04450917e-01  1.17079329e+00
   7.66421616e-01  3.78886998e-01  4.80625778e-01  5.20128787e-01
  -1.32516891e-01  6.16118574e+00  5.75406122e+00  5.40333414e+00
   6.63102102e+00  5.21047068e+00  7.07647753e+00  4.88325834e+00
   5.35431242e+00  6.75984478e+00  4.64775753e+00  5.97828197e+00
   5.58714914e+00  5.56923914e+00  5.39346886e+00  4.82176161e+00
   6.27259302e+00  5.21114731e+00  5.66448927e+00  6.26440716e+00
   5.69774199e+00  5.76147699e+00  5.73954916e+00  6.00982523e+00
   5.27191830e+00  5.95434046e+00  5.62752056e+00  5.57443285e+00
   5.75316811e+00  5.73585081e+00  6.02224398e+00  5.22012663e+00
   5.86090136e+00  5.76322412e+00  4.77456570e+00  5.14269304e+00
   4.96435261e+00  5.37571955e+00  6.24124002e+00  4.76107454e+00
   6.05836630e+00  6.52273941e+00  4.37835026e+00  6.26119757e+00
   6.64884377e+00  6.64229584e+00  6.40260839e+00  6.57358217e+00
   6.76955938e+00  5.36666012e+00  5.56455708e+00  4.32879877e+00
   5.13100481e+00  5.23460054e+00  6.04719687e+00  5.50147057e+00
   5.33896542e+00  5.34837341e+00  5.63347530e+00  5.48872280e+00
   6.46997213e-01  1.48905468e+00  7.23309636e-01  1.85323572e+00
   2.08132315e+00  1.87953305e+00  2.36380434e+00  1.31830525e+00
   1.62707913e+00  2.54847884e-01  1.49801803e+00  2.28148222e+00
   3.20683539e-01  2.08802319e+00  7.08364904e-01  1.40738988e+00
   2.20095277e-01  4.31243658e-01  7.86445558e-01  8.21792185e-01
   1.07822025e+00  1.24185038e+00  1.54700387e+00  1.36093128e+00
   1.59239841e+00  1.45054162e+00  5.92532516e-01  9.69137788e-01
   2.12147617e+00  4.82081652e-01  1.39404011e+00  1.04226756e+00
   4.21761811e-01  1.63059986e+00  1.94910288e+00  1.95742154e+00
   7.70140886e-01  5.28411567e-01  5.32355666e-01  1.34163976e+00
   8.79908144e-01  1.74520910e+00  2.06165731e-01  1.17192638e+00
   7.35227764e-01  5.20175040e-01  9.30983841e-01  1.55019927e+00
   8.41609597e-01  8.60749364e-01  2.08383942e+00  5.91586053e-01
   2.40760326e+00  1.77061057e+00  8.46269488e-01  1.93376207e+00
   5.72641432e-01  1.48241401e+00  8.54685307e-01  9.08090711e-01
   1.57188356e+00  4.97648656e-01  8.54799747e-01  2.10045171e+00
   1.27832210e+00  2.29915476e+00  6.25891328e-01  1.58051562e+00
   8.12191427e-01  9.18753147e-01  1.34931469e+00  1.31008506e+00
   1.31697273e+00  1.02072012e+00  5.28367400e-01  4.95761037e-01
   8.69298816e-01  7.48390436e-01  2.19280624e+00  1.79260135e+00
   1.17985499e+00  1.10481048e+00  3.20852578e-01  1.28121400e+00
   1.08636892e+00  1.27729058e+00  8.82198513e-01  3.62888277e-01
   6.62701011e-01  8.20360184e-01  2.13129950e+00  4.59904790e-01
   2.40581870e-01  9.42278445e-01  2.18197346e+00  3.45230818e-01
   1.91582370e+00  8.60186636e-01  4.26189065e-01  9.90457714e-01
   1.56001496e+00  9.95167434e-01  8.06690335e-01  4.54471767e-01
   3.44715595e-01  1.25598359e+00  1.58605909e+00  1.95933223e+00
   1.79352009e+00  1.82718718e+00  1.50361824e+00  2.94737697e-01
   2.93559670e-01  7.91906595e-01  1.36205411e+00  1.03448331e+00
   1.40243554e+00  1.31322253e+00  7.89145112e-01  1.47382605e+00
   3.24478745e-02  5.44650507e+00  6.12760544e+00  5.12087393e+00
   6.67826366e+00  5.98044395e+00  5.40630674e+00  5.95946026e+00
   6.36830711e+00  6.71657085e+00  6.68233824e+00  6.45005512e+00
   6.71740150e+00  7.27425671e+00  6.59034443e+00  6.64871979e+00
   4.62949276e+00  5.39523697e+00  5.12034750e+00  5.74151182e+00
   5.68972588e+00  6.22517443e+00  5.77879906e+00  5.32270908e+00
   6.96442127e+00  6.19473219e+00  7.12873173e+00  6.51502371e+00
   6.41955805e+00  6.40973330e+00  6.31141901e+00  6.21003485e+00
   5.99396229e+00  6.70162201e+00  6.03256321e+00  6.45189285e+00
   5.97213316e+00  6.38056087e+00  5.67163134e+00  6.65575647e+00
   6.78663635e+00  5.77520275e+00  6.40853643e+00  5.14956188e+00
   6.22907257e+00  5.72836876e+00  5.47571373e+00  5.74501562e+00
   6.99960041e+00  6.28273678e+00  5.61581850e+00  6.47898483e+00
   6.51615858e+00  6.33117294e+00  5.20154428e+00  5.34633636e+00
   6.72870684e+00  6.71911001e+00  6.65211868e+00  6.05725622e+00
  -6.14246607e+00 -1.83225393e+00  5.70951557e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 08:13:19.453078
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.2908
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 08:13:19.457311
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9290.34
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 08:13:19.460436
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.491
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 08:13:19.463547
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -831.016
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139793431703960
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139792490177144
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139792490177648
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139792490178152
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139792490178656
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139792490179160

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2440b24550> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.509366
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.480646
grad_step = 000002, loss = 0.458522
grad_step = 000003, loss = 0.432311
grad_step = 000004, loss = 0.405151
grad_step = 000005, loss = 0.381491
grad_step = 000006, loss = 0.364390
grad_step = 000007, loss = 0.348325
grad_step = 000008, loss = 0.331926
grad_step = 000009, loss = 0.318462
grad_step = 000010, loss = 0.303933
grad_step = 000011, loss = 0.291663
grad_step = 000012, loss = 0.282471
grad_step = 000013, loss = 0.274801
grad_step = 000014, loss = 0.266421
grad_step = 000015, loss = 0.256874
grad_step = 000016, loss = 0.246639
grad_step = 000017, loss = 0.236547
grad_step = 000018, loss = 0.227224
grad_step = 000019, loss = 0.218302
grad_step = 000020, loss = 0.209536
grad_step = 000021, loss = 0.201039
grad_step = 000022, loss = 0.192517
grad_step = 000023, loss = 0.184135
grad_step = 000024, loss = 0.176246
grad_step = 000025, loss = 0.168687
grad_step = 000026, loss = 0.161079
grad_step = 000027, loss = 0.153457
grad_step = 000028, loss = 0.145925
grad_step = 000029, loss = 0.138449
grad_step = 000030, loss = 0.131345
grad_step = 000031, loss = 0.124908
grad_step = 000032, loss = 0.118750
grad_step = 000033, loss = 0.112399
grad_step = 000034, loss = 0.106117
grad_step = 000035, loss = 0.100304
grad_step = 000036, loss = 0.094738
grad_step = 000037, loss = 0.089155
grad_step = 000038, loss = 0.083799
grad_step = 000039, loss = 0.078803
grad_step = 000040, loss = 0.074024
grad_step = 000041, loss = 0.069454
grad_step = 000042, loss = 0.065159
grad_step = 000043, loss = 0.060988
grad_step = 000044, loss = 0.056864
grad_step = 000045, loss = 0.053065
grad_step = 000046, loss = 0.049579
grad_step = 000047, loss = 0.046168
grad_step = 000048, loss = 0.042961
grad_step = 000049, loss = 0.040010
grad_step = 000050, loss = 0.037169
grad_step = 000051, loss = 0.034529
grad_step = 000052, loss = 0.032061
grad_step = 000053, loss = 0.029722
grad_step = 000054, loss = 0.027581
grad_step = 000055, loss = 0.025561
grad_step = 000056, loss = 0.023679
grad_step = 000057, loss = 0.021891
grad_step = 000058, loss = 0.020198
grad_step = 000059, loss = 0.018647
grad_step = 000060, loss = 0.017246
grad_step = 000061, loss = 0.015958
grad_step = 000062, loss = 0.014768
grad_step = 000063, loss = 0.013678
grad_step = 000064, loss = 0.012695
grad_step = 000065, loss = 0.011804
grad_step = 000066, loss = 0.010974
grad_step = 000067, loss = 0.010187
grad_step = 000068, loss = 0.009466
grad_step = 000069, loss = 0.008804
grad_step = 000070, loss = 0.008225
grad_step = 000071, loss = 0.007702
grad_step = 000072, loss = 0.007224
grad_step = 000073, loss = 0.006792
grad_step = 000074, loss = 0.006402
grad_step = 000075, loss = 0.006042
grad_step = 000076, loss = 0.005698
grad_step = 000077, loss = 0.005380
grad_step = 000078, loss = 0.005089
grad_step = 000079, loss = 0.004826
grad_step = 000080, loss = 0.004586
grad_step = 000081, loss = 0.004366
grad_step = 000082, loss = 0.004163
grad_step = 000083, loss = 0.003978
grad_step = 000084, loss = 0.003806
grad_step = 000085, loss = 0.003647
grad_step = 000086, loss = 0.003499
grad_step = 000087, loss = 0.003365
grad_step = 000088, loss = 0.003245
grad_step = 000089, loss = 0.003138
grad_step = 000090, loss = 0.003042
grad_step = 000091, loss = 0.002956
grad_step = 000092, loss = 0.002879
grad_step = 000093, loss = 0.002807
grad_step = 000094, loss = 0.002741
grad_step = 000095, loss = 0.002682
grad_step = 000096, loss = 0.002630
grad_step = 000097, loss = 0.002582
grad_step = 000098, loss = 0.002539
grad_step = 000099, loss = 0.002501
grad_step = 000100, loss = 0.002467
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002436
grad_step = 000102, loss = 0.002407
grad_step = 000103, loss = 0.002380
grad_step = 000104, loss = 0.002357
grad_step = 000105, loss = 0.002337
grad_step = 000106, loss = 0.002320
grad_step = 000107, loss = 0.002304
grad_step = 000108, loss = 0.002290
grad_step = 000109, loss = 0.002278
grad_step = 000110, loss = 0.002268
grad_step = 000111, loss = 0.002258
grad_step = 000112, loss = 0.002250
grad_step = 000113, loss = 0.002243
grad_step = 000114, loss = 0.002237
grad_step = 000115, loss = 0.002232
grad_step = 000116, loss = 0.002228
grad_step = 000117, loss = 0.002224
grad_step = 000118, loss = 0.002220
grad_step = 000119, loss = 0.002217
grad_step = 000120, loss = 0.002214
grad_step = 000121, loss = 0.002212
grad_step = 000122, loss = 0.002210
grad_step = 000123, loss = 0.002207
grad_step = 000124, loss = 0.002205
grad_step = 000125, loss = 0.002203
grad_step = 000126, loss = 0.002202
grad_step = 000127, loss = 0.002200
grad_step = 000128, loss = 0.002198
grad_step = 000129, loss = 0.002197
grad_step = 000130, loss = 0.002195
grad_step = 000131, loss = 0.002193
grad_step = 000132, loss = 0.002190
grad_step = 000133, loss = 0.002188
grad_step = 000134, loss = 0.002187
grad_step = 000135, loss = 0.002185
grad_step = 000136, loss = 0.002182
grad_step = 000137, loss = 0.002180
grad_step = 000138, loss = 0.002178
grad_step = 000139, loss = 0.002176
grad_step = 000140, loss = 0.002174
grad_step = 000141, loss = 0.002171
grad_step = 000142, loss = 0.002169
grad_step = 000143, loss = 0.002167
grad_step = 000144, loss = 0.002165
grad_step = 000145, loss = 0.002163
grad_step = 000146, loss = 0.002161
grad_step = 000147, loss = 0.002159
grad_step = 000148, loss = 0.002157
grad_step = 000149, loss = 0.002154
grad_step = 000150, loss = 0.002153
grad_step = 000151, loss = 0.002151
grad_step = 000152, loss = 0.002149
grad_step = 000153, loss = 0.002148
grad_step = 000154, loss = 0.002146
grad_step = 000155, loss = 0.002145
grad_step = 000156, loss = 0.002144
grad_step = 000157, loss = 0.002143
grad_step = 000158, loss = 0.002141
grad_step = 000159, loss = 0.002138
grad_step = 000160, loss = 0.002135
grad_step = 000161, loss = 0.002132
grad_step = 000162, loss = 0.002130
grad_step = 000163, loss = 0.002127
grad_step = 000164, loss = 0.002126
grad_step = 000165, loss = 0.002124
grad_step = 000166, loss = 0.002124
grad_step = 000167, loss = 0.002123
grad_step = 000168, loss = 0.002124
grad_step = 000169, loss = 0.002126
grad_step = 000170, loss = 0.002130
grad_step = 000171, loss = 0.002134
grad_step = 000172, loss = 0.002133
grad_step = 000173, loss = 0.002126
grad_step = 000174, loss = 0.002115
grad_step = 000175, loss = 0.002107
grad_step = 000176, loss = 0.002104
grad_step = 000177, loss = 0.002107
grad_step = 000178, loss = 0.002110
grad_step = 000179, loss = 0.002112
grad_step = 000180, loss = 0.002110
grad_step = 000181, loss = 0.002104
grad_step = 000182, loss = 0.002098
grad_step = 000183, loss = 0.002092
grad_step = 000184, loss = 0.002090
grad_step = 000185, loss = 0.002090
grad_step = 000186, loss = 0.002091
grad_step = 000187, loss = 0.002092
grad_step = 000188, loss = 0.002093
grad_step = 000189, loss = 0.002094
grad_step = 000190, loss = 0.002095
grad_step = 000191, loss = 0.002094
grad_step = 000192, loss = 0.002092
grad_step = 000193, loss = 0.002089
grad_step = 000194, loss = 0.002085
grad_step = 000195, loss = 0.002081
grad_step = 000196, loss = 0.002076
grad_step = 000197, loss = 0.002072
grad_step = 000198, loss = 0.002068
grad_step = 000199, loss = 0.002065
grad_step = 000200, loss = 0.002062
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002060
grad_step = 000202, loss = 0.002058
grad_step = 000203, loss = 0.002056
grad_step = 000204, loss = 0.002054
grad_step = 000205, loss = 0.002052
grad_step = 000206, loss = 0.002050
grad_step = 000207, loss = 0.002048
grad_step = 000208, loss = 0.002047
grad_step = 000209, loss = 0.002047
grad_step = 000210, loss = 0.002054
grad_step = 000211, loss = 0.002079
grad_step = 000212, loss = 0.002155
grad_step = 000213, loss = 0.002336
grad_step = 000214, loss = 0.002572
grad_step = 000215, loss = 0.002568
grad_step = 000216, loss = 0.002207
grad_step = 000217, loss = 0.002047
grad_step = 000218, loss = 0.002293
grad_step = 000219, loss = 0.002354
grad_step = 000220, loss = 0.002078
grad_step = 000221, loss = 0.002100
grad_step = 000222, loss = 0.002279
grad_step = 000223, loss = 0.002126
grad_step = 000224, loss = 0.002036
grad_step = 000225, loss = 0.002177
grad_step = 000226, loss = 0.002129
grad_step = 000227, loss = 0.002022
grad_step = 000228, loss = 0.002107
grad_step = 000229, loss = 0.002112
grad_step = 000230, loss = 0.002023
grad_step = 000231, loss = 0.002060
grad_step = 000232, loss = 0.002093
grad_step = 000233, loss = 0.002024
grad_step = 000234, loss = 0.002031
grad_step = 000235, loss = 0.002068
grad_step = 000236, loss = 0.002026
grad_step = 000237, loss = 0.002013
grad_step = 000238, loss = 0.002046
grad_step = 000239, loss = 0.002027
grad_step = 000240, loss = 0.002003
grad_step = 000241, loss = 0.002026
grad_step = 000242, loss = 0.002024
grad_step = 000243, loss = 0.001999
grad_step = 000244, loss = 0.002008
grad_step = 000245, loss = 0.002017
grad_step = 000246, loss = 0.001998
grad_step = 000247, loss = 0.001994
grad_step = 000248, loss = 0.002006
grad_step = 000249, loss = 0.001998
grad_step = 000250, loss = 0.001987
grad_step = 000251, loss = 0.001992
grad_step = 000252, loss = 0.001994
grad_step = 000253, loss = 0.001984
grad_step = 000254, loss = 0.001981
grad_step = 000255, loss = 0.001986
grad_step = 000256, loss = 0.001983
grad_step = 000257, loss = 0.001976
grad_step = 000258, loss = 0.001975
grad_step = 000259, loss = 0.001977
grad_step = 000260, loss = 0.001973
grad_step = 000261, loss = 0.001968
grad_step = 000262, loss = 0.001968
grad_step = 000263, loss = 0.001969
grad_step = 000264, loss = 0.001966
grad_step = 000265, loss = 0.001961
grad_step = 000266, loss = 0.001960
grad_step = 000267, loss = 0.001960
grad_step = 000268, loss = 0.001958
grad_step = 000269, loss = 0.001955
grad_step = 000270, loss = 0.001953
grad_step = 000271, loss = 0.001952
grad_step = 000272, loss = 0.001951
grad_step = 000273, loss = 0.001948
grad_step = 000274, loss = 0.001945
grad_step = 000275, loss = 0.001944
grad_step = 000276, loss = 0.001942
grad_step = 000277, loss = 0.001941
grad_step = 000278, loss = 0.001939
grad_step = 000279, loss = 0.001936
grad_step = 000280, loss = 0.001934
grad_step = 000281, loss = 0.001933
grad_step = 000282, loss = 0.001931
grad_step = 000283, loss = 0.001929
grad_step = 000284, loss = 0.001927
grad_step = 000285, loss = 0.001925
grad_step = 000286, loss = 0.001923
grad_step = 000287, loss = 0.001921
grad_step = 000288, loss = 0.001919
grad_step = 000289, loss = 0.001917
grad_step = 000290, loss = 0.001915
grad_step = 000291, loss = 0.001913
grad_step = 000292, loss = 0.001912
grad_step = 000293, loss = 0.001911
grad_step = 000294, loss = 0.001911
grad_step = 000295, loss = 0.001912
grad_step = 000296, loss = 0.001916
grad_step = 000297, loss = 0.001925
grad_step = 000298, loss = 0.001942
grad_step = 000299, loss = 0.001975
grad_step = 000300, loss = 0.002029
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002116
grad_step = 000302, loss = 0.002196
grad_step = 000303, loss = 0.002243
grad_step = 000304, loss = 0.002143
grad_step = 000305, loss = 0.001980
grad_step = 000306, loss = 0.001888
grad_step = 000307, loss = 0.001937
grad_step = 000308, loss = 0.002036
grad_step = 000309, loss = 0.002052
grad_step = 000310, loss = 0.001975
grad_step = 000311, loss = 0.001889
grad_step = 000312, loss = 0.001887
grad_step = 000313, loss = 0.001945
grad_step = 000314, loss = 0.001978
grad_step = 000315, loss = 0.001947
grad_step = 000316, loss = 0.001887
grad_step = 000317, loss = 0.001866
grad_step = 000318, loss = 0.001893
grad_step = 000319, loss = 0.001922
grad_step = 000320, loss = 0.001919
grad_step = 000321, loss = 0.001884
grad_step = 000322, loss = 0.001857
grad_step = 000323, loss = 0.001859
grad_step = 000324, loss = 0.001878
grad_step = 000325, loss = 0.001891
grad_step = 000326, loss = 0.001883
grad_step = 000327, loss = 0.001863
grad_step = 000328, loss = 0.001846
grad_step = 000329, loss = 0.001841
grad_step = 000330, loss = 0.001848
grad_step = 000331, loss = 0.001857
grad_step = 000332, loss = 0.001862
grad_step = 000333, loss = 0.001859
grad_step = 000334, loss = 0.001851
grad_step = 000335, loss = 0.001839
grad_step = 000336, loss = 0.001830
grad_step = 000337, loss = 0.001823
grad_step = 000338, loss = 0.001819
grad_step = 000339, loss = 0.001818
grad_step = 000340, loss = 0.001819
grad_step = 000341, loss = 0.001822
grad_step = 000342, loss = 0.001829
grad_step = 000343, loss = 0.001843
grad_step = 000344, loss = 0.001870
grad_step = 000345, loss = 0.001925
grad_step = 000346, loss = 0.002018
grad_step = 000347, loss = 0.002169
grad_step = 000348, loss = 0.002307
grad_step = 000349, loss = 0.002356
grad_step = 000350, loss = 0.002148
grad_step = 000351, loss = 0.001879
grad_step = 000352, loss = 0.001798
grad_step = 000353, loss = 0.001937
grad_step = 000354, loss = 0.002060
grad_step = 000355, loss = 0.001967
grad_step = 000356, loss = 0.001815
grad_step = 000357, loss = 0.001810
grad_step = 000358, loss = 0.001910
grad_step = 000359, loss = 0.001930
grad_step = 000360, loss = 0.001839
grad_step = 000361, loss = 0.001787
grad_step = 000362, loss = 0.001828
grad_step = 000363, loss = 0.001864
grad_step = 000364, loss = 0.001835
grad_step = 000365, loss = 0.001789
grad_step = 000366, loss = 0.001785
grad_step = 000367, loss = 0.001809
grad_step = 000368, loss = 0.001813
grad_step = 000369, loss = 0.001792
grad_step = 000370, loss = 0.001769
grad_step = 000371, loss = 0.001766
grad_step = 000372, loss = 0.001778
grad_step = 000373, loss = 0.001785
grad_step = 000374, loss = 0.001772
grad_step = 000375, loss = 0.001749
grad_step = 000376, loss = 0.001741
grad_step = 000377, loss = 0.001750
grad_step = 000378, loss = 0.001761
grad_step = 000379, loss = 0.001756
grad_step = 000380, loss = 0.001740
grad_step = 000381, loss = 0.001727
grad_step = 000382, loss = 0.001724
grad_step = 000383, loss = 0.001725
grad_step = 000384, loss = 0.001723
grad_step = 000385, loss = 0.001720
grad_step = 000386, loss = 0.001720
grad_step = 000387, loss = 0.001726
grad_step = 000388, loss = 0.001732
grad_step = 000389, loss = 0.001740
grad_step = 000390, loss = 0.001751
grad_step = 000391, loss = 0.001777
grad_step = 000392, loss = 0.001818
grad_step = 000393, loss = 0.001887
grad_step = 000394, loss = 0.001967
grad_step = 000395, loss = 0.002044
grad_step = 000396, loss = 0.002030
grad_step = 000397, loss = 0.001921
grad_step = 000398, loss = 0.001763
grad_step = 000399, loss = 0.001676
grad_step = 000400, loss = 0.001692
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001769
grad_step = 000402, loss = 0.001839
grad_step = 000403, loss = 0.001833
grad_step = 000404, loss = 0.001769
grad_step = 000405, loss = 0.001689
grad_step = 000406, loss = 0.001649
grad_step = 000407, loss = 0.001656
grad_step = 000408, loss = 0.001694
grad_step = 000409, loss = 0.001739
grad_step = 000410, loss = 0.001754
grad_step = 000411, loss = 0.001741
grad_step = 000412, loss = 0.001704
grad_step = 000413, loss = 0.001666
grad_step = 000414, loss = 0.001635
grad_step = 000415, loss = 0.001616
grad_step = 000416, loss = 0.001609
grad_step = 000417, loss = 0.001610
grad_step = 000418, loss = 0.001621
grad_step = 000419, loss = 0.001643
grad_step = 000420, loss = 0.001679
grad_step = 000421, loss = 0.001729
grad_step = 000422, loss = 0.001814
grad_step = 000423, loss = 0.001890
grad_step = 000424, loss = 0.001976
grad_step = 000425, loss = 0.001956
grad_step = 000426, loss = 0.001865
grad_step = 000427, loss = 0.001690
grad_step = 000428, loss = 0.001577
grad_step = 000429, loss = 0.001567
grad_step = 000430, loss = 0.001637
grad_step = 000431, loss = 0.001731
grad_step = 000432, loss = 0.001737
grad_step = 000433, loss = 0.001666
grad_step = 000434, loss = 0.001576
grad_step = 000435, loss = 0.001533
grad_step = 000436, loss = 0.001536
grad_step = 000437, loss = 0.001564
grad_step = 000438, loss = 0.001605
grad_step = 000439, loss = 0.001629
grad_step = 000440, loss = 0.001643
grad_step = 000441, loss = 0.001610
grad_step = 000442, loss = 0.001573
grad_step = 000443, loss = 0.001531
grad_step = 000444, loss = 0.001501
grad_step = 000445, loss = 0.001483
grad_step = 000446, loss = 0.001472
grad_step = 000447, loss = 0.001465
grad_step = 000448, loss = 0.001462
grad_step = 000449, loss = 0.001465
grad_step = 000450, loss = 0.001478
grad_step = 000451, loss = 0.001518
grad_step = 000452, loss = 0.001609
grad_step = 000453, loss = 0.001833
grad_step = 000454, loss = 0.002081
grad_step = 000455, loss = 0.002421
grad_step = 000456, loss = 0.002189
grad_step = 000457, loss = 0.001774
grad_step = 000458, loss = 0.001477
grad_step = 000459, loss = 0.001634
grad_step = 000460, loss = 0.001971
grad_step = 000461, loss = 0.001840
grad_step = 000462, loss = 0.001577
grad_step = 000463, loss = 0.001655
grad_step = 000464, loss = 0.001717
grad_step = 000465, loss = 0.001724
grad_step = 000466, loss = 0.001583
grad_step = 000467, loss = 0.001603
grad_step = 000468, loss = 0.001612
grad_step = 000469, loss = 0.001626
grad_step = 000470, loss = 0.001573
grad_step = 000471, loss = 0.001445
grad_step = 000472, loss = 0.001482
grad_step = 000473, loss = 0.001637
grad_step = 000474, loss = 0.001451
grad_step = 000475, loss = 0.001359
grad_step = 000476, loss = 0.001429
grad_step = 000477, loss = 0.001487
grad_step = 000478, loss = 0.001400
grad_step = 000479, loss = 0.001380
grad_step = 000480, loss = 0.001359
grad_step = 000481, loss = 0.001317
grad_step = 000482, loss = 0.001332
grad_step = 000483, loss = 0.001386
grad_step = 000484, loss = 0.001353
grad_step = 000485, loss = 0.001308
grad_step = 000486, loss = 0.001297
grad_step = 000487, loss = 0.001273
grad_step = 000488, loss = 0.001241
grad_step = 000489, loss = 0.001265
grad_step = 000490, loss = 0.001282
grad_step = 000491, loss = 0.001287
grad_step = 000492, loss = 0.001327
grad_step = 000493, loss = 0.001357
grad_step = 000494, loss = 0.001384
grad_step = 000495, loss = 0.001377
grad_step = 000496, loss = 0.001403
grad_step = 000497, loss = 0.001344
grad_step = 000498, loss = 0.001289
grad_step = 000499, loss = 0.001236
grad_step = 000500, loss = 0.001208
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001162
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

  date_run                              2020-05-15 08:13:38.347175
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.17932
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 08:13:38.353219
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0735649
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 08:13:38.359375
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.117749
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 08:13:38.364330
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.117845
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
0   2020-05-15 08:13:10.924120  ...    mean_absolute_error
1   2020-05-15 08:13:10.927880  ...     mean_squared_error
2   2020-05-15 08:13:10.930902  ...  median_absolute_error
3   2020-05-15 08:13:10.933943  ...               r2_score
4   2020-05-15 08:13:19.453078  ...    mean_absolute_error
5   2020-05-15 08:13:19.457311  ...     mean_squared_error
6   2020-05-15 08:13:19.460436  ...  median_absolute_error
7   2020-05-15 08:13:19.463547  ...               r2_score
8   2020-05-15 08:13:38.347175  ...    mean_absolute_error
9   2020-05-15 08:13:38.353219  ...     mean_squared_error
10  2020-05-15 08:13:38.359375  ...  median_absolute_error
11  2020-05-15 08:13:38.364330  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d13ed3fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:22, 446993.74it/s] 97%|█████████▋| 9658368/9912422 [00:00<00:00, 637280.22it/s]9920512it [00:00, 46634093.69it/s]                           
0it [00:00, ?it/s]32768it [00:00, 704031.69it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 482916.33it/s]1654784it [00:00, 11925667.88it/s]                         
0it [00:00, ?it/s]8192it [00:00, 253319.41it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc68d6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc5f060b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc68d6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc5e5b0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc36964e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc3681c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc68d6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc5e19710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc36964e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cc5f06128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff1d38971d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8d3f139b73804675f968a50bbdff98cf45f838667181d480116073613d4027a2
  Stored in directory: /tmp/pip-ephem-wheel-cache-9umsaieo/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff16b692710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1515520/17464789 [=>............................] - ETA: 0s
 6545408/17464789 [==========>...................] - ETA: 0s
12099584/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 08:15:04.465080: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 08:15:04.469643: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 08:15:04.469771: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558faf743ad0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 08:15:04.469784: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5716 - accuracy: 0.5062
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6225 - accuracy: 0.5029
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
12000/25000 [=============>................] - ETA: 3s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6383 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6639 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6835 - accuracy: 0.4989
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6900 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 08:15:17.691569
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 08:15:17.691569  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<46:47:53, 5.12kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<32:59:30, 7.26kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:08:45, 10.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:12:12, 14.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:18:35, 21.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.25M/862M [00:02<7:52:33, 30.1kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:02<5:29:23, 43.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:02<3:49:25, 61.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:02<2:39:57, 87.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:02<1:51:29, 125kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:02<1:17:44, 179kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.6M/862M [00:02<54:14, 255kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<37:52, 363kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:03<26:28, 516kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:03<18:37, 731kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.8M/862M [00:03<13:04, 1.04MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:03<09:40, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<08:39, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<09:43, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<07:44, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:06<05:37, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<09:39, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<08:00, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:16, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<04:31, 2.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<13:29, 985kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:09<12:10, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<09:11, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<06:32, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<4:21:36, 50.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<3:04:23, 71.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<2:09:10, 102kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<1:33:15, 141kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<1:06:36, 197kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:13<46:52, 280kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:15<35:46, 366kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<26:24, 496kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<18:43, 697kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:17<16:08, 807kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<12:37, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<09:09, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<09:27, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<07:56, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:19<05:52, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:21<07:09, 1.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<06:18, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:21<04:44, 2.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<06:20, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<05:46, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:23<04:22, 2.92MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<06:03, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<05:32, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:25<04:11, 3.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<05:56, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:27, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<04:08, 3.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:52, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:12, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<03:57, 3.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<02:55, 4.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<52:29, 239kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<38:01, 330kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<26:50, 466kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<21:41, 575kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<16:27, 757kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<11:49, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<11:11, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<09:06, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:40, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<07:34, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:34, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:54, 2.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:19, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:17, 2.85MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<05:52, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:23, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:01, 3.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:12, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<03:57, 3.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:36, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:12, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<03:57, 3.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:06, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<03:53, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:05, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<03:48, 3.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<05:27, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:02, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<03:49, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:27, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<03:47, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<02:47, 4.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<1:41:16, 116kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<1:11:51, 163kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<50:30, 231kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<35:23, 329kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<1:08:41, 170kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<49:22, 236kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<34:43, 334kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<26:58, 429kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<20:04, 576kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:19, 807kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<10:08, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<2:00:53, 95.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<1:25:45, 134kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:00:11, 191kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<44:42, 256kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<32:16, 354kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<22:57, 498kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<16:08, 705kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<57:01, 199kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<41:05, 277kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<28:59, 391kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<22:53, 494kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<17:11, 657kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<12:18, 916kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<11:14, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<09:00, 1.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:34, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<07:14, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<06:12, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:37, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<05:50, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<06:30, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:06, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:40, 3.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<19:28, 566kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<15:48, 698kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:31, 955kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:16, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<08:57, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:25, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:24, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:55, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<46:02, 237kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<33:18, 327kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<23:30, 462kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<18:57, 571kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<14:23, 752kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<10:19, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<09:44, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:56, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:49, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<06:35, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:42, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:15, 2.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<05:29, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<04:46, 2.22MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<03:36, 2.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<02:41, 3.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<09:44, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:53, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:47, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<06:30, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:38, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:12, 2.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<03:39, 2.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:31, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:23, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:10, 3.24MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<12:53, 796kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<10:06, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<07:18, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<07:28, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:17, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:39, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<05:38, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:59, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:40, 2.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<02:43, 3.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<2:01:26, 82.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<1:27:06, 116kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<1:01:22, 164kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<43:08, 232kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<32:24, 308kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<23:43, 421kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<16:51, 592kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:56, 832kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<13:18, 746kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<10:45, 923kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:51, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:40, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:41, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<06:18, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<04:54, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:38, 2.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:19, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<04:54, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<03:42, 2.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<02:43, 3.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<09:23, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<07:28, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<05:38, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:03, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:53, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<07:15, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:19, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:52, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<05:11, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:53, 2.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:47, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:18, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:13, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:03, 3.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:40, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:05, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:00<05:10, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:47, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:08, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<03:49, 2.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<02:50, 3.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<08:25, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<08:39, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<06:36, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:55, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:12, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<04:42, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<03:33, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:08<04:05, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<03:05, 2.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:15, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<03:54, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<02:57, 3.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:51, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<02:55, 3.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:09, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:50, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<02:51, 3.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:07, 4.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<1:07:32, 132kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<48:01, 185kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<33:51, 262kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<23:41, 373kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<31:00, 285kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<22:36, 390kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<15:58, 550kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<13:13, 662kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<10:09, 862kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:18, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:12, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<23:28, 370kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<18:13, 476kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<13:08, 660kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<09:21, 925kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<08:58, 961kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<07:11, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:11, 1.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:45, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<32:24, 264kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<23:23, 365kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<16:39, 512kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<11:42, 725kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<20:28, 414kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<16:05, 527kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<11:41, 724kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<08:15, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<10:10, 827kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<08:00, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:46, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<05:58, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<05:01, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:02, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:01, 2.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<04:03, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:41, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:45, 2.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:03, 3.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<12:36, 645kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<09:51, 825kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<07:08, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<05:04, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<1:02:55, 128kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<44:59, 179kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<31:36, 254kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<22:09, 361kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<23:15, 344kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<17:11, 465kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<12:30, 638kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<08:49, 900kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<09:00, 879kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<08:00, 990kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<06:00, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:17, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:45, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<06:15, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:31, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:16, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<48:29, 161kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<34:43, 224kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<24:22, 318kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<17:06, 451kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<1:17:27, 99.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<55:44, 138kB/s]   .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<39:21, 196kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<27:26, 279kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<1:08:36, 112kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<48:47, 157kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<34:13, 223kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<23:56, 317kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<54:17, 140kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<38:46, 195kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<27:11, 278kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<19:04, 394kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<23:59, 313kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<17:33, 428kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:26, 601kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<10:25, 714kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<08:04, 922kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:49, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<04:07, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<18:15, 403kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<15:22, 479kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<11:19, 649kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<08:03, 908kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<07:11, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:52, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<04:16, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<04:32, 1.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<04:45, 1.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<03:41, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<02:46, 2.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:31, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<02:25, 2.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:20, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:04, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<02:19, 3.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:16, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<02:54, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:13, 3.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<01:38, 4.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<27:17, 254kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<20:32, 338kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<14:39, 473kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:21, 667kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<09:21, 735kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<07:14, 948kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:13, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<05:14, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<04:21, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:12, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:49, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:15, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:35, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<01:52, 3.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<14:44, 452kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<10:59, 606kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<07:50, 847kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<07:01, 941kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:35, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:03, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<04:22, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:43, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:46, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:25<03:27, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<03:04, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:18, 2.78MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 479M/862M [03:27<03:07, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<02:43, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:07, 2.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<01:34, 4.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<04:33, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:50, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:48, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<03:26, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<03:03, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:16, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<03:03, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<02:46, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:05, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<02:54, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<02:39, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:00, 3.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<03:14, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:32, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:51, 3.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<03:44, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<03:14, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:25, 2.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<03:04, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<02:45, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:02, 2.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:30, 3.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<50:37, 115kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<35:59, 162kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<25:13, 230kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<18:55, 305kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<14:25, 400kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<10:19, 558kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<07:19, 783kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<06:41, 854kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<05:16, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:47<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:58, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:56, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:10, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<03:42, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:51<03:10, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<02:21, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:55, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:37, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<01:58, 2.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:38, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:20, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<01:46, 3.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<01:17, 4.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<22:26, 239kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<16:15, 330kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<11:27, 466kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<09:12, 575kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<06:59, 758kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:58, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<04:42, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:49, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:47, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:09, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:44, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:00, 2.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:28, 3.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<45:38, 112kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<34:02, 149kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<24:19, 209kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<17:03, 297kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<11:55, 421kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<4:44:04, 17.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<3:19:07, 25.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<2:18:47, 36.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<1:37:36, 50.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<1:08:44, 72.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<47:59, 103kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<34:23, 142kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<24:59, 195kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<17:40, 275kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<13:02, 369kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<09:40, 497kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<06:51, 697kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<05:46, 820kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<05:07, 925kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<03:48, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<03:05, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:17<02:38, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<01:57, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:26, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<02:11, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<01:38, 2.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:12, 2.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<01:56, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:04, 4.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<17:35, 254kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<12:45, 350kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<08:59, 493kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<07:16, 604kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<05:32, 792kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<03:58, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<03:05, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:14, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:30, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<02:33, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<01:57, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:25, 2.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<02:59, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:51, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:20, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<04:29, 917kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<04:01, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<03:01, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:08, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<04:48, 840kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:47, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<02:44, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:49, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:23, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<01:45, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:08, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:53, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<01:25, 2.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:55, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:08, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<01:39, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<01:13, 3.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<01:58, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<01:46, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:19, 2.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<01:47, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<01:37, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:13, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<01:33, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:10, 3.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<01:39, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<01:53, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:30, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:04, 3.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<3:24:34, 17.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<2:23:17, 24.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<1:39:38, 34.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<1:09:49, 49.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<49:08, 69.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<34:13, 99.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<24:29, 137kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<17:27, 192kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<12:12, 272kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<09:14, 356kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<06:47, 484kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<04:47, 681kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<04:05, 789kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<03:11, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:17, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<02:19, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<02:17, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:43, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:15, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:48, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:31, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:12, 2.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:52, 3.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<18:34, 163kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<13:36, 222kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<09:38, 311kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<07:06, 415kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<05:15, 559kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<03:43, 782kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<03:15, 886kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:08<02:53, 993kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<02:09, 1.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:32, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:56, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<01:38, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:12, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<01:28, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<00:58, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:18, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:11, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<00:52, 3.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<01:13, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<00:50, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:05, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<00:48, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:03, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:06, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:01, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<00:45, 3.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<01:04, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<00:57, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<00:44, 3.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:31, 4.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<08:54, 254kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<06:26, 350kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<04:30, 494kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<03:37, 605kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:43, 804kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:57, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:22, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<08:58, 237kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<06:29, 327kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<04:32, 461kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<03:36, 570kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<02:43, 752kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:56, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:48, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:40, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:14, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<00:53, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:18, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:48, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:59, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:52, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:39, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:52, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:47, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:35, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<00:48, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:44, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:33, 3.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:42, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:31, 3.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:43, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:40, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:29, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:47<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:38, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:28, 3.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:40, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:45, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:36, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:25, 3.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<11:42, 117kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<08:17, 165kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<05:43, 234kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<04:12, 310kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<03:03, 423kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<02:08, 595kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:44, 709kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:20, 916kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:56, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:55, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:45, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:32, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:37, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:33, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:24, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:19, 3.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:14, 4.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<03:46, 254kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<02:43, 350kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:52, 494kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<01:28, 605kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<01:06, 793kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:46, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:43, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:34, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:24, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:16, 2.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<03:13, 234kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<02:24, 312kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<01:41, 437kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:07, 619kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<01:09, 595kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:52, 781kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:36, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:32, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:26, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:18, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:20, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:20, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:15, 2.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:14, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:13, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:09, 2.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:11, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:10, 2.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:07, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:06, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:06, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:04, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:03, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 3.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:02, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 3.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.12MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 856/400000 [00:00<00:46, 8557.32it/s]  0%|          | 1727/400000 [00:00<00:46, 8600.13it/s]  1%|          | 2601/400000 [00:00<00:45, 8641.54it/s]  1%|          | 3475/400000 [00:00<00:45, 8668.49it/s]  1%|          | 4370/400000 [00:00<00:45, 8749.54it/s]  1%|▏         | 5259/400000 [00:00<00:44, 8790.18it/s]  2%|▏         | 6148/400000 [00:00<00:44, 8818.27it/s]  2%|▏         | 7023/400000 [00:00<00:44, 8796.32it/s]  2%|▏         | 7880/400000 [00:00<00:44, 8726.17it/s]  2%|▏         | 8754/400000 [00:01<00:44, 8728.13it/s]  2%|▏         | 9628/400000 [00:01<00:44, 8728.73it/s]  3%|▎         | 10510/400000 [00:01<00:44, 8753.51it/s]  3%|▎         | 11374/400000 [00:01<00:45, 8598.47it/s]  3%|▎         | 12250/400000 [00:01<00:44, 8643.41it/s]  3%|▎         | 13110/400000 [00:01<00:44, 8602.77it/s]  3%|▎         | 13982/400000 [00:01<00:44, 8636.99it/s]  4%|▎         | 14861/400000 [00:01<00:44, 8680.15it/s]  4%|▍         | 15748/400000 [00:01<00:43, 8733.93it/s]  4%|▍         | 16621/400000 [00:01<00:44, 8710.79it/s]  4%|▍         | 17492/400000 [00:02<00:44, 8606.77it/s]  5%|▍         | 18363/400000 [00:02<00:44, 8636.18it/s]  5%|▍         | 19234/400000 [00:02<00:43, 8657.15it/s]  5%|▌         | 20116/400000 [00:02<00:43, 8703.39it/s]  5%|▌         | 20988/400000 [00:02<00:43, 8708.00it/s]  5%|▌         | 21861/400000 [00:02<00:43, 8712.14it/s]  6%|▌         | 22761/400000 [00:02<00:42, 8796.53it/s]  6%|▌         | 23647/400000 [00:02<00:42, 8814.06it/s]  6%|▌         | 24532/400000 [00:02<00:42, 8823.77it/s]  6%|▋         | 25415/400000 [00:02<00:42, 8774.09it/s]  7%|▋         | 26293/400000 [00:03<00:42, 8767.74it/s]  7%|▋         | 27170/400000 [00:03<00:42, 8729.82it/s]  7%|▋         | 28053/400000 [00:03<00:42, 8757.48it/s]  7%|▋         | 28936/400000 [00:03<00:42, 8778.73it/s]  7%|▋         | 29819/400000 [00:03<00:42, 8793.51it/s]  8%|▊         | 30707/400000 [00:03<00:41, 8818.91it/s]  8%|▊         | 31589/400000 [00:03<00:41, 8801.69it/s]  8%|▊         | 32474/400000 [00:03<00:41, 8816.09it/s]  8%|▊         | 33362/400000 [00:03<00:41, 8832.07it/s]  9%|▊         | 34246/400000 [00:03<00:41, 8805.98it/s]  9%|▉         | 35143/400000 [00:04<00:41, 8853.56it/s]  9%|▉         | 36029/400000 [00:04<00:41, 8840.71it/s]  9%|▉         | 36914/400000 [00:04<00:41, 8696.24it/s]  9%|▉         | 37785/400000 [00:04<00:41, 8628.46it/s] 10%|▉         | 38649/400000 [00:04<00:41, 8631.56it/s] 10%|▉         | 39521/400000 [00:04<00:41, 8656.44it/s] 10%|█         | 40387/400000 [00:04<00:41, 8595.24it/s] 10%|█         | 41255/400000 [00:04<00:41, 8617.86it/s] 11%|█         | 42135/400000 [00:04<00:41, 8669.38it/s] 11%|█         | 43003/400000 [00:04<00:42, 8498.33it/s] 11%|█         | 43854/400000 [00:05<00:42, 8472.58it/s] 11%|█         | 44727/400000 [00:05<00:41, 8546.01it/s] 11%|█▏        | 45601/400000 [00:05<00:41, 8601.52it/s] 12%|█▏        | 46481/400000 [00:05<00:40, 8658.03it/s] 12%|█▏        | 47358/400000 [00:05<00:40, 8690.12it/s] 12%|█▏        | 48239/400000 [00:05<00:40, 8725.01it/s] 12%|█▏        | 49112/400000 [00:05<00:40, 8713.55it/s] 12%|█▏        | 49990/400000 [00:05<00:40, 8733.35it/s] 13%|█▎        | 50868/400000 [00:05<00:39, 8746.20it/s] 13%|█▎        | 51743/400000 [00:05<00:40, 8705.79it/s] 13%|█▎        | 52614/400000 [00:06<00:40, 8631.65it/s] 13%|█▎        | 53478/400000 [00:06<00:40, 8607.30it/s] 14%|█▎        | 54339/400000 [00:06<00:40, 8597.47it/s] 14%|█▍        | 55199/400000 [00:06<00:40, 8552.52it/s] 14%|█▍        | 56064/400000 [00:06<00:40, 8581.18it/s] 14%|█▍        | 56934/400000 [00:06<00:39, 8615.60it/s] 14%|█▍        | 57804/400000 [00:06<00:39, 8639.98it/s] 15%|█▍        | 58669/400000 [00:06<00:39, 8575.93it/s] 15%|█▍        | 59551/400000 [00:06<00:39, 8646.82it/s] 15%|█▌        | 60416/400000 [00:06<00:39, 8595.00it/s] 15%|█▌        | 61276/400000 [00:07<00:40, 8397.90it/s] 16%|█▌        | 62148/400000 [00:07<00:39, 8490.82it/s] 16%|█▌        | 63020/400000 [00:07<00:39, 8557.77it/s] 16%|█▌        | 63877/400000 [00:07<00:39, 8421.53it/s] 16%|█▌        | 64731/400000 [00:07<00:39, 8456.15it/s] 16%|█▋        | 65605/400000 [00:07<00:39, 8536.76it/s] 17%|█▋        | 66483/400000 [00:07<00:38, 8606.28it/s] 17%|█▋        | 67345/400000 [00:07<00:38, 8572.42it/s] 17%|█▋        | 68203/400000 [00:07<00:38, 8534.41it/s] 17%|█▋        | 69057/400000 [00:07<00:39, 8435.25it/s] 17%|█▋        | 69905/400000 [00:08<00:39, 8448.06it/s] 18%|█▊        | 70766/400000 [00:08<00:38, 8495.45it/s] 18%|█▊        | 71616/400000 [00:08<00:38, 8487.63it/s] 18%|█▊        | 72482/400000 [00:08<00:38, 8536.22it/s] 18%|█▊        | 73336/400000 [00:08<00:38, 8522.10it/s] 19%|█▊        | 74189/400000 [00:08<00:38, 8516.28it/s] 19%|█▉        | 75047/400000 [00:08<00:38, 8532.70it/s] 19%|█▉        | 75906/400000 [00:08<00:37, 8549.51it/s] 19%|█▉        | 76762/400000 [00:08<00:37, 8533.52it/s] 19%|█▉        | 77616/400000 [00:08<00:39, 8219.77it/s] 20%|█▉        | 78486/400000 [00:09<00:38, 8356.79it/s] 20%|█▉        | 79373/400000 [00:09<00:37, 8502.68it/s] 20%|██        | 80252/400000 [00:09<00:37, 8585.13it/s] 20%|██        | 81113/400000 [00:09<00:37, 8577.37it/s] 20%|██        | 81979/400000 [00:09<00:36, 8600.20it/s] 21%|██        | 82864/400000 [00:09<00:36, 8672.40it/s] 21%|██        | 83743/400000 [00:09<00:36, 8706.96it/s] 21%|██        | 84620/400000 [00:09<00:36, 8724.70it/s] 21%|██▏       | 85501/400000 [00:09<00:35, 8749.91it/s] 22%|██▏       | 86377/400000 [00:09<00:35, 8722.82it/s] 22%|██▏       | 87252/400000 [00:10<00:35, 8730.30it/s] 22%|██▏       | 88134/400000 [00:10<00:35, 8754.67it/s] 22%|██▏       | 89010/400000 [00:10<00:36, 8485.66it/s] 22%|██▏       | 89892/400000 [00:10<00:36, 8582.99it/s] 23%|██▎       | 90752/400000 [00:10<00:36, 8576.98it/s] 23%|██▎       | 91630/400000 [00:10<00:35, 8634.25it/s] 23%|██▎       | 92509/400000 [00:10<00:35, 8680.09it/s] 23%|██▎       | 93392/400000 [00:10<00:35, 8721.59it/s] 24%|██▎       | 94265/400000 [00:10<00:35, 8711.67it/s] 24%|██▍       | 95137/400000 [00:11<00:35, 8637.10it/s] 24%|██▍       | 96010/400000 [00:11<00:35, 8661.97it/s] 24%|██▍       | 96877/400000 [00:11<00:35, 8646.24it/s] 24%|██▍       | 97753/400000 [00:11<00:34, 8679.39it/s] 25%|██▍       | 98631/400000 [00:11<00:34, 8707.45it/s] 25%|██▍       | 99502/400000 [00:11<00:35, 8554.24it/s] 25%|██▌       | 100388/400000 [00:11<00:34, 8642.67it/s] 25%|██▌       | 101280/400000 [00:11<00:34, 8722.47it/s] 26%|██▌       | 102153/400000 [00:11<00:34, 8625.25it/s] 26%|██▌       | 103017/400000 [00:11<00:34, 8625.65it/s] 26%|██▌       | 103881/400000 [00:12<00:34, 8552.52it/s] 26%|██▌       | 104737/400000 [00:12<00:34, 8540.35it/s] 26%|██▋       | 105619/400000 [00:12<00:34, 8619.77it/s] 27%|██▋       | 106486/400000 [00:12<00:33, 8633.49it/s] 27%|██▋       | 107350/400000 [00:12<00:33, 8624.38it/s] 27%|██▋       | 108213/400000 [00:12<00:34, 8554.63it/s] 27%|██▋       | 109090/400000 [00:12<00:33, 8616.32it/s] 27%|██▋       | 109980/400000 [00:12<00:33, 8699.04it/s] 28%|██▊       | 110860/400000 [00:12<00:33, 8728.04it/s] 28%|██▊       | 111745/400000 [00:12<00:32, 8762.91it/s] 28%|██▊       | 112622/400000 [00:13<00:33, 8662.53it/s] 28%|██▊       | 113489/400000 [00:13<00:33, 8660.72it/s] 29%|██▊       | 114364/400000 [00:13<00:32, 8685.37it/s] 29%|██▉       | 115243/400000 [00:13<00:32, 8714.57it/s] 29%|██▉       | 116115/400000 [00:13<00:33, 8591.28it/s] 29%|██▉       | 116976/400000 [00:13<00:32, 8594.05it/s] 29%|██▉       | 117853/400000 [00:13<00:32, 8643.61it/s] 30%|██▉       | 118718/400000 [00:13<00:32, 8619.80it/s] 30%|██▉       | 119596/400000 [00:13<00:32, 8667.21it/s] 30%|███       | 120463/400000 [00:13<00:32, 8542.26it/s] 30%|███       | 121329/400000 [00:14<00:32, 8577.02it/s] 31%|███       | 122188/400000 [00:14<00:32, 8561.06it/s] 31%|███       | 123068/400000 [00:14<00:32, 8630.78it/s] 31%|███       | 123958/400000 [00:14<00:31, 8707.50it/s] 31%|███       | 124857/400000 [00:14<00:31, 8789.10it/s] 31%|███▏      | 125739/400000 [00:14<00:31, 8797.26it/s] 32%|███▏      | 126620/400000 [00:14<00:31, 8746.90it/s] 32%|███▏      | 127496/400000 [00:14<00:31, 8748.88it/s] 32%|███▏      | 128372/400000 [00:14<00:31, 8589.48it/s] 32%|███▏      | 129232/400000 [00:14<00:31, 8522.72it/s] 33%|███▎      | 130105/400000 [00:15<00:31, 8581.25it/s] 33%|███▎      | 130964/400000 [00:15<00:31, 8442.87it/s] 33%|███▎      | 131848/400000 [00:15<00:31, 8556.04it/s] 33%|███▎      | 132727/400000 [00:15<00:30, 8624.32it/s] 33%|███▎      | 133611/400000 [00:15<00:30, 8687.26it/s] 34%|███▎      | 134481/400000 [00:15<00:30, 8671.40it/s] 34%|███▍      | 135349/400000 [00:15<00:30, 8664.77it/s] 34%|███▍      | 136225/400000 [00:15<00:30, 8691.92it/s] 34%|███▍      | 137095/400000 [00:15<00:30, 8640.57it/s] 34%|███▍      | 137977/400000 [00:15<00:30, 8693.25it/s] 35%|███▍      | 138847/400000 [00:16<00:30, 8672.43it/s] 35%|███▍      | 139715/400000 [00:16<00:30, 8659.68it/s] 35%|███▌      | 140600/400000 [00:16<00:29, 8715.09it/s] 35%|███▌      | 141472/400000 [00:16<00:29, 8699.39it/s] 36%|███▌      | 142353/400000 [00:16<00:29, 8730.73it/s] 36%|███▌      | 143239/400000 [00:16<00:29, 8768.94it/s] 36%|███▌      | 144117/400000 [00:16<00:29, 8765.23it/s] 36%|███▌      | 144994/400000 [00:16<00:29, 8634.86it/s] 36%|███▋      | 145863/400000 [00:16<00:29, 8650.13it/s] 37%|███▋      | 146729/400000 [00:16<00:29, 8575.64it/s] 37%|███▋      | 147587/400000 [00:17<00:30, 8340.48it/s] 37%|███▋      | 148446/400000 [00:17<00:29, 8411.43it/s] 37%|███▋      | 149328/400000 [00:17<00:29, 8528.81it/s] 38%|███▊      | 150201/400000 [00:17<00:29, 8586.79it/s] 38%|███▊      | 151083/400000 [00:17<00:28, 8653.46it/s] 38%|███▊      | 151957/400000 [00:17<00:28, 8677.97it/s] 38%|███▊      | 152827/400000 [00:17<00:28, 8681.83it/s] 38%|███▊      | 153720/400000 [00:17<00:28, 8754.22it/s] 39%|███▊      | 154596/400000 [00:17<00:28, 8729.99it/s] 39%|███▉      | 155470/400000 [00:17<00:28, 8627.28it/s] 39%|███▉      | 156337/400000 [00:18<00:28, 8637.11it/s] 39%|███▉      | 157202/400000 [00:18<00:28, 8492.77it/s] 40%|███▉      | 158053/400000 [00:18<00:28, 8480.39it/s] 40%|███▉      | 158921/400000 [00:18<00:28, 8538.57it/s] 40%|███▉      | 159776/400000 [00:18<00:28, 8540.04it/s] 40%|████      | 160656/400000 [00:18<00:27, 8613.89it/s] 40%|████      | 161524/400000 [00:18<00:27, 8631.85it/s] 41%|████      | 162388/400000 [00:18<00:28, 8432.77it/s] 41%|████      | 163273/400000 [00:18<00:27, 8532.94it/s] 41%|████      | 164129/400000 [00:18<00:27, 8539.49it/s] 41%|████▏     | 165000/400000 [00:19<00:27, 8589.75it/s] 41%|████▏     | 165860/400000 [00:19<00:27, 8585.48it/s] 42%|████▏     | 166719/400000 [00:19<00:27, 8582.02it/s] 42%|████▏     | 167585/400000 [00:19<00:27, 8605.16it/s] 42%|████▏     | 168463/400000 [00:19<00:26, 8655.14it/s] 42%|████▏     | 169340/400000 [00:19<00:26, 8688.31it/s] 43%|████▎     | 170210/400000 [00:19<00:26, 8626.94it/s] 43%|████▎     | 171073/400000 [00:19<00:26, 8609.59it/s] 43%|████▎     | 171935/400000 [00:19<00:26, 8554.12it/s] 43%|████▎     | 172805/400000 [00:20<00:26, 8596.24it/s] 43%|████▎     | 173665/400000 [00:20<00:26, 8587.37it/s] 44%|████▎     | 174525/400000 [00:20<00:26, 8590.72it/s] 44%|████▍     | 175406/400000 [00:20<00:25, 8652.76it/s] 44%|████▍     | 176284/400000 [00:20<00:25, 8688.07it/s] 44%|████▍     | 177158/400000 [00:20<00:25, 8703.40it/s] 45%|████▍     | 178029/400000 [00:20<00:25, 8697.66it/s] 45%|████▍     | 178899/400000 [00:20<00:25, 8669.74it/s] 45%|████▍     | 179773/400000 [00:20<00:25, 8689.23it/s] 45%|████▌     | 180644/400000 [00:20<00:25, 8692.95it/s] 45%|████▌     | 181514/400000 [00:21<00:25, 8583.27it/s] 46%|████▌     | 182373/400000 [00:21<00:25, 8583.58it/s] 46%|████▌     | 183232/400000 [00:21<00:26, 8174.10it/s] 46%|████▌     | 184087/400000 [00:21<00:26, 8283.32it/s] 46%|████▌     | 184960/400000 [00:21<00:25, 8410.37it/s] 46%|████▋     | 185831/400000 [00:21<00:25, 8496.02it/s] 47%|████▋     | 186700/400000 [00:21<00:24, 8552.52it/s] 47%|████▋     | 187569/400000 [00:21<00:24, 8592.28it/s] 47%|████▋     | 188443/400000 [00:21<00:24, 8633.87it/s] 47%|████▋     | 189318/400000 [00:21<00:24, 8667.87it/s] 48%|████▊     | 190189/400000 [00:22<00:24, 8679.20it/s] 48%|████▊     | 191071/400000 [00:22<00:23, 8720.39it/s] 48%|████▊     | 191952/400000 [00:22<00:23, 8746.78it/s] 48%|████▊     | 192827/400000 [00:22<00:23, 8703.35it/s] 48%|████▊     | 193717/400000 [00:22<00:23, 8758.48it/s] 49%|████▊     | 194594/400000 [00:22<00:23, 8737.10it/s] 49%|████▉     | 195468/400000 [00:22<00:23, 8675.28it/s] 49%|████▉     | 196336/400000 [00:22<00:23, 8636.97it/s] 49%|████▉     | 197214/400000 [00:22<00:23, 8678.74it/s] 50%|████▉     | 198083/400000 [00:22<00:23, 8661.47it/s] 50%|████▉     | 198955/400000 [00:23<00:23, 8678.30it/s] 50%|████▉     | 199827/400000 [00:23<00:23, 8690.43it/s] 50%|█████     | 200703/400000 [00:23<00:22, 8708.69it/s] 50%|█████     | 201585/400000 [00:23<00:22, 8739.08it/s] 51%|█████     | 202494/400000 [00:23<00:22, 8840.59it/s] 51%|█████     | 203392/400000 [00:23<00:22, 8879.05it/s] 51%|█████     | 204281/400000 [00:23<00:22, 8817.49it/s] 51%|█████▏    | 205164/400000 [00:23<00:22, 8622.18it/s] 52%|█████▏    | 206053/400000 [00:23<00:22, 8700.12it/s] 52%|█████▏    | 206937/400000 [00:23<00:22, 8740.54it/s] 52%|█████▏    | 207830/400000 [00:24<00:21, 8794.94it/s] 52%|█████▏    | 208714/400000 [00:24<00:21, 8807.19it/s] 52%|█████▏    | 209596/400000 [00:24<00:21, 8777.57it/s] 53%|█████▎    | 210483/400000 [00:24<00:21, 8802.89it/s] 53%|█████▎    | 211369/400000 [00:24<00:21, 8819.21it/s] 53%|█████▎    | 212264/400000 [00:24<00:21, 8855.81it/s] 53%|█████▎    | 213152/400000 [00:24<00:21, 8861.87it/s] 54%|█████▎    | 214039/400000 [00:24<00:21, 8817.42it/s] 54%|█████▎    | 214921/400000 [00:24<00:21, 8771.59it/s] 54%|█████▍    | 215799/400000 [00:24<00:21, 8754.21it/s] 54%|█████▍    | 216675/400000 [00:25<00:21, 8729.58it/s] 54%|█████▍    | 217549/400000 [00:25<00:21, 8681.99it/s] 55%|█████▍    | 218418/400000 [00:25<00:20, 8650.32it/s] 55%|█████▍    | 219310/400000 [00:25<00:20, 8728.11it/s] 55%|█████▌    | 220196/400000 [00:25<00:20, 8765.89it/s] 55%|█████▌    | 221073/400000 [00:25<00:20, 8710.05it/s] 55%|█████▌    | 221945/400000 [00:25<00:20, 8681.73it/s] 56%|█████▌    | 222814/400000 [00:25<00:20, 8638.75it/s] 56%|█████▌    | 223696/400000 [00:25<00:20, 8691.64it/s] 56%|█████▌    | 224570/400000 [00:25<00:20, 8703.92it/s] 56%|█████▋    | 225448/400000 [00:26<00:20, 8724.10it/s] 57%|█████▋    | 226338/400000 [00:26<00:19, 8774.05it/s] 57%|█████▋    | 227216/400000 [00:26<00:19, 8743.16it/s] 57%|█████▋    | 228095/400000 [00:26<00:19, 8756.26it/s] 57%|█████▋    | 228977/400000 [00:26<00:19, 8772.65it/s] 57%|█████▋    | 229861/400000 [00:26<00:19, 8790.07it/s] 58%|█████▊    | 230741/400000 [00:26<00:19, 8767.16it/s] 58%|█████▊    | 231618/400000 [00:26<00:19, 8631.56it/s] 58%|█████▊    | 232486/400000 [00:26<00:19, 8644.35it/s] 58%|█████▊    | 233367/400000 [00:26<00:19, 8691.70it/s] 59%|█████▊    | 234246/400000 [00:27<00:19, 8719.50it/s] 59%|█████▉    | 235140/400000 [00:27<00:18, 8783.44it/s] 59%|█████▉    | 236019/400000 [00:27<00:18, 8745.06it/s] 59%|█████▉    | 236894/400000 [00:27<00:19, 8440.02it/s] 59%|█████▉    | 237791/400000 [00:27<00:18, 8591.74it/s] 60%|█████▉    | 238653/400000 [00:27<00:18, 8592.18it/s] 60%|█████▉    | 239538/400000 [00:27<00:18, 8665.88it/s] 60%|██████    | 240406/400000 [00:27<00:18, 8669.96it/s] 60%|██████    | 241290/400000 [00:27<00:18, 8718.57it/s] 61%|██████    | 242163/400000 [00:27<00:18, 8719.93it/s] 61%|██████    | 243036/400000 [00:28<00:18, 8634.75it/s] 61%|██████    | 243909/400000 [00:28<00:18, 8660.92it/s] 61%|██████    | 244776/400000 [00:28<00:18, 8611.77it/s] 61%|██████▏   | 245657/400000 [00:28<00:17, 8668.56it/s] 62%|██████▏   | 246526/400000 [00:28<00:17, 8674.38it/s] 62%|██████▏   | 247409/400000 [00:28<00:17, 8720.33it/s] 62%|██████▏   | 248282/400000 [00:28<00:17, 8699.59it/s] 62%|██████▏   | 249153/400000 [00:28<00:17, 8663.84it/s] 63%|██████▎   | 250043/400000 [00:28<00:17, 8730.47it/s] 63%|██████▎   | 250917/400000 [00:28<00:17, 8607.32it/s] 63%|██████▎   | 251779/400000 [00:29<00:17, 8605.44it/s] 63%|██████▎   | 252655/400000 [00:29<00:17, 8650.59it/s] 63%|██████▎   | 253521/400000 [00:29<00:16, 8639.46it/s] 64%|██████▎   | 254386/400000 [00:29<00:16, 8621.18it/s] 64%|██████▍   | 255259/400000 [00:29<00:16, 8653.04it/s] 64%|██████▍   | 256138/400000 [00:29<00:16, 8691.56it/s] 64%|██████▍   | 257008/400000 [00:29<00:16, 8638.54it/s] 64%|██████▍   | 257873/400000 [00:29<00:16, 8479.30it/s] 65%|██████▍   | 258748/400000 [00:29<00:16, 8557.17it/s] 65%|██████▍   | 259610/400000 [00:30<00:16, 8574.77it/s] 65%|██████▌   | 260468/400000 [00:30<00:16, 8382.84it/s] 65%|██████▌   | 261344/400000 [00:30<00:16, 8490.95it/s] 66%|██████▌   | 262195/400000 [00:30<00:16, 8430.34it/s] 66%|██████▌   | 263040/400000 [00:30<00:16, 8370.33it/s] 66%|██████▌   | 263919/400000 [00:30<00:16, 8491.58it/s] 66%|██████▌   | 264798/400000 [00:30<00:15, 8578.43it/s] 66%|██████▋   | 265681/400000 [00:30<00:15, 8651.07it/s] 67%|██████▋   | 266561/400000 [00:30<00:15, 8692.76it/s] 67%|██████▋   | 267438/400000 [00:30<00:15, 8713.13it/s] 67%|██████▋   | 268322/400000 [00:31<00:15, 8749.49it/s] 67%|██████▋   | 269217/400000 [00:31<00:14, 8808.39it/s] 68%|██████▊   | 270124/400000 [00:31<00:14, 8883.99it/s] 68%|██████▊   | 271013/400000 [00:31<00:14, 8883.32it/s] 68%|██████▊   | 271908/400000 [00:31<00:14, 8900.32it/s] 68%|██████▊   | 272799/400000 [00:31<00:14, 8894.63it/s] 68%|██████▊   | 273706/400000 [00:31<00:14, 8946.37it/s] 69%|██████▊   | 274601/400000 [00:31<00:14, 8883.49it/s] 69%|██████▉   | 275490/400000 [00:31<00:14, 8858.64it/s] 69%|██████▉   | 276377/400000 [00:31<00:14, 8824.37it/s] 69%|██████▉   | 277260/400000 [00:32<00:14, 8693.16it/s] 70%|██████▉   | 278130/400000 [00:32<00:14, 8687.32it/s] 70%|██████▉   | 279016/400000 [00:32<00:13, 8736.28it/s] 70%|██████▉   | 279890/400000 [00:32<00:13, 8729.79it/s] 70%|███████   | 280764/400000 [00:32<00:13, 8709.16it/s] 70%|███████   | 281645/400000 [00:32<00:13, 8736.78it/s] 71%|███████   | 282519/400000 [00:32<00:13, 8705.27it/s] 71%|███████   | 283399/400000 [00:32<00:13, 8731.53it/s] 71%|███████   | 284291/400000 [00:32<00:13, 8786.82it/s] 71%|███████▏  | 285170/400000 [00:32<00:13, 8733.85it/s] 72%|███████▏  | 286044/400000 [00:33<00:13, 8703.86it/s] 72%|███████▏  | 286924/400000 [00:33<00:12, 8730.95it/s] 72%|███████▏  | 287803/400000 [00:33<00:12, 8746.49it/s] 72%|███████▏  | 288680/400000 [00:33<00:12, 8753.47it/s] 72%|███████▏  | 289556/400000 [00:33<00:12, 8727.39it/s] 73%|███████▎  | 290442/400000 [00:33<00:12, 8763.34it/s] 73%|███████▎  | 291344/400000 [00:33<00:12, 8836.42it/s] 73%|███████▎  | 292228/400000 [00:33<00:12, 8666.28it/s] 73%|███████▎  | 293117/400000 [00:33<00:12, 8730.11it/s] 73%|███████▎  | 293991/400000 [00:33<00:12, 8715.34it/s] 74%|███████▎  | 294891/400000 [00:34<00:11, 8798.50it/s] 74%|███████▍  | 295772/400000 [00:34<00:11, 8738.37it/s] 74%|███████▍  | 296669/400000 [00:34<00:11, 8804.58it/s] 74%|███████▍  | 297550/400000 [00:34<00:11, 8727.13it/s] 75%|███████▍  | 298424/400000 [00:34<00:11, 8704.23it/s] 75%|███████▍  | 299319/400000 [00:34<00:11, 8775.16it/s] 75%|███████▌  | 300201/400000 [00:34<00:11, 8786.89it/s] 75%|███████▌  | 301090/400000 [00:34<00:11, 8815.79it/s] 75%|███████▌  | 301972/400000 [00:34<00:11, 8811.18it/s] 76%|███████▌  | 302854/400000 [00:34<00:11, 8707.07it/s] 76%|███████▌  | 303726/400000 [00:35<00:11, 8615.36it/s] 76%|███████▌  | 304610/400000 [00:35<00:10, 8679.46it/s] 76%|███████▋  | 305486/400000 [00:35<00:10, 8700.82it/s] 77%|███████▋  | 306357/400000 [00:35<00:10, 8620.44it/s] 77%|███████▋  | 307220/400000 [00:35<00:10, 8615.08it/s] 77%|███████▋  | 308082/400000 [00:35<00:10, 8601.92it/s] 77%|███████▋  | 308943/400000 [00:35<00:10, 8588.74it/s] 77%|███████▋  | 309820/400000 [00:35<00:10, 8640.18it/s] 78%|███████▊  | 310699/400000 [00:35<00:10, 8683.83it/s] 78%|███████▊  | 311568/400000 [00:35<00:10, 8668.21it/s] 78%|███████▊  | 312435/400000 [00:36<00:10, 8635.68it/s] 78%|███████▊  | 313313/400000 [00:36<00:09, 8676.22it/s] 79%|███████▊  | 314193/400000 [00:36<00:09, 8712.56it/s] 79%|███████▉  | 315065/400000 [00:36<00:09, 8621.43it/s] 79%|███████▉  | 315933/400000 [00:36<00:09, 8636.64it/s] 79%|███████▉  | 316797/400000 [00:36<00:09, 8602.83it/s] 79%|███████▉  | 317679/400000 [00:36<00:09, 8665.91it/s] 80%|███████▉  | 318565/400000 [00:36<00:09, 8720.71it/s] 80%|███████▉  | 319438/400000 [00:36<00:09, 8722.98it/s] 80%|████████  | 320316/400000 [00:36<00:09, 8738.23it/s] 80%|████████  | 321190/400000 [00:37<00:09, 8444.20it/s] 81%|████████  | 322037/400000 [00:37<00:09, 8424.03it/s] 81%|████████  | 322908/400000 [00:37<00:09, 8506.73it/s] 81%|████████  | 323771/400000 [00:37<00:08, 8540.87it/s] 81%|████████  | 324639/400000 [00:37<00:08, 8581.20it/s] 81%|████████▏ | 325498/400000 [00:37<00:08, 8323.89it/s] 82%|████████▏ | 326374/400000 [00:37<00:08, 8447.84it/s] 82%|████████▏ | 327256/400000 [00:37<00:08, 8553.56it/s] 82%|████████▏ | 328139/400000 [00:37<00:08, 8634.15it/s] 82%|████████▏ | 329004/400000 [00:37<00:08, 8610.89it/s] 82%|████████▏ | 329873/400000 [00:38<00:08, 8631.78it/s] 83%|████████▎ | 330765/400000 [00:38<00:07, 8715.99it/s] 83%|████████▎ | 331656/400000 [00:38<00:07, 8773.06it/s] 83%|████████▎ | 332546/400000 [00:38<00:07, 8809.27it/s] 83%|████████▎ | 333428/400000 [00:38<00:07, 8706.88it/s] 84%|████████▎ | 334305/400000 [00:38<00:07, 8725.00it/s] 84%|████████▍ | 335190/400000 [00:38<00:07, 8761.62it/s] 84%|████████▍ | 336076/400000 [00:38<00:07, 8788.38it/s] 84%|████████▍ | 336975/400000 [00:38<00:07, 8846.78it/s] 84%|████████▍ | 337860/400000 [00:39<00:07, 8765.87it/s] 85%|████████▍ | 338739/400000 [00:39<00:06, 8770.95it/s] 85%|████████▍ | 339632/400000 [00:39<00:06, 8816.29it/s] 85%|████████▌ | 340514/400000 [00:39<00:06, 8813.61it/s] 85%|████████▌ | 341396/400000 [00:39<00:06, 8787.92it/s] 86%|████████▌ | 342275/400000 [00:39<00:06, 8773.62it/s] 86%|████████▌ | 343157/400000 [00:39<00:06, 8785.88it/s] 86%|████████▌ | 344036/400000 [00:39<00:06, 8679.55it/s] 86%|████████▌ | 344927/400000 [00:39<00:06, 8747.19it/s] 86%|████████▋ | 345803/400000 [00:39<00:06, 8745.56it/s] 87%|████████▋ | 346678/400000 [00:40<00:06, 8727.31it/s] 87%|████████▋ | 347557/400000 [00:40<00:05, 8744.05it/s] 87%|████████▋ | 348432/400000 [00:40<00:05, 8732.73it/s] 87%|████████▋ | 349316/400000 [00:40<00:05, 8763.13it/s] 88%|████████▊ | 350213/400000 [00:40<00:05, 8823.30it/s] 88%|████████▊ | 351096/400000 [00:40<00:05, 8807.10it/s] 88%|████████▊ | 351980/400000 [00:40<00:05, 8816.76it/s] 88%|████████▊ | 352864/400000 [00:40<00:05, 8822.17it/s] 88%|████████▊ | 353749/400000 [00:40<00:05, 8827.81it/s] 89%|████████▊ | 354635/400000 [00:40<00:05, 8836.07it/s] 89%|████████▉ | 355519/400000 [00:41<00:05, 8798.94it/s] 89%|████████▉ | 356399/400000 [00:41<00:04, 8787.67it/s] 89%|████████▉ | 357278/400000 [00:41<00:04, 8754.26it/s] 90%|████████▉ | 358154/400000 [00:41<00:04, 8746.78it/s] 90%|████████▉ | 359029/400000 [00:41<00:04, 8741.86it/s] 90%|████████▉ | 359908/400000 [00:41<00:04, 8753.51it/s] 90%|█████████ | 360786/400000 [00:41<00:04, 8759.26it/s] 90%|█████████ | 361662/400000 [00:41<00:04, 8750.46it/s] 91%|█████████ | 362548/400000 [00:41<00:04, 8780.72it/s] 91%|█████████ | 363440/400000 [00:41<00:04, 8821.57it/s] 91%|█████████ | 364336/400000 [00:42<00:04, 8860.66it/s] 91%|█████████▏| 365223/400000 [00:42<00:03, 8846.46it/s] 92%|█████████▏| 366115/400000 [00:42<00:03, 8867.04it/s] 92%|█████████▏| 367002/400000 [00:42<00:03, 8806.05it/s] 92%|█████████▏| 367883/400000 [00:42<00:03, 8527.82it/s] 92%|█████████▏| 368769/400000 [00:42<00:03, 8623.27it/s] 92%|█████████▏| 369633/400000 [00:42<00:03, 8623.62it/s] 93%|█████████▎| 370509/400000 [00:42<00:03, 8663.76it/s] 93%|█████████▎| 371391/400000 [00:42<00:03, 8708.47it/s] 93%|█████████▎| 372263/400000 [00:42<00:03, 8651.78it/s] 93%|█████████▎| 373129/400000 [00:43<00:03, 8653.93it/s] 94%|█████████▎| 374002/400000 [00:43<00:02, 8675.93it/s] 94%|█████████▎| 374883/400000 [00:43<00:02, 8714.99it/s] 94%|█████████▍| 375761/400000 [00:43<00:02, 8732.10it/s] 94%|█████████▍| 376637/400000 [00:43<00:02, 8737.88it/s] 94%|█████████▍| 377511/400000 [00:43<00:02, 8656.18it/s] 95%|█████████▍| 378385/400000 [00:43<00:02, 8681.04it/s] 95%|█████████▍| 379257/400000 [00:43<00:02, 8692.48it/s] 95%|█████████▌| 380155/400000 [00:43<00:02, 8775.73it/s] 95%|█████████▌| 381055/400000 [00:43<00:02, 8839.66it/s] 95%|█████████▌| 381940/400000 [00:44<00:02, 8833.45it/s] 96%|█████████▌| 382824/400000 [00:44<00:01, 8633.59it/s] 96%|█████████▌| 383689/400000 [00:44<00:01, 8604.47it/s] 96%|█████████▌| 384577/400000 [00:44<00:01, 8683.95it/s] 96%|█████████▋| 385477/400000 [00:44<00:01, 8774.54it/s] 97%|█████████▋| 386362/400000 [00:44<00:01, 8794.97it/s] 97%|█████████▋| 387243/400000 [00:44<00:01, 8738.34it/s] 97%|█████████▋| 388118/400000 [00:44<00:01, 8656.49it/s] 97%|█████████▋| 388995/400000 [00:44<00:01, 8690.16it/s] 97%|█████████▋| 389882/400000 [00:44<00:01, 8741.48it/s] 98%|█████████▊| 390769/400000 [00:45<00:01, 8778.47it/s] 98%|█████████▊| 391648/400000 [00:45<00:00, 8731.24it/s] 98%|█████████▊| 392525/400000 [00:45<00:00, 8740.98it/s] 98%|█████████▊| 393410/400000 [00:45<00:00, 8771.49it/s] 99%|█████████▊| 394308/400000 [00:45<00:00, 8830.53it/s] 99%|█████████▉| 395192/400000 [00:45<00:00, 8825.79it/s] 99%|█████████▉| 396075/400000 [00:45<00:00, 8775.68it/s] 99%|█████████▉| 396958/400000 [00:45<00:00, 8782.31it/s] 99%|█████████▉| 397837/400000 [00:45<00:00, 8751.56it/s]100%|█████████▉| 398713/400000 [00:45<00:00, 8747.53it/s]100%|█████████▉| 399588/400000 [00:46<00:00, 8739.59it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8674.17it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5a84368940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010940446021389723 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010951964950880478 	 Accuracy: 69

  model saves at 69% accuracy 

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
2020-05-15 08:24:28.795082: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 08:24:28.799583: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 08:24:28.799720: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559d4639e610 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 08:24:28.799734: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5a3038c908> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5210 - accuracy: 0.5095 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5031 - accuracy: 0.5107
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7024 - accuracy: 0.4977
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7367 - accuracy: 0.4954
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6939 - accuracy: 0.4982
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7310 - accuracy: 0.4958
11000/25000 [============>.................] - ETA: 3s - loss: 7.7614 - accuracy: 0.4938
12000/25000 [=============>................] - ETA: 3s - loss: 7.7803 - accuracy: 0.4926
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7633 - accuracy: 0.4937
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7608 - accuracy: 0.4939
15000/25000 [=================>............] - ETA: 2s - loss: 7.7464 - accuracy: 0.4948
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7203 - accuracy: 0.4965
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7072 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7075 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7062 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7142 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f59e91ba6d8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f5a2d7e11d0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9030 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.7535 - val_crf_viterbi_accuracy: 0.0000e+00

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fccf445cf98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 09:17:29.041198
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 09:17:29.045945
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 09:17:29.049986
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 09:17:29.054122
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fcd00226438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356380.7188
Epoch 2/10

1/1 [==============================] - 0s 121ms/step - loss: 286845.9688
Epoch 3/10

1/1 [==============================] - 0s 105ms/step - loss: 222369.1406
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 151933.4844
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 100427.3828
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 65565.6562
Epoch 7/10

1/1 [==============================] - 0s 111ms/step - loss: 43601.7266
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 29799.5332
Epoch 9/10

1/1 [==============================] - 0s 107ms/step - loss: 21078.6855
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 15419.4326

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.12680954 -1.0404698   0.38834563  0.8675374   1.0665643   0.33056068
   1.0433509   0.01284152  1.0635962  -0.655474   -0.23010167 -1.0843196
   1.1035831   0.9588459   0.93971133 -0.6275152   0.08521181 -1.2252434
  -0.792286    1.7592144  -0.7962365  -0.5820346  -0.5640918  -0.3044012
  -1.6238534   0.40672332 -0.9275404  -0.16466701  0.59298116  1.2867206
   0.48866844 -0.13986745 -0.3525412   1.0687771   0.58541125 -0.54195845
   1.1114502   1.8734913   0.497862    0.8673242  -0.9672555   0.5269027
  -0.52871746  0.6585804   0.5449154   0.19212523 -1.5991507  -1.0673552
   0.37824154  1.089081   -1.1643109  -0.7148575   0.2872494  -0.86993134
   1.0133913  -0.8206355  -0.01976855  0.5445701  -1.2567909   0.65989256
   0.25246626  3.336927    5.12121     3.750726    4.5015745   5.1633143
   5.754431    4.0108414   6.156516    3.882543    5.422984    5.6417704
   5.7605586   6.4160366   3.7247548   5.7754965   5.0586257   6.5057254
   4.3811073   4.1465783   4.674085    5.3195686   6.493602    5.1420946
   4.45352     3.335841    4.1333666   6.2094755   4.254305    4.8534102
   5.474845    4.446326    5.9438086   6.4553905   3.778025    5.957939
   3.9835005   3.5714643   5.033292    4.183853    5.730697    4.12286
   5.2945      4.254964    5.8935204   3.8086333   3.924427    5.2468653
   5.4281907   5.479378    4.4481387   5.4280705   4.656249    4.9784327
   3.433085    3.488369    3.287997    5.8466425   4.717288    4.9485955
   0.1801941   0.8849145  -0.3438244  -0.10693341  0.51457214  0.40835464
   0.33091557  1.498318   -0.765706   -0.7558395   0.22833388 -0.24892372
   0.9001789   0.94092005  0.34805208  0.28383595 -1.0227056   1.2732222
   0.33385265 -0.1755034   0.711362    0.8444698  -1.3025174  -0.9417865
   0.10543025  1.2804239   0.41800085 -0.19221449 -0.22433837  0.23203169
   0.42349702  0.59077114  1.747097   -0.9028917   1.2933404  -1.1600596
   1.4127253   0.7424753  -1.0721843  -1.7628859  -0.79470265 -0.42056248
   0.7085353  -0.09819961  0.22504935 -0.7609428   0.9263632  -0.99184394
  -0.6569505   0.5052355   0.75386596  0.5681468   0.1684005   1.1211987
  -1.4865121   0.25027454  1.8891351  -0.76584554 -0.41780633 -1.6791971
   1.5293183   0.89806426  1.9480557   1.1281445   1.0637481   0.20384657
   1.390901    0.78393334  0.7598895   1.1685596   0.39316714  0.7224915
   0.3144474   2.468542    1.6902924   0.26007926  2.2718349   1.7039766
   1.4357998   1.0760844   0.37874293  1.323332    0.46483064  1.6292222
   2.0193696   0.3021469   0.63907677  0.28024757  0.23912853  1.5825293
   1.9258051   0.6777655   1.0595884   1.58479     0.30138785  0.9880138
   0.6978168   1.6532722   1.5361627   2.2887158   2.7240105   1.0269964
   0.9268089   2.204904    1.6024277   1.4077486   0.5543818   0.16958559
   1.5826465   1.4644984   0.550775    0.9559588   0.7272462   0.61607534
   1.7514312   0.7096064   0.3102886   0.23523486  0.65355754  0.46979177
   0.01746291  4.836152    6.5056005   4.125735    5.317465    5.821607
   6.536709    4.789035    5.6526165   6.33742     5.526361    6.001709
   6.1566353   5.7824206   5.0376043   5.1247764   6.931457    5.5709763
   5.7501316   5.2750316   5.625023    4.812381    4.975884    5.7763634
   5.9453683   6.321364    4.2195215   5.8084917   6.283428    6.238141
   4.9378176   5.7463174   5.1493692   6.615039    6.05731     4.206634
   6.600381    5.2184505   6.395066    5.518094    6.4245143   3.931497
   5.851539    4.9394      6.9366503   6.142642    4.65076     4.2364597
   4.6755853   6.973539    6.0506616   6.3058724   5.263035    6.8669024
   5.2270427   6.4740915   6.7986455   6.300164    5.9053607   4.632951
   2.1479383   1.2849385   1.2498666   1.647873    0.6905626   2.218915
   1.0418913   1.2603849   0.8835328   0.38080412  0.38021952  1.0779073
   1.5687463   2.4832358   1.6131625   0.889151    1.8392264   2.3099694
   0.48666     0.335019    1.2264795   0.4144119   0.9632603   1.7139053
   2.6438265   1.6311024   2.5568166   1.1710742   0.57355136  0.36283946
   0.39368665  2.8134542   0.32593715  1.5959938   0.45880175  1.678253
   0.6032137   1.1377866   0.5831948   0.60646975  0.48484725  1.1768818
   0.18871868  1.9361489   1.956745    1.9670208   2.3331842   1.4695691
   1.1016331   2.381073    0.99233544  0.980035    0.43395495  1.8372025
   1.221209    1.5517297   0.48608345  0.8647845   0.3930893   1.3582757
  -8.72053     1.1320089  -8.602814  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 09:17:39.526180
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.891
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 09:17:39.530670
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9402.97
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 09:17:39.534893
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.0624
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 09:17:39.539762
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -841.102
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140517889139208
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140515376644672
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140515376645176
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140515376645680
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140515376646184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140515376646688

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fccfc0ba518> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.548397
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.526300
grad_step = 000002, loss = 0.507704
grad_step = 000003, loss = 0.491011
grad_step = 000004, loss = 0.477080
grad_step = 000005, loss = 0.461077
grad_step = 000006, loss = 0.442439
grad_step = 000007, loss = 0.424865
grad_step = 000008, loss = 0.410606
grad_step = 000009, loss = 0.399324
grad_step = 000010, loss = 0.386884
grad_step = 000011, loss = 0.374089
grad_step = 000012, loss = 0.359865
grad_step = 000013, loss = 0.345641
grad_step = 000014, loss = 0.331209
grad_step = 000015, loss = 0.317107
grad_step = 000016, loss = 0.303837
grad_step = 000017, loss = 0.291871
grad_step = 000018, loss = 0.280727
grad_step = 000019, loss = 0.269188
grad_step = 000020, loss = 0.258530
grad_step = 000021, loss = 0.248923
grad_step = 000022, loss = 0.239502
grad_step = 000023, loss = 0.230102
grad_step = 000024, loss = 0.220765
grad_step = 000025, loss = 0.211078
grad_step = 000026, loss = 0.200699
grad_step = 000027, loss = 0.190244
grad_step = 000028, loss = 0.180459
grad_step = 000029, loss = 0.171487
grad_step = 000030, loss = 0.162722
grad_step = 000031, loss = 0.154352
grad_step = 000032, loss = 0.146747
grad_step = 000033, loss = 0.139314
grad_step = 000034, loss = 0.132033
grad_step = 000035, loss = 0.124899
grad_step = 000036, loss = 0.117979
grad_step = 000037, loss = 0.111318
grad_step = 000038, loss = 0.104882
grad_step = 000039, loss = 0.098891
grad_step = 000040, loss = 0.093114
grad_step = 000041, loss = 0.087593
grad_step = 000042, loss = 0.082318
grad_step = 000043, loss = 0.077266
grad_step = 000044, loss = 0.072421
grad_step = 000045, loss = 0.067837
grad_step = 000046, loss = 0.063506
grad_step = 000047, loss = 0.059260
grad_step = 000048, loss = 0.055279
grad_step = 000049, loss = 0.051453
grad_step = 000050, loss = 0.047844
grad_step = 000051, loss = 0.044452
grad_step = 000052, loss = 0.041271
grad_step = 000053, loss = 0.038235
grad_step = 000054, loss = 0.035408
grad_step = 000055, loss = 0.032718
grad_step = 000056, loss = 0.030181
grad_step = 000057, loss = 0.027803
grad_step = 000058, loss = 0.025581
grad_step = 000059, loss = 0.023499
grad_step = 000060, loss = 0.021582
grad_step = 000061, loss = 0.019813
grad_step = 000062, loss = 0.018176
grad_step = 000063, loss = 0.016657
grad_step = 000064, loss = 0.015237
grad_step = 000065, loss = 0.013932
grad_step = 000066, loss = 0.012738
grad_step = 000067, loss = 0.011655
grad_step = 000068, loss = 0.010660
grad_step = 000069, loss = 0.009768
grad_step = 000070, loss = 0.008949
grad_step = 000071, loss = 0.008207
grad_step = 000072, loss = 0.007531
grad_step = 000073, loss = 0.006917
grad_step = 000074, loss = 0.006367
grad_step = 000075, loss = 0.005869
grad_step = 000076, loss = 0.005427
grad_step = 000077, loss = 0.005034
grad_step = 000078, loss = 0.004683
grad_step = 000079, loss = 0.004369
grad_step = 000080, loss = 0.004090
grad_step = 000081, loss = 0.003840
grad_step = 000082, loss = 0.003618
grad_step = 000083, loss = 0.003422
grad_step = 000084, loss = 0.003253
grad_step = 000085, loss = 0.003103
grad_step = 000086, loss = 0.002971
grad_step = 000087, loss = 0.002854
grad_step = 000088, loss = 0.002749
grad_step = 000089, loss = 0.002657
grad_step = 000090, loss = 0.002577
grad_step = 000091, loss = 0.002506
grad_step = 000092, loss = 0.002443
grad_step = 000093, loss = 0.002388
grad_step = 000094, loss = 0.002340
grad_step = 000095, loss = 0.002296
grad_step = 000096, loss = 0.002257
grad_step = 000097, loss = 0.002222
grad_step = 000098, loss = 0.002191
grad_step = 000099, loss = 0.002164
grad_step = 000100, loss = 0.002141
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002120
grad_step = 000102, loss = 0.002101
grad_step = 000103, loss = 0.002084
grad_step = 000104, loss = 0.002069
grad_step = 000105, loss = 0.002056
grad_step = 000106, loss = 0.002044
grad_step = 000107, loss = 0.002033
grad_step = 000108, loss = 0.002023
grad_step = 000109, loss = 0.002014
grad_step = 000110, loss = 0.002007
grad_step = 000111, loss = 0.002002
grad_step = 000112, loss = 0.001997
grad_step = 000113, loss = 0.001990
grad_step = 000114, loss = 0.001980
grad_step = 000115, loss = 0.001971
grad_step = 000116, loss = 0.001962
grad_step = 000117, loss = 0.001955
grad_step = 000118, loss = 0.001951
grad_step = 000119, loss = 0.001952
grad_step = 000120, loss = 0.001961
grad_step = 000121, loss = 0.001977
grad_step = 000122, loss = 0.001993
grad_step = 000123, loss = 0.001992
grad_step = 000124, loss = 0.001974
grad_step = 000125, loss = 0.001957
grad_step = 000126, loss = 0.001961
grad_step = 000127, loss = 0.001975
grad_step = 000128, loss = 0.001964
grad_step = 000129, loss = 0.001909
grad_step = 000130, loss = 0.001885
grad_step = 000131, loss = 0.001903
grad_step = 000132, loss = 0.001903
grad_step = 000133, loss = 0.001875
grad_step = 000134, loss = 0.001860
grad_step = 000135, loss = 0.001876
grad_step = 000136, loss = 0.001890
grad_step = 000137, loss = 0.001886
grad_step = 000138, loss = 0.001895
grad_step = 000139, loss = 0.001951
grad_step = 000140, loss = 0.002038
grad_step = 000141, loss = 0.002109
grad_step = 000142, loss = 0.002066
grad_step = 000143, loss = 0.001934
grad_step = 000144, loss = 0.001824
grad_step = 000145, loss = 0.001852
grad_step = 000146, loss = 0.001945
grad_step = 000147, loss = 0.001950
grad_step = 000148, loss = 0.001862
grad_step = 000149, loss = 0.001802
grad_step = 000150, loss = 0.001825
grad_step = 000151, loss = 0.001875
grad_step = 000152, loss = 0.001881
grad_step = 000153, loss = 0.001847
grad_step = 000154, loss = 0.001802
grad_step = 000155, loss = 0.001782
grad_step = 000156, loss = 0.001795
grad_step = 000157, loss = 0.001819
grad_step = 000158, loss = 0.001830
grad_step = 000159, loss = 0.001809
grad_step = 000160, loss = 0.001773
grad_step = 000161, loss = 0.001751
grad_step = 000162, loss = 0.001758
grad_step = 000163, loss = 0.001776
grad_step = 000164, loss = 0.001784
grad_step = 000165, loss = 0.001779
grad_step = 000166, loss = 0.001762
grad_step = 000167, loss = 0.001745
grad_step = 000168, loss = 0.001731
grad_step = 000169, loss = 0.001723
grad_step = 000170, loss = 0.001723
grad_step = 000171, loss = 0.001731
grad_step = 000172, loss = 0.001742
grad_step = 000173, loss = 0.001751
grad_step = 000174, loss = 0.001764
grad_step = 000175, loss = 0.001779
grad_step = 000176, loss = 0.001802
grad_step = 000177, loss = 0.001812
grad_step = 000178, loss = 0.001804
grad_step = 000179, loss = 0.001768
grad_step = 000180, loss = 0.001731
grad_step = 000181, loss = 0.001700
grad_step = 000182, loss = 0.001679
grad_step = 000183, loss = 0.001673
grad_step = 000184, loss = 0.001685
grad_step = 000185, loss = 0.001713
grad_step = 000186, loss = 0.001743
grad_step = 000187, loss = 0.001779
grad_step = 000188, loss = 0.001811
grad_step = 000189, loss = 0.001853
grad_step = 000190, loss = 0.001854
grad_step = 000191, loss = 0.001793
grad_step = 000192, loss = 0.001689
grad_step = 000193, loss = 0.001637
grad_step = 000194, loss = 0.001647
grad_step = 000195, loss = 0.001675
grad_step = 000196, loss = 0.001697
grad_step = 000197, loss = 0.001711
grad_step = 000198, loss = 0.001710
grad_step = 000199, loss = 0.001659
grad_step = 000200, loss = 0.001607
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001591
grad_step = 000202, loss = 0.001605
grad_step = 000203, loss = 0.001616
grad_step = 000204, loss = 0.001615
grad_step = 000205, loss = 0.001621
grad_step = 000206, loss = 0.001634
grad_step = 000207, loss = 0.001635
grad_step = 000208, loss = 0.001613
grad_step = 000209, loss = 0.001588
grad_step = 000210, loss = 0.001575
grad_step = 000211, loss = 0.001569
grad_step = 000212, loss = 0.001554
grad_step = 000213, loss = 0.001537
grad_step = 000214, loss = 0.001529
grad_step = 000215, loss = 0.001530
grad_step = 000216, loss = 0.001528
grad_step = 000217, loss = 0.001521
grad_step = 000218, loss = 0.001512
grad_step = 000219, loss = 0.001509
grad_step = 000220, loss = 0.001512
grad_step = 000221, loss = 0.001519
grad_step = 000222, loss = 0.001540
grad_step = 000223, loss = 0.001596
grad_step = 000224, loss = 0.001754
grad_step = 000225, loss = 0.002016
grad_step = 000226, loss = 0.002408
grad_step = 000227, loss = 0.002291
grad_step = 000228, loss = 0.001755
grad_step = 000229, loss = 0.001516
grad_step = 000230, loss = 0.001849
grad_step = 000231, loss = 0.001993
grad_step = 000232, loss = 0.001624
grad_step = 000233, loss = 0.001537
grad_step = 000234, loss = 0.001834
grad_step = 000235, loss = 0.001774
grad_step = 000236, loss = 0.001515
grad_step = 000237, loss = 0.001553
grad_step = 000238, loss = 0.001776
grad_step = 000239, loss = 0.001709
grad_step = 000240, loss = 0.001494
grad_step = 000241, loss = 0.001571
grad_step = 000242, loss = 0.001743
grad_step = 000243, loss = 0.001591
grad_step = 000244, loss = 0.001480
grad_step = 000245, loss = 0.001566
grad_step = 000246, loss = 0.001617
grad_step = 000247, loss = 0.001496
grad_step = 000248, loss = 0.001478
grad_step = 000249, loss = 0.001535
grad_step = 000250, loss = 0.001533
grad_step = 000251, loss = 0.001450
grad_step = 000252, loss = 0.001479
grad_step = 000253, loss = 0.001508
grad_step = 000254, loss = 0.001474
grad_step = 000255, loss = 0.001434
grad_step = 000256, loss = 0.001475
grad_step = 000257, loss = 0.001474
grad_step = 000258, loss = 0.001438
grad_step = 000259, loss = 0.001436
grad_step = 000260, loss = 0.001462
grad_step = 000261, loss = 0.001445
grad_step = 000262, loss = 0.001426
grad_step = 000263, loss = 0.001432
grad_step = 000264, loss = 0.001443
grad_step = 000265, loss = 0.001429
grad_step = 000266, loss = 0.001420
grad_step = 000267, loss = 0.001422
grad_step = 000268, loss = 0.001428
grad_step = 000269, loss = 0.001421
grad_step = 000270, loss = 0.001411
grad_step = 000271, loss = 0.001411
grad_step = 000272, loss = 0.001417
grad_step = 000273, loss = 0.001413
grad_step = 000274, loss = 0.001404
grad_step = 000275, loss = 0.001402
grad_step = 000276, loss = 0.001406
grad_step = 000277, loss = 0.001405
grad_step = 000278, loss = 0.001399
grad_step = 000279, loss = 0.001396
grad_step = 000280, loss = 0.001395
grad_step = 000281, loss = 0.001395
grad_step = 000282, loss = 0.001394
grad_step = 000283, loss = 0.001391
grad_step = 000284, loss = 0.001388
grad_step = 000285, loss = 0.001386
grad_step = 000286, loss = 0.001385
grad_step = 000287, loss = 0.001385
grad_step = 000288, loss = 0.001384
grad_step = 000289, loss = 0.001381
grad_step = 000290, loss = 0.001378
grad_step = 000291, loss = 0.001377
grad_step = 000292, loss = 0.001375
grad_step = 000293, loss = 0.001374
grad_step = 000294, loss = 0.001373
grad_step = 000295, loss = 0.001372
grad_step = 000296, loss = 0.001371
grad_step = 000297, loss = 0.001368
grad_step = 000298, loss = 0.001366
grad_step = 000299, loss = 0.001365
grad_step = 000300, loss = 0.001363
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001361
grad_step = 000302, loss = 0.001360
grad_step = 000303, loss = 0.001359
grad_step = 000304, loss = 0.001358
grad_step = 000305, loss = 0.001356
grad_step = 000306, loss = 0.001356
grad_step = 000307, loss = 0.001355
grad_step = 000308, loss = 0.001355
grad_step = 000309, loss = 0.001355
grad_step = 000310, loss = 0.001358
grad_step = 000311, loss = 0.001363
grad_step = 000312, loss = 0.001373
grad_step = 000313, loss = 0.001394
grad_step = 000314, loss = 0.001431
grad_step = 000315, loss = 0.001490
grad_step = 000316, loss = 0.001564
grad_step = 000317, loss = 0.001618
grad_step = 000318, loss = 0.001614
grad_step = 000319, loss = 0.001535
grad_step = 000320, loss = 0.001429
grad_step = 000321, loss = 0.001369
grad_step = 000322, loss = 0.001370
grad_step = 000323, loss = 0.001411
grad_step = 000324, loss = 0.001453
grad_step = 000325, loss = 0.001446
grad_step = 000326, loss = 0.001388
grad_step = 000327, loss = 0.001334
grad_step = 000328, loss = 0.001339
grad_step = 000329, loss = 0.001382
grad_step = 000330, loss = 0.001403
grad_step = 000331, loss = 0.001384
grad_step = 000332, loss = 0.001350
grad_step = 000333, loss = 0.001333
grad_step = 000334, loss = 0.001333
grad_step = 000335, loss = 0.001336
grad_step = 000336, loss = 0.001340
grad_step = 000337, loss = 0.001347
grad_step = 000338, loss = 0.001353
grad_step = 000339, loss = 0.001350
grad_step = 000340, loss = 0.001334
grad_step = 000341, loss = 0.001316
grad_step = 000342, loss = 0.001307
grad_step = 000343, loss = 0.001307
grad_step = 000344, loss = 0.001312
grad_step = 000345, loss = 0.001317
grad_step = 000346, loss = 0.001319
grad_step = 000347, loss = 0.001320
grad_step = 000348, loss = 0.001321
grad_step = 000349, loss = 0.001321
grad_step = 000350, loss = 0.001318
grad_step = 000351, loss = 0.001313
grad_step = 000352, loss = 0.001306
grad_step = 000353, loss = 0.001301
grad_step = 000354, loss = 0.001297
grad_step = 000355, loss = 0.001295
grad_step = 000356, loss = 0.001293
grad_step = 000357, loss = 0.001290
grad_step = 000358, loss = 0.001288
grad_step = 000359, loss = 0.001286
grad_step = 000360, loss = 0.001285
grad_step = 000361, loss = 0.001285
grad_step = 000362, loss = 0.001286
grad_step = 000363, loss = 0.001290
grad_step = 000364, loss = 0.001297
grad_step = 000365, loss = 0.001311
grad_step = 000366, loss = 0.001338
grad_step = 000367, loss = 0.001391
grad_step = 000368, loss = 0.001480
grad_step = 000369, loss = 0.001625
grad_step = 000370, loss = 0.001779
grad_step = 000371, loss = 0.001868
grad_step = 000372, loss = 0.001768
grad_step = 000373, loss = 0.001516
grad_step = 000374, loss = 0.001323
grad_step = 000375, loss = 0.001334
grad_step = 000376, loss = 0.001453
grad_step = 000377, loss = 0.001501
grad_step = 000378, loss = 0.001424
grad_step = 000379, loss = 0.001317
grad_step = 000380, loss = 0.001302
grad_step = 000381, loss = 0.001367
grad_step = 000382, loss = 0.001403
grad_step = 000383, loss = 0.001356
grad_step = 000384, loss = 0.001278
grad_step = 000385, loss = 0.001267
grad_step = 000386, loss = 0.001325
grad_step = 000387, loss = 0.001360
grad_step = 000388, loss = 0.001320
grad_step = 000389, loss = 0.001258
grad_step = 000390, loss = 0.001250
grad_step = 000391, loss = 0.001290
grad_step = 000392, loss = 0.001307
grad_step = 000393, loss = 0.001281
grad_step = 000394, loss = 0.001247
grad_step = 000395, loss = 0.001244
grad_step = 000396, loss = 0.001263
grad_step = 000397, loss = 0.001270
grad_step = 000398, loss = 0.001257
grad_step = 000399, loss = 0.001240
grad_step = 000400, loss = 0.001238
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001244
grad_step = 000402, loss = 0.001245
grad_step = 000403, loss = 0.001238
grad_step = 000404, loss = 0.001231
grad_step = 000405, loss = 0.001230
grad_step = 000406, loss = 0.001232
grad_step = 000407, loss = 0.001231
grad_step = 000408, loss = 0.001225
grad_step = 000409, loss = 0.001219
grad_step = 000410, loss = 0.001219
grad_step = 000411, loss = 0.001221
grad_step = 000412, loss = 0.001223
grad_step = 000413, loss = 0.001220
grad_step = 000414, loss = 0.001214
grad_step = 000415, loss = 0.001208
grad_step = 000416, loss = 0.001206
grad_step = 000417, loss = 0.001206
grad_step = 000418, loss = 0.001208
grad_step = 000419, loss = 0.001208
grad_step = 000420, loss = 0.001207
grad_step = 000421, loss = 0.001204
grad_step = 000422, loss = 0.001201
grad_step = 000423, loss = 0.001199
grad_step = 000424, loss = 0.001198
grad_step = 000425, loss = 0.001197
grad_step = 000426, loss = 0.001196
grad_step = 000427, loss = 0.001195
grad_step = 000428, loss = 0.001193
grad_step = 000429, loss = 0.001191
grad_step = 000430, loss = 0.001189
grad_step = 000431, loss = 0.001187
grad_step = 000432, loss = 0.001186
grad_step = 000433, loss = 0.001186
grad_step = 000434, loss = 0.001187
grad_step = 000435, loss = 0.001191
grad_step = 000436, loss = 0.001198
grad_step = 000437, loss = 0.001211
grad_step = 000438, loss = 0.001234
grad_step = 000439, loss = 0.001270
grad_step = 000440, loss = 0.001322
grad_step = 000441, loss = 0.001382
grad_step = 000442, loss = 0.001428
grad_step = 000443, loss = 0.001434
grad_step = 000444, loss = 0.001400
grad_step = 000445, loss = 0.001368
grad_step = 000446, loss = 0.001353
grad_step = 000447, loss = 0.001332
grad_step = 000448, loss = 0.001280
grad_step = 000449, loss = 0.001213
grad_step = 000450, loss = 0.001186
grad_step = 000451, loss = 0.001220
grad_step = 000452, loss = 0.001272
grad_step = 000453, loss = 0.001277
grad_step = 000454, loss = 0.001225
grad_step = 000455, loss = 0.001170
grad_step = 000456, loss = 0.001156
grad_step = 000457, loss = 0.001181
grad_step = 000458, loss = 0.001209
grad_step = 000459, loss = 0.001216
grad_step = 000460, loss = 0.001209
grad_step = 000461, loss = 0.001202
grad_step = 000462, loss = 0.001195
grad_step = 000463, loss = 0.001184
grad_step = 000464, loss = 0.001165
grad_step = 000465, loss = 0.001147
grad_step = 000466, loss = 0.001141
grad_step = 000467, loss = 0.001149
grad_step = 000468, loss = 0.001163
grad_step = 000469, loss = 0.001170
grad_step = 000470, loss = 0.001168
grad_step = 000471, loss = 0.001159
grad_step = 000472, loss = 0.001151
grad_step = 000473, loss = 0.001145
grad_step = 000474, loss = 0.001140
grad_step = 000475, loss = 0.001136
grad_step = 000476, loss = 0.001130
grad_step = 000477, loss = 0.001126
grad_step = 000478, loss = 0.001124
grad_step = 000479, loss = 0.001126
grad_step = 000480, loss = 0.001129
grad_step = 000481, loss = 0.001132
grad_step = 000482, loss = 0.001133
grad_step = 000483, loss = 0.001134
grad_step = 000484, loss = 0.001133
grad_step = 000485, loss = 0.001133
grad_step = 000486, loss = 0.001134
grad_step = 000487, loss = 0.001136
grad_step = 000488, loss = 0.001139
grad_step = 000489, loss = 0.001143
grad_step = 000490, loss = 0.001147
grad_step = 000491, loss = 0.001153
grad_step = 000492, loss = 0.001157
grad_step = 000493, loss = 0.001162
grad_step = 000494, loss = 0.001164
grad_step = 000495, loss = 0.001165
grad_step = 000496, loss = 0.001164
grad_step = 000497, loss = 0.001161
grad_step = 000498, loss = 0.001154
grad_step = 000499, loss = 0.001145
grad_step = 000500, loss = 0.001134
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001122
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

  date_run                              2020-05-15 09:18:04.431011
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.311548
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 09:18:04.438076
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.245395
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 09:18:04.446004
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.172057
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 09:18:04.451763
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.72886
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
0   2020-05-15 09:17:29.041198  ...    mean_absolute_error
1   2020-05-15 09:17:29.045945  ...     mean_squared_error
2   2020-05-15 09:17:29.049986  ...  median_absolute_error
3   2020-05-15 09:17:29.054122  ...               r2_score
4   2020-05-15 09:17:39.526180  ...    mean_absolute_error
5   2020-05-15 09:17:39.530670  ...     mean_squared_error
6   2020-05-15 09:17:39.534893  ...  median_absolute_error
7   2020-05-15 09:17:39.539762  ...               r2_score
8   2020-05-15 09:18:04.431011  ...    mean_absolute_error
9   2020-05-15 09:18:04.438076  ...     mean_squared_error
10  2020-05-15 09:18:04.446004  ...  median_absolute_error
11  2020-05-15 09:18:04.451763  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99edc18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 461344.91it/s] 51%|█████     | 5070848/9912422 [00:00<00:07, 656470.91it/s]9920512it [00:00, 35018585.14it/s]                           
0it [00:00, ?it/s]32768it [00:00, 609539.44it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1051228.85it/s]1654784it [00:00, 12969040.14it/s]                           
0it [00:00, ?it/s]8192it [00:00, 254761.91it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b7c3a7e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99b0f28> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b7c3a7e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99b0f28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b791694a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99b0f28> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b7c3a7e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99b0f28> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b791694a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7bc99f9a58> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdea74d41d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d1b2563f8bfd3f6677b0a1782ed261c24e0d89ffd14dc36b9fae16f7d00f45c5
  Stored in directory: /tmp/pip-ephem-wheel-cache-a1wrxtb9/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fde9d63f048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1744896/17464789 [=>............................] - ETA: 0s
 7053312/17464789 [===========>..................] - ETA: 0s
15564800/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 09:19:33.793458: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 09:19:33.798734: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-15 09:19:33.798888: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563f25f75b80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 09:19:33.798906: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9350 - accuracy: 0.4825
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7637 - accuracy: 0.4937 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7088 - accuracy: 0.4972
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7249 - accuracy: 0.4962
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7170 - accuracy: 0.4967
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6751 - accuracy: 0.4994
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.7168 - accuracy: 0.4967
12000/25000 [=============>................] - ETA: 4s - loss: 7.7101 - accuracy: 0.4972
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7232 - accuracy: 0.4963
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7148 - accuracy: 0.4969
15000/25000 [=================>............] - ETA: 3s - loss: 7.7300 - accuracy: 0.4959
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7308 - accuracy: 0.4958
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7243 - accuracy: 0.4962
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7288 - accuracy: 0.4959
19000/25000 [=====================>........] - ETA: 2s - loss: 7.7255 - accuracy: 0.4962
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7004 - accuracy: 0.4978
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 10s 405us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 09:19:51.767274
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 09:19:51.767274  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<130:26:18, 1.84kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<91:32:21, 2.62kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<64:07:15, 3.73kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:04<44:51:48, 5.33kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<31:18:28, 7.62kB/s].vector_cache/glove.6B.zip:   1%|          | 9.69M/862M [00:05<21:45:46, 10.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:05<15:08:00, 15.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:05<10:31:12, 22.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:05<7:18:56, 31.7kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:05<5:05:19, 45.3kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:05<3:32:20, 64.7kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:05<2:27:42, 92.3kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:06<1:42:47, 132kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:06<1:12:03, 188kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:08<52:08, 258kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:08<38:19, 351kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:08<27:11, 494kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:08<19:10, 698kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:10<23:32, 568kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:10<18:08, 737kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:10<13:05, 1.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:12<11:51, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:12<09:47, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:12<07:12, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:14<07:58, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:14<08:35, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:14<06:44, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.6M/862M [00:14<04:51, 2.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:16<17:28, 754kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:16<13:41, 961kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:16<09:56, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:18<09:47, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:18<09:47, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:18<07:35, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.9M/862M [00:18<05:27, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:20<16:12, 804kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:20<12:34, 1.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:20<09:06, 1.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:20<06:33, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:22<15:55, 814kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:22<12:36, 1.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:22<09:10, 1.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:24<09:13, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:24<07:54, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:24<05:52, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:26<06:53, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:26<07:43, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<06:07, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:26<04:26, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:28<18:30, 689kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:28<14:07, 902kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:28<10:14, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<07:21, 1.72MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<16:56, 748kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<14:44, 860kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<10:54, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<07:52, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:32<08:58, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<07:42, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<05:44, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<06:43, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<07:32, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<05:58, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<04:19, 2.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<16:22, 762kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<12:51, 970kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<09:16, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<09:11, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<09:12, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<07:09, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<06:16, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<07:19, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<06:30, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<04:50, 2.54MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<06:59, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<05:35, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:42<04:03, 3.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<08:40, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<07:26, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<05:29, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<06:27, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<07:14, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:46, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<04:10, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<13:58, 863kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<11:08, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:48<08:03, 1.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:47, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<34:08, 351kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<25:01, 479kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<17:44, 674kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<12:35, 948kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<19:15, 619kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<14:48, 805kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<10:41, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<10:00, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<09:40, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<05:22, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<06:49, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<06:06, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<04:32, 2.58MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:38, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<05:15, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<03:48, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<15:05, 771kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<11:52, 980kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<08:34, 1.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:01<08:28, 1.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<08:33, 1.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<06:32, 1.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<04:42, 2.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<09:11, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<07:41, 1.49MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<05:41, 2.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:05<06:28, 1.76MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:06<07:05, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<05:36, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<04:02, 2.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<14:58, 759kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<11:46, 964kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<08:32, 1.33MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<08:25, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<10:31, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<08:34, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:17, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<06:50, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<07:07, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<05:28, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<03:59, 2.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<07:14, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<07:22, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<05:40, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:04, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<19:58, 555kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<16:09, 686kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<11:51, 933kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<10:03, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<09:24, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<07:09, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<56:10, 195kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<41:23, 264kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<29:26, 371kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<22:19, 487kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<17:57, 606kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<13:02, 832kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<09:14, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<12:03, 896kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<10:29, 1.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<07:45, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<05:32, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<12:34, 854kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<11:00, 975kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<08:11, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<05:49, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<12:41, 841kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<10:45, 991kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<07:59, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<07:22, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<07:20, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<05:41, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<04:05, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<53:08, 198kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<39:02, 270kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<27:44, 379kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<21:07, 495kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<16:56, 617kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<12:23, 843kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<08:45, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<1:00:46, 171kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<44:20, 234kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<31:26, 330kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<22:17, 465kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<17:52, 578kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<14:37, 706kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<10:42, 964kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<07:38, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<08:56, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<08:22, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:39<06:23, 1.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<04:33, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<45:35, 223kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<34:00, 300kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<24:13, 420kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<17:03, 595kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<16:24, 617kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<13:33, 746kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<09:56, 1.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<07:04, 1.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<09:48, 1.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<08:55, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<06:41, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<04:51, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<06:36, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<06:15, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<04:43, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<03:25, 2.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<10:56, 907kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<09:41, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<07:17, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<54:54, 179kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<40:26, 243kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<28:47, 342kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<20:10, 485kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<1:23:51, 117kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<1:00:46, 161kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<43:00, 227kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<30:04, 323kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<1:30:15, 108kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<1:05:14, 149kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:55<46:02, 211kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<32:20, 299kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<25:25, 379kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<19:25, 496kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<13:58, 689kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<11:24, 839kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<10:02, 953kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:59<07:31, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<05:21, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<1:11:29, 133kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<52:04, 182kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<36:50, 258kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<25:46, 366kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<25:15, 374kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<19:42, 478kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<14:11, 663kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<10:04, 933kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<10:15, 913kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<09:11, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<06:54, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<04:56, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<1:09:45, 133kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<50:48, 183kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<36:00, 258kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<25:10, 367kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<1:01:57, 149kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<45:20, 204kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<32:10, 286kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<22:30, 407kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<1:21:01, 113kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<58:39, 156kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<41:25, 221kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<28:59, 314kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<25:57, 350kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<20:06, 452kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<14:27, 628kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<10:14, 884kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<10:26, 864kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<09:14, 977kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<06:52, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<04:53, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<14:52, 602kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<12:14, 731kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<09:02, 990kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<06:23, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<51:01, 174kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<37:31, 237kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<26:42, 332kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<18:41, 472kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<1:15:54, 116kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<54:56, 160kB/s]  .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:20<38:47, 227kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<27:09, 323kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<23:28, 373kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<18:13, 480kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<13:11, 662kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<09:17, 934kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<1:08:18, 127kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<49:12, 176kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<34:44, 249kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<25:42, 335kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<19:45, 436kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<14:11, 606kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<10:04, 851kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<09:33, 894kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<08:02, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<05:57, 1.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<05:38, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<05:37, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<04:21, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:32<04:24, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<04:48, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<03:48, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<02:44, 3.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<45:31, 183kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<33:34, 248kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<23:51, 349kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<16:41, 496kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<19:47, 418kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<15:09, 545kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<10:52, 758kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<07:39, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<59:19, 138kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<43:16, 189kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<30:40, 267kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<21:26, 379kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<1:12:58, 111kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<52:21, 155kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<36:50, 220kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<25:47, 313kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<23:07, 349kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<17:54, 450kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<12:56, 622kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<09:05, 879kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<44:39, 179kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<32:31, 246kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<22:58, 347kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<16:06, 493kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<19:32, 406kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<15:17, 518kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<11:03, 715kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<07:47, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<10:50, 725kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<09:11, 854kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<06:47, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<04:49, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<07:29, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<06:26, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:44, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<03:25, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<07:38, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<06:55, 1.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<05:14, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<04:52, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:59, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:52, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<03:55, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<04:14, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<03:18, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<02:23, 3.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<07:59, 940kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<07:08, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<05:19, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<03:48, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<05:55, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<05:40, 1.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<04:21, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<04:13, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<04:27, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:02<03:29, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<03:37, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<04:02, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<03:11, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<03:23, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<03:51, 1.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:00, 2.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<02:11, 3.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<04:54, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<04:54, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:47, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<04:06, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<03:11, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<02:18, 3.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<06:15, 1.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<05:49, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<04:26, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<04:12, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<04:22, 1.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:22, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<02:27, 2.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<04:44, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<04:44, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:39, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<03:39, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:17<03:57, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<03:07, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<03:15, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<03:40, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<02:51, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<02:05, 3.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<04:26, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<04:29, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<03:28, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<02:29, 2.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<2:09:09, 51.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<1:31:45, 72.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<1:04:26, 103kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<45:48, 143kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<33:24, 196kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<23:37, 277kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<16:34, 393kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<13:46, 471kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<10:59, 590kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<07:57, 814kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<05:40, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<05:54, 1.09MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<05:26, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<04:08, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:56, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<2:04:33, 51.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<1:28:26, 71.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<1:02:07, 102kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<43:12, 145kB/s]  .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:33<2:30:23, 41.8kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<1:46:30, 58.9kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<1:14:40, 83.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<52:03, 120kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<39:22, 158kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<28:49, 215kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<20:24, 304kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<14:18, 431kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<12:39, 485kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<10:07, 607kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<07:20, 835kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<05:12, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<05:59, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<05:26, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<04:06, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<02:55, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<1:57:28, 51.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<1:23:25, 72.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<58:34, 102kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<40:43, 146kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<51:46, 115kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<37:01, 160kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<26:00, 227kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<18:08, 324kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<27:16, 215kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<20:06, 292kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<14:15, 410kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<10:01, 580kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<09:12, 630kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<07:29, 773kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<05:29, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:47, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<04:23, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<03:16, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<03:19, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<02:34, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:43, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<02:53, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<02:16, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<02:30, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<02:46, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<02:11, 2.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<01:34, 3.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<2:40:44, 33.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<1:53:26, 48.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<1:19:24, 68.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<55:14, 97.7kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<42:47, 126kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<30:54, 174kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<21:49, 246kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<16:01, 332kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<12:09, 437kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<08:43, 607kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<06:56, 757kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<05:48, 902kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<04:17, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<03:51, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:38, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:44, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<01:58, 2.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<04:22, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<03:59, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<03:00, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<02:56, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<02:57, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:17, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:25, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:36, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:02, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:14, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:27, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<01:56, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<02:09, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<02:23, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<01:51, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<01:20, 3.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<04:48, 992kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<03:58, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:54, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<02:56, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<03:42, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<03:00, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:45, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<02:07, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:15, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:22, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:49, 2.49MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<01:20, 3.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:24<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<03:04, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:20, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<02:26, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<01:51, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<01:21, 3.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<03:35, 1.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<03:17, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:27, 1.76MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:45, 2.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<04:02, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<03:35, 1.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<02:41, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<02:35, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<02:31, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<01:55, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:23, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<03:07, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:55, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:34<02:11, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:34, 2.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<03:58, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<03:30, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<02:35, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:51, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:58, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:47, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<02:05, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<01:30, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<03:35, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<03:06, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<02:20, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<01:40, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<03:32, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<03:07, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<02:21, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<02:16, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<02:15, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:42, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:14, 3.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<03:05, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<02:48, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:06, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<01:29, 2.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<05:28, 669kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:48<04:28, 818kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<03:16, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:52, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:50<02:38, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:58, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<01:24, 2.50MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<04:51, 727kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<03:59, 882kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<02:55, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<02:04, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<05:52, 590kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<04:42, 734kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<03:25, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<02:23, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<09:42, 350kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<07:22, 459kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<05:16, 640kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<04:12, 789kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<03:31, 941kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:36, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:21, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:13, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<01:40, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<01:13, 2.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<01:59, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<01:56, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:29, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:33, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:15, 2.46MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<00:54, 3.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<02:26, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<02:14, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:40, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:14, 2.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:42, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:43, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:19, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:24, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:29, 1.94MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:09, 2.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<00:49, 3.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<05:49, 488kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<04:33, 623kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<03:17, 857kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<02:44, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<02:24, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:47, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:41, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:39, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:19, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:23, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:05, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:11, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:16, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:00, 2.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:06, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:12, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<00:56, 2.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:03, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:10, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<00:55, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:02, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:08, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<00:53, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:00, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:06, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<00:52, 2.62MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:37, 3.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<43:47, 50.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<30:59, 71.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<21:38, 102kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<15:10, 142kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<10:58, 196kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<07:43, 277kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<05:38, 370kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:33<04:18, 484kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<03:04, 671kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<02:27, 823kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<02:03, 974kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:30, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<01:03, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<02:04, 936kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:37<01:47, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<01:19, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<01:13, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:11, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:55, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:57, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:44, 2.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<00:48, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:40, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<00:44, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<00:48, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<00:37, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<00:46, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:35, 2.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:26, 3.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:54, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:54, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<00:43, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:46, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:35, 2.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:38, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:41, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<00:31, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:22, 3.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:58, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:54, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:40, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:55<00:28, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:56<00:53, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:50, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<00:37, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:26, 2.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:54, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:50, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:37, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:36, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:37, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:28, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:20, 3.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:42, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:40, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<00:30, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:30, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:31, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:24, 2.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:17, 3.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:38, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:36, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:06<00:26, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:19, 2.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:32, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:31, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:23, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:24, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:18, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:19, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:21, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:16, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:11, 3.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:26, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:25, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:18, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:18, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:18, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:13, 2.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:09, 3.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<00:26, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<00:23, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:17, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:11, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:24, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:21, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:15, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:10, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:21, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:22<00:18, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:13, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:11, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:10, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:07, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:05, 2.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:08, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:08, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:05, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:04, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:04, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:03, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:02, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:02, 2.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:01, 2.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 3.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:01, 692kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:01, 839kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.14MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 700/400000 [00:00<00:57, 6998.64it/s]  0%|          | 1403/400000 [00:00<00:56, 7007.13it/s]  1%|          | 2126/400000 [00:00<00:56, 7069.70it/s]  1%|          | 2863/400000 [00:00<00:55, 7157.02it/s]  1%|          | 3587/400000 [00:00<00:55, 7179.78it/s]  1%|          | 4300/400000 [00:00<00:55, 7162.45it/s]  1%|          | 4992/400000 [00:00<00:55, 7085.45it/s]  1%|▏         | 5723/400000 [00:00<00:55, 7150.42it/s]  2%|▏         | 6456/400000 [00:00<00:54, 7202.30it/s]  2%|▏         | 7211/400000 [00:01<00:53, 7300.44it/s]  2%|▏         | 7958/400000 [00:01<00:53, 7349.70it/s]  2%|▏         | 8683/400000 [00:01<00:53, 7316.84it/s]  2%|▏         | 9406/400000 [00:01<00:54, 7198.06it/s]  3%|▎         | 10120/400000 [00:01<00:54, 7180.17it/s]  3%|▎         | 10850/400000 [00:01<00:53, 7214.72it/s]  3%|▎         | 11602/400000 [00:01<00:53, 7302.14it/s]  3%|▎         | 12331/400000 [00:01<00:53, 7230.35it/s]  3%|▎         | 13053/400000 [00:01<00:54, 7135.44it/s]  3%|▎         | 13767/400000 [00:01<00:54, 7049.76it/s]  4%|▎         | 14480/400000 [00:02<00:54, 7072.64it/s]  4%|▍         | 15209/400000 [00:02<00:53, 7134.63it/s]  4%|▍         | 15923/400000 [00:02<00:53, 7124.75it/s]  4%|▍         | 16636/400000 [00:02<00:53, 7107.32it/s]  4%|▍         | 17358/400000 [00:02<00:53, 7138.34it/s]  5%|▍         | 18072/400000 [00:02<00:55, 6900.53it/s]  5%|▍         | 18764/400000 [00:02<00:56, 6806.93it/s]  5%|▍         | 19447/400000 [00:02<00:56, 6752.61it/s]  5%|▌         | 20165/400000 [00:02<00:55, 6874.38it/s]  5%|▌         | 20885/400000 [00:02<00:54, 6968.63it/s]  5%|▌         | 21622/400000 [00:03<00:53, 7082.07it/s]  6%|▌         | 22356/400000 [00:03<00:52, 7155.11it/s]  6%|▌         | 23073/400000 [00:03<00:52, 7147.20it/s]  6%|▌         | 23789/400000 [00:03<00:52, 7139.83it/s]  6%|▌         | 24504/400000 [00:03<00:52, 7133.05it/s]  6%|▋         | 25218/400000 [00:03<00:52, 7083.45it/s]  6%|▋         | 25932/400000 [00:03<00:52, 7099.68it/s]  7%|▋         | 26647/400000 [00:03<00:52, 7090.34it/s]  7%|▋         | 27391/400000 [00:03<00:51, 7188.74it/s]  7%|▋         | 28125/400000 [00:03<00:51, 7232.72it/s]  7%|▋         | 28873/400000 [00:04<00:50, 7304.45it/s]  7%|▋         | 29604/400000 [00:04<00:50, 7298.93it/s]  8%|▊         | 30335/400000 [00:04<00:51, 7233.02it/s]  8%|▊         | 31059/400000 [00:04<00:51, 7198.32it/s]  8%|▊         | 31788/400000 [00:04<00:50, 7223.31it/s]  8%|▊         | 32531/400000 [00:04<00:50, 7282.91it/s]  8%|▊         | 33277/400000 [00:04<00:49, 7334.59it/s]  9%|▊         | 34011/400000 [00:04<00:50, 7214.25it/s]  9%|▊         | 34734/400000 [00:04<00:51, 7124.25it/s]  9%|▉         | 35448/400000 [00:04<00:51, 7107.66it/s]  9%|▉         | 36170/400000 [00:05<00:50, 7138.32it/s]  9%|▉         | 36894/400000 [00:05<00:50, 7167.38it/s]  9%|▉         | 37612/400000 [00:05<00:50, 7168.87it/s] 10%|▉         | 38353/400000 [00:05<00:49, 7238.58it/s] 10%|▉         | 39093/400000 [00:05<00:49, 7273.07it/s] 10%|▉         | 39821/400000 [00:05<00:49, 7209.89it/s] 10%|█         | 40543/400000 [00:05<00:50, 7168.62it/s] 10%|█         | 41271/400000 [00:05<00:49, 7200.02it/s] 11%|█         | 42007/400000 [00:05<00:49, 7246.17it/s] 11%|█         | 42764/400000 [00:05<00:48, 7339.45it/s] 11%|█         | 43499/400000 [00:06<00:48, 7328.84it/s] 11%|█         | 44233/400000 [00:06<00:48, 7326.82it/s] 11%|█         | 44966/400000 [00:06<00:49, 7235.88it/s] 11%|█▏        | 45700/400000 [00:06<00:48, 7265.21it/s] 12%|█▏        | 46427/400000 [00:06<00:48, 7261.07it/s] 12%|█▏        | 47156/400000 [00:06<00:48, 7269.72it/s] 12%|█▏        | 47904/400000 [00:06<00:48, 7330.38it/s] 12%|█▏        | 48644/400000 [00:06<00:47, 7349.72it/s] 12%|█▏        | 49380/400000 [00:06<00:48, 7274.29it/s] 13%|█▎        | 50108/400000 [00:06<00:48, 7213.38it/s] 13%|█▎        | 50838/400000 [00:07<00:48, 7236.77it/s] 13%|█▎        | 51562/400000 [00:07<00:48, 7209.32it/s] 13%|█▎        | 52291/400000 [00:07<00:48, 7231.61it/s] 13%|█▎        | 53015/400000 [00:07<00:48, 7185.81it/s] 13%|█▎        | 53734/400000 [00:07<00:48, 7183.66it/s] 14%|█▎        | 54460/400000 [00:07<00:47, 7205.38it/s] 14%|█▍        | 55192/400000 [00:07<00:47, 7238.04it/s] 14%|█▍        | 55916/400000 [00:07<00:47, 7236.49it/s] 14%|█▍        | 56646/400000 [00:07<00:47, 7252.36it/s] 14%|█▍        | 57377/400000 [00:07<00:47, 7269.24it/s] 15%|█▍        | 58106/400000 [00:08<00:46, 7275.32it/s] 15%|█▍        | 58834/400000 [00:08<00:46, 7275.55it/s] 15%|█▍        | 59562/400000 [00:08<00:46, 7271.43it/s] 15%|█▌        | 60290/400000 [00:08<00:46, 7257.51it/s] 15%|█▌        | 61019/400000 [00:08<00:46, 7265.61it/s] 15%|█▌        | 61746/400000 [00:08<00:48, 7039.77it/s] 16%|█▌        | 62452/400000 [00:08<00:48, 7008.49it/s] 16%|█▌        | 63155/400000 [00:08<00:48, 7014.41it/s] 16%|█▌        | 63875/400000 [00:08<00:47, 7067.34it/s] 16%|█▌        | 64603/400000 [00:08<00:47, 7129.43it/s] 16%|█▋        | 65339/400000 [00:09<00:46, 7195.34it/s] 17%|█▋        | 66085/400000 [00:09<00:45, 7272.14it/s] 17%|█▋        | 66815/400000 [00:09<00:45, 7278.45it/s] 17%|█▋        | 67544/400000 [00:09<00:46, 7159.77it/s] 17%|█▋        | 68261/400000 [00:09<00:47, 6991.02it/s] 17%|█▋        | 68962/400000 [00:09<00:47, 6905.63it/s] 17%|█▋        | 69654/400000 [00:09<00:48, 6840.02it/s] 18%|█▊        | 70339/400000 [00:09<00:48, 6779.50it/s] 18%|█▊        | 71018/400000 [00:09<00:48, 6715.01it/s] 18%|█▊        | 71691/400000 [00:10<00:49, 6684.54it/s] 18%|█▊        | 72403/400000 [00:10<00:48, 6809.21it/s] 18%|█▊        | 73102/400000 [00:10<00:47, 6862.25it/s] 18%|█▊        | 73789/400000 [00:10<00:47, 6810.98it/s] 19%|█▊        | 74497/400000 [00:10<00:47, 6886.70it/s] 19%|█▉        | 75187/400000 [00:10<00:47, 6839.62it/s] 19%|█▉        | 75874/400000 [00:10<00:47, 6847.64it/s] 19%|█▉        | 76603/400000 [00:10<00:46, 6974.45it/s] 19%|█▉        | 77302/400000 [00:10<00:47, 6747.61it/s] 20%|█▉        | 78028/400000 [00:10<00:46, 6891.74it/s] 20%|█▉        | 78720/400000 [00:11<00:46, 6886.82it/s] 20%|█▉        | 79411/400000 [00:11<00:46, 6836.60it/s] 20%|██        | 80096/400000 [00:11<00:46, 6837.72it/s] 20%|██        | 80799/400000 [00:11<00:46, 6894.10it/s] 20%|██        | 81490/400000 [00:11<00:46, 6847.30it/s] 21%|██        | 82201/400000 [00:11<00:45, 6921.80it/s] 21%|██        | 82926/400000 [00:11<00:45, 7014.65it/s] 21%|██        | 83641/400000 [00:11<00:44, 7054.28it/s] 21%|██        | 84347/400000 [00:11<00:45, 7005.59it/s] 21%|██▏       | 85049/400000 [00:11<00:45, 6954.94it/s] 21%|██▏       | 85745/400000 [00:12<00:45, 6909.50it/s] 22%|██▏       | 86478/400000 [00:12<00:44, 7029.37it/s] 22%|██▏       | 87200/400000 [00:12<00:44, 7083.81it/s] 22%|██▏       | 87910/400000 [00:12<00:44, 7067.53it/s] 22%|██▏       | 88618/400000 [00:12<00:44, 6968.81it/s] 22%|██▏       | 89316/400000 [00:12<00:44, 6909.46it/s] 23%|██▎       | 90008/400000 [00:12<00:45, 6821.29it/s] 23%|██▎       | 90710/400000 [00:12<00:44, 6878.10it/s] 23%|██▎       | 91431/400000 [00:12<00:44, 6972.97it/s] 23%|██▎       | 92150/400000 [00:12<00:43, 7035.11it/s] 23%|██▎       | 92862/400000 [00:13<00:43, 7059.11it/s] 23%|██▎       | 93600/400000 [00:13<00:42, 7152.26it/s] 24%|██▎       | 94326/400000 [00:13<00:42, 7182.39it/s] 24%|██▍       | 95049/400000 [00:13<00:42, 7195.23it/s] 24%|██▍       | 95769/400000 [00:13<00:42, 7185.34it/s] 24%|██▍       | 96488/400000 [00:13<00:42, 7100.01it/s] 24%|██▍       | 97229/400000 [00:13<00:42, 7188.12it/s] 24%|██▍       | 97969/400000 [00:13<00:41, 7247.57it/s] 25%|██▍       | 98719/400000 [00:13<00:41, 7321.32it/s] 25%|██▍       | 99452/400000 [00:13<00:41, 7231.95it/s] 25%|██▌       | 100176/400000 [00:14<00:41, 7141.47it/s] 25%|██▌       | 100891/400000 [00:14<00:42, 7119.27it/s] 25%|██▌       | 101604/400000 [00:14<00:42, 7057.56it/s] 26%|██▌       | 102311/400000 [00:14<00:43, 6889.76it/s] 26%|██▌       | 103002/400000 [00:14<00:43, 6801.28it/s] 26%|██▌       | 103684/400000 [00:14<00:43, 6792.99it/s] 26%|██▌       | 104365/400000 [00:14<00:43, 6786.13it/s] 26%|██▋       | 105045/400000 [00:14<00:43, 6728.97it/s] 26%|██▋       | 105733/400000 [00:14<00:43, 6773.49it/s] 27%|██▋       | 106454/400000 [00:15<00:42, 6896.92it/s] 27%|██▋       | 107177/400000 [00:15<00:41, 6993.19it/s] 27%|██▋       | 107899/400000 [00:15<00:41, 7057.43it/s] 27%|██▋       | 108606/400000 [00:15<00:42, 6857.69it/s] 27%|██▋       | 109294/400000 [00:15<00:43, 6751.59it/s] 28%|██▊       | 110027/400000 [00:15<00:41, 6914.31it/s] 28%|██▊       | 110758/400000 [00:15<00:41, 7027.07it/s] 28%|██▊       | 111539/400000 [00:15<00:39, 7242.83it/s] 28%|██▊       | 112304/400000 [00:15<00:39, 7359.73it/s] 28%|██▊       | 113043/400000 [00:15<00:39, 7301.07it/s] 28%|██▊       | 113777/400000 [00:16<00:39, 7311.82it/s] 29%|██▊       | 114510/400000 [00:16<00:39, 7282.98it/s] 29%|██▉       | 115240/400000 [00:16<00:39, 7279.65it/s] 29%|██▉       | 115969/400000 [00:16<00:39, 7185.52it/s] 29%|██▉       | 116689/400000 [00:16<00:39, 7097.14it/s] 29%|██▉       | 117414/400000 [00:16<00:39, 7141.63it/s] 30%|██▉       | 118139/400000 [00:16<00:39, 7171.85it/s] 30%|██▉       | 118857/400000 [00:16<00:39, 7155.85it/s] 30%|██▉       | 119573/400000 [00:16<00:39, 7067.98it/s] 30%|███       | 120314/400000 [00:16<00:39, 7165.45it/s] 30%|███       | 121032/400000 [00:17<00:39, 7136.05it/s] 30%|███       | 121747/400000 [00:17<00:40, 6881.31it/s] 31%|███       | 122438/400000 [00:17<00:40, 6807.53it/s] 31%|███       | 123121/400000 [00:17<00:41, 6704.62it/s] 31%|███       | 123796/400000 [00:17<00:41, 6716.16it/s] 31%|███       | 124523/400000 [00:17<00:40, 6872.95it/s] 31%|███▏      | 125253/400000 [00:17<00:39, 6993.52it/s] 31%|███▏      | 125970/400000 [00:17<00:38, 7043.37it/s] 32%|███▏      | 126691/400000 [00:17<00:38, 7091.12it/s] 32%|███▏      | 127428/400000 [00:17<00:38, 7171.78it/s] 32%|███▏      | 128147/400000 [00:18<00:37, 7166.14it/s] 32%|███▏      | 128865/400000 [00:18<00:38, 7015.85it/s] 32%|███▏      | 129568/400000 [00:18<00:38, 6949.27it/s] 33%|███▎      | 130264/400000 [00:18<00:39, 6883.88it/s] 33%|███▎      | 130954/400000 [00:18<00:39, 6759.76it/s] 33%|███▎      | 131635/400000 [00:18<00:39, 6773.88it/s] 33%|███▎      | 132314/400000 [00:18<00:39, 6750.35it/s] 33%|███▎      | 132999/400000 [00:18<00:39, 6779.31it/s] 33%|███▎      | 133678/400000 [00:18<00:39, 6748.32it/s] 34%|███▎      | 134403/400000 [00:18<00:38, 6890.39it/s] 34%|███▍      | 135135/400000 [00:19<00:37, 7012.71it/s] 34%|███▍      | 135892/400000 [00:19<00:36, 7170.44it/s] 34%|███▍      | 136622/400000 [00:19<00:36, 7207.41it/s] 34%|███▍      | 137348/400000 [00:19<00:36, 7221.18it/s] 35%|███▍      | 138071/400000 [00:19<00:36, 7200.02it/s] 35%|███▍      | 138806/400000 [00:19<00:36, 7243.31it/s] 35%|███▍      | 139535/400000 [00:19<00:35, 7255.28it/s] 35%|███▌      | 140261/400000 [00:19<00:36, 7139.94it/s] 35%|███▌      | 141020/400000 [00:19<00:35, 7266.79it/s] 35%|███▌      | 141763/400000 [00:19<00:35, 7313.22it/s] 36%|███▌      | 142510/400000 [00:20<00:34, 7357.41it/s] 36%|███▌      | 143269/400000 [00:20<00:34, 7423.96it/s] 36%|███▌      | 144049/400000 [00:20<00:33, 7532.25it/s] 36%|███▌      | 144804/400000 [00:20<00:34, 7403.43it/s] 36%|███▋      | 145546/400000 [00:20<00:34, 7286.77it/s] 37%|███▋      | 146276/400000 [00:20<00:35, 7206.46it/s] 37%|███▋      | 146998/400000 [00:20<00:35, 7199.18it/s] 37%|███▋      | 147719/400000 [00:20<00:35, 7060.34it/s] 37%|███▋      | 148450/400000 [00:20<00:35, 7131.53it/s] 37%|███▋      | 149174/400000 [00:21<00:35, 7162.02it/s] 37%|███▋      | 149918/400000 [00:21<00:34, 7242.90it/s] 38%|███▊      | 150678/400000 [00:21<00:33, 7345.92it/s] 38%|███▊      | 151424/400000 [00:21<00:33, 7376.62it/s] 38%|███▊      | 152168/400000 [00:21<00:33, 7395.08it/s] 38%|███▊      | 152908/400000 [00:21<00:33, 7305.80it/s] 38%|███▊      | 153640/400000 [00:21<00:33, 7287.42it/s] 39%|███▊      | 154370/400000 [00:21<00:34, 7142.07it/s] 39%|███▉      | 155102/400000 [00:21<00:34, 7191.68it/s] 39%|███▉      | 155829/400000 [00:21<00:33, 7214.70it/s] 39%|███▉      | 156561/400000 [00:22<00:33, 7244.77it/s] 39%|███▉      | 157286/400000 [00:22<00:34, 7092.38it/s] 39%|███▉      | 157997/400000 [00:22<00:35, 6840.40it/s] 40%|███▉      | 158689/400000 [00:22<00:35, 6862.46it/s] 40%|███▉      | 159405/400000 [00:22<00:34, 6948.08it/s] 40%|████      | 160102/400000 [00:22<00:35, 6807.50it/s] 40%|████      | 160807/400000 [00:22<00:34, 6878.42it/s] 40%|████      | 161546/400000 [00:22<00:33, 7022.27it/s] 41%|████      | 162284/400000 [00:22<00:33, 7123.58it/s] 41%|████      | 163042/400000 [00:22<00:32, 7252.56it/s] 41%|████      | 163772/400000 [00:23<00:32, 7264.34it/s] 41%|████      | 164500/400000 [00:23<00:32, 7267.85it/s] 41%|████▏     | 165228/400000 [00:23<00:32, 7261.03it/s] 41%|████▏     | 165955/400000 [00:23<00:32, 7263.52it/s] 42%|████▏     | 166682/400000 [00:23<00:33, 7041.17it/s] 42%|████▏     | 167393/400000 [00:23<00:32, 7059.99it/s] 42%|████▏     | 168139/400000 [00:23<00:32, 7173.76it/s] 42%|████▏     | 168858/400000 [00:23<00:32, 7094.29it/s] 42%|████▏     | 169569/400000 [00:23<00:33, 6969.21it/s] 43%|████▎     | 170268/400000 [00:23<00:33, 6932.50it/s] 43%|████▎     | 170994/400000 [00:24<00:32, 7027.44it/s] 43%|████▎     | 171705/400000 [00:24<00:32, 7051.71it/s] 43%|████▎     | 172447/400000 [00:24<00:31, 7155.26it/s] 43%|████▎     | 173168/400000 [00:24<00:31, 7170.22it/s] 43%|████▎     | 173900/400000 [00:24<00:31, 7213.95it/s] 44%|████▎     | 174638/400000 [00:24<00:31, 7260.43it/s] 44%|████▍     | 175376/400000 [00:24<00:30, 7293.17it/s] 44%|████▍     | 176106/400000 [00:24<00:30, 7285.37it/s] 44%|████▍     | 176841/400000 [00:24<00:30, 7301.92it/s] 44%|████▍     | 177573/400000 [00:24<00:30, 7306.28it/s] 45%|████▍     | 178304/400000 [00:25<00:30, 7300.04it/s] 45%|████▍     | 179035/400000 [00:25<00:30, 7287.91it/s] 45%|████▍     | 179767/400000 [00:25<00:30, 7296.61it/s] 45%|████▌     | 180525/400000 [00:25<00:29, 7379.25it/s] 45%|████▌     | 181264/400000 [00:25<00:29, 7347.86it/s] 45%|████▌     | 181999/400000 [00:25<00:29, 7307.06it/s] 46%|████▌     | 182751/400000 [00:25<00:29, 7366.16it/s] 46%|████▌     | 183488/400000 [00:25<00:29, 7364.42it/s] 46%|████▌     | 184225/400000 [00:25<00:29, 7309.76it/s] 46%|████▌     | 184958/400000 [00:25<00:29, 7314.97it/s] 46%|████▋     | 185690/400000 [00:26<00:29, 7293.76it/s] 47%|████▋     | 186446/400000 [00:26<00:28, 7371.24it/s] 47%|████▋     | 187191/400000 [00:26<00:28, 7394.61it/s] 47%|████▋     | 187931/400000 [00:26<00:29, 7288.01it/s] 47%|████▋     | 188661/400000 [00:26<00:29, 7249.40it/s] 47%|████▋     | 189387/400000 [00:26<00:29, 7178.81it/s] 48%|████▊     | 190141/400000 [00:26<00:28, 7281.71it/s] 48%|████▊     | 190870/400000 [00:26<00:29, 7112.24it/s] 48%|████▊     | 191583/400000 [00:26<00:29, 7058.52it/s] 48%|████▊     | 192314/400000 [00:27<00:29, 7131.03it/s] 48%|████▊     | 193046/400000 [00:27<00:28, 7185.96it/s] 48%|████▊     | 193772/400000 [00:27<00:28, 7207.82it/s] 49%|████▊     | 194499/400000 [00:27<00:28, 7223.02it/s] 49%|████▉     | 195230/400000 [00:27<00:28, 7246.93it/s] 49%|████▉     | 195977/400000 [00:27<00:27, 7312.24it/s] 49%|████▉     | 196709/400000 [00:27<00:28, 7252.28it/s] 49%|████▉     | 197452/400000 [00:27<00:27, 7304.32it/s] 50%|████▉     | 198207/400000 [00:27<00:27, 7374.02it/s] 50%|████▉     | 198969/400000 [00:27<00:27, 7445.18it/s] 50%|████▉     | 199714/400000 [00:28<00:27, 7405.25it/s] 50%|█████     | 200455/400000 [00:28<00:27, 7337.13it/s] 50%|█████     | 201192/400000 [00:28<00:27, 7345.28it/s] 50%|█████     | 201945/400000 [00:28<00:26, 7399.59it/s] 51%|█████     | 202686/400000 [00:28<00:26, 7399.30it/s] 51%|█████     | 203427/400000 [00:28<00:26, 7378.92it/s] 51%|█████     | 204171/400000 [00:28<00:26, 7394.74it/s] 51%|█████     | 204911/400000 [00:28<00:26, 7283.38it/s] 51%|█████▏    | 205640/400000 [00:28<00:27, 7089.57it/s] 52%|█████▏    | 206351/400000 [00:28<00:27, 7012.24it/s] 52%|█████▏    | 207054/400000 [00:29<00:27, 7003.87it/s] 52%|█████▏    | 207756/400000 [00:29<00:27, 6976.18it/s] 52%|█████▏    | 208484/400000 [00:29<00:27, 7062.25it/s] 52%|█████▏    | 209191/400000 [00:29<00:27, 7024.20it/s] 52%|█████▏    | 209894/400000 [00:29<00:27, 6992.10it/s] 53%|█████▎    | 210629/400000 [00:29<00:26, 7094.49it/s] 53%|█████▎    | 211372/400000 [00:29<00:26, 7190.21it/s] 53%|█████▎    | 212128/400000 [00:29<00:25, 7295.33it/s] 53%|█████▎    | 212887/400000 [00:29<00:25, 7379.13it/s] 53%|█████▎    | 213649/400000 [00:29<00:25, 7446.82it/s] 54%|█████▎    | 214395/400000 [00:30<00:24, 7439.16it/s] 54%|█████▍    | 215140/400000 [00:30<00:24, 7411.46it/s] 54%|█████▍    | 215882/400000 [00:30<00:24, 7379.84it/s] 54%|█████▍    | 216628/400000 [00:30<00:24, 7402.92it/s] 54%|█████▍    | 217374/400000 [00:30<00:24, 7418.91it/s] 55%|█████▍    | 218117/400000 [00:30<00:24, 7399.54it/s] 55%|█████▍    | 218858/400000 [00:30<00:24, 7361.38it/s] 55%|█████▍    | 219595/400000 [00:30<00:25, 7168.64it/s] 55%|█████▌    | 220314/400000 [00:30<00:25, 6961.97it/s] 55%|█████▌    | 221013/400000 [00:30<00:25, 6900.33it/s] 55%|█████▌    | 221705/400000 [00:31<00:25, 6862.92it/s] 56%|█████▌    | 222393/400000 [00:31<00:25, 6864.08it/s] 56%|█████▌    | 223091/400000 [00:31<00:25, 6896.95it/s] 56%|█████▌    | 223782/400000 [00:31<00:25, 6874.85it/s] 56%|█████▌    | 224531/400000 [00:31<00:24, 7046.39it/s] 56%|█████▋    | 225278/400000 [00:31<00:24, 7167.26it/s] 57%|█████▋    | 226009/400000 [00:31<00:24, 7208.59it/s] 57%|█████▋    | 226751/400000 [00:31<00:23, 7267.59it/s] 57%|█████▋    | 227479/400000 [00:31<00:23, 7248.71it/s] 57%|█████▋    | 228215/400000 [00:31<00:23, 7279.45it/s] 57%|█████▋    | 228954/400000 [00:32<00:23, 7309.55it/s] 57%|█████▋    | 229686/400000 [00:32<00:23, 7295.15it/s] 58%|█████▊    | 230416/400000 [00:32<00:23, 7275.64it/s] 58%|█████▊    | 231144/400000 [00:32<00:23, 7191.14it/s] 58%|█████▊    | 231864/400000 [00:32<00:24, 6981.77it/s] 58%|█████▊    | 232564/400000 [00:32<00:24, 6925.89it/s] 58%|█████▊    | 233258/400000 [00:32<00:24, 6871.48it/s] 58%|█████▊    | 233947/400000 [00:32<00:24, 6876.18it/s] 59%|█████▊    | 234652/400000 [00:32<00:23, 6926.87it/s] 59%|█████▉    | 235346/400000 [00:33<00:23, 6883.10it/s] 59%|█████▉    | 236045/400000 [00:33<00:23, 6914.27it/s] 59%|█████▉    | 236737/400000 [00:33<00:23, 6876.28it/s] 59%|█████▉    | 237438/400000 [00:33<00:23, 6913.20it/s] 60%|█████▉    | 238168/400000 [00:33<00:23, 7023.07it/s] 60%|█████▉    | 238877/400000 [00:33<00:22, 7042.47it/s] 60%|█████▉    | 239611/400000 [00:33<00:22, 7127.28it/s] 60%|██████    | 240352/400000 [00:33<00:22, 7208.84it/s] 60%|██████    | 241089/400000 [00:33<00:21, 7255.41it/s] 60%|██████    | 241834/400000 [00:33<00:21, 7311.86it/s] 61%|██████    | 242567/400000 [00:34<00:21, 7314.96it/s] 61%|██████    | 243299/400000 [00:34<00:21, 7294.60it/s] 61%|██████    | 244035/400000 [00:34<00:21, 7310.87it/s] 61%|██████    | 244767/400000 [00:34<00:21, 7313.25it/s] 61%|██████▏   | 245499/400000 [00:34<00:21, 7298.36it/s] 62%|██████▏   | 246229/400000 [00:34<00:21, 7272.73it/s] 62%|██████▏   | 246987/400000 [00:34<00:20, 7361.48it/s] 62%|██████▏   | 247728/400000 [00:34<00:20, 7374.80it/s] 62%|██████▏   | 248466/400000 [00:34<00:20, 7365.73it/s] 62%|██████▏   | 249213/400000 [00:34<00:20, 7393.91it/s] 62%|██████▏   | 249961/400000 [00:35<00:20, 7418.00it/s] 63%|██████▎   | 250715/400000 [00:35<00:20, 7452.13it/s] 63%|██████▎   | 251469/400000 [00:35<00:19, 7477.60it/s] 63%|██████▎   | 252217/400000 [00:35<00:19, 7405.34it/s] 63%|██████▎   | 252962/400000 [00:35<00:19, 7417.51it/s] 63%|██████▎   | 253704/400000 [00:35<00:19, 7382.76it/s] 64%|██████▎   | 254443/400000 [00:35<00:19, 7372.19it/s] 64%|██████▍   | 255181/400000 [00:35<00:19, 7310.46it/s] 64%|██████▍   | 255923/400000 [00:35<00:19, 7342.59it/s] 64%|██████▍   | 256658/400000 [00:35<00:19, 7342.91it/s] 64%|██████▍   | 257408/400000 [00:36<00:19, 7387.57it/s] 65%|██████▍   | 258147/400000 [00:36<00:19, 7375.39it/s] 65%|██████▍   | 258894/400000 [00:36<00:19, 7402.72it/s] 65%|██████▍   | 259635/400000 [00:36<00:19, 7362.80it/s] 65%|██████▌   | 260380/400000 [00:36<00:18, 7387.55it/s] 65%|██████▌   | 261119/400000 [00:36<00:18, 7357.82it/s] 65%|██████▌   | 261860/400000 [00:36<00:18, 7372.98it/s] 66%|██████▌   | 262598/400000 [00:36<00:18, 7354.02it/s] 66%|██████▌   | 263334/400000 [00:36<00:18, 7351.12it/s] 66%|██████▌   | 264070/400000 [00:36<00:18, 7267.69it/s] 66%|██████▌   | 264798/400000 [00:37<00:18, 7244.59it/s] 66%|██████▋   | 265526/400000 [00:37<00:18, 7253.22it/s] 67%|██████▋   | 266264/400000 [00:37<00:18, 7290.66it/s] 67%|██████▋   | 266994/400000 [00:37<00:18, 7279.58it/s] 67%|██████▋   | 267723/400000 [00:37<00:18, 7173.59it/s] 67%|██████▋   | 268441/400000 [00:37<00:18, 7158.45it/s] 67%|██████▋   | 269158/400000 [00:37<00:18, 7121.62it/s] 67%|██████▋   | 269888/400000 [00:37<00:18, 7171.74it/s] 68%|██████▊   | 270611/400000 [00:37<00:17, 7188.36it/s] 68%|██████▊   | 271367/400000 [00:37<00:17, 7294.45it/s] 68%|██████▊   | 272123/400000 [00:38<00:17, 7370.28it/s] 68%|██████▊   | 272863/400000 [00:38<00:17, 7376.53it/s] 68%|██████▊   | 273620/400000 [00:38<00:17, 7431.52it/s] 69%|██████▊   | 274364/400000 [00:38<00:16, 7424.08it/s] 69%|██████▉   | 275107/400000 [00:38<00:16, 7361.83it/s] 69%|██████▉   | 275844/400000 [00:38<00:16, 7364.25it/s] 69%|██████▉   | 276581/400000 [00:38<00:17, 7200.78it/s] 69%|██████▉   | 277303/400000 [00:38<00:17, 7163.12it/s] 70%|██████▉   | 278021/400000 [00:38<00:17, 6975.91it/s] 70%|██████▉   | 278721/400000 [00:38<00:17, 6950.39it/s] 70%|██████▉   | 279418/400000 [00:39<00:17, 6884.95it/s] 70%|███████   | 280108/400000 [00:39<00:17, 6796.35it/s] 70%|███████   | 280838/400000 [00:39<00:17, 6937.63it/s] 70%|███████   | 281534/400000 [00:39<00:17, 6940.12it/s] 71%|███████   | 282286/400000 [00:39<00:16, 7101.77it/s] 71%|███████   | 283025/400000 [00:39<00:16, 7185.74it/s] 71%|███████   | 283764/400000 [00:39<00:16, 7245.31it/s] 71%|███████   | 284490/400000 [00:39<00:16, 7157.37it/s] 71%|███████▏  | 285207/400000 [00:39<00:16, 7067.05it/s] 71%|███████▏  | 285930/400000 [00:39<00:16, 7114.85it/s] 72%|███████▏  | 286670/400000 [00:40<00:15, 7197.06it/s] 72%|███████▏  | 287393/400000 [00:40<00:15, 7206.26it/s] 72%|███████▏  | 288125/400000 [00:40<00:15, 7239.53it/s] 72%|███████▏  | 288850/400000 [00:40<00:15, 7202.39it/s] 72%|███████▏  | 289571/400000 [00:40<00:15, 7186.19it/s] 73%|███████▎  | 290302/400000 [00:40<00:15, 7220.62it/s] 73%|███████▎  | 291025/400000 [00:40<00:15, 7149.41it/s] 73%|███████▎  | 291741/400000 [00:40<00:15, 7108.82it/s] 73%|███████▎  | 292453/400000 [00:40<00:15, 6958.25it/s] 73%|███████▎  | 293193/400000 [00:40<00:15, 7081.93it/s] 73%|███████▎  | 293918/400000 [00:41<00:14, 7129.66it/s] 74%|███████▎  | 294673/400000 [00:41<00:14, 7248.19it/s] 74%|███████▍  | 295425/400000 [00:41<00:14, 7324.92it/s] 74%|███████▍  | 296159/400000 [00:41<00:14, 7329.03it/s] 74%|███████▍  | 296893/400000 [00:41<00:14, 7331.41it/s] 74%|███████▍  | 297627/400000 [00:41<00:14, 7221.86it/s] 75%|███████▍  | 298350/400000 [00:41<00:14, 7195.97it/s] 75%|███████▍  | 299071/400000 [00:41<00:14, 7153.97it/s] 75%|███████▍  | 299787/400000 [00:41<00:14, 7141.39it/s] 75%|███████▌  | 300516/400000 [00:42<00:13, 7184.20it/s] 75%|███████▌  | 301251/400000 [00:42<00:13, 7232.63it/s] 75%|███████▌  | 301975/400000 [00:42<00:13, 7207.42it/s] 76%|███████▌  | 302702/400000 [00:42<00:13, 7225.67it/s] 76%|███████▌  | 303425/400000 [00:42<00:13, 7216.72it/s] 76%|███████▌  | 304147/400000 [00:42<00:13, 7121.15it/s] 76%|███████▌  | 304878/400000 [00:42<00:13, 7175.47it/s] 76%|███████▋  | 305628/400000 [00:42<00:12, 7267.97it/s] 77%|███████▋  | 306377/400000 [00:42<00:12, 7330.65it/s] 77%|███████▋  | 307126/400000 [00:42<00:12, 7375.73it/s] 77%|███████▋  | 307877/400000 [00:43<00:12, 7413.92it/s] 77%|███████▋  | 308619/400000 [00:43<00:12, 7411.67it/s] 77%|███████▋  | 309361/400000 [00:43<00:12, 7378.75it/s] 78%|███████▊  | 310106/400000 [00:43<00:12, 7399.47it/s] 78%|███████▊  | 310847/400000 [00:43<00:12, 7361.86it/s] 78%|███████▊  | 311584/400000 [00:43<00:12, 7312.05it/s] 78%|███████▊  | 312316/400000 [00:43<00:12, 7275.12it/s] 78%|███████▊  | 313051/400000 [00:43<00:11, 7296.74it/s] 78%|███████▊  | 313781/400000 [00:43<00:11, 7255.29it/s] 79%|███████▊  | 314507/400000 [00:43<00:11, 7214.57it/s] 79%|███████▉  | 315231/400000 [00:44<00:11, 7220.78it/s] 79%|███████▉  | 315992/400000 [00:44<00:11, 7332.68it/s] 79%|███████▉  | 316742/400000 [00:44<00:11, 7380.00it/s] 79%|███████▉  | 317481/400000 [00:44<00:11, 7370.16it/s] 80%|███████▉  | 318219/400000 [00:44<00:11, 7368.25it/s] 80%|███████▉  | 318957/400000 [00:44<00:11, 7258.32it/s] 80%|███████▉  | 319684/400000 [00:44<00:11, 7243.08it/s] 80%|████████  | 320412/400000 [00:44<00:10, 7252.22it/s] 80%|████████  | 321138/400000 [00:44<00:10, 7212.45it/s] 80%|████████  | 321861/400000 [00:44<00:10, 7215.58it/s] 81%|████████  | 322600/400000 [00:45<00:10, 7265.83it/s] 81%|████████  | 323333/400000 [00:45<00:10, 7284.08it/s] 81%|████████  | 324063/400000 [00:45<00:10, 7286.55it/s] 81%|████████  | 324794/400000 [00:45<00:10, 7292.85it/s] 81%|████████▏ | 325524/400000 [00:45<00:10, 7245.89it/s] 82%|████████▏ | 326249/400000 [00:45<00:10, 7243.69it/s] 82%|████████▏ | 326979/400000 [00:45<00:10, 7258.08it/s] 82%|████████▏ | 327707/400000 [00:45<00:09, 7263.84it/s] 82%|████████▏ | 328438/400000 [00:45<00:09, 7275.21it/s] 82%|████████▏ | 329166/400000 [00:45<00:09, 7190.95it/s] 82%|████████▏ | 329886/400000 [00:46<00:09, 7049.57it/s] 83%|████████▎ | 330611/400000 [00:46<00:09, 7106.46it/s] 83%|████████▎ | 331345/400000 [00:46<00:09, 7174.62it/s] 83%|████████▎ | 332064/400000 [00:46<00:09, 7063.72it/s] 83%|████████▎ | 332772/400000 [00:46<00:09, 7033.21it/s] 83%|████████▎ | 333478/400000 [00:46<00:09, 7039.94it/s] 84%|████████▎ | 334202/400000 [00:46<00:09, 7096.36it/s] 84%|████████▎ | 334922/400000 [00:46<00:09, 7125.27it/s] 84%|████████▍ | 335635/400000 [00:46<00:09, 6993.33it/s] 84%|████████▍ | 336336/400000 [00:46<00:09, 6919.36it/s] 84%|████████▍ | 337029/400000 [00:47<00:09, 6907.01it/s] 84%|████████▍ | 337751/400000 [00:47<00:08, 6995.88it/s] 85%|████████▍ | 338497/400000 [00:47<00:08, 7126.39it/s] 85%|████████▍ | 339230/400000 [00:47<00:08, 7186.10it/s] 85%|████████▍ | 339950/400000 [00:47<00:08, 7186.96it/s] 85%|████████▌ | 340679/400000 [00:47<00:08, 7216.23it/s] 85%|████████▌ | 341406/400000 [00:47<00:08, 7229.84it/s] 86%|████████▌ | 342130/400000 [00:47<00:08, 7119.08it/s] 86%|████████▌ | 342855/400000 [00:47<00:07, 7155.26it/s] 86%|████████▌ | 343582/400000 [00:47<00:07, 7187.44it/s] 86%|████████▌ | 344312/400000 [00:48<00:07, 7220.66it/s] 86%|████████▋ | 345037/400000 [00:48<00:07, 7228.03it/s] 86%|████████▋ | 345772/400000 [00:48<00:07, 7263.71it/s] 87%|████████▋ | 346533/400000 [00:48<00:07, 7363.47it/s] 87%|████████▋ | 347278/400000 [00:48<00:07, 7388.85it/s] 87%|████████▋ | 348018/400000 [00:48<00:07, 7383.61it/s] 87%|████████▋ | 348779/400000 [00:48<00:06, 7448.61it/s] 87%|████████▋ | 349546/400000 [00:48<00:06, 7512.00it/s] 88%|████████▊ | 350305/400000 [00:48<00:06, 7533.42it/s] 88%|████████▊ | 351059/400000 [00:48<00:06, 7465.86it/s] 88%|████████▊ | 351806/400000 [00:49<00:06, 7448.32it/s] 88%|████████▊ | 352552/400000 [00:49<00:06, 7443.08it/s] 88%|████████▊ | 353297/400000 [00:49<00:06, 7301.80it/s] 89%|████████▊ | 354028/400000 [00:49<00:06, 6965.29it/s] 89%|████████▊ | 354740/400000 [00:49<00:06, 7010.07it/s] 89%|████████▉ | 355495/400000 [00:49<00:06, 7162.22it/s] 89%|████████▉ | 356243/400000 [00:49<00:06, 7254.36it/s] 89%|████████▉ | 356976/400000 [00:49<00:05, 7276.34it/s] 89%|████████▉ | 357713/400000 [00:49<00:05, 7300.92it/s] 90%|████████▉ | 358445/400000 [00:50<00:05, 7297.66it/s] 90%|████████▉ | 359191/400000 [00:50<00:05, 7345.22it/s] 90%|████████▉ | 359948/400000 [00:50<00:05, 7411.01it/s] 90%|█████████ | 360690/400000 [00:50<00:05, 7284.87it/s] 90%|█████████ | 361420/400000 [00:50<00:05, 7221.86it/s] 91%|█████████ | 362143/400000 [00:50<00:05, 6886.25it/s] 91%|█████████ | 362869/400000 [00:50<00:05, 6994.08it/s] 91%|█████████ | 363614/400000 [00:50<00:05, 7124.67it/s] 91%|█████████ | 364348/400000 [00:50<00:04, 7186.91it/s] 91%|█████████▏| 365077/400000 [00:50<00:04, 7216.28it/s] 91%|█████████▏| 365804/400000 [00:51<00:04, 7230.80it/s] 92%|█████████▏| 366529/400000 [00:51<00:04, 7096.95it/s] 92%|█████████▏| 367240/400000 [00:51<00:04, 7020.06it/s] 92%|█████████▏| 367944/400000 [00:51<00:04, 6980.28it/s] 92%|█████████▏| 368667/400000 [00:51<00:04, 7051.04it/s] 92%|█████████▏| 369403/400000 [00:51<00:04, 7139.64it/s] 93%|█████████▎| 370159/400000 [00:51<00:04, 7259.36it/s] 93%|█████████▎| 370916/400000 [00:51<00:03, 7347.27it/s] 93%|█████████▎| 371652/400000 [00:51<00:03, 7328.88it/s] 93%|█████████▎| 372390/400000 [00:51<00:03, 7342.56it/s] 93%|█████████▎| 373136/400000 [00:52<00:03, 7376.58it/s] 93%|█████████▎| 373875/400000 [00:52<00:03, 7297.65it/s] 94%|█████████▎| 374628/400000 [00:52<00:03, 7362.35it/s] 94%|█████████▍| 375365/400000 [00:52<00:03, 7299.04it/s] 94%|█████████▍| 376100/400000 [00:52<00:03, 7314.06it/s] 94%|█████████▍| 376832/400000 [00:52<00:03, 7269.86it/s] 94%|█████████▍| 377571/400000 [00:52<00:03, 7304.34it/s] 95%|█████████▍| 378311/400000 [00:52<00:02, 7330.46it/s] 95%|█████████▍| 379045/400000 [00:52<00:02, 7266.51it/s] 95%|█████████▍| 379795/400000 [00:52<00:02, 7333.12it/s] 95%|█████████▌| 380549/400000 [00:53<00:02, 7393.67it/s] 95%|█████████▌| 381312/400000 [00:53<00:02, 7462.81it/s] 96%|█████████▌| 382065/400000 [00:53<00:02, 7480.46it/s] 96%|█████████▌| 382815/400000 [00:53<00:02, 7484.70it/s] 96%|█████████▌| 383564/400000 [00:53<00:02, 7453.80it/s] 96%|█████████▌| 384310/400000 [00:53<00:02, 7389.55it/s] 96%|█████████▋| 385050/400000 [00:53<00:02, 7300.91it/s] 96%|█████████▋| 385781/400000 [00:53<00:01, 7259.00it/s] 97%|█████████▋| 386510/400000 [00:53<00:01, 7265.95it/s] 97%|█████████▋| 387249/400000 [00:53<00:01, 7302.69it/s] 97%|█████████▋| 387980/400000 [00:54<00:01, 7254.81it/s] 97%|█████████▋| 388709/400000 [00:54<00:01, 7262.62it/s] 97%|█████████▋| 389446/400000 [00:54<00:01, 7291.85it/s] 98%|█████████▊| 390181/400000 [00:54<00:01, 7307.07it/s] 98%|█████████▊| 390912/400000 [00:54<00:01, 7180.25it/s] 98%|█████████▊| 391631/400000 [00:54<00:01, 7179.53it/s] 98%|█████████▊| 392380/400000 [00:54<00:01, 7268.53it/s] 98%|█████████▊| 393131/400000 [00:54<00:00, 7338.30it/s] 98%|█████████▊| 393905/400000 [00:54<00:00, 7451.25it/s] 99%|█████████▊| 394673/400000 [00:54<00:00, 7516.18it/s] 99%|█████████▉| 395426/400000 [00:55<00:00, 7480.62it/s] 99%|█████████▉| 396175/400000 [00:55<00:00, 7483.16it/s] 99%|█████████▉| 396930/400000 [00:55<00:00, 7502.72it/s] 99%|█████████▉| 397681/400000 [00:55<00:00, 7466.60it/s]100%|█████████▉| 398428/400000 [00:55<00:00, 7385.11it/s]100%|█████████▉| 399192/400000 [00:55<00:00, 7459.20it/s]100%|█████████▉| 399949/400000 [00:55<00:00, 7490.45it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7182.61it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f041366b940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011220440758826566 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011042184255593597 	 Accuracy: 65

  model saves at 65% accuracy 

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
2020-05-15 09:29:16.264684: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 09:29:16.268936: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-15 09:29:16.269095: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555962710800 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 09:29:16.269218: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f03c004a0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5900 - accuracy: 0.5050
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5388 - accuracy: 0.5083 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5777 - accuracy: 0.5058
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5491 - accuracy: 0.5077
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5374 - accuracy: 0.5084
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5555 - accuracy: 0.5073
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5946 - accuracy: 0.5047
11000/25000 [============>.................] - ETA: 4s - loss: 7.6039 - accuracy: 0.5041
12000/25000 [=============>................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6502 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 3s - loss: 7.6554 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6570 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6470 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6406 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6429 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 10s 400us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f037867d6a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f03bca7f6a0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1244 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.0491 - val_crf_viterbi_accuracy: 0.6800

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f342c0f8fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 10:12:59.496077
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 10:12:59.499490
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 10:12:59.502563
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 10:12:59.505449
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f34381104a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356225.2500
Epoch 2/10

1/1 [==============================] - 0s 95ms/step - loss: 268541.6875
Epoch 3/10

1/1 [==============================] - 0s 85ms/step - loss: 168358.7344
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 94355.9297
Epoch 5/10

1/1 [==============================] - 0s 86ms/step - loss: 48834.8945
Epoch 6/10

1/1 [==============================] - 0s 92ms/step - loss: 25281.4570
Epoch 7/10

1/1 [==============================] - 0s 94ms/step - loss: 14458.2715
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 9195.4297
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 6363.4805
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 4681.7349

  #### Inference Need return ypred, ytrue ######################### 
[[ -2.2982068    0.38398057  -0.09399366  -2.315547     0.9918759
    1.0359734   -0.40141124   0.1537206    1.565243     0.32598132
   -1.4718082   -1.0500556   -0.6383602   -1.2444824    0.12120238
    1.0878266    0.0542683    0.7029013   -0.3785937   -0.0937099
    0.6911741   -0.3697184    2.341723     0.43886566   0.5301143
   -0.7859218    0.9102817    0.4750352    2.01365      1.3046014
   -0.93961334   1.065253     0.7754673   -1.71255     -0.629542
    1.7600447   -1.3420044    2.1573994   -0.5133269   -1.074249
   -1.9776305   -0.09366849  -0.9432584    0.6775501    1.4391794
    0.3674532   -0.12251794   2.307684     1.4811363   -1.4769206
    0.47969797  -0.895793     1.0627851    1.1618423    0.12318899
   -2.742876    -0.01437697   0.18161893  -1.353169     1.2790174
   -0.03712834  10.957409     9.231239     9.785879    10.038092
    9.753619     9.952619    11.223034    11.513872    10.518134
   10.248886    10.017431    12.097985    12.154442    11.061382
    9.631786    11.522548    10.788873    11.875737     8.414336
   10.643561     9.165368     9.8115225    8.665368     7.916102
   11.8184      10.648987    10.882277    10.253981     8.191944
   11.0455675   10.671712     7.942522    11.577808    10.519469
   11.579642    10.775653    10.807672    12.091111    10.325212
    9.572813     9.897804     8.370127    10.78795      8.537092
    9.893645     9.844934    10.116977    10.549904     7.5658493
   10.238008     9.146428    10.757559    10.168198     9.556617
    9.963857    10.983966    10.338618    10.040273    10.425478
    0.63192225   0.5841758   -1.6451182    1.0509467   -0.78178114
   -0.24877572  -2.6175659   -0.8643218    1.5382122   -1.0622591
   -0.09558636   1.8615035   -0.28003272  -0.719691    -0.41103634
    0.94046307   0.49421978  -0.62764627   0.19055837   0.9569654
   -0.63549274   2.6374607    0.33463657  -0.72822744  -0.4818481
    0.21952882   0.7322021   -0.24668586   1.2067597    0.26556846
   -0.04016215   0.4755132    1.1655364   -0.50966144   0.7898619
   -0.3839555    1.6342355    1.094409     1.1542133    1.194483
    0.2634386   -1.1545019   -0.3524835    1.983635     0.67150325
   -0.41749823   0.31153977  -0.5384763    1.3395936   -1.0609701
    0.77573955  -1.8966081    0.90859723  -0.60575086  -0.12782823
    0.03779697   1.1314673    1.2936195    0.2260249   -2.7978895
    0.3668031    1.1548861    0.8123778    0.90762657   0.2628224
    0.90128726   1.1126289    0.10087377   0.15245527   1.5036923
    0.6070774    1.9496294    0.9945025    0.12057829   1.2969034
    0.90824854   2.0617414    2.6781588    1.5056632    0.4899683
    1.337122     0.96219826   2.9662986    0.7309649    1.5006368
    0.7012456    1.2496285    2.978012     0.2420243    0.91050637
    0.48434627   3.0473442    1.3939863    0.16138023   0.23082674
    1.6782966    1.2510314    0.8731238    0.39207006   0.31377286
    2.7917824    1.9988577    0.90823305   0.11095756   1.2356349
    0.26537573   0.24996793   2.406166     1.1567519    0.1595521
    1.6796948    0.73636514   0.20048344   1.2594928    1.9893484
    1.7329987    0.24360442   0.09034061   0.26785696   2.6492622
    0.10209656  10.198257     9.268394    11.12305      9.780301
    9.852831     9.129334     9.535721     9.717933    10.793949
   10.330125     9.632368    10.292543    10.18782     11.070438
    9.210509    10.863459    11.194398     8.42679     11.207187
   12.459281     9.680417     8.9705515   10.833628    11.030882
    8.749966     9.331598    10.27488      8.849186    11.450607
   10.977394    10.721034    12.244099    10.8991      11.561406
    8.425143     9.353846    10.341074    11.470577    10.267398
   10.530524    11.19987      8.400424    11.287962    10.157111
   10.721528     9.609192     9.34378     13.700459    12.933129
   12.626692    10.231358     8.411196     8.522744     9.621103
    9.882455    11.458809    11.639282    10.142543     9.918085
    3.1699939    0.2636975    2.0609713    0.36430728   0.16196585
    0.42895758   0.87795264   1.5739572    1.2434044    2.2901945
    0.4206227    1.1682692    0.51557755   0.20245379   0.6519253
    0.3723362    0.86406016   2.0040488    0.12677878   0.1005314
    0.5851699    0.5339965    0.95086426   0.48640573   0.37099624
    0.99438184   0.5364715    1.2244384    2.4636936    0.09597081
    0.9433756    1.1896584    0.7843758    0.73357177   2.8409104
    0.5393237    0.32936049   0.14049208   2.3015413    2.4229612
    0.6578291    0.6518928    0.23406696   2.3277552    0.2493273
    0.23529899   0.50038517   0.655791     0.36298323   0.3837304
    1.9888129    1.9378103    0.7787204    0.13881862   1.6494379
    1.9992427    2.818399     0.39820755   3.8034234    1.2118874
   -8.684961     9.871046   -11.713899  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 10:13:08.927443
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.3466
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 10:13:08.931077
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8375.13
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 10:13:08.934113
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.9924
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 10:13:08.937583
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -749.052
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139861714331520
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139859185119808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139859185120312
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139859185120816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139859185121320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139859185121824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3417eef358> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.439405
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.401153
grad_step = 000002, loss = 0.376952
grad_step = 000003, loss = 0.352968
grad_step = 000004, loss = 0.329162
grad_step = 000005, loss = 0.310730
grad_step = 000006, loss = 0.304936
grad_step = 000007, loss = 0.297737
grad_step = 000008, loss = 0.280694
grad_step = 000009, loss = 0.261975
grad_step = 000010, loss = 0.247861
grad_step = 000011, loss = 0.238129
grad_step = 000012, loss = 0.230408
grad_step = 000013, loss = 0.222813
grad_step = 000014, loss = 0.214792
grad_step = 000015, loss = 0.206550
grad_step = 000016, loss = 0.198210
grad_step = 000017, loss = 0.190075
grad_step = 000018, loss = 0.181597
grad_step = 000019, loss = 0.173301
grad_step = 000020, loss = 0.165804
grad_step = 000021, loss = 0.158675
grad_step = 000022, loss = 0.151708
grad_step = 000023, loss = 0.144876
grad_step = 000024, loss = 0.138001
grad_step = 000025, loss = 0.131031
grad_step = 000026, loss = 0.124196
grad_step = 000027, loss = 0.117872
grad_step = 000028, loss = 0.112079
grad_step = 000029, loss = 0.106199
grad_step = 000030, loss = 0.100331
grad_step = 000031, loss = 0.094881
grad_step = 000032, loss = 0.089669
grad_step = 000033, loss = 0.084478
grad_step = 000034, loss = 0.079499
grad_step = 000035, loss = 0.074786
grad_step = 000036, loss = 0.070176
grad_step = 000037, loss = 0.065540
grad_step = 000038, loss = 0.061231
grad_step = 000039, loss = 0.057352
grad_step = 000040, loss = 0.053663
grad_step = 000041, loss = 0.050063
grad_step = 000042, loss = 0.046602
grad_step = 000043, loss = 0.043295
grad_step = 000044, loss = 0.040183
grad_step = 000045, loss = 0.037234
grad_step = 000046, loss = 0.034369
grad_step = 000047, loss = 0.031703
grad_step = 000048, loss = 0.029302
grad_step = 000049, loss = 0.027058
grad_step = 000050, loss = 0.024934
grad_step = 000051, loss = 0.022952
grad_step = 000052, loss = 0.021054
grad_step = 000053, loss = 0.019278
grad_step = 000054, loss = 0.017702
grad_step = 000055, loss = 0.016277
grad_step = 000056, loss = 0.014960
grad_step = 000057, loss = 0.013737
grad_step = 000058, loss = 0.012603
grad_step = 000059, loss = 0.011564
grad_step = 000060, loss = 0.010601
grad_step = 000061, loss = 0.009738
grad_step = 000062, loss = 0.008977
grad_step = 000063, loss = 0.008278
grad_step = 000064, loss = 0.007638
grad_step = 000065, loss = 0.007049
grad_step = 000066, loss = 0.006508
grad_step = 000067, loss = 0.006024
grad_step = 000068, loss = 0.005592
grad_step = 000069, loss = 0.005203
grad_step = 000070, loss = 0.004843
grad_step = 000071, loss = 0.004523
grad_step = 000072, loss = 0.004235
grad_step = 000073, loss = 0.003974
grad_step = 000074, loss = 0.003743
grad_step = 000075, loss = 0.003541
grad_step = 000076, loss = 0.003360
grad_step = 000077, loss = 0.003195
grad_step = 000078, loss = 0.003052
grad_step = 000079, loss = 0.002927
grad_step = 000080, loss = 0.002819
grad_step = 000081, loss = 0.002724
grad_step = 000082, loss = 0.002641
grad_step = 000083, loss = 0.002567
grad_step = 000084, loss = 0.002503
grad_step = 000085, loss = 0.002447
grad_step = 000086, loss = 0.002399
grad_step = 000087, loss = 0.002359
grad_step = 000088, loss = 0.002322
grad_step = 000089, loss = 0.002288
grad_step = 000090, loss = 0.002258
grad_step = 000091, loss = 0.002232
grad_step = 000092, loss = 0.002210
grad_step = 000093, loss = 0.002189
grad_step = 000094, loss = 0.002170
grad_step = 000095, loss = 0.002153
grad_step = 000096, loss = 0.002136
grad_step = 000097, loss = 0.002122
grad_step = 000098, loss = 0.002109
grad_step = 000099, loss = 0.002096
grad_step = 000100, loss = 0.002084
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002072
grad_step = 000102, loss = 0.002062
grad_step = 000103, loss = 0.002051
grad_step = 000104, loss = 0.002042
grad_step = 000105, loss = 0.002032
grad_step = 000106, loss = 0.002022
grad_step = 000107, loss = 0.002012
grad_step = 000108, loss = 0.002003
grad_step = 000109, loss = 0.001994
grad_step = 000110, loss = 0.001985
grad_step = 000111, loss = 0.001976
grad_step = 000112, loss = 0.001967
grad_step = 000113, loss = 0.001958
grad_step = 000114, loss = 0.001950
grad_step = 000115, loss = 0.001941
grad_step = 000116, loss = 0.001932
grad_step = 000117, loss = 0.001924
grad_step = 000118, loss = 0.001916
grad_step = 000119, loss = 0.001908
grad_step = 000120, loss = 0.001899
grad_step = 000121, loss = 0.001891
grad_step = 000122, loss = 0.001883
grad_step = 000123, loss = 0.001875
grad_step = 000124, loss = 0.001867
grad_step = 000125, loss = 0.001859
grad_step = 000126, loss = 0.001851
grad_step = 000127, loss = 0.001843
grad_step = 000128, loss = 0.001835
grad_step = 000129, loss = 0.001827
grad_step = 000130, loss = 0.001819
grad_step = 000131, loss = 0.001811
grad_step = 000132, loss = 0.001804
grad_step = 000133, loss = 0.001796
grad_step = 000134, loss = 0.001788
grad_step = 000135, loss = 0.001780
grad_step = 000136, loss = 0.001772
grad_step = 000137, loss = 0.001765
grad_step = 000138, loss = 0.001758
grad_step = 000139, loss = 0.001752
grad_step = 000140, loss = 0.001748
grad_step = 000141, loss = 0.001748
grad_step = 000142, loss = 0.001747
grad_step = 000143, loss = 0.001740
grad_step = 000144, loss = 0.001725
grad_step = 000145, loss = 0.001708
grad_step = 000146, loss = 0.001699
grad_step = 000147, loss = 0.001697
grad_step = 000148, loss = 0.001701
grad_step = 000149, loss = 0.001703
grad_step = 000150, loss = 0.001699
grad_step = 000151, loss = 0.001687
grad_step = 000152, loss = 0.001668
grad_step = 000153, loss = 0.001657
grad_step = 000154, loss = 0.001655
grad_step = 000155, loss = 0.001659
grad_step = 000156, loss = 0.001657
grad_step = 000157, loss = 0.001647
grad_step = 000158, loss = 0.001631
grad_step = 000159, loss = 0.001618
grad_step = 000160, loss = 0.001613
grad_step = 000161, loss = 0.001616
grad_step = 000162, loss = 0.001626
grad_step = 000163, loss = 0.001648
grad_step = 000164, loss = 0.001703
grad_step = 000165, loss = 0.001716
grad_step = 000166, loss = 0.001690
grad_step = 000167, loss = 0.001586
grad_step = 000168, loss = 0.001629
grad_step = 000169, loss = 0.001685
grad_step = 000170, loss = 0.001586
grad_step = 000171, loss = 0.001592
grad_step = 000172, loss = 0.001642
grad_step = 000173, loss = 0.001579
grad_step = 000174, loss = 0.001577
grad_step = 000175, loss = 0.001601
grad_step = 000176, loss = 0.001559
grad_step = 000177, loss = 0.001567
grad_step = 000178, loss = 0.001584
grad_step = 000179, loss = 0.001541
grad_step = 000180, loss = 0.001540
grad_step = 000181, loss = 0.001562
grad_step = 000182, loss = 0.001533
grad_step = 000183, loss = 0.001515
grad_step = 000184, loss = 0.001531
grad_step = 000185, loss = 0.001527
grad_step = 000186, loss = 0.001507
grad_step = 000187, loss = 0.001504
grad_step = 000188, loss = 0.001506
grad_step = 000189, loss = 0.001502
grad_step = 000190, loss = 0.001493
grad_step = 000191, loss = 0.001484
grad_step = 000192, loss = 0.001482
grad_step = 000193, loss = 0.001485
grad_step = 000194, loss = 0.001476
grad_step = 000195, loss = 0.001467
grad_step = 000196, loss = 0.001475
grad_step = 000197, loss = 0.001505
grad_step = 000198, loss = 0.001600
grad_step = 000199, loss = 0.001769
grad_step = 000200, loss = 0.001909
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001532
grad_step = 000202, loss = 0.001535
grad_step = 000203, loss = 0.001710
grad_step = 000204, loss = 0.001480
grad_step = 000205, loss = 0.001602
grad_step = 000206, loss = 0.001587
grad_step = 000207, loss = 0.001490
grad_step = 000208, loss = 0.001608
grad_step = 000209, loss = 0.001484
grad_step = 000210, loss = 0.001520
grad_step = 000211, loss = 0.001517
grad_step = 000212, loss = 0.001462
grad_step = 000213, loss = 0.001504
grad_step = 000214, loss = 0.001459
grad_step = 000215, loss = 0.001472
grad_step = 000216, loss = 0.001466
grad_step = 000217, loss = 0.001442
grad_step = 000218, loss = 0.001465
grad_step = 000219, loss = 0.001438
grad_step = 000220, loss = 0.001436
grad_step = 000221, loss = 0.001443
grad_step = 000222, loss = 0.001420
grad_step = 000223, loss = 0.001429
grad_step = 000224, loss = 0.001420
grad_step = 000225, loss = 0.001407
grad_step = 000226, loss = 0.001419
grad_step = 000227, loss = 0.001404
grad_step = 000228, loss = 0.001397
grad_step = 000229, loss = 0.001405
grad_step = 000230, loss = 0.001391
grad_step = 000231, loss = 0.001390
grad_step = 000232, loss = 0.001392
grad_step = 000233, loss = 0.001379
grad_step = 000234, loss = 0.001381
grad_step = 000235, loss = 0.001381
grad_step = 000236, loss = 0.001370
grad_step = 000237, loss = 0.001372
grad_step = 000238, loss = 0.001372
grad_step = 000239, loss = 0.001363
grad_step = 000240, loss = 0.001362
grad_step = 000241, loss = 0.001362
grad_step = 000242, loss = 0.001356
grad_step = 000243, loss = 0.001354
grad_step = 000244, loss = 0.001352
grad_step = 000245, loss = 0.001347
grad_step = 000246, loss = 0.001346
grad_step = 000247, loss = 0.001345
grad_step = 000248, loss = 0.001340
grad_step = 000249, loss = 0.001337
grad_step = 000250, loss = 0.001336
grad_step = 000251, loss = 0.001332
grad_step = 000252, loss = 0.001329
grad_step = 000253, loss = 0.001327
grad_step = 000254, loss = 0.001324
grad_step = 000255, loss = 0.001321
grad_step = 000256, loss = 0.001320
grad_step = 000257, loss = 0.001317
grad_step = 000258, loss = 0.001314
grad_step = 000259, loss = 0.001313
grad_step = 000260, loss = 0.001314
grad_step = 000261, loss = 0.001317
grad_step = 000262, loss = 0.001331
grad_step = 000263, loss = 0.001358
grad_step = 000264, loss = 0.001426
grad_step = 000265, loss = 0.001455
grad_step = 000266, loss = 0.001484
grad_step = 000267, loss = 0.001365
grad_step = 000268, loss = 0.001315
grad_step = 000269, loss = 0.001353
grad_step = 000270, loss = 0.001344
grad_step = 000271, loss = 0.001309
grad_step = 000272, loss = 0.001315
grad_step = 000273, loss = 0.001334
grad_step = 000274, loss = 0.001307
grad_step = 000275, loss = 0.001272
grad_step = 000276, loss = 0.001294
grad_step = 000277, loss = 0.001319
grad_step = 000278, loss = 0.001288
grad_step = 000279, loss = 0.001259
grad_step = 000280, loss = 0.001269
grad_step = 000281, loss = 0.001284
grad_step = 000282, loss = 0.001277
grad_step = 000283, loss = 0.001253
grad_step = 000284, loss = 0.001247
grad_step = 000285, loss = 0.001258
grad_step = 000286, loss = 0.001259
grad_step = 000287, loss = 0.001248
grad_step = 000288, loss = 0.001233
grad_step = 000289, loss = 0.001232
grad_step = 000290, loss = 0.001240
grad_step = 000291, loss = 0.001238
grad_step = 000292, loss = 0.001228
grad_step = 000293, loss = 0.001218
grad_step = 000294, loss = 0.001216
grad_step = 000295, loss = 0.001218
grad_step = 000296, loss = 0.001217
grad_step = 000297, loss = 0.001212
grad_step = 000298, loss = 0.001206
grad_step = 000299, loss = 0.001203
grad_step = 000300, loss = 0.001201
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001199
grad_step = 000302, loss = 0.001194
grad_step = 000303, loss = 0.001188
grad_step = 000304, loss = 0.001183
grad_step = 000305, loss = 0.001180
grad_step = 000306, loss = 0.001179
grad_step = 000307, loss = 0.001177
grad_step = 000308, loss = 0.001175
grad_step = 000309, loss = 0.001172
grad_step = 000310, loss = 0.001170
grad_step = 000311, loss = 0.001170
grad_step = 000312, loss = 0.001179
grad_step = 000313, loss = 0.001204
grad_step = 000314, loss = 0.001277
grad_step = 000315, loss = 0.001351
grad_step = 000316, loss = 0.001481
grad_step = 000317, loss = 0.001310
grad_step = 000318, loss = 0.001209
grad_step = 000319, loss = 0.001212
grad_step = 000320, loss = 0.001223
grad_step = 000321, loss = 0.001194
grad_step = 000322, loss = 0.001173
grad_step = 000323, loss = 0.001209
grad_step = 000324, loss = 0.001202
grad_step = 000325, loss = 0.001137
grad_step = 000326, loss = 0.001153
grad_step = 000327, loss = 0.001185
grad_step = 000328, loss = 0.001146
grad_step = 000329, loss = 0.001125
grad_step = 000330, loss = 0.001141
grad_step = 000331, loss = 0.001134
grad_step = 000332, loss = 0.001117
grad_step = 000333, loss = 0.001114
grad_step = 000334, loss = 0.001118
grad_step = 000335, loss = 0.001113
grad_step = 000336, loss = 0.001095
grad_step = 000337, loss = 0.001092
grad_step = 000338, loss = 0.001102
grad_step = 000339, loss = 0.001095
grad_step = 000340, loss = 0.001081
grad_step = 000341, loss = 0.001072
grad_step = 000342, loss = 0.001075
grad_step = 000343, loss = 0.001079
grad_step = 000344, loss = 0.001070
grad_step = 000345, loss = 0.001060
grad_step = 000346, loss = 0.001056
grad_step = 000347, loss = 0.001057
grad_step = 000348, loss = 0.001054
grad_step = 000349, loss = 0.001047
grad_step = 000350, loss = 0.001044
grad_step = 000351, loss = 0.001043
grad_step = 000352, loss = 0.001038
grad_step = 000353, loss = 0.001032
grad_step = 000354, loss = 0.001028
grad_step = 000355, loss = 0.001027
grad_step = 000356, loss = 0.001026
grad_step = 000357, loss = 0.001021
grad_step = 000358, loss = 0.001015
grad_step = 000359, loss = 0.001012
grad_step = 000360, loss = 0.001009
grad_step = 000361, loss = 0.001005
grad_step = 000362, loss = 0.001002
grad_step = 000363, loss = 0.000998
grad_step = 000364, loss = 0.000994
grad_step = 000365, loss = 0.000991
grad_step = 000366, loss = 0.000988
grad_step = 000367, loss = 0.000986
grad_step = 000368, loss = 0.000984
grad_step = 000369, loss = 0.000981
grad_step = 000370, loss = 0.000981
grad_step = 000371, loss = 0.000983
grad_step = 000372, loss = 0.000992
grad_step = 000373, loss = 0.001011
grad_step = 000374, loss = 0.001064
grad_step = 000375, loss = 0.001125
grad_step = 000376, loss = 0.001251
grad_step = 000377, loss = 0.001181
grad_step = 000378, loss = 0.001080
grad_step = 000379, loss = 0.000969
grad_step = 000380, loss = 0.001013
grad_step = 000381, loss = 0.001075
grad_step = 000382, loss = 0.000990
grad_step = 000383, loss = 0.000974
grad_step = 000384, loss = 0.001029
grad_step = 000385, loss = 0.000986
grad_step = 000386, loss = 0.000943
grad_step = 000387, loss = 0.000968
grad_step = 000388, loss = 0.000970
grad_step = 000389, loss = 0.000942
grad_step = 000390, loss = 0.000935
grad_step = 000391, loss = 0.000949
grad_step = 000392, loss = 0.000944
grad_step = 000393, loss = 0.000916
grad_step = 000394, loss = 0.000914
grad_step = 000395, loss = 0.000925
grad_step = 000396, loss = 0.000920
grad_step = 000397, loss = 0.000909
grad_step = 000398, loss = 0.000902
grad_step = 000399, loss = 0.000899
grad_step = 000400, loss = 0.000896
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000892
grad_step = 000402, loss = 0.000890
grad_step = 000403, loss = 0.000884
grad_step = 000404, loss = 0.000878
grad_step = 000405, loss = 0.000877
grad_step = 000406, loss = 0.000876
grad_step = 000407, loss = 0.000870
grad_step = 000408, loss = 0.000861
grad_step = 000409, loss = 0.000858
grad_step = 000410, loss = 0.000860
grad_step = 000411, loss = 0.000859
grad_step = 000412, loss = 0.000853
grad_step = 000413, loss = 0.000847
grad_step = 000414, loss = 0.000843
grad_step = 000415, loss = 0.000841
grad_step = 000416, loss = 0.000837
grad_step = 000417, loss = 0.000833
grad_step = 000418, loss = 0.000832
grad_step = 000419, loss = 0.000831
grad_step = 000420, loss = 0.000827
grad_step = 000421, loss = 0.000823
grad_step = 000422, loss = 0.000819
grad_step = 000423, loss = 0.000816
grad_step = 000424, loss = 0.000813
grad_step = 000425, loss = 0.000810
grad_step = 000426, loss = 0.000807
grad_step = 000427, loss = 0.000805
grad_step = 000428, loss = 0.000803
grad_step = 000429, loss = 0.000801
grad_step = 000430, loss = 0.000800
grad_step = 000431, loss = 0.000800
grad_step = 000432, loss = 0.000803
grad_step = 000433, loss = 0.000809
grad_step = 000434, loss = 0.000828
grad_step = 000435, loss = 0.000856
grad_step = 000436, loss = 0.000931
grad_step = 000437, loss = 0.000975
grad_step = 000438, loss = 0.001057
grad_step = 000439, loss = 0.000925
grad_step = 000440, loss = 0.000822
grad_step = 000441, loss = 0.000821
grad_step = 000442, loss = 0.000885
grad_step = 000443, loss = 0.000890
grad_step = 000444, loss = 0.000796
grad_step = 000445, loss = 0.000805
grad_step = 000446, loss = 0.000848
grad_step = 000447, loss = 0.000788
grad_step = 000448, loss = 0.000760
grad_step = 000449, loss = 0.000800
grad_step = 000450, loss = 0.000794
grad_step = 000451, loss = 0.000760
grad_step = 000452, loss = 0.000763
grad_step = 000453, loss = 0.000784
grad_step = 000454, loss = 0.000772
grad_step = 000455, loss = 0.000743
grad_step = 000456, loss = 0.000743
grad_step = 000457, loss = 0.000758
grad_step = 000458, loss = 0.000747
grad_step = 000459, loss = 0.000726
grad_step = 000460, loss = 0.000722
grad_step = 000461, loss = 0.000732
grad_step = 000462, loss = 0.000734
grad_step = 000463, loss = 0.000722
grad_step = 000464, loss = 0.000712
grad_step = 000465, loss = 0.000715
grad_step = 000466, loss = 0.000721
grad_step = 000467, loss = 0.000719
grad_step = 000468, loss = 0.000710
grad_step = 000469, loss = 0.000705
grad_step = 000470, loss = 0.000706
grad_step = 000471, loss = 0.000708
grad_step = 000472, loss = 0.000707
grad_step = 000473, loss = 0.000702
grad_step = 000474, loss = 0.000698
grad_step = 000475, loss = 0.000698
grad_step = 000476, loss = 0.000700
grad_step = 000477, loss = 0.000702
grad_step = 000478, loss = 0.000702
grad_step = 000479, loss = 0.000702
grad_step = 000480, loss = 0.000700
grad_step = 000481, loss = 0.000699
grad_step = 000482, loss = 0.000698
grad_step = 000483, loss = 0.000695
grad_step = 000484, loss = 0.000693
grad_step = 000485, loss = 0.000686
grad_step = 000486, loss = 0.000678
grad_step = 000487, loss = 0.000668
grad_step = 000488, loss = 0.000660
grad_step = 000489, loss = 0.000654
grad_step = 000490, loss = 0.000650
grad_step = 000491, loss = 0.000648
grad_step = 000492, loss = 0.000647
grad_step = 000493, loss = 0.000647
grad_step = 000494, loss = 0.000648
grad_step = 000495, loss = 0.000648
grad_step = 000496, loss = 0.000647
grad_step = 000497, loss = 0.000645
grad_step = 000498, loss = 0.000643
grad_step = 000499, loss = 0.000642
grad_step = 000500, loss = 0.000640
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000638
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

  date_run                              2020-05-15 10:13:26.760834
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.270038
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 10:13:26.766293
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.226393
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 10:13:26.773787
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130254
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 10:13:26.778658
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.44012
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
0   2020-05-15 10:12:59.496077  ...    mean_absolute_error
1   2020-05-15 10:12:59.499490  ...     mean_squared_error
2   2020-05-15 10:12:59.502563  ...  median_absolute_error
3   2020-05-15 10:12:59.505449  ...               r2_score
4   2020-05-15 10:13:08.927443  ...    mean_absolute_error
5   2020-05-15 10:13:08.931077  ...     mean_squared_error
6   2020-05-15 10:13:08.934113  ...  median_absolute_error
7   2020-05-15 10:13:08.937583  ...               r2_score
8   2020-05-15 10:13:26.760834  ...    mean_absolute_error
9   2020-05-15 10:13:26.766293  ...     mean_squared_error
10  2020-05-15 10:13:26.773787  ...  median_absolute_error
11  2020-05-15 10:13:26.778658  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b05d6c898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▊       | 2826240/9912422 [00:00<00:00, 28257374.14it/s]9920512it [00:00, 32875664.11it/s]                             
0it [00:00, ?it/s]32768it [00:00, 900660.25it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 465763.19it/s]1654784it [00:00, 11931838.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 235966.15it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab871ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab55680b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab871ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab7ca3080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab54dc470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab54c7be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab871ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab7c616a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7ab54dc470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b05d24e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdbd7c3b1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4b9ad7e9a2040c221a90e7473d76776d001150bf09e72e763a790ddf6348c5c6
  Stored in directory: /tmp/pip-ephem-wheel-cache-tftp6kzd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdb77965d68> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1916928/17464789 [==>...........................] - ETA: 0s
 8986624/17464789 [==============>...............] - ETA: 0s
16211968/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 10:14:52.981390: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 10:14:52.985686: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 10:14:52.985880: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f71aeb9e50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 10:14:52.985894: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7740 - accuracy: 0.4930
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7931 - accuracy: 0.4918
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8021 - accuracy: 0.4912
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8002 - accuracy: 0.4913
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7471 - accuracy: 0.4947
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7927 - accuracy: 0.4918
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7387 - accuracy: 0.4953
11000/25000 [============>.................] - ETA: 3s - loss: 7.7558 - accuracy: 0.4942
12000/25000 [=============>................] - ETA: 3s - loss: 7.7714 - accuracy: 0.4932
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7421 - accuracy: 0.4951
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7422 - accuracy: 0.4951
15000/25000 [=================>............] - ETA: 2s - loss: 7.7412 - accuracy: 0.4951
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7385 - accuracy: 0.4953
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7505 - accuracy: 0.4945
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7280 - accuracy: 0.4960
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7223 - accuracy: 0.4964
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7149 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7068 - accuracy: 0.4974
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6945 - accuracy: 0.4982
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 7s 273us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 10:15:06.350355
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 10:15:06.350355  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:34:03, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:37:53, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:17:47, 23.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:12:53, 33.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.57M/862M [00:01<5:02:41, 47.3kB/s].vector_cache/glove.6B.zip:   1%|          | 6.55M/862M [00:01<3:31:02, 67.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:26:51, 96.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:01<1:42:32, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:01<1:11:24, 196kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<49:55, 280kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<34:47, 399kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<24:23, 567kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<17:03, 806kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:02<12:00, 1.14MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.8M/862M [00:02<08:26, 1.61MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<05:59, 2.26MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.0M/862M [00:03<05:26, 2.47MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:03<04:17, 3.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<05:26, 2.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<05:50, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:17, 3.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<10:33, 1.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<08:57, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:07<06:36, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<07:25, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<07:55, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<06:14, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<04:31, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<1:37:10, 136kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<1:09:21, 190kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<48:47, 270kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<37:08, 353kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<28:41, 457kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<20:40, 634kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:13<14:37, 894kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:15<16:11, 807kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<12:40, 1.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<09:08, 1.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<09:25, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<07:55, 1.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:17<05:52, 2.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:19<07:09, 1.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<04:45, 2.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:21<06:22, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<05:47, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<04:22, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:23<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<05:33, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:23<04:10, 3.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<05:56, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<05:27, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:08, 3.06MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:52, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:42, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:13, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<03:51, 3.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<07:15, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<06:23, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:45, 2.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<06:15, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<06:57, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:30, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<03:58, 3.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<11:59:35, 17.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<8:24:43, 24.6kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<5:52:53, 35.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<4:09:12, 49.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<2:56:56, 69.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<2:04:15, 99.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<1:26:51, 142kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<1:08:28, 180kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<49:11, 250kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<34:40, 354kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<27:03, 452kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<21:35, 567kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<15:37, 782kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<11:06, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<12:22, 983kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<09:56, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:15, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:52, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:44, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:01, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:21, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:41, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<04:16, 2.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:48, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:32, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:11, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:07, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<03:53, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:47, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:02, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<03:48, 3.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:25, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:50, 2.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<03:31, 3.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:11, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:37, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<05:37, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:38, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:00, 1.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:30, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:59, 2.88MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:24:39, 136kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<1:00:25, 190kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<42:27, 270kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<32:17, 353kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<24:56, 457kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<17:58, 634kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<12:39, 897kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<21:00, 540kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<15:53, 714kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<11:20, 997kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:33, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:32, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<06:14, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:00, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:29, 2.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:46, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:51, 2.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:20, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:40, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:29, 3.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<02:35, 4.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<43:18, 254kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<32:33, 338kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<23:19, 471kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<18:01, 606kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<13:45, 794kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<09:54, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:25, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<07:44, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:41, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:28, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:44, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:16, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:24, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:41, 2.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:04, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:38, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:31, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:55, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:33, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:27, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:34, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:23, 2.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:10, 3.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<11:08, 938kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<08:52, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:25, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:55, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:54, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:23, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:30, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:54, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:41, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:00, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:26, 2.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:49, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:22, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:16, 2.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:37, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:17, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:15, 3.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:36, 2.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:15, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:13, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:16, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:12, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:02, 3.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<1:23:26, 119kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<59:23, 167kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<41:43, 237kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<31:24, 313kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<23:55, 411kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<17:08, 572kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<12:06, 808kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:58, 753kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<10:05, 967kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<07:18, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:21, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:08, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:29, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:55, 2.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<9:18:28, 17.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<6:31:39, 24.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<4:33:39, 35.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<3:13:05, 49.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<2:17:05, 69.7kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:36:17, 99.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<1:07:10, 141kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<57:56, 164kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<41:31, 228kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<29:14, 324kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<22:34, 417kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<17:44, 531kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<12:49, 734kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<09:02, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<17:20, 539kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<13:07, 712kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<09:24, 991kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:43, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:04, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:46, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:56, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:34, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:17, 2.78MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<11:35, 789kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<09:03, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:31, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:41, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:32, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:57, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:34, 2.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<08:45, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:04, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<05:09, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:41, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:49, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:29, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:12, 2.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<20:15, 438kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<15:05, 587kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<10:46, 821kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:34, 920kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<07:36, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:31, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:55, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:02, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:44, 2.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:39, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:09, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:05, 2.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:12, 2.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:50, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<02:52, 2.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:01, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:42, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<02:48, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:57, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:38, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<02:43, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:24, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:28, 2.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:31, 3.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:45, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:35, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:05, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:47, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:11, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:08, 2.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:07, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:43, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:48, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:52, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:23, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:25, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:30, 3.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:07, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:25, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:17, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:10, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:33, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:35, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:47, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:27, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:37, 3.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:39, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:32, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:19, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:31, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:12, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:19, 3.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<08:38, 884kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:49, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<04:57, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:13, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:27, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:18, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:03, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:36, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:42, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:38, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:18, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:29, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:29, 2.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:11, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:23, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:24, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:19, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:03, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:17, 3.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:17, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:01, 2.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:17, 3.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:16, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:44, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:58, 2.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:12, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:58, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:15, 3.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:12, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:57, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:14, 3.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:10, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:38, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:53, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:06, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:53, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:11, 3.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:06, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:51, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:10, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:05, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:50, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:09, 3.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:03, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:49, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:07, 3.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:02, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:47, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:07, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:00, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:46, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:05, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:58, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:04, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:21, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:40, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:52, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:39, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:00, 3.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:51, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:39, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:00, 3.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:50, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:37, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<01:59, 3.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:48, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:35, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:56, 3.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:45, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:32, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:55, 3.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:31, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:54, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:29, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:53, 3.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:40, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:03, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:24, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<01:44, 3.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:31, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:44, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:45, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:14, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:51, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:07, 2.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:47, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:06, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:26, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:35, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:23, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:19, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:45, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:18, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:44, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:16, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:43, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:26, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:14, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:41, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:24, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:12, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:40, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:22, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:11, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:37, 3.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:08, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:37, 3.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:18, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:01, 2.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:28, 3.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:27, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:55, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:10, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:38, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:50, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:36, 2.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<4:35:43, 17.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<3:13:13, 24.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<2:14:38, 35.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<1:34:38, 49.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<1:06:37, 70.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<46:29, 100kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<33:23, 138kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<24:18, 190kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<17:12, 268kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<11:57, 381kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<4:29:49, 16.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<3:09:05, 24.0kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<2:11:43, 34.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<1:32:31, 48.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:05:07, 68.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<45:26, 98.0kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<32:36, 135kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<23:13, 190kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<16:16, 269kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<12:19, 353kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<09:03, 479kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<06:25, 672kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:28, 783kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:42, 909kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:27, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<02:28, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:28, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:51, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<02:06, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:25, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:26, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:12, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:38, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:04, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:23, 2.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:54, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:44, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:18, 2.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:49, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:04, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:36, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:10, 3.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:31, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:10, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:36, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:58, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:42, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:15, 2.92MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:42<00:55, 3.94MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<15:22, 238kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<11:07, 329kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<07:49, 464kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<06:16, 573kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:45, 754kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<03:23, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:11, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:32, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:51, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:05, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:10, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:40, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:11, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:43, 910kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:57, 1.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:08, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:15, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:45, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:53, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:27, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:03, 3.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:18, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:25, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:51, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:27, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:00<01:02, 2.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<22:24, 136kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<15:58, 190kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<11:09, 270kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<08:25, 353kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<06:11, 480kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<04:22, 674kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<03:42, 784kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:53, 1.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:04, 1.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:06, 1.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:45, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:17, 2.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:33, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:22, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:00, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:20, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:13, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:54, 2.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:15, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:25, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:06, 2.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:47, 3.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:48, 913kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:13, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:36, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:41, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:26, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:03, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:18, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:25, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:06, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:49, 2.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:11, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:04, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:48, 2.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:05, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:00, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:45, 2.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:12, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:57, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:25<00:40, 3.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<15:47, 136kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<11:14, 191kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<07:49, 271kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<05:52, 355kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<04:18, 481kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<03:02, 676kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:33, 785kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:59, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:25, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:26, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:12, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:53, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:03, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:55, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:41, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:53, 2.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:48, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:36, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:49, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:56, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:44, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:46, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:32, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:44, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:40, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:30, 3.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:42, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:39, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:29, 3.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:37, 2.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:28, 3.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:35, 2.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:26, 3.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:25, 3.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:35, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:40, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:32, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:30, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:22, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:28, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:21, 3.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:26, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:19, 3.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:27, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:18, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:25, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:17, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:21, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:19, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:19, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:17, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:20, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:14, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:10, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:07, 3.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:05, 3.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:16, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:09, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 3.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.33MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 839/400000 [00:00<00:47, 8384.99it/s]  0%|          | 1704/400000 [00:00<00:47, 8461.96it/s]  1%|          | 2548/400000 [00:00<00:47, 8454.69it/s]  1%|          | 3427/400000 [00:00<00:46, 8549.74it/s]  1%|          | 4254/400000 [00:00<00:46, 8461.39it/s]  1%|▏         | 5088/400000 [00:00<00:46, 8423.76it/s]  1%|▏         | 5959/400000 [00:00<00:46, 8506.08it/s]  2%|▏         | 6816/400000 [00:00<00:46, 8524.27it/s]  2%|▏         | 7675/400000 [00:00<00:45, 8542.49it/s]  2%|▏         | 8551/400000 [00:01<00:45, 8603.79it/s]  2%|▏         | 9434/400000 [00:01<00:45, 8669.13it/s]  3%|▎         | 10285/400000 [00:01<00:45, 8583.76it/s]  3%|▎         | 11133/400000 [00:01<00:46, 8358.95it/s]  3%|▎         | 11963/400000 [00:01<00:47, 8145.34it/s]  3%|▎         | 12831/400000 [00:01<00:46, 8297.27it/s]  3%|▎         | 13660/400000 [00:01<00:47, 8204.77it/s]  4%|▎         | 14480/400000 [00:01<00:47, 8180.47it/s]  4%|▍         | 15298/400000 [00:01<00:47, 8169.68it/s]  4%|▍         | 16115/400000 [00:01<00:47, 8057.59it/s]  4%|▍         | 16971/400000 [00:02<00:46, 8199.54it/s]  4%|▍         | 17842/400000 [00:02<00:45, 8344.02it/s]  5%|▍         | 18721/400000 [00:02<00:45, 8470.51it/s]  5%|▍         | 19588/400000 [00:02<00:44, 8529.34it/s]  5%|▌         | 20471/400000 [00:02<00:44, 8616.40it/s]  5%|▌         | 21341/400000 [00:02<00:43, 8640.63it/s]  6%|▌         | 22206/400000 [00:02<00:43, 8614.02it/s]  6%|▌         | 23082/400000 [00:02<00:43, 8656.07it/s]  6%|▌         | 23955/400000 [00:02<00:43, 8676.37it/s]  6%|▌         | 24836/400000 [00:02<00:43, 8713.38it/s]  6%|▋         | 25708/400000 [00:03<00:42, 8706.63it/s]  7%|▋         | 26579/400000 [00:03<00:43, 8560.15it/s]  7%|▋         | 27456/400000 [00:03<00:43, 8620.47it/s]  7%|▋         | 28331/400000 [00:03<00:42, 8657.79it/s]  7%|▋         | 29198/400000 [00:03<00:42, 8649.70it/s]  8%|▊         | 30064/400000 [00:03<00:43, 8554.04it/s]  8%|▊         | 30920/400000 [00:03<00:43, 8458.68it/s]  8%|▊         | 31798/400000 [00:03<00:43, 8551.02it/s]  8%|▊         | 32674/400000 [00:03<00:42, 8610.62it/s]  8%|▊         | 33553/400000 [00:03<00:42, 8663.34it/s]  9%|▊         | 34435/400000 [00:04<00:41, 8707.36it/s]  9%|▉         | 35312/400000 [00:04<00:41, 8724.62it/s]  9%|▉         | 36186/400000 [00:04<00:41, 8728.38it/s]  9%|▉         | 37060/400000 [00:04<00:41, 8730.46it/s]  9%|▉         | 37934/400000 [00:04<00:41, 8733.05it/s] 10%|▉         | 38810/400000 [00:04<00:41, 8740.13it/s] 10%|▉         | 39691/400000 [00:04<00:41, 8760.03it/s] 10%|█         | 40568/400000 [00:04<00:41, 8737.80it/s] 10%|█         | 41453/400000 [00:04<00:40, 8768.64it/s] 11%|█         | 42330/400000 [00:04<00:41, 8697.92it/s] 11%|█         | 43200/400000 [00:05<00:41, 8667.84it/s] 11%|█         | 44067/400000 [00:05<00:41, 8656.55it/s] 11%|█         | 44933/400000 [00:05<00:41, 8460.92it/s] 11%|█▏        | 45816/400000 [00:05<00:41, 8567.26it/s] 12%|█▏        | 46678/400000 [00:05<00:41, 8581.01it/s] 12%|█▏        | 47557/400000 [00:05<00:40, 8640.20it/s] 12%|█▏        | 48423/400000 [00:05<00:40, 8643.35it/s] 12%|█▏        | 49296/400000 [00:05<00:40, 8667.03it/s] 13%|█▎        | 50165/400000 [00:05<00:40, 8673.36it/s] 13%|█▎        | 51033/400000 [00:05<00:40, 8603.22it/s] 13%|█▎        | 51894/400000 [00:06<00:40, 8580.28it/s] 13%|█▎        | 52759/400000 [00:06<00:40, 8599.80it/s] 13%|█▎        | 53631/400000 [00:06<00:40, 8634.05it/s] 14%|█▎        | 54495/400000 [00:06<00:40, 8495.91it/s] 14%|█▍        | 55365/400000 [00:06<00:40, 8555.71it/s] 14%|█▍        | 56222/400000 [00:06<00:40, 8555.48it/s] 14%|█▍        | 57078/400000 [00:06<00:40, 8411.07it/s] 14%|█▍        | 57949/400000 [00:06<00:40, 8498.43it/s] 15%|█▍        | 58800/400000 [00:06<00:40, 8381.72it/s] 15%|█▍        | 59646/400000 [00:06<00:40, 8404.36it/s] 15%|█▌        | 60491/400000 [00:07<00:40, 8415.89it/s] 15%|█▌        | 61359/400000 [00:07<00:39, 8492.37it/s] 16%|█▌        | 62209/400000 [00:07<00:40, 8420.05it/s] 16%|█▌        | 63052/400000 [00:07<00:40, 8371.63it/s] 16%|█▌        | 63912/400000 [00:07<00:39, 8436.57it/s] 16%|█▌        | 64757/400000 [00:07<00:40, 8189.66it/s] 16%|█▋        | 65640/400000 [00:07<00:39, 8371.41it/s] 17%|█▋        | 66507/400000 [00:07<00:39, 8458.50it/s] 17%|█▋        | 67355/400000 [00:07<00:39, 8404.35it/s] 17%|█▋        | 68197/400000 [00:07<00:39, 8402.67it/s] 17%|█▋        | 69039/400000 [00:08<00:39, 8309.22it/s] 17%|█▋        | 69903/400000 [00:08<00:39, 8404.42it/s] 18%|█▊        | 70753/400000 [00:08<00:39, 8430.65it/s] 18%|█▊        | 71631/400000 [00:08<00:38, 8532.44it/s] 18%|█▊        | 72507/400000 [00:08<00:38, 8599.27it/s] 18%|█▊        | 73368/400000 [00:08<00:38, 8568.62it/s] 19%|█▊        | 74226/400000 [00:08<00:38, 8397.23it/s] 19%|█▉        | 75067/400000 [00:08<00:39, 8294.24it/s] 19%|█▉        | 75898/400000 [00:08<00:39, 8169.25it/s] 19%|█▉        | 76731/400000 [00:09<00:39, 8214.91it/s] 19%|█▉        | 77554/400000 [00:09<00:39, 8213.32it/s] 20%|█▉        | 78410/400000 [00:09<00:38, 8314.01it/s] 20%|█▉        | 79259/400000 [00:09<00:38, 8365.64it/s] 20%|██        | 80124/400000 [00:09<00:37, 8446.40it/s] 20%|██        | 80986/400000 [00:09<00:37, 8496.82it/s] 20%|██        | 81853/400000 [00:09<00:37, 8547.33it/s] 21%|██        | 82709/400000 [00:09<00:37, 8352.10it/s] 21%|██        | 83546/400000 [00:09<00:38, 8210.53it/s] 21%|██        | 84426/400000 [00:09<00:37, 8376.51it/s] 21%|██▏       | 85303/400000 [00:10<00:37, 8488.73it/s] 22%|██▏       | 86177/400000 [00:10<00:36, 8562.36it/s] 22%|██▏       | 87042/400000 [00:10<00:36, 8473.76it/s] 22%|██▏       | 87895/400000 [00:10<00:36, 8488.71it/s] 22%|██▏       | 88767/400000 [00:10<00:36, 8554.64it/s] 22%|██▏       | 89639/400000 [00:10<00:36, 8601.46it/s] 23%|██▎       | 90517/400000 [00:10<00:35, 8653.54it/s] 23%|██▎       | 91404/400000 [00:10<00:35, 8715.48it/s] 23%|██▎       | 92283/400000 [00:10<00:35, 8735.51it/s] 23%|██▎       | 93168/400000 [00:10<00:34, 8768.90it/s] 24%|██▎       | 94046/400000 [00:11<00:35, 8619.75it/s] 24%|██▎       | 94909/400000 [00:11<00:35, 8604.87it/s] 24%|██▍       | 95781/400000 [00:11<00:35, 8636.67it/s] 24%|██▍       | 96656/400000 [00:11<00:34, 8669.68it/s] 24%|██▍       | 97525/400000 [00:11<00:34, 8673.90it/s] 25%|██▍       | 98393/400000 [00:11<00:34, 8628.23it/s] 25%|██▍       | 99257/400000 [00:11<00:35, 8446.82it/s] 25%|██▌       | 100103/400000 [00:11<00:36, 8119.08it/s] 25%|██▌       | 100966/400000 [00:11<00:36, 8263.40it/s] 25%|██▌       | 101843/400000 [00:11<00:35, 8408.53it/s] 26%|██▌       | 102687/400000 [00:12<00:35, 8406.84it/s] 26%|██▌       | 103560/400000 [00:12<00:34, 8500.69it/s] 26%|██▌       | 104445/400000 [00:12<00:34, 8600.77it/s] 26%|██▋       | 105315/400000 [00:12<00:34, 8628.89it/s] 27%|██▋       | 106196/400000 [00:12<00:33, 8682.00it/s] 27%|██▋       | 107067/400000 [00:12<00:33, 8688.24it/s] 27%|██▋       | 107937/400000 [00:12<00:33, 8675.21it/s] 27%|██▋       | 108825/400000 [00:12<00:33, 8735.62it/s] 27%|██▋       | 109711/400000 [00:12<00:33, 8771.76it/s] 28%|██▊       | 110595/400000 [00:12<00:32, 8790.53it/s] 28%|██▊       | 111475/400000 [00:13<00:33, 8659.83it/s] 28%|██▊       | 112353/400000 [00:13<00:33, 8693.57it/s] 28%|██▊       | 113223/400000 [00:13<00:33, 8563.25it/s] 29%|██▊       | 114105/400000 [00:13<00:33, 8636.62it/s] 29%|██▊       | 114981/400000 [00:13<00:32, 8671.55it/s] 29%|██▉       | 115849/400000 [00:13<00:32, 8624.46it/s] 29%|██▉       | 116712/400000 [00:13<00:33, 8545.06it/s] 29%|██▉       | 117567/400000 [00:13<00:33, 8499.20it/s] 30%|██▉       | 118441/400000 [00:13<00:32, 8567.73it/s] 30%|██▉       | 119321/400000 [00:13<00:32, 8633.33it/s] 30%|███       | 120185/400000 [00:14<00:32, 8555.28it/s] 30%|███       | 121071/400000 [00:14<00:32, 8642.84it/s] 30%|███       | 121946/400000 [00:14<00:32, 8672.12it/s] 31%|███       | 122829/400000 [00:14<00:31, 8717.71it/s] 31%|███       | 123717/400000 [00:14<00:31, 8764.97it/s] 31%|███       | 124603/400000 [00:14<00:31, 8792.03it/s] 31%|███▏      | 125485/400000 [00:14<00:31, 8799.34it/s] 32%|███▏      | 126373/400000 [00:14<00:31, 8822.73it/s] 32%|███▏      | 127261/400000 [00:14<00:30, 8838.44it/s] 32%|███▏      | 128149/400000 [00:14<00:30, 8848.96it/s] 32%|███▏      | 129038/400000 [00:15<00:30, 8859.24it/s] 32%|███▏      | 129924/400000 [00:15<00:30, 8838.91it/s] 33%|███▎      | 130808/400000 [00:15<00:30, 8834.66it/s] 33%|███▎      | 131692/400000 [00:15<00:30, 8831.15it/s] 33%|███▎      | 132576/400000 [00:15<00:30, 8817.98it/s] 33%|███▎      | 133458/400000 [00:15<00:30, 8755.02it/s] 34%|███▎      | 134334/400000 [00:15<00:30, 8748.43it/s] 34%|███▍      | 135216/400000 [00:15<00:30, 8769.55it/s] 34%|███▍      | 136105/400000 [00:15<00:29, 8802.54it/s] 34%|███▍      | 136994/400000 [00:15<00:29, 8825.79it/s] 34%|███▍      | 137877/400000 [00:16<00:29, 8825.44it/s] 35%|███▍      | 138760/400000 [00:16<00:29, 8777.84it/s] 35%|███▍      | 139638/400000 [00:16<00:30, 8542.05it/s] 35%|███▌      | 140513/400000 [00:16<00:30, 8602.00it/s] 35%|███▌      | 141395/400000 [00:16<00:29, 8663.90it/s] 36%|███▌      | 142275/400000 [00:16<00:29, 8702.74it/s] 36%|███▌      | 143146/400000 [00:16<00:29, 8695.34it/s] 36%|███▌      | 144022/400000 [00:16<00:29, 8714.12it/s] 36%|███▌      | 144894/400000 [00:16<00:29, 8669.00it/s] 36%|███▋      | 145762/400000 [00:17<00:29, 8524.02it/s] 37%|███▋      | 146616/400000 [00:17<00:30, 8419.16it/s] 37%|███▋      | 147496/400000 [00:17<00:29, 8529.03it/s] 37%|███▋      | 148350/400000 [00:17<00:29, 8503.42it/s] 37%|███▋      | 149202/400000 [00:17<00:30, 8348.56it/s] 38%|███▊      | 150038/400000 [00:17<00:30, 8288.40it/s] 38%|███▊      | 150895/400000 [00:17<00:29, 8369.74it/s] 38%|███▊      | 151776/400000 [00:17<00:29, 8496.74it/s] 38%|███▊      | 152658/400000 [00:17<00:28, 8590.79it/s] 38%|███▊      | 153519/400000 [00:17<00:28, 8580.12it/s] 39%|███▊      | 154378/400000 [00:18<00:29, 8452.73it/s] 39%|███▉      | 155251/400000 [00:18<00:28, 8531.30it/s] 39%|███▉      | 156134/400000 [00:18<00:28, 8615.90it/s] 39%|███▉      | 156997/400000 [00:18<00:28, 8575.15it/s] 39%|███▉      | 157878/400000 [00:18<00:28, 8642.32it/s] 40%|███▉      | 158743/400000 [00:18<00:27, 8629.26it/s] 40%|███▉      | 159607/400000 [00:18<00:28, 8551.24it/s] 40%|████      | 160463/400000 [00:18<00:28, 8329.93it/s] 40%|████      | 161344/400000 [00:18<00:28, 8468.05it/s] 41%|████      | 162200/400000 [00:18<00:28, 8492.66it/s] 41%|████      | 163074/400000 [00:19<00:27, 8563.23it/s] 41%|████      | 163961/400000 [00:19<00:27, 8651.90it/s] 41%|████      | 164848/400000 [00:19<00:26, 8714.62it/s] 41%|████▏     | 165721/400000 [00:19<00:26, 8709.11it/s] 42%|████▏     | 166608/400000 [00:19<00:26, 8756.19it/s] 42%|████▏     | 167485/400000 [00:19<00:26, 8731.98it/s] 42%|████▏     | 168364/400000 [00:19<00:26, 8748.77it/s] 42%|████▏     | 169240/400000 [00:19<00:26, 8732.82it/s] 43%|████▎     | 170128/400000 [00:19<00:26, 8775.18it/s] 43%|████▎     | 171009/400000 [00:19<00:26, 8785.27it/s] 43%|████▎     | 171888/400000 [00:20<00:25, 8781.85it/s] 43%|████▎     | 172771/400000 [00:20<00:25, 8793.86it/s] 43%|████▎     | 173651/400000 [00:20<00:25, 8791.72it/s] 44%|████▎     | 174531/400000 [00:20<00:25, 8761.82it/s] 44%|████▍     | 175408/400000 [00:20<00:25, 8757.32it/s] 44%|████▍     | 176284/400000 [00:20<00:25, 8720.70it/s] 44%|████▍     | 177163/400000 [00:20<00:25, 8739.98it/s] 45%|████▍     | 178038/400000 [00:20<00:25, 8702.17it/s] 45%|████▍     | 178920/400000 [00:20<00:25, 8736.29it/s] 45%|████▍     | 179804/400000 [00:20<00:25, 8764.32it/s] 45%|████▌     | 180691/400000 [00:21<00:24, 8794.82it/s] 45%|████▌     | 181574/400000 [00:21<00:24, 8803.49it/s] 46%|████▌     | 182459/400000 [00:21<00:24, 8814.47it/s] 46%|████▌     | 183341/400000 [00:21<00:25, 8354.03it/s] 46%|████▌     | 184217/400000 [00:21<00:25, 8469.69it/s] 46%|████▋     | 185092/400000 [00:21<00:25, 8549.40it/s] 46%|████▋     | 185965/400000 [00:21<00:24, 8602.11it/s] 47%|████▋     | 186843/400000 [00:21<00:24, 8652.45it/s] 47%|████▋     | 187710/400000 [00:21<00:24, 8527.33it/s] 47%|████▋     | 188565/400000 [00:21<00:24, 8511.32it/s] 47%|████▋     | 189435/400000 [00:22<00:24, 8564.25it/s] 48%|████▊     | 190295/400000 [00:22<00:24, 8572.97it/s] 48%|████▊     | 191153/400000 [00:22<00:24, 8489.19it/s] 48%|████▊     | 192003/400000 [00:22<00:24, 8336.38it/s] 48%|████▊     | 192857/400000 [00:22<00:24, 8396.36it/s] 48%|████▊     | 193714/400000 [00:22<00:24, 8445.47it/s] 49%|████▊     | 194560/400000 [00:22<00:24, 8420.57it/s] 49%|████▉     | 195431/400000 [00:22<00:24, 8503.02it/s] 49%|████▉     | 196301/400000 [00:22<00:23, 8559.05it/s] 49%|████▉     | 197186/400000 [00:22<00:23, 8643.52it/s] 50%|████▉     | 198054/400000 [00:23<00:23, 8654.14it/s] 50%|████▉     | 198928/400000 [00:23<00:23, 8679.63it/s] 50%|████▉     | 199811/400000 [00:23<00:22, 8722.74it/s] 50%|█████     | 200684/400000 [00:23<00:22, 8723.22it/s] 50%|█████     | 201557/400000 [00:23<00:22, 8697.38it/s] 51%|█████     | 202427/400000 [00:23<00:22, 8693.41it/s] 51%|█████     | 203309/400000 [00:23<00:22, 8730.34it/s] 51%|█████     | 204183/400000 [00:23<00:22, 8534.40it/s] 51%|█████▏    | 205038/400000 [00:23<00:23, 8421.68it/s] 51%|█████▏    | 205918/400000 [00:24<00:22, 8529.65it/s] 52%|█████▏    | 206797/400000 [00:24<00:22, 8604.92it/s] 52%|█████▏    | 207664/400000 [00:24<00:22, 8623.76it/s] 52%|█████▏    | 208538/400000 [00:24<00:22, 8657.72it/s] 52%|█████▏    | 209405/400000 [00:24<00:22, 8629.54it/s] 53%|█████▎    | 210281/400000 [00:24<00:21, 8666.49it/s] 53%|█████▎    | 211160/400000 [00:24<00:21, 8701.95it/s] 53%|█████▎    | 212043/400000 [00:24<00:21, 8737.35it/s] 53%|█████▎    | 212922/400000 [00:24<00:21, 8752.79it/s] 53%|█████▎    | 213798/400000 [00:24<00:21, 8702.88it/s] 54%|█████▎    | 214677/400000 [00:25<00:21, 8725.71it/s] 54%|█████▍    | 215550/400000 [00:25<00:21, 8609.51it/s] 54%|█████▍    | 216412/400000 [00:25<00:21, 8514.98it/s] 54%|█████▍    | 217265/400000 [00:25<00:21, 8492.33it/s] 55%|█████▍    | 218136/400000 [00:25<00:21, 8554.18it/s] 55%|█████▍    | 219008/400000 [00:25<00:21, 8603.18it/s] 55%|█████▍    | 219889/400000 [00:25<00:20, 8661.49it/s] 55%|█████▌    | 220769/400000 [00:25<00:20, 8702.02it/s] 55%|█████▌    | 221653/400000 [00:25<00:20, 8741.99it/s] 56%|█████▌    | 222528/400000 [00:25<00:20, 8661.89it/s] 56%|█████▌    | 223409/400000 [00:26<00:20, 8704.19it/s] 56%|█████▌    | 224294/400000 [00:26<00:20, 8747.36it/s] 56%|█████▋    | 225169/400000 [00:26<00:20, 8722.26it/s] 57%|█████▋    | 226042/400000 [00:26<00:19, 8711.97it/s] 57%|█████▋    | 226914/400000 [00:26<00:19, 8671.46it/s] 57%|█████▋    | 227782/400000 [00:26<00:20, 8603.00it/s] 57%|█████▋    | 228643/400000 [00:26<00:20, 8540.21it/s] 57%|█████▋    | 229519/400000 [00:26<00:19, 8604.20it/s] 58%|█████▊    | 230380/400000 [00:26<00:19, 8535.55it/s] 58%|█████▊    | 231234/400000 [00:26<00:19, 8471.13it/s] 58%|█████▊    | 232116/400000 [00:27<00:19, 8570.96it/s] 58%|█████▊    | 232993/400000 [00:27<00:19, 8627.59it/s] 58%|█████▊    | 233879/400000 [00:27<00:19, 8695.44it/s] 59%|█████▊    | 234761/400000 [00:27<00:18, 8732.23it/s] 59%|█████▉    | 235635/400000 [00:27<00:18, 8724.26it/s] 59%|█████▉    | 236508/400000 [00:27<00:18, 8714.96it/s] 59%|█████▉    | 237383/400000 [00:27<00:18, 8724.85it/s] 60%|█████▉    | 238268/400000 [00:27<00:18, 8759.95it/s] 60%|█████▉    | 239152/400000 [00:27<00:18, 8781.61it/s] 60%|██████    | 240031/400000 [00:27<00:18, 8608.66it/s] 60%|██████    | 240913/400000 [00:28<00:18, 8670.06it/s] 60%|██████    | 241796/400000 [00:28<00:18, 8717.20it/s] 61%|██████    | 242669/400000 [00:28<00:18, 8416.77it/s] 61%|██████    | 243514/400000 [00:28<00:18, 8345.10it/s] 61%|██████    | 244356/400000 [00:28<00:18, 8365.02it/s] 61%|██████▏   | 245204/400000 [00:28<00:18, 8398.67it/s] 62%|██████▏   | 246085/400000 [00:28<00:18, 8515.50it/s] 62%|██████▏   | 246968/400000 [00:28<00:17, 8605.79it/s] 62%|██████▏   | 247852/400000 [00:28<00:17, 8674.41it/s] 62%|██████▏   | 248721/400000 [00:28<00:17, 8669.41it/s] 62%|██████▏   | 249603/400000 [00:29<00:17, 8711.27it/s] 63%|██████▎   | 250475/400000 [00:29<00:17, 8691.15it/s] 63%|██████▎   | 251345/400000 [00:29<00:17, 8675.36it/s] 63%|██████▎   | 252213/400000 [00:29<00:17, 8670.14it/s] 63%|██████▎   | 253086/400000 [00:29<00:16, 8685.71it/s] 63%|██████▎   | 253955/400000 [00:29<00:17, 8588.47it/s] 64%|██████▎   | 254826/400000 [00:29<00:16, 8622.23it/s] 64%|██████▍   | 255698/400000 [00:29<00:16, 8651.21it/s] 64%|██████▍   | 256579/400000 [00:29<00:16, 8696.11it/s] 64%|██████▍   | 257449/400000 [00:29<00:16, 8604.56it/s] 65%|██████▍   | 258316/400000 [00:30<00:16, 8622.06it/s] 65%|██████▍   | 259190/400000 [00:30<00:16, 8655.83it/s] 65%|██████▌   | 260064/400000 [00:30<00:16, 8680.74it/s] 65%|██████▌   | 260933/400000 [00:30<00:16, 8467.66it/s] 65%|██████▌   | 261787/400000 [00:30<00:16, 8307.13it/s] 66%|██████▌   | 262646/400000 [00:30<00:16, 8387.98it/s] 66%|██████▌   | 263573/400000 [00:30<00:15, 8632.70it/s] 66%|██████▌   | 264516/400000 [00:30<00:15, 8857.13it/s] 66%|██████▋   | 265495/400000 [00:30<00:14, 9116.78it/s] 67%|██████▋   | 266430/400000 [00:30<00:14, 9182.61it/s] 67%|██████▋   | 267352/400000 [00:31<00:14, 9160.48it/s] 67%|██████▋   | 268287/400000 [00:31<00:14, 9214.20it/s] 67%|██████▋   | 269287/400000 [00:31<00:13, 9435.07it/s] 68%|██████▊   | 270254/400000 [00:31<00:13, 9503.56it/s] 68%|██████▊   | 271207/400000 [00:31<00:13, 9475.97it/s] 68%|██████▊   | 272156/400000 [00:31<00:13, 9334.71it/s] 68%|██████▊   | 273164/400000 [00:31<00:13, 9544.65it/s] 69%|██████▊   | 274190/400000 [00:31<00:12, 9746.79it/s] 69%|██████▉   | 275200/400000 [00:31<00:12, 9849.43it/s] 69%|██████▉   | 276187/400000 [00:32<00:12, 9588.71it/s] 69%|██████▉   | 277149/400000 [00:32<00:12, 9533.24it/s] 70%|██████▉   | 278105/400000 [00:32<00:12, 9405.57it/s] 70%|██████▉   | 279048/400000 [00:32<00:13, 9246.49it/s] 70%|██████▉   | 279994/400000 [00:32<00:12, 9307.46it/s] 70%|███████   | 280927/400000 [00:32<00:12, 9270.35it/s] 70%|███████   | 281909/400000 [00:32<00:12, 9427.07it/s] 71%|███████   | 282854/400000 [00:32<00:12, 9310.13it/s] 71%|███████   | 283787/400000 [00:32<00:12, 9260.23it/s] 71%|███████   | 284714/400000 [00:32<00:12, 9178.41it/s] 71%|███████▏  | 285633/400000 [00:33<00:12, 9161.03it/s] 72%|███████▏  | 286627/400000 [00:33<00:12, 9379.43it/s] 72%|███████▏  | 287656/400000 [00:33<00:11, 9634.51it/s] 72%|███████▏  | 288662/400000 [00:33<00:11, 9757.52it/s] 72%|███████▏  | 289641/400000 [00:33<00:11, 9761.47it/s] 73%|███████▎  | 290619/400000 [00:33<00:11, 9588.19it/s] 73%|███████▎  | 291580/400000 [00:33<00:11, 9521.34it/s] 73%|███████▎  | 292534/400000 [00:33<00:11, 9413.55it/s] 73%|███████▎  | 293477/400000 [00:33<00:11, 9302.86it/s] 74%|███████▎  | 294409/400000 [00:33<00:11, 9289.08it/s] 74%|███████▍  | 295339/400000 [00:34<00:11, 9170.43it/s] 74%|███████▍  | 296257/400000 [00:34<00:11, 9121.20it/s] 74%|███████▍  | 297205/400000 [00:34<00:11, 9224.77it/s] 75%|███████▍  | 298129/400000 [00:34<00:11, 9217.89it/s] 75%|███████▍  | 299111/400000 [00:34<00:10, 9387.71it/s] 75%|███████▌  | 300051/400000 [00:34<00:10, 9386.29it/s] 75%|███████▌  | 300991/400000 [00:34<00:10, 9306.18it/s] 75%|███████▌  | 301946/400000 [00:34<00:10, 9376.33it/s] 76%|███████▌  | 302892/400000 [00:34<00:10, 9400.90it/s] 76%|███████▌  | 303855/400000 [00:34<00:10, 9466.79it/s] 76%|███████▌  | 304853/400000 [00:35<00:09, 9613.75it/s] 76%|███████▋  | 305843/400000 [00:35<00:09, 9697.33it/s] 77%|███████▋  | 306854/400000 [00:35<00:09, 9814.64it/s] 77%|███████▋  | 307859/400000 [00:35<00:09, 9880.87it/s] 77%|███████▋  | 308853/400000 [00:35<00:09, 9896.99it/s] 77%|███████▋  | 309844/400000 [00:35<00:09, 9512.69it/s] 78%|███████▊  | 310799/400000 [00:35<00:09, 9493.73it/s] 78%|███████▊  | 311751/400000 [00:35<00:09, 9471.34it/s] 78%|███████▊  | 312700/400000 [00:35<00:09, 9079.93it/s] 78%|███████▊  | 313624/400000 [00:35<00:09, 9125.48it/s] 79%|███████▊  | 314558/400000 [00:36<00:09, 9180.56it/s] 79%|███████▉  | 315479/400000 [00:36<00:09, 9162.30it/s] 79%|███████▉  | 316406/400000 [00:36<00:09, 9194.16it/s] 79%|███████▉  | 317370/400000 [00:36<00:08, 9322.70it/s] 80%|███████▉  | 318304/400000 [00:36<00:08, 9210.36it/s] 80%|███████▉  | 319227/400000 [00:36<00:08, 9186.04it/s] 80%|████████  | 320251/400000 [00:36<00:08, 9478.41it/s] 80%|████████  | 321202/400000 [00:36<00:08, 9456.15it/s] 81%|████████  | 322167/400000 [00:36<00:08, 9510.56it/s] 81%|████████  | 323180/400000 [00:36<00:07, 9687.73it/s] 81%|████████  | 324170/400000 [00:37<00:07, 9747.82it/s] 81%|████████▏ | 325147/400000 [00:37<00:07, 9536.43it/s] 82%|████████▏ | 326103/400000 [00:37<00:07, 9417.10it/s] 82%|████████▏ | 327047/400000 [00:37<00:08, 9015.60it/s] 82%|████████▏ | 327994/400000 [00:37<00:07, 9145.42it/s] 82%|████████▏ | 328913/400000 [00:37<00:07, 9153.94it/s] 82%|████████▏ | 329909/400000 [00:37<00:07, 9379.92it/s] 83%|████████▎ | 330870/400000 [00:37<00:07, 9445.43it/s] 83%|████████▎ | 331817/400000 [00:37<00:07, 9359.61it/s] 83%|████████▎ | 332759/400000 [00:38<00:07, 9375.34it/s] 83%|████████▎ | 333732/400000 [00:38<00:06, 9478.37it/s] 84%|████████▎ | 334775/400000 [00:38<00:06, 9744.54it/s] 84%|████████▍ | 335753/400000 [00:38<00:06, 9497.30it/s] 84%|████████▍ | 336718/400000 [00:38<00:06, 9541.82it/s] 84%|████████▍ | 337675/400000 [00:38<00:06, 9122.15it/s] 85%|████████▍ | 338593/400000 [00:38<00:06, 9031.78it/s] 85%|████████▍ | 339501/400000 [00:38<00:06, 9034.45it/s] 85%|████████▌ | 340408/400000 [00:38<00:06, 8991.86it/s] 85%|████████▌ | 341367/400000 [00:38<00:06, 9161.94it/s] 86%|████████▌ | 342309/400000 [00:39<00:06, 9235.41it/s] 86%|████████▌ | 343235/400000 [00:39<00:06, 9203.43it/s] 86%|████████▌ | 344163/400000 [00:39<00:06, 9216.38it/s] 86%|████████▋ | 345124/400000 [00:39<00:05, 9330.23it/s] 87%|████████▋ | 346058/400000 [00:39<00:05, 9267.31it/s] 87%|████████▋ | 346986/400000 [00:39<00:05, 9146.83it/s] 87%|████████▋ | 347902/400000 [00:39<00:05, 9030.94it/s] 87%|████████▋ | 348807/400000 [00:39<00:05, 8997.17it/s] 87%|████████▋ | 349716/400000 [00:39<00:05, 9023.75it/s] 88%|████████▊ | 350619/400000 [00:39<00:05, 9014.75it/s] 88%|████████▊ | 351598/400000 [00:40<00:05, 9233.76it/s] 88%|████████▊ | 352523/400000 [00:40<00:05, 9131.13it/s] 88%|████████▊ | 353481/400000 [00:40<00:05, 9147.61it/s] 89%|████████▊ | 354453/400000 [00:40<00:04, 9311.17it/s] 89%|████████▉ | 355405/400000 [00:40<00:04, 9371.40it/s] 89%|████████▉ | 356344/400000 [00:40<00:04, 9297.38it/s] 89%|████████▉ | 357275/400000 [00:40<00:04, 9226.75it/s] 90%|████████▉ | 358227/400000 [00:40<00:04, 9311.79it/s] 90%|████████▉ | 359191/400000 [00:40<00:04, 9407.49it/s] 90%|█████████ | 360179/400000 [00:40<00:04, 9541.65it/s] 90%|█████████ | 361135/400000 [00:41<00:04, 9423.86it/s] 91%|█████████ | 362117/400000 [00:41<00:03, 9537.74it/s] 91%|█████████ | 363072/400000 [00:41<00:03, 9498.30it/s] 91%|█████████ | 364032/400000 [00:41<00:03, 9527.35it/s] 91%|█████████ | 364986/400000 [00:41<00:03, 9478.61it/s] 91%|█████████▏| 365941/400000 [00:41<00:03, 9499.79it/s] 92%|█████████▏| 366892/400000 [00:41<00:03, 9402.66it/s] 92%|█████████▏| 367833/400000 [00:41<00:03, 9256.72it/s] 92%|█████████▏| 368760/400000 [00:41<00:03, 9222.68it/s] 92%|█████████▏| 369761/400000 [00:42<00:03, 9444.76it/s] 93%|█████████▎| 370765/400000 [00:42<00:03, 9613.84it/s] 93%|█████████▎| 371729/400000 [00:42<00:02, 9490.97it/s] 93%|█████████▎| 372680/400000 [00:42<00:02, 9483.22it/s] 93%|█████████▎| 373630/400000 [00:42<00:02, 9462.09it/s] 94%|█████████▎| 374644/400000 [00:42<00:02, 9655.20it/s] 94%|█████████▍| 375612/400000 [00:42<00:02, 9635.12it/s] 94%|█████████▍| 376577/400000 [00:42<00:02, 9582.17it/s] 94%|█████████▍| 377588/400000 [00:42<00:02, 9733.32it/s] 95%|█████████▍| 378563/400000 [00:42<00:02, 9732.32it/s] 95%|█████████▍| 379538/400000 [00:43<00:02, 9730.17it/s] 95%|█████████▌| 380512/400000 [00:43<00:02, 9710.06it/s] 95%|█████████▌| 381484/400000 [00:43<00:01, 9519.81it/s] 96%|█████████▌| 382438/400000 [00:43<00:01, 9518.69it/s] 96%|█████████▌| 383442/400000 [00:43<00:01, 9668.17it/s] 96%|█████████▌| 384433/400000 [00:43<00:01, 9739.05it/s] 96%|█████████▋| 385430/400000 [00:43<00:01, 9807.18it/s] 97%|█████████▋| 386412/400000 [00:43<00:01, 9764.88it/s] 97%|█████████▋| 387390/400000 [00:43<00:01, 9663.45it/s] 97%|█████████▋| 388357/400000 [00:43<00:01, 9608.92it/s] 97%|█████████▋| 389319/400000 [00:44<00:01, 9549.26it/s] 98%|█████████▊| 390275/400000 [00:44<00:01, 9474.01it/s] 98%|█████████▊| 391223/400000 [00:44<00:00, 9405.04it/s] 98%|█████████▊| 392229/400000 [00:44<00:00, 9592.05it/s] 98%|█████████▊| 393239/400000 [00:44<00:00, 9736.43it/s] 99%|█████████▊| 394215/400000 [00:44<00:00, 9726.09it/s] 99%|█████████▉| 395230/400000 [00:44<00:00, 9848.71it/s] 99%|█████████▉| 396216/400000 [00:44<00:00, 9608.83it/s] 99%|█████████▉| 397203/400000 [00:44<00:00, 9682.85it/s]100%|█████████▉| 398173/400000 [00:44<00:00, 9397.16it/s]100%|█████████▉| 399128/400000 [00:45<00:00, 9439.42it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8860.13it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f92dc604940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011071935573190408 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.0109087394232734 	 Accuracy: 69

  model saves at 69% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15612 out of table with 15498 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15612 out of table with 15498 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 10:24:07.666668: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 10:24:07.671048: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 10:24:07.671181: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55aa1350c220 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 10:24:07.671194: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f92ea86c160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8238 - accuracy: 0.4897
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8200 - accuracy: 0.4900
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8455 - accuracy: 0.4883
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8265 - accuracy: 0.4896
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7295 - accuracy: 0.4959
11000/25000 [============>.................] - ETA: 3s - loss: 7.7238 - accuracy: 0.4963
12000/25000 [=============>................] - ETA: 3s - loss: 7.7343 - accuracy: 0.4956
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7539 - accuracy: 0.4943
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7740 - accuracy: 0.4930
15000/25000 [=================>............] - ETA: 2s - loss: 7.7719 - accuracy: 0.4931
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7395 - accuracy: 0.4952
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7262 - accuracy: 0.4961
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7271 - accuracy: 0.4961
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7029 - accuracy: 0.4976
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6866 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f923771a128> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9240dc6128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0121 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.8686 - val_crf_viterbi_accuracy: 0.3467

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

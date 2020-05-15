
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f86b3864fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 11:12:39.913686
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 11:12:39.917863
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 11:12:39.921324
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 11:12:39.924338
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f86bf87c4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356405.9062
Epoch 2/10

1/1 [==============================] - 0s 99ms/step - loss: 256377.0312
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 151531.3438
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 84334.7891
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 47417.4844
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 28645.1934
Epoch 7/10

1/1 [==============================] - 0s 88ms/step - loss: 18881.7168
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 13284.0918
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 9832.8418
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 7555.5278

  #### Inference Need return ypred, ytrue ######################### 
[[-2.37199292e-02  8.29855633e+00  8.22774029e+00  6.38283205e+00
   9.53127670e+00  8.83636951e+00  6.46815014e+00  7.73590326e+00
   8.80431557e+00  7.16235161e+00  8.19585323e+00  8.12606525e+00
   7.05353212e+00  9.94497204e+00  6.88825703e+00  7.20735264e+00
   6.46489429e+00  6.73817253e+00  8.11146164e+00  9.14689732e+00
   9.17371464e+00  6.57242393e+00  7.13484335e+00  8.16192818e+00
   7.11381149e+00  7.33586121e+00  7.75852442e+00  7.13589001e+00
   7.11647129e+00  6.52237606e+00  7.66625309e+00  7.20944357e+00
   8.85056686e+00  6.79682064e+00  6.97863722e+00  6.19362116e+00
   5.89766359e+00  8.83169174e+00  8.42554188e+00  8.13194180e+00
   8.83713341e+00  5.54831457e+00  6.72954369e+00  6.65961361e+00
   8.43936920e+00  8.55478001e+00  7.64520788e+00  7.64383459e+00
   6.36687231e+00  9.23931980e+00  1.02528839e+01  5.80777693e+00
   7.49154377e+00  7.21267080e+00  7.40793657e+00  8.27665615e+00
   9.03367519e+00  6.94742107e+00  7.60012722e+00  7.78026056e+00
  -1.33529752e-01 -7.84481466e-01 -3.08963358e-01 -8.55022192e-01
  -5.74409008e-01  2.27237821e+00  9.58619416e-02  5.50683200e-01
   1.13723040e+00  1.00322992e-01  8.52253675e-01 -1.35847652e+00
  -1.06841695e+00  3.35885018e-01  1.75940096e+00 -8.25708449e-01
  -3.63693327e-01 -1.63434136e+00 -9.52520072e-01 -1.86508226e+00
   2.30707216e+00 -5.75456858e-01 -2.11784154e-01  5.47185838e-01
  -8.55287910e-02 -2.31347609e+00  4.12754714e-02 -5.79654574e-01
  -1.67362058e+00 -6.63125753e-01 -1.31135225e+00 -3.62189114e-02
   1.36482620e+00 -1.19517341e-01 -1.52751803e+00 -3.27883363e-02
  -1.12277222e+00 -1.44357479e+00 -9.47007835e-01 -1.02998286e-01
  -6.20873451e-01  2.31102943e-01 -1.71211207e+00  1.93828297e+00
  -1.65422529e-01  5.29981554e-02 -1.05338264e+00  2.80325115e-01
  -1.60468245e+00 -8.42146814e-01 -2.72137284e-01  7.02366889e-01
   1.61853695e+00  2.12484384e+00 -1.98787892e+00  1.26197442e-01
  -8.21606040e-01 -1.46340430e+00  7.47354925e-02 -1.70033050e+00
  -1.24103546e-01 -3.40391457e-01  4.32652771e-01  2.96220601e-01
  -9.10411537e-01 -2.96943873e-01 -3.97135794e-01  6.22774005e-01
  -4.09674615e-01  1.35315180e+00 -1.58091009e-01  1.29547149e-01
  -7.88835049e-01 -2.66105592e-01  8.59824538e-01 -1.22350192e+00
  -7.48266220e-01 -1.24073168e-02  1.80219424e+00  1.04344881e+00
   3.04122031e-01  6.66524768e-01  1.36147141e-02 -6.97724104e-01
  -6.12199903e-02 -1.77078700e+00  1.31009388e+00  3.18695426e-01
  -1.47612363e-01  2.35925078e-01  1.11426330e+00 -7.30861425e-02
   1.65540886e+00 -6.74579442e-01 -5.76505005e-01 -3.89580905e-01
  -3.97625297e-01 -1.62802577e-01  3.19514871e-01  1.57086337e+00
  -1.42898083e+00 -8.56069207e-01 -1.56625128e+00 -3.70806515e-01
   2.45999813e+00  1.70475006e-01  1.01847339e+00  1.09248507e+00
   8.61004889e-01  1.87541020e+00 -6.93291724e-02 -1.35778308e+00
  -9.83502865e-02  1.44598365e+00 -6.78431809e-01 -3.49690199e-01
  -6.26140177e-01  1.02776337e+00  4.54148650e-02  4.24987674e-02
   2.26873100e-01  8.11737728e+00  7.49571657e+00  7.89982462e+00
   8.69396687e+00  7.08011866e+00  7.69311714e+00  7.29133129e+00
   6.58072472e+00  8.22041607e+00  7.57486200e+00  6.40438747e+00
   9.99030113e+00  6.23544693e+00  7.90356255e+00  7.68605757e+00
   7.75917816e+00  6.95232582e+00  7.77901840e+00  8.73877525e+00
   9.07122517e+00  6.87280560e+00  7.57025862e+00  7.44119072e+00
   7.90243053e+00  8.39467049e+00  8.28588104e+00  8.22433949e+00
   6.62974453e+00  7.16620255e+00  8.16236115e+00  8.81354904e+00
   8.27275848e+00  6.85023355e+00  8.81312084e+00  7.57025719e+00
   6.91403580e+00  5.98488951e+00  7.68995476e+00  6.56715393e+00
   8.18057537e+00  7.18634796e+00  8.35715389e+00  8.60023689e+00
   7.26340866e+00  8.03588104e+00  6.45983315e+00  8.73363400e+00
   8.92964554e+00  8.52813339e+00  7.63174582e+00  9.61415100e+00
   8.83859348e+00  8.41690254e+00  8.03211975e+00  8.69396305e+00
   7.50778818e+00  9.57194233e+00  8.67957687e+00  7.97167969e+00
   2.65598774e-01  5.32620013e-01  4.90120530e-01  1.15936232e+00
   4.59648490e-01  3.68785083e-01  2.02830017e-01  5.49362600e-01
   1.48427629e+00  3.26568723e-01  3.13760400e-01  1.30563676e+00
   3.94078493e-01  1.57227969e+00  8.38901579e-01  8.71556997e-02
   1.19622338e+00  8.86806190e-01  3.97499204e-01  1.10159516e+00
   9.04647052e-01  1.60493064e+00  3.11999702e+00  4.88191009e-01
   1.75494504e+00  1.68506742e+00  5.23074508e-01  5.84055305e-01
   1.87639761e+00  1.55184495e+00  5.56885362e-01  3.84018064e-01
   4.32226896e-01  1.04953957e+00  2.64813757e+00  2.89615250e+00
   6.79561019e-01  9.65968788e-01  1.43270540e+00  3.66999447e-01
   1.47370529e+00  7.62174010e-01  5.02619445e-01  5.45401990e-01
   8.28389108e-01  2.39051723e+00  3.48057747e+00  3.89884591e-01
   1.97918296e-01  8.48655820e-01  2.37349725e+00  1.45825255e+00
   5.69811583e-01  7.33152628e-01  2.94389844e-01  1.55750203e+00
   3.55501473e-01  9.70325351e-01  9.93270397e-01  1.38697791e+00
   1.53054762e+00  3.61514926e-01  6.05234921e-01  2.34027863e+00
   8.70195329e-01  6.57073617e-01  3.72426510e-01  2.04163551e+00
   2.30649781e+00  8.39743614e-01  1.11540043e+00  4.38768148e-01
   8.19709778e-01  1.80542397e+00  7.27005124e-01  6.11528218e-01
   1.00948513e-01  2.72491026e+00  1.92384112e+00  4.14062679e-01
   4.50478196e-01  5.78595042e-01  7.95225263e-01  9.00102854e-01
   3.07227278e+00  1.53953338e+00  2.39216566e-01  1.06293011e+00
   1.17904437e+00  1.44031191e+00  1.57309616e+00  2.31162882e+00
   8.21274281e-01  3.72531354e-01  4.31158483e-01  8.92940164e-01
   3.99215579e-01  6.06959105e-01  7.77820885e-01  5.47418892e-01
   1.28392446e+00  1.32433057e-01  1.41022742e+00  1.36623359e+00
   1.37050581e+00  1.03347683e+00  3.74778986e-01  2.45668411e-01
   3.25841725e-01  8.23467791e-01  2.76407361e-01  4.48411226e-01
   2.61025858e+00  1.85329008e+00  2.10162044e-01  2.25032187e+00
   1.02417529e-01  7.21136332e-01  2.47737312e+00  1.50739062e+00
   1.26099129e+01 -6.72726345e+00 -2.83917069e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 11:12:48.514650
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.7006
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 11:12:48.518490
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8992.32
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 11:12:48.521815
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.7399
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 11:12:48.525002
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -804.325
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140216174457856
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140214947074624
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140214947075128
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140214947075632
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140214947076136
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140214947076640

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f86bb6fdef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.623981
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.577976
grad_step = 000002, loss = 0.542001
grad_step = 000003, loss = 0.502304
grad_step = 000004, loss = 0.460057
grad_step = 000005, loss = 0.422400
grad_step = 000006, loss = 0.403452
grad_step = 000007, loss = 0.400209
grad_step = 000008, loss = 0.382726
grad_step = 000009, loss = 0.356820
grad_step = 000010, loss = 0.337890
grad_step = 000011, loss = 0.326799
grad_step = 000012, loss = 0.318212
grad_step = 000013, loss = 0.307828
grad_step = 000014, loss = 0.294459
grad_step = 000015, loss = 0.279357
grad_step = 000016, loss = 0.264973
grad_step = 000017, loss = 0.253187
grad_step = 000018, loss = 0.243511
grad_step = 000019, loss = 0.233587
grad_step = 000020, loss = 0.221827
grad_step = 000021, loss = 0.209286
grad_step = 000022, loss = 0.198059
grad_step = 000023, loss = 0.188542
grad_step = 000024, loss = 0.179668
grad_step = 000025, loss = 0.170519
grad_step = 000026, loss = 0.160944
grad_step = 000027, loss = 0.151326
grad_step = 000028, loss = 0.142375
grad_step = 000029, loss = 0.134384
grad_step = 000030, loss = 0.126656
grad_step = 000031, loss = 0.118675
grad_step = 000032, loss = 0.110962
grad_step = 000033, loss = 0.104141
grad_step = 000034, loss = 0.098009
grad_step = 000035, loss = 0.091951
grad_step = 000036, loss = 0.085822
grad_step = 000037, loss = 0.079943
grad_step = 000038, loss = 0.074638
grad_step = 000039, loss = 0.069792
grad_step = 000040, loss = 0.065047
grad_step = 000041, loss = 0.060507
grad_step = 000042, loss = 0.056373
grad_step = 000043, loss = 0.052585
grad_step = 000044, loss = 0.048971
grad_step = 000045, loss = 0.045517
grad_step = 000046, loss = 0.042331
grad_step = 000047, loss = 0.039449
grad_step = 000048, loss = 0.036796
grad_step = 000049, loss = 0.034233
grad_step = 000050, loss = 0.031817
grad_step = 000051, loss = 0.029650
grad_step = 000052, loss = 0.027651
grad_step = 000053, loss = 0.025717
grad_step = 000054, loss = 0.023925
grad_step = 000055, loss = 0.022359
grad_step = 000056, loss = 0.020884
grad_step = 000057, loss = 0.019406
grad_step = 000058, loss = 0.018029
grad_step = 000059, loss = 0.016832
grad_step = 000060, loss = 0.015690
grad_step = 000061, loss = 0.014570
grad_step = 000062, loss = 0.013560
grad_step = 000063, loss = 0.012656
grad_step = 000064, loss = 0.011769
grad_step = 000065, loss = 0.010906
grad_step = 000066, loss = 0.010137
grad_step = 000067, loss = 0.009443
grad_step = 000068, loss = 0.008770
grad_step = 000069, loss = 0.008137
grad_step = 000070, loss = 0.007579
grad_step = 000071, loss = 0.007057
grad_step = 000072, loss = 0.006556
grad_step = 000073, loss = 0.006105
grad_step = 000074, loss = 0.005709
grad_step = 000075, loss = 0.005332
grad_step = 000076, loss = 0.004982
grad_step = 000077, loss = 0.004676
grad_step = 000078, loss = 0.004398
grad_step = 000079, loss = 0.004139
grad_step = 000080, loss = 0.003914
grad_step = 000081, loss = 0.003712
grad_step = 000082, loss = 0.003522
grad_step = 000083, loss = 0.003357
grad_step = 000084, loss = 0.003214
grad_step = 000085, loss = 0.003080
grad_step = 000086, loss = 0.002963
grad_step = 000087, loss = 0.002862
grad_step = 000088, loss = 0.002769
grad_step = 000089, loss = 0.002683
grad_step = 000090, loss = 0.002612
grad_step = 000091, loss = 0.002549
grad_step = 000092, loss = 0.002490
grad_step = 000093, loss = 0.002439
grad_step = 000094, loss = 0.002395
grad_step = 000095, loss = 0.002354
grad_step = 000096, loss = 0.002318
grad_step = 000097, loss = 0.002287
grad_step = 000098, loss = 0.002258
grad_step = 000099, loss = 0.002233
grad_step = 000100, loss = 0.002211
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002191
grad_step = 000102, loss = 0.002173
grad_step = 000103, loss = 0.002158
grad_step = 000104, loss = 0.002143
grad_step = 000105, loss = 0.002130
grad_step = 000106, loss = 0.002120
grad_step = 000107, loss = 0.002110
grad_step = 000108, loss = 0.002101
grad_step = 000109, loss = 0.002093
grad_step = 000110, loss = 0.002086
grad_step = 000111, loss = 0.002080
grad_step = 000112, loss = 0.002075
grad_step = 000113, loss = 0.002070
grad_step = 000114, loss = 0.002065
grad_step = 000115, loss = 0.002061
grad_step = 000116, loss = 0.002057
grad_step = 000117, loss = 0.002054
grad_step = 000118, loss = 0.002051
grad_step = 000119, loss = 0.002048
grad_step = 000120, loss = 0.002045
grad_step = 000121, loss = 0.002042
grad_step = 000122, loss = 0.002039
grad_step = 000123, loss = 0.002036
grad_step = 000124, loss = 0.002034
grad_step = 000125, loss = 0.002031
grad_step = 000126, loss = 0.002028
grad_step = 000127, loss = 0.002026
grad_step = 000128, loss = 0.002023
grad_step = 000129, loss = 0.002020
grad_step = 000130, loss = 0.002017
grad_step = 000131, loss = 0.002014
grad_step = 000132, loss = 0.002012
grad_step = 000133, loss = 0.002009
grad_step = 000134, loss = 0.002006
grad_step = 000135, loss = 0.002003
grad_step = 000136, loss = 0.002000
grad_step = 000137, loss = 0.001998
grad_step = 000138, loss = 0.001995
grad_step = 000139, loss = 0.001992
grad_step = 000140, loss = 0.001989
grad_step = 000141, loss = 0.001986
grad_step = 000142, loss = 0.001983
grad_step = 000143, loss = 0.001980
grad_step = 000144, loss = 0.001978
grad_step = 000145, loss = 0.001975
grad_step = 000146, loss = 0.001972
grad_step = 000147, loss = 0.001969
grad_step = 000148, loss = 0.001966
grad_step = 000149, loss = 0.001963
grad_step = 000150, loss = 0.001961
grad_step = 000151, loss = 0.001958
grad_step = 000152, loss = 0.001956
grad_step = 000153, loss = 0.001953
grad_step = 000154, loss = 0.001950
grad_step = 000155, loss = 0.001947
grad_step = 000156, loss = 0.001944
grad_step = 000157, loss = 0.001942
grad_step = 000158, loss = 0.001940
grad_step = 000159, loss = 0.001937
grad_step = 000160, loss = 0.001935
grad_step = 000161, loss = 0.001932
grad_step = 000162, loss = 0.001929
grad_step = 000163, loss = 0.001927
grad_step = 000164, loss = 0.001925
grad_step = 000165, loss = 0.001923
grad_step = 000166, loss = 0.001920
grad_step = 000167, loss = 0.001918
grad_step = 000168, loss = 0.001916
grad_step = 000169, loss = 0.001914
grad_step = 000170, loss = 0.001912
grad_step = 000171, loss = 0.001910
grad_step = 000172, loss = 0.001908
grad_step = 000173, loss = 0.001906
grad_step = 000174, loss = 0.001903
grad_step = 000175, loss = 0.001901
grad_step = 000176, loss = 0.001899
grad_step = 000177, loss = 0.001897
grad_step = 000178, loss = 0.001895
grad_step = 000179, loss = 0.001894
grad_step = 000180, loss = 0.001892
grad_step = 000181, loss = 0.001891
grad_step = 000182, loss = 0.001889
grad_step = 000183, loss = 0.001889
grad_step = 000184, loss = 0.001887
grad_step = 000185, loss = 0.001885
grad_step = 000186, loss = 0.001882
grad_step = 000187, loss = 0.001879
grad_step = 000188, loss = 0.001878
grad_step = 000189, loss = 0.001877
grad_step = 000190, loss = 0.001876
grad_step = 000191, loss = 0.001874
grad_step = 000192, loss = 0.001871
grad_step = 000193, loss = 0.001869
grad_step = 000194, loss = 0.001867
grad_step = 000195, loss = 0.001866
grad_step = 000196, loss = 0.001865
grad_step = 000197, loss = 0.001864
grad_step = 000198, loss = 0.001862
grad_step = 000199, loss = 0.001860
grad_step = 000200, loss = 0.001858
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001856
grad_step = 000202, loss = 0.001854
grad_step = 000203, loss = 0.001852
grad_step = 000204, loss = 0.001851
grad_step = 000205, loss = 0.001850
grad_step = 000206, loss = 0.001848
grad_step = 000207, loss = 0.001847
grad_step = 000208, loss = 0.001846
grad_step = 000209, loss = 0.001845
grad_step = 000210, loss = 0.001843
grad_step = 000211, loss = 0.001842
grad_step = 000212, loss = 0.001840
grad_step = 000213, loss = 0.001838
grad_step = 000214, loss = 0.001835
grad_step = 000215, loss = 0.001833
grad_step = 000216, loss = 0.001831
grad_step = 000217, loss = 0.001829
grad_step = 000218, loss = 0.001828
grad_step = 000219, loss = 0.001826
grad_step = 000220, loss = 0.001825
grad_step = 000221, loss = 0.001824
grad_step = 000222, loss = 0.001823
grad_step = 000223, loss = 0.001822
grad_step = 000224, loss = 0.001820
grad_step = 000225, loss = 0.001819
grad_step = 000226, loss = 0.001817
grad_step = 000227, loss = 0.001815
grad_step = 000228, loss = 0.001812
grad_step = 000229, loss = 0.001810
grad_step = 000230, loss = 0.001807
grad_step = 000231, loss = 0.001805
grad_step = 000232, loss = 0.001803
grad_step = 000233, loss = 0.001801
grad_step = 000234, loss = 0.001800
grad_step = 000235, loss = 0.001798
grad_step = 000236, loss = 0.001797
grad_step = 000237, loss = 0.001796
grad_step = 000238, loss = 0.001794
grad_step = 000239, loss = 0.001794
grad_step = 000240, loss = 0.001793
grad_step = 000241, loss = 0.001793
grad_step = 000242, loss = 0.001792
grad_step = 000243, loss = 0.001791
grad_step = 000244, loss = 0.001788
grad_step = 000245, loss = 0.001785
grad_step = 000246, loss = 0.001781
grad_step = 000247, loss = 0.001777
grad_step = 000248, loss = 0.001774
grad_step = 000249, loss = 0.001772
grad_step = 000250, loss = 0.001771
grad_step = 000251, loss = 0.001770
grad_step = 000252, loss = 0.001770
grad_step = 000253, loss = 0.001769
grad_step = 000254, loss = 0.001769
grad_step = 000255, loss = 0.001767
grad_step = 000256, loss = 0.001765
grad_step = 000257, loss = 0.001762
grad_step = 000258, loss = 0.001758
grad_step = 000259, loss = 0.001754
grad_step = 000260, loss = 0.001751
grad_step = 000261, loss = 0.001749
grad_step = 000262, loss = 0.001747
grad_step = 000263, loss = 0.001745
grad_step = 000264, loss = 0.001744
grad_step = 000265, loss = 0.001743
grad_step = 000266, loss = 0.001742
grad_step = 000267, loss = 0.001741
grad_step = 000268, loss = 0.001740
grad_step = 000269, loss = 0.001739
grad_step = 000270, loss = 0.001738
grad_step = 000271, loss = 0.001736
grad_step = 000272, loss = 0.001733
grad_step = 000273, loss = 0.001730
grad_step = 000274, loss = 0.001728
grad_step = 000275, loss = 0.001727
grad_step = 000276, loss = 0.001724
grad_step = 000277, loss = 0.001720
grad_step = 000278, loss = 0.001716
grad_step = 000279, loss = 0.001712
grad_step = 000280, loss = 0.001710
grad_step = 000281, loss = 0.001707
grad_step = 000282, loss = 0.001704
grad_step = 000283, loss = 0.001701
grad_step = 000284, loss = 0.001699
grad_step = 000285, loss = 0.001698
grad_step = 000286, loss = 0.001697
grad_step = 000287, loss = 0.001696
grad_step = 000288, loss = 0.001696
grad_step = 000289, loss = 0.001696
grad_step = 000290, loss = 0.001699
grad_step = 000291, loss = 0.001702
grad_step = 000292, loss = 0.001703
grad_step = 000293, loss = 0.001697
grad_step = 000294, loss = 0.001686
grad_step = 000295, loss = 0.001675
grad_step = 000296, loss = 0.001669
grad_step = 000297, loss = 0.001665
grad_step = 000298, loss = 0.001664
grad_step = 000299, loss = 0.001665
grad_step = 000300, loss = 0.001663
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001656
grad_step = 000302, loss = 0.001647
grad_step = 000303, loss = 0.001642
grad_step = 000304, loss = 0.001642
grad_step = 000305, loss = 0.001641
grad_step = 000306, loss = 0.001637
grad_step = 000307, loss = 0.001631
grad_step = 000308, loss = 0.001626
grad_step = 000309, loss = 0.001624
grad_step = 000310, loss = 0.001625
grad_step = 000311, loss = 0.001623
grad_step = 000312, loss = 0.001618
grad_step = 000313, loss = 0.001611
grad_step = 000314, loss = 0.001607
grad_step = 000315, loss = 0.001605
grad_step = 000316, loss = 0.001603
grad_step = 000317, loss = 0.001600
grad_step = 000318, loss = 0.001596
grad_step = 000319, loss = 0.001592
grad_step = 000320, loss = 0.001589
grad_step = 000321, loss = 0.001589
grad_step = 000322, loss = 0.001590
grad_step = 000323, loss = 0.001595
grad_step = 000324, loss = 0.001601
grad_step = 000325, loss = 0.001615
grad_step = 000326, loss = 0.001614
grad_step = 000327, loss = 0.001608
grad_step = 000328, loss = 0.001584
grad_step = 000329, loss = 0.001578
grad_step = 000330, loss = 0.001587
grad_step = 000331, loss = 0.001579
grad_step = 000332, loss = 0.001563
grad_step = 000333, loss = 0.001557
grad_step = 000334, loss = 0.001565
grad_step = 000335, loss = 0.001571
grad_step = 000336, loss = 0.001563
grad_step = 000337, loss = 0.001552
grad_step = 000338, loss = 0.001549
grad_step = 000339, loss = 0.001553
grad_step = 000340, loss = 0.001560
grad_step = 000341, loss = 0.001561
grad_step = 000342, loss = 0.001561
grad_step = 000343, loss = 0.001564
grad_step = 000344, loss = 0.001578
grad_step = 000345, loss = 0.001595
grad_step = 000346, loss = 0.001613
grad_step = 000347, loss = 0.001596
grad_step = 000348, loss = 0.001564
grad_step = 000349, loss = 0.001535
grad_step = 000350, loss = 0.001529
grad_step = 000351, loss = 0.001537
grad_step = 000352, loss = 0.001538
grad_step = 000353, loss = 0.001533
grad_step = 000354, loss = 0.001531
grad_step = 000355, loss = 0.001535
grad_step = 000356, loss = 0.001534
grad_step = 000357, loss = 0.001521
grad_step = 000358, loss = 0.001506
grad_step = 000359, loss = 0.001500
grad_step = 000360, loss = 0.001504
grad_step = 000361, loss = 0.001505
grad_step = 000362, loss = 0.001500
grad_step = 000363, loss = 0.001492
grad_step = 000364, loss = 0.001492
grad_step = 000365, loss = 0.001502
grad_step = 000366, loss = 0.001523
grad_step = 000367, loss = 0.001572
grad_step = 000368, loss = 0.001670
grad_step = 000369, loss = 0.001833
grad_step = 000370, loss = 0.001932
grad_step = 000371, loss = 0.001830
grad_step = 000372, loss = 0.001599
grad_step = 000373, loss = 0.001503
grad_step = 000374, loss = 0.001638
grad_step = 000375, loss = 0.001725
grad_step = 000376, loss = 0.001570
grad_step = 000377, loss = 0.001476
grad_step = 000378, loss = 0.001571
grad_step = 000379, loss = 0.001633
grad_step = 000380, loss = 0.001551
grad_step = 000381, loss = 0.001477
grad_step = 000382, loss = 0.001506
grad_step = 000383, loss = 0.001553
grad_step = 000384, loss = 0.001542
grad_step = 000385, loss = 0.001481
grad_step = 000386, loss = 0.001468
grad_step = 000387, loss = 0.001516
grad_step = 000388, loss = 0.001526
grad_step = 000389, loss = 0.001476
grad_step = 000390, loss = 0.001458
grad_step = 000391, loss = 0.001481
grad_step = 000392, loss = 0.001496
grad_step = 000393, loss = 0.001476
grad_step = 000394, loss = 0.001459
grad_step = 000395, loss = 0.001457
grad_step = 000396, loss = 0.001460
grad_step = 000397, loss = 0.001466
grad_step = 000398, loss = 0.001463
grad_step = 000399, loss = 0.001448
grad_step = 000400, loss = 0.001439
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001446
grad_step = 000402, loss = 0.001454
grad_step = 000403, loss = 0.001449
grad_step = 000404, loss = 0.001436
grad_step = 000405, loss = 0.001433
grad_step = 000406, loss = 0.001435
grad_step = 000407, loss = 0.001433
grad_step = 000408, loss = 0.001432
grad_step = 000409, loss = 0.001433
grad_step = 000410, loss = 0.001432
grad_step = 000411, loss = 0.001426
grad_step = 000412, loss = 0.001420
grad_step = 000413, loss = 0.001418
grad_step = 000414, loss = 0.001418
grad_step = 000415, loss = 0.001418
grad_step = 000416, loss = 0.001416
grad_step = 000417, loss = 0.001416
grad_step = 000418, loss = 0.001416
grad_step = 000419, loss = 0.001415
grad_step = 000420, loss = 0.001412
grad_step = 000421, loss = 0.001411
grad_step = 000422, loss = 0.001410
grad_step = 000423, loss = 0.001410
grad_step = 000424, loss = 0.001411
grad_step = 000425, loss = 0.001413
grad_step = 000426, loss = 0.001419
grad_step = 000427, loss = 0.001431
grad_step = 000428, loss = 0.001455
grad_step = 000429, loss = 0.001492
grad_step = 000430, loss = 0.001561
grad_step = 000431, loss = 0.001642
grad_step = 000432, loss = 0.001728
grad_step = 000433, loss = 0.001705
grad_step = 000434, loss = 0.001594
grad_step = 000435, loss = 0.001451
grad_step = 000436, loss = 0.001396
grad_step = 000437, loss = 0.001444
grad_step = 000438, loss = 0.001522
grad_step = 000439, loss = 0.001559
grad_step = 000440, loss = 0.001508
grad_step = 000441, loss = 0.001423
grad_step = 000442, loss = 0.001379
grad_step = 000443, loss = 0.001407
grad_step = 000444, loss = 0.001459
grad_step = 000445, loss = 0.001470
grad_step = 000446, loss = 0.001432
grad_step = 000447, loss = 0.001388
grad_step = 000448, loss = 0.001376
grad_step = 000449, loss = 0.001390
grad_step = 000450, loss = 0.001408
grad_step = 000451, loss = 0.001413
grad_step = 000452, loss = 0.001400
grad_step = 000453, loss = 0.001378
grad_step = 000454, loss = 0.001363
grad_step = 000455, loss = 0.001365
grad_step = 000456, loss = 0.001378
grad_step = 000457, loss = 0.001387
grad_step = 000458, loss = 0.001383
grad_step = 000459, loss = 0.001369
grad_step = 000460, loss = 0.001357
grad_step = 000461, loss = 0.001353
grad_step = 000462, loss = 0.001354
grad_step = 000463, loss = 0.001358
grad_step = 000464, loss = 0.001362
grad_step = 000465, loss = 0.001364
grad_step = 000466, loss = 0.001361
grad_step = 000467, loss = 0.001355
grad_step = 000468, loss = 0.001348
grad_step = 000469, loss = 0.001342
grad_step = 000470, loss = 0.001339
grad_step = 000471, loss = 0.001337
grad_step = 000472, loss = 0.001335
grad_step = 000473, loss = 0.001334
grad_step = 000474, loss = 0.001334
grad_step = 000475, loss = 0.001336
grad_step = 000476, loss = 0.001341
grad_step = 000477, loss = 0.001349
grad_step = 000478, loss = 0.001365
grad_step = 000479, loss = 0.001389
grad_step = 000480, loss = 0.001433
grad_step = 000481, loss = 0.001482
grad_step = 000482, loss = 0.001548
grad_step = 000483, loss = 0.001561
grad_step = 000484, loss = 0.001533
grad_step = 000485, loss = 0.001435
grad_step = 000486, loss = 0.001350
grad_step = 000487, loss = 0.001334
grad_step = 000488, loss = 0.001373
grad_step = 000489, loss = 0.001406
grad_step = 000490, loss = 0.001399
grad_step = 000491, loss = 0.001367
grad_step = 000492, loss = 0.001334
grad_step = 000493, loss = 0.001320
grad_step = 000494, loss = 0.001333
grad_step = 000495, loss = 0.001356
grad_step = 000496, loss = 0.001375
grad_step = 000497, loss = 0.001382
grad_step = 000498, loss = 0.001374
grad_step = 000499, loss = 0.001350
grad_step = 000500, loss = 0.001328
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001312
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

  date_run                              2020-05-15 11:13:07.457747
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.220034
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 11:13:07.463551
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.114261
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 11:13:07.471527
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13228
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 11:13:07.476744
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.736235
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
0   2020-05-15 11:12:39.913686  ...    mean_absolute_error
1   2020-05-15 11:12:39.917863  ...     mean_squared_error
2   2020-05-15 11:12:39.921324  ...  median_absolute_error
3   2020-05-15 11:12:39.924338  ...               r2_score
4   2020-05-15 11:12:48.514650  ...    mean_absolute_error
5   2020-05-15 11:12:48.518490  ...     mean_squared_error
6   2020-05-15 11:12:48.521815  ...  median_absolute_error
7   2020-05-15 11:12:48.525002  ...               r2_score
8   2020-05-15 11:13:07.457747  ...    mean_absolute_error
9   2020-05-15 11:13:07.463551  ...     mean_squared_error
10  2020-05-15 11:13:07.471527  ...  median_absolute_error
11  2020-05-15 11:13:07.476744  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44d5db0ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|████      | 3997696/9912422 [00:00<00:00, 39844098.97it/s]9920512it [00:00, 35597904.63it/s]                             
0it [00:00, ?it/s]32768it [00:00, 604765.26it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478671.22it/s]1654784it [00:00, 11547858.53it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207452.52it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4488769e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4487d9b0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f448552b4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4487cf00b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4488769e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4485516be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f448552b4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4487caf6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4488769e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44d5dbba58> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fae26fac208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=cf3c13b25b5d45bbad4a9b9ce2402100686897b878fbd36df94a66665ca8a91e
  Stored in directory: /tmp/pip-ephem-wheel-cache-to6vmncu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fadbeda76d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2768896/17464789 [===>..........................] - ETA: 0s
 9912320/17464789 [================>.............] - ETA: 0s
16539648/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 11:14:33.763360: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 11:14:33.768087: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-15 11:14:33.768243: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559ba537a1c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 11:14:33.768257: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3216 - accuracy: 0.5225 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4315 - accuracy: 0.5153
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4941 - accuracy: 0.5113
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5878 - accuracy: 0.5051
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5555 - accuracy: 0.5073
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5525 - accuracy: 0.5074
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5700 - accuracy: 0.5063
11000/25000 [============>.................] - ETA: 3s - loss: 7.5830 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 3s - loss: 7.6015 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6159 - accuracy: 0.5033
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6316 - accuracy: 0.5023
15000/25000 [=================>............] - ETA: 2s - loss: 7.6605 - accuracy: 0.5004
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6251 - accuracy: 0.5027
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6255 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6314 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 11:14:47.020538
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 11:14:47.020538  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:34:45, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:14:18, 18.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:19:14, 25.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:32:00, 36.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:33:44, 52.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.44M/862M [00:01<3:10:23, 74.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:12:50, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:32:36, 152kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.6M/862M [00:01<1:04:35, 217kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:01<45:07, 309kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<31:30, 440kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:01<22:03, 626kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:01<15:26, 889kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<10:51, 1.26MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<07:39, 1.77MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<05:27, 2.48MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.8M/862M [00:02<05:07, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<05:29, 2.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:07, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<04:45, 2.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:05<03:29, 3.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<11:53, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<10:02, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<07:22, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:08<07:47, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<08:34, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<06:39, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<04:50, 2.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:10<09:02, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<07:26, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<05:29, 2.40MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:11<04:00, 3.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<55:47, 236kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<40:22, 325kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:13<28:28, 460kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<23:00, 568kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<17:11, 760kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<12:17, 1.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:15<08:45, 1.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<55:21, 235kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<39:50, 326kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<28:09, 461kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:17<19:49, 653kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<1:06:02, 196kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:18<47:29, 272kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:18<33:27, 386kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<26:25, 487kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<19:48, 649kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<14:07, 909kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<12:54, 991kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<10:20, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:22<07:30, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<08:17, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<08:24, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<06:31, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:41, 2.70MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<21:45, 582kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<16:20, 775kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<11:42, 1.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<08:20, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<36:15, 347kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<27:56, 451kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<20:06, 626kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<14:13, 882kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<15:06, 829kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<11:51, 1.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<08:36, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<06:52, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<8:03:13, 25.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<5:38:09, 36.8kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<3:58:05, 52.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:49:30, 73.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:59:09, 104kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<1:23:17, 148kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:04:54, 190kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<46:45, 264kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<32:56, 373kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<25:44, 476kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<20:30, 598kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<14:58, 817kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<10:34, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<54:24, 224kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<39:18, 310kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<27:46, 438kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<22:13, 545kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<17:55, 676kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<13:04, 926kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<09:17, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:04, 922kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:24, 1.16MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<07:34, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:06, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:14, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:17, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:32, 2.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<10:21, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:28, 1.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:13, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<07:07, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:12, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:38, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:01, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:37, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:09, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:44, 3.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<09:53, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<08:08, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:59, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:54, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:01, 1.93MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:28, 2.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:50, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:15, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:58, 2.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:30, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:00, 2.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<03:47, 3.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:21, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:05, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<04:45, 2.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:31, 3.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:18, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:34, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:11, 2.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:34, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:13, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:51, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:31, 3.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:13, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:36, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:36, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:31, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:55, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:25, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:19, 2.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<6:44:52, 27.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<4:43:19, 39.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<3:19:28, 55.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:22:27, 77.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:40:19, 110kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<1:10:06, 157kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<55:12, 199kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<39:52, 275kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<28:09, 388kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<21:59, 496kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<16:33, 658kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<11:51, 917kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:45, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:44, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:17, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<05:12, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<14:46, 729kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:27, 939kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<08:13, 1.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<08:16, 1.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:57, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:06, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:58, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:17, 2.01MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:55, 2.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<05:13, 2.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:48, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:32, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:16, 3.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<12:58, 808kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<10:10, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:22, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<07:34, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<06:23, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:42, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:06, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:47, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:00, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:26, 2.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:48, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:26, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:19, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:39, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<03:14, 3.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:35, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:16, 1.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:08, 2.43MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<02:59, 3.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<13:39, 733kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<10:35, 944kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<07:39, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:40, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:23, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:41, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<04:04, 2.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<38:27, 257kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<27:56, 353kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<19:46, 498kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<14:29, 677kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<6:16:33, 26.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<4:23:25, 37.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<3:05:15, 52.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<2:12:09, 73.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:33:01, 105kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<1:04:59, 149kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<50:55, 190kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<36:42, 263kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<25:54, 372kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<20:08, 477kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<16:08, 595kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<11:44, 816kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<08:17, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<15:51, 601kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:05, 788kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<08:41, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:15, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:43, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:48, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:13, 2.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:59, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:09, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:51, 2.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:51, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:20, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:16, 2.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:28, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:04, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:04, 2.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:19, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:54, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:50, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:47, 3.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<08:45, 1.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<07:03, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:09, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:44, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:51, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:15, 2.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:15, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<06:42, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:53, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:29, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:39, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:24, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:30, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:03, 2.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:11, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:43, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:40, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<10:43, 812kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<08:24, 1.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<06:03, 1.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:15, 1.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:09, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<04:44, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:41, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:09, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:07, 2.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:48, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<5:16:02, 27.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<3:41:02, 38.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<2:35:22, 54.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<1:50:47, 76.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<1:17:55, 108kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<54:23, 154kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<43:51, 191kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<31:34, 265kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<22:13, 376kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<17:20, 479kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<13:50, 600kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<10:05, 822kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:22, 984kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:42, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:53, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:17, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:32, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:21, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:48, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:52, 2.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:54, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:32, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:40, 2.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:45, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:15, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:23, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<02:26, 3.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<58:39, 135kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<41:49, 189kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<29:23, 268kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<22:18, 351kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<17:12, 455kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<12:22, 631kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<08:43, 892kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<10:28, 741kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<08:08, 953kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:52, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<05:54, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<05:42, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:22, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:17, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:47, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:50, 2.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:44, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:23, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:31, 2.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:32, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:00, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:17, 3.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<7:10:26, 17.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<5:01:49, 24.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<3:30:44, 35.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<2:28:30, 49.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<1:44:30, 70.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<1:13:04, 100kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<51:00, 143kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:03:10, 115kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<44:55, 162kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<31:31, 230kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<23:39, 305kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<17:17, 417kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<12:12, 588kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<09:02, 792kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<4:41:17, 25.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<3:16:37, 36.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<2:17:59, 51.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:38:03, 72.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<1:08:51, 103kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<48:01, 146kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<37:12, 189kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<26:45, 262kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<18:50, 371kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<14:44, 471kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<11:44, 591kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:33, 810kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<07:03, 974kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:39, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:24, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:26, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:49, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:48, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:32, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:09, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:22, 2.83MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:14, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:38, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:53, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:04, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:49, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:08, 3.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:47, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:06, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:59, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<03:11, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:33, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<01:53, 3.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:47, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:19, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:27, 2.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:11, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:52, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:08, 2.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:59, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:43, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:03, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:54, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:40, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:00, 3.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:51, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:14, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:34, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<01:51, 3.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<45:04, 134kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<32:08, 188kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<22:33, 267kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<17:06, 350kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<13:11, 454kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<09:29, 630kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<06:40, 890kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<10:37, 558kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<3:27:15, 28.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<2:24:48, 40.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<1:40:51, 58.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<1:13:44, 79.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<53:02, 110kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<37:24, 156kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<26:04, 223kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<22:25, 258kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<16:14, 356kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<11:29, 502kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<09:13, 621kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<07:36, 751kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<05:36, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<03:57, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<49:32, 114kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<35:13, 160kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<24:41, 228kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<18:28, 302kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<13:29, 414kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<09:32, 582kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<07:56, 695kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<06:06, 902kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<04:23, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:21, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:09, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:08, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:15, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:08, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:27, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:32, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:01, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:33, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:53, 2.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:23, 3.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<20:46, 252kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<15:37, 336kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<11:08, 469kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<07:49, 664kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<07:30, 689kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<05:46, 895kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<04:08, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:04, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:53, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:56, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:15, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:29, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:33, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:56, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:04, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:24, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:43, 2.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<18:53, 259kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<13:37, 359kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<09:38, 506kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:08<06:44, 717kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<24:11, 200kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<17:24, 277kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<12:13, 393kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<09:37, 495kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<07:13, 659kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<05:07, 923kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:41, 1.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<04:14, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:11, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:14<02:16, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<4:27:55, 17.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<3:07:45, 24.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<2:10:52, 35.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<1:31:20, 50.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<3:58:24, 19.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<2:46:12, 27.4kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<1:56:01, 38.8kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<1:22:07, 54.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<57:31, 77.9kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<40:02, 111kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<29:44, 149kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<21:14, 208kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<14:53, 295kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<11:21, 384kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<08:23, 519kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:55, 730kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:08, 834kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:01, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:54, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:01, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:58, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:17, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:15, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:00, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:29, 2.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:59, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:48, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:21, 2.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:53, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:44, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:18, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:50, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:37, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:13, 3.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:46, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:02, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:35, 2.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:09, 3.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:58, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:28, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:08, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:15, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:44, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:15, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:50, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:43, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:02, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:47, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:19, 2.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:35, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:11, 2.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:38, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:51, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:28, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:34, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:26, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:05, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:32, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:24, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:03, 3.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:00, 3.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<1:58:37, 27.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<1:22:40, 39.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<57:03, 56.2kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<49:14, 65.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<35:11, 91.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<24:42, 129kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<17:06, 184kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<13:55, 225kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<10:03, 311kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<07:02, 441kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<05:35, 548kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:32, 675kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<03:18, 924kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<02:18, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:46, 629kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:38, 823kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:36, 1.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:29, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:20, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:46, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:59, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:41, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:29, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:37, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:54, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:40, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:27, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:04, 2.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:21, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:29, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:10, 2.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:13, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:07, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:50, 3.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:20, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:02, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:44, 3.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:24, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:55, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:23, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:33, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:34, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:13, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:12, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:04, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:52, 2.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:37, 3.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:19, 676kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<02:46, 807kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:02, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<01:25, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<16:43, 130kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<11:53, 182kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<08:16, 259kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<06:11, 340kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<04:32, 463kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<03:11, 651kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:40, 762kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:04, 980kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:29, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:29, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:26, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:06, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:03, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:56, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:41, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:49, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:36, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:32, 3.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<1:12:23, 24.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<50:28, 34.9kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<34:30, 49.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<24:51, 68.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<17:45, 95.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<12:25, 136kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<08:28, 193kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<07:53, 207kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<05:40, 287kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:56, 406kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:03, 512kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:27, 637kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:46, 868kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<01:13, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<11:31, 130kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<08:11, 182kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<05:39, 258kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<04:12, 340kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:13, 441kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<02:17, 614kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<01:34, 867kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:58, 686kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:31, 890kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:04, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:59, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:31, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<09:04, 134kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<06:27, 188kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<04:27, 267kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<03:17, 350kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:35, 444kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:51, 614kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<01:15, 867kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<08:26, 128kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<05:59, 180kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<04:06, 256kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<03:01, 336kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:12, 458kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:31, 645kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:15, 755kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:58, 972kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:40, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:39, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:33, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:27, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:23, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:17, 2.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:33, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:27, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:22, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:16, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:10, 2.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:07, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<07:53, 25.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<04:57, 36.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<02:33, 51.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<01:46, 71.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<01:07, 102kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:27, 145kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:22, 165kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:14, 230kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:05, 326kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 839/400000 [00:00<00:47, 8382.91it/s]  0%|          | 1723/400000 [00:00<00:46, 8513.80it/s]  1%|          | 2601/400000 [00:00<00:46, 8590.75it/s]  1%|          | 3495/400000 [00:00<00:45, 8690.64it/s]  1%|          | 4376/400000 [00:00<00:45, 8725.72it/s]  1%|▏         | 5262/400000 [00:00<00:45, 8763.60it/s]  2%|▏         | 6126/400000 [00:00<00:45, 8726.00it/s]  2%|▏         | 6970/400000 [00:00<00:45, 8636.68it/s]  2%|▏         | 7829/400000 [00:00<00:45, 8621.63it/s]  2%|▏         | 8730/400000 [00:01<00:44, 8734.47it/s]  2%|▏         | 9627/400000 [00:01<00:44, 8802.67it/s]  3%|▎         | 10547/400000 [00:01<00:43, 8916.71it/s]  3%|▎         | 11428/400000 [00:01<00:43, 8848.25it/s]  3%|▎         | 12317/400000 [00:01<00:43, 8858.44it/s]  3%|▎         | 13210/400000 [00:01<00:43, 8877.34it/s]  4%|▎         | 14106/400000 [00:01<00:43, 8899.31it/s]  4%|▎         | 14994/400000 [00:01<00:43, 8884.38it/s]  4%|▍         | 15881/400000 [00:01<00:43, 8842.89it/s]  4%|▍         | 16773/400000 [00:01<00:43, 8864.45it/s]  4%|▍         | 17659/400000 [00:02<00:43, 8765.82it/s]  5%|▍         | 18547/400000 [00:02<00:43, 8797.50it/s]  5%|▍         | 19427/400000 [00:02<00:43, 8753.19it/s]  5%|▌         | 20303/400000 [00:02<00:43, 8726.94it/s]  5%|▌         | 21176/400000 [00:02<00:43, 8673.96it/s]  6%|▌         | 22048/400000 [00:02<00:43, 8686.00it/s]  6%|▌         | 22927/400000 [00:02<00:43, 8715.85it/s]  6%|▌         | 23807/400000 [00:02<00:43, 8739.78it/s]  6%|▌         | 24693/400000 [00:02<00:42, 8773.79it/s]  6%|▋         | 25571/400000 [00:02<00:42, 8719.60it/s]  7%|▋         | 26444/400000 [00:03<00:42, 8722.32it/s]  7%|▋         | 27335/400000 [00:03<00:42, 8776.01it/s]  7%|▋         | 28257/400000 [00:03<00:41, 8901.83it/s]  7%|▋         | 29165/400000 [00:03<00:41, 8951.90it/s]  8%|▊         | 30085/400000 [00:03<00:40, 9023.87it/s]  8%|▊         | 30988/400000 [00:03<00:41, 8857.81it/s]  8%|▊         | 31877/400000 [00:03<00:41, 8866.72it/s]  8%|▊         | 32765/400000 [00:03<00:41, 8856.01it/s]  8%|▊         | 33652/400000 [00:03<00:41, 8757.39it/s]  9%|▊         | 34546/400000 [00:03<00:41, 8811.01it/s]  9%|▉         | 35428/400000 [00:04<00:41, 8692.95it/s]  9%|▉         | 36298/400000 [00:04<00:42, 8606.85it/s]  9%|▉         | 37161/400000 [00:04<00:42, 8611.07it/s] 10%|▉         | 38034/400000 [00:04<00:41, 8645.03it/s] 10%|▉         | 38908/400000 [00:04<00:41, 8672.59it/s] 10%|▉         | 39777/400000 [00:04<00:41, 8676.74it/s] 10%|█         | 40649/400000 [00:04<00:41, 8688.09it/s] 10%|█         | 41518/400000 [00:04<00:41, 8688.17it/s] 11%|█         | 42387/400000 [00:04<00:41, 8584.43it/s] 11%|█         | 43275/400000 [00:04<00:41, 8629.79it/s] 11%|█         | 44139/400000 [00:05<00:41, 8612.64it/s] 11%|█▏        | 45001/400000 [00:05<00:41, 8563.65it/s] 11%|█▏        | 45867/400000 [00:05<00:41, 8590.80it/s] 12%|█▏        | 46727/400000 [00:05<00:41, 8571.57it/s] 12%|█▏        | 47585/400000 [00:05<00:41, 8572.38it/s] 12%|█▏        | 48443/400000 [00:05<00:41, 8497.08it/s] 12%|█▏        | 49310/400000 [00:05<00:41, 8547.71it/s] 13%|█▎        | 50165/400000 [00:05<00:41, 8457.39it/s] 13%|█▎        | 51012/400000 [00:05<00:41, 8379.39it/s] 13%|█▎        | 51889/400000 [00:05<00:40, 8491.78it/s] 13%|█▎        | 52747/400000 [00:06<00:40, 8515.29it/s] 13%|█▎        | 53602/400000 [00:06<00:40, 8524.93it/s] 14%|█▎        | 54476/400000 [00:06<00:40, 8587.24it/s] 14%|█▍        | 55342/400000 [00:06<00:40, 8607.59it/s] 14%|█▍        | 56206/400000 [00:06<00:39, 8611.11it/s] 14%|█▍        | 57068/400000 [00:06<00:39, 8601.67it/s] 14%|█▍        | 57929/400000 [00:06<00:39, 8603.28it/s] 15%|█▍        | 58790/400000 [00:06<00:39, 8584.47it/s] 15%|█▍        | 59671/400000 [00:06<00:39, 8650.01it/s] 15%|█▌        | 60537/400000 [00:06<00:39, 8631.63it/s] 15%|█▌        | 61412/400000 [00:07<00:39, 8663.00it/s] 16%|█▌        | 62280/400000 [00:07<00:38, 8665.98it/s] 16%|█▌        | 63147/400000 [00:07<00:39, 8624.53it/s] 16%|█▌        | 64020/400000 [00:07<00:38, 8654.91it/s] 16%|█▌        | 64886/400000 [00:07<00:38, 8613.16it/s] 16%|█▋        | 65748/400000 [00:07<00:38, 8576.95it/s] 17%|█▋        | 66606/400000 [00:07<00:39, 8516.62it/s] 17%|█▋        | 67480/400000 [00:07<00:38, 8581.49it/s] 17%|█▋        | 68348/400000 [00:07<00:38, 8610.13it/s] 17%|█▋        | 69215/400000 [00:07<00:38, 8625.19it/s] 18%|█▊        | 70084/400000 [00:08<00:38, 8643.21it/s] 18%|█▊        | 70949/400000 [00:08<00:38, 8625.24it/s] 18%|█▊        | 71824/400000 [00:08<00:37, 8659.92it/s] 18%|█▊        | 72691/400000 [00:08<00:38, 8473.80it/s] 18%|█▊        | 73555/400000 [00:08<00:38, 8522.09it/s] 19%|█▊        | 74429/400000 [00:08<00:37, 8585.28it/s] 19%|█▉        | 75293/400000 [00:08<00:37, 8601.40it/s] 19%|█▉        | 76184/400000 [00:08<00:37, 8690.61it/s] 19%|█▉        | 77065/400000 [00:08<00:37, 8723.61it/s] 19%|█▉        | 77949/400000 [00:08<00:36, 8758.18it/s] 20%|█▉        | 78884/400000 [00:09<00:35, 8926.74it/s] 20%|█▉        | 79792/400000 [00:09<00:35, 8970.54it/s] 20%|██        | 80690/400000 [00:09<00:35, 8956.23it/s] 20%|██        | 81597/400000 [00:09<00:35, 8987.25it/s] 21%|██        | 82497/400000 [00:09<00:35, 8849.97it/s] 21%|██        | 83385/400000 [00:09<00:35, 8856.09it/s] 21%|██        | 84281/400000 [00:09<00:35, 8885.48it/s] 21%|██▏       | 85170/400000 [00:09<00:35, 8778.90it/s] 22%|██▏       | 86049/400000 [00:09<00:36, 8659.04it/s] 22%|██▏       | 86921/400000 [00:09<00:36, 8677.19it/s] 22%|██▏       | 87825/400000 [00:10<00:35, 8781.88it/s] 22%|██▏       | 88726/400000 [00:10<00:35, 8847.82it/s] 22%|██▏       | 89626/400000 [00:10<00:34, 8892.30it/s] 23%|██▎       | 90545/400000 [00:10<00:34, 8977.96it/s] 23%|██▎       | 91444/400000 [00:10<00:34, 8885.49it/s] 23%|██▎       | 92334/400000 [00:10<00:34, 8835.00it/s] 23%|██▎       | 93218/400000 [00:10<00:35, 8718.38it/s] 24%|██▎       | 94091/400000 [00:10<00:35, 8520.91it/s] 24%|██▎       | 94945/400000 [00:10<00:36, 8364.23it/s] 24%|██▍       | 95788/400000 [00:11<00:36, 8382.98it/s] 24%|██▍       | 96671/400000 [00:11<00:35, 8510.37it/s] 24%|██▍       | 97557/400000 [00:11<00:35, 8609.97it/s] 25%|██▍       | 98442/400000 [00:11<00:34, 8678.69it/s] 25%|██▍       | 99320/400000 [00:11<00:34, 8708.46it/s] 25%|██▌       | 100192/400000 [00:11<00:34, 8634.46it/s] 25%|██▌       | 101104/400000 [00:11<00:34, 8773.87it/s] 25%|██▌       | 101983/400000 [00:11<00:34, 8705.67it/s] 26%|██▌       | 102855/400000 [00:11<00:34, 8709.64it/s] 26%|██▌       | 103727/400000 [00:11<00:34, 8700.80it/s] 26%|██▌       | 104616/400000 [00:12<00:33, 8756.21it/s] 26%|██▋       | 105514/400000 [00:12<00:33, 8819.54it/s] 27%|██▋       | 106403/400000 [00:12<00:33, 8838.55it/s] 27%|██▋       | 107298/400000 [00:12<00:33, 8868.41it/s] 27%|██▋       | 108199/400000 [00:12<00:32, 8908.37it/s] 27%|██▋       | 109091/400000 [00:12<00:32, 8848.62it/s] 27%|██▋       | 109977/400000 [00:12<00:34, 8516.68it/s] 28%|██▊       | 110832/400000 [00:12<00:34, 8409.91it/s] 28%|██▊       | 111701/400000 [00:12<00:33, 8491.66it/s] 28%|██▊       | 112569/400000 [00:12<00:33, 8546.22it/s] 28%|██▊       | 113432/400000 [00:13<00:33, 8570.24it/s] 29%|██▊       | 114290/400000 [00:13<00:33, 8550.53it/s] 29%|██▉       | 115161/400000 [00:13<00:33, 8597.46it/s] 29%|██▉       | 116022/400000 [00:13<00:33, 8520.67it/s] 29%|██▉       | 116919/400000 [00:13<00:32, 8650.56it/s] 29%|██▉       | 117785/400000 [00:13<00:32, 8556.99it/s] 30%|██▉       | 118642/400000 [00:13<00:33, 8388.49it/s] 30%|██▉       | 119492/400000 [00:13<00:33, 8419.06it/s] 30%|███       | 120335/400000 [00:13<00:33, 8352.95it/s] 30%|███       | 121178/400000 [00:13<00:33, 8374.75it/s] 31%|███       | 122044/400000 [00:14<00:32, 8456.25it/s] 31%|███       | 122935/400000 [00:14<00:32, 8586.57it/s] 31%|███       | 123818/400000 [00:14<00:31, 8655.65it/s] 31%|███       | 124685/400000 [00:14<00:32, 8529.16it/s] 31%|███▏      | 125555/400000 [00:14<00:31, 8579.35it/s] 32%|███▏      | 126414/400000 [00:14<00:31, 8572.96it/s] 32%|███▏      | 127322/400000 [00:14<00:31, 8716.87it/s] 32%|███▏      | 128195/400000 [00:14<00:31, 8718.85it/s] 32%|███▏      | 129081/400000 [00:14<00:30, 8760.17it/s] 33%|███▎      | 130001/400000 [00:14<00:30, 8886.05it/s] 33%|███▎      | 130893/400000 [00:15<00:30, 8893.96it/s] 33%|███▎      | 131783/400000 [00:15<00:30, 8686.84it/s] 33%|███▎      | 132672/400000 [00:15<00:30, 8744.78it/s] 33%|███▎      | 133548/400000 [00:15<00:30, 8736.77it/s] 34%|███▎      | 134423/400000 [00:15<00:30, 8685.88it/s] 34%|███▍      | 135293/400000 [00:15<00:30, 8622.78it/s] 34%|███▍      | 136156/400000 [00:15<00:31, 8476.24it/s] 34%|███▍      | 137032/400000 [00:15<00:30, 8556.86it/s] 34%|███▍      | 137909/400000 [00:15<00:30, 8619.37it/s] 35%|███▍      | 138797/400000 [00:15<00:30, 8694.87it/s] 35%|███▍      | 139687/400000 [00:16<00:29, 8753.91it/s] 35%|███▌      | 140622/400000 [00:16<00:29, 8924.59it/s] 35%|███▌      | 141556/400000 [00:16<00:28, 9042.10it/s] 36%|███▌      | 142462/400000 [00:16<00:28, 9037.11it/s] 36%|███▌      | 143371/400000 [00:16<00:28, 9050.89it/s] 36%|███▌      | 144277/400000 [00:16<00:28, 8858.37it/s] 36%|███▋      | 145175/400000 [00:16<00:28, 8891.53it/s] 37%|███▋      | 146107/400000 [00:16<00:28, 9013.20it/s] 37%|███▋      | 147010/400000 [00:16<00:28, 8855.86it/s] 37%|███▋      | 147897/400000 [00:16<00:28, 8852.73it/s] 37%|███▋      | 148794/400000 [00:17<00:28, 8885.47it/s] 37%|███▋      | 149684/400000 [00:17<00:28, 8886.03it/s] 38%|███▊      | 150586/400000 [00:17<00:27, 8924.62it/s] 38%|███▊      | 151501/400000 [00:17<00:27, 8990.34it/s] 38%|███▊      | 152427/400000 [00:17<00:27, 9069.14it/s] 38%|███▊      | 153335/400000 [00:17<00:28, 8531.97it/s] 39%|███▊      | 154196/400000 [00:17<00:29, 8443.04it/s] 39%|███▉      | 155046/400000 [00:17<00:29, 8267.14it/s] 39%|███▉      | 155889/400000 [00:17<00:29, 8313.49it/s] 39%|███▉      | 156724/400000 [00:18<00:29, 8312.84it/s] 39%|███▉      | 157606/400000 [00:18<00:28, 8457.93it/s] 40%|███▉      | 158527/400000 [00:18<00:27, 8668.15it/s] 40%|███▉      | 159422/400000 [00:18<00:27, 8748.44it/s] 40%|████      | 160308/400000 [00:18<00:27, 8779.97it/s] 40%|████      | 161206/400000 [00:18<00:27, 8838.13it/s] 41%|████      | 162091/400000 [00:18<00:26, 8819.13it/s] 41%|████      | 162974/400000 [00:18<00:26, 8808.95it/s] 41%|████      | 163856/400000 [00:18<00:27, 8732.53it/s] 41%|████      | 164730/400000 [00:18<00:27, 8695.91it/s] 41%|████▏     | 165605/400000 [00:19<00:26, 8711.41it/s] 42%|████▏     | 166477/400000 [00:19<00:27, 8646.80it/s] 42%|████▏     | 167343/400000 [00:19<00:27, 8514.59it/s] 42%|████▏     | 168196/400000 [00:19<00:27, 8451.55it/s] 42%|████▏     | 169046/400000 [00:19<00:27, 8465.62it/s] 42%|████▏     | 169893/400000 [00:19<00:27, 8408.84it/s] 43%|████▎     | 170780/400000 [00:19<00:26, 8539.31it/s] 43%|████▎     | 171635/400000 [00:19<00:27, 8456.59it/s] 43%|████▎     | 172482/400000 [00:19<00:27, 8326.00it/s] 43%|████▎     | 173358/400000 [00:19<00:26, 8449.35it/s] 44%|████▎     | 174257/400000 [00:20<00:26, 8603.47it/s] 44%|████▍     | 175123/400000 [00:20<00:26, 8618.63it/s] 44%|████▍     | 175986/400000 [00:20<00:26, 8505.84it/s] 44%|████▍     | 176872/400000 [00:20<00:25, 8608.73it/s] 44%|████▍     | 177762/400000 [00:20<00:25, 8692.66it/s] 45%|████▍     | 178633/400000 [00:20<00:25, 8677.85it/s] 45%|████▍     | 179519/400000 [00:20<00:25, 8730.35it/s] 45%|████▌     | 180393/400000 [00:20<00:25, 8700.90it/s] 45%|████▌     | 181282/400000 [00:20<00:24, 8755.84it/s] 46%|████▌     | 182158/400000 [00:20<00:25, 8498.07it/s] 46%|████▌     | 183010/400000 [00:21<00:25, 8428.95it/s] 46%|████▌     | 183888/400000 [00:21<00:25, 8529.35it/s] 46%|████▌     | 184801/400000 [00:21<00:24, 8698.35it/s] 46%|████▋     | 185673/400000 [00:21<00:24, 8669.33it/s] 47%|████▋     | 186542/400000 [00:21<00:24, 8665.78it/s] 47%|████▋     | 187410/400000 [00:21<00:24, 8589.49it/s] 47%|████▋     | 188323/400000 [00:21<00:24, 8742.43it/s] 47%|████▋     | 189199/400000 [00:21<00:24, 8694.05it/s] 48%|████▊     | 190076/400000 [00:21<00:24, 8713.75it/s] 48%|████▊     | 190949/400000 [00:21<00:23, 8717.13it/s] 48%|████▊     | 191859/400000 [00:22<00:23, 8826.64it/s] 48%|████▊     | 192769/400000 [00:22<00:23, 8906.88it/s] 48%|████▊     | 193678/400000 [00:22<00:23, 8958.93it/s] 49%|████▊     | 194622/400000 [00:22<00:22, 9095.21it/s] 49%|████▉     | 195533/400000 [00:22<00:22, 8993.69it/s] 49%|████▉     | 196434/400000 [00:22<00:22, 8918.49it/s] 49%|████▉     | 197327/400000 [00:22<00:22, 8832.35it/s] 50%|████▉     | 198211/400000 [00:22<00:23, 8706.99it/s] 50%|████▉     | 199083/400000 [00:22<00:23, 8657.27it/s] 50%|████▉     | 199950/400000 [00:23<00:23, 8608.65it/s] 50%|█████     | 200855/400000 [00:23<00:22, 8734.85it/s] 50%|█████     | 201730/400000 [00:23<00:22, 8677.22it/s] 51%|█████     | 202658/400000 [00:23<00:22, 8848.20it/s] 51%|█████     | 203568/400000 [00:23<00:22, 8922.18it/s] 51%|█████     | 204478/400000 [00:23<00:21, 8963.67it/s] 51%|█████▏    | 205385/400000 [00:23<00:21, 8993.23it/s] 52%|█████▏    | 206295/400000 [00:23<00:21, 9024.69it/s] 52%|█████▏    | 207198/400000 [00:23<00:21, 8968.53it/s] 52%|█████▏    | 208120/400000 [00:23<00:21, 9042.33it/s] 52%|█████▏    | 209025/400000 [00:24<00:21, 8919.15it/s] 52%|█████▏    | 209927/400000 [00:24<00:21, 8948.80it/s] 53%|█████▎    | 210823/400000 [00:24<00:21, 8930.59it/s] 53%|█████▎    | 211717/400000 [00:24<00:21, 8894.06it/s] 53%|█████▎    | 212607/400000 [00:24<00:21, 8853.73it/s] 53%|█████▎    | 213493/400000 [00:24<00:21, 8829.62it/s] 54%|█████▎    | 214381/400000 [00:24<00:20, 8844.53it/s] 54%|█████▍    | 215266/400000 [00:24<00:21, 8781.46it/s] 54%|█████▍    | 216145/400000 [00:24<00:20, 8758.55it/s] 54%|█████▍    | 217041/400000 [00:24<00:20, 8817.69it/s] 54%|█████▍    | 217944/400000 [00:25<00:20, 8879.26it/s] 55%|█████▍    | 218839/400000 [00:25<00:20, 8898.41it/s] 55%|█████▍    | 219730/400000 [00:25<00:20, 8771.68it/s] 55%|█████▌    | 220663/400000 [00:25<00:20, 8931.33it/s] 55%|█████▌    | 221583/400000 [00:25<00:19, 9008.28it/s] 56%|█████▌    | 222485/400000 [00:25<00:19, 8963.55it/s] 56%|█████▌    | 223386/400000 [00:25<00:19, 8975.11it/s] 56%|█████▌    | 224284/400000 [00:25<00:19, 8834.74it/s] 56%|█████▋    | 225180/400000 [00:25<00:19, 8869.80it/s] 57%|█████▋    | 226068/400000 [00:25<00:19, 8858.54it/s] 57%|█████▋    | 226960/400000 [00:26<00:19, 8876.28it/s] 57%|█████▋    | 227852/400000 [00:26<00:19, 8886.66it/s] 57%|█████▋    | 228741/400000 [00:26<00:19, 8865.73it/s] 57%|█████▋    | 229628/400000 [00:26<00:19, 8831.60it/s] 58%|█████▊    | 230512/400000 [00:26<00:19, 8743.75it/s] 58%|█████▊    | 231387/400000 [00:26<00:19, 8524.19it/s] 58%|█████▊    | 232252/400000 [00:26<00:19, 8560.34it/s] 58%|█████▊    | 233126/400000 [00:26<00:19, 8612.13it/s] 59%|█████▊    | 234002/400000 [00:26<00:19, 8653.36it/s] 59%|█████▊    | 234885/400000 [00:26<00:18, 8702.71it/s] 59%|█████▉    | 235756/400000 [00:27<00:19, 8560.67it/s] 59%|█████▉    | 236613/400000 [00:27<00:19, 8309.88it/s] 59%|█████▉    | 237493/400000 [00:27<00:19, 8449.37it/s] 60%|█████▉    | 238393/400000 [00:27<00:18, 8606.02it/s] 60%|█████▉    | 239297/400000 [00:27<00:18, 8730.54it/s] 60%|██████    | 240187/400000 [00:27<00:18, 8778.91it/s] 60%|██████    | 241067/400000 [00:27<00:18, 8634.26it/s] 60%|██████    | 241932/400000 [00:27<00:18, 8568.17it/s] 61%|██████    | 242791/400000 [00:27<00:18, 8475.42it/s] 61%|██████    | 243665/400000 [00:27<00:18, 8552.54it/s] 61%|██████    | 244581/400000 [00:28<00:17, 8724.43it/s] 61%|██████▏   | 245490/400000 [00:28<00:17, 8830.24it/s] 62%|██████▏   | 246403/400000 [00:28<00:17, 8916.14it/s] 62%|██████▏   | 247319/400000 [00:28<00:16, 8987.65it/s] 62%|██████▏   | 248219/400000 [00:28<00:17, 8838.99it/s] 62%|██████▏   | 249105/400000 [00:28<00:17, 8747.29it/s] 62%|██████▏   | 249996/400000 [00:28<00:17, 8792.49it/s] 63%|██████▎   | 250877/400000 [00:28<00:16, 8790.10it/s] 63%|██████▎   | 251768/400000 [00:28<00:16, 8822.71it/s] 63%|██████▎   | 252665/400000 [00:28<00:16, 8865.11it/s] 63%|██████▎   | 253594/400000 [00:29<00:16, 8988.32it/s] 64%|██████▎   | 254494/400000 [00:29<00:16, 8989.47it/s] 64%|██████▍   | 255394/400000 [00:29<00:16, 8928.25it/s] 64%|██████▍   | 256288/400000 [00:29<00:16, 8780.62it/s] 64%|██████▍   | 257167/400000 [00:29<00:16, 8651.46it/s] 65%|██████▍   | 258034/400000 [00:29<00:16, 8637.18it/s] 65%|██████▍   | 258944/400000 [00:29<00:16, 8769.48it/s] 65%|██████▍   | 259835/400000 [00:29<00:15, 8809.32it/s] 65%|██████▌   | 260766/400000 [00:29<00:15, 8953.45it/s] 65%|██████▌   | 261702/400000 [00:30<00:15, 9068.82it/s] 66%|██████▌   | 262611/400000 [00:30<00:15, 8922.95it/s] 66%|██████▌   | 263509/400000 [00:30<00:15, 8938.89it/s] 66%|██████▌   | 264426/400000 [00:30<00:15, 9005.75it/s] 66%|██████▋   | 265337/400000 [00:30<00:14, 9035.80it/s] 67%|██████▋   | 266242/400000 [00:30<00:15, 8877.34it/s] 67%|██████▋   | 267131/400000 [00:30<00:15, 8818.08it/s] 67%|██████▋   | 268014/400000 [00:30<00:15, 8760.91it/s] 67%|██████▋   | 268891/400000 [00:30<00:15, 8679.70it/s] 67%|██████▋   | 269812/400000 [00:30<00:14, 8830.00it/s] 68%|██████▊   | 270737/400000 [00:31<00:14, 8951.93it/s] 68%|██████▊   | 271634/400000 [00:31<00:14, 8913.11it/s] 68%|██████▊   | 272527/400000 [00:31<00:14, 8592.71it/s] 68%|██████▊   | 273442/400000 [00:31<00:14, 8751.54it/s] 69%|██████▊   | 274401/400000 [00:31<00:13, 8984.57it/s] 69%|██████▉   | 275334/400000 [00:31<00:13, 9083.94it/s] 69%|██████▉   | 276247/400000 [00:31<00:13, 9097.34it/s] 69%|██████▉   | 277174/400000 [00:31<00:13, 9145.25it/s] 70%|██████▉   | 278090/400000 [00:31<00:13, 9142.17it/s] 70%|██████▉   | 279006/400000 [00:31<00:13, 9073.72it/s] 70%|██████▉   | 279915/400000 [00:32<00:13, 9055.99it/s] 70%|███████   | 280834/400000 [00:32<00:13, 9094.34it/s] 70%|███████   | 281744/400000 [00:32<00:13, 9062.18it/s] 71%|███████   | 282651/400000 [00:32<00:13, 8948.26it/s] 71%|███████   | 283555/400000 [00:32<00:12, 8974.44it/s] 71%|███████   | 284453/400000 [00:32<00:13, 8879.57it/s] 71%|███████▏  | 285397/400000 [00:32<00:12, 9040.54it/s] 72%|███████▏  | 286305/400000 [00:32<00:12, 9050.89it/s] 72%|███████▏  | 287211/400000 [00:32<00:12, 8986.80it/s] 72%|███████▏  | 288111/400000 [00:32<00:12, 8972.37it/s] 72%|███████▏  | 289013/400000 [00:33<00:12, 8984.93it/s] 72%|███████▏  | 289931/400000 [00:33<00:12, 9040.54it/s] 73%|███████▎  | 290838/400000 [00:33<00:12, 9047.46it/s] 73%|███████▎  | 291743/400000 [00:33<00:12, 8998.85it/s] 73%|███████▎  | 292644/400000 [00:33<00:11, 8971.76it/s] 73%|███████▎  | 293542/400000 [00:33<00:12, 8834.06it/s] 74%|███████▎  | 294437/400000 [00:33<00:11, 8867.82it/s] 74%|███████▍  | 295340/400000 [00:33<00:11, 8914.14it/s] 74%|███████▍  | 296232/400000 [00:33<00:11, 8862.29it/s] 74%|███████▍  | 297163/400000 [00:33<00:11, 8991.79it/s] 75%|███████▍  | 298066/400000 [00:34<00:11, 9001.73it/s] 75%|███████▍  | 298967/400000 [00:34<00:11, 8947.99it/s] 75%|███████▍  | 299863/400000 [00:34<00:11, 8808.70it/s] 75%|███████▌  | 300763/400000 [00:34<00:11, 8863.28it/s] 75%|███████▌  | 301650/400000 [00:34<00:11, 8735.62it/s] 76%|███████▌  | 302533/400000 [00:34<00:11, 8763.65it/s] 76%|███████▌  | 303435/400000 [00:34<00:10, 8836.19it/s] 76%|███████▌  | 304320/400000 [00:34<00:10, 8840.16it/s] 76%|███████▋  | 305205/400000 [00:34<00:10, 8808.27it/s] 77%|███████▋  | 306087/400000 [00:34<00:10, 8570.10it/s] 77%|███████▋  | 306946/400000 [00:35<00:11, 8413.29it/s] 77%|███████▋  | 307790/400000 [00:35<00:10, 8414.12it/s] 77%|███████▋  | 308664/400000 [00:35<00:10, 8507.39it/s] 77%|███████▋  | 309516/400000 [00:35<00:10, 8418.63it/s] 78%|███████▊  | 310382/400000 [00:35<00:10, 8488.42it/s] 78%|███████▊  | 311232/400000 [00:35<00:11, 7995.88it/s] 78%|███████▊  | 312070/400000 [00:35<00:10, 8104.92it/s] 78%|███████▊  | 312913/400000 [00:35<00:10, 8197.36it/s] 78%|███████▊  | 313773/400000 [00:35<00:10, 8312.08it/s] 79%|███████▊  | 314608/400000 [00:36<00:10, 8285.51it/s] 79%|███████▉  | 315490/400000 [00:36<00:10, 8438.81it/s] 79%|███████▉  | 316360/400000 [00:36<00:09, 8513.30it/s] 79%|███████▉  | 317237/400000 [00:36<00:09, 8586.58it/s] 80%|███████▉  | 318101/400000 [00:36<00:09, 8602.37it/s] 80%|███████▉  | 318971/400000 [00:36<00:09, 8631.26it/s] 80%|███████▉  | 319835/400000 [00:36<00:09, 8599.76it/s] 80%|████████  | 320705/400000 [00:36<00:09, 8627.06it/s] 80%|████████  | 321569/400000 [00:36<00:09, 8612.24it/s] 81%|████████  | 322431/400000 [00:36<00:09, 8494.60it/s] 81%|████████  | 323315/400000 [00:37<00:08, 8592.49it/s] 81%|████████  | 324197/400000 [00:37<00:08, 8657.20it/s] 81%|████████▏ | 325064/400000 [00:37<00:08, 8614.97it/s] 81%|████████▏ | 325947/400000 [00:37<00:08, 8677.07it/s] 82%|████████▏ | 326832/400000 [00:37<00:08, 8727.46it/s] 82%|████████▏ | 327706/400000 [00:37<00:08, 8676.53it/s] 82%|████████▏ | 328580/400000 [00:37<00:08, 8695.18it/s] 82%|████████▏ | 329450/400000 [00:37<00:08, 8542.49it/s] 83%|████████▎ | 330306/400000 [00:37<00:08, 8518.65it/s] 83%|████████▎ | 331159/400000 [00:37<00:08, 8446.05it/s] 83%|████████▎ | 332052/400000 [00:38<00:07, 8584.40it/s] 83%|████████▎ | 332953/400000 [00:38<00:07, 8705.87it/s] 83%|████████▎ | 333825/400000 [00:38<00:07, 8672.19it/s] 84%|████████▎ | 334693/400000 [00:38<00:07, 8546.23it/s] 84%|████████▍ | 335571/400000 [00:38<00:07, 8612.26it/s] 84%|████████▍ | 336434/400000 [00:38<00:07, 8610.01it/s] 84%|████████▍ | 337296/400000 [00:38<00:07, 8598.70it/s] 85%|████████▍ | 338157/400000 [00:38<00:07, 8579.53it/s] 85%|████████▍ | 339045/400000 [00:38<00:07, 8666.14it/s] 85%|████████▍ | 339923/400000 [00:38<00:06, 8698.56it/s] 85%|████████▌ | 340794/400000 [00:39<00:06, 8698.41it/s] 85%|████████▌ | 341670/400000 [00:39<00:06, 8713.70it/s] 86%|████████▌ | 342545/400000 [00:39<00:06, 8722.34it/s] 86%|████████▌ | 343418/400000 [00:39<00:06, 8689.43it/s] 86%|████████▌ | 344288/400000 [00:39<00:06, 8658.59it/s] 86%|████████▋ | 345154/400000 [00:39<00:06, 8283.11it/s] 87%|████████▋ | 346011/400000 [00:39<00:06, 8365.26it/s] 87%|████████▋ | 346877/400000 [00:39<00:06, 8451.37it/s] 87%|████████▋ | 347725/400000 [00:39<00:06, 8413.40it/s] 87%|████████▋ | 348575/400000 [00:39<00:06, 8438.07it/s] 87%|████████▋ | 349451/400000 [00:40<00:05, 8530.92it/s] 88%|████████▊ | 350352/400000 [00:40<00:05, 8666.68it/s] 88%|████████▊ | 351220/400000 [00:40<00:05, 8616.47it/s] 88%|████████▊ | 352093/400000 [00:40<00:05, 8647.47it/s] 88%|████████▊ | 352959/400000 [00:40<00:05, 8619.62it/s] 88%|████████▊ | 353822/400000 [00:40<00:05, 8385.95it/s] 89%|████████▊ | 354671/400000 [00:40<00:05, 8416.13it/s] 89%|████████▉ | 355514/400000 [00:40<00:05, 8375.77it/s] 89%|████████▉ | 356397/400000 [00:40<00:05, 8505.85it/s] 89%|████████▉ | 357249/400000 [00:40<00:05, 8470.23it/s] 90%|████████▉ | 358113/400000 [00:41<00:04, 8520.28it/s] 90%|████████▉ | 358985/400000 [00:41<00:04, 8579.14it/s] 90%|████████▉ | 359844/400000 [00:41<00:04, 8439.83it/s] 90%|█████████ | 360689/400000 [00:41<00:04, 8402.45it/s] 90%|█████████ | 361585/400000 [00:41<00:04, 8561.44it/s] 91%|█████████ | 362513/400000 [00:41<00:04, 8763.85it/s] 91%|█████████ | 363456/400000 [00:41<00:04, 8951.28it/s] 91%|█████████ | 364370/400000 [00:41<00:03, 9005.94it/s] 91%|█████████▏| 365273/400000 [00:41<00:03, 8999.34it/s] 92%|█████████▏| 366175/400000 [00:42<00:03, 8934.90it/s] 92%|█████████▏| 367070/400000 [00:42<00:03, 8910.53it/s] 92%|█████████▏| 367962/400000 [00:42<00:03, 8737.00it/s] 92%|█████████▏| 368837/400000 [00:42<00:03, 8728.12it/s] 92%|█████████▏| 369726/400000 [00:42<00:03, 8775.56it/s] 93%|█████████▎| 370605/400000 [00:42<00:03, 8759.40it/s] 93%|█████████▎| 371503/400000 [00:42<00:03, 8821.64it/s] 93%|█████████▎| 372386/400000 [00:42<00:03, 8752.87it/s] 93%|█████████▎| 373305/400000 [00:42<00:03, 8879.01it/s] 94%|█████████▎| 374194/400000 [00:42<00:02, 8808.95it/s] 94%|█████████▍| 375076/400000 [00:43<00:02, 8750.47it/s] 94%|█████████▍| 375989/400000 [00:43<00:02, 8860.85it/s] 94%|█████████▍| 376893/400000 [00:43<00:02, 8912.42it/s] 94%|█████████▍| 377785/400000 [00:43<00:02, 8830.00it/s] 95%|█████████▍| 378690/400000 [00:43<00:02, 8892.78it/s] 95%|█████████▍| 379580/400000 [00:43<00:02, 8737.86it/s] 95%|█████████▌| 380510/400000 [00:43<00:02, 8897.82it/s] 95%|█████████▌| 381402/400000 [00:43<00:02, 8752.30it/s] 96%|█████████▌| 382328/400000 [00:43<00:01, 8896.53it/s] 96%|█████████▌| 383220/400000 [00:43<00:01, 8885.45it/s] 96%|█████████▌| 384125/400000 [00:44<00:01, 8933.74it/s] 96%|█████████▋| 385039/400000 [00:44<00:01, 8993.18it/s] 96%|█████████▋| 385940/400000 [00:44<00:01, 8900.81it/s] 97%|█████████▋| 386831/400000 [00:44<00:01, 8765.74it/s] 97%|█████████▋| 387709/400000 [00:44<00:01, 8747.91it/s] 97%|█████████▋| 388585/400000 [00:44<00:01, 8718.04it/s] 97%|█████████▋| 389458/400000 [00:44<00:01, 8612.51it/s] 98%|█████████▊| 390323/400000 [00:44<00:01, 8621.31it/s] 98%|█████████▊| 391189/400000 [00:44<00:01, 8631.21it/s] 98%|█████████▊| 392074/400000 [00:44<00:00, 8695.59it/s] 98%|█████████▊| 392954/400000 [00:45<00:00, 8726.32it/s] 98%|█████████▊| 393837/400000 [00:45<00:00, 8756.63it/s] 99%|█████████▊| 394729/400000 [00:45<00:00, 8803.67it/s] 99%|█████████▉| 395610/400000 [00:45<00:00, 8762.37it/s] 99%|█████████▉| 396487/400000 [00:45<00:00, 8763.79it/s] 99%|█████████▉| 397364/400000 [00:45<00:00, 8740.21it/s]100%|█████████▉| 398257/400000 [00:45<00:00, 8793.85it/s]100%|█████████▉| 399137/400000 [00:45<00:00, 8767.25it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8723.32it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f447c0b6a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011175079066013232 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010943039205162023 	 Accuracy: 61

  model saves at 61% accuracy 

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
2020-05-15 11:23:55.173984: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 11:23:55.178642: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-15 11:23:55.178786: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564df2311e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 11:23:55.178798: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4425486fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.1760 - accuracy: 0.5320
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4136 - accuracy: 0.5165 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5337 - accuracy: 0.5087
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6022 - accuracy: 0.5042
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6104 - accuracy: 0.5037
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6162 - accuracy: 0.5033
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5763 - accuracy: 0.5059
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5992 - accuracy: 0.5044
11000/25000 [============>.................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
12000/25000 [=============>................] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6324 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6447 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6637 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6692 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6980 - accuracy: 0.4980
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6938 - accuracy: 0.4982
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f43e4e58668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f43e0762d68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.0356 - crf_viterbi_accuracy: 0.6800 - val_loss: 0.9273 - val_crf_viterbi_accuracy: 0.6533

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f20cffc4f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 12:13:57.550402
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 12:13:57.555491
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 12:13:57.559401
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 12:13:57.563673
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f20dbd8e438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356127.4688
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 273965.1562
Epoch 3/10

1/1 [==============================] - 0s 101ms/step - loss: 190583.5312
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 117834.8203
Epoch 5/10

1/1 [==============================] - 0s 106ms/step - loss: 67894.3828
Epoch 6/10

1/1 [==============================] - 0s 105ms/step - loss: 38728.0469
Epoch 7/10

1/1 [==============================] - 0s 113ms/step - loss: 23922.1973
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 15959.9131
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 11320.8184
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 8474.2988

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.44681925e-01 -5.12362957e-01  5.66823483e-02 -7.73171186e-01
   4.68467206e-01  1.20556571e-01 -3.93103302e-01 -4.27821837e-02
   6.65458024e-01 -1.58480331e-01 -8.50026429e-01  7.38968432e-01
   8.97207081e-01  2.04661894e+00 -5.59539080e-01 -1.45123565e+00
   6.06152534e-01  4.57385540e-01 -4.23112631e-01 -9.52780843e-01
   4.43244040e-01  1.31088185e+00 -1.76404643e+00  5.14621377e-01
  -8.75760674e-01 -5.47579646e-01  7.16460586e-01  2.90732205e-01
  -2.33402967e-01  8.65903974e-01  9.10264403e-02  1.26154566e+00
   4.12453949e-01 -5.09402394e-01  1.78537238e+00  4.64345723e-01
   8.48377943e-02  1.12654066e+00 -4.95467782e-02  2.92855859e-01
   8.95318449e-01  1.43664062e-01  1.08181751e+00 -8.14487219e-01
  -9.84935403e-01 -4.62049156e-01  8.57808113e-01 -1.53295577e+00
  -8.47213864e-02 -4.12641823e-01 -9.77760911e-01  4.35545325e-01
  -1.79677451e+00  7.52260029e-01  9.65760052e-01  9.32603329e-02
  -2.61595702e+00 -1.31380486e+00  9.28687632e-01 -1.19930565e-01
   1.68948025e-01  7.48332644e+00  6.90764809e+00  9.16884995e+00
   5.36901331e+00  7.02157354e+00  7.22270060e+00  6.64008713e+00
   6.39799118e+00  7.55993414e+00  6.06877041e+00  6.44454432e+00
   6.53376245e+00  6.33620214e+00  6.83684397e+00  7.85349131e+00
   6.21662807e+00  6.77373886e+00  6.77347326e+00  6.15245676e+00
   7.79362726e+00  6.45069742e+00  7.89041901e+00  8.23175144e+00
   6.64406443e+00  8.63254833e+00  7.37391615e+00  8.95791912e+00
   8.17412758e+00  7.00087738e+00  6.42680120e+00  5.92527246e+00
   7.91677141e+00  7.62418556e+00  8.09329510e+00  6.78335142e+00
   6.95209169e+00  6.87058115e+00  7.97370195e+00  6.74101782e+00
   6.56115818e+00  7.35826683e+00  8.05675411e+00  6.70430565e+00
   7.92819548e+00  6.10026407e+00  6.28168392e+00  6.94800663e+00
   7.07678032e+00  8.35347176e+00  6.07545948e+00  8.06072521e+00
   7.75391769e+00  6.97370338e+00  6.73483896e+00  6.91516447e+00
   6.62937450e+00  7.06225872e+00  7.64731598e+00  7.15081024e+00
  -1.85357153e+00  1.22053456e+00 -1.93020535e+00  3.58214259e-01
   3.38202119e-01  1.28224224e-01  1.68189716e+00 -8.07957768e-01
   9.98909593e-01 -2.74493039e-01  1.91209686e+00  3.75871301e-01
  -6.40292287e-01 -8.06626976e-02  3.60453546e-01 -6.47417963e-01
   3.92576069e-01 -3.01067650e-01 -4.87096250e-01  2.19544813e-01
  -1.23097861e+00 -2.69679397e-01  1.85527945e+00  8.22942436e-01
  -4.34090197e-02 -1.29786789e-01 -1.85744107e-01 -2.36984879e-01
   5.97572207e-01  1.17373800e+00 -6.00104630e-01  1.04413915e+00
   6.83857858e-01  1.64081061e+00  6.88905895e-01  6.11439466e-01
  -9.28955749e-02  2.07339525e-02 -1.30170971e-01 -1.51564509e-01
   1.69703507e+00  5.02678990e-01 -6.80574000e-01  1.48001206e+00
  -6.21288598e-01  3.37088346e-01  6.59373641e-01  1.53799415e+00
   3.30169141e-01  9.32647526e-01 -2.28685045e+00  4.18103695e-01
  -1.41725242e+00  5.81240654e-03 -2.93923676e-01  6.96349978e-01
  -1.21223405e-01  1.50895298e-01 -1.32977962e-03  1.10903752e+00
   7.25139439e-01  6.86899304e-01  2.15746117e+00  2.09318781e+00
   6.05906129e-01  1.87174463e+00  1.68044066e+00  7.27079570e-01
   1.37507045e+00  4.75246072e-01  1.53064418e+00  2.14865065e+00
   2.04374790e+00  2.02199841e+00  4.92340326e-01  3.45475078e-01
   2.11669922e+00  2.27788258e+00  1.62454927e+00  1.72348201e-01
   4.24297273e-01  1.02580070e+00  2.26916170e+00  3.29415083e-01
   4.45983946e-01  6.61861837e-01  1.39886224e+00  3.64676893e-01
   2.79759169e-01  6.38707161e-01  1.19983768e+00  1.87116742e-01
   3.33087265e-01  6.14903629e-01  2.44896984e+00  2.80168295e-01
   1.30013227e+00  1.33615017e+00  2.11868095e+00  1.33985376e+00
   1.63132322e+00  6.81765676e-01  2.66492176e+00  2.60889649e-01
   1.82474530e+00  1.54807436e+00  1.01299798e+00  1.80696106e+00
   8.59972656e-01  1.10153437e+00  1.74041152e+00  6.15799785e-01
   6.41128182e-01  8.37645173e-01  7.16373920e-01  1.69256532e+00
   1.92631423e-01  5.23664832e-01  8.98637056e-01  6.09947920e-01
   1.02631211e-01  8.03598309e+00  8.58459568e+00  5.79421616e+00
   8.86956596e+00  9.08273697e+00  8.27267361e+00  7.64962101e+00
   7.45183802e+00  6.90438223e+00  7.67287588e+00  8.36890125e+00
   6.78962326e+00  6.01178312e+00  5.97066975e+00  7.07700443e+00
   6.92213392e+00  7.16231155e+00  7.56624174e+00  8.18050385e+00
   6.89536428e+00  7.13710690e+00  8.00984097e+00  6.40561819e+00
   8.20792198e+00  8.31358242e+00  8.32857990e+00  6.98645353e+00
   7.38010883e+00  9.30076694e+00  7.70440769e+00  8.36571884e+00
   8.28485584e+00  5.80451488e+00  6.80019855e+00  5.89752626e+00
   7.04612064e+00  8.94456863e+00  8.14508629e+00  8.69615841e+00
   7.63861227e+00  6.52747917e+00  8.90216923e+00  8.06024265e+00
   8.41627312e+00  6.57237720e+00  8.46222973e+00  7.19506788e+00
   7.95358801e+00  7.13041496e+00  9.20968437e+00  6.37109232e+00
   7.30075979e+00  5.47821426e+00  6.70627594e+00  6.79142857e+00
   7.78341007e+00  7.40732574e+00  7.10134315e+00  7.40360594e+00
   1.06148314e+00  1.48085141e+00  5.24878800e-01  2.14057779e+00
   1.92962527e+00  9.65298176e-01  1.57079148e+00  8.43902528e-01
   5.32515287e-01  1.67473650e+00  5.36518574e-01  5.00448585e-01
   2.14293957e+00  5.39683342e-01  1.37291074e+00  1.18206573e+00
   9.42013443e-01  6.23917103e-01  1.50310826e+00  1.78640115e+00
   4.98635650e-01  5.80237806e-01  2.07917809e-01  1.00875807e+00
   6.77897990e-01  3.43546391e-01  8.34644556e-01  1.48797762e+00
   2.69536495e-01  1.29958963e+00  1.57541382e+00  7.86856890e-01
   2.10953236e+00  6.79485619e-01  1.14618254e+00  3.05355072e-01
   2.62500000e+00  1.51480091e+00  2.23372281e-01  9.06588674e-01
   2.03932822e-01  1.18703592e+00  1.79000342e+00  7.90423989e-01
   1.53926492e+00  2.64051795e-01  4.56151783e-01  7.76993155e-01
   8.92266035e-01  1.77128327e+00  8.98092568e-01  1.06902647e+00
   9.08352017e-01  1.40078425e+00  1.57390106e+00  1.00907004e+00
   8.66473675e-01  9.68758821e-01  4.00499225e-01  1.85034585e+00
  -1.04353294e+01  9.26729870e+00 -3.97422075e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 12:14:07.300015
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.5008
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 12:14:07.304391
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8953.55
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 12:14:07.308092
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.846
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 12:14:07.311831
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -800.854
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139778562847744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139777336218176
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139777336218680
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139777336219184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139777336219688
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139777336220192

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f20c96fb550> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.578219
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.538272
grad_step = 000002, loss = 0.507771
grad_step = 000003, loss = 0.477696
grad_step = 000004, loss = 0.442245
grad_step = 000005, loss = 0.406442
grad_step = 000006, loss = 0.378883
grad_step = 000007, loss = 0.370985
grad_step = 000008, loss = 0.356439
grad_step = 000009, loss = 0.338031
grad_step = 000010, loss = 0.322988
grad_step = 000011, loss = 0.308851
grad_step = 000012, loss = 0.294704
grad_step = 000013, loss = 0.282756
grad_step = 000014, loss = 0.271895
grad_step = 000015, loss = 0.261210
grad_step = 000016, loss = 0.250344
grad_step = 000017, loss = 0.239387
grad_step = 000018, loss = 0.228943
grad_step = 000019, loss = 0.219152
grad_step = 000020, loss = 0.209349
grad_step = 000021, loss = 0.198733
grad_step = 000022, loss = 0.188996
grad_step = 000023, loss = 0.180764
grad_step = 000024, loss = 0.172690
grad_step = 000025, loss = 0.164164
grad_step = 000026, loss = 0.156008
grad_step = 000027, loss = 0.148709
grad_step = 000028, loss = 0.141754
grad_step = 000029, loss = 0.134409
grad_step = 000030, loss = 0.126742
grad_step = 000031, loss = 0.119537
grad_step = 000032, loss = 0.113234
grad_step = 000033, loss = 0.107449
grad_step = 000034, loss = 0.101681
grad_step = 000035, loss = 0.096000
grad_step = 000036, loss = 0.090662
grad_step = 000037, loss = 0.085596
grad_step = 000038, loss = 0.080614
grad_step = 000039, loss = 0.075694
grad_step = 000040, loss = 0.070923
grad_step = 000041, loss = 0.066430
grad_step = 000042, loss = 0.062339
grad_step = 000043, loss = 0.058533
grad_step = 000044, loss = 0.054902
grad_step = 000045, loss = 0.051463
grad_step = 000046, loss = 0.048131
grad_step = 000047, loss = 0.044870
grad_step = 000048, loss = 0.041820
grad_step = 000049, loss = 0.039027
grad_step = 000050, loss = 0.036374
grad_step = 000051, loss = 0.033808
grad_step = 000052, loss = 0.031374
grad_step = 000053, loss = 0.029076
grad_step = 000054, loss = 0.026900
grad_step = 000055, loss = 0.024872
grad_step = 000056, loss = 0.022997
grad_step = 000057, loss = 0.021247
grad_step = 000058, loss = 0.019592
grad_step = 000059, loss = 0.018021
grad_step = 000060, loss = 0.016558
grad_step = 000061, loss = 0.015200
grad_step = 000062, loss = 0.013930
grad_step = 000063, loss = 0.012762
grad_step = 000064, loss = 0.011685
grad_step = 000065, loss = 0.010678
grad_step = 000066, loss = 0.009766
grad_step = 000067, loss = 0.008937
grad_step = 000068, loss = 0.008172
grad_step = 000069, loss = 0.007479
grad_step = 000070, loss = 0.006844
grad_step = 000071, loss = 0.006258
grad_step = 000072, loss = 0.005753
grad_step = 000073, loss = 0.005313
grad_step = 000074, loss = 0.004916
grad_step = 000075, loss = 0.004561
grad_step = 000076, loss = 0.004238
grad_step = 000077, loss = 0.003951
grad_step = 000078, loss = 0.003709
grad_step = 000079, loss = 0.003501
grad_step = 000080, loss = 0.003318
grad_step = 000081, loss = 0.003158
grad_step = 000082, loss = 0.003017
grad_step = 000083, loss = 0.002896
grad_step = 000084, loss = 0.002796
grad_step = 000085, loss = 0.002707
grad_step = 000086, loss = 0.002629
grad_step = 000087, loss = 0.002565
grad_step = 000088, loss = 0.002508
grad_step = 000089, loss = 0.002461
grad_step = 000090, loss = 0.002421
grad_step = 000091, loss = 0.002385
grad_step = 000092, loss = 0.002356
grad_step = 000093, loss = 0.002331
grad_step = 000094, loss = 0.002311
grad_step = 000095, loss = 0.002293
grad_step = 000096, loss = 0.002276
grad_step = 000097, loss = 0.002262
grad_step = 000098, loss = 0.002250
grad_step = 000099, loss = 0.002241
grad_step = 000100, loss = 0.002232
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002223
grad_step = 000102, loss = 0.002215
grad_step = 000103, loss = 0.002208
grad_step = 000104, loss = 0.002201
grad_step = 000105, loss = 0.002195
grad_step = 000106, loss = 0.002188
grad_step = 000107, loss = 0.002182
grad_step = 000108, loss = 0.002175
grad_step = 000109, loss = 0.002169
grad_step = 000110, loss = 0.002162
grad_step = 000111, loss = 0.002155
grad_step = 000112, loss = 0.002149
grad_step = 000113, loss = 0.002142
grad_step = 000114, loss = 0.002135
grad_step = 000115, loss = 0.002128
grad_step = 000116, loss = 0.002121
grad_step = 000117, loss = 0.002114
grad_step = 000118, loss = 0.002107
grad_step = 000119, loss = 0.002099
grad_step = 000120, loss = 0.002092
grad_step = 000121, loss = 0.002085
grad_step = 000122, loss = 0.002079
grad_step = 000123, loss = 0.002075
grad_step = 000124, loss = 0.002068
grad_step = 000125, loss = 0.002060
grad_step = 000126, loss = 0.002056
grad_step = 000127, loss = 0.002051
grad_step = 000128, loss = 0.002043
grad_step = 000129, loss = 0.002039
grad_step = 000130, loss = 0.002035
grad_step = 000131, loss = 0.002029
grad_step = 000132, loss = 0.002024
grad_step = 000133, loss = 0.002021
grad_step = 000134, loss = 0.002016
grad_step = 000135, loss = 0.002011
grad_step = 000136, loss = 0.002007
grad_step = 000137, loss = 0.002003
grad_step = 000138, loss = 0.001999
grad_step = 000139, loss = 0.001995
grad_step = 000140, loss = 0.001990
grad_step = 000141, loss = 0.001986
grad_step = 000142, loss = 0.001982
grad_step = 000143, loss = 0.001978
grad_step = 000144, loss = 0.001974
grad_step = 000145, loss = 0.001969
grad_step = 000146, loss = 0.001965
grad_step = 000147, loss = 0.001960
grad_step = 000148, loss = 0.001955
grad_step = 000149, loss = 0.001950
grad_step = 000150, loss = 0.001945
grad_step = 000151, loss = 0.001940
grad_step = 000152, loss = 0.001935
grad_step = 000153, loss = 0.001930
grad_step = 000154, loss = 0.001927
grad_step = 000155, loss = 0.001930
grad_step = 000156, loss = 0.001941
grad_step = 000157, loss = 0.001945
grad_step = 000158, loss = 0.001923
grad_step = 000159, loss = 0.001900
grad_step = 000160, loss = 0.001911
grad_step = 000161, loss = 0.001914
grad_step = 000162, loss = 0.001895
grad_step = 000163, loss = 0.001885
grad_step = 000164, loss = 0.001888
grad_step = 000165, loss = 0.001886
grad_step = 000166, loss = 0.001871
grad_step = 000167, loss = 0.001868
grad_step = 000168, loss = 0.001867
grad_step = 000169, loss = 0.001862
grad_step = 000170, loss = 0.001853
grad_step = 000171, loss = 0.001847
grad_step = 000172, loss = 0.001846
grad_step = 000173, loss = 0.001842
grad_step = 000174, loss = 0.001836
grad_step = 000175, loss = 0.001831
grad_step = 000176, loss = 0.001824
grad_step = 000177, loss = 0.001819
grad_step = 000178, loss = 0.001817
grad_step = 000179, loss = 0.001815
grad_step = 000180, loss = 0.001814
grad_step = 000181, loss = 0.001812
grad_step = 000182, loss = 0.001806
grad_step = 000183, loss = 0.001800
grad_step = 000184, loss = 0.001794
grad_step = 000185, loss = 0.001788
grad_step = 000186, loss = 0.001783
grad_step = 000187, loss = 0.001779
grad_step = 000188, loss = 0.001775
grad_step = 000189, loss = 0.001772
grad_step = 000190, loss = 0.001770
grad_step = 000191, loss = 0.001771
grad_step = 000192, loss = 0.001781
grad_step = 000193, loss = 0.001820
grad_step = 000194, loss = 0.001852
grad_step = 000195, loss = 0.001898
grad_step = 000196, loss = 0.001833
grad_step = 000197, loss = 0.001810
grad_step = 000198, loss = 0.001822
grad_step = 000199, loss = 0.001789
grad_step = 000200, loss = 0.001781
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001785
grad_step = 000202, loss = 0.001784
grad_step = 000203, loss = 0.001776
grad_step = 000204, loss = 0.001758
grad_step = 000205, loss = 0.001776
grad_step = 000206, loss = 0.001773
grad_step = 000207, loss = 0.001743
grad_step = 000208, loss = 0.001737
grad_step = 000209, loss = 0.001751
grad_step = 000210, loss = 0.001762
grad_step = 000211, loss = 0.001740
grad_step = 000212, loss = 0.001723
grad_step = 000213, loss = 0.001733
grad_step = 000214, loss = 0.001736
grad_step = 000215, loss = 0.001729
grad_step = 000216, loss = 0.001721
grad_step = 000217, loss = 0.001724
grad_step = 000218, loss = 0.001730
grad_step = 000219, loss = 0.001722
grad_step = 000220, loss = 0.001712
grad_step = 000221, loss = 0.001713
grad_step = 000222, loss = 0.001714
grad_step = 000223, loss = 0.001712
grad_step = 000224, loss = 0.001705
grad_step = 000225, loss = 0.001701
grad_step = 000226, loss = 0.001704
grad_step = 000227, loss = 0.001705
grad_step = 000228, loss = 0.001705
grad_step = 000229, loss = 0.001707
grad_step = 000230, loss = 0.001716
grad_step = 000231, loss = 0.001736
grad_step = 000232, loss = 0.001773
grad_step = 000233, loss = 0.001798
grad_step = 000234, loss = 0.001830
grad_step = 000235, loss = 0.001794
grad_step = 000236, loss = 0.001744
grad_step = 000237, loss = 0.001703
grad_step = 000238, loss = 0.001701
grad_step = 000239, loss = 0.001724
grad_step = 000240, loss = 0.001732
grad_step = 000241, loss = 0.001715
grad_step = 000242, loss = 0.001686
grad_step = 000243, loss = 0.001682
grad_step = 000244, loss = 0.001707
grad_step = 000245, loss = 0.001717
grad_step = 000246, loss = 0.001699
grad_step = 000247, loss = 0.001679
grad_step = 000248, loss = 0.001679
grad_step = 000249, loss = 0.001683
grad_step = 000250, loss = 0.001678
grad_step = 000251, loss = 0.001673
grad_step = 000252, loss = 0.001678
grad_step = 000253, loss = 0.001683
grad_step = 000254, loss = 0.001676
grad_step = 000255, loss = 0.001664
grad_step = 000256, loss = 0.001658
grad_step = 000257, loss = 0.001662
grad_step = 000258, loss = 0.001663
grad_step = 000259, loss = 0.001659
grad_step = 000260, loss = 0.001656
grad_step = 000261, loss = 0.001658
grad_step = 000262, loss = 0.001661
grad_step = 000263, loss = 0.001659
grad_step = 000264, loss = 0.001654
grad_step = 000265, loss = 0.001651
grad_step = 000266, loss = 0.001651
grad_step = 000267, loss = 0.001651
grad_step = 000268, loss = 0.001649
grad_step = 000269, loss = 0.001645
grad_step = 000270, loss = 0.001645
grad_step = 000271, loss = 0.001647
grad_step = 000272, loss = 0.001650
grad_step = 000273, loss = 0.001654
grad_step = 000274, loss = 0.001663
grad_step = 000275, loss = 0.001678
grad_step = 000276, loss = 0.001709
grad_step = 000277, loss = 0.001742
grad_step = 000278, loss = 0.001787
grad_step = 000279, loss = 0.001789
grad_step = 000280, loss = 0.001766
grad_step = 000281, loss = 0.001695
grad_step = 000282, loss = 0.001640
grad_step = 000283, loss = 0.001628
grad_step = 000284, loss = 0.001656
grad_step = 000285, loss = 0.001688
grad_step = 000286, loss = 0.001683
grad_step = 000287, loss = 0.001653
grad_step = 000288, loss = 0.001621
grad_step = 000289, loss = 0.001616
grad_step = 000290, loss = 0.001636
grad_step = 000291, loss = 0.001651
grad_step = 000292, loss = 0.001644
grad_step = 000293, loss = 0.001620
grad_step = 000294, loss = 0.001605
grad_step = 000295, loss = 0.001610
grad_step = 000296, loss = 0.001622
grad_step = 000297, loss = 0.001625
grad_step = 000298, loss = 0.001615
grad_step = 000299, loss = 0.001603
grad_step = 000300, loss = 0.001598
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001601
grad_step = 000302, loss = 0.001604
grad_step = 000303, loss = 0.001604
grad_step = 000304, loss = 0.001601
grad_step = 000305, loss = 0.001596
grad_step = 000306, loss = 0.001591
grad_step = 000307, loss = 0.001588
grad_step = 000308, loss = 0.001586
grad_step = 000309, loss = 0.001587
grad_step = 000310, loss = 0.001587
grad_step = 000311, loss = 0.001588
grad_step = 000312, loss = 0.001585
grad_step = 000313, loss = 0.001581
grad_step = 000314, loss = 0.001577
grad_step = 000315, loss = 0.001573
grad_step = 000316, loss = 0.001572
grad_step = 000317, loss = 0.001570
grad_step = 000318, loss = 0.001569
grad_step = 000319, loss = 0.001568
grad_step = 000320, loss = 0.001567
grad_step = 000321, loss = 0.001567
grad_step = 000322, loss = 0.001567
grad_step = 000323, loss = 0.001567
grad_step = 000324, loss = 0.001566
grad_step = 000325, loss = 0.001564
grad_step = 000326, loss = 0.001564
grad_step = 000327, loss = 0.001565
grad_step = 000328, loss = 0.001568
grad_step = 000329, loss = 0.001575
grad_step = 000330, loss = 0.001588
grad_step = 000331, loss = 0.001606
grad_step = 000332, loss = 0.001640
grad_step = 000333, loss = 0.001675
grad_step = 000334, loss = 0.001731
grad_step = 000335, loss = 0.001743
grad_step = 000336, loss = 0.001742
grad_step = 000337, loss = 0.001663
grad_step = 000338, loss = 0.001583
grad_step = 000339, loss = 0.001535
grad_step = 000340, loss = 0.001547
grad_step = 000341, loss = 0.001591
grad_step = 000342, loss = 0.001612
grad_step = 000343, loss = 0.001597
grad_step = 000344, loss = 0.001552
grad_step = 000345, loss = 0.001527
grad_step = 000346, loss = 0.001537
grad_step = 000347, loss = 0.001564
grad_step = 000348, loss = 0.001574
grad_step = 000349, loss = 0.001558
grad_step = 000350, loss = 0.001533
grad_step = 000351, loss = 0.001525
grad_step = 000352, loss = 0.001531
grad_step = 000353, loss = 0.001540
grad_step = 000354, loss = 0.001535
grad_step = 000355, loss = 0.001523
grad_step = 000356, loss = 0.001510
grad_step = 000357, loss = 0.001507
grad_step = 000358, loss = 0.001511
grad_step = 000359, loss = 0.001516
grad_step = 000360, loss = 0.001515
grad_step = 000361, loss = 0.001510
grad_step = 000362, loss = 0.001509
grad_step = 000363, loss = 0.001518
grad_step = 000364, loss = 0.001518
grad_step = 000365, loss = 0.001523
grad_step = 000366, loss = 0.001509
grad_step = 000367, loss = 0.001499
grad_step = 000368, loss = 0.001490
grad_step = 000369, loss = 0.001484
grad_step = 000370, loss = 0.001483
grad_step = 000371, loss = 0.001484
grad_step = 000372, loss = 0.001484
grad_step = 000373, loss = 0.001481
grad_step = 000374, loss = 0.001479
grad_step = 000375, loss = 0.001478
grad_step = 000376, loss = 0.001476
grad_step = 000377, loss = 0.001473
grad_step = 000378, loss = 0.001471
grad_step = 000379, loss = 0.001471
grad_step = 000380, loss = 0.001477
grad_step = 000381, loss = 0.001501
grad_step = 000382, loss = 0.001490
grad_step = 000383, loss = 0.001506
grad_step = 000384, loss = 0.001522
grad_step = 000385, loss = 0.001561
grad_step = 000386, loss = 0.001533
grad_step = 000387, loss = 0.001514
grad_step = 000388, loss = 0.001498
grad_step = 000389, loss = 0.001495
grad_step = 000390, loss = 0.001525
grad_step = 000391, loss = 0.001558
grad_step = 000392, loss = 0.001579
grad_step = 000393, loss = 0.001567
grad_step = 000394, loss = 0.001543
grad_step = 000395, loss = 0.001495
grad_step = 000396, loss = 0.001459
grad_step = 000397, loss = 0.001446
grad_step = 000398, loss = 0.001465
grad_step = 000399, loss = 0.001495
grad_step = 000400, loss = 0.001514
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001509
grad_step = 000402, loss = 0.001548
grad_step = 000403, loss = 0.001516
grad_step = 000404, loss = 0.001523
grad_step = 000405, loss = 0.001496
grad_step = 000406, loss = 0.001476
grad_step = 000407, loss = 0.001473
grad_step = 000408, loss = 0.001474
grad_step = 000409, loss = 0.001472
grad_step = 000410, loss = 0.001483
grad_step = 000411, loss = 0.001524
grad_step = 000412, loss = 0.001471
grad_step = 000413, loss = 0.001554
grad_step = 000414, loss = 0.001621
grad_step = 000415, loss = 0.001603
grad_step = 000416, loss = 0.001590
grad_step = 000417, loss = 0.001477
grad_step = 000418, loss = 0.001491
grad_step = 000419, loss = 0.001501
grad_step = 000420, loss = 0.001500
grad_step = 000421, loss = 0.001591
grad_step = 000422, loss = 0.001462
grad_step = 000423, loss = 0.001519
grad_step = 000424, loss = 0.001454
grad_step = 000425, loss = 0.001534
grad_step = 000426, loss = 0.001515
grad_step = 000427, loss = 0.001426
grad_step = 000428, loss = 0.001447
grad_step = 000429, loss = 0.001430
grad_step = 000430, loss = 0.001532
grad_step = 000431, loss = 0.001557
grad_step = 000432, loss = 0.001507
grad_step = 000433, loss = 0.001448
grad_step = 000434, loss = 0.001427
grad_step = 000435, loss = 0.001492
grad_step = 000436, loss = 0.001464
grad_step = 000437, loss = 0.001434
grad_step = 000438, loss = 0.001406
grad_step = 000439, loss = 0.001406
grad_step = 000440, loss = 0.001432
grad_step = 000441, loss = 0.001420
grad_step = 000442, loss = 0.001409
grad_step = 000443, loss = 0.001386
grad_step = 000444, loss = 0.001392
grad_step = 000445, loss = 0.001414
grad_step = 000446, loss = 0.001411
grad_step = 000447, loss = 0.001404
grad_step = 000448, loss = 0.001402
grad_step = 000449, loss = 0.001379
grad_step = 000450, loss = 0.001413
grad_step = 000451, loss = 0.001426
grad_step = 000452, loss = 0.001414
grad_step = 000453, loss = 0.001406
grad_step = 000454, loss = 0.001368
grad_step = 000455, loss = 0.001383
grad_step = 000456, loss = 0.001390
grad_step = 000457, loss = 0.001378
grad_step = 000458, loss = 0.001384
grad_step = 000459, loss = 0.001351
grad_step = 000460, loss = 0.001359
grad_step = 000461, loss = 0.001371
grad_step = 000462, loss = 0.001365
grad_step = 000463, loss = 0.001367
grad_step = 000464, loss = 0.001343
grad_step = 000465, loss = 0.001355
grad_step = 000466, loss = 0.001362
grad_step = 000467, loss = 0.001350
grad_step = 000468, loss = 0.001355
grad_step = 000469, loss = 0.001341
grad_step = 000470, loss = 0.001358
grad_step = 000471, loss = 0.001383
grad_step = 000472, loss = 0.001391
grad_step = 000473, loss = 0.001412
grad_step = 000474, loss = 0.001422
grad_step = 000475, loss = 0.001436
grad_step = 000476, loss = 0.001469
grad_step = 000477, loss = 0.001422
grad_step = 000478, loss = 0.001381
grad_step = 000479, loss = 0.001332
grad_step = 000480, loss = 0.001316
grad_step = 000481, loss = 0.001340
grad_step = 000482, loss = 0.001361
grad_step = 000483, loss = 0.001354
grad_step = 000484, loss = 0.001329
grad_step = 000485, loss = 0.001304
grad_step = 000486, loss = 0.001300
grad_step = 000487, loss = 0.001315
grad_step = 000488, loss = 0.001327
grad_step = 000489, loss = 0.001321
grad_step = 000490, loss = 0.001313
grad_step = 000491, loss = 0.001295
grad_step = 000492, loss = 0.001285
grad_step = 000493, loss = 0.001284
grad_step = 000494, loss = 0.001287
grad_step = 000495, loss = 0.001293
grad_step = 000496, loss = 0.001295
grad_step = 000497, loss = 0.001292
grad_step = 000498, loss = 0.001287
grad_step = 000499, loss = 0.001280
grad_step = 000500, loss = 0.001270
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001264
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

  date_run                              2020-05-15 12:14:31.418126
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.24026
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 12:14:31.424935
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139663
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 12:14:31.433645
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.144366
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 12:14:31.439169
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.12222
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
0   2020-05-15 12:13:57.550402  ...    mean_absolute_error
1   2020-05-15 12:13:57.555491  ...     mean_squared_error
2   2020-05-15 12:13:57.559401  ...  median_absolute_error
3   2020-05-15 12:13:57.563673  ...               r2_score
4   2020-05-15 12:14:07.300015  ...    mean_absolute_error
5   2020-05-15 12:14:07.304391  ...     mean_squared_error
6   2020-05-15 12:14:07.308092  ...  median_absolute_error
7   2020-05-15 12:14:07.311831  ...               r2_score
8   2020-05-15 12:14:31.418126  ...    mean_absolute_error
9   2020-05-15 12:14:31.424935  ...     mean_squared_error
10  2020-05-15 12:14:31.433645  ...  median_absolute_error
11  2020-05-15 12:14:31.439169  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a1df98be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3031040/9912422 [00:00<00:00, 29565906.33it/s]9920512it [00:00, 33177379.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 600369.35it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476149.46it/s]1654784it [00:00, 11747972.91it/s]                         
0it [00:00, ?it/s]8192it [00:00, 164168.94it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69d0951e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cff810b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69d0951e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cfed80f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cd7134e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cd6ffc50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69d0951e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cfe96710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cd7134e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f69cff81128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f61e7808208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a3c07eefb922f8b5f73e56e62c718fef8b4cbf56351b4ac2bae6f4151c849190
  Stored in directory: /tmp/pip-ephem-wheel-cache-p99t_tiu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f61dd972048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2965504/17464789 [====>.........................] - ETA: 0s
11141120/17464789 [==================>...........] - ETA: 0s
16130048/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 12:15:58.933397: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 12:15:58.938135: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 12:15:58.938286: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5582ad9420d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 12:15:58.938302: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9350 - accuracy: 0.4825
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9120 - accuracy: 0.4840 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8276 - accuracy: 0.4895
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8077 - accuracy: 0.4908
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8251 - accuracy: 0.4897
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.8309 - accuracy: 0.4893
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7663 - accuracy: 0.4935
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7314 - accuracy: 0.4958
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 4s - loss: 7.6903 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6867 - accuracy: 0.4987
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6885 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6612 - accuracy: 0.5004
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 10s 390us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 12:16:16.003273
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 12:16:16.003273  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<75:32:24, 3.17kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<53:06:42, 4.51kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<37:13:35, 6.43kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<26:03:11, 9.18kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<18:10:57, 13.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.83M/862M [00:03<12:38:15, 18.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:03<8:47:55, 26.8kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.1M/862M [00:03<6:07:42, 38.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.5M/862M [00:03<4:15:48, 54.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:03<2:58:28, 77.9kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:03<2:04:19, 111kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:03<1:26:34, 159kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:03<1:00:23, 226kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:04<42:08, 323kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:04<29:26, 459kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:04<21:42, 622kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:06<17:02, 788kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:06<13:53, 966kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:06<10:11, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:08<09:37, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:08<08:14, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:08<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:10<07:14, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:10<06:25, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:10<04:50, 2.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:12<06:28, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:12<07:13, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:12<05:49, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:13<04:45, 2.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:14<7:58:04, 27.5kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:14<5:34:32, 39.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:16<3:55:49, 55.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:16<2:47:29, 78.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:16<1:57:44, 111kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:18<1:24:15, 155kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:18<1:01:51, 211kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:18<43:53, 297kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:18<30:47, 421kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:20<29:28, 440kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:20<21:56, 591kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:20<15:36, 828kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:22<13:56, 925kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<12:16, 1.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<09:08, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:22<06:32, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:24<13:17, 965kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:24<10:38, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:24<07:45, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:26<08:24, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:26<08:37, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:26<06:40, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<04:48, 2.64MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<1:24:36, 150kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<1:00:30, 210kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<42:32, 297kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<32:40, 386kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<25:25, 496kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<18:25, 684kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<14:52, 843kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<11:42, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<08:30, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:33<08:50, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<08:46, 1.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:46, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<04:52, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<11:51:14, 17.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<8:18:54, 24.9kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<5:48:49, 35.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<4:06:21, 50.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<2:53:37, 71.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<2:01:35, 101kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<1:27:46, 140kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<1:02:39, 196kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<44:02, 278kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<33:39, 363kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<26:03, 468kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<18:46, 649kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<13:14, 918kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<20:36, 589kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<15:40, 774kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<11:15, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<10:41, 1.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<09:56, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<07:33, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<05:24, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:47<32:45, 366kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<24:08, 497kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<17:08, 698kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<14:45, 808kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<12:47, 933kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<09:27, 1.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:45, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<12:05, 981kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<09:42, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<07:03, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<07:41, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<07:46, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:02, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<04:46, 2.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<7:28:18, 26.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<5:14:03, 37.3kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<3:40:18, 53.1kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<2:34:42, 75.6kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<1:48:50, 107kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<1:16:43, 152kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<1:02:47, 186kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<50:16, 232kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<36:53, 316kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<26:21, 442kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<18:57, 614kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<14:03, 827kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<10:19, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<07:42, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<17:54, 647kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<18:14, 636kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<14:02, 826kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<10:32, 1.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<07:48, 1.48MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<05:59, 1.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:42, 2.45MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<14:41, 785kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<15:54, 725kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<12:32, 919kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<09:17, 1.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<06:59, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<05:26, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:14, 2.70MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<27:50, 412kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<25:04, 457kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<18:52, 607kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<13:45, 831kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<10:02, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<07:29, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:46, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<2:47:05, 68.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<2:01:59, 93.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<1:26:36, 131kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<1:00:59, 186kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<43:06, 263kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<30:36, 370kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<26:28, 428kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<23:35, 480kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:07<17:45, 637kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<12:55, 874kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<09:30, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<07:06, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<10:08, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<12:08, 927kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<09:32, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<07:11, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:22, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<04:25, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<03:45, 2.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<08:08, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<10:39, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<08:31, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<06:28, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<04:58, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<04:04, 2.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<03:12, 3.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<12:41, 876kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<13:49, 804kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<10:42, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<07:52, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<06:19, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<04:48, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<03:42, 2.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<13:03, 846kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<14:02, 787kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<11:04, 997kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<08:15, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<06:13, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<04:48, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<08:07, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<10:34, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<08:26, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<06:24, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<05:09, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<03:56, 2.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<03:22, 3.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<07:31, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<09:42, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<07:58, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<06:01, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<04:38, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<03:40, 2.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<07:49, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<09:53, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<08:03, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<06:07, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<03:41, 2.92MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<08:34, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<10:22, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<08:21, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<06:19, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<04:49, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<03:45, 2.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<09:56, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<11:18, 946kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<08:57, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<06:39, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<05:02, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<03:53, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<11:46, 902kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<12:13, 869kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<09:33, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<05:17, 2.00MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<04:02, 2.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<38:56, 271kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<30:55, 341kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<22:24, 471kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<16:03, 656kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<11:33, 910kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<11:47, 889kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<11:41, 897kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<08:53, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<06:37, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:52, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<03:55, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<11:36, 898kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<14:28, 720kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<11:31, 904kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<08:31, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<06:11, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<04:47, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<09:00, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<09:06, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:59, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<05:13, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<03:53, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<12:59, 791kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<14:06, 729kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<11:14, 914kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<08:15, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<07:49, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<10:28, 976kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<08:34, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<06:17, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:36, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<09:10, 1.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<10:57, 926kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<08:37, 1.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<06:21, 1.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:37, 2.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<08:41, 1.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<10:12, 988kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<08:00, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<04:15, 2.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<19:36, 510kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<17:26, 574kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<13:06, 763kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<09:47, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:04, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:05, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<1:10:39, 141kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<52:56, 188kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<37:47, 263kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<26:38, 372kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<20:55, 472kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<18:20, 538kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<13:43, 718kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<09:48, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<09:06, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<09:21, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<07:11, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<05:11, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<5:48:24, 28.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<4:04:29, 39.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<2:50:49, 56.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<2:01:53, 79.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<1:31:15, 106kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<1:05:09, 148kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<45:49, 211kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<33:39, 285kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<26:20, 365kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<19:06, 502kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<13:31, 707kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<12:29, 763kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<11:03, 862kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:58<08:14, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:56, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<08:03, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<08:21, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<06:27, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<04:44, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<06:08, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<08:16, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<06:34, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:46, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<05:59, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<07:05, 1.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<05:33, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:02, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:02, 3.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<29:07, 318kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<23:12, 399kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<16:55, 546kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<11:57, 770kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<11:17, 813kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<10:32, 872kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<07:56, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<05:41, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<4:19:24, 35.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<3:04:09, 49.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<2:09:22, 70.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<1:30:25, 100kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<1:06:09, 137kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<48:55, 185kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<34:44, 260kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<24:27, 369kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:14<19:07, 469kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<15:58, 562kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<11:42, 766kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<08:20, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<08:51, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<08:36, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<06:39, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:48, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<06:21, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:18<06:43, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<05:16, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<03:50, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<05:57, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<06:25, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<05:03, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<03:40, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<05:50, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<06:34, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<05:06, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<05:36, 1.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<06:08, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<04:50, 1.78MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<03:32, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<05:46, 1.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<06:29, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<05:03, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<03:45, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<05:15, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:12, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:05, 2.73MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<05:25, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<06:04, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<04:50, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<03:31, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<05:15, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<05:49, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<04:37, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:21, 2.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<05:25, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<06:11, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:54, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<03:34, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<05:59, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<04:39, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:23, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<04:49, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<05:35, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<04:23, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<03:13, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<05:02, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<05:43, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<03:14, 2.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:51, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<05:29, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<04:21, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:11, 2.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<05:11, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<04:22, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:44<03:12, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<04:16, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<04:49, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<02:49, 2.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<05:21, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<05:38, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:12, 2.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<05:26, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<05:47, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<03:15, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<05:26, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<05:40, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<04:22, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<03:09, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:53, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<04:34, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:19, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<05:31, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<06:01, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:44, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<05:03, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<05:34, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<04:21, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<03:09, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<04:31, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<04:08, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:00, 2.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<05:17, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<05:24, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:13, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<03:03, 2.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<05:13, 1.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:37, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<04:22, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:10, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:18, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<05:04, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:04, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<02:57, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<04:28, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<04:57, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:55, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:51, 2.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<04:50, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<05:04, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:54, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:48, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<04:43, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<05:13, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<04:02, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:56, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<04:10, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<04:31, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<03:30, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:32, 2.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<04:24, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:15<04:40, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<03:39, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:40, 2.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<05:03, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<05:02, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:51, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:47, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<04:15, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<04:27, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:29, 2.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:15, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<3:54:48, 28.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<2:44:32, 40.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<1:54:38, 57.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<1:22:40, 79.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<1:00:50, 108kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<43:17, 151kB/s]  .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<30:21, 215kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<21:14, 306kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<24:58, 260kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<19:21, 335kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<14:01, 462kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<09:52, 653kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<09:16, 693kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<08:13, 781kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<06:10, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:26, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<05:06, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<05:47, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<04:32, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:17, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:27, 2.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<06:10, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<06:39, 943kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<05:13, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<03:45, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<04:14, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<05:07, 1.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<04:07, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:59, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<03:44, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<04:46, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<02:45, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<03:36, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<04:38, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:44, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<04:27, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:36, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:37, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<01:59, 2.99MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<08:32, 695kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<08:16, 718kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<06:15, 948kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<04:28, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<04:45, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<05:06, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<04:02, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:43, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<04:22, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<03:30, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<02:32, 2.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:27, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<04:09, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:26, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<03:20, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<04:11, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<03:22, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<02:28, 2.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:20, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<04:02, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<03:10, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<02:18, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<01:44, 3.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<14:32, 380kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<11:45, 469kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<08:38, 638kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<06:06, 897kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<05:53, 925kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<05:47, 941kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<04:24, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<03:12, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<03:49, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<04:20, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<03:26, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:19, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<03:09, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:17, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<03:12, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<03:00, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:11, 2.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<01:37, 3.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<10:32, 491kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<08:51, 584kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<06:34, 785kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<04:40, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<04:47, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<04:55, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<03:49, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<02:45, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<03:28, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<03:52, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<03:05, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<02:14, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<03:06, 1.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<03:41, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<02:53, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<02:08, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<01:33, 3.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<10:01, 489kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<08:31, 575kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<06:19, 773kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<04:28, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<04:34, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<04:40, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<03:38, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<03:14, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<03:42, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<02:53, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<02:06, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<01:32, 3.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<11:39, 402kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<09:35, 489kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<07:03, 664kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<04:59, 931kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<04:51, 952kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<04:48, 960kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<03:42, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<02:40, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<03:15, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<03:35, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:50, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:04, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:48, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<01:54, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<03:13, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:32, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:51, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<02:36, 1.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<03:04, 1.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:25, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:45, 2.45MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:18, 3.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<15:56, 268kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<12:23, 345kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<08:57, 476kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<06:18, 671kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<05:42, 736kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<05:13, 805kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<03:54, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:48, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<03:15, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<03:24, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:37, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<02:39, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<02:54, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<02:16, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:39, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<02:29, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<02:49, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<02:12, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:36, 2.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:33, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<2:21:05, 27.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<1:38:51, 39.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<1:08:42, 56.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<48:55, 79.1kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<35:17, 110kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<24:52, 155kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<17:19, 221kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<13:29, 282kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<10:22, 366kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<07:26, 510kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<05:12, 720kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<05:09, 724kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<05:26, 686kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<04:14, 879kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<03:03, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:46<02:56, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:46<03:42, 989kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<03:01, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<02:12, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:48<02:15, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<02:22, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<01:51, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:20, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<03:33, 991kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<03:14, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:50<02:36, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<01:52, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:22, 2.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<04:40, 740kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<04:09, 830kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<03:07, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<03:06, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<03:02, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:40, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<02:38, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<02:42, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<02:04, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<01:57, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<02:07, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:58<01:38, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:11, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<02:17, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<02:27, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:23, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<02:20, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<02:10, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:02<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:10, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<02:26, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<02:25, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:19, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<02:44, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<02:28, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<01:46, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<01:51, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<01:27, 1.98MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:02, 2.72MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<05:03, 559kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<04:08, 683kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<03:02, 928kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:07, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<11:21, 243kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<08:24, 328kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<05:57, 460kB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:14<04:34, 588kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<03:45, 717kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<02:44, 975kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<02:18, 1.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<02:02, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:38, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:11, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:24, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:09, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<00:49, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<02:14, 1.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<02:04, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:33, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:06, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<01:45, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<01:42, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<01:17, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<00:55, 2.57MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:43, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:41, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:24<01:17, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:15, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:11, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<00:54, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:01, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<02:15, 977kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:55, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:29, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:04, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<00:58, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<01:01, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<01:05, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<00:51, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<00:55, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<01:23, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:09, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:50, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:04, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:06, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<00:51, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<00:56, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<00:43, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:31, 3.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:17, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:13, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<00:55, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:55, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:57, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<00:47, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<01:10, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<00:57, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:44<00:42, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:53, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:54, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:42, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:48<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<00:36, 2.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:39, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<00:43, 2.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:33, 2.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:36, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:40, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:31, 2.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:34, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:38, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:29, 2.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<00:21, 3.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:46, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:45, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:34, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:24, 2.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:59, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:53, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:40, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:38, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:38, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:29, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:30, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:25, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:26, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:22, 2.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:15, 3.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:05<00:54, 979kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:47, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:34, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:23, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<01:24, 585kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<01:18, 635kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:58, 840kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:40, 1.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:39, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:35, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:26, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:17, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:55, 748kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:53, 777kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:40, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:27, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:28, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:26, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:19, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:18, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:18, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:13, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:19, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:11, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:12, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:13, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:09, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:06, 3.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:49, 416kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:38, 536kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:26, 740kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:18, 897kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:15, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:11, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:08, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:08, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:05, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:04, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:04, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:03, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:01, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:02, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:01, 2.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 3.47MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 397kB/s] .vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 797/400000 [00:00<00:50, 7964.98it/s]  0%|          | 1542/400000 [00:00<00:51, 7799.16it/s]  1%|          | 2325/400000 [00:00<00:50, 7806.32it/s]  1%|          | 3158/400000 [00:00<00:49, 7955.45it/s]  1%|          | 3976/400000 [00:00<00:49, 8019.82it/s]  1%|          | 4798/400000 [00:00<00:48, 8074.28it/s]  1%|▏         | 5615/400000 [00:00<00:48, 8100.29it/s]  2%|▏         | 6435/400000 [00:00<00:48, 8127.52it/s]  2%|▏         | 7270/400000 [00:00<00:47, 8191.23it/s]  2%|▏         | 8058/400000 [00:01<00:48, 8086.37it/s]  2%|▏         | 8880/400000 [00:01<00:48, 8125.45it/s]  2%|▏         | 9727/400000 [00:01<00:47, 8222.72it/s]  3%|▎         | 10550/400000 [00:01<00:47, 8220.81it/s]  3%|▎         | 11389/400000 [00:01<00:46, 8269.38it/s]  3%|▎         | 12213/400000 [00:01<00:46, 8258.87it/s]  3%|▎         | 13045/400000 [00:01<00:46, 8276.87it/s]  3%|▎         | 13875/400000 [00:01<00:46, 8282.44it/s]  4%|▎         | 14702/400000 [00:01<00:47, 8165.96it/s]  4%|▍         | 15559/400000 [00:01<00:46, 8281.57it/s]  4%|▍         | 16387/400000 [00:02<00:46, 8238.78it/s]  4%|▍         | 17211/400000 [00:02<00:46, 8173.01it/s]  5%|▍         | 18029/400000 [00:02<00:47, 8048.75it/s]  5%|▍         | 18848/400000 [00:02<00:47, 8089.96it/s]  5%|▍         | 19658/400000 [00:02<00:47, 8032.75it/s]  5%|▌         | 20462/400000 [00:02<00:47, 7930.63it/s]  5%|▌         | 21256/400000 [00:02<00:48, 7887.63it/s]  6%|▌         | 22046/400000 [00:02<00:48, 7735.35it/s]  6%|▌         | 22843/400000 [00:02<00:48, 7799.77it/s]  6%|▌         | 23630/400000 [00:02<00:48, 7818.59it/s]  6%|▌         | 24419/400000 [00:03<00:47, 7839.75it/s]  6%|▋         | 25219/400000 [00:03<00:47, 7886.24it/s]  7%|▋         | 26008/400000 [00:03<00:47, 7860.26it/s]  7%|▋         | 26814/400000 [00:03<00:47, 7918.83it/s]  7%|▋         | 27607/400000 [00:03<00:47, 7881.66it/s]  7%|▋         | 28412/400000 [00:03<00:46, 7930.88it/s]  7%|▋         | 29206/400000 [00:03<00:46, 7908.56it/s]  7%|▋         | 29998/400000 [00:03<00:46, 7894.23it/s]  8%|▊         | 30788/400000 [00:03<00:46, 7883.68it/s]  8%|▊         | 31577/400000 [00:03<00:47, 7718.77it/s]  8%|▊         | 32395/400000 [00:04<00:46, 7849.82it/s]  8%|▊         | 33200/400000 [00:04<00:46, 7906.88it/s]  8%|▊         | 33992/400000 [00:04<00:47, 7750.55it/s]  9%|▊         | 34769/400000 [00:04<00:47, 7722.95it/s]  9%|▉         | 35558/400000 [00:04<00:46, 7770.09it/s]  9%|▉         | 36356/400000 [00:04<00:46, 7831.67it/s]  9%|▉         | 37140/400000 [00:04<00:46, 7795.72it/s]  9%|▉         | 37942/400000 [00:04<00:46, 7861.06it/s] 10%|▉         | 38755/400000 [00:04<00:45, 7937.60it/s] 10%|▉         | 39570/400000 [00:04<00:45, 7998.75it/s] 10%|█         | 40371/400000 [00:05<00:45, 7975.30it/s] 10%|█         | 41169/400000 [00:05<00:45, 7914.74it/s] 10%|█         | 41961/400000 [00:05<00:45, 7908.40it/s] 11%|█         | 42758/400000 [00:05<00:45, 7924.62it/s] 11%|█         | 43551/400000 [00:05<00:45, 7916.61it/s] 11%|█         | 44365/400000 [00:05<00:44, 7980.14it/s] 11%|█▏        | 45164/400000 [00:05<00:45, 7772.35it/s] 11%|█▏        | 45961/400000 [00:05<00:45, 7829.91it/s] 12%|█▏        | 46746/400000 [00:05<00:45, 7727.23it/s] 12%|█▏        | 47520/400000 [00:05<00:45, 7698.26it/s] 12%|█▏        | 48291/400000 [00:06<00:48, 7266.41it/s] 12%|█▏        | 49066/400000 [00:06<00:47, 7404.62it/s] 12%|█▏        | 49854/400000 [00:06<00:46, 7539.49it/s] 13%|█▎        | 50670/400000 [00:06<00:45, 7713.28it/s] 13%|█▎        | 51459/400000 [00:06<00:44, 7765.05it/s] 13%|█▎        | 52269/400000 [00:06<00:44, 7861.09it/s] 13%|█▎        | 53104/400000 [00:06<00:43, 8001.20it/s] 13%|█▎        | 53907/400000 [00:06<00:43, 7958.15it/s] 14%|█▎        | 54724/400000 [00:06<00:43, 8019.86it/s] 14%|█▍        | 55545/400000 [00:06<00:42, 8075.56it/s] 14%|█▍        | 56354/400000 [00:07<00:42, 8055.18it/s] 14%|█▍        | 57161/400000 [00:07<00:42, 8058.58it/s] 14%|█▍        | 57968/400000 [00:07<00:43, 7880.83it/s] 15%|█▍        | 58790/400000 [00:07<00:42, 7979.30it/s] 15%|█▍        | 59631/400000 [00:07<00:42, 8103.23it/s] 15%|█▌        | 60446/400000 [00:07<00:41, 8113.66it/s] 15%|█▌        | 61259/400000 [00:07<00:41, 8100.92it/s] 16%|█▌        | 62070/400000 [00:07<00:41, 8100.16it/s] 16%|█▌        | 62888/400000 [00:07<00:41, 8123.46it/s] 16%|█▌        | 63709/400000 [00:08<00:41, 8148.75it/s] 16%|█▌        | 64525/400000 [00:08<00:41, 8125.89it/s] 16%|█▋        | 65338/400000 [00:08<00:41, 8055.50it/s] 17%|█▋        | 66144/400000 [00:08<00:41, 8042.23it/s] 17%|█▋        | 66976/400000 [00:08<00:41, 8121.14it/s] 17%|█▋        | 67802/400000 [00:08<00:40, 8160.78it/s] 17%|█▋        | 68633/400000 [00:08<00:40, 8203.13it/s] 17%|█▋        | 69461/400000 [00:08<00:40, 8223.22it/s] 18%|█▊        | 70284/400000 [00:08<00:40, 8185.24it/s] 18%|█▊        | 71111/400000 [00:08<00:40, 8209.11it/s] 18%|█▊        | 71938/400000 [00:09<00:39, 8224.84it/s] 18%|█▊        | 72773/400000 [00:09<00:39, 8260.46it/s] 18%|█▊        | 73600/400000 [00:09<00:39, 8173.93it/s] 19%|█▊        | 74418/400000 [00:09<00:40, 8097.41it/s] 19%|█▉        | 75229/400000 [00:09<00:40, 8022.74it/s] 19%|█▉        | 76032/400000 [00:09<00:40, 8023.07it/s] 19%|█▉        | 76842/400000 [00:09<00:40, 8043.67it/s] 19%|█▉        | 77661/400000 [00:09<00:39, 8084.61it/s] 20%|█▉        | 78473/400000 [00:09<00:39, 8093.72it/s] 20%|█▉        | 79313/400000 [00:09<00:39, 8181.22it/s] 20%|██        | 80133/400000 [00:10<00:39, 8183.50it/s] 20%|██        | 80968/400000 [00:10<00:38, 8232.12it/s] 20%|██        | 81803/400000 [00:10<00:38, 8266.18it/s] 21%|██        | 82630/400000 [00:10<00:39, 8131.95it/s] 21%|██        | 83461/400000 [00:10<00:38, 8184.34it/s] 21%|██        | 84280/400000 [00:10<00:38, 8120.85it/s] 21%|██▏       | 85093/400000 [00:10<00:38, 8107.08it/s] 21%|██▏       | 85905/400000 [00:10<00:38, 8080.26it/s] 22%|██▏       | 86714/400000 [00:10<00:39, 8030.23it/s] 22%|██▏       | 87518/400000 [00:10<00:39, 7886.44it/s] 22%|██▏       | 88308/400000 [00:11<00:40, 7696.75it/s] 22%|██▏       | 89092/400000 [00:11<00:40, 7737.74it/s] 22%|██▏       | 89907/400000 [00:11<00:39, 7855.14it/s] 23%|██▎       | 90711/400000 [00:11<00:39, 7908.66it/s] 23%|██▎       | 91548/400000 [00:11<00:38, 8041.04it/s] 23%|██▎       | 92366/400000 [00:11<00:38, 8081.58it/s] 23%|██▎       | 93199/400000 [00:11<00:37, 8153.39it/s] 24%|██▎       | 94016/400000 [00:11<00:37, 8122.39it/s] 24%|██▎       | 94829/400000 [00:11<00:38, 7995.90it/s] 24%|██▍       | 95643/400000 [00:11<00:37, 8038.02it/s] 24%|██▍       | 96455/400000 [00:12<00:37, 8059.73it/s] 24%|██▍       | 97283/400000 [00:12<00:37, 8124.07it/s] 25%|██▍       | 98096/400000 [00:12<00:37, 7979.04it/s] 25%|██▍       | 98902/400000 [00:12<00:37, 8000.90it/s] 25%|██▍       | 99758/400000 [00:12<00:36, 8157.70it/s] 25%|██▌       | 100575/400000 [00:12<00:36, 8154.85it/s] 25%|██▌       | 101395/400000 [00:12<00:36, 8166.29it/s] 26%|██▌       | 102224/400000 [00:12<00:36, 8202.65it/s] 26%|██▌       | 103045/400000 [00:12<00:36, 8035.40it/s] 26%|██▌       | 103878/400000 [00:12<00:36, 8118.75it/s] 26%|██▌       | 104691/400000 [00:13<00:36, 8111.01it/s] 26%|██▋       | 105524/400000 [00:13<00:36, 8174.72it/s] 27%|██▋       | 106344/400000 [00:13<00:35, 8179.56it/s] 27%|██▋       | 107163/400000 [00:13<00:35, 8170.77it/s] 27%|██▋       | 107982/400000 [00:13<00:35, 8174.94it/s] 27%|██▋       | 108800/400000 [00:13<00:35, 8150.56it/s] 27%|██▋       | 109629/400000 [00:13<00:35, 8191.23it/s] 28%|██▊       | 110455/400000 [00:13<00:35, 8209.40it/s] 28%|██▊       | 111277/400000 [00:13<00:35, 8176.66it/s] 28%|██▊       | 112095/400000 [00:13<00:35, 8044.74it/s] 28%|██▊       | 112901/400000 [00:14<00:35, 7995.83it/s] 28%|██▊       | 113703/400000 [00:14<00:35, 8000.75it/s] 29%|██▊       | 114530/400000 [00:14<00:35, 8078.00it/s] 29%|██▉       | 115339/400000 [00:14<00:35, 8002.55it/s] 29%|██▉       | 116155/400000 [00:14<00:35, 8046.85it/s] 29%|██▉       | 116961/400000 [00:14<00:35, 8022.52it/s] 29%|██▉       | 117769/400000 [00:14<00:35, 8039.52it/s] 30%|██▉       | 118574/400000 [00:14<00:34, 8041.33it/s] 30%|██▉       | 119379/400000 [00:14<00:35, 8007.10it/s] 30%|███       | 120208/400000 [00:14<00:34, 8089.80it/s] 30%|███       | 121028/400000 [00:15<00:34, 8122.09it/s] 30%|███       | 121851/400000 [00:15<00:34, 8152.79it/s] 31%|███       | 122674/400000 [00:15<00:33, 8173.96it/s] 31%|███       | 123497/400000 [00:15<00:33, 8188.03it/s] 31%|███       | 124327/400000 [00:15<00:33, 8217.70it/s] 31%|███▏      | 125157/400000 [00:15<00:33, 8241.62it/s] 31%|███▏      | 125997/400000 [00:15<00:33, 8288.22it/s] 32%|███▏      | 126826/400000 [00:15<00:33, 8247.11it/s] 32%|███▏      | 127655/400000 [00:15<00:32, 8258.28it/s] 32%|███▏      | 128481/400000 [00:15<00:33, 8152.21it/s] 32%|███▏      | 129313/400000 [00:16<00:33, 8198.86it/s] 33%|███▎      | 130137/400000 [00:16<00:32, 8209.02it/s] 33%|███▎      | 130959/400000 [00:16<00:32, 8178.16it/s] 33%|███▎      | 131790/400000 [00:16<00:32, 8214.25it/s] 33%|███▎      | 132612/400000 [00:16<00:32, 8212.36it/s] 33%|███▎      | 133434/400000 [00:16<00:33, 8069.52it/s] 34%|███▎      | 134269/400000 [00:16<00:32, 8150.81it/s] 34%|███▍      | 135104/400000 [00:16<00:32, 8207.62it/s] 34%|███▍      | 135935/400000 [00:16<00:32, 8236.12it/s] 34%|███▍      | 136760/400000 [00:17<00:32, 8103.55it/s] 34%|███▍      | 137572/400000 [00:17<00:32, 8043.55it/s] 35%|███▍      | 138404/400000 [00:17<00:32, 8122.99it/s] 35%|███▍      | 139249/400000 [00:17<00:31, 8217.57it/s] 35%|███▌      | 140090/400000 [00:17<00:31, 8272.23it/s] 35%|███▌      | 140918/400000 [00:17<00:31, 8215.64it/s] 35%|███▌      | 141741/400000 [00:17<00:31, 8207.11it/s] 36%|███▌      | 142580/400000 [00:17<00:31, 8260.49it/s] 36%|███▌      | 143407/400000 [00:17<00:31, 8244.09it/s] 36%|███▌      | 144232/400000 [00:17<00:31, 8240.33it/s] 36%|███▋      | 145057/400000 [00:18<00:31, 8188.68it/s] 36%|███▋      | 145888/400000 [00:18<00:30, 8224.15it/s] 37%|███▋      | 146711/400000 [00:18<00:30, 8199.74it/s] 37%|███▋      | 147553/400000 [00:18<00:30, 8263.99it/s] 37%|███▋      | 148380/400000 [00:18<00:30, 8228.61it/s] 37%|███▋      | 149204/400000 [00:18<00:30, 8205.85it/s] 38%|███▊      | 150025/400000 [00:18<00:30, 8162.58it/s] 38%|███▊      | 150844/400000 [00:18<00:30, 8169.22it/s] 38%|███▊      | 151685/400000 [00:18<00:30, 8238.04it/s] 38%|███▊      | 152510/400000 [00:18<00:30, 8128.50it/s] 38%|███▊      | 153324/400000 [00:19<00:30, 8088.32it/s] 39%|███▊      | 154139/400000 [00:19<00:30, 8105.20it/s] 39%|███▊      | 154950/400000 [00:19<00:30, 8092.84it/s] 39%|███▉      | 155760/400000 [00:19<00:30, 8059.83it/s] 39%|███▉      | 156567/400000 [00:19<00:30, 7982.64it/s] 39%|███▉      | 157366/400000 [00:19<00:30, 7948.10it/s] 40%|███▉      | 158162/400000 [00:19<00:30, 7860.11it/s] 40%|███▉      | 158949/400000 [00:19<00:31, 7743.42it/s] 40%|███▉      | 159739/400000 [00:19<00:30, 7787.43it/s] 40%|████      | 160524/400000 [00:19<00:30, 7805.31it/s] 40%|████      | 161305/400000 [00:20<00:30, 7722.80it/s] 41%|████      | 162098/400000 [00:20<00:30, 7783.51it/s] 41%|████      | 162877/400000 [00:20<00:30, 7779.72it/s] 41%|████      | 163683/400000 [00:20<00:30, 7859.90it/s] 41%|████      | 164509/400000 [00:20<00:29, 7974.27it/s] 41%|████▏     | 165308/400000 [00:20<00:29, 7902.81it/s] 42%|████▏     | 166103/400000 [00:20<00:29, 7915.37it/s] 42%|████▏     | 166912/400000 [00:20<00:29, 7964.27it/s] 42%|████▏     | 167726/400000 [00:20<00:28, 8014.07it/s] 42%|████▏     | 168532/400000 [00:20<00:28, 8027.34it/s] 42%|████▏     | 169335/400000 [00:21<00:28, 7968.86it/s] 43%|████▎     | 170133/400000 [00:21<00:29, 7906.59it/s] 43%|████▎     | 170933/400000 [00:21<00:28, 7932.18it/s] 43%|████▎     | 171754/400000 [00:21<00:28, 8011.30it/s] 43%|████▎     | 172556/400000 [00:21<00:28, 7980.68it/s] 43%|████▎     | 173355/400000 [00:21<00:28, 7927.35it/s] 44%|████▎     | 174149/400000 [00:21<00:29, 7706.93it/s] 44%|████▎     | 174950/400000 [00:21<00:28, 7793.80it/s] 44%|████▍     | 175753/400000 [00:21<00:28, 7860.80it/s] 44%|████▍     | 176562/400000 [00:21<00:28, 7926.16it/s] 44%|████▍     | 177356/400000 [00:22<00:28, 7850.71it/s] 45%|████▍     | 178142/400000 [00:22<00:28, 7753.57it/s] 45%|████▍     | 178925/400000 [00:22<00:28, 7772.09it/s] 45%|████▍     | 179748/400000 [00:22<00:27, 7902.37it/s] 45%|████▌     | 180558/400000 [00:22<00:27, 7960.08it/s] 45%|████▌     | 181355/400000 [00:22<00:27, 7940.33it/s] 46%|████▌     | 182150/400000 [00:22<00:27, 7878.11it/s] 46%|████▌     | 182939/400000 [00:22<00:27, 7866.24it/s] 46%|████▌     | 183726/400000 [00:22<00:27, 7834.51it/s] 46%|████▌     | 184529/400000 [00:22<00:27, 7889.99it/s] 46%|████▋     | 185319/400000 [00:23<00:27, 7833.52it/s] 47%|████▋     | 186112/400000 [00:23<00:27, 7860.11it/s] 47%|████▋     | 186916/400000 [00:23<00:26, 7912.01it/s] 47%|████▋     | 187747/400000 [00:23<00:26, 8026.73it/s] 47%|████▋     | 188563/400000 [00:23<00:26, 8064.85it/s] 47%|████▋     | 189370/400000 [00:23<00:26, 8050.04it/s] 48%|████▊     | 190176/400000 [00:23<00:26, 7930.71it/s] 48%|████▊     | 190970/400000 [00:23<00:26, 7894.77it/s] 48%|████▊     | 191767/400000 [00:23<00:26, 7916.26it/s] 48%|████▊     | 192559/400000 [00:23<00:26, 7911.78it/s] 48%|████▊     | 193351/400000 [00:24<00:26, 7870.33it/s] 49%|████▊     | 194139/400000 [00:24<00:26, 7844.82it/s] 49%|████▊     | 194924/400000 [00:24<00:26, 7839.40it/s] 49%|████▉     | 195709/400000 [00:24<00:26, 7775.64it/s] 49%|████▉     | 196489/400000 [00:24<00:26, 7781.86it/s] 49%|████▉     | 197288/400000 [00:24<00:25, 7839.72it/s] 50%|████▉     | 198073/400000 [00:24<00:25, 7805.84it/s] 50%|████▉     | 198856/400000 [00:24<00:25, 7812.95it/s] 50%|████▉     | 199641/400000 [00:24<00:25, 7821.93it/s] 50%|█████     | 200428/400000 [00:25<00:25, 7836.10it/s] 50%|█████     | 201212/400000 [00:25<00:25, 7816.85it/s] 51%|█████     | 202016/400000 [00:25<00:25, 7880.51it/s] 51%|█████     | 202805/400000 [00:25<00:25, 7847.63it/s] 51%|█████     | 203590/400000 [00:25<00:25, 7820.84it/s] 51%|█████     | 204383/400000 [00:25<00:24, 7851.02it/s] 51%|█████▏    | 205169/400000 [00:25<00:24, 7802.43it/s] 51%|█████▏    | 205977/400000 [00:25<00:24, 7883.40it/s] 52%|█████▏    | 206807/400000 [00:25<00:24, 8001.18it/s] 52%|█████▏    | 207617/400000 [00:25<00:23, 8027.85it/s] 52%|█████▏    | 208435/400000 [00:26<00:23, 8072.10it/s] 52%|█████▏    | 209243/400000 [00:26<00:23, 8011.40it/s] 53%|█████▎    | 210046/400000 [00:26<00:23, 8015.81it/s] 53%|█████▎    | 210891/400000 [00:26<00:23, 8137.66it/s] 53%|█████▎    | 211725/400000 [00:26<00:22, 8196.49it/s] 53%|█████▎    | 212559/400000 [00:26<00:22, 8238.96it/s] 53%|█████▎    | 213384/400000 [00:26<00:22, 8207.18it/s] 54%|█████▎    | 214206/400000 [00:26<00:22, 8122.86it/s] 54%|█████▍    | 215019/400000 [00:26<00:22, 8091.29it/s] 54%|█████▍    | 215829/400000 [00:26<00:22, 8089.30it/s] 54%|█████▍    | 216639/400000 [00:27<00:22, 8060.60it/s] 54%|█████▍    | 217446/400000 [00:27<00:22, 7994.93it/s] 55%|█████▍    | 218252/400000 [00:27<00:22, 8012.19it/s] 55%|█████▍    | 219074/400000 [00:27<00:22, 8072.41it/s] 55%|█████▍    | 219889/400000 [00:27<00:22, 8092.83it/s] 55%|█████▌    | 220714/400000 [00:27<00:22, 8136.61it/s] 55%|█████▌    | 221528/400000 [00:27<00:21, 8116.22it/s] 56%|█████▌    | 222370/400000 [00:27<00:21, 8203.19it/s] 56%|█████▌    | 223194/400000 [00:27<00:21, 8212.89it/s] 56%|█████▌    | 224016/400000 [00:27<00:21, 8195.21it/s] 56%|█████▌    | 224836/400000 [00:28<00:21, 8123.24it/s] 56%|█████▋    | 225649/400000 [00:28<00:21, 8034.83it/s] 57%|█████▋    | 226458/400000 [00:28<00:21, 8051.14it/s] 57%|█████▋    | 227277/400000 [00:28<00:21, 8092.11it/s] 57%|█████▋    | 228112/400000 [00:28<00:21, 8166.79it/s] 57%|█████▋    | 228940/400000 [00:28<00:20, 8199.80it/s] 57%|█████▋    | 229761/400000 [00:28<00:20, 8179.68it/s] 58%|█████▊    | 230587/400000 [00:28<00:20, 8201.50it/s] 58%|█████▊    | 231422/400000 [00:28<00:20, 8243.99it/s] 58%|█████▊    | 232247/400000 [00:28<00:20, 8227.36it/s] 58%|█████▊    | 233070/400000 [00:29<00:20, 8147.15it/s] 58%|█████▊    | 233885/400000 [00:29<00:20, 8064.67it/s] 59%|█████▊    | 234692/400000 [00:29<00:20, 8028.08it/s] 59%|█████▉    | 235499/400000 [00:29<00:20, 8039.69it/s] 59%|█████▉    | 236304/400000 [00:29<00:20, 7994.78it/s] 59%|█████▉    | 237104/400000 [00:29<00:20, 7937.45it/s] 59%|█████▉    | 237898/400000 [00:29<00:20, 7833.93it/s] 60%|█████▉    | 238682/400000 [00:29<00:20, 7773.11it/s] 60%|█████▉    | 239463/400000 [00:29<00:20, 7781.69it/s] 60%|██████    | 240269/400000 [00:29<00:20, 7862.93it/s] 60%|██████    | 241063/400000 [00:30<00:20, 7883.87it/s] 60%|██████    | 241852/400000 [00:30<00:20, 7823.17it/s] 61%|██████    | 242635/400000 [00:30<00:20, 7783.53it/s] 61%|██████    | 243440/400000 [00:30<00:19, 7856.68it/s] 61%|██████    | 244257/400000 [00:30<00:19, 7945.99it/s] 61%|██████▏   | 245053/400000 [00:30<00:19, 7939.02it/s] 61%|██████▏   | 245848/400000 [00:30<00:19, 7903.54it/s] 62%|██████▏   | 246639/400000 [00:30<00:19, 7843.14it/s] 62%|██████▏   | 247430/400000 [00:30<00:19, 7860.51it/s] 62%|██████▏   | 248217/400000 [00:30<00:19, 7857.58it/s] 62%|██████▏   | 249019/400000 [00:31<00:19, 7904.67it/s] 62%|██████▏   | 249810/400000 [00:31<00:19, 7862.51it/s] 63%|██████▎   | 250597/400000 [00:31<00:19, 7856.61it/s] 63%|██████▎   | 251409/400000 [00:31<00:18, 7933.12it/s] 63%|██████▎   | 252226/400000 [00:31<00:18, 8002.64it/s] 63%|██████▎   | 253043/400000 [00:31<00:18, 8050.45it/s] 63%|██████▎   | 253849/400000 [00:31<00:18, 7993.92it/s] 64%|██████▎   | 254649/400000 [00:31<00:18, 7921.03it/s] 64%|██████▍   | 255442/400000 [00:31<00:18, 7907.59it/s] 64%|██████▍   | 256234/400000 [00:31<00:18, 7757.26it/s] 64%|██████▍   | 257048/400000 [00:32<00:18, 7868.23it/s] 64%|██████▍   | 257849/400000 [00:32<00:17, 7909.02it/s] 65%|██████▍   | 258647/400000 [00:32<00:17, 7927.31it/s] 65%|██████▍   | 259441/400000 [00:32<00:17, 7919.63it/s] 65%|██████▌   | 260234/400000 [00:32<00:17, 7911.53it/s] 65%|██████▌   | 261045/400000 [00:32<00:17, 7967.17it/s] 65%|██████▌   | 261842/400000 [00:32<00:17, 7967.39it/s] 66%|██████▌   | 262641/400000 [00:32<00:17, 7972.89it/s] 66%|██████▌   | 263439/400000 [00:32<00:17, 7920.14it/s] 66%|██████▌   | 264232/400000 [00:32<00:17, 7895.53it/s] 66%|██████▋   | 265026/400000 [00:33<00:17, 7906.39it/s] 66%|██████▋   | 265817/400000 [00:33<00:17, 7884.67it/s] 67%|██████▋   | 266617/400000 [00:33<00:16, 7918.82it/s] 67%|██████▋   | 267422/400000 [00:33<00:16, 7955.99it/s] 67%|██████▋   | 268231/400000 [00:33<00:16, 7995.05it/s] 67%|██████▋   | 269036/400000 [00:33<00:16, 8010.08it/s] 67%|██████▋   | 269838/400000 [00:33<00:16, 7916.06it/s] 68%|██████▊   | 270641/400000 [00:33<00:16, 7949.15it/s] 68%|██████▊   | 271453/400000 [00:33<00:16, 7997.59it/s] 68%|██████▊   | 272254/400000 [00:33<00:16, 7915.18it/s] 68%|██████▊   | 273059/400000 [00:34<00:15, 7954.93it/s] 68%|██████▊   | 273867/400000 [00:34<00:15, 7990.95it/s] 69%|██████▊   | 274667/400000 [00:34<00:15, 7945.93it/s] 69%|██████▉   | 275462/400000 [00:34<00:15, 7916.55it/s] 69%|██████▉   | 276254/400000 [00:34<00:15, 7864.33it/s] 69%|██████▉   | 277058/400000 [00:34<00:15, 7914.92it/s] 69%|██████▉   | 277870/400000 [00:34<00:15, 7974.86it/s] 70%|██████▉   | 278670/400000 [00:34<00:15, 7980.70it/s] 70%|██████▉   | 279477/400000 [00:34<00:15, 8006.60it/s] 70%|███████   | 280278/400000 [00:35<00:14, 8003.01it/s] 70%|███████   | 281107/400000 [00:35<00:14, 8084.36it/s] 70%|███████   | 281926/400000 [00:35<00:14, 8114.60it/s] 71%|███████   | 282738/400000 [00:35<00:14, 8029.86it/s] 71%|███████   | 283542/400000 [00:35<00:14, 8027.54it/s] 71%|███████   | 284351/400000 [00:35<00:14, 8046.09it/s] 71%|███████▏  | 285156/400000 [00:35<00:14, 7765.29it/s] 71%|███████▏  | 285974/400000 [00:35<00:14, 7884.85it/s] 72%|███████▏  | 286789/400000 [00:35<00:14, 7958.28it/s] 72%|███████▏  | 287620/400000 [00:35<00:13, 8060.34it/s] 72%|███████▏  | 288429/400000 [00:36<00:13, 8068.14it/s] 72%|███████▏  | 289237/400000 [00:36<00:13, 8038.31it/s] 73%|███████▎  | 290052/400000 [00:36<00:13, 8069.47it/s] 73%|███████▎  | 290875/400000 [00:36<00:13, 8116.74it/s] 73%|███████▎  | 291731/400000 [00:36<00:13, 8243.28it/s] 73%|███████▎  | 292564/400000 [00:36<00:12, 8268.90it/s] 73%|███████▎  | 293406/400000 [00:36<00:12, 8312.83it/s] 74%|███████▎  | 294238/400000 [00:36<00:12, 8287.80it/s] 74%|███████▍  | 295068/400000 [00:36<00:12, 8170.38it/s] 74%|███████▍  | 295886/400000 [00:36<00:12, 8137.82it/s] 74%|███████▍  | 296701/400000 [00:37<00:12, 8068.88it/s] 74%|███████▍  | 297509/400000 [00:37<00:12, 7971.95it/s] 75%|███████▍  | 298307/400000 [00:37<00:12, 7946.64it/s] 75%|███████▍  | 299103/400000 [00:37<00:12, 7933.34it/s] 75%|███████▍  | 299901/400000 [00:37<00:12, 7943.41it/s] 75%|███████▌  | 300722/400000 [00:37<00:12, 8019.62it/s] 75%|███████▌  | 301525/400000 [00:37<00:12, 7972.54it/s] 76%|███████▌  | 302324/400000 [00:37<00:12, 7976.99it/s] 76%|███████▌  | 303122/400000 [00:37<00:12, 7823.88it/s] 76%|███████▌  | 303912/400000 [00:37<00:12, 7843.99it/s] 76%|███████▌  | 304707/400000 [00:38<00:12, 7874.45it/s] 76%|███████▋  | 305512/400000 [00:38<00:11, 7925.86it/s] 77%|███████▋  | 306305/400000 [00:38<00:11, 7923.80it/s] 77%|███████▋  | 307104/400000 [00:38<00:11, 7941.61it/s] 77%|███████▋  | 307909/400000 [00:38<00:11, 7971.62it/s] 77%|███████▋  | 308707/400000 [00:38<00:11, 7964.46it/s] 77%|███████▋  | 309511/400000 [00:38<00:11, 7985.13it/s] 78%|███████▊  | 310310/400000 [00:38<00:11, 7938.16it/s] 78%|███████▊  | 311104/400000 [00:38<00:11, 7937.63it/s] 78%|███████▊  | 311904/400000 [00:38<00:11, 7952.27it/s] 78%|███████▊  | 312700/400000 [00:39<00:10, 7945.36it/s] 78%|███████▊  | 313495/400000 [00:39<00:10, 7901.84it/s] 79%|███████▊  | 314301/400000 [00:39<00:10, 7947.26it/s] 79%|███████▉  | 315096/400000 [00:39<00:10, 7922.84it/s] 79%|███████▉  | 315911/400000 [00:39<00:10, 7986.37it/s] 79%|███████▉  | 316720/400000 [00:39<00:10, 8013.63it/s] 79%|███████▉  | 317523/400000 [00:39<00:10, 8017.24it/s] 80%|███████▉  | 318325/400000 [00:39<00:10, 7953.72it/s] 80%|███████▉  | 319121/400000 [00:39<00:10, 7878.82it/s] 80%|███████▉  | 319910/400000 [00:39<00:10, 7840.43it/s] 80%|████████  | 320695/400000 [00:40<00:10, 7766.14it/s] 80%|████████  | 321472/400000 [00:40<00:10, 7737.42it/s] 81%|████████  | 322247/400000 [00:40<00:10, 7720.04it/s] 81%|████████  | 323020/400000 [00:40<00:09, 7702.04it/s] 81%|████████  | 323791/400000 [00:40<00:09, 7648.87it/s] 81%|████████  | 324557/400000 [00:40<00:09, 7598.54it/s] 81%|████████▏ | 325327/400000 [00:40<00:09, 7627.82it/s] 82%|████████▏ | 326119/400000 [00:40<00:09, 7711.13it/s] 82%|████████▏ | 326891/400000 [00:40<00:09, 7654.73it/s] 82%|████████▏ | 327657/400000 [00:40<00:09, 7600.20it/s] 82%|████████▏ | 328448/400000 [00:41<00:09, 7689.21it/s] 82%|████████▏ | 329218/400000 [00:41<00:09, 7606.45it/s] 82%|████████▏ | 329986/400000 [00:41<00:09, 7625.96it/s] 83%|████████▎ | 330749/400000 [00:41<00:09, 7616.56it/s] 83%|████████▎ | 331551/400000 [00:41<00:08, 7731.80it/s] 83%|████████▎ | 332346/400000 [00:41<00:08, 7794.09it/s] 83%|████████▎ | 333163/400000 [00:41<00:08, 7900.93it/s] 83%|████████▎ | 333974/400000 [00:41<00:08, 7961.38it/s] 84%|████████▎ | 334771/400000 [00:41<00:08, 7958.42it/s] 84%|████████▍ | 335569/400000 [00:41<00:08, 7964.34it/s] 84%|████████▍ | 336395/400000 [00:42<00:07, 8049.64it/s] 84%|████████▍ | 337203/400000 [00:42<00:07, 8058.55it/s] 85%|████████▍ | 338019/400000 [00:42<00:07, 8088.71it/s] 85%|████████▍ | 338839/400000 [00:42<00:07, 8119.24it/s] 85%|████████▍ | 339655/400000 [00:42<00:07, 8131.01it/s] 85%|████████▌ | 340471/400000 [00:42<00:07, 8138.76it/s] 85%|████████▌ | 341314/400000 [00:42<00:07, 8222.07it/s] 86%|████████▌ | 342137/400000 [00:42<00:07, 8210.25it/s] 86%|████████▌ | 342959/400000 [00:42<00:07, 8099.22it/s] 86%|████████▌ | 343770/400000 [00:42<00:06, 8055.55it/s] 86%|████████▌ | 344581/400000 [00:43<00:06, 8069.75it/s] 86%|████████▋ | 345427/400000 [00:43<00:06, 8181.76it/s] 87%|████████▋ | 346285/400000 [00:43<00:06, 8297.26it/s] 87%|████████▋ | 347116/400000 [00:43<00:06, 8241.99it/s] 87%|████████▋ | 347941/400000 [00:43<00:06, 8218.80it/s] 87%|████████▋ | 348773/400000 [00:43<00:06, 8247.54it/s] 87%|████████▋ | 349609/400000 [00:43<00:06, 8278.48it/s] 88%|████████▊ | 350438/400000 [00:43<00:05, 8280.01it/s] 88%|████████▊ | 351267/400000 [00:43<00:05, 8224.87it/s] 88%|████████▊ | 352099/400000 [00:43<00:05, 8252.68it/s] 88%|████████▊ | 352925/400000 [00:44<00:05, 8218.61it/s] 88%|████████▊ | 353748/400000 [00:44<00:05, 8168.92it/s] 89%|████████▊ | 354566/400000 [00:44<00:05, 8163.15it/s] 89%|████████▉ | 355383/400000 [00:44<00:05, 8085.47it/s] 89%|████████▉ | 356206/400000 [00:44<00:05, 8127.86it/s] 89%|████████▉ | 357020/400000 [00:44<00:05, 8073.40it/s] 89%|████████▉ | 357828/400000 [00:44<00:05, 7997.84it/s] 90%|████████▉ | 358633/400000 [00:44<00:05, 8012.26it/s] 90%|████████▉ | 359435/400000 [00:44<00:05, 8004.46it/s] 90%|█████████ | 360261/400000 [00:45<00:04, 8077.68it/s] 90%|█████████ | 361105/400000 [00:45<00:04, 8181.71it/s] 90%|█████████ | 361940/400000 [00:45<00:04, 8227.69it/s] 91%|█████████ | 362764/400000 [00:45<00:04, 8201.13it/s] 91%|█████████ | 363585/400000 [00:45<00:04, 8103.92it/s] 91%|█████████ | 364412/400000 [00:45<00:04, 8151.11it/s] 91%|█████████▏| 365228/400000 [00:45<00:04, 8087.55it/s] 92%|█████████▏| 366056/400000 [00:45<00:04, 8143.02it/s] 92%|█████████▏| 366875/400000 [00:45<00:04, 8154.78it/s] 92%|█████████▏| 367691/400000 [00:45<00:03, 8087.09it/s] 92%|█████████▏| 368501/400000 [00:46<00:03, 8050.82it/s] 92%|█████████▏| 369330/400000 [00:46<00:03, 8119.32it/s] 93%|█████████▎| 370143/400000 [00:46<00:03, 8115.50it/s] 93%|█████████▎| 370955/400000 [00:46<00:03, 8086.73it/s] 93%|█████████▎| 371764/400000 [00:46<00:03, 7947.29it/s] 93%|█████████▎| 372560/400000 [00:46<00:03, 7942.29it/s] 93%|█████████▎| 373355/400000 [00:46<00:03, 7799.09it/s] 94%|█████████▎| 374136/400000 [00:46<00:03, 7779.80it/s] 94%|█████████▎| 374915/400000 [00:46<00:03, 7740.15it/s] 94%|█████████▍| 375690/400000 [00:46<00:03, 7660.48it/s] 94%|█████████▍| 376457/400000 [00:47<00:03, 7654.54it/s] 94%|█████████▍| 377234/400000 [00:47<00:02, 7688.49it/s] 95%|█████████▍| 378021/400000 [00:47<00:02, 7739.64it/s] 95%|█████████▍| 378819/400000 [00:47<00:02, 7808.46it/s] 95%|█████████▍| 379601/400000 [00:47<00:02, 7765.16it/s] 95%|█████████▌| 380385/400000 [00:47<00:02, 7785.45it/s] 95%|█████████▌| 381164/400000 [00:47<00:02, 7712.50it/s] 95%|█████████▌| 381936/400000 [00:47<00:02, 7693.70it/s] 96%|█████████▌| 382706/400000 [00:47<00:02, 7691.96it/s] 96%|█████████▌| 383479/400000 [00:47<00:02, 7700.68it/s] 96%|█████████▌| 384250/400000 [00:48<00:02, 7703.39it/s] 96%|█████████▋| 385044/400000 [00:48<00:01, 7772.03it/s] 96%|█████████▋| 385822/400000 [00:48<00:01, 7737.87it/s] 97%|█████████▋| 386596/400000 [00:48<00:01, 7685.38it/s] 97%|█████████▋| 387365/400000 [00:48<00:01, 7667.07it/s] 97%|█████████▋| 388132/400000 [00:48<00:01, 7629.10it/s] 97%|█████████▋| 388896/400000 [00:48<00:01, 7533.13it/s] 97%|█████████▋| 389670/400000 [00:48<00:01, 7592.38it/s] 98%|█████████▊| 390440/400000 [00:48<00:01, 7622.91it/s] 98%|█████████▊| 391203/400000 [00:48<00:01, 7599.82it/s] 98%|█████████▊| 391977/400000 [00:49<00:01, 7641.12it/s] 98%|█████████▊| 392742/400000 [00:49<00:00, 7632.10it/s] 98%|█████████▊| 393506/400000 [00:49<00:00, 7624.79it/s] 99%|█████████▊| 394269/400000 [00:49<00:00, 7347.68it/s] 99%|█████████▉| 395007/400000 [00:49<00:00, 7278.13it/s] 99%|█████████▉| 395750/400000 [00:49<00:00, 7321.75it/s] 99%|█████████▉| 396484/400000 [00:49<00:00, 7314.11it/s] 99%|█████████▉| 397217/400000 [00:49<00:00, 7127.83it/s] 99%|█████████▉| 397932/400000 [00:49<00:00, 7076.14it/s]100%|█████████▉| 398678/400000 [00:49<00:00, 7185.35it/s]100%|█████████▉| 399441/400000 [00:50<00:00, 7310.87it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7973.10it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6449b94c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011169771353403727 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011153210166306 	 Accuracy: 53

  model saves at 53% accuracy 

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
2020-05-15 12:25:26.996498: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 12:25:27.001444: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 12:25:27.001660: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e9b2115910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 12:25:27.001679: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f63f5bc3898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7203 - accuracy: 0.4965
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7356 - accuracy: 0.4955
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6544 - accuracy: 0.5008
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
11000/25000 [============>.................] - ETA: 4s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6879 - accuracy: 0.4986
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6388 - accuracy: 0.5018
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6352 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6314 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6279 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6346 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6446 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f63ae99a5f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f63b665a1d0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.0263 - crf_viterbi_accuracy: 0.6533 - val_loss: 0.9840 - val_crf_viterbi_accuracy: 0.6800

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

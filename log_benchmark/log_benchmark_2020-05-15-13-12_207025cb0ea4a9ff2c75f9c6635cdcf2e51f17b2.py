
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fbbf4deef28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 13:12:35.464373
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 13:12:35.468027
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 13:12:35.471512
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 13:12:35.474763
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fbc00bb8400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355388.0000
Epoch 2/10

1/1 [==============================] - 0s 101ms/step - loss: 269536.4062
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 182884.9688
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 109900.9688
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 63579.8867
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 37643.6914
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 23452.1641
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 15318.3301
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 10551.3525
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 7556.2944

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.70483655 -0.18211043  1.8385525  -0.3130398  -0.8898819   0.5272729
  -0.1562959  -0.28262028  1.1384238  -0.41872555 -0.22728962  1.3500434
   0.7406937  -1.3759193   0.38860756  0.86587214 -1.3663855   0.3698092
  -0.30005175  1.5920205   0.47414613 -0.07794271  0.96718585 -0.5255914
  -0.484347    0.06943238  0.6739532   0.17750445  1.3615234   0.76909816
   0.46402597 -0.5266707   1.9106398   1.6778824   0.6917167  -0.34937358
  -1.1068741   0.24547735  0.64937925  0.9221566   1.4749005  -0.37291166
  -0.09382245  1.3554361   0.2555446   0.54389954 -1.2629435  -0.59739
  -1.4575332  -0.8356168  -0.28520212 -0.7305697   0.04653192  0.11887075
  -1.6510212  -0.78218913  0.17042685 -0.5446569  -1.644614    0.02954644
   0.34492922  6.082378    8.757563    7.3255806   7.588416    7.529054
   6.6380787   7.0838566   6.8312473   6.2535176   7.1041594  10.200108
   7.0007415   7.6782384   6.761956    7.151676    7.102382    5.9047275
   5.9658694   8.094783    8.200373    7.7423577   7.361746    7.17237
   7.16933     6.672702    8.644601    6.7855663   6.099576    8.094781
   6.405801    7.8045897   6.5637274   7.231652    6.2450275   8.662485
   6.7264915   6.1719213   8.05992     7.6569476   9.2172575   7.572688
   6.851137    6.6343427   6.006054    6.799852    7.967907    7.71217
   6.5324936   7.7515993   5.795207    7.113588    5.5281105   9.748625
   8.950854    8.934628    7.7925773   8.320186    8.517812    7.6730356
   0.29635227  0.21641873 -0.19150352 -0.56977785  0.08201273  1.2782056
  -1.1891339  -0.6212822  -0.43918115  0.8259059   0.23269475  1.4437531
  -0.27969486  0.43644974 -0.0364905   0.03404439 -0.57760847  0.535574
  -0.7323429  -0.6669494   0.44899362 -0.9855832   1.3561301  -1.2964053
   0.08049925 -0.51649976  0.75921786  0.49504524  0.08153158  0.01330715
   0.3008223   0.34874797 -0.7733732   1.0190942  -0.6067325   0.83504665
  -0.11926766 -0.7613231   0.04753244  0.5554428   0.7894834  -0.7310068
   1.1162815  -0.21590215 -0.32880482 -1.5753511  -0.95981354 -0.15378496
  -0.26788166  0.9517209  -0.18602905  1.3011005  -0.91016185 -0.45349783
  -1.2383564  -1.970689   -0.7103117   0.29019123  0.95547944  0.9710075
   1.6957871   0.36589986  2.9816074   0.46523     0.4487456   1.2570239
   2.6233249   1.0979826   2.1683617   1.1317732   0.79310817  0.8526378
   1.8346922   3.5255523   1.1651546   0.50531065  0.5981969   0.30842292
   0.43922263  0.6161378   0.42811918  0.5076549   1.3278104   2.0119197
   0.26479155  2.0244815   2.2045963   1.3182263   2.211749    0.47529817
   0.38058895  1.936108    0.5703253   0.8848131   0.5215459   1.1295726
   0.3797921   0.58836937  1.1440747   2.4278584   0.36922473  1.9111664
   0.4686123   0.26041812  2.363986    1.8161864   0.9745354   1.2962785
   1.7411809   1.1961448   0.20484304  0.60910106  0.44121432  1.735544
   2.1874404   0.2263825   2.0310674   1.6898575   1.135647    0.86986065
   0.02334267  7.767559    7.947018    7.567803    9.167562    8.206221
   6.2502413   8.776375    7.1921077   9.342221    5.8063674   8.295832
   8.090774    7.2312527   8.67923     7.816933    8.758198    6.709485
   8.872125    9.035032    7.5763574   7.7727456   7.489539    7.4265037
   8.789009    8.042104    9.467075    7.1672463   7.836153    8.470484
   8.499468    7.008581    7.9990144   8.431839    6.6883163   8.99675
   9.432252    8.949824    6.8620524   8.298371    8.916303    8.948292
   8.401015    8.177344    7.712895    9.019182    6.295774    7.7811303
   9.481414    7.710785    7.51211     8.455777    8.170486    7.855387
   7.042425    8.759383    7.84907     6.716563    8.006793    7.99795
   1.3641473   0.6229608   0.8492627   0.73557234  2.073977    1.0649778
   2.6433396   1.0918216   0.5948564   0.39211786  0.503779    0.5977294
   0.5289077   0.63552254  1.6552908   1.0646409   1.6871177   1.5285733
   0.5256793   0.6471075   1.679148    0.70035386  1.6675612   0.8638673
   0.39284432  2.1076908   1.0414805   1.1335939   2.0741944   0.7760785
   1.6857069   2.0586514   2.6903915   0.4532627   0.18694878  0.81099063
   0.6694168   0.38247663  1.3283627   1.5746272   1.0264374   0.65255326
   0.16978532  1.0596488   2.1583638   0.4568491   0.7985084   0.6652228
   0.49305856  1.1110162   1.9411328   0.2078048   0.37270474  1.5015489
   0.55541384  0.21776432  1.0395765   0.7169611   1.603045    2.1402807
  -9.309863    2.7497551  -6.6074314 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 13:12:44.416718
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.762
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 13:12:44.420586
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8999.56
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 13:12:44.424861
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.0651
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 13:12:44.428738
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -804.975
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140444901605784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140443674448336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140443674448840
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140443674449344
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140443674449848
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140443674450352

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fbbe11fe438> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.519534
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.480851
grad_step = 000002, loss = 0.448942
grad_step = 000003, loss = 0.416716
grad_step = 000004, loss = 0.383090
grad_step = 000005, loss = 0.355332
grad_step = 000006, loss = 0.347826
grad_step = 000007, loss = 0.340343
grad_step = 000008, loss = 0.320301
grad_step = 000009, loss = 0.300457
grad_step = 000010, loss = 0.286986
grad_step = 000011, loss = 0.277154
grad_step = 000012, loss = 0.267714
grad_step = 000013, loss = 0.256715
grad_step = 000014, loss = 0.244328
grad_step = 000015, loss = 0.233257
grad_step = 000016, loss = 0.225178
grad_step = 000017, loss = 0.217818
grad_step = 000018, loss = 0.208552
grad_step = 000019, loss = 0.198216
grad_step = 000020, loss = 0.189031
grad_step = 000021, loss = 0.181213
grad_step = 000022, loss = 0.173565
grad_step = 000023, loss = 0.165642
grad_step = 000024, loss = 0.157593
grad_step = 000025, loss = 0.149686
grad_step = 000026, loss = 0.142313
grad_step = 000027, loss = 0.135703
grad_step = 000028, loss = 0.129375
grad_step = 000029, loss = 0.122558
grad_step = 000030, loss = 0.115552
grad_step = 000031, loss = 0.109333
grad_step = 000032, loss = 0.103901
grad_step = 000033, loss = 0.098527
grad_step = 000034, loss = 0.092971
grad_step = 000035, loss = 0.087591
grad_step = 000036, loss = 0.082646
grad_step = 000037, loss = 0.078161
grad_step = 000038, loss = 0.073863
grad_step = 000039, loss = 0.069490
grad_step = 000040, loss = 0.065214
grad_step = 000041, loss = 0.061393
grad_step = 000042, loss = 0.057916
grad_step = 000043, loss = 0.054453
grad_step = 000044, loss = 0.051066
grad_step = 000045, loss = 0.047978
grad_step = 000046, loss = 0.045137
grad_step = 000047, loss = 0.042417
grad_step = 000048, loss = 0.039780
grad_step = 000049, loss = 0.037265
grad_step = 000050, loss = 0.034967
grad_step = 000051, loss = 0.032825
grad_step = 000052, loss = 0.030742
grad_step = 000053, loss = 0.028789
grad_step = 000054, loss = 0.027004
grad_step = 000055, loss = 0.025310
grad_step = 000056, loss = 0.023701
grad_step = 000057, loss = 0.022179
grad_step = 000058, loss = 0.020755
grad_step = 000059, loss = 0.019440
grad_step = 000060, loss = 0.018193
grad_step = 000061, loss = 0.017016
grad_step = 000062, loss = 0.015922
grad_step = 000063, loss = 0.014889
grad_step = 000064, loss = 0.013809
grad_step = 000065, loss = 0.012824
grad_step = 000066, loss = 0.011952
grad_step = 000067, loss = 0.011171
grad_step = 000068, loss = 0.010429
grad_step = 000069, loss = 0.009733
grad_step = 000070, loss = 0.009064
grad_step = 000071, loss = 0.008421
grad_step = 000072, loss = 0.007866
grad_step = 000073, loss = 0.007344
grad_step = 000074, loss = 0.006839
grad_step = 000075, loss = 0.006382
grad_step = 000076, loss = 0.005968
grad_step = 000077, loss = 0.005580
grad_step = 000078, loss = 0.005202
grad_step = 000079, loss = 0.004878
grad_step = 000080, loss = 0.004593
grad_step = 000081, loss = 0.004319
grad_step = 000082, loss = 0.004066
grad_step = 000083, loss = 0.003844
grad_step = 000084, loss = 0.003647
grad_step = 000085, loss = 0.003457
grad_step = 000086, loss = 0.003294
grad_step = 000087, loss = 0.003150
grad_step = 000088, loss = 0.003018
grad_step = 000089, loss = 0.002902
grad_step = 000090, loss = 0.002800
grad_step = 000091, loss = 0.002707
grad_step = 000092, loss = 0.002624
grad_step = 000093, loss = 0.002557
grad_step = 000094, loss = 0.002495
grad_step = 000095, loss = 0.002441
grad_step = 000096, loss = 0.002396
grad_step = 000097, loss = 0.002357
grad_step = 000098, loss = 0.002320
grad_step = 000099, loss = 0.002290
grad_step = 000100, loss = 0.002266
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002245
grad_step = 000102, loss = 0.002225
grad_step = 000103, loss = 0.002207
grad_step = 000104, loss = 0.002192
grad_step = 000105, loss = 0.002179
grad_step = 000106, loss = 0.002168
grad_step = 000107, loss = 0.002156
grad_step = 000108, loss = 0.002146
grad_step = 000109, loss = 0.002137
grad_step = 000110, loss = 0.002128
grad_step = 000111, loss = 0.002119
grad_step = 000112, loss = 0.002111
grad_step = 000113, loss = 0.002103
grad_step = 000114, loss = 0.002095
grad_step = 000115, loss = 0.002088
grad_step = 000116, loss = 0.002080
grad_step = 000117, loss = 0.002073
grad_step = 000118, loss = 0.002066
grad_step = 000119, loss = 0.002058
grad_step = 000120, loss = 0.002051
grad_step = 000121, loss = 0.002044
grad_step = 000122, loss = 0.002036
grad_step = 000123, loss = 0.002029
grad_step = 000124, loss = 0.002021
grad_step = 000125, loss = 0.002013
grad_step = 000126, loss = 0.002005
grad_step = 000127, loss = 0.001996
grad_step = 000128, loss = 0.001987
grad_step = 000129, loss = 0.001977
grad_step = 000130, loss = 0.001967
grad_step = 000131, loss = 0.001957
grad_step = 000132, loss = 0.001945
grad_step = 000133, loss = 0.001933
grad_step = 000134, loss = 0.001926
grad_step = 000135, loss = 0.001915
grad_step = 000136, loss = 0.001894
grad_step = 000137, loss = 0.001875
grad_step = 000138, loss = 0.001862
grad_step = 000139, loss = 0.001905
grad_step = 000140, loss = 0.001954
grad_step = 000141, loss = 0.001880
grad_step = 000142, loss = 0.001842
grad_step = 000143, loss = 0.001828
grad_step = 000144, loss = 0.001826
grad_step = 000145, loss = 0.001816
grad_step = 000146, loss = 0.001790
grad_step = 000147, loss = 0.001743
grad_step = 000148, loss = 0.001755
grad_step = 000149, loss = 0.001951
grad_step = 000150, loss = 0.002054
grad_step = 000151, loss = 0.001710
grad_step = 000152, loss = 0.002085
grad_step = 000153, loss = 0.002216
grad_step = 000154, loss = 0.001953
grad_step = 000155, loss = 0.002075
grad_step = 000156, loss = 0.001952
grad_step = 000157, loss = 0.001948
grad_step = 000158, loss = 0.001966
grad_step = 000159, loss = 0.001868
grad_step = 000160, loss = 0.001917
grad_step = 000161, loss = 0.001861
grad_step = 000162, loss = 0.001856
grad_step = 000163, loss = 0.001850
grad_step = 000164, loss = 0.001811
grad_step = 000165, loss = 0.001831
grad_step = 000166, loss = 0.001793
grad_step = 000167, loss = 0.001787
grad_step = 000168, loss = 0.001792
grad_step = 000169, loss = 0.001745
grad_step = 000170, loss = 0.001780
grad_step = 000171, loss = 0.001718
grad_step = 000172, loss = 0.001756
grad_step = 000173, loss = 0.001707
grad_step = 000174, loss = 0.001725
grad_step = 000175, loss = 0.001703
grad_step = 000176, loss = 0.001692
grad_step = 000177, loss = 0.001700
grad_step = 000178, loss = 0.001666
grad_step = 000179, loss = 0.001689
grad_step = 000180, loss = 0.001649
grad_step = 000181, loss = 0.001671
grad_step = 000182, loss = 0.001639
grad_step = 000183, loss = 0.001645
grad_step = 000184, loss = 0.001631
grad_step = 000185, loss = 0.001625
grad_step = 000186, loss = 0.001620
grad_step = 000187, loss = 0.001609
grad_step = 000188, loss = 0.001605
grad_step = 000189, loss = 0.001594
grad_step = 000190, loss = 0.001592
grad_step = 000191, loss = 0.001579
grad_step = 000192, loss = 0.001576
grad_step = 000193, loss = 0.001569
grad_step = 000194, loss = 0.001561
grad_step = 000195, loss = 0.001551
grad_step = 000196, loss = 0.001549
grad_step = 000197, loss = 0.001538
grad_step = 000198, loss = 0.001538
grad_step = 000199, loss = 0.001530
grad_step = 000200, loss = 0.001527
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001523
grad_step = 000202, loss = 0.001522
grad_step = 000203, loss = 0.001527
grad_step = 000204, loss = 0.001541
grad_step = 000205, loss = 0.001549
grad_step = 000206, loss = 0.001565
grad_step = 000207, loss = 0.001539
grad_step = 000208, loss = 0.001497
grad_step = 000209, loss = 0.001461
grad_step = 000210, loss = 0.001456
grad_step = 000211, loss = 0.001474
grad_step = 000212, loss = 0.001487
grad_step = 000213, loss = 0.001490
grad_step = 000214, loss = 0.001467
grad_step = 000215, loss = 0.001441
grad_step = 000216, loss = 0.001416
grad_step = 000217, loss = 0.001400
grad_step = 000218, loss = 0.001396
grad_step = 000219, loss = 0.001399
grad_step = 000220, loss = 0.001419
grad_step = 000221, loss = 0.001471
grad_step = 000222, loss = 0.001621
grad_step = 000223, loss = 0.001719
grad_step = 000224, loss = 0.001758
grad_step = 000225, loss = 0.001419
grad_step = 000226, loss = 0.001416
grad_step = 000227, loss = 0.001604
grad_step = 000228, loss = 0.001406
grad_step = 000229, loss = 0.001403
grad_step = 000230, loss = 0.001509
grad_step = 000231, loss = 0.001352
grad_step = 000232, loss = 0.001404
grad_step = 000233, loss = 0.001466
grad_step = 000234, loss = 0.001325
grad_step = 000235, loss = 0.001382
grad_step = 000236, loss = 0.001468
grad_step = 000237, loss = 0.001329
grad_step = 000238, loss = 0.001323
grad_step = 000239, loss = 0.001403
grad_step = 000240, loss = 0.001325
grad_step = 000241, loss = 0.001286
grad_step = 000242, loss = 0.001327
grad_step = 000243, loss = 0.001304
grad_step = 000244, loss = 0.001264
grad_step = 000245, loss = 0.001272
grad_step = 000246, loss = 0.001280
grad_step = 000247, loss = 0.001248
grad_step = 000248, loss = 0.001227
grad_step = 000249, loss = 0.001253
grad_step = 000250, loss = 0.001248
grad_step = 000251, loss = 0.001206
grad_step = 000252, loss = 0.001207
grad_step = 000253, loss = 0.001219
grad_step = 000254, loss = 0.001211
grad_step = 000255, loss = 0.001208
grad_step = 000256, loss = 0.001187
grad_step = 000257, loss = 0.001165
grad_step = 000258, loss = 0.001168
grad_step = 000259, loss = 0.001171
grad_step = 000260, loss = 0.001167
grad_step = 000261, loss = 0.001164
grad_step = 000262, loss = 0.001154
grad_step = 000263, loss = 0.001136
grad_step = 000264, loss = 0.001127
grad_step = 000265, loss = 0.001120
grad_step = 000266, loss = 0.001111
grad_step = 000267, loss = 0.001109
grad_step = 000268, loss = 0.001110
grad_step = 000269, loss = 0.001111
grad_step = 000270, loss = 0.001123
grad_step = 000271, loss = 0.001154
grad_step = 000272, loss = 0.001197
grad_step = 000273, loss = 0.001260
grad_step = 000274, loss = 0.001263
grad_step = 000275, loss = 0.001212
grad_step = 000276, loss = 0.001096
grad_step = 000277, loss = 0.001051
grad_step = 000278, loss = 0.001094
grad_step = 000279, loss = 0.001123
grad_step = 000280, loss = 0.001086
grad_step = 000281, loss = 0.001031
grad_step = 000282, loss = 0.001034
grad_step = 000283, loss = 0.001067
grad_step = 000284, loss = 0.001060
grad_step = 000285, loss = 0.001024
grad_step = 000286, loss = 0.000995
grad_step = 000287, loss = 0.000992
grad_step = 000288, loss = 0.001007
grad_step = 000289, loss = 0.001025
grad_step = 000290, loss = 0.001032
grad_step = 000291, loss = 0.001018
grad_step = 000292, loss = 0.000992
grad_step = 000293, loss = 0.000965
grad_step = 000294, loss = 0.000947
grad_step = 000295, loss = 0.000939
grad_step = 000296, loss = 0.000939
grad_step = 000297, loss = 0.000945
grad_step = 000298, loss = 0.000953
grad_step = 000299, loss = 0.000965
grad_step = 000300, loss = 0.000976
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000990
grad_step = 000302, loss = 0.000989
grad_step = 000303, loss = 0.000980
grad_step = 000304, loss = 0.000948
grad_step = 000305, loss = 0.000909
grad_step = 000306, loss = 0.000880
grad_step = 000307, loss = 0.000870
grad_step = 000308, loss = 0.000877
grad_step = 000309, loss = 0.000891
grad_step = 000310, loss = 0.000902
grad_step = 000311, loss = 0.000901
grad_step = 000312, loss = 0.000892
grad_step = 000313, loss = 0.000872
grad_step = 000314, loss = 0.000852
grad_step = 000315, loss = 0.000833
grad_step = 000316, loss = 0.000820
grad_step = 000317, loss = 0.000812
grad_step = 000318, loss = 0.000807
grad_step = 000319, loss = 0.000806
grad_step = 000320, loss = 0.000808
grad_step = 000321, loss = 0.000819
grad_step = 000322, loss = 0.000842
grad_step = 000323, loss = 0.000892
grad_step = 000324, loss = 0.000952
grad_step = 000325, loss = 0.001041
grad_step = 000326, loss = 0.001011
grad_step = 000327, loss = 0.000917
grad_step = 000328, loss = 0.000779
grad_step = 000329, loss = 0.000773
grad_step = 000330, loss = 0.000857
grad_step = 000331, loss = 0.000849
grad_step = 000332, loss = 0.000778
grad_step = 000333, loss = 0.000739
grad_step = 000334, loss = 0.000779
grad_step = 000335, loss = 0.000804
grad_step = 000336, loss = 0.000754
grad_step = 000337, loss = 0.000716
grad_step = 000338, loss = 0.000727
grad_step = 000339, loss = 0.000752
grad_step = 000340, loss = 0.000760
grad_step = 000341, loss = 0.000734
grad_step = 000342, loss = 0.000706
grad_step = 000343, loss = 0.000689
grad_step = 000344, loss = 0.000687
grad_step = 000345, loss = 0.000693
grad_step = 000346, loss = 0.000702
grad_step = 000347, loss = 0.000705
grad_step = 000348, loss = 0.000697
grad_step = 000349, loss = 0.000678
grad_step = 000350, loss = 0.000658
grad_step = 000351, loss = 0.000645
grad_step = 000352, loss = 0.000640
grad_step = 000353, loss = 0.000637
grad_step = 000354, loss = 0.000633
grad_step = 000355, loss = 0.000630
grad_step = 000356, loss = 0.000632
grad_step = 000357, loss = 0.000642
grad_step = 000358, loss = 0.000669
grad_step = 000359, loss = 0.000708
grad_step = 000360, loss = 0.000778
grad_step = 000361, loss = 0.000799
grad_step = 000362, loss = 0.000821
grad_step = 000363, loss = 0.000732
grad_step = 000364, loss = 0.000645
grad_step = 000365, loss = 0.000600
grad_step = 000366, loss = 0.000619
grad_step = 000367, loss = 0.000663
grad_step = 000368, loss = 0.000660
grad_step = 000369, loss = 0.000623
grad_step = 000370, loss = 0.000583
grad_step = 000371, loss = 0.000577
grad_step = 000372, loss = 0.000598
grad_step = 000373, loss = 0.000610
grad_step = 000374, loss = 0.000596
grad_step = 000375, loss = 0.000564
grad_step = 000376, loss = 0.000545
grad_step = 000377, loss = 0.000551
grad_step = 000378, loss = 0.000567
grad_step = 000379, loss = 0.000577
grad_step = 000380, loss = 0.000574
grad_step = 000381, loss = 0.000569
grad_step = 000382, loss = 0.000559
grad_step = 000383, loss = 0.000544
grad_step = 000384, loss = 0.000526
grad_step = 000385, loss = 0.000515
grad_step = 000386, loss = 0.000514
grad_step = 000387, loss = 0.000517
grad_step = 000388, loss = 0.000520
grad_step = 000389, loss = 0.000523
grad_step = 000390, loss = 0.000530
grad_step = 000391, loss = 0.000541
grad_step = 000392, loss = 0.000559
grad_step = 000393, loss = 0.000567
grad_step = 000394, loss = 0.000583
grad_step = 000395, loss = 0.000568
grad_step = 000396, loss = 0.000559
grad_step = 000397, loss = 0.000527
grad_step = 000398, loss = 0.000500
grad_step = 000399, loss = 0.000480
grad_step = 000400, loss = 0.000473
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000475
grad_step = 000402, loss = 0.000482
grad_step = 000403, loss = 0.000487
grad_step = 000404, loss = 0.000486
grad_step = 000405, loss = 0.000481
grad_step = 000406, loss = 0.000472
grad_step = 000407, loss = 0.000463
grad_step = 000408, loss = 0.000455
grad_step = 000409, loss = 0.000450
grad_step = 000410, loss = 0.000448
grad_step = 000411, loss = 0.000448
grad_step = 000412, loss = 0.000449
grad_step = 000413, loss = 0.000452
grad_step = 000414, loss = 0.000456
grad_step = 000415, loss = 0.000463
grad_step = 000416, loss = 0.000478
grad_step = 000417, loss = 0.000490
grad_step = 000418, loss = 0.000513
grad_step = 000419, loss = 0.000505
grad_step = 000420, loss = 0.000506
grad_step = 000421, loss = 0.000477
grad_step = 000422, loss = 0.000458
grad_step = 000423, loss = 0.000437
grad_step = 000424, loss = 0.000424
grad_step = 000425, loss = 0.000418
grad_step = 000426, loss = 0.000416
grad_step = 000427, loss = 0.000420
grad_step = 000428, loss = 0.000427
grad_step = 000429, loss = 0.000435
grad_step = 000430, loss = 0.000440
grad_step = 000431, loss = 0.000445
grad_step = 000432, loss = 0.000439
grad_step = 000433, loss = 0.000434
grad_step = 000434, loss = 0.000422
grad_step = 000435, loss = 0.000413
grad_step = 000436, loss = 0.000404
grad_step = 000437, loss = 0.000398
grad_step = 000438, loss = 0.000395
grad_step = 000439, loss = 0.000393
grad_step = 000440, loss = 0.000394
grad_step = 000441, loss = 0.000396
grad_step = 000442, loss = 0.000401
grad_step = 000443, loss = 0.000407
grad_step = 000444, loss = 0.000420
grad_step = 000445, loss = 0.000431
grad_step = 000446, loss = 0.000456
grad_step = 000447, loss = 0.000458
grad_step = 000448, loss = 0.000476
grad_step = 000449, loss = 0.000455
grad_step = 000450, loss = 0.000448
grad_step = 000451, loss = 0.000423
grad_step = 000452, loss = 0.000405
grad_step = 000453, loss = 0.000387
grad_step = 000454, loss = 0.000377
grad_step = 000455, loss = 0.000372
grad_step = 000456, loss = 0.000372
grad_step = 000457, loss = 0.000376
grad_step = 000458, loss = 0.000383
grad_step = 000459, loss = 0.000392
grad_step = 000460, loss = 0.000398
grad_step = 000461, loss = 0.000405
grad_step = 000462, loss = 0.000402
grad_step = 000463, loss = 0.000402
grad_step = 000464, loss = 0.000389
grad_step = 000465, loss = 0.000379
grad_step = 000466, loss = 0.000368
grad_step = 000467, loss = 0.000361
grad_step = 000468, loss = 0.000357
grad_step = 000469, loss = 0.000356
grad_step = 000470, loss = 0.000356
grad_step = 000471, loss = 0.000358
grad_step = 000472, loss = 0.000363
grad_step = 000473, loss = 0.000368
grad_step = 000474, loss = 0.000378
grad_step = 000475, loss = 0.000388
grad_step = 000476, loss = 0.000406
grad_step = 000477, loss = 0.000409
grad_step = 000478, loss = 0.000422
grad_step = 000479, loss = 0.000410
grad_step = 000480, loss = 0.000405
grad_step = 000481, loss = 0.000385
grad_step = 000482, loss = 0.000371
grad_step = 000483, loss = 0.000360
grad_step = 000484, loss = 0.000353
grad_step = 000485, loss = 0.000350
grad_step = 000486, loss = 0.000350
grad_step = 000487, loss = 0.000352
grad_step = 000488, loss = 0.000356
grad_step = 000489, loss = 0.000360
grad_step = 000490, loss = 0.000363
grad_step = 000491, loss = 0.000366
grad_step = 000492, loss = 0.000366
grad_step = 000493, loss = 0.000365
grad_step = 000494, loss = 0.000361
grad_step = 000495, loss = 0.000356
grad_step = 000496, loss = 0.000349
grad_step = 000497, loss = 0.000344
grad_step = 000498, loss = 0.000341
grad_step = 000499, loss = 0.000340
grad_step = 000500, loss = 0.000341
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000344
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

  date_run                              2020-05-15 13:13:02.587790
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.301507
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 13:13:02.593483
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.257601
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 13:13:02.600442
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140071
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 13:13:02.605391
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.91434
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
0   2020-05-15 13:12:35.464373  ...    mean_absolute_error
1   2020-05-15 13:12:35.468027  ...     mean_squared_error
2   2020-05-15 13:12:35.471512  ...  median_absolute_error
3   2020-05-15 13:12:35.474763  ...               r2_score
4   2020-05-15 13:12:44.416718  ...    mean_absolute_error
5   2020-05-15 13:12:44.420586  ...     mean_squared_error
6   2020-05-15 13:12:44.424861  ...  median_absolute_error
7   2020-05-15 13:12:44.428738  ...               r2_score
8   2020-05-15 13:13:02.587790  ...    mean_absolute_error
9   2020-05-15 13:13:02.593483  ...     mean_squared_error
10  2020-05-15 13:13:02.600442  ...  median_absolute_error
11  2020-05-15 13:13:02.605391  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6e3f633cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 3022848/9912422 [00:00<00:00, 30226707.17it/s]9920512it [00:00, 34857092.37it/s]                             
0it [00:00, ?it/s]32768it [00:00, 546320.55it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162976.37it/s]1654784it [00:00, 11163115.64it/s]                         
0it [00:00, ?it/s]8192it [00:00, 235132.92it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6df1fece80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6df161d0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6df1fece80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6e3f63ea20> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6deedae4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6e3f63ea20> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6df1fece80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6e3f63ea20> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6deedae4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6e3f5f6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f60d1b0f208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8e495e3527f89c4ff938c350d6b47095b784f72ee6806f15fc7e8486271a7340
  Stored in directory: /tmp/pip-ephem-wheel-cache-0zj1wq_s/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f60c7c7a048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2531328/17464789 [===>..........................] - ETA: 0s
10412032/17464789 [================>.............] - ETA: 0s
16048128/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 13:14:28.889632: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 13:14:28.893555: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-15 13:14:28.893697: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561deb920040 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 13:14:28.893711: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7433 - accuracy: 0.4950
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6641 - accuracy: 0.5002
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 3s - loss: 7.6164 - accuracy: 0.5033
12000/25000 [=============>................] - ETA: 3s - loss: 7.6232 - accuracy: 0.5028
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6469 - accuracy: 0.5013
15000/25000 [=================>............] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6935 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6982 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6990 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7054 - accuracy: 0.4975
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7004 - accuracy: 0.4978
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6849 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6854 - accuracy: 0.4988
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 13:14:42.143048
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 13:14:42.143048  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:57:56, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:54:51, 16.1kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:00<10:30:03, 22.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 860k/862M [00:01<7:21:36, 32.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.47M/862M [00:01<5:08:24, 46.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.42M/862M [00:01<3:34:27, 66.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:29:43, 94.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<1:44:12, 135kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:01<1:12:32, 193kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:01<50:31, 275kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:02<35:13, 391kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:02<24:35, 556kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:02<17:12, 791kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:02<12:07, 1.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<08:36, 1.57MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<08:05, 1.66MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<07:31, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<05:43, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<06:32, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<06:11, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:06<04:44, 2.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<06:03, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<07:13, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.3M/862M [00:08<04:07, 3.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<15:17, 866kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<12:09, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:10<08:47, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<09:04, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<09:11, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:02, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<05:11, 2.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<07:19, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<06:42, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<05:05, 2.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<06:13, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<07:34, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:08, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:16<04:27, 2.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<10:33, 1.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<08:59, 1.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<06:40, 1.94MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<07:18, 1.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<08:19, 1.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<04:47, 2.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<11:19, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<09:30, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<06:58, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:22<05:01, 2.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<1:52:29, 113kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<1:21:52, 156kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<58:03, 219kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<40:41, 312kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<36:19, 349kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<26:58, 470kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<19:11, 660kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<15:58, 789kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:44, 989kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:13, 1.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<06:35, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<1:11:59, 174kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<51:55, 242kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<36:37, 342kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<28:05, 444kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<21:12, 588kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<15:08, 823kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<10:43, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<34:03, 364kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<25:22, 489kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<18:06, 683kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:09, 814kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<12:05, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:47, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:38, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:05, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:00, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<05:06, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:09, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:32, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:56, 2.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:55, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:19, 2.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:29, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:21, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:06, 2.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:18, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<04:58, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:51, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:07, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:32, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:18, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:53, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<10:15, 1.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:39, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:22, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:51, 1.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:14, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:43, 2.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:41, 2.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:52, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:32, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:02, 2.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:49, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:19, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:39, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:32, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:53, 1.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:18, 2.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:37, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:04, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:36, 2.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:32, 2.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:44, 1.69MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:25, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:57, 2.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:59, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:23, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:13, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:38, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:03, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:31, 2.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:28, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:37, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:19, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:51, 2.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:39, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:08, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:58, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:18, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<22:37, 490kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<18:34, 597kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<13:36, 814kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<09:37, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<12:45, 863kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:14, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:28, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:30, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:36, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:54, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:33, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<36:19, 300kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<26:41, 407kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<18:55, 574kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<13:19, 811kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:14:07, 146kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<54:18, 199kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<38:32, 280kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<27:01, 398kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<28:34, 376kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<21:18, 504kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<15:11, 705kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<12:46, 835kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<11:35, 921kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:46, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:16, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<10:57, 968kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:58, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<06:33, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:44, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:28, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:16, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:59, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:48, 2.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:46, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:58, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:49, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:31, 2.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:01, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:28, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<05:30, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:04, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:47, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:23, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:54, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:12, 1.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:43, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:38, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<04:04, 2.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<21:49, 464kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<17:46, 569kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<12:58, 779kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<09:10, 1.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<11:43, 857kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:27, 1.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<06:54, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:51, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:15, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:38, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<04:02, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:34, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:11, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:19, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:43, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:13, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:54, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:15, 2.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<7:17:26, 22.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<5:06:24, 31.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<3:33:33, 45.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<2:37:56, 61.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<1:52:41, 86.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<1:19:15, 122kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<55:25, 174kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<42:44, 225kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<30:59, 311kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<21:52, 439kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<17:20, 552kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<14:29, 660kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<10:38, 898kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<07:34, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:23, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:02, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:12, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:13, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:56, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:35, 2.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:23, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:01, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:11, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:31, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:10, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:54, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:32, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:12, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:46, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:57, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:27, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:06, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:45, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:27, 2.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:42, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:07, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:49, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:30, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:15, 2.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:06, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:59, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:03, 2.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:56, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:51, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:55, 3.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<03:51, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<02:54, 3.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:49, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:45, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:53, 3.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:47, 2.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:47, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<02:46, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:01, 1.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:57, 1.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:25, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:49, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:26, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:21, 2.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:03, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:56, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<02:50, 2.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:09, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:37, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:27, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:06, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:45, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:43, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:43, 3.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:24, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:46, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:35, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:10, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:57, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:54, 2.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:50, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:11, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:34, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:23, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:06, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:49, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:47, 2.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:38, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:13, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:09, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:48, 2.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:38, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:46, 2.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:37, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<03:25, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:34, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:29, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:21, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:27, 2.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<02:32, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:52, 2.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:33, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:24, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:36, 2.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:22, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:17, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<02:31, 2.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:17, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:28, 3.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:52, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:36, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:45, 2.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:18, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:14, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:04, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:21, 3.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:50, 1.49MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:16, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<03:10, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:17, 3.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<25:27, 281kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<19:35, 364kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<14:03, 507kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<09:55, 715kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<09:03, 781kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<07:03, 1.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:08, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:01, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:23, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:41, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:26, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:37, 2.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:13, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:05, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:22, 2.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:02, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:48, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:00, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:13, 3.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:29, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:16, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:27, 2.72MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:59, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:17, 2.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:56, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:42, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<02:09, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:47, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:26, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:34, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:06, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:47, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:10, 2.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:08, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:41, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:46, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:13, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:48, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:03, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:12, 2.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:07, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:21, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:13, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:30, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:04, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:18, 2.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<01:42, 3.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:04, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:04, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:55, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:45, 1.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:46, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:31, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:40, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:18, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:29, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:02, 2.90MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<4:28:09, 22.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<3:07:40, 31.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<2:10:23, 44.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<1:36:16, 60.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<1:08:39, 84.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<48:14, 120kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<33:37, 172kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<26:33, 217kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<19:17, 298kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<13:35, 422kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<10:37, 536kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<08:07, 699kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:49, 972kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:11, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:18, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:09, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:20, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:01, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:16, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:43, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:16, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:34, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:52, 2.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:09, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:48, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:05, 2.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:37, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:29, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:54, 2.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:24, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:20, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:46, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:17, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:53, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:18, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:41, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:52, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:37, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:58, 2.60MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:24, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:56, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:19, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:41, 2.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:52, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:37, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:57, 2.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:22, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:16, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:44, 2.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:12, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:44, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:13, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:36, 2.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:54, 1.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:18, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:26, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:39, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:02, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:24, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:44, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:53, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:18, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:26, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:37, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:24, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:48, 2.51MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:10, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:32, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:01, 2.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:27, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:26, 816kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:21, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:09, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<02:14, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<2:12:24, 33.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<1:33:42, 46.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<1:05:40, 66.4kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<45:35, 94.7kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<34:18, 125kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<24:30, 175kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<17:11, 249kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<13:01, 325kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<11:14, 376kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<08:23, 504kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:58, 703kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:34, 917kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<03:21, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<4:09:20, 16.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<2:54:45, 23.7kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<2:01:48, 33.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<1:25:14, 47.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<1:00:05, 67.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<41:57, 96.6kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<29:53, 134kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<21:54, 183kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<15:30, 258kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<10:49, 366kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<09:02, 436kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<06:48, 579kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:51, 806kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:09, 932kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<03:51, 1.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:54, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:04, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:40, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:18, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:43, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<01:59, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<02:14, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:44, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:16, 2.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:45<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:23, 2.60MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:42, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:38, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:14, 2.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:35, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<01:58, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<01:33, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:08, 3.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:50, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:43, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:17, 2.66MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:34, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:31, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:08, 2.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:28, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:50, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:28, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:04, 3.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:54, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:44, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:18, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:33, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:28, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:06, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:24, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:22, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:02, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:20, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:43, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:21, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:00, 2.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:30, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:04, 2.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:46, 3.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<22:14, 130kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<15:53, 182kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<11:07, 258kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<08:15, 343kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<06:28, 436kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<04:40, 603kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:16, 850kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:14, 851kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:36, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:53, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:48, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<02:35, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:08, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:33, 1.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:34, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:25, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:03, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:15, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:11, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:51, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:20, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:14, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<00:55, 2.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:40, 3.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<06:13, 386kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<04:38, 516kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<03:17, 724kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:17, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<14:16, 163kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<10:15, 227kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<07:10, 322kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<05:23, 419kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<04:02, 558kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:51, 782kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:25, 908kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:57, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:24, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:59, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<43:05, 49.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<30:39, 69.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<21:27, 98.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<14:49, 140kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<10:59, 187kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<07:55, 259kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<05:32, 366kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:12, 472kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<03:11, 622kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:15, 869kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:56, 988kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:50, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:22, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:58, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:48, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:34, 3.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<1:46:58, 16.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<1:15:14, 23.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<52:27, 33.7kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<35:52, 48.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<25:53, 66.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<18:16, 93.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<12:40, 133kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<08:59, 182kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<06:38, 246kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<04:42, 345kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<03:13, 490kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:55, 401kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:54, 538kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:02, 754kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:43, 872kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:23, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:59, 1.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:41, 2.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<10:00, 143kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<07:09, 199kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<04:58, 282kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<03:39, 372kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:51, 475kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<02:03, 657kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:25, 926kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:27, 885kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:10, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:50, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:49, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:50, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:27, 2.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:43, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:37, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:27, 2.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:19, 3.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:55, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:46, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:35, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:32, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:23, 2.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:27, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:33, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:26, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:18, 2.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:42, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:35, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:25, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:32, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:07<00:25, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:17, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:36, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:27, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:18, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:25, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:29, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:18, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:20, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:10, 2.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:40, 685kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:31, 880kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:21, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:18, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:19, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:18, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:09, 694kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 882kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.63MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 859/400000 [00:00<00:46, 8588.82it/s]  0%|          | 1700/400000 [00:00<00:46, 8532.36it/s]  1%|          | 2559/400000 [00:00<00:46, 8549.24it/s]  1%|          | 3422/400000 [00:00<00:46, 8571.10it/s]  1%|          | 4224/400000 [00:00<00:47, 8396.64it/s]  1%|▏         | 5065/400000 [00:00<00:47, 8399.84it/s]  1%|▏         | 5929/400000 [00:00<00:46, 8469.78it/s]  2%|▏         | 6794/400000 [00:00<00:46, 8522.18it/s]  2%|▏         | 7645/400000 [00:00<00:46, 8516.30it/s]  2%|▏         | 8489/400000 [00:01<00:46, 8492.49it/s]  2%|▏         | 9356/400000 [00:01<00:45, 8544.53it/s]  3%|▎         | 10217/400000 [00:01<00:45, 8563.02it/s]  3%|▎         | 11082/400000 [00:01<00:45, 8588.36it/s]  3%|▎         | 11933/400000 [00:01<00:45, 8545.97it/s]  3%|▎         | 12793/400000 [00:01<00:45, 8559.82it/s]  3%|▎         | 13646/400000 [00:01<00:45, 8453.78it/s]  4%|▎         | 14509/400000 [00:01<00:45, 8505.79it/s]  4%|▍         | 15358/400000 [00:01<00:45, 8472.80it/s]  4%|▍         | 16207/400000 [00:01<00:45, 8475.81it/s]  4%|▍         | 17058/400000 [00:02<00:45, 8485.72it/s]  4%|▍         | 17924/400000 [00:02<00:44, 8537.02it/s]  5%|▍         | 18778/400000 [00:02<00:44, 8524.51it/s]  5%|▍         | 19643/400000 [00:02<00:44, 8559.93it/s]  5%|▌         | 20509/400000 [00:02<00:44, 8587.46it/s]  5%|▌         | 21368/400000 [00:02<00:44, 8523.24it/s]  6%|▌         | 22228/400000 [00:02<00:44, 8543.57it/s]  6%|▌         | 23083/400000 [00:02<00:44, 8533.56it/s]  6%|▌         | 23949/400000 [00:02<00:43, 8568.94it/s]  6%|▌         | 24818/400000 [00:02<00:43, 8603.33it/s]  6%|▋         | 25680/400000 [00:03<00:43, 8605.45it/s]  7%|▋         | 26541/400000 [00:03<00:43, 8555.05it/s]  7%|▋         | 27408/400000 [00:03<00:43, 8587.08it/s]  7%|▋         | 28275/400000 [00:03<00:43, 8609.22it/s]  7%|▋         | 29137/400000 [00:03<00:43, 8534.24it/s]  7%|▋         | 29997/400000 [00:03<00:43, 8551.29it/s]  8%|▊         | 30853/400000 [00:03<00:43, 8481.47it/s]  8%|▊         | 31702/400000 [00:03<00:43, 8445.81it/s]  8%|▊         | 32565/400000 [00:03<00:43, 8499.79it/s]  8%|▊         | 33432/400000 [00:03<00:42, 8549.56it/s]  9%|▊         | 34288/400000 [00:04<00:42, 8515.71it/s]  9%|▉         | 35140/400000 [00:04<00:43, 8463.14it/s]  9%|▉         | 35998/400000 [00:04<00:42, 8497.52it/s]  9%|▉         | 36860/400000 [00:04<00:42, 8531.17it/s]  9%|▉         | 37716/400000 [00:04<00:42, 8538.16it/s] 10%|▉         | 38570/400000 [00:04<00:42, 8514.57it/s] 10%|▉         | 39427/400000 [00:04<00:42, 8528.79it/s] 10%|█         | 40291/400000 [00:04<00:42, 8559.98it/s] 10%|█         | 41157/400000 [00:04<00:41, 8586.83it/s] 11%|█         | 42016/400000 [00:04<00:41, 8567.05it/s] 11%|█         | 42883/400000 [00:05<00:41, 8595.97it/s] 11%|█         | 43750/400000 [00:05<00:41, 8615.94it/s] 11%|█         | 44625/400000 [00:05<00:41, 8653.86it/s] 11%|█▏        | 45491/400000 [00:05<00:41, 8637.13it/s] 12%|█▏        | 46355/400000 [00:05<00:40, 8627.11it/s] 12%|█▏        | 47218/400000 [00:05<00:40, 8612.52it/s] 12%|█▏        | 48083/400000 [00:05<00:40, 8621.93it/s] 12%|█▏        | 48960/400000 [00:05<00:40, 8663.10it/s] 12%|█▏        | 49827/400000 [00:05<00:40, 8630.33it/s] 13%|█▎        | 50691/400000 [00:05<00:40, 8631.89it/s] 13%|█▎        | 51628/400000 [00:06<00:39, 8840.10it/s] 13%|█▎        | 52514/400000 [00:06<00:39, 8792.41it/s] 13%|█▎        | 53395/400000 [00:06<00:39, 8755.58it/s] 14%|█▎        | 54272/400000 [00:06<00:39, 8748.58it/s] 14%|█▍        | 55148/400000 [00:06<00:39, 8726.59it/s] 14%|█▍        | 56022/400000 [00:06<00:40, 8547.00it/s] 14%|█▍        | 56878/400000 [00:06<00:41, 8339.54it/s] 14%|█▍        | 57746/400000 [00:06<00:40, 8437.19it/s] 15%|█▍        | 58615/400000 [00:06<00:40, 8510.80it/s] 15%|█▍        | 59477/400000 [00:06<00:39, 8540.49it/s] 15%|█▌        | 60338/400000 [00:07<00:39, 8560.66it/s] 15%|█▌        | 61195/400000 [00:07<00:40, 8403.99it/s] 16%|█▌        | 62064/400000 [00:07<00:39, 8485.96it/s] 16%|█▌        | 62932/400000 [00:07<00:39, 8540.62it/s] 16%|█▌        | 63795/400000 [00:07<00:39, 8566.72it/s] 16%|█▌        | 64661/400000 [00:07<00:39, 8594.39it/s] 16%|█▋        | 65521/400000 [00:07<00:39, 8575.59it/s] 17%|█▋        | 66379/400000 [00:07<00:38, 8572.68it/s] 17%|█▋        | 67237/400000 [00:07<00:38, 8573.59it/s] 17%|█▋        | 68107/400000 [00:07<00:38, 8611.05it/s] 17%|█▋        | 68970/400000 [00:08<00:38, 8614.17it/s] 17%|█▋        | 69832/400000 [00:08<00:39, 8410.50it/s] 18%|█▊        | 70699/400000 [00:08<00:38, 8484.78it/s] 18%|█▊        | 71568/400000 [00:08<00:38, 8544.34it/s] 18%|█▊        | 72432/400000 [00:08<00:38, 8572.11it/s] 18%|█▊        | 73299/400000 [00:08<00:37, 8600.50it/s] 19%|█▊        | 74160/400000 [00:08<00:37, 8595.43it/s] 19%|█▉        | 75028/400000 [00:08<00:37, 8619.45it/s] 19%|█▉        | 75898/400000 [00:08<00:37, 8641.89it/s] 19%|█▉        | 76768/400000 [00:08<00:37, 8656.57it/s] 19%|█▉        | 77639/400000 [00:09<00:37, 8670.43it/s] 20%|█▉        | 78507/400000 [00:09<00:37, 8662.88it/s] 20%|█▉        | 79375/400000 [00:09<00:36, 8667.23it/s] 20%|██        | 80244/400000 [00:09<00:36, 8673.20it/s] 20%|██        | 81112/400000 [00:09<00:36, 8632.80it/s] 20%|██        | 81987/400000 [00:09<00:36, 8665.63it/s] 21%|██        | 82854/400000 [00:09<00:36, 8656.04it/s] 21%|██        | 83720/400000 [00:09<00:36, 8654.26it/s] 21%|██        | 84586/400000 [00:09<00:36, 8626.03it/s] 21%|██▏       | 85449/400000 [00:09<00:36, 8610.65it/s] 22%|██▏       | 86311/400000 [00:10<00:36, 8600.74it/s] 22%|██▏       | 87172/400000 [00:10<00:36, 8535.52it/s] 22%|██▏       | 88026/400000 [00:10<00:36, 8529.82it/s] 22%|██▏       | 88880/400000 [00:10<00:36, 8530.87it/s] 22%|██▏       | 89749/400000 [00:10<00:36, 8575.77it/s] 23%|██▎       | 90619/400000 [00:10<00:35, 8610.22it/s] 23%|██▎       | 91484/400000 [00:10<00:35, 8621.39it/s] 23%|██▎       | 92347/400000 [00:10<00:35, 8575.84it/s] 23%|██▎       | 93205/400000 [00:10<00:35, 8538.07it/s] 24%|██▎       | 94067/400000 [00:10<00:35, 8561.62it/s] 24%|██▎       | 94931/400000 [00:11<00:35, 8584.82it/s] 24%|██▍       | 95790/400000 [00:11<00:35, 8578.77it/s] 24%|██▍       | 96656/400000 [00:11<00:35, 8601.63it/s] 24%|██▍       | 97517/400000 [00:11<00:35, 8445.44it/s] 25%|██▍       | 98369/400000 [00:11<00:35, 8465.21it/s] 25%|██▍       | 99242/400000 [00:11<00:35, 8542.09it/s] 25%|██▌       | 100099/400000 [00:11<00:35, 8548.41it/s] 25%|██▌       | 100965/400000 [00:11<00:34, 8579.58it/s] 25%|██▌       | 101829/400000 [00:11<00:34, 8596.15it/s] 26%|██▌       | 102696/400000 [00:11<00:34, 8617.60it/s] 26%|██▌       | 103558/400000 [00:12<00:34, 8611.38it/s] 26%|██▌       | 104420/400000 [00:12<00:34, 8605.56it/s] 26%|██▋       | 105287/400000 [00:12<00:34, 8623.99it/s] 27%|██▋       | 106152/400000 [00:12<00:34, 8630.62it/s] 27%|██▋       | 107016/400000 [00:12<00:33, 8623.19it/s] 27%|██▋       | 107887/400000 [00:12<00:33, 8647.37it/s] 27%|██▋       | 108752/400000 [00:12<00:33, 8610.43it/s] 27%|██▋       | 109618/400000 [00:12<00:33, 8622.55it/s] 28%|██▊       | 110481/400000 [00:12<00:33, 8616.96it/s] 28%|██▊       | 111345/400000 [00:12<00:33, 8622.13it/s] 28%|██▊       | 112208/400000 [00:13<00:33, 8588.22it/s] 28%|██▊       | 113069/400000 [00:13<00:33, 8591.74it/s] 28%|██▊       | 113934/400000 [00:13<00:33, 8606.27it/s] 29%|██▊       | 114799/400000 [00:13<00:33, 8617.69it/s] 29%|██▉       | 115667/400000 [00:13<00:32, 8635.57it/s] 29%|██▉       | 116532/400000 [00:13<00:32, 8638.89it/s] 29%|██▉       | 117401/400000 [00:13<00:32, 8653.27it/s] 30%|██▉       | 118267/400000 [00:13<00:32, 8652.75it/s] 30%|██▉       | 119133/400000 [00:13<00:32, 8649.76it/s] 30%|██▉       | 119998/400000 [00:13<00:32, 8641.59it/s] 30%|███       | 120869/400000 [00:14<00:32, 8659.66it/s] 30%|███       | 121737/400000 [00:14<00:32, 8664.51it/s] 31%|███       | 122604/400000 [00:14<00:32, 8660.38it/s] 31%|███       | 123471/400000 [00:14<00:31, 8653.72it/s] 31%|███       | 124337/400000 [00:14<00:31, 8649.52it/s] 31%|███▏      | 125204/400000 [00:14<00:31, 8654.12it/s] 32%|███▏      | 126070/400000 [00:14<00:31, 8653.28it/s] 32%|███▏      | 126936/400000 [00:14<00:31, 8637.24it/s] 32%|███▏      | 127808/400000 [00:14<00:31, 8659.48it/s] 32%|███▏      | 128677/400000 [00:14<00:31, 8668.14it/s] 32%|███▏      | 129546/400000 [00:15<00:31, 8674.59it/s] 33%|███▎      | 130414/400000 [00:15<00:31, 8658.31it/s] 33%|███▎      | 131280/400000 [00:15<00:31, 8526.34it/s] 33%|███▎      | 132143/400000 [00:15<00:31, 8555.27it/s] 33%|███▎      | 133004/400000 [00:15<00:31, 8570.71it/s] 33%|███▎      | 133867/400000 [00:15<00:30, 8587.85it/s] 34%|███▎      | 134726/400000 [00:15<00:30, 8582.57it/s] 34%|███▍      | 135585/400000 [00:15<00:30, 8582.02it/s] 34%|███▍      | 136449/400000 [00:15<00:30, 8596.69it/s] 34%|███▍      | 137309/400000 [00:16<00:30, 8546.88it/s] 35%|███▍      | 138164/400000 [00:16<00:30, 8539.74it/s] 35%|███▍      | 139019/400000 [00:16<00:30, 8484.71it/s] 35%|███▍      | 139868/400000 [00:16<00:30, 8467.58it/s] 35%|███▌      | 140715/400000 [00:16<00:30, 8395.34it/s] 35%|███▌      | 141575/400000 [00:16<00:30, 8452.99it/s] 36%|███▌      | 142435/400000 [00:16<00:30, 8495.77it/s] 36%|███▌      | 143288/400000 [00:16<00:30, 8505.43it/s] 36%|███▌      | 144139/400000 [00:16<00:30, 8482.13it/s] 36%|███▌      | 144994/400000 [00:16<00:29, 8501.41it/s] 36%|███▋      | 145845/400000 [00:17<00:29, 8483.10it/s] 37%|███▋      | 146711/400000 [00:17<00:29, 8533.05it/s] 37%|███▋      | 147565/400000 [00:17<00:29, 8461.93it/s] 37%|███▋      | 148417/400000 [00:17<00:29, 8479.13it/s] 37%|███▋      | 149279/400000 [00:17<00:29, 8518.55it/s] 38%|███▊      | 150139/400000 [00:17<00:29, 8541.76it/s] 38%|███▊      | 151002/400000 [00:17<00:29, 8567.55it/s] 38%|███▊      | 151870/400000 [00:17<00:28, 8598.17it/s] 38%|███▊      | 152730/400000 [00:17<00:29, 8393.14it/s] 38%|███▊      | 153571/400000 [00:17<00:29, 8368.94it/s] 39%|███▊      | 154436/400000 [00:18<00:29, 8451.21it/s] 39%|███▉      | 155304/400000 [00:18<00:28, 8518.46it/s] 39%|███▉      | 156175/400000 [00:18<00:28, 8574.66it/s] 39%|███▉      | 157039/400000 [00:18<00:28, 8592.90it/s] 39%|███▉      | 157907/400000 [00:18<00:28, 8617.36it/s] 40%|███▉      | 158771/400000 [00:18<00:27, 8622.91it/s] 40%|███▉      | 159634/400000 [00:18<00:27, 8623.21it/s] 40%|████      | 160497/400000 [00:18<00:27, 8623.69it/s] 40%|████      | 161360/400000 [00:18<00:27, 8618.23it/s] 41%|████      | 162228/400000 [00:18<00:27, 8636.28it/s] 41%|████      | 163097/400000 [00:19<00:27, 8651.28it/s] 41%|████      | 163963/400000 [00:19<00:27, 8633.85it/s] 41%|████      | 164833/400000 [00:19<00:27, 8651.35it/s] 41%|████▏     | 165699/400000 [00:19<00:27, 8643.87it/s] 42%|████▏     | 166564/400000 [00:19<00:27, 8526.63it/s] 42%|████▏     | 167418/400000 [00:19<00:27, 8340.44it/s] 42%|████▏     | 168279/400000 [00:19<00:27, 8418.71it/s] 42%|████▏     | 169142/400000 [00:19<00:27, 8479.51it/s] 42%|████▏     | 169991/400000 [00:19<00:27, 8393.78it/s] 43%|████▎     | 170854/400000 [00:19<00:27, 8461.13it/s] 43%|████▎     | 171718/400000 [00:20<00:26, 8513.32it/s] 43%|████▎     | 172570/400000 [00:20<00:26, 8505.30it/s] 43%|████▎     | 173437/400000 [00:20<00:26, 8552.11it/s] 44%|████▎     | 174293/400000 [00:20<00:26, 8525.07it/s] 44%|████▍     | 175151/400000 [00:20<00:26, 8540.36it/s] 44%|████▍     | 176020/400000 [00:20<00:26, 8584.09it/s] 44%|████▍     | 176881/400000 [00:20<00:25, 8591.18it/s] 44%|████▍     | 177755/400000 [00:20<00:25, 8635.27it/s] 45%|████▍     | 178619/400000 [00:20<00:25, 8572.35it/s] 45%|████▍     | 179477/400000 [00:20<00:25, 8537.77it/s] 45%|████▌     | 180346/400000 [00:21<00:25, 8580.28it/s] 45%|████▌     | 181207/400000 [00:21<00:25, 8588.38it/s] 46%|████▌     | 182066/400000 [00:21<00:25, 8552.33it/s] 46%|████▌     | 182924/400000 [00:21<00:25, 8559.99it/s] 46%|████▌     | 183789/400000 [00:21<00:25, 8584.21it/s] 46%|████▌     | 184648/400000 [00:21<00:25, 8421.23it/s] 46%|████▋     | 185506/400000 [00:21<00:25, 8467.93it/s] 47%|████▋     | 186354/400000 [00:21<00:25, 8256.70it/s] 47%|████▋     | 187209/400000 [00:21<00:25, 8341.82it/s] 47%|████▋     | 188045/400000 [00:21<00:25, 8304.72it/s] 47%|████▋     | 188901/400000 [00:22<00:25, 8377.12it/s] 47%|████▋     | 189761/400000 [00:22<00:24, 8441.94it/s] 48%|████▊     | 190626/400000 [00:22<00:24, 8501.67it/s] 48%|████▊     | 191477/400000 [00:22<00:24, 8502.64it/s] 48%|████▊     | 192328/400000 [00:22<00:24, 8487.54it/s] 48%|████▊     | 193178/400000 [00:22<00:24, 8487.64it/s] 49%|████▊     | 194027/400000 [00:22<00:24, 8434.48it/s] 49%|████▊     | 194889/400000 [00:22<00:24, 8488.29it/s] 49%|████▉     | 195753/400000 [00:22<00:23, 8532.71it/s] 49%|████▉     | 196622/400000 [00:22<00:23, 8578.94it/s] 49%|████▉     | 197493/400000 [00:23<00:23, 8617.25it/s] 50%|████▉     | 198357/400000 [00:23<00:23, 8622.79it/s] 50%|████▉     | 199234/400000 [00:23<00:23, 8664.44it/s] 50%|█████     | 200101/400000 [00:23<00:23, 8629.13it/s] 50%|█████     | 200965/400000 [00:23<00:24, 8278.78it/s] 50%|█████     | 201817/400000 [00:23<00:23, 8349.09it/s] 51%|█████     | 202672/400000 [00:23<00:23, 8406.54it/s] 51%|█████     | 203537/400000 [00:23<00:23, 8478.11it/s] 51%|█████     | 204400/400000 [00:23<00:22, 8520.93it/s] 51%|█████▏    | 205275/400000 [00:23<00:22, 8586.81it/s] 52%|█████▏    | 206148/400000 [00:24<00:22, 8627.69it/s] 52%|█████▏    | 207015/400000 [00:24<00:22, 8637.74it/s] 52%|█████▏    | 207889/400000 [00:24<00:22, 8666.55it/s] 52%|█████▏    | 208756/400000 [00:24<00:22, 8665.29it/s] 52%|█████▏    | 209623/400000 [00:24<00:21, 8654.15it/s] 53%|█████▎    | 210494/400000 [00:24<00:21, 8669.10it/s] 53%|█████▎    | 211363/400000 [00:24<00:21, 8673.24it/s] 53%|█████▎    | 212239/400000 [00:24<00:21, 8696.64it/s] 53%|█████▎    | 213116/400000 [00:24<00:21, 8718.25it/s] 53%|█████▎    | 213988/400000 [00:24<00:21, 8544.97it/s] 54%|█████▎    | 214844/400000 [00:25<00:21, 8486.65it/s] 54%|█████▍    | 215712/400000 [00:25<00:21, 8543.54it/s] 54%|█████▍    | 216567/400000 [00:25<00:21, 8466.56it/s] 54%|█████▍    | 217415/400000 [00:25<00:21, 8344.00it/s] 55%|█████▍    | 218279/400000 [00:25<00:21, 8427.97it/s] 55%|█████▍    | 219150/400000 [00:25<00:21, 8508.05it/s] 55%|█████▌    | 220019/400000 [00:25<00:21, 8560.04it/s] 55%|█████▌    | 220876/400000 [00:25<00:20, 8561.98it/s] 55%|█████▌    | 221735/400000 [00:25<00:20, 8570.14it/s] 56%|█████▌    | 222605/400000 [00:26<00:20, 8605.97it/s] 56%|█████▌    | 223485/400000 [00:26<00:20, 8662.50it/s] 56%|█████▌    | 224352/400000 [00:26<00:20, 8642.41it/s] 56%|█████▋    | 225229/400000 [00:26<00:20, 8679.71it/s] 57%|█████▋    | 226098/400000 [00:26<00:20, 8673.48it/s] 57%|█████▋    | 226966/400000 [00:26<00:19, 8675.05it/s] 57%|█████▋    | 227838/400000 [00:26<00:19, 8687.09it/s] 57%|█████▋    | 228707/400000 [00:26<00:19, 8648.82it/s] 57%|█████▋    | 229576/400000 [00:26<00:19, 8660.40it/s] 58%|█████▊    | 230446/400000 [00:26<00:19, 8671.82it/s] 58%|█████▊    | 231314/400000 [00:27<00:19, 8634.78it/s] 58%|█████▊    | 232181/400000 [00:27<00:19, 8645.26it/s] 58%|█████▊    | 233054/400000 [00:27<00:19, 8669.33it/s] 58%|█████▊    | 233925/400000 [00:27<00:19, 8678.66it/s] 59%|█████▊    | 234793/400000 [00:27<00:19, 8569.92it/s] 59%|█████▉    | 235663/400000 [00:27<00:19, 8607.01it/s] 59%|█████▉    | 236538/400000 [00:27<00:18, 8648.68it/s] 59%|█████▉    | 237409/400000 [00:27<00:18, 8666.44it/s] 60%|█████▉    | 238280/400000 [00:27<00:18, 8678.41it/s] 60%|█████▉    | 239148/400000 [00:27<00:18, 8651.23it/s] 60%|██████    | 240014/400000 [00:28<00:18, 8612.17it/s] 60%|██████    | 240876/400000 [00:28<00:18, 8537.46it/s] 60%|██████    | 241745/400000 [00:28<00:18, 8579.97it/s] 61%|██████    | 242604/400000 [00:28<00:18, 8426.36it/s] 61%|██████    | 243465/400000 [00:28<00:18, 8479.93it/s] 61%|██████    | 244314/400000 [00:28<00:18, 8473.72it/s] 61%|██████▏   | 245181/400000 [00:28<00:18, 8531.27it/s] 62%|██████▏   | 246059/400000 [00:28<00:17, 8601.54it/s] 62%|██████▏   | 246924/400000 [00:28<00:17, 8613.74it/s] 62%|██████▏   | 247798/400000 [00:28<00:17, 8648.21it/s] 62%|██████▏   | 248664/400000 [00:29<00:17, 8643.57it/s] 62%|██████▏   | 249529/400000 [00:29<00:17, 8636.93it/s] 63%|██████▎   | 250399/400000 [00:29<00:17, 8653.78it/s] 63%|██████▎   | 251265/400000 [00:29<00:17, 8645.40it/s] 63%|██████▎   | 252130/400000 [00:29<00:17, 8632.55it/s] 63%|██████▎   | 252994/400000 [00:29<00:17, 8623.44it/s] 63%|██████▎   | 253862/400000 [00:29<00:16, 8639.57it/s] 64%|██████▎   | 254731/400000 [00:29<00:16, 8651.86it/s] 64%|██████▍   | 255597/400000 [00:29<00:17, 8472.59it/s] 64%|██████▍   | 256468/400000 [00:29<00:16, 8540.29it/s] 64%|██████▍   | 257340/400000 [00:30<00:16, 8590.79it/s] 65%|██████▍   | 258218/400000 [00:30<00:16, 8643.88it/s] 65%|██████▍   | 259083/400000 [00:30<00:16, 8636.36it/s] 65%|██████▍   | 259947/400000 [00:30<00:16, 8634.73it/s] 65%|██████▌   | 260811/400000 [00:30<00:16, 8628.76it/s] 65%|██████▌   | 261675/400000 [00:30<00:16, 8623.73it/s] 66%|██████▌   | 262546/400000 [00:30<00:15, 8649.23it/s] 66%|██████▌   | 263420/400000 [00:30<00:15, 8674.57it/s] 66%|██████▌   | 264290/400000 [00:30<00:15, 8681.90it/s] 66%|██████▋   | 265159/400000 [00:30<00:15, 8679.07it/s] 67%|██████▋   | 266027/400000 [00:31<00:15, 8661.79it/s] 67%|██████▋   | 266899/400000 [00:31<00:15, 8674.25it/s] 67%|██████▋   | 267782/400000 [00:31<00:15, 8719.68it/s] 67%|██████▋   | 268659/400000 [00:31<00:15, 8734.09it/s] 67%|██████▋   | 269533/400000 [00:31<00:14, 8706.54it/s] 68%|██████▊   | 270404/400000 [00:31<00:14, 8696.33it/s] 68%|██████▊   | 271274/400000 [00:31<00:14, 8671.56it/s] 68%|██████▊   | 272148/400000 [00:31<00:14, 8691.35it/s] 68%|██████▊   | 273023/400000 [00:31<00:14, 8707.94it/s] 68%|██████▊   | 273894/400000 [00:31<00:14, 8701.85it/s] 69%|██████▊   | 274765/400000 [00:32<00:14, 8581.50it/s] 69%|██████▉   | 275634/400000 [00:32<00:14, 8613.01it/s] 69%|██████▉   | 276496/400000 [00:32<00:14, 8605.00it/s] 69%|██████▉   | 277357/400000 [00:32<00:14, 8590.50it/s] 70%|██████▉   | 278224/400000 [00:32<00:14, 8613.22it/s] 70%|██████▉   | 279086/400000 [00:32<00:14, 8603.39it/s] 70%|██████▉   | 279947/400000 [00:32<00:14, 8571.67it/s] 70%|███████   | 280808/400000 [00:32<00:13, 8581.37it/s] 70%|███████   | 281667/400000 [00:32<00:13, 8551.38it/s] 71%|███████   | 282523/400000 [00:32<00:13, 8479.15it/s] 71%|███████   | 283372/400000 [00:33<00:13, 8408.99it/s] 71%|███████   | 284224/400000 [00:33<00:13, 8439.66it/s] 71%|███████▏  | 285092/400000 [00:33<00:13, 8508.97it/s] 71%|███████▏  | 285962/400000 [00:33<00:13, 8562.50it/s] 72%|███████▏  | 286828/400000 [00:33<00:13, 8590.87it/s] 72%|███████▏  | 287688/400000 [00:33<00:13, 8318.75it/s] 72%|███████▏  | 288553/400000 [00:33<00:13, 8414.46it/s] 72%|███████▏  | 289421/400000 [00:33<00:13, 8491.05it/s] 73%|███████▎  | 290282/400000 [00:33<00:12, 8524.07it/s] 73%|███████▎  | 291152/400000 [00:33<00:12, 8573.15it/s] 73%|███████▎  | 292011/400000 [00:34<00:12, 8565.80it/s] 73%|███████▎  | 292880/400000 [00:34<00:12, 8600.08it/s] 73%|███████▎  | 293751/400000 [00:34<00:12, 8630.40it/s] 74%|███████▎  | 294623/400000 [00:34<00:12, 8655.81it/s] 74%|███████▍  | 295491/400000 [00:34<00:12, 8661.73it/s] 74%|███████▍  | 296358/400000 [00:34<00:12, 8634.65it/s] 74%|███████▍  | 297222/400000 [00:34<00:12, 8555.39it/s] 75%|███████▍  | 298085/400000 [00:34<00:11, 8577.16it/s] 75%|███████▍  | 298959/400000 [00:34<00:11, 8622.76it/s] 75%|███████▍  | 299835/400000 [00:34<00:11, 8662.61it/s] 75%|███████▌  | 300702/400000 [00:35<00:11, 8650.55it/s] 75%|███████▌  | 301573/400000 [00:35<00:11, 8665.46it/s] 76%|███████▌  | 302451/400000 [00:35<00:11, 8696.80it/s] 76%|███████▌  | 303321/400000 [00:35<00:11, 8639.50it/s] 76%|███████▌  | 304186/400000 [00:35<00:11, 8545.01it/s] 76%|███████▋  | 305041/400000 [00:35<00:11, 8518.66it/s] 76%|███████▋  | 305912/400000 [00:35<00:10, 8574.99it/s] 77%|███████▋  | 306782/400000 [00:35<00:10, 8610.35it/s] 77%|███████▋  | 307648/400000 [00:35<00:10, 8625.12it/s] 77%|███████▋  | 308511/400000 [00:35<00:10, 8607.16it/s] 77%|███████▋  | 309372/400000 [00:36<00:10, 8587.79it/s] 78%|███████▊  | 310233/400000 [00:36<00:10, 8591.82it/s] 78%|███████▊  | 311097/400000 [00:36<00:10, 8603.49it/s] 78%|███████▊  | 311964/400000 [00:36<00:10, 8621.09it/s] 78%|███████▊  | 312837/400000 [00:36<00:10, 8651.28it/s] 78%|███████▊  | 313703/400000 [00:36<00:09, 8643.23it/s] 79%|███████▊  | 314585/400000 [00:36<00:09, 8693.54it/s] 79%|███████▉  | 315455/400000 [00:36<00:09, 8688.64it/s] 79%|███████▉  | 316324/400000 [00:36<00:09, 8688.06it/s] 79%|███████▉  | 317194/400000 [00:36<00:09, 8689.18it/s] 80%|███████▉  | 318063/400000 [00:37<00:09, 8661.02it/s] 80%|███████▉  | 318930/400000 [00:37<00:09, 8631.80it/s] 80%|███████▉  | 319801/400000 [00:37<00:09, 8653.73it/s] 80%|████████  | 320667/400000 [00:37<00:09, 8651.72it/s] 80%|████████  | 321533/400000 [00:37<00:09, 8624.46it/s] 81%|████████  | 322399/400000 [00:37<00:08, 8632.33it/s] 81%|████████  | 323263/400000 [00:37<00:08, 8620.27it/s] 81%|████████  | 324131/400000 [00:37<00:08, 8635.15it/s] 81%|████████▏ | 325002/400000 [00:37<00:08, 8655.19it/s] 81%|████████▏ | 325868/400000 [00:37<00:08, 8422.79it/s] 82%|████████▏ | 326741/400000 [00:38<00:08, 8511.11it/s] 82%|████████▏ | 327613/400000 [00:38<00:08, 8572.42it/s] 82%|████████▏ | 328475/400000 [00:38<00:08, 8583.84it/s] 82%|████████▏ | 329340/400000 [00:38<00:08, 8602.99it/s] 83%|████████▎ | 330211/400000 [00:38<00:08, 8633.81it/s] 83%|████████▎ | 331082/400000 [00:38<00:07, 8653.95it/s] 83%|████████▎ | 331954/400000 [00:38<00:07, 8671.64it/s] 83%|████████▎ | 332822/400000 [00:38<00:07, 8649.86it/s] 83%|████████▎ | 333688/400000 [00:38<00:07, 8638.75it/s] 84%|████████▎ | 334552/400000 [00:38<00:07, 8637.61it/s] 84%|████████▍ | 335416/400000 [00:39<00:07, 8532.63it/s] 84%|████████▍ | 336270/400000 [00:39<00:07, 8514.52it/s] 84%|████████▍ | 337131/400000 [00:39<00:07, 8542.80it/s] 84%|████████▍ | 337997/400000 [00:39<00:07, 8576.92it/s] 85%|████████▍ | 338855/400000 [00:39<00:07, 8553.21it/s] 85%|████████▍ | 339722/400000 [00:39<00:07, 8585.99it/s] 85%|████████▌ | 340581/400000 [00:39<00:06, 8519.59it/s] 85%|████████▌ | 341448/400000 [00:39<00:06, 8564.07it/s] 86%|████████▌ | 342319/400000 [00:39<00:06, 8606.19it/s] 86%|████████▌ | 343185/400000 [00:40<00:06, 8620.93it/s] 86%|████████▌ | 344053/400000 [00:40<00:06, 8635.67it/s] 86%|████████▌ | 344917/400000 [00:40<00:06, 8590.60it/s] 86%|████████▋ | 345789/400000 [00:40<00:06, 8627.07it/s] 87%|████████▋ | 346654/400000 [00:40<00:06, 8633.64it/s] 87%|████████▋ | 347519/400000 [00:40<00:06, 8636.52it/s] 87%|████████▋ | 348389/400000 [00:40<00:05, 8652.87it/s] 87%|████████▋ | 349255/400000 [00:40<00:05, 8604.17it/s] 88%|████████▊ | 350118/400000 [00:40<00:05, 8609.24it/s] 88%|████████▊ | 350981/400000 [00:40<00:05, 8614.55it/s] 88%|████████▊ | 351843/400000 [00:41<00:05, 8611.01it/s] 88%|████████▊ | 352709/400000 [00:41<00:05, 8623.34it/s] 88%|████████▊ | 353572/400000 [00:41<00:05, 8585.35it/s] 89%|████████▊ | 354441/400000 [00:41<00:05, 8614.72it/s] 89%|████████▉ | 355306/400000 [00:41<00:05, 8622.51it/s] 89%|████████▉ | 356169/400000 [00:41<00:05, 8535.90it/s] 89%|████████▉ | 357025/400000 [00:41<00:05, 8540.69it/s] 89%|████████▉ | 357890/400000 [00:41<00:04, 8571.39it/s] 90%|████████▉ | 358766/400000 [00:41<00:04, 8624.88it/s] 90%|████████▉ | 359637/400000 [00:41<00:04, 8648.92it/s] 90%|█████████ | 360503/400000 [00:42<00:04, 8647.32it/s] 90%|█████████ | 361370/400000 [00:42<00:04, 8652.88it/s] 91%|█████████ | 362236/400000 [00:42<00:04, 8536.58it/s] 91%|█████████ | 363097/400000 [00:42<00:04, 8555.63it/s] 91%|█████████ | 363962/400000 [00:42<00:04, 8583.15it/s] 91%|█████████ | 364835/400000 [00:42<00:04, 8623.86it/s] 91%|█████████▏| 365704/400000 [00:42<00:03, 8640.69it/s] 92%|█████████▏| 366569/400000 [00:42<00:03, 8641.78it/s] 92%|█████████▏| 367440/400000 [00:42<00:03, 8661.64it/s] 92%|█████████▏| 368314/400000 [00:42<00:03, 8682.49it/s] 92%|█████████▏| 369191/400000 [00:43<00:03, 8707.64it/s] 93%|█████████▎| 370063/400000 [00:43<00:03, 8711.28it/s] 93%|█████████▎| 370935/400000 [00:43<00:03, 8642.52it/s] 93%|█████████▎| 371807/400000 [00:43<00:03, 8663.70it/s] 93%|█████████▎| 372676/400000 [00:43<00:03, 8669.14it/s] 93%|█████████▎| 373544/400000 [00:43<00:03, 8654.64it/s] 94%|█████████▎| 374410/400000 [00:43<00:02, 8649.54it/s] 94%|█████████▍| 375276/400000 [00:43<00:02, 8611.48it/s] 94%|█████████▍| 376147/400000 [00:43<00:02, 8640.67it/s] 94%|█████████▍| 377025/400000 [00:43<00:02, 8679.61it/s] 94%|█████████▍| 377896/400000 [00:44<00:02, 8687.59it/s] 95%|█████████▍| 378770/400000 [00:44<00:02, 8702.08it/s] 95%|█████████▍| 379641/400000 [00:44<00:02, 8668.90it/s] 95%|█████████▌| 380512/400000 [00:44<00:02, 8680.11it/s] 95%|█████████▌| 381381/400000 [00:44<00:02, 8672.43it/s] 96%|█████████▌| 382249/400000 [00:44<00:02, 8642.31it/s] 96%|█████████▌| 383123/400000 [00:44<00:01, 8670.02it/s] 96%|█████████▌| 383991/400000 [00:44<00:01, 8639.34it/s] 96%|█████████▌| 384863/400000 [00:44<00:01, 8663.15it/s] 96%|█████████▋| 385730/400000 [00:44<00:01, 8639.02it/s] 97%|█████████▋| 386599/400000 [00:45<00:01, 8651.12it/s] 97%|█████████▋| 387480/400000 [00:45<00:01, 8697.00it/s] 97%|█████████▋| 388350/400000 [00:45<00:01, 8677.11it/s] 97%|█████████▋| 389218/400000 [00:45<00:01, 8644.94it/s] 98%|█████████▊| 390083/400000 [00:45<00:01, 8619.25it/s] 98%|█████████▊| 390945/400000 [00:45<00:01, 8604.36it/s] 98%|█████████▊| 391806/400000 [00:45<00:00, 8598.47it/s] 98%|█████████▊| 392666/400000 [00:45<00:00, 8550.79it/s] 98%|█████████▊| 393522/400000 [00:45<00:00, 8346.63it/s] 99%|█████████▊| 394374/400000 [00:45<00:00, 8396.32it/s] 99%|█████████▉| 395215/400000 [00:46<00:00, 8372.94it/s] 99%|█████████▉| 396053/400000 [00:46<00:00, 8318.02it/s] 99%|█████████▉| 396900/400000 [00:46<00:00, 8360.82it/s] 99%|█████████▉| 397768/400000 [00:46<00:00, 8451.36it/s]100%|█████████▉| 398636/400000 [00:46<00:00, 8516.10it/s]100%|█████████▉| 399502/400000 [00:46<00:00, 8557.30it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8581.06it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdc44047940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011230754153070191 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.010819484956288417 	 Accuracy: 73

  model saves at 73% accuracy 

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
2020-05-15 13:23:51.223987: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 13:23:51.227848: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-15 13:23:51.227978: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557ed7d88f20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 13:23:51.227992: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdbe9aecd30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8200 - accuracy: 0.4900 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7455 - accuracy: 0.4949
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7529 - accuracy: 0.4944
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7382 - accuracy: 0.4953
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 3s - loss: 7.7043 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 3s - loss: 7.7152 - accuracy: 0.4968
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7020 - accuracy: 0.4977
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6557 - accuracy: 0.5007
15000/25000 [=================>............] - ETA: 2s - loss: 7.6380 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6300 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6182 - accuracy: 0.5032
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fdbc02e0668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fdbf7599be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3903 - crf_viterbi_accuracy: 0.2667 - val_loss: 1.3043 - val_crf_viterbi_accuracy: 0.3200

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

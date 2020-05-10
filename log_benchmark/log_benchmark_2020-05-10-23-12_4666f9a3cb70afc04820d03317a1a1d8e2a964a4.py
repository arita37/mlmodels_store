
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f16f02044a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:12:58.403794
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 23:12:58.407701
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 23:12:58.411165
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 23:12:58.414288
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f16e8554470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351270.5938
Epoch 2/10

1/1 [==============================] - 0s 93ms/step - loss: 241306.7969
Epoch 3/10

1/1 [==============================] - 0s 90ms/step - loss: 152313.4688
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 90703.3516
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 55903.9883
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 36240.5391
Epoch 7/10

1/1 [==============================] - 0s 87ms/step - loss: 24109.4473
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 16908.2246
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 12389.5752
Epoch 10/10

1/1 [==============================] - 0s 88ms/step - loss: 9376.6602

  #### Inference Need return ypred, ytrue ######################### 
[[-2.17569208e+00 -2.04200125e+00  1.34097874e+00 -2.86555082e-01
  -2.05511093e-01 -8.44933689e-01  4.65252906e-01  1.26711893e+00
  -7.32590079e-01  9.62317288e-01 -9.97295856e-01 -1.19724131e+00
   1.89250875e+00 -1.70290387e+00  9.03502226e-01 -5.18581390e-01
   1.73206449e-01  5.22440612e-01  1.09908426e+00  4.26119447e-01
  -1.00669456e+00 -1.05912268e+00 -7.73617923e-02 -7.64746666e-01
   1.11505508e-01 -2.09285879e+00 -1.16260529e+00  4.36911106e-01
   1.22952402e-01 -7.85342276e-01 -2.84280837e-01 -2.52828240e-01
   1.76133871e-01  4.84788418e-03 -5.99293232e-01 -1.25157344e+00
  -4.15215492e-01  1.82609558e-01 -1.51771474e+00  1.71686709e+00
  -1.26790988e+00  1.32873249e+00 -1.03419900e+00 -4.08679783e-01
   4.13120925e-01 -6.55442476e-01  2.82641575e-02  2.85517305e-01
  -2.62468547e-01  1.48370922e+00 -8.49109292e-01 -6.75290108e-01
   2.24007368e+00  1.48960203e-02  1.76015520e+00 -1.52301979e+00
   7.11180806e-01  1.01738572e-02 -1.00729394e+00  1.37969244e+00
  -1.87107071e-01  7.71281338e+00  8.38660431e+00  7.97883749e+00
   7.09869909e+00  5.97792816e+00  5.41293812e+00  4.18162394e+00
   6.71513271e+00  8.02294731e+00  6.03381968e+00  6.26320791e+00
   7.62005281e+00  7.83742952e+00  6.99203491e+00  6.75451422e+00
   6.58939981e+00  8.28634834e+00  6.61197376e+00  5.75374413e+00
   5.69860601e+00  5.84540844e+00  4.82842493e+00  5.06543398e+00
   5.85834551e+00  6.49058199e+00  8.94728088e+00  6.54171467e+00
   8.42424488e+00  7.28893852e+00  7.55974960e+00  8.22773743e+00
   6.65383816e+00  4.77352190e+00  5.83676386e+00  5.85824537e+00
   7.37862778e+00  5.12350845e+00  5.76387453e+00  7.43663454e+00
   6.76175833e+00  5.75450134e+00  4.49041319e+00  7.76335001e+00
   7.30230999e+00  5.00998068e+00  5.69500113e+00  6.94285917e+00
   6.20651722e+00  6.53915691e+00  5.38581419e+00  5.31215715e+00
   7.70203733e+00  7.52037764e+00  5.26809931e+00  7.70219803e+00
   8.59180927e+00  8.97210312e+00  6.04662037e+00  6.19064140e+00
  -1.51906157e+00  2.23716348e-02 -1.20156419e+00 -5.96114397e-02
   1.95637488e+00 -1.22416580e+00 -1.05410814e+00 -1.72744125e-01
  -4.11273241e-02  7.66085625e-01  3.68244946e-01 -5.54369569e-01
  -3.24579179e-01 -1.29032218e+00  6.34791195e-01 -3.30168009e-01
   1.69551134e-01 -9.99618769e-01  1.08953381e+00 -5.67065239e-01
  -7.52815962e-01 -2.44875401e-01  3.67897004e-01 -3.35263848e-01
   6.46041155e-01 -7.14251399e-02 -2.14160180e+00 -9.17577267e-01
  -1.13678229e+00 -7.15512037e-02  8.56005728e-01 -1.30007076e+00
   3.68946731e-01 -8.12677741e-01  1.87142700e-01 -1.37978005e+00
  -1.32930410e+00  2.54071927e+00  2.22265887e+00  1.13027453e-01
  -2.04686856e+00  5.72523028e-02  1.34013188e+00  4.38441992e-01
   2.41744423e+00  1.77115464e+00 -2.49663639e+00 -2.90159762e-01
  -4.20700938e-01  2.62956023e-02 -1.14528000e+00 -2.83306360e-01
   1.41849232e+00 -5.74004829e-01 -4.52263415e-01  1.64612323e-01
   6.87343836e-01 -8.96789074e-01 -8.10184598e-01 -7.76237190e-01
   1.40551996e+00  1.37561190e+00  1.97382748e+00  6.91846609e-01
   5.13778806e-01  2.02435732e-01  1.36212277e+00  2.74915338e-01
   4.73195910e-01  7.24749684e-01  1.37997925e-01  1.34589612e+00
   5.13270617e-01  1.02058423e+00  2.73766422e+00  1.41822791e+00
   3.45574796e-01  9.46736455e-01  1.62886751e+00  1.17230892e+00
   2.90886521e-01  6.98832273e-01  2.52983928e-01  3.79660904e-01
   9.05467093e-01  1.77683628e+00  3.75047684e-01  3.10353577e-01
   2.07137918e+00  2.59679437e-01  7.06114888e-01  2.49105453e+00
   5.76398134e-01  4.85871553e-01  4.89416599e-01  2.19289970e+00
   1.99738741e-01  1.99226451e+00  1.59352660e-01  2.40681648e-01
   8.20065856e-01  4.42341328e-01  1.35238624e+00  1.11463451e+00
   8.36991370e-01  1.57493818e+00  9.93752658e-01  1.34752905e+00
   2.30569696e+00  1.54223967e+00  1.12565482e+00  1.41065896e+00
   3.60029817e-01  2.37099338e+00  7.04548836e-01  6.00716472e-01
   1.22333682e+00  1.05859041e-01  7.71368027e-01  1.33342624e+00
   9.35964584e-02  6.53196335e+00  8.34565639e+00  8.91127300e+00
   6.41326523e+00  6.18286419e+00  6.85253286e+00  4.99363995e+00
   5.23811674e+00  7.67196751e+00  6.51262093e+00  7.76800156e+00
   6.85496521e+00  7.49629545e+00  7.42042065e+00  7.38085842e+00
   8.76647568e+00  6.11599779e+00  7.25348425e+00  8.62512302e+00
   5.63728857e+00  7.45710707e+00  7.93253374e+00  5.60057306e+00
   5.37479448e+00  7.84502840e+00  7.51471758e+00  6.66764259e+00
   8.16364956e+00  7.92065811e+00  5.90228224e+00  8.26704311e+00
   8.53026390e+00  6.26744652e+00  9.03618050e+00  7.03308678e+00
   9.01734161e+00  5.28142881e+00  5.71227741e+00  5.45626259e+00
   9.08001423e+00  7.32011509e+00  9.06374931e+00  8.93606949e+00
   5.88152313e+00  8.03607273e+00  8.91999817e+00  7.17159081e+00
   7.95990801e+00  8.42444229e+00  8.11400318e+00  7.74244308e+00
   8.88926411e+00  6.44241333e+00  6.61742115e+00  6.53759193e+00
   7.89511824e+00  9.30463028e+00  5.30162811e+00  8.02445984e+00
   8.20812106e-01  1.48709321e+00  6.15253210e-01  6.70210779e-01
   2.34190989e+00  8.33582282e-01  3.83344352e-01  1.71879721e+00
   1.01789320e+00  1.81062889e+00  8.15129340e-01  1.90049648e-01
   1.87973738e-01  1.76477575e+00  1.96870637e+00  6.05706334e-01
   1.70101130e+00  5.97819030e-01  1.52859902e+00  2.68427610e-01
   8.94713402e-01  4.21253085e-01  8.57064486e-01  8.04610133e-01
   3.96816969e-01  4.27031636e-01  3.93537998e-01  1.78665471e+00
   4.69031632e-01  3.60497189e+00  4.11045253e-01  1.50516272e-01
   2.13350725e+00  2.62050223e+00  4.03833628e-01  2.63574886e+00
   1.81722021e+00  1.03061390e+00  1.11083925e-01  7.94399679e-01
   1.97345150e+00  1.00014544e+00  2.32414722e-01  6.99124098e-01
   5.33854187e-01  9.05500054e-02  1.56546593e+00  3.77516985e-01
   3.98273587e-01  8.50883782e-01  4.00637269e-01  5.59544027e-01
   1.17138481e+00  3.69059920e-01  2.98282790e+00  1.68951941e+00
   7.60360956e-02  7.61129379e-01  1.30164969e+00  9.12890494e-01
  -8.36846447e+00  9.28706360e+00 -3.32973170e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:13:08.938209
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4367
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 23:13:08.941864
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9132.78
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 23:13:08.944901
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.7766
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 23:13:08.947981
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -816.905
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139735822933352
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139734595915000
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139734595915504
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734595506472
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734595506976
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734595507480

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f16d5fa3da0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.504418
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.471859
grad_step = 000002, loss = 0.447732
grad_step = 000003, loss = 0.424260
grad_step = 000004, loss = 0.401341
grad_step = 000005, loss = 0.378610
grad_step = 000006, loss = 0.358461
grad_step = 000007, loss = 0.344089
grad_step = 000008, loss = 0.336605
grad_step = 000009, loss = 0.324193
grad_step = 000010, loss = 0.307709
grad_step = 000011, loss = 0.294061
grad_step = 000012, loss = 0.285639
grad_step = 000013, loss = 0.278335
grad_step = 000014, loss = 0.269038
grad_step = 000015, loss = 0.258621
grad_step = 000016, loss = 0.248064
grad_step = 000017, loss = 0.237489
grad_step = 000018, loss = 0.227300
grad_step = 000019, loss = 0.218065
grad_step = 000020, loss = 0.209357
grad_step = 000021, loss = 0.200419
grad_step = 000022, loss = 0.191178
grad_step = 000023, loss = 0.182343
grad_step = 000024, loss = 0.174159
grad_step = 000025, loss = 0.166019
grad_step = 000026, loss = 0.157570
grad_step = 000027, loss = 0.149201
grad_step = 000028, loss = 0.141531
grad_step = 000029, loss = 0.134558
grad_step = 000030, loss = 0.127632
grad_step = 000031, loss = 0.120573
grad_step = 000032, loss = 0.113698
grad_step = 000033, loss = 0.107182
grad_step = 000034, loss = 0.100983
grad_step = 000035, loss = 0.094980
grad_step = 000036, loss = 0.089201
grad_step = 000037, loss = 0.083672
grad_step = 000038, loss = 0.078437
grad_step = 000039, loss = 0.073479
grad_step = 000040, loss = 0.068736
grad_step = 000041, loss = 0.064132
grad_step = 000042, loss = 0.059799
grad_step = 000043, loss = 0.055798
grad_step = 000044, loss = 0.052007
grad_step = 000045, loss = 0.048342
grad_step = 000046, loss = 0.044935
grad_step = 000047, loss = 0.041793
grad_step = 000048, loss = 0.038777
grad_step = 000049, loss = 0.035900
grad_step = 000050, loss = 0.033265
grad_step = 000051, loss = 0.030844
grad_step = 000052, loss = 0.028569
grad_step = 000053, loss = 0.026440
grad_step = 000054, loss = 0.024459
grad_step = 000055, loss = 0.022620
grad_step = 000056, loss = 0.020917
grad_step = 000057, loss = 0.019331
grad_step = 000058, loss = 0.017861
grad_step = 000059, loss = 0.016499
grad_step = 000060, loss = 0.015237
grad_step = 000061, loss = 0.014084
grad_step = 000062, loss = 0.013015
grad_step = 000063, loss = 0.012017
grad_step = 000064, loss = 0.011114
grad_step = 000065, loss = 0.010279
grad_step = 000066, loss = 0.009501
grad_step = 000067, loss = 0.008810
grad_step = 000068, loss = 0.008177
grad_step = 000069, loss = 0.007583
grad_step = 000070, loss = 0.007047
grad_step = 000071, loss = 0.006563
grad_step = 000072, loss = 0.006122
grad_step = 000073, loss = 0.005720
grad_step = 000074, loss = 0.005359
grad_step = 000075, loss = 0.005035
grad_step = 000076, loss = 0.004734
grad_step = 000077, loss = 0.004465
grad_step = 000078, loss = 0.004229
grad_step = 000079, loss = 0.004009
grad_step = 000080, loss = 0.003810
grad_step = 000081, loss = 0.003635
grad_step = 000082, loss = 0.003475
grad_step = 000083, loss = 0.003330
grad_step = 000084, loss = 0.003201
grad_step = 000085, loss = 0.003086
grad_step = 000086, loss = 0.002980
grad_step = 000087, loss = 0.002887
grad_step = 000088, loss = 0.002803
grad_step = 000089, loss = 0.002726
grad_step = 000090, loss = 0.002659
grad_step = 000091, loss = 0.002599
grad_step = 000092, loss = 0.002544
grad_step = 000093, loss = 0.002496
grad_step = 000094, loss = 0.002452
grad_step = 000095, loss = 0.002414
grad_step = 000096, loss = 0.002379
grad_step = 000097, loss = 0.002349
grad_step = 000098, loss = 0.002322
grad_step = 000099, loss = 0.002298
grad_step = 000100, loss = 0.002276
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002257
grad_step = 000102, loss = 0.002241
grad_step = 000103, loss = 0.002225
grad_step = 000104, loss = 0.002212
grad_step = 000105, loss = 0.002201
grad_step = 000106, loss = 0.002190
grad_step = 000107, loss = 0.002181
grad_step = 000108, loss = 0.002172
grad_step = 000109, loss = 0.002165
grad_step = 000110, loss = 0.002158
grad_step = 000111, loss = 0.002152
grad_step = 000112, loss = 0.002146
grad_step = 000113, loss = 0.002140
grad_step = 000114, loss = 0.002135
grad_step = 000115, loss = 0.002130
grad_step = 000116, loss = 0.002125
grad_step = 000117, loss = 0.002120
grad_step = 000118, loss = 0.002115
grad_step = 000119, loss = 0.002110
grad_step = 000120, loss = 0.002106
grad_step = 000121, loss = 0.002101
grad_step = 000122, loss = 0.002096
grad_step = 000123, loss = 0.002090
grad_step = 000124, loss = 0.002085
grad_step = 000125, loss = 0.002079
grad_step = 000126, loss = 0.002074
grad_step = 000127, loss = 0.002068
grad_step = 000128, loss = 0.002062
grad_step = 000129, loss = 0.002056
grad_step = 000130, loss = 0.002050
grad_step = 000131, loss = 0.002044
grad_step = 000132, loss = 0.002039
grad_step = 000133, loss = 0.002035
grad_step = 000134, loss = 0.002035
grad_step = 000135, loss = 0.002042
grad_step = 000136, loss = 0.002053
grad_step = 000137, loss = 0.002070
grad_step = 000138, loss = 0.002063
grad_step = 000139, loss = 0.002038
grad_step = 000140, loss = 0.001999
grad_step = 000141, loss = 0.001986
grad_step = 000142, loss = 0.002000
grad_step = 000143, loss = 0.002008
grad_step = 000144, loss = 0.001994
grad_step = 000145, loss = 0.001967
grad_step = 000146, loss = 0.001958
grad_step = 000147, loss = 0.001966
grad_step = 000148, loss = 0.001967
grad_step = 000149, loss = 0.001953
grad_step = 000150, loss = 0.001935
grad_step = 000151, loss = 0.001931
grad_step = 000152, loss = 0.001934
grad_step = 000153, loss = 0.001931
grad_step = 000154, loss = 0.001919
grad_step = 000155, loss = 0.001906
grad_step = 000156, loss = 0.001902
grad_step = 000157, loss = 0.001902
grad_step = 000158, loss = 0.001898
grad_step = 000159, loss = 0.001889
grad_step = 000160, loss = 0.001879
grad_step = 000161, loss = 0.001871
grad_step = 000162, loss = 0.001868
grad_step = 000163, loss = 0.001865
grad_step = 000164, loss = 0.001861
grad_step = 000165, loss = 0.001854
grad_step = 000166, loss = 0.001845
grad_step = 000167, loss = 0.001838
grad_step = 000168, loss = 0.001831
grad_step = 000169, loss = 0.001826
grad_step = 000170, loss = 0.001822
grad_step = 000171, loss = 0.001819
grad_step = 000172, loss = 0.001816
grad_step = 000173, loss = 0.001813
grad_step = 000174, loss = 0.001812
grad_step = 000175, loss = 0.001811
grad_step = 000176, loss = 0.001811
grad_step = 000177, loss = 0.001811
grad_step = 000178, loss = 0.001811
grad_step = 000179, loss = 0.001810
grad_step = 000180, loss = 0.001805
grad_step = 000181, loss = 0.001797
grad_step = 000182, loss = 0.001783
grad_step = 000183, loss = 0.001769
grad_step = 000184, loss = 0.001757
grad_step = 000185, loss = 0.001748
grad_step = 000186, loss = 0.001741
grad_step = 000187, loss = 0.001737
grad_step = 000188, loss = 0.001734
grad_step = 000189, loss = 0.001731
grad_step = 000190, loss = 0.001731
grad_step = 000191, loss = 0.001738
grad_step = 000192, loss = 0.001761
grad_step = 000193, loss = 0.001810
grad_step = 000194, loss = 0.001876
grad_step = 000195, loss = 0.001938
grad_step = 000196, loss = 0.001926
grad_step = 000197, loss = 0.001793
grad_step = 000198, loss = 0.001696
grad_step = 000199, loss = 0.001750
grad_step = 000200, loss = 0.001810
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001762
grad_step = 000202, loss = 0.001711
grad_step = 000203, loss = 0.001724
grad_step = 000204, loss = 0.001729
grad_step = 000205, loss = 0.001718
grad_step = 000206, loss = 0.001719
grad_step = 000207, loss = 0.001696
grad_step = 000208, loss = 0.001676
grad_step = 000209, loss = 0.001706
grad_step = 000210, loss = 0.001704
grad_step = 000211, loss = 0.001661
grad_step = 000212, loss = 0.001675
grad_step = 000213, loss = 0.001684
grad_step = 000214, loss = 0.001671
grad_step = 000215, loss = 0.001665
grad_step = 000216, loss = 0.001658
grad_step = 000217, loss = 0.001662
grad_step = 000218, loss = 0.001647
grad_step = 000219, loss = 0.001644
grad_step = 000220, loss = 0.001657
grad_step = 000221, loss = 0.001640
grad_step = 000222, loss = 0.001629
grad_step = 000223, loss = 0.001636
grad_step = 000224, loss = 0.001631
grad_step = 000225, loss = 0.001623
grad_step = 000226, loss = 0.001624
grad_step = 000227, loss = 0.001618
grad_step = 000228, loss = 0.001617
grad_step = 000229, loss = 0.001621
grad_step = 000230, loss = 0.001612
grad_step = 000231, loss = 0.001606
grad_step = 000232, loss = 0.001608
grad_step = 000233, loss = 0.001607
grad_step = 000234, loss = 0.001603
grad_step = 000235, loss = 0.001601
grad_step = 000236, loss = 0.001599
grad_step = 000237, loss = 0.001599
grad_step = 000238, loss = 0.001609
grad_step = 000239, loss = 0.001633
grad_step = 000240, loss = 0.001670
grad_step = 000241, loss = 0.001732
grad_step = 000242, loss = 0.001772
grad_step = 000243, loss = 0.001830
grad_step = 000244, loss = 0.001789
grad_step = 000245, loss = 0.001661
grad_step = 000246, loss = 0.001581
grad_step = 000247, loss = 0.001631
grad_step = 000248, loss = 0.001694
grad_step = 000249, loss = 0.001670
grad_step = 000250, loss = 0.001609
grad_step = 000251, loss = 0.001587
grad_step = 000252, loss = 0.001610
grad_step = 000253, loss = 0.001639
grad_step = 000254, loss = 0.001622
grad_step = 000255, loss = 0.001579
grad_step = 000256, loss = 0.001571
grad_step = 000257, loss = 0.001604
grad_step = 000258, loss = 0.001608
grad_step = 000259, loss = 0.001575
grad_step = 000260, loss = 0.001564
grad_step = 000261, loss = 0.001576
grad_step = 000262, loss = 0.001581
grad_step = 000263, loss = 0.001574
grad_step = 000264, loss = 0.001562
grad_step = 000265, loss = 0.001556
grad_step = 000266, loss = 0.001562
grad_step = 000267, loss = 0.001568
grad_step = 000268, loss = 0.001558
grad_step = 000269, loss = 0.001547
grad_step = 000270, loss = 0.001549
grad_step = 000271, loss = 0.001554
grad_step = 000272, loss = 0.001551
grad_step = 000273, loss = 0.001546
grad_step = 000274, loss = 0.001543
grad_step = 000275, loss = 0.001541
grad_step = 000276, loss = 0.001539
grad_step = 000277, loss = 0.001540
grad_step = 000278, loss = 0.001540
grad_step = 000279, loss = 0.001536
grad_step = 000280, loss = 0.001531
grad_step = 000281, loss = 0.001530
grad_step = 000282, loss = 0.001530
grad_step = 000283, loss = 0.001530
grad_step = 000284, loss = 0.001528
grad_step = 000285, loss = 0.001527
grad_step = 000286, loss = 0.001525
grad_step = 000287, loss = 0.001522
grad_step = 000288, loss = 0.001519
grad_step = 000289, loss = 0.001518
grad_step = 000290, loss = 0.001517
grad_step = 000291, loss = 0.001517
grad_step = 000292, loss = 0.001516
grad_step = 000293, loss = 0.001514
grad_step = 000294, loss = 0.001513
grad_step = 000295, loss = 0.001513
grad_step = 000296, loss = 0.001513
grad_step = 000297, loss = 0.001512
grad_step = 000298, loss = 0.001512
grad_step = 000299, loss = 0.001513
grad_step = 000300, loss = 0.001516
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001520
grad_step = 000302, loss = 0.001527
grad_step = 000303, loss = 0.001535
grad_step = 000304, loss = 0.001545
grad_step = 000305, loss = 0.001551
grad_step = 000306, loss = 0.001554
grad_step = 000307, loss = 0.001546
grad_step = 000308, loss = 0.001534
grad_step = 000309, loss = 0.001517
grad_step = 000310, loss = 0.001500
grad_step = 000311, loss = 0.001488
grad_step = 000312, loss = 0.001486
grad_step = 000313, loss = 0.001490
grad_step = 000314, loss = 0.001496
grad_step = 000315, loss = 0.001501
grad_step = 000316, loss = 0.001505
grad_step = 000317, loss = 0.001506
grad_step = 000318, loss = 0.001504
grad_step = 000319, loss = 0.001501
grad_step = 000320, loss = 0.001494
grad_step = 000321, loss = 0.001487
grad_step = 000322, loss = 0.001481
grad_step = 000323, loss = 0.001475
grad_step = 000324, loss = 0.001471
grad_step = 000325, loss = 0.001467
grad_step = 000326, loss = 0.001465
grad_step = 000327, loss = 0.001463
grad_step = 000328, loss = 0.001462
grad_step = 000329, loss = 0.001461
grad_step = 000330, loss = 0.001460
grad_step = 000331, loss = 0.001459
grad_step = 000332, loss = 0.001459
grad_step = 000333, loss = 0.001461
grad_step = 000334, loss = 0.001466
grad_step = 000335, loss = 0.001478
grad_step = 000336, loss = 0.001503
grad_step = 000337, loss = 0.001547
grad_step = 000338, loss = 0.001615
grad_step = 000339, loss = 0.001692
grad_step = 000340, loss = 0.001756
grad_step = 000341, loss = 0.001745
grad_step = 000342, loss = 0.001632
grad_step = 000343, loss = 0.001491
grad_step = 000344, loss = 0.001456
grad_step = 000345, loss = 0.001534
grad_step = 000346, loss = 0.001597
grad_step = 000347, loss = 0.001541
grad_step = 000348, loss = 0.001451
grad_step = 000349, loss = 0.001449
grad_step = 000350, loss = 0.001512
grad_step = 000351, loss = 0.001530
grad_step = 000352, loss = 0.001476
grad_step = 000353, loss = 0.001433
grad_step = 000354, loss = 0.001449
grad_step = 000355, loss = 0.001476
grad_step = 000356, loss = 0.001466
grad_step = 000357, loss = 0.001445
grad_step = 000358, loss = 0.001444
grad_step = 000359, loss = 0.001450
grad_step = 000360, loss = 0.001437
grad_step = 000361, loss = 0.001419
grad_step = 000362, loss = 0.001423
grad_step = 000363, loss = 0.001438
grad_step = 000364, loss = 0.001441
grad_step = 000365, loss = 0.001427
grad_step = 000366, loss = 0.001411
grad_step = 000367, loss = 0.001408
grad_step = 000368, loss = 0.001413
grad_step = 000369, loss = 0.001414
grad_step = 000370, loss = 0.001408
grad_step = 000371, loss = 0.001403
grad_step = 000372, loss = 0.001404
grad_step = 000373, loss = 0.001408
grad_step = 000374, loss = 0.001408
grad_step = 000375, loss = 0.001403
grad_step = 000376, loss = 0.001397
grad_step = 000377, loss = 0.001393
grad_step = 000378, loss = 0.001393
grad_step = 000379, loss = 0.001391
grad_step = 000380, loss = 0.001388
grad_step = 000381, loss = 0.001383
grad_step = 000382, loss = 0.001379
grad_step = 000383, loss = 0.001378
grad_step = 000384, loss = 0.001378
grad_step = 000385, loss = 0.001377
grad_step = 000386, loss = 0.001376
grad_step = 000387, loss = 0.001375
grad_step = 000388, loss = 0.001376
grad_step = 000389, loss = 0.001381
grad_step = 000390, loss = 0.001394
grad_step = 000391, loss = 0.001418
grad_step = 000392, loss = 0.001464
grad_step = 000393, loss = 0.001515
grad_step = 000394, loss = 0.001577
grad_step = 000395, loss = 0.001552
grad_step = 000396, loss = 0.001491
grad_step = 000397, loss = 0.001405
grad_step = 000398, loss = 0.001392
grad_step = 000399, loss = 0.001425
grad_step = 000400, loss = 0.001414
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001383
grad_step = 000402, loss = 0.001378
grad_step = 000403, loss = 0.001405
grad_step = 000404, loss = 0.001415
grad_step = 000405, loss = 0.001383
grad_step = 000406, loss = 0.001349
grad_step = 000407, loss = 0.001345
grad_step = 000408, loss = 0.001366
grad_step = 000409, loss = 0.001384
grad_step = 000410, loss = 0.001381
grad_step = 000411, loss = 0.001359
grad_step = 000412, loss = 0.001337
grad_step = 000413, loss = 0.001332
grad_step = 000414, loss = 0.001341
grad_step = 000415, loss = 0.001353
grad_step = 000416, loss = 0.001356
grad_step = 000417, loss = 0.001347
grad_step = 000418, loss = 0.001333
grad_step = 000419, loss = 0.001323
grad_step = 000420, loss = 0.001320
grad_step = 000421, loss = 0.001324
grad_step = 000422, loss = 0.001329
grad_step = 000423, loss = 0.001332
grad_step = 000424, loss = 0.001329
grad_step = 000425, loss = 0.001323
grad_step = 000426, loss = 0.001316
grad_step = 000427, loss = 0.001311
grad_step = 000428, loss = 0.001309
grad_step = 000429, loss = 0.001310
grad_step = 000430, loss = 0.001312
grad_step = 000431, loss = 0.001315
grad_step = 000432, loss = 0.001319
grad_step = 000433, loss = 0.001324
grad_step = 000434, loss = 0.001331
grad_step = 000435, loss = 0.001344
grad_step = 000436, loss = 0.001367
grad_step = 000437, loss = 0.001406
grad_step = 000438, loss = 0.001451
grad_step = 000439, loss = 0.001509
grad_step = 000440, loss = 0.001505
grad_step = 000441, loss = 0.001470
grad_step = 000442, loss = 0.001386
grad_step = 000443, loss = 0.001338
grad_step = 000444, loss = 0.001339
grad_step = 000445, loss = 0.001347
grad_step = 000446, loss = 0.001333
grad_step = 000447, loss = 0.001300
grad_step = 000448, loss = 0.001299
grad_step = 000449, loss = 0.001330
grad_step = 000450, loss = 0.001352
grad_step = 000451, loss = 0.001344
grad_step = 000452, loss = 0.001308
grad_step = 000453, loss = 0.001278
grad_step = 000454, loss = 0.001271
grad_step = 000455, loss = 0.001285
grad_step = 000456, loss = 0.001303
grad_step = 000457, loss = 0.001306
grad_step = 000458, loss = 0.001296
grad_step = 000459, loss = 0.001278
grad_step = 000460, loss = 0.001266
grad_step = 000461, loss = 0.001264
grad_step = 000462, loss = 0.001270
grad_step = 000463, loss = 0.001276
grad_step = 000464, loss = 0.001276
grad_step = 000465, loss = 0.001271
grad_step = 000466, loss = 0.001261
grad_step = 000467, loss = 0.001252
grad_step = 000468, loss = 0.001247
grad_step = 000469, loss = 0.001247
grad_step = 000470, loss = 0.001250
grad_step = 000471, loss = 0.001254
grad_step = 000472, loss = 0.001255
grad_step = 000473, loss = 0.001253
grad_step = 000474, loss = 0.001250
grad_step = 000475, loss = 0.001245
grad_step = 000476, loss = 0.001241
grad_step = 000477, loss = 0.001238
grad_step = 000478, loss = 0.001238
grad_step = 000479, loss = 0.001240
grad_step = 000480, loss = 0.001244
grad_step = 000481, loss = 0.001249
grad_step = 000482, loss = 0.001260
grad_step = 000483, loss = 0.001280
grad_step = 000484, loss = 0.001316
grad_step = 000485, loss = 0.001357
grad_step = 000486, loss = 0.001405
grad_step = 000487, loss = 0.001426
grad_step = 000488, loss = 0.001426
grad_step = 000489, loss = 0.001393
grad_step = 000490, loss = 0.001398
grad_step = 000491, loss = 0.001436
grad_step = 000492, loss = 0.001523
grad_step = 000493, loss = 0.001423
grad_step = 000494, loss = 0.001304
grad_step = 000495, loss = 0.001257
grad_step = 000496, loss = 0.001321
grad_step = 000497, loss = 0.001339
grad_step = 000498, loss = 0.001245
grad_step = 000499, loss = 0.001232
grad_step = 000500, loss = 0.001303
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001313
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

  date_run                              2020-05-10 23:13:27.649328
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.22469
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 23:13:27.655188
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.116742
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 23:13:27.663698
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.142383
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 23:13:27.668761
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.773941
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
0   2020-05-10 23:12:58.403794  ...    mean_absolute_error
1   2020-05-10 23:12:58.407701  ...     mean_squared_error
2   2020-05-10 23:12:58.411165  ...  median_absolute_error
3   2020-05-10 23:12:58.414288  ...               r2_score
4   2020-05-10 23:13:08.938209  ...    mean_absolute_error
5   2020-05-10 23:13:08.941864  ...     mean_squared_error
6   2020-05-10 23:13:08.944901  ...  median_absolute_error
7   2020-05-10 23:13:08.947981  ...               r2_score
8   2020-05-10 23:13:27.649328  ...    mean_absolute_error
9   2020-05-10 23:13:27.655188  ...     mean_squared_error
10  2020-05-10 23:13:27.663698  ...  median_absolute_error
11  2020-05-10 23:13:27.668761  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do timeseries 





 ************************************************************************************************************************

  vision_mnist 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_cnn/mnist 

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▌      | 3497984/9912422 [00:00<00:00, 34785390.99it/s]9920512it [00:00, 39195439.28it/s]                             
0it [00:00, ?it/s]32768it [00:00, 581301.91it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 460221.30it/s]1654784it [00:00, 11525671.46it/s]                         
0it [00:00, ?it/s]8192it [00:00, 206709.93it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d138e9e8> <class 'mlmodels.model_tch.torchhub.Model'>
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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa96eaddc18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d1346e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa96eaddda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d1346e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa983d43cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d1393f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d138e9e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa983d55be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa9d138e9e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa983d43cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f72dbba01d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2cd04d025aba875531bdaffcbc0ed3dbb4c7f19d7f6efdf73ba9d12d9dccbe96
  Stored in directory: /tmp/pip-ephem-wheel-cache-njywmaur/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.0.2; however, version 20.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f72d3a21f98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2408448/17464789 [===>..........................] - ETA: 0s
 6799360/17464789 [==========>...................] - ETA: 0s
11091968/17464789 [==================>...........] - ETA: 0s
15368192/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 23:14:54.024732: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 23:14:54.029189: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-10 23:14:54.029388: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556afdd62760 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 23:14:54.029402: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4903 - accuracy: 0.5115 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6871 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7024 - accuracy: 0.4977
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6622 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6455 - accuracy: 0.5014
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 3s - loss: 7.6597 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 3s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6383 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 2s - loss: 7.6707 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6896 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6856 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6903 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6786 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6807 - accuracy: 0.4991
25000/25000 [==============================] - 7s 281us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:15:07.695472
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 23:15:07.695472  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}] 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Using TensorFlow backend.
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
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 75, 40)            1720      
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
2020-05-10 23:15:13.906133: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 23:15:13.912419: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-10 23:15:13.912643: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dc1cb4ea50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 23:15:13.912659: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1753b7bd68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2379 - crf_viterbi_accuracy: 0.6667 - val_loss: 1.1960 - val_crf_viterbi_accuracy: 0.6933

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
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
dense_2 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1770002fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.1453 - accuracy: 0.5340
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6329 - accuracy: 0.5022
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6075 - accuracy: 0.5039
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6398 - accuracy: 0.5017
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 3s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6572 - accuracy: 0.5006
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6683 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6505 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 7s 287us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f170eb53470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:58:46, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:19:05, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:28:35, 20.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:01<8:02:23, 29.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.63M/862M [00:01<5:37:14, 42.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.66M/862M [00:01<3:55:05, 60.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<2:43:37, 86.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.2M/862M [00:01<1:54:04, 124kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<1:19:26, 176kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:01<55:30, 251kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<38:44, 358kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:02<27:05, 510kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:02<18:57, 725kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<13:18, 1.03MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<09:20, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:02<06:37, 2.04MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<05:34, 2.42MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:02<03:59, 3.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:04<30:54, 435kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<24:55, 539kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<18:15, 735kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<12:56, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<17:34, 761kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<13:56, 958kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<10:09, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<09:47, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<08:14, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:09<04:31, 2.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<10:03, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<08:39, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<06:23, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:11<04:37, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<6:37:14, 33.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<4:41:24, 46.7kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<3:17:31, 66.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<2:17:58, 95.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<1:44:34, 125kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<1:14:32, 176kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<52:30, 249kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<39:16, 331kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<29:06, 447kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<20:42, 627kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<17:05, 757kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<15:17, 846kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<11:23, 1.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:18<08:07, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<11:02, 1.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<09:18, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<06:51, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:23, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<08:18, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<06:31, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:22<04:43, 2.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<07:45, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<05:45, 2.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<04:10, 3.04MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:40:55, 16.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<8:55:37, 23.7kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<6:15:16, 33.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<4:22:13, 48.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<3:07:32, 67.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:12:48, 94.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<1:33:08, 135kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:07:31, 186kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<50:19, 249kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<35:51, 349kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<25:17, 494kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<21:02, 593kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<16:17, 765kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<11:41, 1.06MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<08:18, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<4:13:50, 48.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<2:59:10, 69.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<2:05:31, 98.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<1:27:42, 141kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<4:06:09, 50.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:53:47, 70.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<2:01:49, 101kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:27:28, 140kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:04:13, 191kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<45:34, 269kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<31:56, 382kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<28:54, 422kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<21:45, 560kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<15:31, 784kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<10:59, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<36:46, 330kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<27:12, 445kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<19:22, 624kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:58, 754kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<14:09, 851kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<10:31, 1.14MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<07:35, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:28, 1.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:23, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:15, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:28, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:51, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:17, 2.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:38, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:05, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:37, 2.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:36, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:22, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:55, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:27, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:39, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:00, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:51, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:56, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:03, 2.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<10:00, 1.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:25, 1.37MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<06:11, 1.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<04:28, 2.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<29:36, 388kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<22:04, 520kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<15:46, 727kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<13:22, 854kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<12:11, 937kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<09:07, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<06:37, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:24, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:34, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:43, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:46, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:53, 2.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:31, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:29, 2.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:24, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:39, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:20, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:53, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:47, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:01, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:26, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:12, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:36, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:04, 2.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:20, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:28, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:49, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:33, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:15, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:00, 2.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:59, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:41, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:32, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:48, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:51, 2.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<03:33, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<09:16, 1.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:47, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:44, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:09, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:28, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:00, 2.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:06, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:03, 2.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:00, 3.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:54, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:13, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:23, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<09:12, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:43, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:42, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:04, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:46, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:16, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:49, 2.68MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:50, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:02, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:32, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:13, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:08, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:49, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:36, 2.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:56, 2.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:32, 2.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:23, 2.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:19, 3.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:21, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:30, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:22, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:15, 3.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:51, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:36, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:31, 2.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:27, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:19, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:12, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:04, 3.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<06:18, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:36, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:12, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:54, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:48, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:19, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:58, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:21, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:01, 2.39MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:43, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:38, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:26, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:14, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:03, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:23, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:03, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:44, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:29, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:25, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:16, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<05:18, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:13, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:05, 3.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:26, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:58, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:43, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:28, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:23, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:08, 2.91MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:27, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:22, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<04:41, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:07, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:47, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:32, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:21, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:40, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:22, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:17, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:07, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:04, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:01, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:58, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:39, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:20, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:17, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:05, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:00, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:57, 2.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:53, 3.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:19, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:47, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:34, 2.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<02:35, 3.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<2:55:22, 49.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<2:04:50, 69.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<1:27:43, 98.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<1:01:20, 140kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<45:25, 189kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<32:49, 261kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<23:09, 369kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<17:51, 476kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<14:41, 579kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<10:43, 792kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<07:40, 1.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:26, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:14, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:37, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:55, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:19, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:07, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:03, 2.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:14, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:02, 2.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:34, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:41, 3.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:38, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:32, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:38, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:40, 3.04MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:20, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:25, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:59, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:25, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:03, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:02, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<02:55, 2.73MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:37, 1.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:36, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:07, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:28, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:03, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:01, 1.96MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:43<02:55, 2.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:58, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:44, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:12, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:36, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:07, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:03, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<02:56, 2.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:57, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:47, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:30, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:06, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:57, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:53, 2.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:14, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:53, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:56, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:34, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:25, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:35, 2.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<01:53, 3.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<24:39, 300kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<18:09, 407kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<12:51, 573kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<10:26, 702kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<09:06, 805kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<06:44, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:52, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:08, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:30, 1.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:20, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:47, 1.90MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:31, 2.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:39, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:18, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:08, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<03:19, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:25, 2.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<06:12, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:12, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:50, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:35, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:43, 2.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<01:59, 3.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<20:17, 341kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<15:03, 459kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<10:41, 645kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<08:49, 776kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:01, 974kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:05, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:55, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:56, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:50, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:44, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<09:06, 736kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<07:03, 949kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:08, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<03:40, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<09:26, 704kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<07:25, 893kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:23, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:05, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:09, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:01, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:53, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:01, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:00, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:41, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:52, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:17, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:23, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:26, 2.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<05:22, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:31, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:19, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:35, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:03, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:10, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:17, 2.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:47, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:02, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:58, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:24, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:54, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:03, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:12, 2.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:11, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:40, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:57, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:38, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:00, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:57, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:16, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:44, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:58, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:09, 2.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<05:15, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:23, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:14, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:26, 1.69MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:51, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:02, 1.90MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:12, 2.61MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:11, 1.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:20, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:10, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:21, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:00, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:09, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:00, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:11, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:03, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:15, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:42, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:55, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:06, 2.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:57, 1.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:08, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:02, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:12, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:33, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:46, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:01, 2.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:53, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:40, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:01, 2.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:28, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:47, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:18, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:53, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:17, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:43, 2.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:23, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:17, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:45, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:14, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:11, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:39, 3.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:13, 4.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:50, 1.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:00, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:55, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<11:56, 411kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<08:57, 547kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:23, 763kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:25, 893kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:02, 958kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:46, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:42, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:27, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:52, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:08, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:30, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:19, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:45, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:09, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:05, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:34, 2.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:01, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:32, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:03, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:29, 3.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:52, 1.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:23, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<01:43, 2.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<08:22, 528kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<06:50, 646kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<05:01, 877kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:31, 1.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<06:53, 632kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<05:17, 820kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<03:47, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<02:41, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<30:29, 140kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<22:23, 191kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<15:53, 269kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<11:04, 382kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<10:25, 404kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<07:45, 543kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<05:29, 761kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:42, 879kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:48, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:46, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:45, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:26, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:04, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:25, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:54, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:23, 2.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:06, 1.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:58, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:29, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:44, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:20, 2.88MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:01, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:35, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:09, 3.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:07, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:56, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:26, 2.56MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:44, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:40, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:16, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:33, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:10, 3.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<00:51, 4.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<30:54, 114kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<22:01, 160kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<15:25, 226kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<11:22, 303kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<08:47, 392kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<06:19, 544kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<04:25, 769kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:28, 755kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:33, 950kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:33, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:26, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:06, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:44, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:00, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:34, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:09, 2.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:37, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:31, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:09, 2.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:25, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:21, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:01, 2.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:37, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:16, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<00:55, 3.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:47, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:37, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:12, 2.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.70MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:21, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:58, 2.89MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:26, 1.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:03, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:30, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:35, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:20, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:57, 2.81MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:43, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:30, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:06, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:20, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:34, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:14, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:53, 2.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:20, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:00, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:44, 3.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:52, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:56, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:28, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:04, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:24, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:16, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:57, 2.50MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:07, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:18, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:01, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:45, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:08, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:04, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:48, 2.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:00, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:00, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:43, 2.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:49, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:32, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:07, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:11, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:17, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:00, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:43, 2.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:55, 682kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:16, 871kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:38, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:30, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:29, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:07, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:48, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:06, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:49, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:55, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:52, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:39, 2.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:58, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:45, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:33, 3.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:56, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:38, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:45, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:31, 2.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:17, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:02, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:45, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:32, 2.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:40, 857kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:20, 1.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:58, 1.46MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:56, 1.46MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:59, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:32, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:11, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:59, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:43, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:44, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:50, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:38, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:27, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:47, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:30, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:22, 2.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:16, 3.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:14, 819kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:58, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:41, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:40, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:22, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:48, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:40, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:29, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:32, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:24, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:17, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:29, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:25, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:24, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:13, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:18, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:13, 2.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:15, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:10, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:07, 3.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:29, 820kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:22, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:15, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:14, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:02, 3.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 3.46MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 979kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.63MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 908/400000 [00:00<00:43, 9077.30it/s]  0%|          | 1791/400000 [00:00<00:44, 8999.00it/s]  1%|          | 2719/400000 [00:00<00:43, 9079.05it/s]  1%|          | 3620/400000 [00:00<00:43, 9056.73it/s]  1%|          | 4408/400000 [00:00<00:45, 8666.49it/s]  1%|▏         | 5308/400000 [00:00<00:45, 8761.40it/s]  2%|▏         | 6204/400000 [00:00<00:44, 8817.59it/s]  2%|▏         | 7035/400000 [00:00<00:45, 8657.26it/s]  2%|▏         | 7936/400000 [00:00<00:44, 8757.78it/s]  2%|▏         | 8831/400000 [00:01<00:44, 8811.91it/s]  2%|▏         | 9688/400000 [00:01<00:44, 8685.23it/s]  3%|▎         | 10540/400000 [00:01<00:45, 8543.49it/s]  3%|▎         | 11438/400000 [00:01<00:44, 8667.63it/s]  3%|▎         | 12324/400000 [00:01<00:44, 8722.81it/s]  3%|▎         | 13209/400000 [00:01<00:44, 8742.70it/s]  4%|▎         | 14080/400000 [00:01<00:44, 8611.57it/s]  4%|▎         | 14957/400000 [00:01<00:44, 8658.39it/s]  4%|▍         | 15857/400000 [00:01<00:43, 8756.96it/s]  4%|▍         | 16741/400000 [00:01<00:43, 8781.30it/s]  4%|▍         | 17619/400000 [00:02<00:43, 8779.30it/s]  5%|▍         | 18497/400000 [00:02<00:43, 8756.77it/s]  5%|▍         | 19415/400000 [00:02<00:42, 8876.90it/s]  5%|▌         | 20325/400000 [00:02<00:42, 8942.03it/s]  5%|▌         | 21227/400000 [00:02<00:42, 8962.65it/s]  6%|▌         | 22146/400000 [00:02<00:41, 9028.51it/s]  6%|▌         | 23050/400000 [00:02<00:42, 8969.40it/s]  6%|▌         | 23948/400000 [00:02<00:42, 8893.95it/s]  6%|▌         | 24845/400000 [00:02<00:42, 8915.85it/s]  6%|▋         | 25749/400000 [00:02<00:41, 8950.85it/s]  7%|▋         | 26645/400000 [00:03<00:41, 8922.98it/s]  7%|▋         | 27538/400000 [00:03<00:41, 8900.96it/s]  7%|▋         | 28453/400000 [00:03<00:41, 8971.93it/s]  7%|▋         | 29351/400000 [00:03<00:41, 8956.92it/s]  8%|▊         | 30247/400000 [00:03<00:41, 8940.51it/s]  8%|▊         | 31142/400000 [00:03<00:41, 8916.99it/s]  8%|▊         | 32055/400000 [00:03<00:40, 8978.52it/s]  8%|▊         | 32954/400000 [00:03<00:41, 8882.23it/s]  8%|▊         | 33843/400000 [00:03<00:41, 8836.98it/s]  9%|▊         | 34727/400000 [00:03<00:41, 8778.10it/s]  9%|▉         | 35606/400000 [00:04<00:41, 8749.23it/s]  9%|▉         | 36488/400000 [00:04<00:41, 8767.37it/s]  9%|▉         | 37383/400000 [00:04<00:41, 8819.53it/s] 10%|▉         | 38266/400000 [00:04<00:41, 8811.37it/s] 10%|▉         | 39148/400000 [00:04<00:41, 8787.53it/s] 10%|█         | 40027/400000 [00:04<00:41, 8735.43it/s] 10%|█         | 40901/400000 [00:04<00:41, 8552.70it/s] 10%|█         | 41758/400000 [00:04<00:42, 8515.13it/s] 11%|█         | 42611/400000 [00:04<00:42, 8497.14it/s] 11%|█         | 43491/400000 [00:04<00:41, 8584.74it/s] 11%|█         | 44358/400000 [00:05<00:41, 8609.46it/s] 11%|█▏        | 45229/400000 [00:05<00:41, 8637.46it/s] 12%|█▏        | 46106/400000 [00:05<00:40, 8676.13it/s] 12%|█▏        | 46999/400000 [00:05<00:40, 8748.24it/s] 12%|█▏        | 47893/400000 [00:05<00:40, 8802.20it/s] 12%|█▏        | 48778/400000 [00:05<00:39, 8815.49it/s] 12%|█▏        | 49660/400000 [00:05<00:40, 8551.58it/s] 13%|█▎        | 50518/400000 [00:05<00:41, 8450.92it/s] 13%|█▎        | 51365/400000 [00:05<00:41, 8433.86it/s] 13%|█▎        | 52229/400000 [00:05<00:40, 8493.46it/s] 13%|█▎        | 53108/400000 [00:06<00:40, 8579.08it/s] 14%|█▎        | 54002/400000 [00:06<00:39, 8681.96it/s] 14%|█▎        | 54872/400000 [00:06<00:40, 8516.68it/s] 14%|█▍        | 55735/400000 [00:06<00:40, 8549.93it/s] 14%|█▍        | 56611/400000 [00:06<00:39, 8609.71it/s] 14%|█▍        | 57488/400000 [00:06<00:39, 8656.64it/s] 15%|█▍        | 58361/400000 [00:06<00:39, 8675.83it/s] 15%|█▍        | 59230/400000 [00:06<00:39, 8600.09it/s] 15%|█▌        | 60091/400000 [00:06<00:39, 8542.55it/s] 15%|█▌        | 60951/400000 [00:06<00:39, 8559.25it/s] 15%|█▌        | 61808/400000 [00:07<00:41, 8229.77it/s] 16%|█▌        | 62670/400000 [00:07<00:40, 8341.97it/s] 16%|█▌        | 63507/400000 [00:07<00:41, 8180.34it/s] 16%|█▌        | 64328/400000 [00:07<00:40, 8187.52it/s] 16%|█▋        | 65149/400000 [00:07<00:45, 7333.85it/s] 17%|█▋        | 66004/400000 [00:07<00:43, 7659.14it/s] 17%|█▋        | 66866/400000 [00:07<00:42, 7922.71it/s] 17%|█▋        | 67755/400000 [00:07<00:40, 8188.31it/s] 17%|█▋        | 68618/400000 [00:07<00:39, 8313.80it/s] 17%|█▋        | 69459/400000 [00:08<00:40, 8187.04it/s] 18%|█▊        | 70350/400000 [00:08<00:39, 8390.42it/s] 18%|█▊        | 71240/400000 [00:08<00:38, 8535.93it/s] 18%|█▊        | 72137/400000 [00:08<00:37, 8661.54it/s] 18%|█▊        | 73007/400000 [00:08<00:37, 8662.62it/s] 18%|█▊        | 73895/400000 [00:08<00:37, 8724.78it/s] 19%|█▊        | 74797/400000 [00:08<00:36, 8811.18it/s] 19%|█▉        | 75680/400000 [00:08<00:36, 8811.08it/s] 19%|█▉        | 76563/400000 [00:08<00:38, 8327.11it/s] 19%|█▉        | 77444/400000 [00:08<00:38, 8465.54it/s] 20%|█▉        | 78318/400000 [00:09<00:37, 8545.05it/s] 20%|█▉        | 79196/400000 [00:09<00:37, 8611.82it/s] 20%|██        | 80060/400000 [00:09<00:37, 8609.86it/s] 20%|██        | 80964/400000 [00:09<00:36, 8732.67it/s] 20%|██        | 81839/400000 [00:09<00:36, 8663.00it/s] 21%|██        | 82707/400000 [00:09<00:36, 8628.56it/s] 21%|██        | 83586/400000 [00:09<00:36, 8674.33it/s] 21%|██        | 84455/400000 [00:09<00:36, 8628.50it/s] 21%|██▏       | 85319/400000 [00:09<00:40, 7755.74it/s] 22%|██▏       | 86112/400000 [00:10<00:42, 7318.18it/s] 22%|██▏       | 86974/400000 [00:10<00:40, 7665.22it/s] 22%|██▏       | 87852/400000 [00:10<00:39, 7968.02it/s] 22%|██▏       | 88718/400000 [00:10<00:38, 8161.39it/s] 22%|██▏       | 89546/400000 [00:10<00:38, 8033.46it/s] 23%|██▎       | 90396/400000 [00:10<00:37, 8165.40it/s] 23%|██▎       | 91291/400000 [00:10<00:36, 8385.36it/s] 23%|██▎       | 92226/400000 [00:10<00:35, 8651.43it/s] 23%|██▎       | 93143/400000 [00:10<00:34, 8798.17it/s] 24%|██▎       | 94074/400000 [00:10<00:34, 8944.78it/s] 24%|██▍       | 95009/400000 [00:11<00:33, 9061.73it/s] 24%|██▍       | 95919/400000 [00:11<00:34, 8885.78it/s] 24%|██▍       | 96811/400000 [00:11<00:34, 8816.90it/s] 24%|██▍       | 97695/400000 [00:11<00:34, 8721.65it/s] 25%|██▍       | 98569/400000 [00:11<00:35, 8528.89it/s] 25%|██▍       | 99425/400000 [00:11<00:36, 8206.53it/s] 25%|██▌       | 100304/400000 [00:11<00:35, 8371.19it/s] 25%|██▌       | 101147/400000 [00:11<00:35, 8388.61it/s] 26%|██▌       | 102015/400000 [00:11<00:35, 8473.87it/s] 26%|██▌       | 102865/400000 [00:11<00:35, 8371.44it/s] 26%|██▌       | 103704/400000 [00:12<00:35, 8279.48it/s] 26%|██▌       | 104534/400000 [00:12<00:36, 8149.70it/s] 26%|██▋       | 105408/400000 [00:12<00:35, 8317.07it/s] 27%|██▋       | 106325/400000 [00:12<00:34, 8554.46it/s] 27%|██▋       | 107202/400000 [00:12<00:33, 8616.00it/s] 27%|██▋       | 108066/400000 [00:12<00:34, 8386.51it/s] 27%|██▋       | 108943/400000 [00:12<00:34, 8497.86it/s] 27%|██▋       | 109797/400000 [00:12<00:34, 8507.62it/s] 28%|██▊       | 110672/400000 [00:12<00:33, 8576.27it/s] 28%|██▊       | 111531/400000 [00:12<00:33, 8552.00it/s] 28%|██▊       | 112390/400000 [00:13<00:33, 8560.60it/s] 28%|██▊       | 113258/400000 [00:13<00:33, 8593.38it/s] 29%|██▊       | 114118/400000 [00:13<00:33, 8550.37it/s] 29%|██▉       | 115004/400000 [00:13<00:32, 8639.91it/s] 29%|██▉       | 115875/400000 [00:13<00:32, 8659.60it/s] 29%|██▉       | 116799/400000 [00:13<00:32, 8823.60it/s] 29%|██▉       | 117690/400000 [00:13<00:31, 8847.04it/s] 30%|██▉       | 118601/400000 [00:13<00:31, 8923.24it/s] 30%|██▉       | 119494/400000 [00:13<00:31, 8882.41it/s] 30%|███       | 120383/400000 [00:14<00:31, 8857.63it/s] 30%|███       | 121270/400000 [00:14<00:31, 8822.21it/s] 31%|███       | 122153/400000 [00:14<00:31, 8790.77it/s] 31%|███       | 123040/400000 [00:14<00:31, 8814.38it/s] 31%|███       | 123922/400000 [00:14<00:31, 8670.18it/s] 31%|███       | 124790/400000 [00:14<00:32, 8493.38it/s] 31%|███▏      | 125641/400000 [00:14<00:32, 8445.37it/s] 32%|███▏      | 126487/400000 [00:14<00:33, 8055.31it/s] 32%|███▏      | 127369/400000 [00:14<00:32, 8268.32it/s] 32%|███▏      | 128251/400000 [00:14<00:32, 8425.26it/s] 32%|███▏      | 129098/400000 [00:15<00:33, 8032.72it/s] 32%|███▏      | 129938/400000 [00:15<00:33, 8139.08it/s] 33%|███▎      | 130784/400000 [00:15<00:32, 8232.13it/s] 33%|███▎      | 131688/400000 [00:15<00:31, 8457.33it/s] 33%|███▎      | 132565/400000 [00:15<00:31, 8547.68it/s] 33%|███▎      | 133468/400000 [00:15<00:30, 8686.52it/s] 34%|███▎      | 134340/400000 [00:15<00:30, 8681.88it/s] 34%|███▍      | 135228/400000 [00:15<00:30, 8739.73it/s] 34%|███▍      | 136156/400000 [00:15<00:29, 8893.00it/s] 34%|███▍      | 137099/400000 [00:15<00:29, 9045.44it/s] 35%|███▍      | 138019/400000 [00:16<00:28, 9089.67it/s] 35%|███▍      | 138930/400000 [00:16<00:29, 8913.05it/s] 35%|███▍      | 139823/400000 [00:16<00:29, 8842.13it/s] 35%|███▌      | 140712/400000 [00:16<00:29, 8854.26it/s] 35%|███▌      | 141599/400000 [00:16<00:29, 8786.63it/s] 36%|███▌      | 142490/400000 [00:16<00:29, 8822.97it/s] 36%|███▌      | 143417/400000 [00:16<00:28, 8951.49it/s] 36%|███▌      | 144313/400000 [00:16<00:28, 8879.92it/s] 36%|███▋      | 145202/400000 [00:16<00:28, 8880.68it/s] 37%|███▋      | 146105/400000 [00:16<00:28, 8923.40it/s] 37%|███▋      | 147013/400000 [00:17<00:28, 8968.50it/s] 37%|███▋      | 147911/400000 [00:17<00:28, 8936.38it/s] 37%|███▋      | 148805/400000 [00:17<00:28, 8907.20it/s] 37%|███▋      | 149727/400000 [00:17<00:27, 8997.33it/s] 38%|███▊      | 150670/400000 [00:17<00:27, 9121.90it/s] 38%|███▊      | 151599/400000 [00:17<00:27, 9169.26it/s] 38%|███▊      | 152517/400000 [00:17<00:27, 8999.77it/s] 38%|███▊      | 153419/400000 [00:17<00:27, 8995.51it/s] 39%|███▊      | 154320/400000 [00:17<00:27, 8790.19it/s] 39%|███▉      | 155249/400000 [00:17<00:27, 8933.35it/s] 39%|███▉      | 156156/400000 [00:18<00:27, 8973.55it/s] 39%|███▉      | 157075/400000 [00:18<00:26, 9036.02it/s] 40%|███▉      | 158003/400000 [00:18<00:26, 9106.07it/s] 40%|███▉      | 158921/400000 [00:18<00:26, 9128.07it/s] 40%|███▉      | 159835/400000 [00:18<00:26, 9004.41it/s] 40%|████      | 160737/400000 [00:18<00:27, 8848.65it/s] 40%|████      | 161624/400000 [00:18<00:26, 8830.34it/s] 41%|████      | 162545/400000 [00:18<00:26, 8938.43it/s] 41%|████      | 163493/400000 [00:18<00:26, 9093.23it/s] 41%|████      | 164406/400000 [00:18<00:25, 9102.52it/s] 41%|████▏     | 165336/400000 [00:19<00:25, 9159.72it/s] 42%|████▏     | 166277/400000 [00:19<00:25, 9233.36it/s] 42%|████▏     | 167201/400000 [00:19<00:25, 9177.33it/s] 42%|████▏     | 168120/400000 [00:19<00:25, 9051.04it/s] 42%|████▏     | 169084/400000 [00:19<00:25, 9218.37it/s] 43%|████▎     | 170025/400000 [00:19<00:24, 9273.50it/s] 43%|████▎     | 170954/400000 [00:19<00:24, 9269.40it/s] 43%|████▎     | 171900/400000 [00:19<00:24, 9324.25it/s] 43%|████▎     | 172833/400000 [00:19<00:24, 9279.97it/s] 43%|████▎     | 173762/400000 [00:20<00:24, 9266.91it/s] 44%|████▎     | 174690/400000 [00:20<00:24, 9110.68it/s] 44%|████▍     | 175602/400000 [00:20<00:24, 9075.51it/s] 44%|████▍     | 176511/400000 [00:20<00:24, 9064.89it/s] 44%|████▍     | 177418/400000 [00:20<00:24, 9005.28it/s] 45%|████▍     | 178325/400000 [00:20<00:24, 9023.83it/s] 45%|████▍     | 179231/400000 [00:20<00:24, 9021.26it/s] 45%|████▌     | 180134/400000 [00:20<00:24, 8958.99it/s] 45%|████▌     | 181031/400000 [00:20<00:24, 8907.20it/s] 45%|████▌     | 181926/400000 [00:20<00:24, 8918.09it/s] 46%|████▌     | 182834/400000 [00:21<00:24, 8960.81it/s] 46%|████▌     | 183736/400000 [00:21<00:24, 8976.92it/s] 46%|████▌     | 184693/400000 [00:21<00:23, 9146.56it/s] 46%|████▋     | 185679/400000 [00:21<00:22, 9347.81it/s] 47%|████▋     | 186618/400000 [00:21<00:22, 9359.45it/s] 47%|████▋     | 187556/400000 [00:21<00:22, 9337.78it/s] 47%|████▋     | 188513/400000 [00:21<00:22, 9404.40it/s] 47%|████▋     | 189455/400000 [00:21<00:22, 9279.60it/s] 48%|████▊     | 190384/400000 [00:21<00:22, 9183.86it/s] 48%|████▊     | 191304/400000 [00:21<00:22, 9164.03it/s] 48%|████▊     | 192258/400000 [00:22<00:22, 9272.90it/s] 48%|████▊     | 193236/400000 [00:22<00:21, 9416.90it/s] 49%|████▊     | 194179/400000 [00:22<00:22, 9320.06it/s] 49%|████▉     | 195112/400000 [00:22<00:22, 9156.70it/s] 49%|████▉     | 196029/400000 [00:22<00:22, 9136.80it/s] 49%|████▉     | 196944/400000 [00:22<00:22, 8994.44it/s] 49%|████▉     | 197845/400000 [00:22<00:23, 8512.78it/s] 50%|████▉     | 198703/400000 [00:22<00:23, 8517.81it/s] 50%|████▉     | 199593/400000 [00:22<00:23, 8628.41it/s] 50%|█████     | 200478/400000 [00:22<00:22, 8693.66it/s] 50%|█████     | 201350/400000 [00:23<00:23, 8530.38it/s] 51%|█████     | 202207/400000 [00:23<00:23, 8540.59it/s] 51%|█████     | 203109/400000 [00:23<00:22, 8676.65it/s] 51%|█████     | 203990/400000 [00:23<00:22, 8715.49it/s] 51%|█████     | 204863/400000 [00:23<00:22, 8711.71it/s] 51%|█████▏    | 205740/400000 [00:23<00:22, 8727.59it/s] 52%|█████▏    | 206635/400000 [00:23<00:21, 8790.96it/s] 52%|█████▏    | 207541/400000 [00:23<00:21, 8869.30it/s] 52%|█████▏    | 208467/400000 [00:23<00:21, 8981.28it/s] 52%|█████▏    | 209381/400000 [00:23<00:21, 9025.78it/s] 53%|█████▎    | 210326/400000 [00:24<00:20, 9148.63it/s] 53%|█████▎    | 211272/400000 [00:24<00:20, 9239.73it/s] 53%|█████▎    | 212197/400000 [00:24<00:20, 9214.06it/s] 53%|█████▎    | 213119/400000 [00:24<00:21, 8771.28it/s] 54%|█████▎    | 214042/400000 [00:24<00:20, 8903.22it/s] 54%|█████▎    | 214942/400000 [00:24<00:20, 8930.44it/s] 54%|█████▍    | 215841/400000 [00:24<00:20, 8947.77it/s] 54%|█████▍    | 216758/400000 [00:24<00:20, 9010.35it/s] 54%|█████▍    | 217674/400000 [00:24<00:20, 9053.10it/s] 55%|█████▍    | 218581/400000 [00:25<00:20, 8912.29it/s] 55%|█████▍    | 219474/400000 [00:25<00:20, 8846.40it/s] 55%|█████▌    | 220360/400000 [00:25<00:20, 8679.95it/s] 55%|█████▌    | 221253/400000 [00:25<00:20, 8753.02it/s] 56%|█████▌    | 222147/400000 [00:25<00:20, 8807.27it/s] 56%|█████▌    | 223029/400000 [00:25<00:20, 8791.22it/s] 56%|█████▌    | 223937/400000 [00:25<00:19, 8874.45it/s] 56%|█████▌    | 224826/400000 [00:25<00:19, 8847.78it/s] 56%|█████▋    | 225726/400000 [00:25<00:19, 8892.19it/s] 57%|█████▋    | 226616/400000 [00:25<00:20, 8653.19it/s] 57%|█████▋    | 227550/400000 [00:26<00:19, 8847.16it/s] 57%|█████▋    | 228469/400000 [00:26<00:19, 8944.92it/s] 57%|█████▋    | 229384/400000 [00:26<00:18, 9003.73it/s] 58%|█████▊    | 230286/400000 [00:26<00:18, 8967.86it/s] 58%|█████▊    | 231184/400000 [00:26<00:19, 8836.35it/s] 58%|█████▊    | 232069/400000 [00:26<00:18, 8839.11it/s] 58%|█████▊    | 232975/400000 [00:26<00:18, 8903.37it/s] 58%|█████▊    | 233867/400000 [00:26<00:18, 8833.18it/s] 59%|█████▊    | 234751/400000 [00:26<00:18, 8804.56it/s] 59%|█████▉    | 235647/400000 [00:26<00:18, 8848.80it/s] 59%|█████▉    | 236533/400000 [00:27<00:18, 8782.38it/s] 59%|█████▉    | 237415/400000 [00:27<00:18, 8790.73it/s] 60%|█████▉    | 238298/400000 [00:27<00:18, 8800.40it/s] 60%|█████▉    | 239179/400000 [00:27<00:18, 8754.15it/s] 60%|██████    | 240055/400000 [00:27<00:18, 8725.65it/s] 60%|██████    | 240928/400000 [00:27<00:18, 8674.52it/s] 60%|██████    | 241836/400000 [00:27<00:17, 8791.16it/s] 61%|██████    | 242732/400000 [00:27<00:17, 8840.65it/s] 61%|██████    | 243649/400000 [00:27<00:17, 8934.27it/s] 61%|██████    | 244582/400000 [00:27<00:17, 9046.73it/s] 61%|██████▏   | 245488/400000 [00:28<00:17, 9003.96it/s] 62%|██████▏   | 246389/400000 [00:28<00:17, 8959.05it/s] 62%|██████▏   | 247286/400000 [00:28<00:17, 8956.99it/s] 62%|██████▏   | 248200/400000 [00:28<00:16, 9010.34it/s] 62%|██████▏   | 249117/400000 [00:28<00:16, 9055.64it/s] 63%|██████▎   | 250077/400000 [00:28<00:16, 9212.11it/s] 63%|██████▎   | 251000/400000 [00:28<00:16, 9187.92it/s] 63%|██████▎   | 251920/400000 [00:28<00:16, 9139.77it/s] 63%|██████▎   | 252835/400000 [00:28<00:16, 9123.30it/s] 63%|██████▎   | 253769/400000 [00:28<00:15, 9184.80it/s] 64%|██████▎   | 254716/400000 [00:29<00:15, 9266.23it/s] 64%|██████▍   | 255673/400000 [00:29<00:15, 9355.12it/s] 64%|██████▍   | 256610/400000 [00:29<00:15, 9324.80it/s] 64%|██████▍   | 257543/400000 [00:29<00:15, 9317.78it/s] 65%|██████▍   | 258476/400000 [00:29<00:15, 9078.09it/s] 65%|██████▍   | 259386/400000 [00:29<00:16, 8663.24it/s] 65%|██████▌   | 260258/400000 [00:29<00:16, 8649.01it/s] 65%|██████▌   | 261139/400000 [00:29<00:15, 8694.46it/s] 66%|██████▌   | 262070/400000 [00:29<00:15, 8868.37it/s] 66%|██████▌   | 263004/400000 [00:29<00:15, 9003.98it/s] 66%|██████▌   | 263907/400000 [00:30<00:15, 9007.76it/s] 66%|██████▌   | 264820/400000 [00:30<00:14, 9043.61it/s] 66%|██████▋   | 265726/400000 [00:30<00:15, 8856.30it/s] 67%|██████▋   | 266625/400000 [00:30<00:14, 8894.59it/s] 67%|██████▋   | 267516/400000 [00:30<00:14, 8844.52it/s] 67%|██████▋   | 268451/400000 [00:30<00:14, 8988.57it/s] 67%|██████▋   | 269375/400000 [00:30<00:14, 9061.75it/s] 68%|██████▊   | 270285/400000 [00:30<00:14, 9072.76it/s] 68%|██████▊   | 271193/400000 [00:30<00:14, 9047.67it/s] 68%|██████▊   | 272099/400000 [00:30<00:14, 8856.24it/s] 68%|██████▊   | 272986/400000 [00:31<00:14, 8712.81it/s] 68%|██████▊   | 273859/400000 [00:31<00:14, 8631.09it/s] 69%|██████▊   | 274757/400000 [00:31<00:14, 8730.37it/s] 69%|██████▉   | 275694/400000 [00:31<00:13, 8911.77it/s] 69%|██████▉   | 276587/400000 [00:31<00:13, 8818.32it/s] 69%|██████▉   | 277513/400000 [00:31<00:13, 8945.59it/s] 70%|██████▉   | 278409/400000 [00:31<00:14, 8678.43it/s] 70%|██████▉   | 279302/400000 [00:31<00:13, 8750.28it/s] 70%|███████   | 280180/400000 [00:31<00:13, 8643.80it/s] 70%|███████   | 281097/400000 [00:32<00:13, 8792.83it/s] 70%|███████   | 281979/400000 [00:32<00:13, 8785.85it/s] 71%|███████   | 282871/400000 [00:32<00:13, 8824.79it/s] 71%|███████   | 283761/400000 [00:32<00:13, 8845.03it/s] 71%|███████   | 284668/400000 [00:32<00:12, 8909.24it/s] 71%|███████▏  | 285575/400000 [00:32<00:12, 8955.62it/s] 72%|███████▏  | 286473/400000 [00:32<00:12, 8960.54it/s] 72%|███████▏  | 287370/400000 [00:32<00:12, 8819.05it/s] 72%|███████▏  | 288288/400000 [00:32<00:12, 8922.34it/s] 72%|███████▏  | 289207/400000 [00:32<00:12, 8998.34it/s] 73%|███████▎  | 290108/400000 [00:33<00:12, 8965.29it/s] 73%|███████▎  | 291006/400000 [00:33<00:12, 8900.28it/s] 73%|███████▎  | 291928/400000 [00:33<00:12, 8992.68it/s] 73%|███████▎  | 292828/400000 [00:33<00:11, 8952.52it/s] 73%|███████▎  | 293730/400000 [00:33<00:11, 8972.45it/s] 74%|███████▎  | 294635/400000 [00:33<00:11, 8992.82it/s] 74%|███████▍  | 295535/400000 [00:33<00:11, 8874.85it/s] 74%|███████▍  | 296452/400000 [00:33<00:11, 8959.88it/s] 74%|███████▍  | 297365/400000 [00:33<00:11, 9008.82it/s] 75%|███████▍  | 298282/400000 [00:33<00:11, 9055.38it/s] 75%|███████▍  | 299188/400000 [00:34<00:11, 8984.24it/s] 75%|███████▌  | 300129/400000 [00:34<00:10, 9107.73it/s] 75%|███████▌  | 301059/400000 [00:34<00:10, 9162.76it/s] 75%|███████▌  | 301976/400000 [00:34<00:10, 9151.20it/s] 76%|███████▌  | 302905/400000 [00:34<00:10, 9192.06it/s] 76%|███████▌  | 303825/400000 [00:34<00:10, 9168.14it/s] 76%|███████▌  | 304743/400000 [00:34<00:10, 9117.50it/s] 76%|███████▋  | 305680/400000 [00:34<00:10, 9189.56it/s] 77%|███████▋  | 306620/400000 [00:34<00:10, 9249.74it/s] 77%|███████▋  | 307612/400000 [00:34<00:09, 9438.24it/s] 77%|███████▋  | 308636/400000 [00:35<00:09, 9664.89it/s] 77%|███████▋  | 309605/400000 [00:35<00:09, 9662.82it/s] 78%|███████▊  | 310606/400000 [00:35<00:09, 9762.23it/s] 78%|███████▊  | 311584/400000 [00:35<00:09, 9763.15it/s] 78%|███████▊  | 312562/400000 [00:35<00:08, 9737.45it/s] 78%|███████▊  | 313537/400000 [00:35<00:08, 9713.81it/s] 79%|███████▊  | 314516/400000 [00:35<00:08, 9733.45it/s] 79%|███████▉  | 315490/400000 [00:35<00:08, 9634.14it/s] 79%|███████▉  | 316454/400000 [00:35<00:08, 9491.26it/s] 79%|███████▉  | 317433/400000 [00:35<00:08, 9578.84it/s] 80%|███████▉  | 318392/400000 [00:36<00:08, 9297.56it/s] 80%|███████▉  | 319325/400000 [00:36<00:08, 9133.45it/s] 80%|████████  | 320241/400000 [00:36<00:08, 9051.23it/s] 80%|████████  | 321148/400000 [00:36<00:08, 8937.21it/s] 81%|████████  | 322044/400000 [00:36<00:08, 8806.37it/s] 81%|████████  | 322927/400000 [00:36<00:09, 8526.76it/s] 81%|████████  | 323783/400000 [00:36<00:08, 8490.50it/s] 81%|████████  | 324670/400000 [00:36<00:08, 8599.89it/s] 81%|████████▏ | 325545/400000 [00:36<00:08, 8643.91it/s] 82%|████████▏ | 326471/400000 [00:36<00:08, 8817.77it/s] 82%|████████▏ | 327374/400000 [00:37<00:08, 8877.98it/s] 82%|████████▏ | 328293/400000 [00:37<00:07, 8968.63it/s] 82%|████████▏ | 329198/400000 [00:37<00:07, 8992.33it/s] 83%|████████▎ | 330099/400000 [00:37<00:07, 8936.02it/s] 83%|████████▎ | 331000/400000 [00:37<00:07, 8957.28it/s] 83%|████████▎ | 331920/400000 [00:37<00:07, 9025.91it/s] 83%|████████▎ | 332855/400000 [00:37<00:07, 9119.98it/s] 83%|████████▎ | 333781/400000 [00:37<00:07, 9160.72it/s] 84%|████████▎ | 334698/400000 [00:37<00:07, 9091.98it/s] 84%|████████▍ | 335612/400000 [00:38<00:07, 9105.80it/s] 84%|████████▍ | 336554/400000 [00:38<00:06, 9196.17it/s] 84%|████████▍ | 337519/400000 [00:38<00:06, 9326.87it/s] 85%|████████▍ | 338476/400000 [00:38<00:06, 9397.21it/s] 85%|████████▍ | 339417/400000 [00:38<00:06, 9254.71it/s] 85%|████████▌ | 340344/400000 [00:38<00:06, 9219.40it/s] 85%|████████▌ | 341295/400000 [00:38<00:06, 9304.26it/s] 86%|████████▌ | 342227/400000 [00:38<00:06, 9179.85it/s] 86%|████████▌ | 343146/400000 [00:38<00:06, 9167.03it/s] 86%|████████▌ | 344064/400000 [00:38<00:06, 9059.89it/s] 86%|████████▌ | 344971/400000 [00:39<00:06, 8954.15it/s] 86%|████████▋ | 345868/400000 [00:39<00:06, 8953.79it/s] 87%|████████▋ | 346781/400000 [00:39<00:05, 9005.93it/s] 87%|████████▋ | 347708/400000 [00:39<00:05, 9082.79it/s] 87%|████████▋ | 348617/400000 [00:39<00:05, 9022.61it/s] 87%|████████▋ | 349520/400000 [00:39<00:05, 8967.40it/s] 88%|████████▊ | 350420/400000 [00:39<00:05, 8975.09it/s] 88%|████████▊ | 351318/400000 [00:39<00:05, 8887.35it/s] 88%|████████▊ | 352209/400000 [00:39<00:05, 8891.77it/s] 88%|████████▊ | 353099/400000 [00:39<00:05, 8841.81it/s] 88%|████████▊ | 353984/400000 [00:40<00:05, 8815.79it/s] 89%|████████▊ | 354892/400000 [00:40<00:05, 8891.94it/s] 89%|████████▉ | 355814/400000 [00:40<00:04, 8986.32it/s] 89%|████████▉ | 356781/400000 [00:40<00:04, 9178.64it/s] 89%|████████▉ | 357701/400000 [00:40<00:04, 9152.84it/s] 90%|████████▉ | 358618/400000 [00:40<00:04, 9029.89it/s] 90%|████████▉ | 359588/400000 [00:40<00:04, 9220.46it/s] 90%|█████████ | 360512/400000 [00:40<00:04, 9218.90it/s] 90%|█████████ | 361441/400000 [00:40<00:04, 9238.56it/s] 91%|█████████ | 362366/400000 [00:40<00:04, 9173.43it/s] 91%|█████████ | 363285/400000 [00:41<00:04, 9134.52it/s] 91%|█████████ | 364199/400000 [00:41<00:03, 9070.31it/s] 91%|█████████▏| 365122/400000 [00:41<00:03, 9115.92it/s] 92%|█████████▏| 366061/400000 [00:41<00:03, 9195.04it/s] 92%|█████████▏| 366981/400000 [00:41<00:03, 9019.94it/s] 92%|█████████▏| 367885/400000 [00:41<00:03, 8864.25it/s] 92%|█████████▏| 368802/400000 [00:41<00:03, 8951.48it/s] 92%|█████████▏| 369733/400000 [00:41<00:03, 9055.74it/s] 93%|█████████▎| 370656/400000 [00:41<00:03, 9105.49it/s] 93%|█████████▎| 371568/400000 [00:41<00:03, 9070.18it/s] 93%|█████████▎| 372476/400000 [00:42<00:03, 9040.49it/s] 93%|█████████▎| 373381/400000 [00:42<00:02, 9022.35it/s] 94%|█████████▎| 374284/400000 [00:42<00:02, 8978.86it/s] 94%|█████████▍| 375213/400000 [00:42<00:02, 9068.73it/s] 94%|█████████▍| 376142/400000 [00:42<00:02, 9132.22it/s] 94%|█████████▍| 377076/400000 [00:42<00:02, 9191.68it/s] 95%|█████████▍| 378035/400000 [00:42<00:02, 9305.24it/s] 95%|█████████▍| 378967/400000 [00:42<00:02, 9194.34it/s] 95%|█████████▍| 379888/400000 [00:42<00:02, 9138.56it/s] 95%|█████████▌| 380803/400000 [00:42<00:02, 9116.77it/s] 95%|█████████▌| 381716/400000 [00:43<00:02, 9102.19it/s] 96%|█████████▌| 382627/400000 [00:43<00:01, 9012.25it/s] 96%|█████████▌| 383529/400000 [00:43<00:01, 8995.92it/s] 96%|█████████▌| 384444/400000 [00:43<00:01, 9040.62it/s] 96%|█████████▋| 385349/400000 [00:43<00:01, 9026.37it/s] 97%|█████████▋| 386252/400000 [00:43<00:01, 9016.60it/s] 97%|█████████▋| 387158/400000 [00:43<00:01, 9028.81it/s] 97%|█████████▋| 388061/400000 [00:43<00:01, 8978.68it/s] 97%|█████████▋| 388986/400000 [00:43<00:01, 9054.05it/s] 97%|█████████▋| 389892/400000 [00:43<00:01, 9034.01it/s] 98%|█████████▊| 390796/400000 [00:44<00:01, 8976.06it/s] 98%|█████████▊| 391694/400000 [00:44<00:00, 8943.18it/s] 98%|█████████▊| 392621/400000 [00:44<00:00, 9038.29it/s] 98%|█████████▊| 393543/400000 [00:44<00:00, 9090.05it/s] 99%|█████████▊| 394456/400000 [00:44<00:00, 9099.54it/s] 99%|█████████▉| 395367/400000 [00:44<00:00, 9098.95it/s] 99%|█████████▉| 396278/400000 [00:44<00:00, 9008.29it/s] 99%|█████████▉| 397180/400000 [00:44<00:00, 8922.16it/s]100%|█████████▉| 398073/400000 [00:44<00:00, 8621.96it/s]100%|█████████▉| 398945/400000 [00:45<00:00, 8648.93it/s]100%|█████████▉| 399828/400000 [00:45<00:00, 8700.25it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8864.15it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f170e588a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011672838904144767 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011780181656713072 	 Accuracy: 49

  model saves at 49% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15982 out of table with 15975 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} 'model_pars' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text/ 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 

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
RuntimeError: index out of range: Tried to access index 15982 out of table with 15975 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

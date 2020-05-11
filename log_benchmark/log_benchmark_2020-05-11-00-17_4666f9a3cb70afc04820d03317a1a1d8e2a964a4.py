
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5a51701470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 00:18:07.359990
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 00:18:07.366501
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 00:18:07.370633
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 00:18:07.374678
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5a49a51400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355487.9688
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 275668.1562
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 181359.4844
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 107180.6797
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 59285.0156
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 33821.7617
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 20495.9746
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 13275.1689
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 9091.0723
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 6579.3931

  #### Inference Need return ypred, ytrue ######################### 
[[-6.80919826e-01 -1.11354366e-01  1.10381722e+00  1.23417568e+00
   9.88587499e-01 -4.74627942e-01  9.57222462e-01  3.89987797e-01
   1.02787709e+00 -1.74176025e+00  1.48344254e+00  8.48319530e-01
   5.62446535e-01  1.35255027e+00  4.18193638e-01  4.50993538e-01
   1.35703540e+00 -2.01164693e-01 -4.02305126e-02  2.34198928e-01
   2.12228864e-01 -1.74249351e-01 -1.51014721e+00  1.24269366e+00
   3.85580897e-01  6.93365812e-01 -7.39357948e-01  8.06622148e-01
   7.25404739e-01  1.50226963e+00 -2.33900279e-01 -4.36083198e-01
  -8.38685811e-01  6.59473062e-01  5.63006878e-01 -1.44548565e-02
  -3.23850393e-01  2.19786167e+00 -2.51253635e-01  6.44077539e-01
  -1.03526783e+00  4.58427519e-01 -4.38986301e-01 -4.02578264e-01
  -8.24878097e-01  1.02970409e+00  1.93440661e-01  8.63075018e-01
  -5.33373713e-01  5.84952474e-01 -5.63003838e-01  9.44301188e-01
  -7.42540300e-01 -3.56537074e-01  2.40751028e-01  6.98393643e-01
   1.11313438e+00  5.06526947e-01  6.80765152e-01 -1.56917655e+00
   4.66911703e-01  8.36091328e+00  6.66021729e+00  7.89531946e+00
   7.13581181e+00  8.81692886e+00  7.79755783e+00  8.66382313e+00
   8.48723412e+00  8.83378315e+00  9.45732689e+00  7.84090710e+00
   7.30162382e+00  8.84890938e+00  9.15854263e+00  8.53202343e+00
   6.73070621e+00  9.51478577e+00  8.46081638e+00  7.68635321e+00
   7.83810139e+00  9.61499310e+00  8.26263046e+00  7.58403730e+00
   6.69376183e+00  9.77260780e+00  7.49324036e+00  8.16685963e+00
   7.43722010e+00  9.41259003e+00  8.52779102e+00  7.36089468e+00
   8.37587738e+00  8.98464298e+00  7.82968330e+00  8.45491028e+00
   9.66255760e+00  7.91722107e+00  6.87000322e+00  9.20654678e+00
   9.61510468e+00  9.56485939e+00  8.64443111e+00  7.35612965e+00
   7.43413639e+00  9.33863640e+00  8.45663261e+00  9.25117016e+00
   8.82134819e+00  9.04966354e+00  6.82064533e+00  9.36307335e+00
   9.10725880e+00  7.44430971e+00  7.82361650e+00  9.07295513e+00
   5.99066734e+00  9.22491169e+00  8.63494492e+00  9.02247143e+00
  -1.09647560e+00  2.79526889e-01  1.70760572e-01  9.02948752e-02
   3.56899500e-02  1.57560304e-01 -5.22707105e-01 -8.44589889e-01
  -4.92294729e-01 -5.78879476e-01 -5.21185994e-01 -3.37675095e-01
   6.42765045e-01 -5.05754948e-02  2.25785232e+00 -4.91109416e-02
   6.35985136e-01  5.21846890e-01 -9.15432215e-01  4.08697397e-01
  -8.72678995e-01 -1.98306888e-01 -8.21756363e-01 -2.91318208e-01
  -2.28834763e-01 -1.65277556e-01 -2.13310194e+00 -8.67912292e-01
   3.90052527e-01  3.07812959e-01  5.41060686e-01 -9.47832227e-01
   4.59013283e-01  6.48397982e-01  8.73412285e-03  7.43721187e-01
  -8.12201023e-01  1.66378164e+00 -2.25167781e-01 -2.08538622e-01
   2.17190176e-01  6.01128161e-01  1.29970646e+00  1.69074643e+00
  -1.25746274e+00 -3.54762971e-01  2.02565566e-01 -2.84398794e+00
   5.59772730e-01  6.61238909e-01 -3.23761433e-01  3.65405172e-01
  -7.91107476e-01 -1.63115430e+00 -1.20769334e+00  4.82057810e-01
   4.15444881e-01 -1.53106046e+00 -4.36602116e-01  1.45841390e-01
   6.93693459e-01  1.71402633e-01  1.30302370e+00  6.14096582e-01
   1.76353836e+00  2.61381578e+00  2.08943844e+00  4.06020701e-01
   3.40115070e-01  1.99348938e+00  1.36224151e-01  1.19879663e-01
   1.75707459e-01  2.21342659e+00  7.28371024e-01  1.86448956e+00
   9.39054191e-01  2.35125518e+00  8.92152786e-01  1.31562424e+00
   1.11490309e-01  1.25369906e+00  1.05843353e+00  2.63280201e+00
   3.11836338e+00  2.10894823e+00  2.68842411e+00  1.15942276e+00
   3.23614836e-01  2.61566877e+00  6.65221930e-01  1.95576549e-01
   8.41525793e-01  9.11523223e-01  2.66726041e+00  2.27349639e-01
   2.19766736e+00  2.03933907e+00  2.14377069e+00  4.73815739e-01
   7.66380370e-01  2.56111026e-01  6.69429839e-01  1.98526442e+00
   1.31031334e+00  7.98144341e-01  2.54108548e-01  7.08222628e-01
   9.72384453e-01  1.52560604e+00  2.55819058e+00  2.19956040e-01
   8.49433661e-01  2.19155431e-01  2.98331857e-01  4.34246182e-01
   2.20391607e+00  3.72370780e-01  1.54399455e-01  4.57741678e-01
   1.63071275e-01  8.15787506e+00  8.90951157e+00  9.59137058e+00
   9.42039204e+00  7.86460781e+00  7.81641245e+00  9.07986736e+00
   9.26667500e+00  8.56703472e+00  7.94950914e+00  1.02457275e+01
   8.01349068e+00  9.49316025e+00  8.81071186e+00  8.30228615e+00
   6.73250294e+00  9.04760265e+00  8.56725311e+00  8.30664253e+00
   9.09789276e+00  9.15226364e+00  8.79981899e+00  8.93192577e+00
   7.32267332e+00  7.48566818e+00  8.39774132e+00  8.93449020e+00
   1.02914562e+01  8.97068691e+00  8.32233715e+00  9.58828354e+00
   7.90535545e+00  9.81525326e+00  8.07801437e+00  7.56191874e+00
   7.47259378e+00  9.76490879e+00  1.03079433e+01  9.61232281e+00
   8.47899437e+00  8.55565739e+00  8.78272629e+00  9.29232502e+00
   8.47739506e+00  7.49506664e+00  8.37336540e+00  8.88255978e+00
   8.95056725e+00  8.02817917e+00  6.97498608e+00  7.66087389e+00
   7.93843031e+00  1.04272985e+01  9.88993168e+00  8.07010365e+00
   8.03898144e+00  9.81456757e+00  8.97860527e+00  9.18405247e+00
   2.05375671e+00  3.82805169e-01  1.34823704e+00  1.34938073e+00
   6.38147652e-01  9.76076663e-01  3.80772114e-01  1.09863830e+00
   7.82708466e-01  3.81328523e-01  5.27565598e-01  1.17923367e+00
   3.60440373e-01  1.00751233e+00  1.74717319e+00  1.02580416e+00
   1.78263688e+00  6.64251149e-01  1.44388032e+00  5.64341247e-01
   9.18177903e-01  1.82446194e+00  1.24995756e+00  1.71212316e-01
   1.82896137e-01  2.75950813e+00  1.32873225e+00  5.60653269e-01
   1.77295220e+00  1.71479774e+00  1.57759190e+00  1.63771749e-01
   5.04930973e-01  1.29376197e+00  2.23916626e+00  6.68973029e-01
   4.80104983e-01  1.10123634e+00  2.11363912e-01  1.80445170e+00
   2.04930997e+00  9.70369458e-01  9.39296842e-01  2.97354245e+00
   3.76644552e-01  4.90172625e-01  1.69899106e-01  1.82790041e+00
   1.36028028e+00  1.30971813e+00  2.32032597e-01  2.03399539e-01
   3.86126041e-01  8.83108795e-01  8.86286199e-01  1.97209358e-01
   6.37907088e-01  4.39243376e-01  1.93203163e+00  1.17355776e+00
  -1.11451387e+01  3.33780360e+00 -1.29510374e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 00:18:15.555677
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2289
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 00:18:15.559742
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8713.01
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 00:18:15.563642
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4182
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 00:18:15.567319
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -779.312
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140025218362784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140024394009792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140024394010296
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140024393597168
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140024393597672
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140024393598176

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5a3d98c400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.515347
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.487690
grad_step = 000002, loss = 0.465064
grad_step = 000003, loss = 0.440415
grad_step = 000004, loss = 0.415281
grad_step = 000005, loss = 0.391089
grad_step = 000006, loss = 0.376165
grad_step = 000007, loss = 0.362997
grad_step = 000008, loss = 0.346850
grad_step = 000009, loss = 0.326226
grad_step = 000010, loss = 0.310804
grad_step = 000011, loss = 0.298773
grad_step = 000012, loss = 0.288377
grad_step = 000013, loss = 0.278908
grad_step = 000014, loss = 0.267884
grad_step = 000015, loss = 0.254912
grad_step = 000016, loss = 0.242289
grad_step = 000017, loss = 0.230819
grad_step = 000018, loss = 0.220069
grad_step = 000019, loss = 0.209910
grad_step = 000020, loss = 0.199244
grad_step = 000021, loss = 0.189068
grad_step = 000022, loss = 0.179940
grad_step = 000023, loss = 0.170536
grad_step = 000024, loss = 0.161355
grad_step = 000025, loss = 0.152746
grad_step = 000026, loss = 0.144223
grad_step = 000027, loss = 0.135440
grad_step = 000028, loss = 0.126529
grad_step = 000029, loss = 0.118213
grad_step = 000030, loss = 0.110827
grad_step = 000031, loss = 0.103379
grad_step = 000032, loss = 0.096009
grad_step = 000033, loss = 0.089214
grad_step = 000034, loss = 0.083006
grad_step = 000035, loss = 0.077018
grad_step = 000036, loss = 0.071170
grad_step = 000037, loss = 0.065597
grad_step = 000038, loss = 0.060173
grad_step = 000039, loss = 0.055194
grad_step = 000040, loss = 0.050668
grad_step = 000041, loss = 0.046374
grad_step = 000042, loss = 0.042231
grad_step = 000043, loss = 0.038611
grad_step = 000044, loss = 0.035327
grad_step = 000045, loss = 0.032146
grad_step = 000046, loss = 0.029172
grad_step = 000047, loss = 0.026549
grad_step = 000048, loss = 0.024129
grad_step = 000049, loss = 0.021906
grad_step = 000050, loss = 0.019846
grad_step = 000051, loss = 0.017952
grad_step = 000052, loss = 0.016304
grad_step = 000053, loss = 0.014856
grad_step = 000054, loss = 0.013458
grad_step = 000055, loss = 0.012242
grad_step = 000056, loss = 0.011184
grad_step = 000057, loss = 0.010189
grad_step = 000058, loss = 0.009277
grad_step = 000059, loss = 0.008473
grad_step = 000060, loss = 0.007769
grad_step = 000061, loss = 0.007133
grad_step = 000062, loss = 0.006541
grad_step = 000063, loss = 0.006019
grad_step = 000064, loss = 0.005578
grad_step = 000065, loss = 0.005167
grad_step = 000066, loss = 0.004788
grad_step = 000067, loss = 0.004458
grad_step = 000068, loss = 0.004169
grad_step = 000069, loss = 0.003902
grad_step = 000070, loss = 0.003664
grad_step = 000071, loss = 0.003464
grad_step = 000072, loss = 0.003285
grad_step = 000073, loss = 0.003127
grad_step = 000074, loss = 0.002990
grad_step = 000075, loss = 0.002881
grad_step = 000076, loss = 0.002782
grad_step = 000077, loss = 0.002693
grad_step = 000078, loss = 0.002626
grad_step = 000079, loss = 0.002569
grad_step = 000080, loss = 0.002521
grad_step = 000081, loss = 0.002479
grad_step = 000082, loss = 0.002448
grad_step = 000083, loss = 0.002424
grad_step = 000084, loss = 0.002400
grad_step = 000085, loss = 0.002384
grad_step = 000086, loss = 0.002370
grad_step = 000087, loss = 0.002357
grad_step = 000088, loss = 0.002345
grad_step = 000089, loss = 0.002335
grad_step = 000090, loss = 0.002326
grad_step = 000091, loss = 0.002314
grad_step = 000092, loss = 0.002306
grad_step = 000093, loss = 0.002296
grad_step = 000094, loss = 0.002287
grad_step = 000095, loss = 0.002277
grad_step = 000096, loss = 0.002268
grad_step = 000097, loss = 0.002257
grad_step = 000098, loss = 0.002247
grad_step = 000099, loss = 0.002237
grad_step = 000100, loss = 0.002227
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002218
grad_step = 000102, loss = 0.002209
grad_step = 000103, loss = 0.002200
grad_step = 000104, loss = 0.002191
grad_step = 000105, loss = 0.002183
grad_step = 000106, loss = 0.002175
grad_step = 000107, loss = 0.002167
grad_step = 000108, loss = 0.002159
grad_step = 000109, loss = 0.002152
grad_step = 000110, loss = 0.002145
grad_step = 000111, loss = 0.002138
grad_step = 000112, loss = 0.002131
grad_step = 000113, loss = 0.002125
grad_step = 000114, loss = 0.002118
grad_step = 000115, loss = 0.002112
grad_step = 000116, loss = 0.002105
grad_step = 000117, loss = 0.002099
grad_step = 000118, loss = 0.002093
grad_step = 000119, loss = 0.002087
grad_step = 000120, loss = 0.002081
grad_step = 000121, loss = 0.002075
grad_step = 000122, loss = 0.002070
grad_step = 000123, loss = 0.002064
grad_step = 000124, loss = 0.002059
grad_step = 000125, loss = 0.002055
grad_step = 000126, loss = 0.002058
grad_step = 000127, loss = 0.002079
grad_step = 000128, loss = 0.002143
grad_step = 000129, loss = 0.002267
grad_step = 000130, loss = 0.002316
grad_step = 000131, loss = 0.002224
grad_step = 000132, loss = 0.002042
grad_step = 000133, loss = 0.002045
grad_step = 000134, loss = 0.002172
grad_step = 000135, loss = 0.002158
grad_step = 000136, loss = 0.002034
grad_step = 000137, loss = 0.002009
grad_step = 000138, loss = 0.002093
grad_step = 000139, loss = 0.002107
grad_step = 000140, loss = 0.002011
grad_step = 000141, loss = 0.001989
grad_step = 000142, loss = 0.002051
grad_step = 000143, loss = 0.002054
grad_step = 000144, loss = 0.001988
grad_step = 000145, loss = 0.001967
grad_step = 000146, loss = 0.002007
grad_step = 000147, loss = 0.002020
grad_step = 000148, loss = 0.001972
grad_step = 000149, loss = 0.001946
grad_step = 000150, loss = 0.001967
grad_step = 000151, loss = 0.001984
grad_step = 000152, loss = 0.001961
grad_step = 000153, loss = 0.001930
grad_step = 000154, loss = 0.001929
grad_step = 000155, loss = 0.001945
grad_step = 000156, loss = 0.001947
grad_step = 000157, loss = 0.001928
grad_step = 000158, loss = 0.001907
grad_step = 000159, loss = 0.001901
grad_step = 000160, loss = 0.001908
grad_step = 000161, loss = 0.001914
grad_step = 000162, loss = 0.001910
grad_step = 000163, loss = 0.001898
grad_step = 000164, loss = 0.001883
grad_step = 000165, loss = 0.001872
grad_step = 000166, loss = 0.001867
grad_step = 000167, loss = 0.001867
grad_step = 000168, loss = 0.001869
grad_step = 000169, loss = 0.001874
grad_step = 000170, loss = 0.001881
grad_step = 000171, loss = 0.001896
grad_step = 000172, loss = 0.001915
grad_step = 000173, loss = 0.001952
grad_step = 000174, loss = 0.001987
grad_step = 000175, loss = 0.002022
grad_step = 000176, loss = 0.001996
grad_step = 000177, loss = 0.001929
grad_step = 000178, loss = 0.001847
grad_step = 000179, loss = 0.001810
grad_step = 000180, loss = 0.001831
grad_step = 000181, loss = 0.001878
grad_step = 000182, loss = 0.001919
grad_step = 000183, loss = 0.001915
grad_step = 000184, loss = 0.001878
grad_step = 000185, loss = 0.001820
grad_step = 000186, loss = 0.001786
grad_step = 000187, loss = 0.001785
grad_step = 000188, loss = 0.001810
grad_step = 000189, loss = 0.001849
grad_step = 000190, loss = 0.001870
grad_step = 000191, loss = 0.001872
grad_step = 000192, loss = 0.001838
grad_step = 000193, loss = 0.001795
grad_step = 000194, loss = 0.001763
grad_step = 000195, loss = 0.001755
grad_step = 000196, loss = 0.001768
grad_step = 000197, loss = 0.001790
grad_step = 000198, loss = 0.001808
grad_step = 000199, loss = 0.001811
grad_step = 000200, loss = 0.001801
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001776
grad_step = 000202, loss = 0.001753
grad_step = 000203, loss = 0.001736
grad_step = 000204, loss = 0.001727
grad_step = 000205, loss = 0.001727
grad_step = 000206, loss = 0.001733
grad_step = 000207, loss = 0.001745
grad_step = 000208, loss = 0.001766
grad_step = 000209, loss = 0.001803
grad_step = 000210, loss = 0.001836
grad_step = 000211, loss = 0.001878
grad_step = 000212, loss = 0.001850
grad_step = 000213, loss = 0.001806
grad_step = 000214, loss = 0.001761
grad_step = 000215, loss = 0.001748
grad_step = 000216, loss = 0.001756
grad_step = 000217, loss = 0.001729
grad_step = 000218, loss = 0.001714
grad_step = 000219, loss = 0.001733
grad_step = 000220, loss = 0.001742
grad_step = 000221, loss = 0.001723
grad_step = 000222, loss = 0.001689
grad_step = 000223, loss = 0.001693
grad_step = 000224, loss = 0.001716
grad_step = 000225, loss = 0.001708
grad_step = 000226, loss = 0.001689
grad_step = 000227, loss = 0.001684
grad_step = 000228, loss = 0.001698
grad_step = 000229, loss = 0.001709
grad_step = 000230, loss = 0.001699
grad_step = 000231, loss = 0.001687
grad_step = 000232, loss = 0.001687
grad_step = 000233, loss = 0.001699
grad_step = 000234, loss = 0.001714
grad_step = 000235, loss = 0.001719
grad_step = 000236, loss = 0.001725
grad_step = 000237, loss = 0.001743
grad_step = 000238, loss = 0.001769
grad_step = 000239, loss = 0.001810
grad_step = 000240, loss = 0.001801
grad_step = 000241, loss = 0.001776
grad_step = 000242, loss = 0.001720
grad_step = 000243, loss = 0.001685
grad_step = 000244, loss = 0.001680
grad_step = 000245, loss = 0.001680
grad_step = 000246, loss = 0.001677
grad_step = 000247, loss = 0.001674
grad_step = 000248, loss = 0.001684
grad_step = 000249, loss = 0.001700
grad_step = 000250, loss = 0.001691
grad_step = 000251, loss = 0.001668
grad_step = 000252, loss = 0.001642
grad_step = 000253, loss = 0.001639
grad_step = 000254, loss = 0.001652
grad_step = 000255, loss = 0.001659
grad_step = 000256, loss = 0.001653
grad_step = 000257, loss = 0.001643
grad_step = 000258, loss = 0.001641
grad_step = 000259, loss = 0.001646
grad_step = 000260, loss = 0.001645
grad_step = 000261, loss = 0.001637
grad_step = 000262, loss = 0.001625
grad_step = 000263, loss = 0.001619
grad_step = 000264, loss = 0.001620
grad_step = 000265, loss = 0.001623
grad_step = 000266, loss = 0.001624
grad_step = 000267, loss = 0.001620
grad_step = 000268, loss = 0.001615
grad_step = 000269, loss = 0.001611
grad_step = 000270, loss = 0.001611
grad_step = 000271, loss = 0.001613
grad_step = 000272, loss = 0.001617
grad_step = 000273, loss = 0.001622
grad_step = 000274, loss = 0.001627
grad_step = 000275, loss = 0.001637
grad_step = 000276, loss = 0.001650
grad_step = 000277, loss = 0.001674
grad_step = 000278, loss = 0.001712
grad_step = 000279, loss = 0.001764
grad_step = 000280, loss = 0.001824
grad_step = 000281, loss = 0.001850
grad_step = 000282, loss = 0.001828
grad_step = 000283, loss = 0.001741
grad_step = 000284, loss = 0.001640
grad_step = 000285, loss = 0.001592
grad_step = 000286, loss = 0.001602
grad_step = 000287, loss = 0.001637
grad_step = 000288, loss = 0.001644
grad_step = 000289, loss = 0.001627
grad_step = 000290, loss = 0.001608
grad_step = 000291, loss = 0.001597
grad_step = 000292, loss = 0.001595
grad_step = 000293, loss = 0.001581
grad_step = 000294, loss = 0.001569
grad_step = 000295, loss = 0.001561
grad_step = 000296, loss = 0.001563
grad_step = 000297, loss = 0.001569
grad_step = 000298, loss = 0.001564
grad_step = 000299, loss = 0.001547
grad_step = 000300, loss = 0.001523
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001511
grad_step = 000302, loss = 0.001515
grad_step = 000303, loss = 0.001524
grad_step = 000304, loss = 0.001528
grad_step = 000305, loss = 0.001519
grad_step = 000306, loss = 0.001508
grad_step = 000307, loss = 0.001498
grad_step = 000308, loss = 0.001491
grad_step = 000309, loss = 0.001486
grad_step = 000310, loss = 0.001477
grad_step = 000311, loss = 0.001469
grad_step = 000312, loss = 0.001462
grad_step = 000313, loss = 0.001464
grad_step = 000314, loss = 0.001478
grad_step = 000315, loss = 0.001517
grad_step = 000316, loss = 0.001605
grad_step = 000317, loss = 0.001664
grad_step = 000318, loss = 0.001746
grad_step = 000319, loss = 0.001616
grad_step = 000320, loss = 0.001497
grad_step = 000321, loss = 0.001450
grad_step = 000322, loss = 0.001485
grad_step = 000323, loss = 0.001519
grad_step = 000324, loss = 0.001463
grad_step = 000325, loss = 0.001438
grad_step = 000326, loss = 0.001458
grad_step = 000327, loss = 0.001453
grad_step = 000328, loss = 0.001429
grad_step = 000329, loss = 0.001403
grad_step = 000330, loss = 0.001411
grad_step = 000331, loss = 0.001424
grad_step = 000332, loss = 0.001404
grad_step = 000333, loss = 0.001381
grad_step = 000334, loss = 0.001381
grad_step = 000335, loss = 0.001404
grad_step = 000336, loss = 0.001416
grad_step = 000337, loss = 0.001398
grad_step = 000338, loss = 0.001369
grad_step = 000339, loss = 0.001358
grad_step = 000340, loss = 0.001360
grad_step = 000341, loss = 0.001361
grad_step = 000342, loss = 0.001351
grad_step = 000343, loss = 0.001342
grad_step = 000344, loss = 0.001344
grad_step = 000345, loss = 0.001353
grad_step = 000346, loss = 0.001359
grad_step = 000347, loss = 0.001351
grad_step = 000348, loss = 0.001343
grad_step = 000349, loss = 0.001338
grad_step = 000350, loss = 0.001338
grad_step = 000351, loss = 0.001336
grad_step = 000352, loss = 0.001329
grad_step = 000353, loss = 0.001321
grad_step = 000354, loss = 0.001316
grad_step = 000355, loss = 0.001316
grad_step = 000356, loss = 0.001317
grad_step = 000357, loss = 0.001316
grad_step = 000358, loss = 0.001313
grad_step = 000359, loss = 0.001310
grad_step = 000360, loss = 0.001310
grad_step = 000361, loss = 0.001313
grad_step = 000362, loss = 0.001318
grad_step = 000363, loss = 0.001331
grad_step = 000364, loss = 0.001348
grad_step = 000365, loss = 0.001393
grad_step = 000366, loss = 0.001424
grad_step = 000367, loss = 0.001488
grad_step = 000368, loss = 0.001452
grad_step = 000369, loss = 0.001400
grad_step = 000370, loss = 0.001337
grad_step = 000371, loss = 0.001320
grad_step = 000372, loss = 0.001339
grad_step = 000373, loss = 0.001339
grad_step = 000374, loss = 0.001323
grad_step = 000375, loss = 0.001309
grad_step = 000376, loss = 0.001318
grad_step = 000377, loss = 0.001334
grad_step = 000378, loss = 0.001322
grad_step = 000379, loss = 0.001297
grad_step = 000380, loss = 0.001276
grad_step = 000381, loss = 0.001278
grad_step = 000382, loss = 0.001295
grad_step = 000383, loss = 0.001304
grad_step = 000384, loss = 0.001299
grad_step = 000385, loss = 0.001279
grad_step = 000386, loss = 0.001264
grad_step = 000387, loss = 0.001261
grad_step = 000388, loss = 0.001268
grad_step = 000389, loss = 0.001276
grad_step = 000390, loss = 0.001274
grad_step = 000391, loss = 0.001266
grad_step = 000392, loss = 0.001255
grad_step = 000393, loss = 0.001249
grad_step = 000394, loss = 0.001249
grad_step = 000395, loss = 0.001252
grad_step = 000396, loss = 0.001255
grad_step = 000397, loss = 0.001253
grad_step = 000398, loss = 0.001248
grad_step = 000399, loss = 0.001241
grad_step = 000400, loss = 0.001237
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001234
grad_step = 000402, loss = 0.001234
grad_step = 000403, loss = 0.001235
grad_step = 000404, loss = 0.001235
grad_step = 000405, loss = 0.001234
grad_step = 000406, loss = 0.001231
grad_step = 000407, loss = 0.001227
grad_step = 000408, loss = 0.001223
grad_step = 000409, loss = 0.001220
grad_step = 000410, loss = 0.001218
grad_step = 000411, loss = 0.001216
grad_step = 000412, loss = 0.001215
grad_step = 000413, loss = 0.001214
grad_step = 000414, loss = 0.001214
grad_step = 000415, loss = 0.001214
grad_step = 000416, loss = 0.001217
grad_step = 000417, loss = 0.001226
grad_step = 000418, loss = 0.001247
grad_step = 000419, loss = 0.001304
grad_step = 000420, loss = 0.001392
grad_step = 000421, loss = 0.001589
grad_step = 000422, loss = 0.001645
grad_step = 000423, loss = 0.001688
grad_step = 000424, loss = 0.001417
grad_step = 000425, loss = 0.001266
grad_step = 000426, loss = 0.001364
grad_step = 000427, loss = 0.001461
grad_step = 000428, loss = 0.001395
grad_step = 000429, loss = 0.001224
grad_step = 000430, loss = 0.001250
grad_step = 000431, loss = 0.001350
grad_step = 000432, loss = 0.001284
grad_step = 000433, loss = 0.001202
grad_step = 000434, loss = 0.001233
grad_step = 000435, loss = 0.001297
grad_step = 000436, loss = 0.001279
grad_step = 000437, loss = 0.001199
grad_step = 000438, loss = 0.001187
grad_step = 000439, loss = 0.001233
grad_step = 000440, loss = 0.001229
grad_step = 000441, loss = 0.001190
grad_step = 000442, loss = 0.001173
grad_step = 000443, loss = 0.001198
grad_step = 000444, loss = 0.001215
grad_step = 000445, loss = 0.001187
grad_step = 000446, loss = 0.001159
grad_step = 000447, loss = 0.001163
grad_step = 000448, loss = 0.001180
grad_step = 000449, loss = 0.001178
grad_step = 000450, loss = 0.001158
grad_step = 000451, loss = 0.001149
grad_step = 000452, loss = 0.001157
grad_step = 000453, loss = 0.001165
grad_step = 000454, loss = 0.001160
grad_step = 000455, loss = 0.001145
grad_step = 000456, loss = 0.001137
grad_step = 000457, loss = 0.001140
grad_step = 000458, loss = 0.001145
grad_step = 000459, loss = 0.001144
grad_step = 000460, loss = 0.001135
grad_step = 000461, loss = 0.001128
grad_step = 000462, loss = 0.001126
grad_step = 000463, loss = 0.001128
grad_step = 000464, loss = 0.001130
grad_step = 000465, loss = 0.001127
grad_step = 000466, loss = 0.001122
grad_step = 000467, loss = 0.001117
grad_step = 000468, loss = 0.001115
grad_step = 000469, loss = 0.001116
grad_step = 000470, loss = 0.001116
grad_step = 000471, loss = 0.001114
grad_step = 000472, loss = 0.001111
grad_step = 000473, loss = 0.001107
grad_step = 000474, loss = 0.001105
grad_step = 000475, loss = 0.001104
grad_step = 000476, loss = 0.001104
grad_step = 000477, loss = 0.001105
grad_step = 000478, loss = 0.001107
grad_step = 000479, loss = 0.001109
grad_step = 000480, loss = 0.001114
grad_step = 000481, loss = 0.001120
grad_step = 000482, loss = 0.001133
grad_step = 000483, loss = 0.001146
grad_step = 000484, loss = 0.001170
grad_step = 000485, loss = 0.001183
grad_step = 000486, loss = 0.001204
grad_step = 000487, loss = 0.001196
grad_step = 000488, loss = 0.001183
grad_step = 000489, loss = 0.001147
grad_step = 000490, loss = 0.001115
grad_step = 000491, loss = 0.001096
grad_step = 000492, loss = 0.001093
grad_step = 000493, loss = 0.001100
grad_step = 000494, loss = 0.001104
grad_step = 000495, loss = 0.001102
grad_step = 000496, loss = 0.001087
grad_step = 000497, loss = 0.001073
grad_step = 000498, loss = 0.001062
grad_step = 000499, loss = 0.001060
grad_step = 000500, loss = 0.001066
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001074
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

  date_run                              2020-05-11 00:18:38.046180
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.246888
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 00:18:38.056562
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.187234
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 00:18:38.064885
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.123444
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 00:18:38.070901
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.84508
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
0   2020-05-11 00:18:07.359990  ...    mean_absolute_error
1   2020-05-11 00:18:07.366501  ...     mean_squared_error
2   2020-05-11 00:18:07.370633  ...  median_absolute_error
3   2020-05-11 00:18:07.374678  ...               r2_score
4   2020-05-11 00:18:15.555677  ...    mean_absolute_error
5   2020-05-11 00:18:15.559742  ...     mean_squared_error
6   2020-05-11 00:18:15.563642  ...  median_absolute_error
7   2020-05-11 00:18:15.567319  ...               r2_score
8   2020-05-11 00:18:38.046180  ...    mean_absolute_error
9   2020-05-11 00:18:38.056562  ...     mean_squared_error
10  2020-05-11 00:18:38.064885  ...  median_absolute_error
11  2020-05-11 00:18:38.070901  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:23, 118589.81it/s] 73%|███████▎  | 7217152/9912422 [00:00<00:15, 169291.76it/s]9920512it [00:00, 37187758.38it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578982.87it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483433.85it/s]1654784it [00:00, 11922431.17it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196176.57it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a3cda208> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc54033f9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a2bade48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc54033fda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a3cda208> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5555a6cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a3cda208> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc549a5b518> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a3cda208> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5555a6cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc5a3cda208> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0c43c2e1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=446a0743b281bbd689c07c67a5121d0d5d65a1f2823eaac3ad9f652733b51d22
  Stored in directory: /tmp/pip-ephem-wheel-cache-vcsmktu1/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0bdc90ad30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   57344/17464789 [..............................] - ETA: 20s
  212992/17464789 [..............................] - ETA: 9s 
  892928/17464789 [>.............................] - ETA: 3s
 3661824/17464789 [=====>........................] - ETA: 0s
 7929856/17464789 [============>.................] - ETA: 0s
10690560/17464789 [=================>............] - ETA: 0s
15736832/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 00:20:03.787622: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 00:20:03.792859: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 00:20:03.793068: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56226aeef7f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 00:20:03.793087: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2680 - accuracy: 0.5260
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7701 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8230 - accuracy: 0.4898
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7765 - accuracy: 0.4928
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6929 - accuracy: 0.4983
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6570 - accuracy: 0.5006
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 4s - loss: 7.6597 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6676 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6401 - accuracy: 0.5017
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6473 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 00:20:19.767942
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 00:20:19.767942  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 00:20:25.741794: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 00:20:25.746127: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 00:20:25.746308: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c2790d02c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 00:20:25.746323: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fe528dbfbe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 963ms/step - loss: 1.6124 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.5162 - val_crf_viterbi_accuracy: 0.0667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe52ace3fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7816 - accuracy: 0.4925
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7740 - accuracy: 0.4930
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7510 - accuracy: 0.4945
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7484 - accuracy: 0.4947
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 4s - loss: 7.6806 - accuracy: 0.4991
12000/25000 [=============>................] - ETA: 4s - loss: 7.7369 - accuracy: 0.4954
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7232 - accuracy: 0.4963
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7137 - accuracy: 0.4969
15000/25000 [=================>............] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7117 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6955 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7245 - accuracy: 0.4962
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7159 - accuracy: 0.4968
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7257 - accuracy: 0.4961
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7053 - accuracy: 0.4975
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6920 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 9s 365us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fe4bfb080f0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:43:03, 9.31kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:14:09, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 197k/862M [00:01<12:48:58, 18.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 811k/862M [00:01<8:58:51, 26.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.07M/862M [00:01<6:16:23, 38.0kB/s].vector_cache/glove.6B.zip:   1%|          | 6.28M/862M [00:01<4:22:37, 54.3kB/s].vector_cache/glove.6B.zip:   1%|          | 10.4M/862M [00:01<3:03:04, 77.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:01<2:07:30, 111kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:28:44, 158kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<1:01:46, 225kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:02<43:13, 321kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:02<30:10, 457kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:02<21:08, 650kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<14:47, 923kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<10:25, 1.30MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<07:21, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.9M/862M [00:03<07:03, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:03<05:19, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:03<03:53, 3.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<09:10, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<9:45:43, 22.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<6:50:27, 32.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:05<4:46:32, 46.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<3:29:55, 63.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<2:29:44, 89.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<1:45:22, 126kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:07<1:13:40, 180kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<1:26:15, 154kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<1:01:42, 215kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:09<43:23, 305kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<33:24, 395kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<26:10, 504kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<18:54, 697kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:11<13:19, 986kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<30:50, 426kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<22:56, 572kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<16:21, 801kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<14:29, 902kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<12:49, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<09:33, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:15<06:48, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<20:24, 637kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<15:36, 833kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:17<11:11, 1.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<10:51, 1.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<08:54, 1.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:19<06:33, 1.97MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<07:37, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<06:38, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<04:54, 2.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<06:29, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:23<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<04:24, 2.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<06:06, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:25<05:34, 2.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:09, 3.05MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:43, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:18, 2.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:45, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:19, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:00, 3.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:17, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<03:57, 3.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:32, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<05:12, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:38, 2.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<03:56, 3.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<06:26, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:02, 2.44MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<03:41, 3.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:09, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:39, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:47, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:21, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:48, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<04:12, 2.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<33:02, 366kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<24:24, 496kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<17:19, 697kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<12:54, 933kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:08:56, 24.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<5:42:00, 35.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<4:00:47, 49.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:51:03, 70.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:00:15, 99.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<1:23:58, 142kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:28:38, 134kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:03:17, 188kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<44:31, 267kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<33:44, 351kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<26:07, 453kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<18:47, 629kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<13:17, 887kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<14:09, 831kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:10, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<08:03, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:17, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:02, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:11, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:14, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:11, 2.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:33, 2.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:20, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:57, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:36, 3.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<09:22, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:45, 1.99MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<06:35, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:02, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:28, 2.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:56, 2.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:12, 860kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:14, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:25, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<18:31, 609kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<15:21, 735kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<11:15, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<07:59, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<11:53, 943kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:31, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:54, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:20, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:31, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:47, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:12, 2.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:36, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:30, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:49, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:26, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:05, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:41, 2.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<37:15, 294kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<27:12, 402kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<19:14, 567kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<15:53, 684kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<13:25, 810kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<09:57, 1.09MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<07:03, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<29:55, 361kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<22:04, 489kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<15:42, 686kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<13:22, 802kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<11:43, 915kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:46, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<06:14, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<14:02, 759kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<10:57, 972kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:56, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<07:55, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:50, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:56, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:18, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:15, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:11, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:36, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:34, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:00, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:44, 2.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:57, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:25, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:30, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:19, 2.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:08, 3.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:42, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:11, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:15, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:57, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<06:19, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:57, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:34, 2.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<26:50, 377kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<19:52, 509kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<14:07, 714kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<12:06, 829kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<10:36, 947kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:51, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:37, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:57, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:26, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:42, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:26, 2.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<15:23, 644kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<12:52, 770kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<09:26, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:42, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<11:00, 894kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:44, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:21, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:01, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<6:43:00, 24.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<4:42:23, 34.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<3:16:55, 49.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<2:23:08, 67.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<1:42:19, 94.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<1:12:03, 134kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<50:18, 192kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<47:16, 204kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<34:05, 282kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<24:03, 399kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<18:55, 505kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<15:16, 626kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<11:07, 858kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<07:52, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:03, 859kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:46, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<06:22, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:36, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:37, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:07, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:40, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<47:24, 198kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<34:09, 274kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<24:05, 387kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<18:53, 492kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<15:11, 611kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<11:03, 838kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<07:50, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<09:51, 935kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:52, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<05:42, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:03, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:12, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:45, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:15, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:05, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:57, 3.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:00, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:51, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:17, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:03, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:29, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:20, 2.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:20, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:50, 2.31MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<02:47, 3.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:31, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:28, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:50, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:18, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:14, 2.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:12, 2.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:52, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:53, 2.98MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:35, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:35, 2.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:37, 3.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<05:36, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:49, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:35, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:27, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:54, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:52, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<02:50, 2.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:43, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<07:25, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:02, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:25, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:58, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:52, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:37, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:05, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:59, 2.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<13:44, 601kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<12:42, 650kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:33, 863kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:52, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<06:46, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:35, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:02, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:23, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:04, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<06:25, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:44, 2.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<24:02, 335kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<19:38, 410kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<14:18, 562kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<10:09, 789kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<07:14, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<10:37, 752kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:52, 809kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:27, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:23, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:50, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<06:39, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:11, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:46, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:47, 2.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:38, 907kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:36, 911kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<06:37, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:47, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:26, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:11, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:56, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:13, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<4:35:04, 28.0kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<3:12:55, 39.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<2:14:49, 57.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<1:34:12, 81.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<1:10:31, 108kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<54:43, 140kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<39:34, 193kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<27:54, 273kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<19:33, 388kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<18:04, 419kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<14:49, 511kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<10:56, 691kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<07:47, 966kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:31, 998kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:42, 974kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:59, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<04:20, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:59, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:38, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:24, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<03:14, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:25, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:12, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:05, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:58, 2.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:13, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<17:23, 420kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<14:15, 512kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<10:24, 700kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<07:22, 984kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<05:17, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<26:17, 275kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<20:36, 351kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<14:55, 483kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<10:32, 682kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<09:27, 757kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<08:40, 825kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:35, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<04:44, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:23, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:48, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:29, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:17, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:23, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:12, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:04, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:07, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:53, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:51, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:49, 2.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:05, 3.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<12:51, 535kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<11:05, 620kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<08:15, 832kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<05:52, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<06:07, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:14, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:46, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:28, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:24, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:56, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:53, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:58, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:49, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:46, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:45, 2.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:04, 3.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<15:00, 440kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<12:23, 533kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<09:04, 727kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<06:27, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:26, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:56, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:34, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<04:24, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:50, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:45, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:45, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:01, 3.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<15:18, 418kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<12:32, 510kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<09:08, 698kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<06:29, 980kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<04:38, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<20:14, 313kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<15:58, 396kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<11:32, 547kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<08:09, 771kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<05:49, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<12:05, 517kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<10:38, 588kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<07:57, 785kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:39, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:36, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:43, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:25, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:12, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:20, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:48, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:42, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:41, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:41, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:20, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:27, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:31, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:41, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:12, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:27, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<03:38, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:28, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:37, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:38, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:27, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:55, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:07, 1.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:15, 2.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:37, 1.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:01, 1.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:06, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:17, 2.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:01, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:35, 1.59MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:51, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:04, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:59, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:06, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:19, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:24, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<04:22, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:23, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:25, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:50, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:36, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:32, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:32, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:08, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:55, 1.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:46, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:42, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:15, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:10, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:13, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:18, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:33, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:15, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:15, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:20, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<05:27, 956kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:53, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:38, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:38, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:24, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:23, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:35, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:53, 2.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:15, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:19, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:34, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:51, 2.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:08, 702kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<06:13, 805kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:39, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<03:17, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<03:57, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<3:01:54, 27.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<2:07:22, 38.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<1:28:29, 55.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<1:04:17, 76.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<47:23, 103kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<33:37, 145kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<23:34, 206kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<16:28, 293kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<14:28, 333kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<11:18, 425kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<08:08, 590kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<05:45, 829kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:17, 896kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:50, 980kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:36, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:36, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:05, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:15, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:31, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:48, 2.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:09, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:21, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:35, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:53, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:52, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:17, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:39, 2.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:49, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:43, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:52, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<02:03, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:14, 1.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:30, 1.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:41, 1.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:57, 2.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:55, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:53, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:28, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:51, 2.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:04, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:16, 1.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:48, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:55, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:12, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:44, 2.39MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:52, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:07, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:41, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:06, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:11, 3.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<08:06, 491kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<06:20, 628kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:37, 859kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<03:16, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:40, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:20, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:29, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:46, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<06:11, 622kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<04:46, 805kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:27, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<02:27, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<04:41, 807kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:57, 954kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:55, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:39, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:30, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:54, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:21, 2.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<05:54, 617kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:47, 760kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:29, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<02:26, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<12:50, 278kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<09:38, 371kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<06:51, 519kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<04:47, 734kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<05:38, 621kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<04:35, 764kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:21, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<02:21, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<08:34, 401kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<06:35, 521kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:45, 719kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:50, 876kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:17, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:25, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:43, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:02, 818kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:25, 965kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:30, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:46, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:35, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<1:56:51, 27.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<1:21:34, 39.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<56:40, 56.0kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<40:50, 77.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<28:47, 110kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<20:02, 156kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<14:41, 211kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<10:48, 286kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<07:40, 402kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<05:47, 523kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<04:36, 657kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:20, 900kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:48, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:29, 1.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:52, 1.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:46, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:20, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:57, 2.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:44, 1.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:24, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:42, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:41, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:18, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:21, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:25, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:06, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:12, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:19, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:00, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:43, 3.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:40, 954kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:18, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:43, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:36, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:34, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:14, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:17, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:00, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:05, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:11, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:55, 2.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:01, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:06, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:52, 2.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:58, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:04, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:50, 2.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:35, 3.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<10:57, 195kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<08:02, 266kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<05:41, 373kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<04:14, 489kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<03:20, 619kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:24, 851kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:59, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<02:05, 959kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:38, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:10, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:18, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:15, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:57, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:58, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:00, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:50, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:53, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:29, 3.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:10, 795kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:49, 943kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:20, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:00, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<1:03:04, 26.4kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<43:50, 37.7kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<29:56, 53.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<21:29, 74.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<15:07, 105kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<10:25, 150kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<07:33, 202kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<05:33, 275kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<03:54, 386kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:53, 504kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:17, 637kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:38, 874kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:20, 1.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:11, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:53, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:49, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:48, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:37, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:39, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:30, 2.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:21, 3.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:13, 965kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:54, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:37, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:03, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:56, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:41, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:38, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:38, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:29, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:29, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:31, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:24, 2.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:16, 3.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<05:31, 165kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<04:00, 226kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:48, 318kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:59, 422kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:32, 545kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:05, 753kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:50, 910kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:43, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:32, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:28, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:24, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:19, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:26, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:13, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 1.98MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:09, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:14, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:07, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:04, 2.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 3.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<03:26, 27.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<02:01, 39.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<01:43, 46.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:23, 65.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:15, 89.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:07, 127kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 817/400000 [00:00<00:48, 8162.11it/s]  0%|          | 1675/400000 [00:00<00:48, 8281.95it/s]  1%|          | 2587/400000 [00:00<00:46, 8515.95it/s]  1%|          | 3461/400000 [00:00<00:46, 8580.49it/s]  1%|          | 4244/400000 [00:00<00:47, 8336.78it/s]  1%|▏         | 5074/400000 [00:00<00:47, 8322.64it/s]  2%|▏         | 6003/400000 [00:00<00:45, 8590.57it/s]  2%|▏         | 6874/400000 [00:00<00:45, 8622.18it/s]  2%|▏         | 7788/400000 [00:00<00:44, 8769.08it/s]  2%|▏         | 8635/400000 [00:01<00:45, 8636.31it/s]  2%|▏         | 9478/400000 [00:01<00:47, 8150.39it/s]  3%|▎         | 10316/400000 [00:01<00:47, 8216.56it/s]  3%|▎         | 11200/400000 [00:01<00:46, 8393.59it/s]  3%|▎         | 12036/400000 [00:01<00:46, 8288.92it/s]  3%|▎         | 12863/400000 [00:01<00:47, 8180.36it/s]  3%|▎         | 13713/400000 [00:01<00:46, 8272.11it/s]  4%|▎         | 14635/400000 [00:01<00:45, 8532.34it/s]  4%|▍         | 15536/400000 [00:01<00:44, 8667.99it/s]  4%|▍         | 16428/400000 [00:01<00:43, 8741.25it/s]  4%|▍         | 17304/400000 [00:02<00:44, 8675.47it/s]  5%|▍         | 18173/400000 [00:02<00:44, 8604.25it/s]  5%|▍         | 19035/400000 [00:02<00:45, 8405.67it/s]  5%|▍         | 19878/400000 [00:02<00:45, 8305.71it/s]  5%|▌         | 20745/400000 [00:02<00:45, 8410.22it/s]  5%|▌         | 21588/400000 [00:02<00:45, 8244.84it/s]  6%|▌         | 22415/400000 [00:02<00:46, 8158.51it/s]  6%|▌         | 23243/400000 [00:02<00:45, 8193.92it/s]  6%|▌         | 24073/400000 [00:02<00:45, 8224.70it/s]  6%|▌         | 24942/400000 [00:02<00:44, 8357.18it/s]  6%|▋         | 25779/400000 [00:03<00:45, 8205.99it/s]  7%|▋         | 26610/400000 [00:03<00:45, 8236.01it/s]  7%|▋         | 27499/400000 [00:03<00:44, 8419.62it/s]  7%|▋         | 28367/400000 [00:03<00:43, 8495.08it/s]  7%|▋         | 29249/400000 [00:03<00:43, 8589.54it/s]  8%|▊         | 30110/400000 [00:03<00:43, 8473.67it/s]  8%|▊         | 30991/400000 [00:03<00:43, 8571.76it/s]  8%|▊         | 31902/400000 [00:03<00:42, 8725.15it/s]  8%|▊         | 32776/400000 [00:03<00:43, 8508.04it/s]  8%|▊         | 33630/400000 [00:03<00:43, 8460.58it/s]  9%|▊         | 34503/400000 [00:04<00:42, 8537.33it/s]  9%|▉         | 35359/400000 [00:04<00:43, 8392.39it/s]  9%|▉         | 36283/400000 [00:04<00:42, 8628.58it/s]  9%|▉         | 37237/400000 [00:04<00:40, 8881.94it/s] 10%|▉         | 38129/400000 [00:04<00:41, 8682.08it/s] 10%|▉         | 39006/400000 [00:04<00:41, 8705.08it/s] 10%|▉         | 39880/400000 [00:04<00:41, 8708.14it/s] 10%|█         | 40809/400000 [00:04<00:40, 8874.23it/s] 10%|█         | 41728/400000 [00:04<00:39, 8962.94it/s] 11%|█         | 42659/400000 [00:04<00:39, 9059.94it/s] 11%|█         | 43567/400000 [00:05<00:39, 8989.52it/s] 11%|█         | 44468/400000 [00:05<00:41, 8539.59it/s] 11%|█▏        | 45340/400000 [00:05<00:41, 8592.55it/s] 12%|█▏        | 46204/400000 [00:05<00:42, 8343.82it/s] 12%|█▏        | 47043/400000 [00:05<00:43, 8178.40it/s] 12%|█▏        | 47865/400000 [00:05<00:43, 8108.84it/s] 12%|█▏        | 48772/400000 [00:05<00:41, 8374.01it/s] 12%|█▏        | 49682/400000 [00:05<00:40, 8575.87it/s] 13%|█▎        | 50613/400000 [00:05<00:39, 8782.84it/s] 13%|█▎        | 51510/400000 [00:06<00:39, 8834.63it/s] 13%|█▎        | 52397/400000 [00:06<00:39, 8726.43it/s] 13%|█▎        | 53273/400000 [00:06<00:40, 8473.93it/s] 14%|█▎        | 54232/400000 [00:06<00:39, 8777.93it/s] 14%|█▍        | 55115/400000 [00:06<00:40, 8483.59it/s] 14%|█▍        | 55970/400000 [00:06<00:41, 8194.43it/s] 14%|█▍        | 56796/400000 [00:06<00:42, 8136.09it/s] 14%|█▍        | 57666/400000 [00:06<00:41, 8293.54it/s] 15%|█▍        | 58504/400000 [00:06<00:41, 8317.17it/s] 15%|█▍        | 59342/400000 [00:06<00:40, 8333.03it/s] 15%|█▌        | 60224/400000 [00:07<00:40, 8470.15it/s] 15%|█▌        | 61073/400000 [00:07<00:40, 8329.65it/s] 15%|█▌        | 61908/400000 [00:07<00:41, 8115.10it/s] 16%|█▌        | 62773/400000 [00:07<00:40, 8268.22it/s] 16%|█▌        | 63655/400000 [00:07<00:39, 8424.47it/s] 16%|█▌        | 64500/400000 [00:07<00:39, 8394.60it/s] 16%|█▋        | 65415/400000 [00:07<00:38, 8606.40it/s] 17%|█▋        | 66279/400000 [00:07<00:38, 8566.77it/s] 17%|█▋        | 67138/400000 [00:07<00:38, 8566.92it/s] 17%|█▋        | 68020/400000 [00:07<00:38, 8638.78it/s] 17%|█▋        | 68896/400000 [00:08<00:38, 8672.57it/s] 17%|█▋        | 69765/400000 [00:08<00:38, 8518.68it/s] 18%|█▊        | 70721/400000 [00:08<00:37, 8806.27it/s] 18%|█▊        | 71605/400000 [00:08<00:37, 8796.48it/s] 18%|█▊        | 72487/400000 [00:08<00:37, 8757.74it/s] 18%|█▊        | 73365/400000 [00:08<00:37, 8713.15it/s] 19%|█▊        | 74293/400000 [00:08<00:36, 8875.23it/s] 19%|█▉        | 75236/400000 [00:08<00:35, 9030.66it/s] 19%|█▉        | 76141/400000 [00:08<00:37, 8748.80it/s] 19%|█▉        | 77020/400000 [00:09<00:37, 8627.33it/s] 19%|█▉        | 77886/400000 [00:09<00:37, 8630.45it/s] 20%|█▉        | 78751/400000 [00:09<00:38, 8318.53it/s] 20%|█▉        | 79587/400000 [00:09<00:38, 8237.35it/s] 20%|██        | 80418/400000 [00:09<00:38, 8257.32it/s] 20%|██        | 81297/400000 [00:09<00:37, 8409.93it/s] 21%|██        | 82141/400000 [00:09<00:37, 8410.37it/s] 21%|██        | 83067/400000 [00:09<00:36, 8648.01it/s] 21%|██        | 83970/400000 [00:09<00:36, 8757.54it/s] 21%|██        | 84901/400000 [00:09<00:35, 8914.39it/s] 21%|██▏       | 85799/400000 [00:10<00:35, 8932.94it/s] 22%|██▏       | 86725/400000 [00:10<00:34, 9021.77it/s] 22%|██▏       | 87636/400000 [00:10<00:34, 9045.43it/s] 22%|██▏       | 88586/400000 [00:10<00:33, 9175.01it/s] 22%|██▏       | 89505/400000 [00:10<00:35, 8736.30it/s] 23%|██▎       | 90384/400000 [00:10<00:37, 8285.87it/s] 23%|██▎       | 91222/400000 [00:10<00:38, 8039.70it/s] 23%|██▎       | 92057/400000 [00:10<00:37, 8129.30it/s] 23%|██▎       | 93015/400000 [00:10<00:36, 8514.11it/s] 23%|██▎       | 93876/400000 [00:10<00:36, 8452.70it/s] 24%|██▎       | 94769/400000 [00:11<00:35, 8589.67it/s] 24%|██▍       | 95633/400000 [00:11<00:36, 8436.69it/s] 24%|██▍       | 96507/400000 [00:11<00:35, 8512.74it/s] 24%|██▍       | 97403/400000 [00:11<00:35, 8641.36it/s] 25%|██▍       | 98270/400000 [00:11<00:35, 8446.11it/s] 25%|██▍       | 99118/400000 [00:11<00:36, 8299.94it/s] 25%|██▍       | 99962/400000 [00:11<00:35, 8340.85it/s] 25%|██▌       | 100859/400000 [00:11<00:35, 8518.01it/s] 25%|██▌       | 101773/400000 [00:11<00:34, 8693.49it/s] 26%|██▌       | 102679/400000 [00:12<00:33, 8798.14it/s] 26%|██▌       | 103561/400000 [00:12<00:33, 8796.34it/s] 26%|██▌       | 104443/400000 [00:12<00:35, 8434.60it/s] 26%|██▋       | 105302/400000 [00:12<00:34, 8479.73it/s] 27%|██▋       | 106153/400000 [00:12<00:35, 8317.28it/s] 27%|██▋       | 106992/400000 [00:12<00:35, 8336.89it/s] 27%|██▋       | 107828/400000 [00:12<00:35, 8315.96it/s] 27%|██▋       | 108703/400000 [00:12<00:34, 8440.59it/s] 27%|██▋       | 109610/400000 [00:12<00:33, 8619.32it/s] 28%|██▊       | 110521/400000 [00:12<00:33, 8759.37it/s] 28%|██▊       | 111399/400000 [00:13<00:33, 8692.55it/s] 28%|██▊       | 112270/400000 [00:13<00:33, 8693.65it/s] 28%|██▊       | 113141/400000 [00:13<00:33, 8503.47it/s] 29%|██▊       | 114029/400000 [00:13<00:33, 8611.95it/s] 29%|██▊       | 114892/400000 [00:13<00:33, 8596.36it/s] 29%|██▉       | 115753/400000 [00:13<00:33, 8418.29it/s] 29%|██▉       | 116597/400000 [00:13<00:33, 8342.94it/s] 29%|██▉       | 117465/400000 [00:13<00:33, 8440.97it/s] 30%|██▉       | 118311/400000 [00:13<00:33, 8346.32it/s] 30%|██▉       | 119176/400000 [00:13<00:33, 8434.91it/s] 30%|███       | 120021/400000 [00:14<00:33, 8373.32it/s] 30%|███       | 120860/400000 [00:14<00:33, 8295.39it/s] 30%|███       | 121745/400000 [00:14<00:32, 8452.27it/s] 31%|███       | 122655/400000 [00:14<00:32, 8634.16it/s] 31%|███       | 123523/400000 [00:14<00:31, 8647.70it/s] 31%|███       | 124390/400000 [00:14<00:32, 8536.88it/s] 31%|███▏      | 125245/400000 [00:14<00:32, 8484.53it/s] 32%|███▏      | 126154/400000 [00:14<00:31, 8657.43it/s] 32%|███▏      | 127030/400000 [00:14<00:31, 8687.42it/s] 32%|███▏      | 127969/400000 [00:14<00:30, 8884.93it/s] 32%|███▏      | 128895/400000 [00:15<00:30, 8993.51it/s] 32%|███▏      | 129796/400000 [00:15<00:30, 8815.37it/s] 33%|███▎      | 130680/400000 [00:15<00:30, 8723.24it/s] 33%|███▎      | 131579/400000 [00:15<00:30, 8800.32it/s] 33%|███▎      | 132461/400000 [00:15<00:31, 8476.86it/s] 33%|███▎      | 133313/400000 [00:15<00:32, 8186.71it/s] 34%|███▎      | 134171/400000 [00:15<00:32, 8299.97it/s] 34%|███▍      | 135070/400000 [00:15<00:31, 8493.49it/s] 34%|███▍      | 135923/400000 [00:15<00:31, 8494.55it/s] 34%|███▍      | 136817/400000 [00:16<00:30, 8622.77it/s] 34%|███▍      | 137682/400000 [00:16<00:31, 8368.80it/s] 35%|███▍      | 138523/400000 [00:16<00:32, 8120.29it/s] 35%|███▍      | 139339/400000 [00:16<00:32, 8084.61it/s] 35%|███▌      | 140236/400000 [00:16<00:31, 8329.96it/s] 35%|███▌      | 141114/400000 [00:16<00:30, 8458.23it/s] 35%|███▌      | 141963/400000 [00:16<00:31, 8255.68it/s] 36%|███▌      | 142855/400000 [00:16<00:30, 8442.22it/s] 36%|███▌      | 143804/400000 [00:16<00:29, 8731.23it/s] 36%|███▌      | 144752/400000 [00:16<00:28, 8939.46it/s] 36%|███▋      | 145651/400000 [00:17<00:28, 8948.36it/s] 37%|███▋      | 146550/400000 [00:17<00:28, 8887.52it/s] 37%|███▋      | 147442/400000 [00:17<00:30, 8382.97it/s] 37%|███▋      | 148288/400000 [00:17<00:31, 7886.56it/s] 37%|███▋      | 149088/400000 [00:17<00:32, 7810.52it/s] 37%|███▋      | 149886/400000 [00:17<00:31, 7859.56it/s] 38%|███▊      | 150691/400000 [00:17<00:31, 7913.74it/s] 38%|███▊      | 151497/400000 [00:17<00:31, 7956.01it/s] 38%|███▊      | 152351/400000 [00:17<00:30, 8121.74it/s] 38%|███▊      | 153260/400000 [00:17<00:29, 8388.26it/s] 39%|███▊      | 154185/400000 [00:18<00:28, 8628.50it/s] 39%|███▉      | 155053/400000 [00:18<00:28, 8584.36it/s] 39%|███▉      | 155942/400000 [00:18<00:28, 8671.02it/s] 39%|███▉      | 156897/400000 [00:18<00:27, 8915.33it/s] 39%|███▉      | 157793/400000 [00:18<00:27, 8927.72it/s] 40%|███▉      | 158689/400000 [00:18<00:27, 8770.03it/s] 40%|███▉      | 159569/400000 [00:18<00:28, 8494.02it/s] 40%|████      | 160422/400000 [00:18<00:29, 8218.00it/s] 40%|████      | 161286/400000 [00:18<00:28, 8339.15it/s] 41%|████      | 162197/400000 [00:19<00:27, 8556.29it/s] 41%|████      | 163063/400000 [00:19<00:27, 8585.80it/s] 41%|████      | 163925/400000 [00:19<00:27, 8484.16it/s] 41%|████      | 164776/400000 [00:19<00:28, 8387.45it/s] 41%|████▏     | 165638/400000 [00:19<00:27, 8454.97it/s] 42%|████▏     | 166521/400000 [00:19<00:27, 8561.54it/s] 42%|████▏     | 167379/400000 [00:19<00:27, 8429.90it/s] 42%|████▏     | 168224/400000 [00:19<00:27, 8327.30it/s] 42%|████▏     | 169134/400000 [00:19<00:27, 8544.73it/s] 43%|████▎     | 170053/400000 [00:19<00:26, 8727.26it/s] 43%|████▎     | 170969/400000 [00:20<00:25, 8851.94it/s] 43%|████▎     | 171905/400000 [00:20<00:25, 8996.43it/s] 43%|████▎     | 172807/400000 [00:20<00:25, 8938.74it/s] 43%|████▎     | 173710/400000 [00:20<00:25, 8963.60it/s] 44%|████▎     | 174608/400000 [00:20<00:25, 8838.97it/s] 44%|████▍     | 175494/400000 [00:20<00:26, 8490.61it/s] 44%|████▍     | 176347/400000 [00:20<00:26, 8453.74it/s] 44%|████▍     | 177196/400000 [00:20<00:26, 8456.38it/s] 45%|████▍     | 178094/400000 [00:20<00:25, 8606.61it/s] 45%|████▍     | 179020/400000 [00:20<00:25, 8791.62it/s] 45%|████▍     | 179956/400000 [00:21<00:24, 8952.29it/s] 45%|████▌     | 180878/400000 [00:21<00:24, 9029.50it/s] 45%|████▌     | 181783/400000 [00:21<00:24, 8744.81it/s] 46%|████▌     | 182661/400000 [00:21<00:25, 8613.01it/s] 46%|████▌     | 183526/400000 [00:21<00:25, 8543.75it/s] 46%|████▌     | 184383/400000 [00:21<00:25, 8358.06it/s] 46%|████▋     | 185318/400000 [00:21<00:24, 8631.68it/s] 47%|████▋     | 186186/400000 [00:21<00:25, 8409.75it/s] 47%|████▋     | 187126/400000 [00:21<00:24, 8682.13it/s] 47%|████▋     | 188038/400000 [00:22<00:24, 8808.88it/s] 47%|████▋     | 188970/400000 [00:22<00:23, 8953.85it/s] 47%|████▋     | 189869/400000 [00:22<00:24, 8743.02it/s] 48%|████▊     | 190747/400000 [00:22<00:24, 8409.03it/s] 48%|████▊     | 191594/400000 [00:22<00:25, 8270.90it/s] 48%|████▊     | 192497/400000 [00:22<00:24, 8482.61it/s] 48%|████▊     | 193380/400000 [00:22<00:24, 8575.20it/s] 49%|████▊     | 194268/400000 [00:22<00:23, 8663.93it/s] 49%|████▉     | 195137/400000 [00:22<00:24, 8465.75it/s] 49%|████▉     | 195998/400000 [00:22<00:23, 8508.42it/s] 49%|████▉     | 196915/400000 [00:23<00:23, 8693.24it/s] 49%|████▉     | 197853/400000 [00:23<00:22, 8887.80it/s] 50%|████▉     | 198745/400000 [00:23<00:22, 8796.60it/s] 50%|████▉     | 199627/400000 [00:23<00:22, 8781.12it/s] 50%|█████     | 200523/400000 [00:23<00:22, 8833.69it/s] 50%|█████     | 201415/400000 [00:23<00:22, 8857.88it/s] 51%|█████     | 202302/400000 [00:23<00:22, 8825.34it/s] 51%|█████     | 203186/400000 [00:23<00:22, 8826.56it/s] 51%|█████     | 204070/400000 [00:23<00:22, 8817.23it/s] 51%|█████     | 204953/400000 [00:23<00:22, 8787.09it/s] 51%|█████▏    | 205892/400000 [00:24<00:21, 8957.84it/s] 52%|█████▏    | 206789/400000 [00:24<00:21, 8863.08it/s] 52%|█████▏    | 207677/400000 [00:24<00:22, 8717.90it/s] 52%|█████▏    | 208578/400000 [00:24<00:21, 8802.81it/s] 52%|█████▏    | 209460/400000 [00:24<00:21, 8772.30it/s] 53%|█████▎    | 210372/400000 [00:24<00:21, 8871.50it/s] 53%|█████▎    | 211260/400000 [00:24<00:21, 8622.20it/s] 53%|█████▎    | 212141/400000 [00:24<00:21, 8676.29it/s] 53%|█████▎    | 213011/400000 [00:24<00:22, 8442.31it/s] 53%|█████▎    | 213858/400000 [00:24<00:22, 8268.42it/s] 54%|█████▎    | 214791/400000 [00:25<00:21, 8559.62it/s] 54%|█████▍    | 215705/400000 [00:25<00:21, 8723.40it/s] 54%|█████▍    | 216582/400000 [00:25<00:21, 8651.58it/s] 54%|█████▍    | 217450/400000 [00:25<00:21, 8583.18it/s] 55%|█████▍    | 218311/400000 [00:25<00:21, 8584.79it/s] 55%|█████▍    | 219171/400000 [00:25<00:21, 8389.19it/s] 55%|█████▌    | 220029/400000 [00:25<00:21, 8445.42it/s] 55%|█████▌    | 220876/400000 [00:25<00:21, 8400.41it/s] 55%|█████▌    | 221718/400000 [00:25<00:21, 8223.98it/s] 56%|█████▌    | 222542/400000 [00:26<00:21, 8196.13it/s] 56%|█████▌    | 223447/400000 [00:26<00:20, 8434.21it/s] 56%|█████▌    | 224349/400000 [00:26<00:20, 8601.01it/s] 56%|█████▋    | 225212/400000 [00:26<00:20, 8537.04it/s] 57%|█████▋    | 226068/400000 [00:26<00:20, 8438.16it/s] 57%|█████▋    | 226970/400000 [00:26<00:20, 8603.65it/s] 57%|█████▋    | 227833/400000 [00:26<00:20, 8458.63it/s] 57%|█████▋    | 228742/400000 [00:26<00:19, 8636.79it/s] 57%|█████▋    | 229657/400000 [00:26<00:19, 8781.67it/s] 58%|█████▊    | 230592/400000 [00:26<00:18, 8944.39it/s] 58%|█████▊    | 231562/400000 [00:27<00:18, 9156.94it/s] 58%|█████▊    | 232508/400000 [00:27<00:18, 9244.49it/s] 58%|█████▊    | 233435/400000 [00:27<00:18, 9049.92it/s] 59%|█████▊    | 234343/400000 [00:27<00:18, 8772.24it/s] 59%|█████▉    | 235224/400000 [00:27<00:18, 8707.62it/s] 59%|█████▉    | 236100/400000 [00:27<00:18, 8720.76it/s] 59%|█████▉    | 236974/400000 [00:27<00:19, 8530.81it/s] 59%|█████▉    | 237897/400000 [00:27<00:18, 8728.12it/s] 60%|█████▉    | 238773/400000 [00:27<00:18, 8674.83it/s] 60%|█████▉    | 239657/400000 [00:27<00:18, 8722.02it/s] 60%|██████    | 240531/400000 [00:28<00:18, 8574.19it/s] 60%|██████    | 241428/400000 [00:28<00:18, 8688.63it/s] 61%|██████    | 242346/400000 [00:28<00:17, 8828.40it/s] 61%|██████    | 243241/400000 [00:28<00:17, 8862.31it/s] 61%|██████    | 244129/400000 [00:28<00:17, 8779.22it/s] 61%|██████▏   | 245008/400000 [00:28<00:17, 8663.60it/s] 61%|██████▏   | 245876/400000 [00:28<00:17, 8612.78it/s] 62%|██████▏   | 246762/400000 [00:28<00:17, 8685.44it/s] 62%|██████▏   | 247632/400000 [00:28<00:17, 8619.08it/s] 62%|██████▏   | 248495/400000 [00:28<00:17, 8488.95it/s] 62%|██████▏   | 249345/400000 [00:29<00:18, 8330.10it/s] 63%|██████▎   | 250229/400000 [00:29<00:17, 8474.37it/s] 63%|██████▎   | 251098/400000 [00:29<00:17, 8536.84it/s] 63%|██████▎   | 251953/400000 [00:29<00:17, 8467.75it/s] 63%|██████▎   | 252835/400000 [00:29<00:17, 8569.04it/s] 63%|██████▎   | 253733/400000 [00:29<00:16, 8686.62it/s] 64%|██████▎   | 254657/400000 [00:29<00:16, 8844.59it/s] 64%|██████▍   | 255543/400000 [00:29<00:16, 8817.74it/s] 64%|██████▍   | 256440/400000 [00:29<00:16, 8859.89it/s] 64%|██████▍   | 257398/400000 [00:29<00:15, 9062.27it/s] 65%|██████▍   | 258321/400000 [00:30<00:15, 9110.44it/s] 65%|██████▍   | 259249/400000 [00:30<00:15, 9157.95it/s] 65%|██████▌   | 260166/400000 [00:30<00:15, 9011.84it/s] 65%|██████▌   | 261069/400000 [00:30<00:15, 8775.73it/s] 65%|██████▌   | 261949/400000 [00:30<00:16, 8598.81it/s] 66%|██████▌   | 262812/400000 [00:30<00:16, 8353.17it/s] 66%|██████▌   | 263651/400000 [00:30<00:16, 8346.83it/s] 66%|██████▌   | 264521/400000 [00:30<00:16, 8447.23it/s] 66%|██████▋   | 265368/400000 [00:30<00:16, 8388.46it/s] 67%|██████▋   | 266282/400000 [00:31<00:15, 8600.03it/s] 67%|██████▋   | 267145/400000 [00:31<00:15, 8399.64it/s] 67%|██████▋   | 267988/400000 [00:31<00:15, 8314.82it/s] 67%|██████▋   | 268833/400000 [00:31<00:15, 8354.09it/s] 67%|██████▋   | 269711/400000 [00:31<00:15, 8476.00it/s] 68%|██████▊   | 270561/400000 [00:31<00:15, 8473.05it/s] 68%|██████▊   | 271410/400000 [00:31<00:15, 8367.82it/s] 68%|██████▊   | 272328/400000 [00:31<00:14, 8595.34it/s] 68%|██████▊   | 273242/400000 [00:31<00:14, 8750.12it/s] 69%|██████▊   | 274135/400000 [00:31<00:14, 8802.16it/s] 69%|██████▉   | 275060/400000 [00:32<00:13, 8929.00it/s] 69%|██████▉   | 275955/400000 [00:32<00:14, 8681.69it/s] 69%|██████▉   | 276826/400000 [00:32<00:14, 8412.90it/s] 69%|██████▉   | 277671/400000 [00:32<00:14, 8305.46it/s] 70%|██████▉   | 278518/400000 [00:32<00:14, 8351.35it/s] 70%|██████▉   | 279387/400000 [00:32<00:14, 8449.39it/s] 70%|███████   | 280234/400000 [00:32<00:14, 8422.64it/s] 70%|███████   | 281078/400000 [00:32<00:14, 8412.85it/s] 70%|███████   | 281921/400000 [00:32<00:14, 8283.12it/s] 71%|███████   | 282810/400000 [00:32<00:13, 8456.30it/s] 71%|███████   | 283727/400000 [00:33<00:13, 8657.49it/s] 71%|███████   | 284610/400000 [00:33<00:13, 8707.80it/s] 71%|███████▏  | 285518/400000 [00:33<00:12, 8813.77it/s] 72%|███████▏  | 286440/400000 [00:33<00:12, 8928.75it/s] 72%|███████▏  | 287335/400000 [00:33<00:13, 8570.37it/s] 72%|███████▏  | 288198/400000 [00:33<00:13, 8587.23it/s] 72%|███████▏  | 289118/400000 [00:33<00:12, 8762.20it/s] 73%|███████▎  | 290009/400000 [00:33<00:12, 8804.45it/s] 73%|███████▎  | 290914/400000 [00:33<00:12, 8876.29it/s] 73%|███████▎  | 291804/400000 [00:34<00:12, 8610.49it/s] 73%|███████▎  | 292668/400000 [00:34<00:12, 8547.15it/s] 73%|███████▎  | 293525/400000 [00:34<00:12, 8511.45it/s] 74%|███████▎  | 294378/400000 [00:34<00:12, 8482.21it/s] 74%|███████▍  | 295250/400000 [00:34<00:12, 8551.60it/s] 74%|███████▍  | 296107/400000 [00:34<00:12, 8415.40it/s] 74%|███████▍  | 296950/400000 [00:34<00:12, 8341.49it/s] 74%|███████▍  | 297850/400000 [00:34<00:11, 8526.88it/s] 75%|███████▍  | 298741/400000 [00:34<00:11, 8636.91it/s] 75%|███████▍  | 299615/400000 [00:34<00:11, 8666.64it/s] 75%|███████▌  | 300504/400000 [00:35<00:11, 8731.04it/s] 75%|███████▌  | 301429/400000 [00:35<00:11, 8879.78it/s] 76%|███████▌  | 302319/400000 [00:35<00:11, 8686.50it/s] 76%|███████▌  | 303190/400000 [00:35<00:11, 8626.83it/s] 76%|███████▌  | 304055/400000 [00:35<00:11, 8594.26it/s] 76%|███████▌  | 304916/400000 [00:35<00:11, 8457.76it/s] 76%|███████▋  | 305763/400000 [00:35<00:11, 8339.14it/s] 77%|███████▋  | 306618/400000 [00:35<00:11, 8398.81it/s] 77%|███████▋  | 307529/400000 [00:35<00:10, 8599.66it/s] 77%|███████▋  | 308474/400000 [00:35<00:10, 8837.17it/s] 77%|███████▋  | 309361/400000 [00:36<00:10, 8721.05it/s] 78%|███████▊  | 310254/400000 [00:36<00:10, 8780.56it/s] 78%|███████▊  | 311134/400000 [00:36<00:10, 8782.37it/s] 78%|███████▊  | 312023/400000 [00:36<00:09, 8813.74it/s] 78%|███████▊  | 312943/400000 [00:36<00:09, 8925.86it/s] 78%|███████▊  | 313837/400000 [00:36<00:09, 8855.10it/s] 79%|███████▊  | 314724/400000 [00:36<00:09, 8819.49it/s] 79%|███████▉  | 315688/400000 [00:36<00:09, 9050.19it/s] 79%|███████▉  | 316595/400000 [00:36<00:09, 9033.23it/s] 79%|███████▉  | 317534/400000 [00:36<00:09, 9135.27it/s] 80%|███████▉  | 318449/400000 [00:37<00:08, 9094.37it/s] 80%|███████▉  | 319360/400000 [00:37<00:08, 9008.21it/s] 80%|████████  | 320262/400000 [00:37<00:09, 8803.23it/s] 80%|████████  | 321144/400000 [00:37<00:09, 8459.06it/s] 80%|████████  | 321994/400000 [00:37<00:09, 8420.81it/s] 81%|████████  | 322839/400000 [00:37<00:09, 8217.54it/s] 81%|████████  | 323714/400000 [00:37<00:09, 8369.71it/s] 81%|████████  | 324668/400000 [00:37<00:08, 8689.26it/s] 81%|████████▏ | 325544/400000 [00:37<00:08, 8709.08it/s] 82%|████████▏ | 326419/400000 [00:37<00:08, 8634.06it/s] 82%|████████▏ | 327286/400000 [00:38<00:08, 8621.04it/s] 82%|████████▏ | 328184/400000 [00:38<00:08, 8724.21it/s] 82%|████████▏ | 329093/400000 [00:38<00:08, 8830.04it/s] 82%|████████▏ | 329985/400000 [00:38<00:07, 8854.41it/s] 83%|████████▎ | 330872/400000 [00:38<00:08, 8565.58it/s] 83%|████████▎ | 331732/400000 [00:38<00:08, 8422.24it/s] 83%|████████▎ | 332671/400000 [00:38<00:07, 8689.63it/s] 83%|████████▎ | 333563/400000 [00:38<00:07, 8754.96it/s] 84%|████████▎ | 334495/400000 [00:38<00:07, 8915.08it/s] 84%|████████▍ | 335390/400000 [00:39<00:07, 8643.11it/s] 84%|████████▍ | 336285/400000 [00:39<00:07, 8731.21it/s] 84%|████████▍ | 337161/400000 [00:39<00:07, 8647.40it/s] 85%|████████▍ | 338086/400000 [00:39<00:07, 8818.28it/s] 85%|████████▍ | 339001/400000 [00:39<00:06, 8913.33it/s] 85%|████████▍ | 339895/400000 [00:39<00:06, 8645.81it/s] 85%|████████▌ | 340763/400000 [00:39<00:06, 8623.61it/s] 85%|████████▌ | 341661/400000 [00:39<00:06, 8726.62it/s] 86%|████████▌ | 342573/400000 [00:39<00:06, 8840.72it/s] 86%|████████▌ | 343467/400000 [00:39<00:06, 8869.04it/s] 86%|████████▌ | 344376/400000 [00:40<00:06, 8934.00it/s] 86%|████████▋ | 345271/400000 [00:40<00:06, 8870.00it/s] 87%|████████▋ | 346167/400000 [00:40<00:06, 8895.93it/s] 87%|████████▋ | 347099/400000 [00:40<00:05, 9015.99it/s] 87%|████████▋ | 348002/400000 [00:40<00:05, 8943.24it/s] 87%|████████▋ | 348897/400000 [00:40<00:06, 8482.62it/s] 87%|████████▋ | 349751/400000 [00:40<00:06, 8373.92it/s] 88%|████████▊ | 350635/400000 [00:40<00:05, 8506.34it/s] 88%|████████▊ | 351566/400000 [00:40<00:05, 8731.33it/s] 88%|████████▊ | 352463/400000 [00:40<00:05, 8799.61it/s] 88%|████████▊ | 353346/400000 [00:41<00:05, 8734.08it/s] 89%|████████▊ | 354222/400000 [00:41<00:05, 8669.72it/s] 89%|████████▉ | 355091/400000 [00:41<00:05, 8491.13it/s] 89%|████████▉ | 355943/400000 [00:41<00:05, 8405.96it/s] 89%|████████▉ | 356786/400000 [00:41<00:05, 8289.20it/s] 89%|████████▉ | 357617/400000 [00:41<00:05, 8047.29it/s] 90%|████████▉ | 358467/400000 [00:41<00:05, 8177.49it/s] 90%|████████▉ | 359444/400000 [00:41<00:04, 8596.61it/s] 90%|█████████ | 360363/400000 [00:41<00:04, 8766.24it/s] 90%|█████████ | 361274/400000 [00:42<00:04, 8865.64it/s] 91%|█████████ | 362199/400000 [00:42<00:04, 8976.98it/s] 91%|█████████ | 363170/400000 [00:42<00:04, 9182.43it/s] 91%|█████████ | 364094/400000 [00:42<00:03, 9197.36it/s] 91%|█████████▏| 365017/400000 [00:42<00:03, 9017.36it/s] 91%|█████████▏| 365941/400000 [00:42<00:03, 9081.80it/s] 92%|█████████▏| 366852/400000 [00:42<00:03, 8808.72it/s] 92%|█████████▏| 367785/400000 [00:42<00:03, 8958.69it/s] 92%|█████████▏| 368746/400000 [00:42<00:03, 9144.26it/s] 92%|█████████▏| 369664/400000 [00:42<00:03, 9033.76it/s] 93%|█████████▎| 370614/400000 [00:43<00:03, 9166.78it/s] 93%|█████████▎| 371533/400000 [00:43<00:03, 9078.42it/s] 93%|█████████▎| 372443/400000 [00:43<00:03, 8990.92it/s] 93%|█████████▎| 373364/400000 [00:43<00:02, 9053.34it/s] 94%|█████████▎| 374271/400000 [00:43<00:02, 8882.03it/s] 94%|█████████▍| 375161/400000 [00:43<00:02, 8774.71it/s] 94%|█████████▍| 376040/400000 [00:43<00:02, 8430.23it/s] 94%|█████████▍| 376916/400000 [00:43<00:02, 8524.05it/s] 94%|█████████▍| 377860/400000 [00:43<00:02, 8776.75it/s] 95%|█████████▍| 378744/400000 [00:43<00:02, 8795.44it/s] 95%|█████████▍| 379645/400000 [00:44<00:02, 8856.43it/s] 95%|█████████▌| 380533/400000 [00:44<00:02, 8811.54it/s] 95%|█████████▌| 381428/400000 [00:44<00:02, 8851.61it/s] 96%|█████████▌| 382325/400000 [00:44<00:01, 8885.93it/s] 96%|█████████▌| 383215/400000 [00:44<00:01, 8785.19it/s] 96%|█████████▌| 384095/400000 [00:44<00:01, 8773.96it/s] 96%|█████████▌| 384973/400000 [00:44<00:01, 8529.89it/s] 96%|█████████▋| 385894/400000 [00:44<00:01, 8723.04it/s] 97%|█████████▋| 386832/400000 [00:44<00:01, 8907.34it/s] 97%|█████████▋| 387726/400000 [00:44<00:01, 8905.31it/s] 97%|█████████▋| 388619/400000 [00:45<00:01, 8845.22it/s] 97%|█████████▋| 389505/400000 [00:45<00:01, 8771.16it/s] 98%|█████████▊| 390394/400000 [00:45<00:01, 8804.88it/s] 98%|█████████▊| 391312/400000 [00:45<00:00, 8911.53it/s] 98%|█████████▊| 392205/400000 [00:45<00:00, 8867.78it/s] 98%|█████████▊| 393093/400000 [00:45<00:00, 8676.38it/s] 98%|█████████▊| 393963/400000 [00:45<00:00, 8361.66it/s] 99%|█████████▊| 394811/400000 [00:45<00:00, 8395.63it/s] 99%|█████████▉| 395697/400000 [00:45<00:00, 8527.41it/s] 99%|█████████▉| 396552/400000 [00:46<00:00, 8458.67it/s] 99%|█████████▉| 397400/400000 [00:46<00:00, 8372.00it/s]100%|█████████▉| 398239/400000 [00:46<00:00, 8325.44it/s]100%|█████████▉| 399133/400000 [00:46<00:00, 8498.69it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8618.57it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe4c92faba8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0110307300039627 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.010994942291923191 	 Accuracy: 71

  model saves at 71% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15806 out of table with 15803 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15806 out of table with 15803 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

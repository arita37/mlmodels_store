
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb4091974a8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 01:12:44.405290
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 01:12:44.410472
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 01:12:44.414385
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 01:12:44.417753
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb4014e7400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355885.4688
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 293367.5312
Epoch 3/10

1/1 [==============================] - 0s 87ms/step - loss: 223861.6562
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 160651.9062
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 114088.7344
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 81031.8047
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 58356.4258
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 43109.8633
Epoch 9/10

1/1 [==============================] - 0s 87ms/step - loss: 32738.8301
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 25555.0977

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.7764733e-01  2.6138806e-01 -1.9445794e-03  6.4387403e-02
   3.3617175e-01  8.5902822e-01  5.3864378e-01  9.5930916e-01
   6.5920621e-01  1.0341709e+00  4.2615557e-01 -5.1305920e-01
   1.0903342e+00  6.1436856e-01  3.6043429e-01 -1.0155238e+00
   8.0055791e-01 -7.3917884e-01  1.0874239e+00 -7.6267523e-01
  -9.5231193e-01 -9.7497754e-02 -4.1251463e-01  6.5887415e-01
  -9.7269095e-02 -4.5839351e-01  2.8741890e-01 -3.7173346e-01
  -4.1064262e-01  8.0501515e-01 -4.1829832e-02 -7.1777385e-01
   1.0054079e+00  1.0557054e+00  2.3922901e-01  7.9723591e-01
   4.6673974e-01  1.0804547e+00 -2.6210111e-01 -4.2724660e-01
  -3.4844312e-01  1.1090599e-01  6.7055154e-01 -7.5088584e-01
   8.7176949e-01  5.5575687e-01  1.0936226e+00 -4.7680235e-01
  -1.0525593e+00 -2.0226551e-02 -1.0432401e+00 -3.5348538e-01
   7.1586043e-01 -9.1920233e-01  5.1355231e-01 -1.8066859e-01
   9.5050290e-02  1.8629909e-01 -1.6213097e-01  6.6806251e-01
  -1.0790366e+00  9.0103042e-01 -1.0883961e+00 -5.4704130e-01
   4.4654974e-01 -5.2578932e-01  3.7595015e-02 -4.4484934e-01
  -4.0002739e-01 -4.3755311e-01  9.2677069e-01  5.6231117e-01
   1.0905657e-01 -1.0634695e+00  3.5583027e-02  5.8458900e-01
  -9.6043801e-01 -2.9092151e-01  6.4353538e-01  1.0547360e+00
   5.5648309e-01  3.7742296e-01 -2.7651504e-01 -8.8753894e-02
   5.2528638e-01  7.4762151e-02  2.4997154e-01 -3.3047676e-01
   7.6785856e-01  6.8784970e-01 -7.5416398e-01 -7.3936063e-01
  -3.9995408e-01 -5.4695349e-02 -1.0843390e+00  1.8343416e-01
   9.2470086e-01  8.9943558e-01  6.5243965e-01  1.0638558e+00
  -3.0848667e-01 -8.3221257e-01 -8.2658368e-01 -7.8476876e-01
   4.8332745e-01  8.3074117e-01  8.4991395e-01  9.5167482e-01
   6.2623328e-01  8.8069212e-01 -6.9572055e-01 -5.1035845e-01
   4.0513851e-02 -2.0554894e-01 -9.3403000e-01 -6.2952244e-01
   2.7267763e-01  6.4632785e-01  1.0824405e+00 -2.8622743e-01
  -8.3505988e-02  4.4438848e+00  4.3844204e+00  4.1660957e+00
   3.1665881e+00  3.7154698e+00  2.5774527e+00  4.2811794e+00
   4.3726039e+00  3.2509236e+00  4.0756974e+00  2.4664116e+00
   3.3580236e+00  4.0966582e+00  4.1227517e+00  4.2616343e+00
   3.3482115e+00  3.4559138e+00  4.4066076e+00  3.8070700e+00
   2.7921591e+00  2.9555354e+00  3.7722707e+00  3.7283244e+00
   3.6256764e+00  3.6758511e+00  3.6203983e+00  4.4700093e+00
   2.8441174e+00  3.1383367e+00  2.6113188e+00  2.7720530e+00
   4.1199923e+00  2.8034127e+00  4.2910542e+00  2.3185043e+00
   3.6021345e+00  3.7079344e+00  3.6005549e+00  3.3557968e+00
   2.5207012e+00  3.8146925e+00  2.6932440e+00  3.8069713e+00
   2.2923002e+00  2.7929776e+00  2.7311208e+00  3.7268510e+00
   2.4700332e+00  3.6630852e+00  2.6989384e+00  3.4598231e+00
   2.3150361e+00  2.5225258e+00  2.9030273e+00  3.9148445e+00
   3.8542202e+00  3.5712240e+00  3.0091434e+00  2.5721395e+00
   4.3935990e-01  1.2635876e+00  3.9146817e-01  1.2272689e+00
   8.3787918e-01  9.9160165e-01  7.9481214e-01  1.6897256e+00
   7.4399418e-01  3.2919204e-01  5.2493447e-01  3.2950503e-01
   1.4933692e+00  1.5650128e+00  1.2799625e+00  9.8361164e-01
   1.0681095e+00  1.8880401e+00  1.8672528e+00  3.2796979e-01
   5.0257903e-01  1.7563171e+00  2.1042066e+00  3.7377572e-01
   4.2568362e-01  1.7767229e+00  1.4523876e+00  1.7567095e+00
   6.2367213e-01  4.7599435e-01  1.7789660e+00  6.8594700e-01
   1.3427131e+00  1.8839315e+00  1.0661439e+00  1.2940278e+00
   3.3768713e-01  1.3807684e+00  7.5900745e-01  1.6956396e+00
   2.0743303e+00  3.6675417e-01  4.8627341e-01  1.5528969e+00
   1.7519867e+00  4.7349328e-01  1.1453029e+00  2.0272517e+00
   1.9587387e+00  1.1734478e+00  3.8770705e-01  8.5966104e-01
   1.0272892e+00  8.2455784e-01  4.5758200e-01  3.9815396e-01
   4.1471517e-01  3.3375198e-01  4.4084060e-01  1.9429796e+00
   2.0280344e+00  3.7945694e-01  4.0511888e-01  1.8020542e+00
   8.6645269e-01  1.3569838e+00  3.4319824e-01  1.1524248e+00
   1.3163671e+00  6.0827959e-01  3.5131538e-01  1.4146421e+00
   5.2590197e-01  1.3956046e+00  3.6183691e-01  1.0101560e+00
   7.2095871e-01  6.6735625e-01  6.9483310e-01  1.7692058e+00
   2.1168728e+00  4.9148750e-01  9.3997163e-01  4.6802425e-01
   5.6759906e-01  7.3099267e-01  5.8565664e-01  4.1940445e-01
   1.7231007e+00  4.1173869e-01  8.2906353e-01  1.5607929e+00
   1.6050364e+00  4.2631078e-01  1.0831858e+00  6.5714580e-01
   1.8599446e+00  1.4518970e+00  1.6368618e+00  3.6044204e-01
   1.4410195e+00  4.0314388e-01  8.9932507e-01  1.1414524e+00
   6.0585815e-01  1.1768873e+00  1.6473677e+00  1.1247184e+00
   2.0967927e+00  1.1866177e+00  2.0304694e+00  9.7868896e-01
   7.3498917e-01  1.2464864e+00  7.5854325e-01  1.1224377e+00
   1.2492946e+00  1.9725804e+00  5.3399706e-01  1.7244866e+00
   3.5914004e-02  3.7142911e+00  3.7212362e+00  4.5941100e+00
   4.1565638e+00  3.3382239e+00  4.1305299e+00  4.0765882e+00
   3.7226319e+00  4.2988238e+00  4.3845448e+00  4.7310200e+00
   4.9809709e+00  4.6883650e+00  3.4558535e+00  3.7426534e+00
   4.5157857e+00  4.5383615e+00  4.4718328e+00  3.9519734e+00
   4.3071165e+00  5.0751348e+00  4.0062461e+00  4.6919298e+00
   4.8228192e+00  4.1231804e+00  4.9849806e+00  4.5309744e+00
   4.7950211e+00  4.0537157e+00  3.9235396e+00  4.2875009e+00
   4.2539148e+00  3.3538599e+00  5.0135574e+00  3.2434072e+00
   4.8475623e+00  3.4404945e+00  4.0135221e+00  4.6933451e+00
   4.3442264e+00  3.3146944e+00  4.7181039e+00  4.7252736e+00
   3.9126616e+00  4.1121030e+00  4.5488148e+00  4.9833899e+00
   3.7025661e+00  4.3189483e+00  3.6012187e+00  3.2605834e+00
   4.4239073e+00  5.0116243e+00  3.7175360e+00  3.5323358e+00
   5.0953517e+00  3.6027894e+00  4.5807853e+00  4.0356383e+00
  -2.9818833e+00 -6.2167816e+00  1.9816026e-01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 01:12:52.807445
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.3087
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 01:12:52.811577
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9681.98
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 01:12:52.815183
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.2554
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 01:12:52.818868
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -866.09
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140410551778944
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140408021982344
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140408021982848
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140408021573816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140408021574320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140408021574824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb3eef36e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.563069
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.508738
grad_step = 000002, loss = 0.471272
grad_step = 000003, loss = 0.431744
grad_step = 000004, loss = 0.389736
grad_step = 000005, loss = 0.355711
grad_step = 000006, loss = 0.337065
grad_step = 000007, loss = 0.333237
grad_step = 000008, loss = 0.321814
grad_step = 000009, loss = 0.302356
grad_step = 000010, loss = 0.286953
grad_step = 000011, loss = 0.277654
grad_step = 000012, loss = 0.271256
grad_step = 000013, loss = 0.264772
grad_step = 000014, loss = 0.256447
grad_step = 000015, loss = 0.246005
grad_step = 000016, loss = 0.234479
grad_step = 000017, loss = 0.223485
grad_step = 000018, loss = 0.214184
grad_step = 000019, loss = 0.206227
grad_step = 000020, loss = 0.198406
grad_step = 000021, loss = 0.190585
grad_step = 000022, loss = 0.183038
grad_step = 000023, loss = 0.175269
grad_step = 000024, loss = 0.167334
grad_step = 000025, loss = 0.159850
grad_step = 000026, loss = 0.153074
grad_step = 000027, loss = 0.146677
grad_step = 000028, loss = 0.140377
grad_step = 000029, loss = 0.134228
grad_step = 000030, loss = 0.128256
grad_step = 000031, loss = 0.122268
grad_step = 000032, loss = 0.116230
grad_step = 000033, loss = 0.110470
grad_step = 000034, loss = 0.105272
grad_step = 000035, loss = 0.100478
grad_step = 000036, loss = 0.095753
grad_step = 000037, loss = 0.091014
grad_step = 000038, loss = 0.086406
grad_step = 000039, loss = 0.082042
grad_step = 000040, loss = 0.077875
grad_step = 000041, loss = 0.073771
grad_step = 000042, loss = 0.069728
grad_step = 000043, loss = 0.065913
grad_step = 000044, loss = 0.062432
grad_step = 000045, loss = 0.059167
grad_step = 000046, loss = 0.055907
grad_step = 000047, loss = 0.052603
grad_step = 000048, loss = 0.049412
grad_step = 000049, loss = 0.046486
grad_step = 000050, loss = 0.043787
grad_step = 000051, loss = 0.041168
grad_step = 000052, loss = 0.038572
grad_step = 000053, loss = 0.036072
grad_step = 000054, loss = 0.033739
grad_step = 000055, loss = 0.031555
grad_step = 000056, loss = 0.029477
grad_step = 000057, loss = 0.027493
grad_step = 000058, loss = 0.025614
grad_step = 000059, loss = 0.023840
grad_step = 000060, loss = 0.022159
grad_step = 000061, loss = 0.020568
grad_step = 000062, loss = 0.019075
grad_step = 000063, loss = 0.017680
grad_step = 000064, loss = 0.016379
grad_step = 000065, loss = 0.015169
grad_step = 000066, loss = 0.014041
grad_step = 000067, loss = 0.012979
grad_step = 000068, loss = 0.011985
grad_step = 000069, loss = 0.011074
grad_step = 000070, loss = 0.010241
grad_step = 000071, loss = 0.009464
grad_step = 000072, loss = 0.008742
grad_step = 000073, loss = 0.008083
grad_step = 000074, loss = 0.007487
grad_step = 000075, loss = 0.006934
grad_step = 000076, loss = 0.006425
grad_step = 000077, loss = 0.005968
grad_step = 000078, loss = 0.005557
grad_step = 000079, loss = 0.005180
grad_step = 000080, loss = 0.004836
grad_step = 000081, loss = 0.004529
grad_step = 000082, loss = 0.004251
grad_step = 000083, loss = 0.004003
grad_step = 000084, loss = 0.003782
grad_step = 000085, loss = 0.003587
grad_step = 000086, loss = 0.003408
grad_step = 000087, loss = 0.003248
grad_step = 000088, loss = 0.003110
grad_step = 000089, loss = 0.002989
grad_step = 000090, loss = 0.002881
grad_step = 000091, loss = 0.002786
grad_step = 000092, loss = 0.002701
grad_step = 000093, loss = 0.002628
grad_step = 000094, loss = 0.002565
grad_step = 000095, loss = 0.002510
grad_step = 000096, loss = 0.002460
grad_step = 000097, loss = 0.002417
grad_step = 000098, loss = 0.002381
grad_step = 000099, loss = 0.002348
grad_step = 000100, loss = 0.002320
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002295
grad_step = 000102, loss = 0.002274
grad_step = 000103, loss = 0.002254
grad_step = 000104, loss = 0.002237
grad_step = 000105, loss = 0.002222
grad_step = 000106, loss = 0.002208
grad_step = 000107, loss = 0.002195
grad_step = 000108, loss = 0.002183
grad_step = 000109, loss = 0.002172
grad_step = 000110, loss = 0.002162
grad_step = 000111, loss = 0.002152
grad_step = 000112, loss = 0.002143
grad_step = 000113, loss = 0.002133
grad_step = 000114, loss = 0.002124
grad_step = 000115, loss = 0.002116
grad_step = 000116, loss = 0.002111
grad_step = 000117, loss = 0.002111
grad_step = 000118, loss = 0.002116
grad_step = 000119, loss = 0.002105
grad_step = 000120, loss = 0.002081
grad_step = 000121, loss = 0.002066
grad_step = 000122, loss = 0.002071
grad_step = 000123, loss = 0.002078
grad_step = 000124, loss = 0.002072
grad_step = 000125, loss = 0.002052
grad_step = 000126, loss = 0.002033
grad_step = 000127, loss = 0.002027
grad_step = 000128, loss = 0.002032
grad_step = 000129, loss = 0.002045
grad_step = 000130, loss = 0.002066
grad_step = 000131, loss = 0.002068
grad_step = 000132, loss = 0.002048
grad_step = 000133, loss = 0.002004
grad_step = 000134, loss = 0.001986
grad_step = 000135, loss = 0.002002
grad_step = 000136, loss = 0.002029
grad_step = 000137, loss = 0.002049
grad_step = 000138, loss = 0.002024
grad_step = 000139, loss = 0.001984
grad_step = 000140, loss = 0.001960
grad_step = 000141, loss = 0.001969
grad_step = 000142, loss = 0.001997
grad_step = 000143, loss = 0.002013
grad_step = 000144, loss = 0.002006
grad_step = 000145, loss = 0.001968
grad_step = 000146, loss = 0.001941
grad_step = 000147, loss = 0.001941
grad_step = 000148, loss = 0.001959
grad_step = 000149, loss = 0.001979
grad_step = 000150, loss = 0.001977
grad_step = 000151, loss = 0.001960
grad_step = 000152, loss = 0.001933
grad_step = 000153, loss = 0.001919
grad_step = 000154, loss = 0.001922
grad_step = 000155, loss = 0.001937
grad_step = 000156, loss = 0.001958
grad_step = 000157, loss = 0.001970
grad_step = 000158, loss = 0.001974
grad_step = 000159, loss = 0.001949
grad_step = 000160, loss = 0.001919
grad_step = 000161, loss = 0.001897
grad_step = 000162, loss = 0.001899
grad_step = 000163, loss = 0.001916
grad_step = 000164, loss = 0.001927
grad_step = 000165, loss = 0.001930
grad_step = 000166, loss = 0.001912
grad_step = 000167, loss = 0.001893
grad_step = 000168, loss = 0.001878
grad_step = 000169, loss = 0.001876
grad_step = 000170, loss = 0.001883
grad_step = 000171, loss = 0.001895
grad_step = 000172, loss = 0.001912
grad_step = 000173, loss = 0.001923
grad_step = 000174, loss = 0.001933
grad_step = 000175, loss = 0.001914
grad_step = 000176, loss = 0.001888
grad_step = 000177, loss = 0.001859
grad_step = 000178, loss = 0.001848
grad_step = 000179, loss = 0.001856
grad_step = 000180, loss = 0.001869
grad_step = 000181, loss = 0.001875
grad_step = 000182, loss = 0.001862
grad_step = 000183, loss = 0.001844
grad_step = 000184, loss = 0.001830
grad_step = 000185, loss = 0.001826
grad_step = 000186, loss = 0.001831
grad_step = 000187, loss = 0.001840
grad_step = 000188, loss = 0.001854
grad_step = 000189, loss = 0.001868
grad_step = 000190, loss = 0.001889
grad_step = 000191, loss = 0.001890
grad_step = 000192, loss = 0.001884
grad_step = 000193, loss = 0.001844
grad_step = 000194, loss = 0.001808
grad_step = 000195, loss = 0.001791
grad_step = 000196, loss = 0.001799
grad_step = 000197, loss = 0.001817
grad_step = 000198, loss = 0.001818
grad_step = 000199, loss = 0.001806
grad_step = 000200, loss = 0.001783
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001769
grad_step = 000202, loss = 0.001768
grad_step = 000203, loss = 0.001776
grad_step = 000204, loss = 0.001790
grad_step = 000205, loss = 0.001800
grad_step = 000206, loss = 0.001815
grad_step = 000207, loss = 0.001816
grad_step = 000208, loss = 0.001821
grad_step = 000209, loss = 0.001799
grad_step = 000210, loss = 0.001778
grad_step = 000211, loss = 0.001747
grad_step = 000212, loss = 0.001727
grad_step = 000213, loss = 0.001721
grad_step = 000214, loss = 0.001726
grad_step = 000215, loss = 0.001738
grad_step = 000216, loss = 0.001753
grad_step = 000217, loss = 0.001778
grad_step = 000218, loss = 0.001796
grad_step = 000219, loss = 0.001830
grad_step = 000220, loss = 0.001815
grad_step = 000221, loss = 0.001791
grad_step = 000222, loss = 0.001724
grad_step = 000223, loss = 0.001685
grad_step = 000224, loss = 0.001688
grad_step = 000225, loss = 0.001714
grad_step = 000226, loss = 0.001733
grad_step = 000227, loss = 0.001713
grad_step = 000228, loss = 0.001684
grad_step = 000229, loss = 0.001661
grad_step = 000230, loss = 0.001657
grad_step = 000231, loss = 0.001670
grad_step = 000232, loss = 0.001691
grad_step = 000233, loss = 0.001723
grad_step = 000234, loss = 0.001754
grad_step = 000235, loss = 0.001796
grad_step = 000236, loss = 0.001795
grad_step = 000237, loss = 0.001774
grad_step = 000238, loss = 0.001697
grad_step = 000239, loss = 0.001636
grad_step = 000240, loss = 0.001624
grad_step = 000241, loss = 0.001655
grad_step = 000242, loss = 0.001688
grad_step = 000243, loss = 0.001676
grad_step = 000244, loss = 0.001641
grad_step = 000245, loss = 0.001607
grad_step = 000246, loss = 0.001603
grad_step = 000247, loss = 0.001622
grad_step = 000248, loss = 0.001646
grad_step = 000249, loss = 0.001668
grad_step = 000250, loss = 0.001670
grad_step = 000251, loss = 0.001663
grad_step = 000252, loss = 0.001637
grad_step = 000253, loss = 0.001611
grad_step = 000254, loss = 0.001586
grad_step = 000255, loss = 0.001570
grad_step = 000256, loss = 0.001564
grad_step = 000257, loss = 0.001565
grad_step = 000258, loss = 0.001573
grad_step = 000259, loss = 0.001588
grad_step = 000260, loss = 0.001616
grad_step = 000261, loss = 0.001659
grad_step = 000262, loss = 0.001743
grad_step = 000263, loss = 0.001807
grad_step = 000264, loss = 0.001864
grad_step = 000265, loss = 0.001723
grad_step = 000266, loss = 0.001574
grad_step = 000267, loss = 0.001542
grad_step = 000268, loss = 0.001626
grad_step = 000269, loss = 0.001651
grad_step = 000270, loss = 0.001556
grad_step = 000271, loss = 0.001531
grad_step = 000272, loss = 0.001590
grad_step = 000273, loss = 0.001579
grad_step = 000274, loss = 0.001521
grad_step = 000275, loss = 0.001515
grad_step = 000276, loss = 0.001553
grad_step = 000277, loss = 0.001556
grad_step = 000278, loss = 0.001511
grad_step = 000279, loss = 0.001490
grad_step = 000280, loss = 0.001510
grad_step = 000281, loss = 0.001530
grad_step = 000282, loss = 0.001524
grad_step = 000283, loss = 0.001494
grad_step = 000284, loss = 0.001471
grad_step = 000285, loss = 0.001467
grad_step = 000286, loss = 0.001478
grad_step = 000287, loss = 0.001493
grad_step = 000288, loss = 0.001500
grad_step = 000289, loss = 0.001502
grad_step = 000290, loss = 0.001490
grad_step = 000291, loss = 0.001479
grad_step = 000292, loss = 0.001463
grad_step = 000293, loss = 0.001450
grad_step = 000294, loss = 0.001438
grad_step = 000295, loss = 0.001429
grad_step = 000296, loss = 0.001422
grad_step = 000297, loss = 0.001417
grad_step = 000298, loss = 0.001413
grad_step = 000299, loss = 0.001409
grad_step = 000300, loss = 0.001405
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001404
grad_step = 000302, loss = 0.001406
grad_step = 000303, loss = 0.001419
grad_step = 000304, loss = 0.001466
grad_step = 000305, loss = 0.001588
grad_step = 000306, loss = 0.001912
grad_step = 000307, loss = 0.002113
grad_step = 000308, loss = 0.002231
grad_step = 000309, loss = 0.001493
grad_step = 000310, loss = 0.001532
grad_step = 000311, loss = 0.001905
grad_step = 000312, loss = 0.001445
grad_step = 000313, loss = 0.001603
grad_step = 000314, loss = 0.001707
grad_step = 000315, loss = 0.001408
grad_step = 000316, loss = 0.001713
grad_step = 000317, loss = 0.001478
grad_step = 000318, loss = 0.001503
grad_step = 000319, loss = 0.001563
grad_step = 000320, loss = 0.001368
grad_step = 000321, loss = 0.001521
grad_step = 000322, loss = 0.001375
grad_step = 000323, loss = 0.001422
grad_step = 000324, loss = 0.001441
grad_step = 000325, loss = 0.001337
grad_step = 000326, loss = 0.001441
grad_step = 000327, loss = 0.001353
grad_step = 000328, loss = 0.001343
grad_step = 000329, loss = 0.001393
grad_step = 000330, loss = 0.001309
grad_step = 000331, loss = 0.001325
grad_step = 000332, loss = 0.001350
grad_step = 000333, loss = 0.001282
grad_step = 000334, loss = 0.001294
grad_step = 000335, loss = 0.001314
grad_step = 000336, loss = 0.001267
grad_step = 000337, loss = 0.001257
grad_step = 000338, loss = 0.001280
grad_step = 000339, loss = 0.001265
grad_step = 000340, loss = 0.001228
grad_step = 000341, loss = 0.001235
grad_step = 000342, loss = 0.001251
grad_step = 000343, loss = 0.001233
grad_step = 000344, loss = 0.001206
grad_step = 000345, loss = 0.001198
grad_step = 000346, loss = 0.001208
grad_step = 000347, loss = 0.001213
grad_step = 000348, loss = 0.001200
grad_step = 000349, loss = 0.001181
grad_step = 000350, loss = 0.001168
grad_step = 000351, loss = 0.001165
grad_step = 000352, loss = 0.001169
grad_step = 000353, loss = 0.001171
grad_step = 000354, loss = 0.001167
grad_step = 000355, loss = 0.001155
grad_step = 000356, loss = 0.001144
grad_step = 000357, loss = 0.001136
grad_step = 000358, loss = 0.001131
grad_step = 000359, loss = 0.001131
grad_step = 000360, loss = 0.001130
grad_step = 000361, loss = 0.001129
grad_step = 000362, loss = 0.001126
grad_step = 000363, loss = 0.001122
grad_step = 000364, loss = 0.001115
grad_step = 000365, loss = 0.001110
grad_step = 000366, loss = 0.001104
grad_step = 000367, loss = 0.001098
grad_step = 000368, loss = 0.001093
grad_step = 000369, loss = 0.001088
grad_step = 000370, loss = 0.001083
grad_step = 000371, loss = 0.001079
grad_step = 000372, loss = 0.001075
grad_step = 000373, loss = 0.001072
grad_step = 000374, loss = 0.001070
grad_step = 000375, loss = 0.001071
grad_step = 000376, loss = 0.001075
grad_step = 000377, loss = 0.001086
grad_step = 000378, loss = 0.001104
grad_step = 000379, loss = 0.001141
grad_step = 000380, loss = 0.001170
grad_step = 000381, loss = 0.001219
grad_step = 000382, loss = 0.001183
grad_step = 000383, loss = 0.001129
grad_step = 000384, loss = 0.001044
grad_step = 000385, loss = 0.001023
grad_step = 000386, loss = 0.001065
grad_step = 000387, loss = 0.001092
grad_step = 000388, loss = 0.001071
grad_step = 000389, loss = 0.001020
grad_step = 000390, loss = 0.001000
grad_step = 000391, loss = 0.001014
grad_step = 000392, loss = 0.001041
grad_step = 000393, loss = 0.001065
grad_step = 000394, loss = 0.001058
grad_step = 000395, loss = 0.001045
grad_step = 000396, loss = 0.001018
grad_step = 000397, loss = 0.000999
grad_step = 000398, loss = 0.000977
grad_step = 000399, loss = 0.000963
grad_step = 000400, loss = 0.000957
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000958
grad_step = 000402, loss = 0.000961
grad_step = 000403, loss = 0.000964
grad_step = 000404, loss = 0.000975
grad_step = 000405, loss = 0.000995
grad_step = 000406, loss = 0.001037
grad_step = 000407, loss = 0.001061
grad_step = 000408, loss = 0.001131
grad_step = 000409, loss = 0.001132
grad_step = 000410, loss = 0.001130
grad_step = 000411, loss = 0.001014
grad_step = 000412, loss = 0.000935
grad_step = 000413, loss = 0.000930
grad_step = 000414, loss = 0.000975
grad_step = 000415, loss = 0.001007
grad_step = 000416, loss = 0.000960
grad_step = 000417, loss = 0.000922
grad_step = 000418, loss = 0.000912
grad_step = 000419, loss = 0.000935
grad_step = 000420, loss = 0.000959
grad_step = 000421, loss = 0.000915
grad_step = 000422, loss = 0.000892
grad_step = 000423, loss = 0.000894
grad_step = 000424, loss = 0.000901
grad_step = 000425, loss = 0.000915
grad_step = 000426, loss = 0.000921
grad_step = 000427, loss = 0.000924
grad_step = 000428, loss = 0.000912
grad_step = 000429, loss = 0.000904
grad_step = 000430, loss = 0.000893
grad_step = 000431, loss = 0.000891
grad_step = 000432, loss = 0.000870
grad_step = 000433, loss = 0.000862
grad_step = 000434, loss = 0.000857
grad_step = 000435, loss = 0.000851
grad_step = 000436, loss = 0.000845
grad_step = 000437, loss = 0.000840
grad_step = 000438, loss = 0.000839
grad_step = 000439, loss = 0.000839
grad_step = 000440, loss = 0.000840
grad_step = 000441, loss = 0.000840
grad_step = 000442, loss = 0.000843
grad_step = 000443, loss = 0.000858
grad_step = 000444, loss = 0.000907
grad_step = 000445, loss = 0.000965
grad_step = 000446, loss = 0.001139
grad_step = 000447, loss = 0.001250
grad_step = 000448, loss = 0.001397
grad_step = 000449, loss = 0.001055
grad_step = 000450, loss = 0.000829
grad_step = 000451, loss = 0.000909
grad_step = 000452, loss = 0.001014
grad_step = 000453, loss = 0.000918
grad_step = 000454, loss = 0.000820
grad_step = 000455, loss = 0.000924
grad_step = 000456, loss = 0.000922
grad_step = 000457, loss = 0.000818
grad_step = 000458, loss = 0.000894
grad_step = 000459, loss = 0.000901
grad_step = 000460, loss = 0.000813
grad_step = 000461, loss = 0.000871
grad_step = 000462, loss = 0.000866
grad_step = 000463, loss = 0.000844
grad_step = 000464, loss = 0.000885
grad_step = 000465, loss = 0.000816
grad_step = 000466, loss = 0.000975
grad_step = 000467, loss = 0.000916
grad_step = 000468, loss = 0.000914
grad_step = 000469, loss = 0.000825
grad_step = 000470, loss = 0.000878
grad_step = 000471, loss = 0.000875
grad_step = 000472, loss = 0.000868
grad_step = 000473, loss = 0.000862
grad_step = 000474, loss = 0.000835
grad_step = 000475, loss = 0.000814
grad_step = 000476, loss = 0.000832
grad_step = 000477, loss = 0.000817
grad_step = 000478, loss = 0.000835
grad_step = 000479, loss = 0.000843
grad_step = 000480, loss = 0.000808
grad_step = 000481, loss = 0.000810
grad_step = 000482, loss = 0.000793
grad_step = 000483, loss = 0.000778
grad_step = 000484, loss = 0.000782
grad_step = 000485, loss = 0.000781
grad_step = 000486, loss = 0.000782
grad_step = 000487, loss = 0.000783
grad_step = 000488, loss = 0.000774
grad_step = 000489, loss = 0.000770
grad_step = 000490, loss = 0.000763
grad_step = 000491, loss = 0.000752
grad_step = 000492, loss = 0.000753
grad_step = 000493, loss = 0.000749
grad_step = 000494, loss = 0.000747
grad_step = 000495, loss = 0.000749
grad_step = 000496, loss = 0.000746
grad_step = 000497, loss = 0.000745
grad_step = 000498, loss = 0.000741
grad_step = 000499, loss = 0.000735
grad_step = 000500, loss = 0.000733
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000727
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

  date_run                              2020-05-11 01:13:15.396448
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.247642
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 01:13:15.402507
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.160258
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 01:13:15.409055
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.129556
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 01:13:15.414566
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.43517
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
0   2020-05-11 01:12:44.405290  ...    mean_absolute_error
1   2020-05-11 01:12:44.410472  ...     mean_squared_error
2   2020-05-11 01:12:44.414385  ...  median_absolute_error
3   2020-05-11 01:12:44.417753  ...               r2_score
4   2020-05-11 01:12:52.807445  ...    mean_absolute_error
5   2020-05-11 01:12:52.811577  ...     mean_squared_error
6   2020-05-11 01:12:52.815183  ...  median_absolute_error
7   2020-05-11 01:12:52.818868  ...               r2_score
8   2020-05-11 01:13:15.396448  ...    mean_absolute_error
9   2020-05-11 01:13:15.402507  ...     mean_squared_error
10  2020-05-11 01:13:15.409055  ...  median_absolute_error
11  2020-05-11 01:13:15.414566  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 307094.59it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 396368.16it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 548554.64it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 774929.11it/s] 78%|███████▊  | 7684096/9912422 [00:00<00:02, 1095726.21it/s]9920512it [00:00, 10246156.90it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 145493.47it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 306748.69it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 396251.61it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 548185.90it/s]1654784it [00:00, 2770129.36it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 26178.01it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7feaea576780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea87cbfa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7feaea576e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7feaea52de48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea87cbb080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea913db4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea87cbfa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7feaea576f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea9cf25cc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7feaea576f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fea9cf25cc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb9ba4971d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4bb121c556761ed1fd4d98bd5bdddcd4cc611fe375b62acf422ab7d91e37b7da
  Stored in directory: /tmp/pip-ephem-wheel-cache-mzcnsy19/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb9aa19be48> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 38s
  122880/17464789 [..............................] - ETA: 27s
  245760/17464789 [..............................] - ETA: 17s
  524288/17464789 [..............................] - ETA: 10s
 1064960/17464789 [>.............................] - ETA: 5s 
 2154496/17464789 [==>...........................] - ETA: 3s
 4317184/17464789 [======>.......................] - ETA: 1s
 7430144/17464789 [===========>..................] - ETA: 0s
10543104/17464789 [=================>............] - ETA: 0s
13672448/17464789 [======================>.......] - ETA: 0s
16769024/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 01:14:44.504064: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 01:14:44.508488: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-11 01:14:44.508643: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d4accf8ab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 01:14:44.508658: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6590 - accuracy: 0.5005 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5874 - accuracy: 0.5052
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5352 - accuracy: 0.5086
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.4807 - accuracy: 0.5121
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5337 - accuracy: 0.5087
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5332 - accuracy: 0.5087
11000/25000 [============>.................] - ETA: 4s - loss: 7.5593 - accuracy: 0.5070
12000/25000 [=============>................] - ETA: 3s - loss: 7.5925 - accuracy: 0.5048
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6141 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6130 - accuracy: 0.5035
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6224 - accuracy: 0.5029
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6232 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6421 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6840 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 9s 341us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 01:15:00.021586
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 01:15:00.021586  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 01:15:06.617128: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 01:15:06.623992: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-11 01:15:06.624186: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555a9144d000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 01:15:06.624205: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd985ebad30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9731 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.8614 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd97b261f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5286 - accuracy: 0.5090 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6912 - accuracy: 0.4984
11000/25000 [============>.................] - ETA: 4s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 3s - loss: 7.6960 - accuracy: 0.4981
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7067 - accuracy: 0.4974
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6995 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7011 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6844 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6907 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 344us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd936fc88d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<35:48:28, 6.69kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:01<25:04:17, 9.55kB/s] .vector_cache/glove.6B.zip:   1%|          | 6.72M/862M [00:01<17:25:25, 13.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.6M/862M [00:01<12:03:22, 19.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:01<8:20:53, 27.8kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<5:46:32, 39.8kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:01<3:59:23, 56.8kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<2:46:46, 81.0kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<1:56:35, 115kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<12:26:58, 18.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<8:42:17, 25.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<11:03:47, 20.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<7:44:01, 28.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<11:07:04, 20.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<7:46:20, 28.6kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<11:02:20, 20.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<7:43:05, 28.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:41:57, 20.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<7:28:51, 29.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:33:53, 20.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<7:23:12, 29.9kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:28:50, 21.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<7:19:36, 30.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:46:32, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<7:31:58, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:50:30, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<7:34:45, 28.9kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:43:51, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<7:30:04, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:49:37, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<7:34:12, 28.8kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:21:55, 21.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<7:14:50, 30.0kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:15:51, 21.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<7:10:30, 30.2kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:34:34, 20.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<7:23:34, 29.2kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:39:16, 20.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<7:26:58, 28.9kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:12:24, 21.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<7:08:05, 30.1kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:30:35, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<7:20:52, 29.2kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:10:39, 21.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<7:06:51, 30.0kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:28:03, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<7:19:00, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:30:25, 20.3kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<7:20:40, 28.9kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:24:47, 20.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<7:16:42, 29.1kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:30:15, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<7:20:31, 28.8kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:28:42, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<7:19:29, 28.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:16:11, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<7:10:41, 29.3kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:24:29, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<7:16:31, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:16:13, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<7:10:47, 29.1kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:02:05, 20.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<7:00:54, 29.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<9:55:57, 21.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<6:56:32, 29.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:12:30, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<7:08:06, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:14:05, 20.2kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<7:09:13, 28.9kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:08:21, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<7:05:10, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:15:45, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<7:10:21, 28.6kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:12:52, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<7:08:21, 28.7kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:05:28, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<7:03:14, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<9:50:22, 20.8kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<6:52:36, 29.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:05:42, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<7:03:19, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:05:09, 20.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<7:02:57, 28.7kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:00:16, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<6:59:37, 28.9kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:35:57, 21.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<6:42:36, 30.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:32:29, 21.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<6:40:06, 30.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:49:14, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<6:51:48, 29.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:51:29, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<6:53:24, 29.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:44:53, 20.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<6:48:44, 29.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:53:07, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<6:54:30, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:51:31, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<6:53:24, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:44:58, 20.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<6:48:47, 29.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:50:00, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<6:52:19, 28.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:45:01, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<6:48:49, 28.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:46:44, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<6:50:02, 28.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:41:31, 20.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<6:46:21, 28.8kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:45:00, 20.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<6:48:48, 28.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:39:56, 20.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<6:45:17, 28.7kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:32:28, 20.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<6:40:01, 29.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:38:14, 20.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<6:44:08, 28.6kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:14:00, 20.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<6:27:08, 29.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:26:25, 20.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<6:35:49, 29.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:26:42, 20.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<6:36:01, 28.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:20:46, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<6:31:55, 29.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:07:08, 20.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<6:22:18, 29.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:23:03, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<6:33:26, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:21:41, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<6:32:29, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:14:59, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<6:27:48, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:12:29, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<6:26:02, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<9:18:52, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<6:30:30, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:14:16, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<6:27:20, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<8:59:50, 20.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<6:17:10, 29.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:13:16, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<6:26:34, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:10:51, 20.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<6:24:54, 28.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<9:04:06, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<6:20:10, 29.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:01:31, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<6:18:20, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<9:07:20, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<6:22:25, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<9:03:08, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<6:19:27, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<9:05:12, 20.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<6:20:55, 28.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<9:00:33, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<6:17:39, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<9:02:05, 20.0kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<6:18:44, 28.6kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:55:51, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<6:14:21, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:58:17, 20.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<6:16:08, 28.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:35:52, 20.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<6:00:24, 29.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:46:08, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<6:07:36, 29.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:38:53, 20.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<6:02:29, 29.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:47:02, 20.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<6:08:18, 28.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:17:51, 21.3kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<5:47:48, 30.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:34:07, 20.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<5:59:11, 29.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:32:01, 20.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<5:57:41, 29.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:41:05, 20.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<6:04:05, 28.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:21:32, 20.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<5:50:25, 29.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:16:14, 21.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<5:46:39, 30.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:30:21, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<5:56:31, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:30:32, 20.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<5:56:37, 29.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:33:58, 20.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<5:59:02, 28.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:31:38, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<5:57:24, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:25:26, 20.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<5:53:02, 29.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:30:41, 20.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<5:56:43, 28.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:26:14, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<5:53:35, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:27:28, 20.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<5:54:28, 28.6kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:22:55, 20.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<5:51:16, 28.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:24:03, 20.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<5:52:04, 28.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:20:59, 20.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<5:49:59, 28.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<7:58:07, 21.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<5:34:03, 29.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<7:43:20, 21.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<5:23:38, 30.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:01:13, 20.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<5:36:11, 29.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<7:47:56, 21.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<5:26:54, 30.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<7:43:14, 21.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<5:23:32, 30.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<8:01:29, 20.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<5:36:17, 29.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<8:03:57, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<5:38:00, 28.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<7:59:47, 20.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<5:35:10, 29.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<7:41:13, 21.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<5:22:07, 30.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:54:10, 20.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<5:31:09, 29.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:57:59, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<5:33:50, 28.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:53:33, 20.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<5:30:42, 29.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:57:25, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<5:33:24, 28.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:57:08, 20.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<5:33:13, 28.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:50:41, 20.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<5:28:41, 28.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:53:14, 20.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<5:30:27, 28.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:52:30, 20.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<5:29:58, 28.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:45:38, 20.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<5:25:09, 28.9kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:48:12, 20.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<5:26:56, 28.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:45:44, 20.1kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<5:25:14, 28.7kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:39:59, 20.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<5:21:11, 28.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:43:09, 20.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<5:23:23, 28.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:41:00, 20.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<5:21:54, 28.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:36:33, 20.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<5:18:51, 28.8kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:16:10, 21.1kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<5:04:37, 30.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:12:15, 21.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<5:01:52, 30.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:10:39, 21.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<5:00:41, 30.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:26:26, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<5:11:46, 29.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:12:01, 20.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<5:01:38, 29.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:21:14, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<5:08:04, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:23:34, 20.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<5:09:45, 28.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:04:10, 21.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<4:56:08, 30.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<7:15:57, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<5:04:21, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<7:18:15, 20.2kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<5:05:59, 28.9kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<7:13:27, 20.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<5:02:36, 29.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<7:17:57, 20.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<5:05:48, 28.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<7:01:17, 20.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<4:54:06, 29.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<7:10:00, 20.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<5:00:11, 29.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<7:10:59, 20.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<5:00:52, 28.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<7:05:25, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<4:56:58, 29.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<7:09:24, 20.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<4:59:44, 28.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<7:07:53, 20.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<4:58:45, 28.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<6:47:30, 21.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<4:44:27, 29.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<6:58:03, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<4:51:49, 29.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<6:59:43, 20.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<4:52:59, 28.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:54:45, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<4:49:33, 29.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<6:44:26, 20.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<4:42:17, 29.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:54:11, 20.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<4:49:06, 28.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<6:54:19, 20.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<4:49:12, 28.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<6:47:25, 20.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<4:44:21, 29.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<6:51:29, 20.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<4:47:14, 28.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<6:37:07, 20.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<4:37:10, 29.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<6:42:36, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<4:41:00, 29.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<6:39:33, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<4:38:52, 29.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<6:38:26, 20.4kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<4:38:03, 29.1kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:40:32, 20.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<4:39:31, 28.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<6:40:52, 20.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<4:39:46, 28.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<6:33:08, 20.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<4:34:20, 29.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<6:36:49, 20.1kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<4:36:55, 28.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<6:35:32, 20.1kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<4:36:05, 28.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<6:15:23, 21.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<4:21:57, 30.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<6:25:27, 20.5kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<4:28:58, 29.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<6:27:06, 20.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<4:30:08, 28.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<6:21:27, 20.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<4:26:10, 29.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<6:25:39, 20.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<4:29:05, 28.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<6:24:35, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<4:28:21, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<6:18:40, 20.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<4:24:12, 29.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<6:21:36, 20.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<4:26:14, 28.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<6:21:05, 20.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<4:25:57, 28.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<6:02:12, 21.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<4:12:45, 30.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<5:58:50, 21.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<4:10:21, 30.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<6:10:06, 20.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<4:18:13, 29.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<6:10:46, 20.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<4:18:41, 28.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<6:04:46, 20.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<4:14:28, 29.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<6:08:30, 20.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<4:17:04, 28.8kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<6:07:27, 20.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<4:16:20, 28.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<6:03:57, 20.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<4:13:53, 28.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<6:00:30, 20.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<4:11:31, 29.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<5:50:32, 20.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<4:04:30, 29.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<5:58:45, 20.2kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<4:10:14, 28.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:58:35, 20.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<4:10:10, 28.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<5:40:58, 21.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<3:57:49, 30.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<5:50:12, 20.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<4:04:18, 29.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<5:40:17, 20.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<3:57:20, 29.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<5:48:23, 20.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<4:02:59, 29.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:48:31, 20.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<4:03:05, 28.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<5:44:35, 20.4kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<4:00:21, 29.0kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:35:49, 20.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<3:54:11, 29.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:42:57, 20.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<3:59:10, 28.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:42:28, 20.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<3:58:50, 28.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:39:27, 20.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<3:56:43, 28.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<5:36:10, 20.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<3:54:25, 29.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<5:38:45, 20.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<3:56:13, 28.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<5:37:03, 20.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<3:55:02, 28.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<5:31:31, 20.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<3:51:12, 29.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:20:06, 20.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<3:43:11, 29.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<5:27:20, 20.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<3:48:14, 29.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<5:28:01, 20.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<3:48:43, 28.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<5:21:30, 20.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<3:44:09, 29.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<5:25:12, 20.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<3:46:46, 28.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<5:12:49, 20.9kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<3:38:05, 29.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<5:19:04, 20.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<3:42:26, 29.0kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<5:18:35, 20.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<3:42:06, 28.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<5:16:29, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<3:40:41, 28.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<4:58:46, 21.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<3:28:16, 30.5kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<5:07:59, 20.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<3:34:41, 29.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<5:09:39, 20.4kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<3:35:54, 29.1kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<4:57:02, 21.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<3:27:02, 30.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<5:04:23, 20.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<3:32:09, 29.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<5:04:26, 20.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<3:32:13, 29.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<4:56:23, 20.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<3:26:34, 29.7kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<5:02:23, 20.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<3:30:44, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<5:00:45, 20.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<3:29:35, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<5:01:51, 20.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<3:30:24, 28.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<4:47:22, 21.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<3:20:18, 30.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<4:43:52, 21.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<3:17:48, 30.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<4:52:38, 20.4kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<3:23:54, 29.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<4:53:30, 20.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<3:24:31, 28.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<4:49:37, 20.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<3:21:47, 29.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<4:51:59, 20.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<3:23:26, 28.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<4:48:57, 20.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<3:21:20, 28.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<4:39:42, 20.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<3:14:51, 29.5kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<4:45:33, 20.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<3:18:55, 28.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<4:45:05, 20.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<3:18:36, 28.6kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<4:39:52, 20.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<3:14:57, 29.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<4:40:53, 20.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<3:15:41, 28.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:28:44, 20.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<3:07:11, 29.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<4:34:15, 20.4kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<3:11:02, 29.0kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<4:33:40, 20.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<3:10:37, 28.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<4:32:40, 20.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<3:09:06, 28.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<2:15:07, 40.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<1:34:53, 57.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:06:21, 81.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<2:49:59, 31.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:58:14, 45.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:23:05, 64.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<58:07, 91.2kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<2:41:44, 32.8kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<1:52:30, 46.5kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<1:19:05, 66.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<55:19, 94.0kB/s]  .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<2:37:53, 32.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<1:49:47, 46.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<1:17:09, 66.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<53:58, 94.4kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<2:33:49, 33.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<1:46:57, 47.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<1:15:09, 66.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<52:36, 94.8kB/s]  .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<2:22:37, 35.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<1:39:09, 49.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<1:09:42, 70.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<48:45, 100kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<2:25:20, 33.6kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<1:40:59, 47.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<1:10:58, 67.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<49:37, 96.3kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<2:23:29, 33.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<1:39:40, 47.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<1:10:02, 67.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<49:05, 95.2kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<1:55:06, 40.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<1:20:03, 57.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<56:20, 81.6kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<39:25, 116kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<2:11:33, 34.7kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<1:31:21, 49.3kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<1:04:13, 70.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<44:55, 99.3kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<2:05:00, 35.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<1:26:47, 50.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<1:01:00, 71.9kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<42:38, 102kB/s]   .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<2:09:06, 33.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:29:34, 47.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<1:02:56, 68.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<44:01, 96.6kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<2:00:03, 35.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<1:23:17, 50.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<58:47, 71.1kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<41:04, 101kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<1:54:57, 36.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<1:19:43, 51.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<56:11, 72.5kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<39:13, 103kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<1:59:43, 33.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<1:22:57, 47.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<58:21, 68.0kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<40:44, 96.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:57:32, 33.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<1:21:23, 47.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<57:22, 67.4kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<40:01, 95.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<1:54:37, 33.4kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<1:19:20, 47.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<55:43, 67.5kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<38:55, 95.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<1:46:05, 35.2kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<1:13:25, 49.9kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<51:49, 70.6kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<36:08, 100kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<1:46:40, 34.0kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<1:13:46, 48.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<51:57, 68.4kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<36:13, 97.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<1:44:49, 33.6kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:12:26, 47.6kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<50:58, 67.6kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<35:32, 96.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<1:41:52, 33.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<1:10:21, 47.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<49:26, 67.6kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<34:28, 96.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<1:39:06, 33.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:08:23, 47.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<47:50, 67.5kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<33:25, 95.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:38:44, 32.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<1:08:05, 46.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<47:47, 65.5kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<33:18, 93.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<1:34:25, 32.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<1:05:03, 46.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<45:46, 66.1kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<31:52, 94.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:30:01, 33.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<1:01:59, 47.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<43:30, 67.1kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<30:19, 95.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:22:18, 35.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<56:39, 49.8kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<39:52, 70.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<27:44, 100kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:22:44, 33.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<56:53, 47.8kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<39:55, 67.9kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<27:48, 96.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:16:06, 35.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<52:16, 50.0kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<36:41, 71.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<25:32, 101kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<1:16:40, 33.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<52:36, 47.7kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<36:54, 67.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<25:41, 96.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<1:09:59, 35.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<47:58, 50.1kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<33:45, 71.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<23:26, 101kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<1:09:54, 33.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<47:50, 48.0kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<33:33, 68.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<23:20, 96.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:04:07, 35.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<43:50, 50.0kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<30:45, 71.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<21:21, 101kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<1:03:52, 33.8kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<43:35, 47.9kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<30:34, 68.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<21:12, 96.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<1:01:51, 33.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<42:08, 47.1kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<29:42, 66.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<20:34, 94.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<54:42, 35.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<37:13, 50.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<26:06, 71.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<18:04, 102kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<55:01, 33.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<37:20, 47.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<26:10, 67.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<18:06, 95.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<51:01, 34.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<34:32, 48.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<24:12, 68.6kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<16:44, 97.5kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<48:54, 33.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<33:01, 47.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<23:07, 67.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<15:58, 95.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<43:16, 35.3kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<29:08, 50.1kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<20:24, 71.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<14:05, 101kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<39:41, 35.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<26:38, 50.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<18:39, 72.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<12:51, 103kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<36:51, 35.8kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<24:38, 50.7kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<17:17, 71.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<11:51, 102kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<54:41, 22.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<36:15, 31.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<25:12, 45.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<17:18, 64.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<38:12, 29.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<25:14, 41.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<17:37, 58.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<12:03, 83.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<29:59, 33.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<19:41, 47.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<13:44, 67.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<09:22, 95.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<27:00, 33.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<17:34, 47.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<12:15, 67.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<08:19, 95.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<22:40, 35.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<14:36, 49.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<10:10, 70.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<06:52, 100kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<20:36, 33.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<13:04, 47.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<09:08, 67.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<06:06, 95.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<16:32, 35.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<10:17, 50.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<07:12, 71.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<04:45, 101kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<14:11, 33.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<08:34, 47.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<05:55, 68.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<03:52, 96.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<11:12, 33.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<06:27, 47.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<04:27, 67.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<02:49, 95.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<07:38, 35.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<04:01, 50.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<02:45, 70.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<01:37, 101kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<04:51, 33.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<01:59, 48.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<01:17, 68.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:36, 96.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:47, 33.3kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 696/400000 [00:00<00:57, 6959.69it/s]  0%|          | 1450/400000 [00:00<00:55, 7122.11it/s]  1%|          | 2216/400000 [00:00<00:54, 7274.11it/s]  1%|          | 2995/400000 [00:00<00:53, 7418.76it/s]  1%|          | 3743/400000 [00:00<00:53, 7434.44it/s]  1%|          | 4497/400000 [00:00<00:52, 7465.13it/s]  1%|▏         | 5251/400000 [00:00<00:52, 7486.92it/s]  1%|▏         | 5944/400000 [00:00<00:53, 7304.22it/s]  2%|▏         | 6697/400000 [00:00<00:53, 7369.43it/s]  2%|▏         | 7418/400000 [00:01<00:53, 7319.48it/s]  2%|▏         | 8132/400000 [00:01<00:54, 7126.61it/s]  2%|▏         | 8894/400000 [00:01<00:53, 7266.57it/s]  2%|▏         | 9613/400000 [00:01<00:54, 7208.60it/s]  3%|▎         | 10329/400000 [00:01<00:54, 7170.95it/s]  3%|▎         | 11043/400000 [00:01<00:54, 7118.61it/s]  3%|▎         | 11753/400000 [00:01<00:55, 6964.59it/s]  3%|▎         | 12507/400000 [00:01<00:54, 7126.03it/s]  3%|▎         | 13255/400000 [00:01<00:53, 7220.03it/s]  3%|▎         | 13978/400000 [00:01<00:53, 7186.52it/s]  4%|▎         | 14702/400000 [00:02<00:53, 7190.64it/s]  4%|▍         | 15446/400000 [00:02<00:52, 7262.35it/s]  4%|▍         | 16214/400000 [00:02<00:51, 7382.54it/s]  4%|▍         | 16972/400000 [00:02<00:51, 7438.84it/s]  4%|▍         | 17730/400000 [00:02<00:51, 7478.12it/s]  5%|▍         | 18479/400000 [00:02<00:51, 7431.55it/s]  5%|▍         | 19223/400000 [00:02<00:51, 7353.06it/s]  5%|▍         | 19959/400000 [00:02<00:52, 7191.29it/s]  5%|▌         | 20733/400000 [00:02<00:51, 7345.74it/s]  5%|▌         | 21470/400000 [00:02<00:52, 7182.26it/s]  6%|▌         | 22223/400000 [00:03<00:51, 7280.24it/s]  6%|▌         | 22963/400000 [00:03<00:51, 7315.09it/s]  6%|▌         | 23720/400000 [00:03<00:50, 7387.68it/s]  6%|▌         | 24460/400000 [00:03<00:51, 7342.11it/s]  6%|▋         | 25195/400000 [00:03<00:51, 7276.68it/s]  6%|▋         | 25924/400000 [00:03<00:51, 7257.94it/s]  7%|▋         | 26698/400000 [00:03<00:50, 7393.63it/s]  7%|▋         | 27462/400000 [00:03<00:49, 7463.93it/s]  7%|▋         | 28210/400000 [00:03<00:49, 7457.49it/s]  7%|▋         | 28961/400000 [00:03<00:49, 7470.65it/s]  7%|▋         | 29709/400000 [00:04<00:49, 7448.46it/s]  8%|▊         | 30455/400000 [00:04<00:49, 7439.80it/s]  8%|▊         | 31213/400000 [00:04<00:49, 7480.45it/s]  8%|▊         | 31970/400000 [00:04<00:49, 7507.00it/s]  8%|▊         | 32725/400000 [00:04<00:48, 7505.98it/s]  8%|▊         | 33476/400000 [00:04<00:50, 7284.45it/s]  9%|▊         | 34206/400000 [00:04<00:50, 7247.90it/s]  9%|▊         | 34961/400000 [00:04<00:49, 7335.59it/s]  9%|▉         | 35723/400000 [00:04<00:49, 7418.63it/s]  9%|▉         | 36470/400000 [00:04<00:48, 7424.70it/s]  9%|▉         | 37214/400000 [00:05<00:49, 7389.25it/s]  9%|▉         | 37966/400000 [00:05<00:48, 7426.64it/s] 10%|▉         | 38733/400000 [00:05<00:48, 7496.88it/s] 10%|▉         | 39484/400000 [00:05<00:48, 7496.98it/s] 10%|█         | 40245/400000 [00:05<00:47, 7530.51it/s] 10%|█         | 40999/400000 [00:05<00:48, 7369.80it/s] 10%|█         | 41753/400000 [00:05<00:48, 7418.42it/s] 11%|█         | 42498/400000 [00:05<00:48, 7425.12it/s] 11%|█         | 43275/400000 [00:05<00:47, 7524.96it/s] 11%|█         | 44029/400000 [00:05<00:47, 7431.00it/s] 11%|█         | 44773/400000 [00:06<00:49, 7146.80it/s] 11%|█▏        | 45491/400000 [00:06<00:50, 7053.08it/s] 12%|█▏        | 46265/400000 [00:06<00:48, 7243.59it/s] 12%|█▏        | 47047/400000 [00:06<00:47, 7405.22it/s] 12%|█▏        | 47807/400000 [00:06<00:47, 7462.35it/s] 12%|█▏        | 48556/400000 [00:06<00:47, 7434.03it/s] 12%|█▏        | 49314/400000 [00:06<00:46, 7475.53it/s] 13%|█▎        | 50070/400000 [00:06<00:46, 7500.09it/s] 13%|█▎        | 50843/400000 [00:06<00:46, 7566.71it/s] 13%|█▎        | 51607/400000 [00:07<00:45, 7586.49it/s] 13%|█▎        | 52388/400000 [00:07<00:45, 7652.00it/s] 13%|█▎        | 53164/400000 [00:07<00:45, 7682.86it/s] 13%|█▎        | 53933/400000 [00:07<00:45, 7580.03it/s] 14%|█▎        | 54703/400000 [00:07<00:45, 7613.36it/s] 14%|█▍        | 55465/400000 [00:07<00:45, 7517.81it/s] 14%|█▍        | 56218/400000 [00:07<00:45, 7479.77it/s] 14%|█▍        | 56967/400000 [00:07<00:45, 7467.94it/s] 14%|█▍        | 57747/400000 [00:07<00:45, 7562.00it/s] 15%|█▍        | 58504/400000 [00:07<00:45, 7503.16it/s] 15%|█▍        | 59255/400000 [00:08<00:46, 7367.48it/s] 15%|█▍        | 59996/400000 [00:08<00:46, 7377.70it/s] 15%|█▌        | 60755/400000 [00:08<00:45, 7438.49it/s] 15%|█▌        | 61527/400000 [00:08<00:45, 7520.55it/s] 16%|█▌        | 62309/400000 [00:08<00:44, 7605.71it/s] 16%|█▌        | 63071/400000 [00:08<00:44, 7526.14it/s] 16%|█▌        | 63839/400000 [00:08<00:44, 7569.72it/s] 16%|█▌        | 64629/400000 [00:08<00:43, 7664.48it/s] 16%|█▋        | 65397/400000 [00:08<00:43, 7647.11it/s] 17%|█▋        | 66175/400000 [00:08<00:43, 7685.06it/s] 17%|█▋        | 66944/400000 [00:09<00:44, 7481.05it/s] 17%|█▋        | 67699/400000 [00:09<00:44, 7499.25it/s] 17%|█▋        | 68450/400000 [00:09<00:44, 7434.72it/s] 17%|█▋        | 69229/400000 [00:09<00:43, 7536.41it/s] 17%|█▋        | 69990/400000 [00:09<00:43, 7557.47it/s] 18%|█▊        | 70747/400000 [00:09<00:43, 7516.58it/s] 18%|█▊        | 71516/400000 [00:09<00:43, 7566.73it/s] 18%|█▊        | 72283/400000 [00:09<00:43, 7595.59it/s] 18%|█▊        | 73055/400000 [00:09<00:42, 7629.85it/s] 18%|█▊        | 73823/400000 [00:09<00:42, 7644.63it/s] 19%|█▊        | 74588/400000 [00:10<00:42, 7573.91it/s] 19%|█▉        | 75348/400000 [00:10<00:42, 7579.93it/s] 19%|█▉        | 76107/400000 [00:10<00:42, 7572.70it/s] 19%|█▉        | 76872/400000 [00:10<00:42, 7593.96it/s] 19%|█▉        | 77632/400000 [00:10<00:42, 7502.74it/s] 20%|█▉        | 78383/400000 [00:10<00:43, 7478.03it/s] 20%|█▉        | 79139/400000 [00:10<00:42, 7502.16it/s] 20%|█▉        | 79926/400000 [00:10<00:42, 7607.16it/s] 20%|██        | 80688/400000 [00:10<00:43, 7393.22it/s] 20%|██        | 81429/400000 [00:10<00:43, 7353.66it/s] 21%|██        | 82172/400000 [00:11<00:43, 7374.20it/s] 21%|██        | 82942/400000 [00:11<00:42, 7468.29it/s] 21%|██        | 83715/400000 [00:11<00:41, 7544.45it/s] 21%|██        | 84511/400000 [00:11<00:41, 7664.10it/s] 21%|██▏       | 85279/400000 [00:11<00:41, 7629.04it/s] 22%|██▏       | 86043/400000 [00:11<00:41, 7552.63it/s] 22%|██▏       | 86802/400000 [00:11<00:41, 7561.35it/s] 22%|██▏       | 87569/400000 [00:11<00:41, 7591.12it/s] 22%|██▏       | 88342/400000 [00:11<00:40, 7630.00it/s] 22%|██▏       | 89106/400000 [00:11<00:40, 7601.74it/s] 22%|██▏       | 89867/400000 [00:12<00:41, 7522.67it/s] 23%|██▎       | 90620/400000 [00:12<00:41, 7519.47it/s] 23%|██▎       | 91373/400000 [00:12<00:41, 7522.28it/s] 23%|██▎       | 92142/400000 [00:12<00:40, 7570.98it/s] 23%|██▎       | 92900/400000 [00:12<00:40, 7513.88it/s] 23%|██▎       | 93652/400000 [00:12<00:40, 7473.03it/s] 24%|██▎       | 94416/400000 [00:12<00:40, 7522.06it/s] 24%|██▍       | 95178/400000 [00:12<00:40, 7550.53it/s] 24%|██▍       | 95935/400000 [00:12<00:40, 7556.03it/s] 24%|██▍       | 96691/400000 [00:12<00:40, 7556.70it/s] 24%|██▍       | 97447/400000 [00:13<00:40, 7493.56it/s] 25%|██▍       | 98197/400000 [00:13<00:40, 7490.99it/s] 25%|██▍       | 98947/400000 [00:13<00:40, 7464.54it/s] 25%|██▍       | 99694/400000 [00:13<00:40, 7464.74it/s] 25%|██▌       | 100450/400000 [00:13<00:39, 7491.07it/s] 25%|██▌       | 101200/400000 [00:13<00:40, 7413.18it/s] 25%|██▌       | 101954/400000 [00:13<00:40, 7447.64it/s] 26%|██▌       | 102699/400000 [00:13<00:40, 7388.72it/s] 26%|██▌       | 103468/400000 [00:13<00:39, 7475.72it/s] 26%|██▌       | 104216/400000 [00:13<00:39, 7442.03it/s] 26%|██▌       | 104961/400000 [00:14<00:39, 7411.96it/s] 26%|██▋       | 105703/400000 [00:14<00:39, 7394.07it/s] 27%|██▋       | 106457/400000 [00:14<00:39, 7436.51it/s] 27%|██▋       | 107232/400000 [00:14<00:38, 7525.98it/s] 27%|██▋       | 107993/400000 [00:14<00:38, 7550.36it/s] 27%|██▋       | 108749/400000 [00:14<00:39, 7456.13it/s] 27%|██▋       | 109496/400000 [00:14<00:38, 7459.73it/s] 28%|██▊       | 110257/400000 [00:14<00:38, 7502.76it/s] 28%|██▊       | 111009/400000 [00:14<00:38, 7506.53it/s] 28%|██▊       | 111760/400000 [00:14<00:38, 7482.90it/s] 28%|██▊       | 112509/400000 [00:15<00:39, 7313.94it/s] 28%|██▊       | 113249/400000 [00:15<00:39, 7337.69it/s] 28%|██▊       | 113999/400000 [00:15<00:38, 7383.82it/s] 29%|██▊       | 114738/400000 [00:15<00:39, 7301.99it/s] 29%|██▉       | 115502/400000 [00:15<00:38, 7399.69it/s] 29%|██▉       | 116243/400000 [00:15<00:38, 7393.68it/s] 29%|██▉       | 116994/400000 [00:15<00:38, 7427.15it/s] 29%|██▉       | 117746/400000 [00:15<00:37, 7451.54it/s] 30%|██▉       | 118528/400000 [00:15<00:37, 7557.25it/s] 30%|██▉       | 119285/400000 [00:16<00:37, 7546.53it/s] 30%|███       | 120041/400000 [00:16<00:37, 7509.34it/s] 30%|███       | 120793/400000 [00:16<00:38, 7276.46it/s] 30%|███       | 121547/400000 [00:16<00:37, 7351.14it/s] 31%|███       | 122293/400000 [00:16<00:37, 7382.24it/s] 31%|███       | 123063/400000 [00:16<00:37, 7472.17it/s] 31%|███       | 123812/400000 [00:16<00:37, 7446.46it/s] 31%|███       | 124575/400000 [00:16<00:36, 7498.27it/s] 31%|███▏      | 125326/400000 [00:16<00:36, 7450.56it/s] 32%|███▏      | 126089/400000 [00:16<00:36, 7503.01it/s] 32%|███▏      | 126849/400000 [00:17<00:36, 7530.32it/s] 32%|███▏      | 127603/400000 [00:17<00:36, 7427.87it/s] 32%|███▏      | 128351/400000 [00:17<00:36, 7443.11it/s] 32%|███▏      | 129109/400000 [00:17<00:36, 7483.35it/s] 32%|███▏      | 129866/400000 [00:17<00:35, 7508.06it/s] 33%|███▎      | 130623/400000 [00:17<00:35, 7525.64it/s] 33%|███▎      | 131388/400000 [00:17<00:35, 7561.96it/s] 33%|███▎      | 132145/400000 [00:17<00:35, 7542.11it/s] 33%|███▎      | 132900/400000 [00:17<00:35, 7439.12it/s] 33%|███▎      | 133649/400000 [00:17<00:35, 7453.94it/s] 34%|███▎      | 134400/400000 [00:18<00:35, 7468.45it/s] 34%|███▍      | 135148/400000 [00:18<00:35, 7442.36it/s] 34%|███▍      | 135893/400000 [00:18<00:35, 7351.10it/s] 34%|███▍      | 136664/400000 [00:18<00:35, 7455.16it/s] 34%|███▍      | 137433/400000 [00:18<00:34, 7522.54it/s] 35%|███▍      | 138204/400000 [00:18<00:34, 7577.04it/s] 35%|███▍      | 138965/400000 [00:18<00:34, 7585.89it/s] 35%|███▍      | 139724/400000 [00:18<00:35, 7429.66it/s] 35%|███▌      | 140468/400000 [00:18<00:35, 7410.05it/s] 35%|███▌      | 141229/400000 [00:18<00:34, 7466.94it/s] 35%|███▌      | 141990/400000 [00:19<00:34, 7508.76it/s] 36%|███▌      | 142742/400000 [00:19<00:34, 7431.55it/s] 36%|███▌      | 143486/400000 [00:19<00:34, 7360.11it/s] 36%|███▌      | 144229/400000 [00:19<00:34, 7378.26it/s] 36%|███▌      | 144982/400000 [00:19<00:34, 7422.68it/s] 36%|███▋      | 145725/400000 [00:19<00:34, 7423.04it/s] 37%|███▋      | 146468/400000 [00:19<00:34, 7370.56it/s] 37%|███▋      | 147206/400000 [00:19<00:34, 7342.31it/s] 37%|███▋      | 147943/400000 [00:19<00:34, 7347.71it/s] 37%|███▋      | 148706/400000 [00:19<00:33, 7428.13it/s] 37%|███▋      | 149450/400000 [00:20<00:34, 7349.14it/s] 38%|███▊      | 150186/400000 [00:20<00:34, 7160.82it/s] 38%|███▊      | 150904/400000 [00:20<00:34, 7144.82it/s] 38%|███▊      | 151644/400000 [00:20<00:34, 7217.22it/s] 38%|███▊      | 152414/400000 [00:20<00:33, 7355.32it/s] 38%|███▊      | 153195/400000 [00:20<00:32, 7484.97it/s] 38%|███▊      | 153945/400000 [00:20<00:32, 7456.23it/s] 39%|███▊      | 154692/400000 [00:20<00:33, 7305.36it/s] 39%|███▉      | 155446/400000 [00:20<00:33, 7373.98it/s] 39%|███▉      | 156189/400000 [00:20<00:32, 7390.23it/s] 39%|███▉      | 156929/400000 [00:21<00:32, 7391.64it/s] 39%|███▉      | 157681/400000 [00:21<00:32, 7429.47it/s] 40%|███▉      | 158425/400000 [00:21<00:32, 7401.57it/s] 40%|███▉      | 159188/400000 [00:21<00:32, 7467.94it/s] 40%|███▉      | 159936/400000 [00:21<00:32, 7470.11it/s] 40%|████      | 160684/400000 [00:21<00:32, 7437.17it/s] 40%|████      | 161428/400000 [00:21<00:32, 7429.85it/s] 41%|████      | 162172/400000 [00:21<00:32, 7415.59it/s] 41%|████      | 162950/400000 [00:21<00:31, 7520.80it/s] 41%|████      | 163703/400000 [00:21<00:32, 7360.39it/s] 41%|████      | 164448/400000 [00:22<00:31, 7386.63it/s] 41%|████▏     | 165188/400000 [00:22<00:32, 7193.40it/s] 41%|████▏     | 165909/400000 [00:22<00:32, 7146.29it/s] 42%|████▏     | 166645/400000 [00:22<00:32, 7208.07it/s] 42%|████▏     | 167406/400000 [00:22<00:31, 7323.29it/s] 42%|████▏     | 168175/400000 [00:22<00:31, 7428.54it/s] 42%|████▏     | 168928/400000 [00:22<00:30, 7457.36it/s] 42%|████▏     | 169675/400000 [00:22<00:31, 7337.19it/s] 43%|████▎     | 170422/400000 [00:22<00:31, 7374.29it/s] 43%|████▎     | 171161/400000 [00:23<00:31, 7172.88it/s] 43%|████▎     | 171906/400000 [00:23<00:31, 7251.65it/s] 43%|████▎     | 172648/400000 [00:23<00:31, 7300.05it/s] 43%|████▎     | 173380/400000 [00:23<00:31, 7085.89it/s] 44%|████▎     | 174091/400000 [00:23<00:33, 6776.90it/s] 44%|████▎     | 174826/400000 [00:23<00:32, 6938.83it/s] 44%|████▍     | 175584/400000 [00:23<00:31, 7118.95it/s] 44%|████▍     | 176332/400000 [00:23<00:30, 7222.08it/s] 44%|████▍     | 177058/400000 [00:23<00:31, 7188.98it/s] 44%|████▍     | 177830/400000 [00:23<00:30, 7338.50it/s] 45%|████▍     | 178583/400000 [00:24<00:29, 7393.14it/s] 45%|████▍     | 179325/400000 [00:24<00:29, 7399.83it/s] 45%|████▌     | 180067/400000 [00:24<00:29, 7404.12it/s] 45%|████▌     | 180809/400000 [00:24<00:30, 7242.31it/s] 45%|████▌     | 181564/400000 [00:24<00:29, 7330.13it/s] 46%|████▌     | 182322/400000 [00:24<00:29, 7402.70it/s] 46%|████▌     | 183076/400000 [00:24<00:29, 7443.23it/s] 46%|████▌     | 183822/400000 [00:24<00:29, 7398.21it/s] 46%|████▌     | 184563/400000 [00:24<00:29, 7264.24it/s] 46%|████▋     | 185329/400000 [00:24<00:29, 7375.86it/s] 47%|████▋     | 186077/400000 [00:25<00:28, 7404.57it/s] 47%|████▋     | 186844/400000 [00:25<00:28, 7479.80it/s] 47%|████▋     | 187593/400000 [00:25<00:28, 7401.15it/s] 47%|████▋     | 188355/400000 [00:25<00:28, 7463.42it/s] 47%|████▋     | 189109/400000 [00:25<00:28, 7484.90it/s] 47%|████▋     | 189863/400000 [00:25<00:28, 7501.25it/s] 48%|████▊     | 190626/400000 [00:25<00:27, 7538.80it/s] 48%|████▊     | 191392/400000 [00:25<00:27, 7574.13it/s] 48%|████▊     | 192150/400000 [00:25<00:28, 7376.64it/s] 48%|████▊     | 192889/400000 [00:25<00:28, 7355.94it/s] 48%|████▊     | 193646/400000 [00:26<00:27, 7418.48it/s] 49%|████▊     | 194401/400000 [00:26<00:27, 7455.96it/s] 49%|████▉     | 195150/400000 [00:26<00:27, 7463.72it/s] 49%|████▉     | 195900/400000 [00:26<00:27, 7472.52it/s] 49%|████▉     | 196648/400000 [00:26<00:27, 7416.88it/s] 49%|████▉     | 197390/400000 [00:26<00:28, 7223.79it/s] 50%|████▉     | 198141/400000 [00:26<00:27, 7306.17it/s] 50%|████▉     | 198882/400000 [00:26<00:27, 7335.70it/s] 50%|████▉     | 199624/400000 [00:26<00:27, 7360.77it/s] 50%|█████     | 200393/400000 [00:26<00:26, 7454.40it/s] 50%|█████     | 201151/400000 [00:27<00:26, 7490.48it/s] 50%|█████     | 201915/400000 [00:27<00:26, 7533.00it/s] 51%|█████     | 202677/400000 [00:27<00:26, 7556.70it/s] 51%|█████     | 203433/400000 [00:27<00:26, 7533.38it/s] 51%|█████     | 204187/400000 [00:27<00:26, 7508.69it/s] 51%|█████     | 204964/400000 [00:27<00:25, 7584.69it/s] 51%|█████▏    | 205742/400000 [00:27<00:25, 7640.82it/s] 52%|█████▏    | 206508/400000 [00:27<00:25, 7644.55it/s] 52%|█████▏    | 207274/400000 [00:27<00:25, 7648.20it/s] 52%|█████▏    | 208050/400000 [00:28<00:24, 7679.15it/s] 52%|█████▏    | 208831/400000 [00:28<00:24, 7715.56it/s] 52%|█████▏    | 209603/400000 [00:28<00:24, 7651.05it/s] 53%|█████▎    | 210369/400000 [00:28<00:25, 7580.43it/s] 53%|█████▎    | 211128/400000 [00:28<00:25, 7540.26it/s] 53%|█████▎    | 211883/400000 [00:28<00:25, 7507.51it/s] 53%|█████▎    | 212634/400000 [00:28<00:24, 7507.10it/s] 53%|█████▎    | 213391/400000 [00:28<00:24, 7524.47it/s] 54%|█████▎    | 214144/400000 [00:28<00:24, 7481.48it/s] 54%|█████▎    | 214893/400000 [00:28<00:24, 7465.22it/s] 54%|█████▍    | 215640/400000 [00:29<00:24, 7456.14it/s] 54%|█████▍    | 216411/400000 [00:29<00:24, 7529.60it/s] 54%|█████▍    | 217177/400000 [00:29<00:24, 7565.40it/s] 54%|█████▍    | 217934/400000 [00:29<00:24, 7543.30it/s] 55%|█████▍    | 218689/400000 [00:29<00:24, 7516.57it/s] 55%|█████▍    | 219454/400000 [00:29<00:23, 7555.13it/s] 55%|█████▌    | 220216/400000 [00:29<00:23, 7572.02it/s] 55%|█████▌    | 220976/400000 [00:29<00:23, 7577.62it/s] 55%|█████▌    | 221744/400000 [00:29<00:23, 7605.25it/s] 56%|█████▌    | 222505/400000 [00:29<00:23, 7597.75it/s] 56%|█████▌    | 223265/400000 [00:30<00:23, 7576.35it/s] 56%|█████▌    | 224023/400000 [00:30<00:23, 7553.32it/s] 56%|█████▌    | 224799/400000 [00:30<00:23, 7614.01it/s] 56%|█████▋    | 225568/400000 [00:30<00:22, 7633.76it/s] 57%|█████▋    | 226332/400000 [00:30<00:22, 7602.46it/s] 57%|█████▋    | 227099/400000 [00:30<00:22, 7622.31it/s] 57%|█████▋    | 227868/400000 [00:30<00:22, 7641.12it/s] 57%|█████▋    | 228652/400000 [00:30<00:22, 7697.50it/s] 57%|█████▋    | 229435/400000 [00:30<00:22, 7735.25it/s] 58%|█████▊    | 230209/400000 [00:30<00:21, 7729.18it/s] 58%|█████▊    | 230983/400000 [00:31<00:21, 7717.67it/s] 58%|█████▊    | 231780/400000 [00:31<00:21, 7790.21it/s] 58%|█████▊    | 232566/400000 [00:31<00:21, 7809.24it/s] 58%|█████▊    | 233348/400000 [00:31<00:21, 7733.47it/s] 59%|█████▊    | 234122/400000 [00:31<00:21, 7723.18it/s] 59%|█████▊    | 234906/400000 [00:31<00:21, 7756.52it/s] 59%|█████▉    | 235685/400000 [00:31<00:21, 7765.67it/s] 59%|█████▉    | 236462/400000 [00:31<00:21, 7764.46it/s] 59%|█████▉    | 237239/400000 [00:31<00:21, 7671.48it/s] 60%|█████▉    | 238007/400000 [00:31<00:21, 7592.21it/s] 60%|█████▉    | 238767/400000 [00:32<00:21, 7495.65it/s] 60%|█████▉    | 239532/400000 [00:32<00:21, 7539.64it/s] 60%|██████    | 240298/400000 [00:32<00:21, 7572.72it/s] 60%|██████    | 241064/400000 [00:32<00:20, 7596.36it/s] 60%|██████    | 241824/400000 [00:32<00:20, 7580.73it/s] 61%|██████    | 242586/400000 [00:32<00:20, 7592.17it/s] 61%|██████    | 243357/400000 [00:32<00:20, 7625.04it/s] 61%|██████    | 244123/400000 [00:32<00:20, 7634.62it/s] 61%|██████    | 244887/400000 [00:32<00:20, 7442.40it/s] 61%|██████▏   | 245633/400000 [00:32<00:21, 7291.49it/s] 62%|██████▏   | 246403/400000 [00:33<00:20, 7408.14it/s] 62%|██████▏   | 247164/400000 [00:33<00:20, 7466.89it/s] 62%|██████▏   | 247913/400000 [00:33<00:20, 7472.76it/s] 62%|██████▏   | 248700/400000 [00:33<00:19, 7585.10it/s] 62%|██████▏   | 249460/400000 [00:33<00:19, 7566.72it/s] 63%|██████▎   | 250218/400000 [00:33<00:19, 7547.47it/s] 63%|██████▎   | 250974/400000 [00:33<00:19, 7530.92it/s] 63%|██████▎   | 251728/400000 [00:33<00:19, 7515.43it/s] 63%|██████▎   | 252490/400000 [00:33<00:19, 7545.40it/s] 63%|██████▎   | 253245/400000 [00:33<00:19, 7520.91it/s] 64%|██████▎   | 254020/400000 [00:34<00:19, 7587.09it/s] 64%|██████▎   | 254801/400000 [00:34<00:18, 7650.35it/s] 64%|██████▍   | 255576/400000 [00:34<00:18, 7676.83it/s] 64%|██████▍   | 256366/400000 [00:34<00:18, 7741.90it/s] 64%|██████▍   | 257141/400000 [00:34<00:18, 7570.42it/s] 64%|██████▍   | 257908/400000 [00:34<00:18, 7596.61it/s] 65%|██████▍   | 258669/400000 [00:34<00:18, 7526.59it/s] 65%|██████▍   | 259427/400000 [00:34<00:18, 7539.99it/s] 65%|██████▌   | 260183/400000 [00:34<00:18, 7545.96it/s] 65%|██████▌   | 260938/400000 [00:34<00:18, 7527.82it/s] 65%|██████▌   | 261692/400000 [00:35<00:18, 7465.41it/s] 66%|██████▌   | 262439/400000 [00:35<00:18, 7346.34it/s] 66%|██████▌   | 263198/400000 [00:35<00:18, 7415.20it/s] 66%|██████▌   | 263975/400000 [00:35<00:18, 7518.10it/s] 66%|██████▌   | 264731/400000 [00:35<00:17, 7527.81it/s] 66%|██████▋   | 265495/400000 [00:35<00:17, 7561.03it/s] 67%|██████▋   | 266266/400000 [00:35<00:17, 7603.41it/s] 67%|██████▋   | 267030/400000 [00:35<00:17, 7613.00it/s] 67%|██████▋   | 267792/400000 [00:35<00:17, 7528.72it/s] 67%|██████▋   | 268546/400000 [00:35<00:17, 7309.66it/s] 67%|██████▋   | 269279/400000 [00:36<00:18, 7210.60it/s] 68%|██████▊   | 270044/400000 [00:36<00:17, 7334.32it/s] 68%|██████▊   | 270808/400000 [00:36<00:17, 7423.08it/s] 68%|██████▊   | 271556/400000 [00:36<00:17, 7437.67it/s] 68%|██████▊   | 272304/400000 [00:36<00:17, 7448.65it/s] 68%|██████▊   | 273070/400000 [00:36<00:16, 7508.31it/s] 68%|██████▊   | 273822/400000 [00:36<00:16, 7510.17it/s] 69%|██████▊   | 274576/400000 [00:36<00:16, 7518.24it/s] 69%|██████▉   | 275332/400000 [00:36<00:16, 7530.61it/s] 69%|██████▉   | 276086/400000 [00:37<00:16, 7526.17it/s] 69%|██████▉   | 276851/400000 [00:37<00:16, 7561.34it/s] 69%|██████▉   | 277614/400000 [00:37<00:16, 7581.33it/s] 70%|██████▉   | 278373/400000 [00:37<00:16, 7565.04it/s] 70%|██████▉   | 279135/400000 [00:37<00:15, 7581.00it/s] 70%|██████▉   | 279894/400000 [00:37<00:15, 7529.72it/s] 70%|███████   | 280648/400000 [00:37<00:16, 7455.62it/s] 70%|███████   | 281409/400000 [00:37<00:15, 7500.63it/s] 71%|███████   | 282160/400000 [00:37<00:15, 7481.73it/s] 71%|███████   | 282917/400000 [00:37<00:15, 7506.93it/s] 71%|███████   | 283668/400000 [00:38<00:15, 7438.22it/s] 71%|███████   | 284421/400000 [00:38<00:15, 7464.67it/s] 71%|███████▏  | 285171/400000 [00:38<00:15, 7473.89it/s] 71%|███████▏  | 285926/400000 [00:38<00:15, 7495.80it/s] 72%|███████▏  | 286679/400000 [00:38<00:15, 7503.75it/s] 72%|███████▏  | 287430/400000 [00:38<00:15, 7367.91it/s] 72%|███████▏  | 288172/400000 [00:38<00:15, 7382.85it/s] 72%|███████▏  | 288954/400000 [00:38<00:14, 7508.07it/s] 72%|███████▏  | 289706/400000 [00:38<00:15, 7339.36it/s] 73%|███████▎  | 290442/400000 [00:38<00:15, 7251.61it/s] 73%|███████▎  | 291169/400000 [00:39<00:15, 7166.12it/s] 73%|███████▎  | 291906/400000 [00:39<00:14, 7225.66it/s] 73%|███████▎  | 292639/400000 [00:39<00:14, 7255.49it/s] 73%|███████▎  | 293402/400000 [00:39<00:14, 7362.37it/s] 74%|███████▎  | 294181/400000 [00:39<00:14, 7484.06it/s] 74%|███████▎  | 294931/400000 [00:39<00:14, 7480.98it/s] 74%|███████▍  | 295705/400000 [00:39<00:13, 7554.96it/s] 74%|███████▍  | 296479/400000 [00:39<00:13, 7607.03it/s] 74%|███████▍  | 297241/400000 [00:39<00:13, 7550.21it/s] 75%|███████▍  | 298010/400000 [00:39<00:13, 7590.46it/s] 75%|███████▍  | 298770/400000 [00:40<00:13, 7540.57it/s] 75%|███████▍  | 299530/400000 [00:40<00:13, 7557.21it/s] 75%|███████▌  | 300292/400000 [00:40<00:13, 7574.98it/s] 75%|███████▌  | 301060/400000 [00:40<00:13, 7605.67it/s] 75%|███████▌  | 301825/400000 [00:40<00:12, 7618.40it/s] 76%|███████▌  | 302587/400000 [00:40<00:12, 7538.19it/s] 76%|███████▌  | 303344/400000 [00:40<00:12, 7545.52it/s] 76%|███████▌  | 304101/400000 [00:40<00:12, 7552.05it/s] 76%|███████▌  | 304857/400000 [00:40<00:12, 7508.64it/s] 76%|███████▋  | 305619/400000 [00:40<00:12, 7539.19it/s] 77%|███████▋  | 306374/400000 [00:41<00:12, 7536.80it/s] 77%|███████▋  | 307129/400000 [00:41<00:12, 7539.17it/s] 77%|███████▋  | 307890/400000 [00:41<00:12, 7559.41it/s] 77%|███████▋  | 308668/400000 [00:41<00:11, 7624.24it/s] 77%|███████▋  | 309436/400000 [00:41<00:11, 7638.08it/s] 78%|███████▊  | 310200/400000 [00:41<00:11, 7618.83it/s] 78%|███████▊  | 310963/400000 [00:41<00:11, 7619.17it/s] 78%|███████▊  | 311733/400000 [00:41<00:11, 7640.12it/s] 78%|███████▊  | 312504/400000 [00:41<00:11, 7659.42it/s] 78%|███████▊  | 313279/400000 [00:41<00:11, 7684.98it/s] 79%|███████▊  | 314048/400000 [00:42<00:11, 7583.43it/s] 79%|███████▊  | 314830/400000 [00:42<00:11, 7650.32it/s] 79%|███████▉  | 315596/400000 [00:42<00:11, 7623.29it/s] 79%|███████▉  | 316359/400000 [00:42<00:10, 7621.04it/s] 79%|███████▉  | 317136/400000 [00:42<00:10, 7663.16it/s] 79%|███████▉  | 317903/400000 [00:42<00:10, 7558.37it/s] 80%|███████▉  | 318673/400000 [00:42<00:10, 7598.62it/s] 80%|███████▉  | 319434/400000 [00:42<00:10, 7589.56it/s] 80%|████████  | 320194/400000 [00:42<00:10, 7574.00it/s] 80%|████████  | 320952/400000 [00:42<00:10, 7554.01it/s] 80%|████████  | 321708/400000 [00:43<00:10, 7547.62it/s] 81%|████████  | 322468/400000 [00:43<00:10, 7562.77it/s] 81%|████████  | 323225/400000 [00:43<00:10, 7469.81it/s] 81%|████████  | 323992/400000 [00:43<00:10, 7526.93it/s] 81%|████████  | 324746/400000 [00:43<00:10, 7479.52it/s] 81%|████████▏ | 325495/400000 [00:43<00:10, 7438.76it/s] 82%|████████▏ | 326252/400000 [00:43<00:09, 7476.23it/s] 82%|████████▏ | 327000/400000 [00:43<00:09, 7459.42it/s] 82%|████████▏ | 327747/400000 [00:43<00:10, 7172.44it/s] 82%|████████▏ | 328509/400000 [00:43<00:09, 7299.21it/s] 82%|████████▏ | 329247/400000 [00:44<00:09, 7322.60it/s] 82%|████████▏ | 329996/400000 [00:44<00:09, 7369.51it/s] 83%|████████▎ | 330735/400000 [00:44<00:09, 7352.53it/s] 83%|████████▎ | 331484/400000 [00:44<00:09, 7392.35it/s] 83%|████████▎ | 332224/400000 [00:44<00:09, 7237.66it/s] 83%|████████▎ | 332949/400000 [00:44<00:09, 7222.04it/s] 83%|████████▎ | 333676/400000 [00:44<00:09, 7235.05it/s] 84%|████████▎ | 334430/400000 [00:44<00:08, 7322.77it/s] 84%|████████▍ | 335191/400000 [00:44<00:08, 7406.22it/s] 84%|████████▍ | 335954/400000 [00:44<00:08, 7469.21it/s] 84%|████████▍ | 336702/400000 [00:45<00:08, 7381.53it/s] 84%|████████▍ | 337457/400000 [00:45<00:08, 7427.90it/s] 85%|████████▍ | 338201/400000 [00:45<00:08, 7411.70it/s] 85%|████████▍ | 338943/400000 [00:45<00:08, 7409.13it/s] 85%|████████▍ | 339685/400000 [00:45<00:08, 7288.50it/s] 85%|████████▌ | 340425/400000 [00:45<00:08, 7319.53it/s] 85%|████████▌ | 341198/400000 [00:45<00:07, 7437.35it/s] 85%|████████▌ | 341943/400000 [00:45<00:07, 7267.84it/s] 86%|████████▌ | 342717/400000 [00:45<00:07, 7400.84it/s] 86%|████████▌ | 343486/400000 [00:46<00:07, 7482.70it/s] 86%|████████▌ | 344236/400000 [00:46<00:07, 7373.87it/s] 86%|████████▌ | 344995/400000 [00:46<00:07, 7435.92it/s] 86%|████████▋ | 345772/400000 [00:46<00:07, 7531.72it/s] 87%|████████▋ | 346554/400000 [00:46<00:07, 7615.29it/s] 87%|████████▋ | 347329/400000 [00:46<00:06, 7653.23it/s] 87%|████████▋ | 348096/400000 [00:46<00:06, 7580.29it/s] 87%|████████▋ | 348869/400000 [00:46<00:06, 7621.92it/s] 87%|████████▋ | 349632/400000 [00:46<00:06, 7327.21it/s] 88%|████████▊ | 350399/400000 [00:46<00:06, 7424.03it/s] 88%|████████▊ | 351149/400000 [00:47<00:06, 7445.91it/s] 88%|████████▊ | 351907/400000 [00:47<00:06, 7483.94it/s] 88%|████████▊ | 352657/400000 [00:47<00:06, 7480.25it/s] 88%|████████▊ | 353406/400000 [00:47<00:06, 7299.89it/s] 89%|████████▊ | 354181/400000 [00:47<00:06, 7428.39it/s] 89%|████████▊ | 354944/400000 [00:47<00:06, 7485.37it/s] 89%|████████▉ | 355699/400000 [00:47<00:05, 7502.56it/s] 89%|████████▉ | 356464/400000 [00:47<00:05, 7543.74it/s] 89%|████████▉ | 357229/400000 [00:47<00:05, 7574.13it/s] 89%|████████▉ | 357991/400000 [00:47<00:05, 7585.48it/s] 90%|████████▉ | 358750/400000 [00:48<00:05, 7561.61it/s] 90%|████████▉ | 359507/400000 [00:48<00:05, 7538.04it/s] 90%|█████████ | 360269/400000 [00:48<00:05, 7560.27it/s] 90%|█████████ | 361037/400000 [00:48<00:05, 7594.56it/s] 90%|█████████ | 361816/400000 [00:48<00:04, 7649.28it/s] 91%|█████████ | 362582/400000 [00:48<00:04, 7518.33it/s] 91%|█████████ | 363335/400000 [00:48<00:05, 7324.51it/s] 91%|█████████ | 364087/400000 [00:48<00:04, 7379.99it/s] 91%|█████████ | 364827/400000 [00:48<00:04, 7366.24it/s] 91%|█████████▏| 365586/400000 [00:48<00:04, 7431.63it/s] 92%|█████████▏| 366363/400000 [00:49<00:04, 7528.36it/s] 92%|█████████▏| 367117/400000 [00:49<00:04, 7393.24it/s] 92%|█████████▏| 367862/400000 [00:49<00:04, 7406.81it/s] 92%|█████████▏| 368604/400000 [00:49<00:04, 7361.41it/s] 92%|█████████▏| 369367/400000 [00:49<00:04, 7439.02it/s] 93%|█████████▎| 370112/400000 [00:49<00:04, 7401.26it/s] 93%|█████████▎| 370853/400000 [00:49<00:03, 7334.00it/s] 93%|█████████▎| 371587/400000 [00:49<00:03, 7310.77it/s] 93%|█████████▎| 372351/400000 [00:49<00:03, 7403.81it/s] 93%|█████████▎| 373092/400000 [00:49<00:03, 7399.62it/s] 93%|█████████▎| 373839/400000 [00:50<00:03, 7420.54it/s] 94%|█████████▎| 374582/400000 [00:50<00:03, 7396.17it/s] 94%|█████████▍| 375333/400000 [00:50<00:03, 7427.70it/s] 94%|█████████▍| 376076/400000 [00:50<00:03, 7420.91it/s] 94%|█████████▍| 376820/400000 [00:50<00:03, 7424.73it/s] 94%|█████████▍| 377611/400000 [00:50<00:02, 7563.30it/s] 95%|█████████▍| 378387/400000 [00:50<00:02, 7618.55it/s] 95%|█████████▍| 379150/400000 [00:50<00:02, 7569.47it/s] 95%|█████████▍| 379940/400000 [00:50<00:02, 7665.27it/s] 95%|█████████▌| 380731/400000 [00:50<00:02, 7736.56it/s] 95%|█████████▌| 381524/400000 [00:51<00:02, 7793.27it/s] 96%|█████████▌| 382304/400000 [00:51<00:02, 7623.15it/s] 96%|█████████▌| 383086/400000 [00:51<00:02, 7680.22it/s] 96%|█████████▌| 383878/400000 [00:51<00:02, 7748.75it/s] 96%|█████████▌| 384663/400000 [00:51<00:01, 7777.45it/s] 96%|█████████▋| 385448/400000 [00:51<00:01, 7798.65it/s] 97%|█████████▋| 386229/400000 [00:51<00:01, 7664.28it/s] 97%|█████████▋| 386997/400000 [00:51<00:01, 7582.66it/s] 97%|█████████▋| 387765/400000 [00:51<00:01, 7610.79it/s] 97%|█████████▋| 388544/400000 [00:52<00:01, 7662.30it/s] 97%|█████████▋| 389317/400000 [00:52<00:01, 7681.21it/s] 98%|█████████▊| 390086/400000 [00:52<00:01, 7504.52it/s] 98%|█████████▊| 390838/400000 [00:52<00:01, 7471.91it/s] 98%|█████████▊| 391587/400000 [00:52<00:01, 7475.33it/s] 98%|█████████▊| 392348/400000 [00:52<00:01, 7513.38it/s] 98%|█████████▊| 393121/400000 [00:52<00:00, 7575.31it/s] 98%|█████████▊| 393879/400000 [00:52<00:00, 7473.37it/s] 99%|█████████▊| 394677/400000 [00:52<00:00, 7617.12it/s] 99%|█████████▉| 395486/400000 [00:52<00:00, 7752.38it/s] 99%|█████████▉| 396263/400000 [00:53<00:00, 7526.94it/s] 99%|█████████▉| 397056/400000 [00:53<00:00, 7642.07it/s] 99%|█████████▉| 397823/400000 [00:53<00:00, 7569.06it/s]100%|█████████▉| 398582/400000 [00:53<00:00, 7561.23it/s]100%|█████████▉| 399381/400000 [00:53<00:00, 7683.41it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7475.59it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd97bd001d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011339959583227897 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011275782034947323 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15873 out of table with 15835 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15873 out of table with 15835 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

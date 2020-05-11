
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7e60f11f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 02:13:39.719963
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 02:13:39.722812
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 02:13:39.725441
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 02:13:39.727993
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7e6ccd53c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 353960.5625
Epoch 2/10

1/1 [==============================] - 0s 90ms/step - loss: 262852.5000
Epoch 3/10

1/1 [==============================] - 0s 83ms/step - loss: 167224.9062
Epoch 4/10

1/1 [==============================] - 0s 86ms/step - loss: 96547.3672
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 55394.6055
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 33546.3086
Epoch 7/10

1/1 [==============================] - 0s 86ms/step - loss: 21881.5684
Epoch 8/10

1/1 [==============================] - 0s 82ms/step - loss: 15276.3369
Epoch 9/10

1/1 [==============================] - 0s 88ms/step - loss: 11388.1982
Epoch 10/10

1/1 [==============================] - 0s 86ms/step - loss: 8729.6055

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.81011105 -1.0376892   0.05252188  0.5610839   1.0454068   2.1571312
  -0.66582656 -0.95480597  1.8483319  -0.1613199   0.8494143   0.42856923
   0.53190047  0.4356719   1.4492427  -0.53610253 -0.14976805  0.34681755
  -1.8079305  -1.8376958  -1.3508817   0.09992385 -1.0629717   0.9324409
   1.9607508   0.54023194 -1.9282701   0.84599566 -1.2245922   0.9464241
  -0.76096004  0.57591647  1.4451857   1.5769691  -0.6179754  -0.12392855
   1.3096749  -1.1757993   1.2305095   0.44477475  0.2857013   0.4209163
  -0.9132232  -0.12498374  0.23455457  0.19041118 -0.97010386  1.0309193
  -0.97399956  0.23623186  0.23380214  1.3184055   0.5668807   0.856294
   0.5033849   0.14925961  1.2870318   0.5211298   1.2886617  -1.6104143
   0.3206678   5.360479    5.0585575   6.9192586   5.2133656   6.241247
   7.127581    6.3174233   7.1844625   8.013463    6.498464    5.397915
   5.2951026   8.421674    7.423657    7.6970444   5.9872117   6.4621043
   6.849171    6.6943183   6.8200717   4.5228987   8.582494    4.9793673
   7.2528944   6.772673    6.249779    5.934543    6.785527    5.0364866
   6.9237714   5.13885     6.157308    6.119576    5.743721    6.6301622
   7.3496513   7.3402724   7.0303617   7.349448    6.1539054   5.52907
   6.353257    7.858436    5.7407694   6.99036     7.4154496   6.3599916
   7.468359    7.825627    6.8314915   6.019514    7.5097756   6.8693433
   5.8058615   6.5813026   7.5050035   6.867591    5.5870633   7.3155327
   1.8193836  -0.59321445  1.8851151  -1.0475969   1.0674256   0.0751471
   1.5660748   1.1945045   0.5754465   1.1187806   2.0209956  -0.8504802
  -1.2509481   1.7782474  -0.3375709  -0.53196776 -1.570287   -1.6532377
   0.5615738  -0.81871367 -0.20656395  1.8312085  -0.6118835   0.32414353
  -1.2069864  -1.3892215  -0.06470145  0.8496879  -0.62216663 -1.7217118
  -1.1583009   1.1854085   0.53530884 -1.5607932   1.9380689  -0.15833095
  -0.235336    0.05522206 -0.62640023  0.31811005 -0.0391897  -0.3489022
  -0.6349983  -0.4775061   0.43165475  0.18073834  1.2669756  -0.3516568
   0.34270644  0.63414776 -0.04330093  0.14127128  0.31110615 -0.4015993
  -0.13730517 -0.6506369   0.5353707   0.9100841  -0.44835186 -0.5671822
   2.7368507   0.2304241   2.8202405   0.59429777  1.5701468   0.5615294
   0.8633648   0.99351996  0.44627702  1.7246878   0.7228186   0.4969749
   2.0457516   0.7177757   0.3375855   2.0534096   1.2469267   2.4604416
   1.2729824   0.3098566   0.697362    1.3168535   0.67043555  0.565715
   1.8591807   0.7827032   1.0406387   1.0171655   0.4403007   1.8075198
   0.3531208   0.850416    1.5875015   1.653676    0.82520604  1.4178481
   1.1031636   0.5093559   1.6318991   0.72248703  2.5546932   0.64034724
   2.5373244   2.3805368   1.3762631   0.17378461  0.8605148   0.8307449
   0.38538158  1.5522102   0.4504422   0.84282655  0.43219125  2.1109216
   0.7629077   2.8143616   2.2904835   0.9701258   0.17265886  0.52240914
   0.02295148  8.205866    8.472294    7.6958604   7.064052    6.377188
   8.517123    7.6740217   7.3418026   6.4162936   7.3407807   7.224083
   7.954022    8.389283    7.592491    6.763434    6.6369934   7.8863273
   6.4685163   7.5224767   8.02819     5.5190034   7.0784044   6.5017104
   6.6348133   8.411581    7.372175    7.4872293   6.576403    7.290158
   7.2149353   6.896074    7.972155    6.9593096   9.094416    8.796106
   8.184746    5.7295747   6.9856005   6.3648314   6.3306274   6.662273
   7.277638    7.5905576   6.199999    5.649179    7.403197    6.747668
   7.1234617   6.7507377   7.4838104   6.3631554   7.4368215   8.209996
   7.389478    7.8184233   7.3731136   5.783832    6.680797    8.62464
   1.7232306   0.6566312   0.21036357  1.3626411   1.8886409   0.97210497
   0.19309986  1.4484398   0.6514527   2.985239    1.6199348   1.253366
   1.0233091   0.62454283  1.3675462   2.0431776   2.061924    1.4994346
   0.4572103   0.5109299   0.17362553  0.12341797  1.6268259   1.6313138
   0.7622473   1.7008874   2.375459    1.9290812   1.6287917   0.6870672
   2.083325    2.493949    1.955679    1.2180772   1.4435927   1.2737398
   1.6091697   2.8195264   0.24523866  0.50759625  1.3364503   1.7435663
   1.347861    1.8102763   1.7097449   0.61378396  1.7705538   0.73971045
   2.8945303   2.9533172   0.6216029   0.09683907  1.1309049   0.31051373
   0.60226506  0.4230731   1.285109    0.45424736  2.0914805   1.1268207
  -9.013226    5.4706774  -3.2827594 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 02:13:47.336860
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.8124
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 02:13:47.340029
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9006.4
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 02:13:47.343165
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.4993
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 02:13:47.346122
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -805.587
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140180409884968
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140179199716488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140179199716992
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140179199316152
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140179199316656
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140179199317160

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7e68b62e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.620165
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.586005
grad_step = 000002, loss = 0.558469
grad_step = 000003, loss = 0.527980
grad_step = 000004, loss = 0.493380
grad_step = 000005, loss = 0.457222
grad_step = 000006, loss = 0.429549
grad_step = 000007, loss = 0.420536
grad_step = 000008, loss = 0.418947
grad_step = 000009, loss = 0.399468
grad_step = 000010, loss = 0.376645
grad_step = 000011, loss = 0.362160
grad_step = 000012, loss = 0.354520
grad_step = 000013, loss = 0.347742
grad_step = 000014, loss = 0.338454
grad_step = 000015, loss = 0.326498
grad_step = 000016, loss = 0.313348
grad_step = 000017, loss = 0.300769
grad_step = 000018, loss = 0.289823
grad_step = 000019, loss = 0.279700
grad_step = 000020, loss = 0.269874
grad_step = 000021, loss = 0.260481
grad_step = 000022, loss = 0.250897
grad_step = 000023, loss = 0.240805
grad_step = 000024, loss = 0.230748
grad_step = 000025, loss = 0.221345
grad_step = 000026, loss = 0.212384
grad_step = 000027, loss = 0.203577
grad_step = 000028, loss = 0.194817
grad_step = 000029, loss = 0.186093
grad_step = 000030, loss = 0.177506
grad_step = 000031, loss = 0.169265
grad_step = 000032, loss = 0.161307
grad_step = 000033, loss = 0.153656
grad_step = 000034, loss = 0.146143
grad_step = 000035, loss = 0.138707
grad_step = 000036, loss = 0.131913
grad_step = 000037, loss = 0.125377
grad_step = 000038, loss = 0.118530
grad_step = 000039, loss = 0.111976
grad_step = 000040, loss = 0.106049
grad_step = 000041, loss = 0.100529
grad_step = 000042, loss = 0.095085
grad_step = 000043, loss = 0.089669
grad_step = 000044, loss = 0.084403
grad_step = 000045, loss = 0.079541
grad_step = 000046, loss = 0.075086
grad_step = 000047, loss = 0.070763
grad_step = 000048, loss = 0.066542
grad_step = 000049, loss = 0.062479
grad_step = 000050, loss = 0.058627
grad_step = 000051, loss = 0.055108
grad_step = 000052, loss = 0.051829
grad_step = 000053, loss = 0.048592
grad_step = 000054, loss = 0.045474
grad_step = 000055, loss = 0.042640
grad_step = 000056, loss = 0.040025
grad_step = 000057, loss = 0.037470
grad_step = 000058, loss = 0.034994
grad_step = 000059, loss = 0.032751
grad_step = 000060, loss = 0.030697
grad_step = 000061, loss = 0.028699
grad_step = 000062, loss = 0.026793
grad_step = 000063, loss = 0.025044
grad_step = 000064, loss = 0.023432
grad_step = 000065, loss = 0.021903
grad_step = 000066, loss = 0.020436
grad_step = 000067, loss = 0.019074
grad_step = 000068, loss = 0.017831
grad_step = 000069, loss = 0.016657
grad_step = 000070, loss = 0.015539
grad_step = 000071, loss = 0.014509
grad_step = 000072, loss = 0.013561
grad_step = 000073, loss = 0.012651
grad_step = 000074, loss = 0.011799
grad_step = 000075, loss = 0.011025
grad_step = 000076, loss = 0.010295
grad_step = 000077, loss = 0.009601
grad_step = 000078, loss = 0.008959
grad_step = 000079, loss = 0.008371
grad_step = 000080, loss = 0.007819
grad_step = 000081, loss = 0.007297
grad_step = 000082, loss = 0.006815
grad_step = 000083, loss = 0.006373
grad_step = 000084, loss = 0.005960
grad_step = 000085, loss = 0.005573
grad_step = 000086, loss = 0.005224
grad_step = 000087, loss = 0.004900
grad_step = 000088, loss = 0.004596
grad_step = 000089, loss = 0.004323
grad_step = 000090, loss = 0.004075
grad_step = 000091, loss = 0.003845
grad_step = 000092, loss = 0.003635
grad_step = 000093, loss = 0.003448
grad_step = 000094, loss = 0.003279
grad_step = 000095, loss = 0.003124
grad_step = 000096, loss = 0.002986
grad_step = 000097, loss = 0.002865
grad_step = 000098, loss = 0.002754
grad_step = 000099, loss = 0.002657
grad_step = 000100, loss = 0.002572
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002496
grad_step = 000102, loss = 0.002430
grad_step = 000103, loss = 0.002372
grad_step = 000104, loss = 0.002322
grad_step = 000105, loss = 0.002279
grad_step = 000106, loss = 0.002243
grad_step = 000107, loss = 0.002216
grad_step = 000108, loss = 0.002204
grad_step = 000109, loss = 0.002223
grad_step = 000110, loss = 0.002303
grad_step = 000111, loss = 0.002337
grad_step = 000112, loss = 0.002271
grad_step = 000113, loss = 0.002105
grad_step = 000114, loss = 0.002146
grad_step = 000115, loss = 0.002243
grad_step = 000116, loss = 0.002125
grad_step = 000117, loss = 0.002073
grad_step = 000118, loss = 0.002159
grad_step = 000119, loss = 0.002117
grad_step = 000120, loss = 0.002052
grad_step = 000121, loss = 0.002100
grad_step = 000122, loss = 0.002100
grad_step = 000123, loss = 0.002047
grad_step = 000124, loss = 0.002065
grad_step = 000125, loss = 0.002083
grad_step = 000126, loss = 0.002047
grad_step = 000127, loss = 0.002044
grad_step = 000128, loss = 0.002066
grad_step = 000129, loss = 0.002048
grad_step = 000130, loss = 0.002032
grad_step = 000131, loss = 0.002049
grad_step = 000132, loss = 0.002046
grad_step = 000133, loss = 0.002026
grad_step = 000134, loss = 0.002034
grad_step = 000135, loss = 0.002040
grad_step = 000136, loss = 0.002024
grad_step = 000137, loss = 0.002020
grad_step = 000138, loss = 0.002029
grad_step = 000139, loss = 0.002023
grad_step = 000140, loss = 0.002012
grad_step = 000141, loss = 0.002015
grad_step = 000142, loss = 0.002017
grad_step = 000143, loss = 0.002009
grad_step = 000144, loss = 0.002004
grad_step = 000145, loss = 0.002006
grad_step = 000146, loss = 0.002006
grad_step = 000147, loss = 0.001999
grad_step = 000148, loss = 0.001995
grad_step = 000149, loss = 0.001996
grad_step = 000150, loss = 0.001995
grad_step = 000151, loss = 0.001990
grad_step = 000152, loss = 0.001987
grad_step = 000153, loss = 0.001986
grad_step = 000154, loss = 0.001986
grad_step = 000155, loss = 0.001982
grad_step = 000156, loss = 0.001978
grad_step = 000157, loss = 0.001977
grad_step = 000158, loss = 0.001976
grad_step = 000159, loss = 0.001974
grad_step = 000160, loss = 0.001971
grad_step = 000161, loss = 0.001968
grad_step = 000162, loss = 0.001966
grad_step = 000163, loss = 0.001965
grad_step = 000164, loss = 0.001964
grad_step = 000165, loss = 0.001961
grad_step = 000166, loss = 0.001959
grad_step = 000167, loss = 0.001956
grad_step = 000168, loss = 0.001954
grad_step = 000169, loss = 0.001953
grad_step = 000170, loss = 0.001951
grad_step = 000171, loss = 0.001949
grad_step = 000172, loss = 0.001947
grad_step = 000173, loss = 0.001945
grad_step = 000174, loss = 0.001943
grad_step = 000175, loss = 0.001941
grad_step = 000176, loss = 0.001939
grad_step = 000177, loss = 0.001937
grad_step = 000178, loss = 0.001935
grad_step = 000179, loss = 0.001933
grad_step = 000180, loss = 0.001931
grad_step = 000181, loss = 0.001929
grad_step = 000182, loss = 0.001927
grad_step = 000183, loss = 0.001925
grad_step = 000184, loss = 0.001924
grad_step = 000185, loss = 0.001922
grad_step = 000186, loss = 0.001922
grad_step = 000187, loss = 0.001925
grad_step = 000188, loss = 0.001936
grad_step = 000189, loss = 0.001965
grad_step = 000190, loss = 0.002043
grad_step = 000191, loss = 0.002168
grad_step = 000192, loss = 0.002369
grad_step = 000193, loss = 0.002299
grad_step = 000194, loss = 0.002068
grad_step = 000195, loss = 0.001908
grad_step = 000196, loss = 0.002028
grad_step = 000197, loss = 0.002167
grad_step = 000198, loss = 0.002035
grad_step = 000199, loss = 0.001901
grad_step = 000200, loss = 0.001986
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002063
grad_step = 000202, loss = 0.001977
grad_step = 000203, loss = 0.001894
grad_step = 000204, loss = 0.001970
grad_step = 000205, loss = 0.002014
grad_step = 000206, loss = 0.001926
grad_step = 000207, loss = 0.001895
grad_step = 000208, loss = 0.001957
grad_step = 000209, loss = 0.001959
grad_step = 000210, loss = 0.001893
grad_step = 000211, loss = 0.001893
grad_step = 000212, loss = 0.001933
grad_step = 000213, loss = 0.001919
grad_step = 000214, loss = 0.001878
grad_step = 000215, loss = 0.001887
grad_step = 000216, loss = 0.001911
grad_step = 000217, loss = 0.001892
grad_step = 000218, loss = 0.001868
grad_step = 000219, loss = 0.001878
grad_step = 000220, loss = 0.001890
grad_step = 000221, loss = 0.001877
grad_step = 000222, loss = 0.001861
grad_step = 000223, loss = 0.001867
grad_step = 000224, loss = 0.001875
grad_step = 000225, loss = 0.001866
grad_step = 000226, loss = 0.001854
grad_step = 000227, loss = 0.001855
grad_step = 000228, loss = 0.001861
grad_step = 000229, loss = 0.001857
grad_step = 000230, loss = 0.001848
grad_step = 000231, loss = 0.001845
grad_step = 000232, loss = 0.001848
grad_step = 000233, loss = 0.001848
grad_step = 000234, loss = 0.001843
grad_step = 000235, loss = 0.001837
grad_step = 000236, loss = 0.001836
grad_step = 000237, loss = 0.001837
grad_step = 000238, loss = 0.001836
grad_step = 000239, loss = 0.001832
grad_step = 000240, loss = 0.001828
grad_step = 000241, loss = 0.001825
grad_step = 000242, loss = 0.001825
grad_step = 000243, loss = 0.001825
grad_step = 000244, loss = 0.001824
grad_step = 000245, loss = 0.001821
grad_step = 000246, loss = 0.001817
grad_step = 000247, loss = 0.001814
grad_step = 000248, loss = 0.001812
grad_step = 000249, loss = 0.001811
grad_step = 000250, loss = 0.001810
grad_step = 000251, loss = 0.001809
grad_step = 000252, loss = 0.001807
grad_step = 000253, loss = 0.001805
grad_step = 000254, loss = 0.001803
grad_step = 000255, loss = 0.001800
grad_step = 000256, loss = 0.001798
grad_step = 000257, loss = 0.001795
grad_step = 000258, loss = 0.001793
grad_step = 000259, loss = 0.001791
grad_step = 000260, loss = 0.001789
grad_step = 000261, loss = 0.001788
grad_step = 000262, loss = 0.001786
grad_step = 000263, loss = 0.001786
grad_step = 000264, loss = 0.001787
grad_step = 000265, loss = 0.001792
grad_step = 000266, loss = 0.001806
grad_step = 000267, loss = 0.001839
grad_step = 000268, loss = 0.001909
grad_step = 000269, loss = 0.002053
grad_step = 000270, loss = 0.002267
grad_step = 000271, loss = 0.002475
grad_step = 000272, loss = 0.002368
grad_step = 000273, loss = 0.001987
grad_step = 000274, loss = 0.001767
grad_step = 000275, loss = 0.001944
grad_step = 000276, loss = 0.002143
grad_step = 000277, loss = 0.001993
grad_step = 000278, loss = 0.001771
grad_step = 000279, loss = 0.001867
grad_step = 000280, loss = 0.002015
grad_step = 000281, loss = 0.001914
grad_step = 000282, loss = 0.001760
grad_step = 000283, loss = 0.001846
grad_step = 000284, loss = 0.001948
grad_step = 000285, loss = 0.001853
grad_step = 000286, loss = 0.001754
grad_step = 000287, loss = 0.001824
grad_step = 000288, loss = 0.001887
grad_step = 000289, loss = 0.001807
grad_step = 000290, loss = 0.001749
grad_step = 000291, loss = 0.001799
grad_step = 000292, loss = 0.001836
grad_step = 000293, loss = 0.001773
grad_step = 000294, loss = 0.001743
grad_step = 000295, loss = 0.001780
grad_step = 000296, loss = 0.001797
grad_step = 000297, loss = 0.001862
grad_step = 000298, loss = 0.001780
grad_step = 000299, loss = 0.001839
grad_step = 000300, loss = 0.001823
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001747
grad_step = 000302, loss = 0.001752
grad_step = 000303, loss = 0.001796
grad_step = 000304, loss = 0.001782
grad_step = 000305, loss = 0.001739
grad_step = 000306, loss = 0.001735
grad_step = 000307, loss = 0.001760
grad_step = 000308, loss = 0.001756
grad_step = 000309, loss = 0.001728
grad_step = 000310, loss = 0.001721
grad_step = 000311, loss = 0.001736
grad_step = 000312, loss = 0.001737
grad_step = 000313, loss = 0.001725
grad_step = 000314, loss = 0.001709
grad_step = 000315, loss = 0.001716
grad_step = 000316, loss = 0.001722
grad_step = 000317, loss = 0.001718
grad_step = 000318, loss = 0.001705
grad_step = 000319, loss = 0.001702
grad_step = 000320, loss = 0.001706
grad_step = 000321, loss = 0.001707
grad_step = 000322, loss = 0.001702
grad_step = 000323, loss = 0.001695
grad_step = 000324, loss = 0.001693
grad_step = 000325, loss = 0.001695
grad_step = 000326, loss = 0.001695
grad_step = 000327, loss = 0.001693
grad_step = 000328, loss = 0.001687
grad_step = 000329, loss = 0.001684
grad_step = 000330, loss = 0.001683
grad_step = 000331, loss = 0.001684
grad_step = 000332, loss = 0.001682
grad_step = 000333, loss = 0.001680
grad_step = 000334, loss = 0.001676
grad_step = 000335, loss = 0.001673
grad_step = 000336, loss = 0.001672
grad_step = 000337, loss = 0.001672
grad_step = 000338, loss = 0.001671
grad_step = 000339, loss = 0.001670
grad_step = 000340, loss = 0.001667
grad_step = 000341, loss = 0.001665
grad_step = 000342, loss = 0.001662
grad_step = 000343, loss = 0.001660
grad_step = 000344, loss = 0.001659
grad_step = 000345, loss = 0.001657
grad_step = 000346, loss = 0.001656
grad_step = 000347, loss = 0.001655
grad_step = 000348, loss = 0.001654
grad_step = 000349, loss = 0.001653
grad_step = 000350, loss = 0.001652
grad_step = 000351, loss = 0.001651
grad_step = 000352, loss = 0.001651
grad_step = 000353, loss = 0.001651
grad_step = 000354, loss = 0.001655
grad_step = 000355, loss = 0.001664
grad_step = 000356, loss = 0.001686
grad_step = 000357, loss = 0.001727
grad_step = 000358, loss = 0.001807
grad_step = 000359, loss = 0.001929
grad_step = 000360, loss = 0.002099
grad_step = 000361, loss = 0.002196
grad_step = 000362, loss = 0.002109
grad_step = 000363, loss = 0.001842
grad_step = 000364, loss = 0.001644
grad_step = 000365, loss = 0.001690
grad_step = 000366, loss = 0.001854
grad_step = 000367, loss = 0.001889
grad_step = 000368, loss = 0.001745
grad_step = 000369, loss = 0.001628
grad_step = 000370, loss = 0.001667
grad_step = 000371, loss = 0.001771
grad_step = 000372, loss = 0.001784
grad_step = 000373, loss = 0.001688
grad_step = 000374, loss = 0.001618
grad_step = 000375, loss = 0.001644
grad_step = 000376, loss = 0.001706
grad_step = 000377, loss = 0.001707
grad_step = 000378, loss = 0.001647
grad_step = 000379, loss = 0.001608
grad_step = 000380, loss = 0.001628
grad_step = 000381, loss = 0.001663
grad_step = 000382, loss = 0.001659
grad_step = 000383, loss = 0.001622
grad_step = 000384, loss = 0.001600
grad_step = 000385, loss = 0.001613
grad_step = 000386, loss = 0.001633
grad_step = 000387, loss = 0.001630
grad_step = 000388, loss = 0.001607
grad_step = 000389, loss = 0.001592
grad_step = 000390, loss = 0.001599
grad_step = 000391, loss = 0.001610
grad_step = 000392, loss = 0.001609
grad_step = 000393, loss = 0.001595
grad_step = 000394, loss = 0.001584
grad_step = 000395, loss = 0.001585
grad_step = 000396, loss = 0.001592
grad_step = 000397, loss = 0.001593
grad_step = 000398, loss = 0.001587
grad_step = 000399, loss = 0.001578
grad_step = 000400, loss = 0.001573
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001573
grad_step = 000402, loss = 0.001576
grad_step = 000403, loss = 0.001576
grad_step = 000404, loss = 0.001573
grad_step = 000405, loss = 0.001568
grad_step = 000406, loss = 0.001563
grad_step = 000407, loss = 0.001560
grad_step = 000408, loss = 0.001558
grad_step = 000409, loss = 0.001558
grad_step = 000410, loss = 0.001557
grad_step = 000411, loss = 0.001555
grad_step = 000412, loss = 0.001553
grad_step = 000413, loss = 0.001549
grad_step = 000414, loss = 0.001546
grad_step = 000415, loss = 0.001543
grad_step = 000416, loss = 0.001541
grad_step = 000417, loss = 0.001539
grad_step = 000418, loss = 0.001537
grad_step = 000419, loss = 0.001535
grad_step = 000420, loss = 0.001534
grad_step = 000421, loss = 0.001533
grad_step = 000422, loss = 0.001532
grad_step = 000423, loss = 0.001532
grad_step = 000424, loss = 0.001533
grad_step = 000425, loss = 0.001536
grad_step = 000426, loss = 0.001543
grad_step = 000427, loss = 0.001556
grad_step = 000428, loss = 0.001582
grad_step = 000429, loss = 0.001623
grad_step = 000430, loss = 0.001698
grad_step = 000431, loss = 0.001780
grad_step = 000432, loss = 0.001879
grad_step = 000433, loss = 0.001880
grad_step = 000434, loss = 0.001787
grad_step = 000435, loss = 0.001618
grad_step = 000436, loss = 0.001508
grad_step = 000437, loss = 0.001524
grad_step = 000438, loss = 0.001617
grad_step = 000439, loss = 0.001697
grad_step = 000440, loss = 0.001681
grad_step = 000441, loss = 0.001613
grad_step = 000442, loss = 0.001522
grad_step = 000443, loss = 0.001482
grad_step = 000444, loss = 0.001508
grad_step = 000445, loss = 0.001556
grad_step = 000446, loss = 0.001574
grad_step = 000447, loss = 0.001542
grad_step = 000448, loss = 0.001493
grad_step = 000449, loss = 0.001469
grad_step = 000450, loss = 0.001483
grad_step = 000451, loss = 0.001510
grad_step = 000452, loss = 0.001517
grad_step = 000453, loss = 0.001501
grad_step = 000454, loss = 0.001473
grad_step = 000455, loss = 0.001456
grad_step = 000456, loss = 0.001456
grad_step = 000457, loss = 0.001467
grad_step = 000458, loss = 0.001477
grad_step = 000459, loss = 0.001476
grad_step = 000460, loss = 0.001465
grad_step = 000461, loss = 0.001449
grad_step = 000462, loss = 0.001439
grad_step = 000463, loss = 0.001436
grad_step = 000464, loss = 0.001440
grad_step = 000465, loss = 0.001445
grad_step = 000466, loss = 0.001445
grad_step = 000467, loss = 0.001442
grad_step = 000468, loss = 0.001434
grad_step = 000469, loss = 0.001426
grad_step = 000470, loss = 0.001419
grad_step = 000471, loss = 0.001415
grad_step = 000472, loss = 0.001413
grad_step = 000473, loss = 0.001413
grad_step = 000474, loss = 0.001413
grad_step = 000475, loss = 0.001415
grad_step = 000476, loss = 0.001417
grad_step = 000477, loss = 0.001421
grad_step = 000478, loss = 0.001428
grad_step = 000479, loss = 0.001436
grad_step = 000480, loss = 0.001450
grad_step = 000481, loss = 0.001463
grad_step = 000482, loss = 0.001481
grad_step = 000483, loss = 0.001486
grad_step = 000484, loss = 0.001490
grad_step = 000485, loss = 0.001468
grad_step = 000486, loss = 0.001442
grad_step = 000487, loss = 0.001408
grad_step = 000488, loss = 0.001383
grad_step = 000489, loss = 0.001372
grad_step = 000490, loss = 0.001375
grad_step = 000491, loss = 0.001387
grad_step = 000492, loss = 0.001405
grad_step = 000493, loss = 0.001434
grad_step = 000494, loss = 0.001463
grad_step = 000495, loss = 0.001514
grad_step = 000496, loss = 0.001535
grad_step = 000497, loss = 0.001552
grad_step = 000498, loss = 0.001501
grad_step = 000499, loss = 0.001436
grad_step = 000500, loss = 0.001371
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001349
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

  date_run                              2020-05-11 02:14:05.154218
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.208859
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 02:14:05.159176
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.102885
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 02:14:05.165492
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.131492
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 02:14:05.169644
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.563376
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
0   2020-05-11 02:13:39.719963  ...    mean_absolute_error
1   2020-05-11 02:13:39.722812  ...     mean_squared_error
2   2020-05-11 02:13:39.725441  ...  median_absolute_error
3   2020-05-11 02:13:39.727993  ...               r2_score
4   2020-05-11 02:13:47.336860  ...    mean_absolute_error
5   2020-05-11 02:13:47.340029  ...     mean_squared_error
6   2020-05-11 02:13:47.343165  ...  median_absolute_error
7   2020-05-11 02:13:47.346122  ...               r2_score
8   2020-05-11 02:14:05.154218  ...    mean_absolute_error
9   2020-05-11 02:14:05.159176  ...     mean_squared_error
10  2020-05-11 02:14:05.165492  ...  median_absolute_error
11  2020-05-11 02:14:05.169644  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▌      | 3489792/9912422 [00:00<00:00, 34896289.29it/s]9920512it [00:00, 33710361.56it/s]                             
0it [00:00, ?it/s]32768it [00:00, 586919.56it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151261.54it/s]1654784it [00:00, 11107502.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196659.39it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fccecb3f0b8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcc891a3c50> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcceba0de10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcc891a3dd8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fccecb3f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcc9e40acf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fccecb3f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcc928bf518> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fccecb3f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcc9e40acf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fccecb3f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9c1fda01d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c3a4cd1a86d60b88c7a485faa3f49c6aff53a456c5e08d0da919fbf71a1e7399
  Stored in directory: /tmp/pip-ephem-wheel-cache-kc1svvvt/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9bb8a8f978> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3055616/17464789 [====>.........................] - ETA: 0s
13582336/17464789 [======================>.......] - ETA: 0s
15917056/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 02:15:27.863346: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 02:15:27.867367: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 02:15:27.867542: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55faeee4eeb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 02:15:27.867555: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 7s - loss: 8.0193 - accuracy: 0.4770 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.9273 - accuracy: 0.4830
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.5478 - accuracy: 0.5077
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5934 - accuracy: 0.5048
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5915 - accuracy: 0.5049
11000/25000 [============>.................] - ETA: 3s - loss: 7.5746 - accuracy: 0.5060
12000/25000 [=============>................] - ETA: 2s - loss: 7.5874 - accuracy: 0.5052
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6305 - accuracy: 0.5024
15000/25000 [=================>............] - ETA: 2s - loss: 7.6339 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6340 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6333 - accuracy: 0.5022
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6392 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 6s 247us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 02:15:39.604369
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 02:15:39.604369  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 02:15:44.959905: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 02:15:44.964617: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 02:15:44.964747: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555def7262f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 02:15:44.964758: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ffa2f8d1d68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 998ms/step - loss: 1.4082 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.3780 - val_crf_viterbi_accuracy: 0.0667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ffa4bd58fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5235 - accuracy: 0.5093
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5286 - accuracy: 0.5090
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.4673 - accuracy: 0.5130
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5184 - accuracy: 0.5097
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5527 - accuracy: 0.5074
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.5497 - accuracy: 0.5076
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5866 - accuracy: 0.5052
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5700 - accuracy: 0.5063
11000/25000 [============>.................] - ETA: 3s - loss: 7.5927 - accuracy: 0.5048
12000/25000 [=============>................] - ETA: 2s - loss: 7.5963 - accuracy: 0.5046
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6316 - accuracy: 0.5023
15000/25000 [=================>............] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6877 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6811 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 6s 244us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ff9f22750b8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:22:34, 9.44kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<17:59:51, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:01<12:39:27, 18.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 852k/862M [00:01<8:52:06, 27.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.44M/862M [00:01<6:11:34, 38.5kB/s].vector_cache/glove.6B.zip:   1%|          | 8.39M/862M [00:01<4:18:41, 55.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<3:00:14, 78.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.3M/862M [00:01<2:05:35, 112kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.6M/862M [00:01<1:27:35, 160kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<1:01:03, 228kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:02<42:39, 325kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:02<29:47, 463kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<20:49, 659kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:02<14:36, 935kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<10:15, 1.32MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<07:14, 1.87MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<12:20, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<10:30, 1.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<09:36, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<07:17, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:23, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:50, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:08, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:17, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:21, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:53, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<04:17, 3.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<18:41, 707kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<14:25, 915kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<10:24, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<10:23, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<08:36, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:13<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<07:31, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<04:56, 2.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<06:31, 1.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<05:53, 2.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<04:26, 2.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:10, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:19<06:57, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<05:31, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<04:09, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:23<05:25, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<04:04, 3.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:52, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<05:24, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:06, 3.09MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:51, 3.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<7:51:40, 26.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<5:30:06, 38.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<3:52:31, 54.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:45:23, 76.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<1:56:19, 108kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<1:21:16, 154kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:12:08, 174kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<51:45, 242kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<36:25, 343kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<28:22, 440kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<21:11, 588kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<15:07, 822kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:24, 925kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:01, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:58, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:29, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:38, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:37, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:59, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:18, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:15, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:38, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:51, 3.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:16, 1.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:04, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:14, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:24, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:46, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:18, 2.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:10, 3.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<3:58:11, 50.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:47:56, 71.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<1:57:38, 102kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:24:44, 141kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:01:53, 193kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<43:50, 272kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<30:50, 385kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<25:08, 472kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<18:51, 628kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<13:29, 877kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<12:07, 972kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<09:44, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:04, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:37, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:54, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:06, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:29, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:08, 2.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:33, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:05, 2.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:49, 3.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:44, 3.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:14, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:53, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:40, 3.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:20, 3.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<7:25:30, 25.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<5:11:40, 36.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<3:39:22, 51.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:35:55, 72.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:49:33, 103kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<1:16:29, 147kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<1:07:55, 166kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<48:42, 231kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<34:16, 327kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<26:29, 422kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<19:43, 566kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<14:04, 792kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:22, 898kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:50, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:09, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:32, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:39, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:56, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<04:16, 2.56MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<36:03, 304kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<26:24, 415kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<18:42, 585kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<15:33, 701kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<13:11, 826kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:46, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<06:58, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:48, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:47, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:44, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:08, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<17:42, 608kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<14:40, 733kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:50, 992kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<07:40, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<31:02, 345kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<22:51, 468kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<16:12, 658kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<13:44, 773kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:44, 989kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:44, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:49, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:35, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:47, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:17, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:36, 2.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:36, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:00, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:44, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:41, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:26, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:17, 3.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:31, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:55, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:42, 2.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:28, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:13, 3.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<51:11, 198kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<36:51, 275kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<26:00, 389kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<18:48, 536kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<6:48:08, 24.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<4:45:27, 35.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<3:20:44, 49.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<2:22:43, 70.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<1:40:18, 99.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<1:10:01, 142kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<59:34, 167kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<42:44, 233kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<30:04, 330kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<23:12, 426kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<18:19, 539kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<13:15, 745kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<09:21, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:13, 742kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<10:18, 952kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<07:27, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:25, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:15, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:35, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<04:01, 2.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<1:11:49, 135kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<51:17, 189kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<36:01, 268kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<27:19, 352kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<21:09, 454kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<15:15, 629kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<10:43, 890kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<18:58, 503kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<14:15, 669kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<10:10, 934kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<09:17, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:30, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:22, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:34, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:23, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:51, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:01, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:39, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:57, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:40, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<03:22, 2.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<30:09, 307kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<22:03, 420kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<15:38, 590kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<13:02, 705kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:06, 909kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:18, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<07:11, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:58, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:24, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:12, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:34, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:25, 2.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:27, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:03, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:03, 2.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:13, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<03:51, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:53, 3.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:40, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:43, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<02:41, 3.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<47:56, 183kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<34:26, 255kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<24:14, 361kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<17:35, 496kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<5:32:30, 26.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<3:52:29, 37.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<2:43:27, 53.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<1:56:13, 74.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<1:21:41, 106kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<56:57, 151kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<59:40, 144kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<42:39, 201kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<29:57, 286kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<22:48, 374kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<17:41, 481kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<12:48, 664kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<09:01, 937kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<1:06:44, 127kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<47:33, 178kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<33:23, 252kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<25:12, 333kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<19:27, 431kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<14:02, 596kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<09:53, 842kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<26:40, 312kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<19:32, 425kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<13:49, 599kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<11:32, 715kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<09:49, 839kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:13, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<05:10, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:21, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:18, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:55, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:35, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:57, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:48, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:48, 745kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:23, 958kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<06:03, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:03, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:55, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:36, 2.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:39, 2.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<08:18, 951kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:37, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:49, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:11, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:19, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:04, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:58, 2.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:53, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:14, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:08, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:58, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<04:24, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:29, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:31, 3.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<25:59, 294kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<18:58, 402kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<13:26, 565kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<11:07, 680kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<08:33, 882kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:09, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<06:02, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:59, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:40, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:04, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<4:45:20, 26.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<3:19:29, 37.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<2:20:03, 52.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<1:39:34, 73.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<1:09:54, 105kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<48:48, 150kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<37:22, 195kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<26:55, 271kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<18:56, 384kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<14:49, 488kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<11:54, 607kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:38, 835kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<06:09, 1.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:32, 1.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:18, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:52, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:19, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:32, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:28, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:32, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:21, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:46, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:49, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:34, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:54, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:03, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:14, 3.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:53, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:26, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:34, 2.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:22, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:03, 2.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:18, 2.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:11, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:55, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:13, 3.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:05, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:44, 2.44MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:05, 3.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:59, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:29, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:47, 2.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:00, 3.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<14:07, 463kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<10:34, 618kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<07:32, 862kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<06:42, 964kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<06:04, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:34, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<47:55, 134kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<34:11, 187kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<24:00, 265kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<18:10, 349kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<13:21, 474kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<09:28, 665kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<08:03, 777kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<06:18, 992kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:32, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:35, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:29, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:25, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:27, 2.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:55, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:05, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:30, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<3:54:28, 25.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<2:43:44, 37.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<1:53:50, 52.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:31:47, 65.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:05:33, 91.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<46:07, 130kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<32:08, 185kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<26:27, 224kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<19:07, 310kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<13:28, 438kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<10:43, 547kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:05, 724kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<05:47, 1.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:23, 1.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:22, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:10, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:32, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:03, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:16, 2.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:55, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:37, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:57, 2.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:39, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:25, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:50, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:34, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:58, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:19, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:41, 3.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:20, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:54, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:09, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:45, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:29, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:51, 2.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:31, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:50, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:13, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:37, 3.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<04:00, 1.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:20, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:54, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:07, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:47, 2.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:37, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:22, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:47, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:24, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:40, 2.99MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:18, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:07, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:36, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:15, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:35, 3.06MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:34, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:01, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:29, 3.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:46, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:27, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:49, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:37, 2.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<2:55:50, 26.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<2:02:42, 38.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<1:25:53, 54.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<1:01:04, 75.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<42:50, 108kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<29:49, 154kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<22:54, 200kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<16:29, 277kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<11:35, 392kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<09:06, 494kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<06:58, 645kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<04:57, 902kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:31, 978kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:38, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:38, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:50, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:49, 2.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:29, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:55, 2.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:24, 3.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:31, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:12, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:38, 2.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:06, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:22, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:50, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:21, 3.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:10, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:57, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:28, 2.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:56, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:43, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:15, 3.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:17, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:01, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:29, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:55, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:46, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:19, 2.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:47, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:39, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:14, 3.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:42, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:59, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:08, 3.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<27:01, 136kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<19:15, 191kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<13:28, 271kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<10:10, 355kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<07:53, 457kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<05:39, 634kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<03:58, 895kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:19, 820kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:23, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:26, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:30, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:23, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:51, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:20, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:11, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:35, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:18, 2.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<2:03:09, 27.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<1:25:44, 38.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<59:48, 54.8kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<42:29, 77.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<29:47, 109kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<21:02, 152kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<15:01, 213kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<10:30, 302kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<07:59, 392kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<06:07, 511kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<04:26, 703kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<03:07, 989kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:22, 911kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:40, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:56, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:01, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:44, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:16, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:33, 1.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:42, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:19, 2.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:59, 2.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:25, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:18, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:58, 2.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:30, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:51, 3.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<07:24, 368kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<05:27, 498kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:51, 699kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<03:17, 810kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:51, 929kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:06, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:29, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:10, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:47, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:18, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:29, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:34, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:52, 2.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:39, 672kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:48, 873kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:00, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:55, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:51, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:24, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:00, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:29, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:56, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:26, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:14, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:54, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:08, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:01, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:45, 2.81MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:00, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:09, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:55, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:44, 2.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<1:11:53, 28.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<49:54, 40.7kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<34:27, 57.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<24:34, 80.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<17:12, 114kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<11:48, 163kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<09:33, 200kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<06:53, 277kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<04:48, 391kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:41, 500kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:58, 620kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:09, 846kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:29, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<04:05, 433kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:02, 582kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<02:08, 813kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:52, 907kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:40, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:52, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<05:48, 282kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<04:13, 387kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<02:57, 545kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:22, 660kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:49, 858kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:17, 1.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:15, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:11, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:53, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:37, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:06, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:45, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:40, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:29, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:38, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:43, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:34, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:23, 3.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<03:13, 382kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<02:22, 516kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:39, 723kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:23, 834kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:05, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:46, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:46, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:46, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:24, 2.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:44, 372kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:56, 518kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<01:18, 733kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<04:27, 214kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<03:11, 296kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<02:12, 419kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:40, 526kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:21, 648kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:59, 882kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:39, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<02:20, 348kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:42, 472kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:10, 664kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:57, 779kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:49, 898kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:36, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:24, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<02:26, 279kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:45, 382kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:11, 540kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:56, 653kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:46, 779kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:33, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:22, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:27, 1.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:22, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:08, 2.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:16, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<15:09, 22.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<10:14, 31.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<06:03, 45.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<04:35, 58.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<03:13, 82.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<02:09, 118kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<01:17, 168kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:57, 212kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:40, 293kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:24, 415kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:15, 520kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:12, 641kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:08, 879kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 915kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 1.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.59MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 987/400000 [00:00<00:40, 9861.19it/s]  0%|          | 1993/400000 [00:00<00:40, 9918.49it/s]  1%|          | 3009/400000 [00:00<00:39, 9988.44it/s]  1%|          | 3975/400000 [00:00<00:40, 9887.22it/s]  1%|          | 4994/400000 [00:00<00:39, 9975.58it/s]  2%|▏         | 6025/400000 [00:00<00:39, 10073.32it/s]  2%|▏         | 7101/400000 [00:00<00:38, 10268.49it/s]  2%|▏         | 8088/400000 [00:00<00:38, 10143.06it/s]  2%|▏         | 9051/400000 [00:00<00:39, 9980.43it/s]   3%|▎         | 10018/400000 [00:01<00:39, 9884.13it/s]  3%|▎         | 10980/400000 [00:01<00:39, 9763.99it/s]  3%|▎         | 11939/400000 [00:01<00:40, 9575.03it/s]  3%|▎         | 12885/400000 [00:01<00:41, 9278.68it/s]  3%|▎         | 13903/400000 [00:01<00:40, 9530.38it/s]  4%|▎         | 14950/400000 [00:01<00:39, 9791.41it/s]  4%|▍         | 15929/400000 [00:01<00:39, 9686.60it/s]  4%|▍         | 16940/400000 [00:01<00:39, 9808.61it/s]  4%|▍         | 17922/400000 [00:01<00:39, 9745.06it/s]  5%|▍         | 18974/400000 [00:01<00:38, 9963.36it/s]  5%|▍         | 19972/400000 [00:02<00:38, 9835.51it/s]  5%|▌         | 20957/400000 [00:02<00:38, 9734.63it/s]  6%|▌         | 22006/400000 [00:02<00:37, 9948.13it/s]  6%|▌         | 23032/400000 [00:02<00:37, 10037.48it/s]  6%|▌         | 24046/400000 [00:02<00:37, 10065.30it/s]  6%|▋         | 25056/400000 [00:02<00:37, 10075.31it/s]  7%|▋         | 26065/400000 [00:02<00:37, 10037.65it/s]  7%|▋         | 27118/400000 [00:02<00:36, 10178.81it/s]  7%|▋         | 28137/400000 [00:02<00:37, 9853.38it/s]   7%|▋         | 29126/400000 [00:02<00:38, 9736.93it/s]  8%|▊         | 30103/400000 [00:03<00:39, 9465.04it/s]  8%|▊         | 31053/400000 [00:03<00:39, 9418.84it/s]  8%|▊         | 32037/400000 [00:03<00:38, 9540.72it/s]  8%|▊         | 33044/400000 [00:03<00:37, 9691.64it/s]  9%|▊         | 34091/400000 [00:03<00:36, 9912.56it/s]  9%|▉         | 35174/400000 [00:03<00:35, 10170.12it/s]  9%|▉         | 36242/400000 [00:03<00:35, 10315.96it/s]  9%|▉         | 37277/400000 [00:03<00:35, 10300.47it/s] 10%|▉         | 38310/400000 [00:03<00:36, 9979.95it/s]  10%|▉         | 39312/400000 [00:03<00:36, 9970.91it/s] 10%|█         | 40312/400000 [00:04<00:36, 9887.84it/s] 10%|█         | 41318/400000 [00:04<00:36, 9938.43it/s] 11%|█         | 42371/400000 [00:04<00:35, 10108.48it/s] 11%|█         | 43400/400000 [00:04<00:35, 10159.43it/s] 11%|█         | 44418/400000 [00:04<00:35, 10118.38it/s] 11%|█▏        | 45431/400000 [00:04<00:35, 9886.44it/s]  12%|█▏        | 46422/400000 [00:04<00:36, 9819.85it/s] 12%|█▏        | 47441/400000 [00:04<00:35, 9925.54it/s] 12%|█▏        | 48435/400000 [00:04<00:36, 9592.87it/s] 12%|█▏        | 49398/400000 [00:05<00:37, 9383.62it/s] 13%|█▎        | 50359/400000 [00:05<00:37, 9448.14it/s] 13%|█▎        | 51328/400000 [00:05<00:36, 9517.22it/s] 13%|█▎        | 52348/400000 [00:05<00:35, 9710.01it/s] 13%|█▎        | 53342/400000 [00:05<00:35, 9776.35it/s] 14%|█▎        | 54322/400000 [00:05<00:35, 9775.06it/s] 14%|█▍        | 55341/400000 [00:05<00:34, 9894.48it/s] 14%|█▍        | 56412/400000 [00:05<00:33, 10124.10it/s] 14%|█▍        | 57479/400000 [00:05<00:33, 10280.56it/s] 15%|█▍        | 58510/400000 [00:05<00:33, 10114.13it/s] 15%|█▍        | 59524/400000 [00:06<00:33, 10042.43it/s] 15%|█▌        | 60577/400000 [00:06<00:33, 10182.75it/s] 15%|█▌        | 61597/400000 [00:06<00:34, 9848.41it/s]  16%|█▌        | 62586/400000 [00:06<00:34, 9755.05it/s] 16%|█▌        | 63618/400000 [00:06<00:33, 9916.37it/s] 16%|█▌        | 64647/400000 [00:06<00:33, 10023.21it/s] 16%|█▋        | 65652/400000 [00:06<00:33, 10028.24it/s] 17%|█▋        | 66733/400000 [00:06<00:32, 10248.48it/s] 17%|█▋        | 67761/400000 [00:06<00:32, 10170.91it/s] 17%|█▋        | 68780/400000 [00:06<00:33, 9909.72it/s]  17%|█▋        | 69774/400000 [00:07<00:33, 9873.22it/s] 18%|█▊        | 70764/400000 [00:07<00:34, 9630.33it/s] 18%|█▊        | 71837/400000 [00:07<00:33, 9934.16it/s] 18%|█▊        | 72927/400000 [00:07<00:32, 10204.59it/s] 18%|█▊        | 73953/400000 [00:07<00:32, 9984.43it/s]  19%|█▊        | 74956/400000 [00:07<00:33, 9806.57it/s] 19%|█▉        | 75941/400000 [00:07<00:33, 9785.20it/s] 19%|█▉        | 76927/400000 [00:07<00:32, 9805.65it/s] 19%|█▉        | 77910/400000 [00:07<00:33, 9708.75it/s] 20%|█▉        | 78883/400000 [00:07<00:33, 9621.60it/s] 20%|█▉        | 79875/400000 [00:08<00:32, 9708.05it/s] 20%|██        | 80907/400000 [00:08<00:32, 9882.11it/s] 20%|██        | 81953/400000 [00:08<00:31, 10046.58it/s] 21%|██        | 83009/400000 [00:08<00:31, 10192.41it/s] 21%|██        | 84051/400000 [00:08<00:30, 10258.28it/s] 21%|██▏       | 85084/400000 [00:08<00:30, 10277.50it/s] 22%|██▏       | 86113/400000 [00:08<00:30, 10229.15it/s] 22%|██▏       | 87137/400000 [00:08<00:30, 10206.01it/s] 22%|██▏       | 88159/400000 [00:08<00:31, 10029.32it/s] 22%|██▏       | 89163/400000 [00:08<00:31, 9878.80it/s]  23%|██▎       | 90153/400000 [00:09<00:32, 9675.41it/s] 23%|██▎       | 91123/400000 [00:09<00:32, 9564.78it/s] 23%|██▎       | 92125/400000 [00:09<00:31, 9695.53it/s] 23%|██▎       | 93121/400000 [00:09<00:31, 9771.48it/s] 24%|██▎       | 94114/400000 [00:09<00:31, 9818.07it/s] 24%|██▍       | 95097/400000 [00:09<00:31, 9730.69it/s] 24%|██▍       | 96071/400000 [00:09<00:31, 9615.45it/s] 24%|██▍       | 97034/400000 [00:09<00:31, 9581.43it/s] 25%|██▍       | 98019/400000 [00:09<00:31, 9660.13it/s] 25%|██▍       | 98986/400000 [00:10<00:31, 9606.50it/s] 25%|██▍       | 99967/400000 [00:10<00:31, 9665.71it/s] 25%|██▌       | 100934/400000 [00:10<00:30, 9666.69it/s] 25%|██▌       | 101942/400000 [00:10<00:30, 9785.16it/s] 26%|██▌       | 102954/400000 [00:10<00:30, 9880.87it/s] 26%|██▌       | 103943/400000 [00:10<00:30, 9817.28it/s] 26%|██▌       | 104926/400000 [00:10<00:30, 9753.79it/s] 26%|██▋       | 105932/400000 [00:10<00:29, 9843.57it/s] 27%|██▋       | 106951/400000 [00:10<00:29, 9943.34it/s] 27%|██▋       | 107952/400000 [00:10<00:29, 9960.81it/s] 27%|██▋       | 108949/400000 [00:11<00:29, 9827.27it/s] 27%|██▋       | 109965/400000 [00:11<00:29, 9923.37it/s] 28%|██▊       | 110964/400000 [00:11<00:29, 9941.58it/s] 28%|██▊       | 111959/400000 [00:11<00:30, 9499.12it/s] 28%|██▊       | 112914/400000 [00:11<00:30, 9426.46it/s] 28%|██▊       | 113928/400000 [00:11<00:29, 9628.31it/s] 29%|██▊       | 114935/400000 [00:11<00:29, 9740.87it/s] 29%|██▉       | 115940/400000 [00:11<00:28, 9829.06it/s] 29%|██▉       | 116925/400000 [00:11<00:29, 9684.00it/s] 29%|██▉       | 117896/400000 [00:11<00:29, 9613.57it/s] 30%|██▉       | 118859/400000 [00:12<00:29, 9492.46it/s] 30%|██▉       | 119810/400000 [00:12<00:30, 9075.60it/s] 30%|███       | 120786/400000 [00:12<00:30, 9269.14it/s] 30%|███       | 121737/400000 [00:12<00:29, 9339.28it/s] 31%|███       | 122691/400000 [00:12<00:29, 9398.49it/s] 31%|███       | 123634/400000 [00:12<00:29, 9288.62it/s] 31%|███       | 124583/400000 [00:12<00:29, 9347.72it/s] 31%|███▏      | 125541/400000 [00:12<00:29, 9413.52it/s] 32%|███▏      | 126509/400000 [00:12<00:28, 9491.09it/s] 32%|███▏      | 127460/400000 [00:12<00:29, 9332.38it/s] 32%|███▏      | 128395/400000 [00:13<00:29, 9158.04it/s] 32%|███▏      | 129315/400000 [00:13<00:29, 9169.25it/s] 33%|███▎      | 130285/400000 [00:13<00:28, 9321.49it/s] 33%|███▎      | 131257/400000 [00:13<00:28, 9435.73it/s] 33%|███▎      | 132287/400000 [00:13<00:27, 9677.86it/s] 33%|███▎      | 133269/400000 [00:13<00:27, 9719.63it/s] 34%|███▎      | 134359/400000 [00:13<00:26, 10043.69it/s] 34%|███▍      | 135400/400000 [00:13<00:26, 10149.89it/s] 34%|███▍      | 136419/400000 [00:13<00:26, 9897.74it/s]  34%|███▍      | 137413/400000 [00:13<00:26, 9888.84it/s] 35%|███▍      | 138405/400000 [00:14<00:26, 9869.81it/s] 35%|███▍      | 139394/400000 [00:14<00:26, 9823.78it/s] 35%|███▌      | 140378/400000 [00:14<00:26, 9807.65it/s] 35%|███▌      | 141360/400000 [00:14<00:27, 9526.39it/s] 36%|███▌      | 142359/400000 [00:14<00:26, 9658.28it/s] 36%|███▌      | 143390/400000 [00:14<00:26, 9843.17it/s] 36%|███▌      | 144377/400000 [00:14<00:26, 9752.85it/s] 36%|███▋      | 145355/400000 [00:14<00:26, 9648.04it/s] 37%|███▋      | 146335/400000 [00:14<00:26, 9691.11it/s] 37%|███▋      | 147306/400000 [00:15<00:26, 9505.72it/s] 37%|███▋      | 148259/400000 [00:15<00:26, 9478.55it/s] 37%|███▋      | 149236/400000 [00:15<00:26, 9562.68it/s] 38%|███▊      | 150200/400000 [00:15<00:26, 9585.12it/s] 38%|███▊      | 151165/400000 [00:15<00:25, 9603.62it/s] 38%|███▊      | 152126/400000 [00:15<00:26, 9461.80it/s] 38%|███▊      | 153089/400000 [00:15<00:25, 9511.42it/s] 39%|███▊      | 154067/400000 [00:15<00:25, 9589.44it/s] 39%|███▉      | 155027/400000 [00:15<00:25, 9542.81it/s] 39%|███▉      | 155982/400000 [00:15<00:25, 9470.80it/s] 39%|███▉      | 156930/400000 [00:16<00:25, 9388.10it/s] 39%|███▉      | 157870/400000 [00:16<00:25, 9345.06it/s] 40%|███▉      | 158805/400000 [00:16<00:26, 9237.89it/s] 40%|███▉      | 159730/400000 [00:16<00:26, 9152.03it/s] 40%|████      | 160649/400000 [00:16<00:26, 9162.67it/s] 40%|████      | 161630/400000 [00:16<00:25, 9346.85it/s] 41%|████      | 162576/400000 [00:16<00:25, 9379.32it/s] 41%|████      | 163591/400000 [00:16<00:24, 9597.23it/s] 41%|████      | 164601/400000 [00:16<00:24, 9740.77it/s] 41%|████▏     | 165598/400000 [00:16<00:23, 9807.43it/s] 42%|████▏     | 166581/400000 [00:17<00:24, 9379.53it/s] 42%|████▏     | 167524/400000 [00:17<00:24, 9363.26it/s] 42%|████▏     | 168470/400000 [00:17<00:24, 9389.26it/s] 42%|████▏     | 169478/400000 [00:17<00:24, 9584.99it/s] 43%|████▎     | 170481/400000 [00:17<00:23, 9712.33it/s] 43%|████▎     | 171458/400000 [00:17<00:23, 9728.08it/s] 43%|████▎     | 172440/400000 [00:17<00:23, 9753.87it/s] 43%|████▎     | 173438/400000 [00:17<00:23, 9820.06it/s] 44%|████▎     | 174435/400000 [00:17<00:22, 9863.99it/s] 44%|████▍     | 175423/400000 [00:17<00:22, 9826.10it/s] 44%|████▍     | 176407/400000 [00:18<00:23, 9543.53it/s] 44%|████▍     | 177364/400000 [00:18<00:23, 9535.62it/s] 45%|████▍     | 178375/400000 [00:18<00:22, 9699.71it/s] 45%|████▍     | 179357/400000 [00:18<00:22, 9734.16it/s] 45%|████▌     | 180332/400000 [00:18<00:22, 9644.44it/s] 45%|████▌     | 181314/400000 [00:18<00:22, 9693.38it/s] 46%|████▌     | 182313/400000 [00:18<00:22, 9778.46it/s] 46%|████▌     | 183307/400000 [00:18<00:22, 9824.02it/s] 46%|████▌     | 184290/400000 [00:18<00:21, 9811.05it/s] 46%|████▋     | 185282/400000 [00:18<00:21, 9842.61it/s] 47%|████▋     | 186267/400000 [00:19<00:21, 9785.39it/s] 47%|████▋     | 187246/400000 [00:19<00:22, 9585.34it/s] 47%|████▋     | 188258/400000 [00:19<00:21, 9738.80it/s] 47%|████▋     | 189270/400000 [00:19<00:21, 9849.74it/s] 48%|████▊     | 190257/400000 [00:19<00:21, 9840.05it/s] 48%|████▊     | 191251/400000 [00:19<00:21, 9867.41it/s] 48%|████▊     | 192239/400000 [00:19<00:21, 9615.89it/s] 48%|████▊     | 193203/400000 [00:19<00:21, 9614.97it/s] 49%|████▊     | 194187/400000 [00:19<00:21, 9679.58it/s] 49%|████▉     | 195156/400000 [00:20<00:21, 9596.47it/s] 49%|████▉     | 196117/400000 [00:20<00:21, 9499.49it/s] 49%|████▉     | 197068/400000 [00:20<00:22, 8982.51it/s] 49%|████▉     | 197973/400000 [00:20<00:22, 8940.21it/s] 50%|████▉     | 198944/400000 [00:20<00:21, 9156.57it/s] 50%|████▉     | 199941/400000 [00:20<00:21, 9384.28it/s] 50%|█████     | 200915/400000 [00:20<00:20, 9487.04it/s] 50%|█████     | 201906/400000 [00:20<00:20, 9607.80it/s] 51%|█████     | 202888/400000 [00:20<00:20, 9668.19it/s] 51%|█████     | 203886/400000 [00:20<00:20, 9757.97it/s] 51%|█████     | 204864/400000 [00:21<00:20, 9582.60it/s] 51%|█████▏    | 205825/400000 [00:21<00:20, 9428.80it/s] 52%|█████▏    | 206770/400000 [00:21<00:20, 9285.32it/s] 52%|█████▏    | 207701/400000 [00:21<00:21, 9141.65it/s] 52%|█████▏    | 208624/400000 [00:21<00:20, 9167.36it/s] 52%|█████▏    | 209574/400000 [00:21<00:20, 9263.35it/s] 53%|█████▎    | 210525/400000 [00:21<00:20, 9333.43it/s] 53%|█████▎    | 211516/400000 [00:21<00:19, 9498.75it/s] 53%|█████▎    | 212575/400000 [00:21<00:19, 9800.65it/s] 53%|█████▎    | 213575/400000 [00:21<00:18, 9857.90it/s] 54%|█████▎    | 214564/400000 [00:22<00:18, 9841.66it/s] 54%|█████▍    | 215556/400000 [00:22<00:18, 9862.70it/s] 54%|█████▍    | 216544/400000 [00:22<00:18, 9864.58it/s] 54%|█████▍    | 217532/400000 [00:22<00:18, 9772.34it/s] 55%|█████▍    | 218543/400000 [00:22<00:18, 9870.55it/s] 55%|█████▍    | 219585/400000 [00:22<00:17, 10028.79it/s] 55%|█████▌    | 220595/400000 [00:22<00:17, 10048.86it/s] 55%|█████▌    | 221601/400000 [00:22<00:17, 9939.91it/s]  56%|█████▌    | 222596/400000 [00:22<00:17, 9935.70it/s] 56%|█████▌    | 223594/400000 [00:22<00:17, 9948.58it/s] 56%|█████▌    | 224590/400000 [00:23<00:17, 9779.12it/s] 56%|█████▋    | 225569/400000 [00:23<00:18, 9622.53it/s] 57%|█████▋    | 226549/400000 [00:23<00:17, 9673.25it/s] 57%|█████▋    | 227524/400000 [00:23<00:17, 9693.98it/s] 57%|█████▋    | 228508/400000 [00:23<00:17, 9735.01it/s] 57%|█████▋    | 229517/400000 [00:23<00:17, 9838.73it/s] 58%|█████▊    | 230509/400000 [00:23<00:17, 9860.90it/s] 58%|█████▊    | 231496/400000 [00:23<00:17, 9657.78it/s] 58%|█████▊    | 232498/400000 [00:23<00:17, 9761.93it/s] 58%|█████▊    | 233501/400000 [00:23<00:16, 9839.30it/s] 59%|█████▊    | 234487/400000 [00:24<00:16, 9844.33it/s] 59%|█████▉    | 235475/400000 [00:24<00:16, 9838.03it/s] 59%|█████▉    | 236460/400000 [00:24<00:16, 9652.76it/s] 59%|█████▉    | 237449/400000 [00:24<00:16, 9719.86it/s] 60%|█████▉    | 238473/400000 [00:24<00:16, 9868.68it/s] 60%|█████▉    | 239483/400000 [00:24<00:16, 9936.64it/s] 60%|██████    | 240478/400000 [00:24<00:16, 9883.68it/s] 60%|██████    | 241468/400000 [00:24<00:16, 9708.51it/s] 61%|██████    | 242441/400000 [00:24<00:16, 9677.56it/s] 61%|██████    | 243449/400000 [00:24<00:15, 9794.66it/s] 61%|██████    | 244430/400000 [00:25<00:15, 9764.80it/s] 61%|██████▏   | 245408/400000 [00:25<00:16, 9637.65it/s] 62%|██████▏   | 246373/400000 [00:25<00:16, 9488.33it/s] 62%|██████▏   | 247367/400000 [00:25<00:15, 9617.93it/s] 62%|██████▏   | 248357/400000 [00:25<00:15, 9698.91it/s] 62%|██████▏   | 249365/400000 [00:25<00:15, 9808.38it/s] 63%|██████▎   | 250347/400000 [00:25<00:15, 9776.41it/s] 63%|██████▎   | 251326/400000 [00:25<00:15, 9720.43it/s] 63%|██████▎   | 252332/400000 [00:25<00:15, 9819.81it/s] 63%|██████▎   | 253315/400000 [00:26<00:14, 9820.23it/s] 64%|██████▎   | 254298/400000 [00:26<00:14, 9780.76it/s] 64%|██████▍   | 255277/400000 [00:26<00:15, 9468.90it/s] 64%|██████▍   | 256258/400000 [00:26<00:15, 9566.95it/s] 64%|██████▍   | 257255/400000 [00:26<00:14, 9684.28it/s] 65%|██████▍   | 258268/400000 [00:26<00:14, 9811.75it/s] 65%|██████▍   | 259251/400000 [00:26<00:14, 9724.54it/s] 65%|██████▌   | 260225/400000 [00:26<00:14, 9631.23it/s] 65%|██████▌   | 261190/400000 [00:26<00:14, 9613.22it/s] 66%|██████▌   | 262214/400000 [00:26<00:14, 9790.57it/s] 66%|██████▌   | 263237/400000 [00:27<00:13, 9917.50it/s] 66%|██████▌   | 264231/400000 [00:27<00:13, 9874.29it/s] 66%|██████▋   | 265230/400000 [00:27<00:13, 9907.64it/s] 67%|██████▋   | 266222/400000 [00:27<00:13, 9687.39it/s] 67%|██████▋   | 267193/400000 [00:27<00:13, 9505.78it/s] 67%|██████▋   | 268146/400000 [00:27<00:13, 9479.57it/s] 67%|██████▋   | 269096/400000 [00:27<00:13, 9461.65it/s] 68%|██████▊   | 270048/400000 [00:27<00:13, 9477.69it/s] 68%|██████▊   | 270997/400000 [00:27<00:13, 9419.28it/s] 68%|██████▊   | 271965/400000 [00:27<00:13, 9495.81it/s] 68%|██████▊   | 272929/400000 [00:28<00:13, 9538.46it/s] 68%|██████▊   | 273892/400000 [00:28<00:13, 9565.23it/s] 69%|██████▊   | 274849/400000 [00:28<00:13, 9534.23it/s] 69%|██████▉   | 275803/400000 [00:28<00:13, 9404.67it/s] 69%|██████▉   | 276745/400000 [00:28<00:13, 9204.51it/s] 69%|██████▉   | 277694/400000 [00:28<00:13, 9287.30it/s] 70%|██████▉   | 278668/400000 [00:28<00:12, 9417.95it/s] 70%|██████▉   | 279612/400000 [00:28<00:12, 9305.90it/s] 70%|███████   | 280544/400000 [00:28<00:12, 9292.58it/s] 70%|███████   | 281475/400000 [00:28<00:13, 8958.38it/s] 71%|███████   | 282419/400000 [00:29<00:12, 9096.44it/s] 71%|███████   | 283392/400000 [00:29<00:12, 9277.00it/s] 71%|███████   | 284341/400000 [00:29<00:12, 9337.77it/s] 71%|███████▏  | 285277/400000 [00:29<00:12, 9183.59it/s] 72%|███████▏  | 286236/400000 [00:29<00:12, 9299.87it/s] 72%|███████▏  | 287172/400000 [00:29<00:12, 9316.11it/s] 72%|███████▏  | 288122/400000 [00:29<00:11, 9369.49it/s] 72%|███████▏  | 289098/400000 [00:29<00:11, 9482.26it/s] 73%|███████▎  | 290064/400000 [00:29<00:11, 9534.67it/s] 73%|███████▎  | 291019/400000 [00:29<00:11, 9484.63it/s] 73%|███████▎  | 291969/400000 [00:30<00:11, 9146.98it/s] 73%|███████▎  | 292887/400000 [00:30<00:11, 9058.79it/s] 73%|███████▎  | 293830/400000 [00:30<00:11, 9165.15it/s] 74%|███████▎  | 294794/400000 [00:30<00:11, 9301.26it/s] 74%|███████▍  | 295741/400000 [00:30<00:11, 9348.40it/s] 74%|███████▍  | 296692/400000 [00:30<00:10, 9393.82it/s] 74%|███████▍  | 297645/400000 [00:30<00:10, 9433.65it/s] 75%|███████▍  | 298599/400000 [00:30<00:10, 9464.68it/s] 75%|███████▍  | 299555/400000 [00:30<00:10, 9493.06it/s] 75%|███████▌  | 300512/400000 [00:31<00:10, 9515.35it/s] 75%|███████▌  | 301464/400000 [00:31<00:10, 9512.30it/s] 76%|███████▌  | 302441/400000 [00:31<00:10, 9586.68it/s] 76%|███████▌  | 303411/400000 [00:31<00:10, 9619.28it/s] 76%|███████▌  | 304374/400000 [00:31<00:10, 9423.63it/s] 76%|███████▋  | 305379/400000 [00:31<00:09, 9601.27it/s] 77%|███████▋  | 306433/400000 [00:31<00:09, 9864.06it/s] 77%|███████▋  | 307423/400000 [00:31<00:09, 9841.98it/s] 77%|███████▋  | 308410/400000 [00:31<00:09, 9687.62it/s] 77%|███████▋  | 309381/400000 [00:31<00:09, 9554.30it/s] 78%|███████▊  | 310339/400000 [00:32<00:09, 9476.17it/s] 78%|███████▊  | 311289/400000 [00:32<00:09, 9395.97it/s] 78%|███████▊  | 312230/400000 [00:32<00:09, 9376.35it/s] 78%|███████▊  | 313216/400000 [00:32<00:09, 9515.39it/s] 79%|███████▊  | 314182/400000 [00:32<00:08, 9555.90it/s] 79%|███████▉  | 315145/400000 [00:32<00:08, 9577.81it/s] 79%|███████▉  | 316104/400000 [00:32<00:08, 9497.85it/s] 79%|███████▉  | 317055/400000 [00:32<00:08, 9466.28it/s] 80%|███████▉  | 318003/400000 [00:32<00:08, 9413.74it/s] 80%|███████▉  | 318985/400000 [00:32<00:08, 9529.04it/s] 80%|███████▉  | 319962/400000 [00:33<00:08, 9597.34it/s] 80%|████████  | 320923/400000 [00:33<00:08, 9501.83it/s] 80%|████████  | 321874/400000 [00:33<00:08, 9475.23it/s] 81%|████████  | 322841/400000 [00:33<00:08, 9531.41it/s] 81%|████████  | 323795/400000 [00:33<00:08, 9443.52it/s] 81%|████████  | 324740/400000 [00:33<00:08, 9173.35it/s] 81%|████████▏ | 325686/400000 [00:33<00:08, 9255.79it/s] 82%|████████▏ | 326614/400000 [00:33<00:07, 9262.40it/s] 82%|████████▏ | 327570/400000 [00:33<00:07, 9349.34it/s] 82%|████████▏ | 328506/400000 [00:33<00:07, 9338.59it/s] 82%|████████▏ | 329499/400000 [00:34<00:07, 9507.59it/s] 83%|████████▎ | 330455/400000 [00:34<00:07, 9520.57it/s] 83%|████████▎ | 331429/400000 [00:34<00:07, 9584.05it/s] 83%|████████▎ | 332389/400000 [00:34<00:07, 9545.80it/s] 83%|████████▎ | 333345/400000 [00:34<00:07, 9468.99it/s] 84%|████████▎ | 334325/400000 [00:34<00:06, 9564.56it/s] 84%|████████▍ | 335283/400000 [00:34<00:06, 9535.19it/s] 84%|████████▍ | 336265/400000 [00:34<00:06, 9618.44it/s] 84%|████████▍ | 337237/400000 [00:34<00:06, 9646.51it/s] 85%|████████▍ | 338203/400000 [00:34<00:06, 9524.68it/s] 85%|████████▍ | 339172/400000 [00:35<00:06, 9572.12it/s] 85%|████████▌ | 340130/400000 [00:35<00:06, 9518.56it/s] 85%|████████▌ | 341083/400000 [00:35<00:06, 9490.29it/s] 86%|████████▌ | 342057/400000 [00:35<00:06, 9562.14it/s] 86%|████████▌ | 343014/400000 [00:35<00:05, 9530.85it/s] 86%|████████▌ | 343968/400000 [00:35<00:05, 9503.13it/s] 86%|████████▌ | 344966/400000 [00:35<00:05, 9639.59it/s] 86%|████████▋ | 345967/400000 [00:35<00:05, 9747.08it/s] 87%|████████▋ | 346973/400000 [00:35<00:05, 9837.35it/s] 87%|████████▋ | 347958/400000 [00:35<00:05, 9791.94it/s] 87%|████████▋ | 348939/400000 [00:36<00:05, 9795.14it/s] 87%|████████▋ | 349919/400000 [00:36<00:05, 9617.82it/s] 88%|████████▊ | 350899/400000 [00:36<00:05, 9670.94it/s] 88%|████████▊ | 351890/400000 [00:36<00:04, 9739.03it/s] 88%|████████▊ | 352865/400000 [00:36<00:04, 9741.27it/s] 88%|████████▊ | 353840/400000 [00:36<00:04, 9738.69it/s] 89%|████████▊ | 354815/400000 [00:36<00:04, 9643.80it/s] 89%|████████▉ | 355780/400000 [00:36<00:04, 9476.79it/s] 89%|████████▉ | 356729/400000 [00:36<00:04, 9384.67it/s] 89%|████████▉ | 357692/400000 [00:36<00:04, 9455.24it/s] 90%|████████▉ | 358690/400000 [00:37<00:04, 9604.94it/s] 90%|████████▉ | 359677/400000 [00:37<00:04, 9681.47it/s] 90%|█████████ | 360647/400000 [00:37<00:04, 9589.75it/s] 90%|█████████ | 361607/400000 [00:37<00:04, 9583.71it/s] 91%|█████████ | 362583/400000 [00:37<00:03, 9633.07it/s] 91%|█████████ | 363547/400000 [00:37<00:03, 9567.87it/s] 91%|█████████ | 364537/400000 [00:37<00:03, 9664.14it/s] 91%|█████████▏| 365504/400000 [00:37<00:03, 9625.24it/s] 92%|█████████▏| 366467/400000 [00:37<00:03, 9404.72it/s] 92%|█████████▏| 367409/400000 [00:38<00:03, 9258.06it/s] 92%|█████████▏| 368365/400000 [00:38<00:03, 9345.61it/s] 92%|█████████▏| 369327/400000 [00:38<00:03, 9426.24it/s] 93%|█████████▎| 370295/400000 [00:38<00:03, 9498.19it/s] 93%|█████████▎| 371258/400000 [00:38<00:03, 9536.30it/s] 93%|█████████▎| 372213/400000 [00:38<00:02, 9342.02it/s] 93%|█████████▎| 373149/400000 [00:38<00:02, 9135.04it/s] 94%|█████████▎| 374065/400000 [00:38<00:02, 9111.98it/s] 94%|█████████▎| 374987/400000 [00:38<00:02, 9143.10it/s] 94%|█████████▍| 375927/400000 [00:38<00:02, 9216.43it/s] 94%|█████████▍| 376856/400000 [00:39<00:02, 9234.73it/s] 94%|█████████▍| 377781/400000 [00:39<00:02, 9221.41it/s] 95%|█████████▍| 378704/400000 [00:39<00:02, 9171.25it/s] 95%|█████████▍| 379668/400000 [00:39<00:02, 9306.24it/s] 95%|█████████▌| 380619/400000 [00:39<00:02, 9364.04it/s] 95%|█████████▌| 381564/400000 [00:39<00:01, 9386.97it/s] 96%|█████████▌| 382504/400000 [00:39<00:01, 9374.09it/s] 96%|█████████▌| 383442/400000 [00:39<00:01, 9368.76it/s] 96%|█████████▌| 384380/400000 [00:39<00:01, 9327.75it/s] 96%|█████████▋| 385327/400000 [00:39<00:01, 9367.82it/s] 97%|█████████▋| 386264/400000 [00:40<00:01, 9362.72it/s] 97%|█████████▋| 387238/400000 [00:40<00:01, 9472.71it/s] 97%|█████████▋| 388190/400000 [00:40<00:01, 9485.04it/s] 97%|█████████▋| 389139/400000 [00:40<00:01, 9283.16it/s] 98%|█████████▊| 390069/400000 [00:40<00:01, 9211.97it/s] 98%|█████████▊| 391013/400000 [00:40<00:00, 9277.46it/s] 98%|█████████▊| 391966/400000 [00:40<00:00, 9351.52it/s] 98%|█████████▊| 392940/400000 [00:40<00:00, 9462.58it/s] 98%|█████████▊| 393899/400000 [00:40<00:00, 9498.12it/s] 99%|█████████▊| 394850/400000 [00:40<00:00, 9476.59it/s] 99%|█████████▉| 395808/400000 [00:41<00:00, 9504.47it/s] 99%|█████████▉| 396769/400000 [00:41<00:00, 9535.45it/s] 99%|█████████▉| 397741/400000 [00:41<00:00, 9587.63it/s]100%|█████████▉| 398701/400000 [00:41<00:00, 9590.07it/s]100%|█████████▉| 399661/400000 [00:41<00:00, 9523.38it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9641.55it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff9ea300ba8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010992555403402434 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.011260268082188125 	 Accuracy: 50

  model saves at 50% accuracy 

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7974aab470> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:12:19.780572
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 03:12:19.784415
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 03:12:19.787796
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 03:12:19.791561
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f796cdfb400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357559.5000
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 281181.8750
Epoch 3/10

1/1 [==============================] - 0s 118ms/step - loss: 167417.0781
Epoch 4/10

1/1 [==============================] - 0s 109ms/step - loss: 88915.3750
Epoch 5/10

1/1 [==============================] - 0s 107ms/step - loss: 45864.9219
Epoch 6/10

1/1 [==============================] - 0s 112ms/step - loss: 24885.4844
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 15005.5205
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 9947.5566
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 7082.5444
Epoch 10/10

1/1 [==============================] - 0s 109ms/step - loss: 5368.4590

  #### Inference Need return ypred, ytrue ######################### 
[[  1.0088881    0.8485119   -0.14694935  -0.96146846  -0.09427631
    0.39417234  -1.1577122    1.6079824   -0.9420249    0.4714212
   -1.9076285    1.5545397    0.38974506   1.2028841    1.4379078
   -1.1241643   -0.45406836  -0.9196108    0.01312166   1.9482825
   -0.83404857  -0.87586474   0.10291883   0.6177322   -2.0240817
    1.1980898   -0.11634612  -1.4458699   -0.20803526   0.58055836
    1.8353764    0.18332493   2.3436983    0.11136574   0.62882185
    0.68089515  -1.1901764   -0.3588693   -1.8801062   -1.20244
   -1.5648707    2.844059    -0.01633608   0.6516123    0.48903847
    1.0361226   -0.08254799  -1.2695385   -0.94033146  -0.7491552
   -1.0622237    0.59399056   1.1712209    0.4668374    0.60827416
    1.2679902    1.3941834    0.24359739  -0.9071827   -1.8184646
    0.10238597  -0.11677134   0.18343657   0.23213083   0.527345
   -0.5812812    2.973819     1.385951    -0.41378054  -0.6260242
   -0.6897275    0.10240015   1.7974985   -0.97825634  -0.03142889
    1.8528981   -1.6373131   -0.31627673   1.8845496   -1.8647368
    2.517099    -1.3722374   -0.22446394  -1.0447451   -1.1568104
    2.2241511    0.24728072   1.3479235    1.3466209   -0.45013678
    0.7786644    1.4844649   -0.03781869   0.67058706   0.68735933
   -1.2019073    0.3747671    1.2889085    1.3042752   -0.1459935
    0.26915076  -0.44555104  -1.445416    -0.36942792   1.129035
    0.5745296   -0.11685792   0.95672     -0.15255451   1.2746159
   -0.6933588   -0.532869     0.16620629   0.15847205   2.5246096
   -0.2916184   -0.46565163   0.95013976  -1.1624687    0.33016294
   -0.46239316   9.460047     9.1666       9.19071      9.044243
    9.620063     9.553202     9.021821     7.9912753    7.522467
    8.279262     9.16381      9.667993     8.504021     6.027046
   10.792724    10.135359     9.461244     8.492902     8.330234
   10.949756    11.70049      9.774288    10.52915      8.289456
    9.461294     8.234345     7.226806     6.17881     10.100629
    7.9660597    9.036941     9.165187     7.858032     8.563918
   10.967102     8.590484     8.393466     8.617091     7.5599895
    7.903485     8.846664     7.6615224   10.14152     10.111389
    9.155918    10.62912     11.498395     9.536009     9.523704
   10.058427     9.307643    10.429335     9.255669     9.435776
    8.478542     8.590352     9.192396     6.5554056    8.065987
    1.0385454    2.9134293    1.7534119    0.6874419    1.4029665
    0.44339442   1.3178893    0.42820513   2.3888745    1.5776479
    0.06091446   0.5099612    1.6392956    1.8777473    0.7760178
    0.6943208    2.0027404    1.8045409    0.11437315   1.3013756
    0.11031491   2.1657896    0.2413075    1.2794152    2.0631328
    0.48706722   0.6988639    0.63229597   0.9509832    1.9217205
    2.0484939    0.6729017    0.7179773    0.68650234   2.4202075
    0.69215083   1.5140188    0.9634932    0.8509441    1.4272492
    0.52584404   1.2432582    1.5379593    1.0952241    3.1572127
    1.2496207    2.3939304    2.0681486    0.47807002   2.2282124
    0.989702     0.3532502    2.2168841    2.291738     0.59351987
    2.0722003    1.5552996    1.6006682    0.99053603   0.8013699
    1.7476804    1.2919655    1.5386828    1.0631857    0.70151305
    0.34273738   0.76234764   2.2870803    1.4773105    0.31187874
    0.10408038   1.1924775    0.89461267   0.74163806   0.12417853
    1.0585622    1.5160105    0.14057118   0.34878314   1.3915311
    1.7834251    0.9422232    1.2796409    1.43368      2.0914993
    0.13920176   3.7780614    0.2503013    1.7427459    0.5278264
    0.2821319    0.73015475   0.61637086   0.3047408    1.189448
    0.23565406   0.7081459    1.03601      0.48000932   2.2662106
    1.1128879    0.9830983    0.56438434   0.11765325   0.36493427
    1.7575843    1.0602145    1.0582228    0.5242929    0.6065275
    0.45312738   0.42152762   2.0213392    0.59192735   1.7359498
    0.465634     0.34779072   1.1686864    1.3407413    0.35743552
    0.23125732   9.044694     9.4385195   10.281472     8.901937
   11.179628     8.808916    10.028174     9.618414    10.30289
    8.800883     7.8296885   11.065034     8.5026865    8.434949
    9.154263    11.872978    10.03121      6.835402    10.237888
    8.402475    10.083903     9.296673    11.470018     8.983673
    9.057794     8.496063     8.328125     8.610857     9.6376915
    8.044366    11.437799     9.694431    10.062993     8.112253
   10.158414     8.86491      8.139449     8.825771     9.163929
   10.327018    10.1915      11.073516     9.700932     7.9732833
    7.706643     8.577079     9.8625      10.5521755    9.950225
    8.448421     8.688092     9.480827     8.271237    10.099713
   10.40869     10.347369    10.958643     9.823296    10.2494
   -3.0894532  -10.834749     8.979102  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:12:29.174023
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6673
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 03:12:29.178704
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8607.06
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 03:12:29.183412
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.8952
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 03:12:29.187169
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.823
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140158956259032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140157726190784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140157726191288
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140157725786352
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140157725786856
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140157725787360

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f794d229278> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.564095
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.534953
grad_step = 000002, loss = 0.511954
grad_step = 000003, loss = 0.487516
grad_step = 000004, loss = 0.462450
grad_step = 000005, loss = 0.442794
grad_step = 000006, loss = 0.433464
grad_step = 000007, loss = 0.420279
grad_step = 000008, loss = 0.402075
grad_step = 000009, loss = 0.385705
grad_step = 000010, loss = 0.373705
grad_step = 000011, loss = 0.362951
grad_step = 000012, loss = 0.350746
grad_step = 000013, loss = 0.337472
grad_step = 000014, loss = 0.324860
grad_step = 000015, loss = 0.314517
grad_step = 000016, loss = 0.305151
grad_step = 000017, loss = 0.294365
grad_step = 000018, loss = 0.282875
grad_step = 000019, loss = 0.272007
grad_step = 000020, loss = 0.261727
grad_step = 000021, loss = 0.251577
grad_step = 000022, loss = 0.241389
grad_step = 000023, loss = 0.231183
grad_step = 000024, loss = 0.221072
grad_step = 000025, loss = 0.211287
grad_step = 000026, loss = 0.202050
grad_step = 000027, loss = 0.193122
grad_step = 000028, loss = 0.184238
grad_step = 000029, loss = 0.175440
grad_step = 000030, loss = 0.166874
grad_step = 000031, loss = 0.158585
grad_step = 000032, loss = 0.150544
grad_step = 000033, loss = 0.142732
grad_step = 000034, loss = 0.135095
grad_step = 000035, loss = 0.127929
grad_step = 000036, loss = 0.121115
grad_step = 000037, loss = 0.114337
grad_step = 000038, loss = 0.107921
grad_step = 000039, loss = 0.101907
grad_step = 000040, loss = 0.096060
grad_step = 000041, loss = 0.090374
grad_step = 000042, loss = 0.084967
grad_step = 000043, loss = 0.079943
grad_step = 000044, loss = 0.075100
grad_step = 000045, loss = 0.070421
grad_step = 000046, loss = 0.066077
grad_step = 000047, loss = 0.061926
grad_step = 000048, loss = 0.057960
grad_step = 000049, loss = 0.054261
grad_step = 000050, loss = 0.050784
grad_step = 000051, loss = 0.047465
grad_step = 000052, loss = 0.044325
grad_step = 000053, loss = 0.041410
grad_step = 000054, loss = 0.038646
grad_step = 000055, loss = 0.036077
grad_step = 000056, loss = 0.033666
grad_step = 000057, loss = 0.031413
grad_step = 000058, loss = 0.029308
grad_step = 000059, loss = 0.027329
grad_step = 000060, loss = 0.025490
grad_step = 000061, loss = 0.023796
grad_step = 000062, loss = 0.022189
grad_step = 000063, loss = 0.020697
grad_step = 000064, loss = 0.019334
grad_step = 000065, loss = 0.018040
grad_step = 000066, loss = 0.016838
grad_step = 000067, loss = 0.015736
grad_step = 000068, loss = 0.014688
grad_step = 000069, loss = 0.013716
grad_step = 000070, loss = 0.012826
grad_step = 000071, loss = 0.011984
grad_step = 000072, loss = 0.011199
grad_step = 000073, loss = 0.010475
grad_step = 000074, loss = 0.009797
grad_step = 000075, loss = 0.009167
grad_step = 000076, loss = 0.008584
grad_step = 000077, loss = 0.008041
grad_step = 000078, loss = 0.007533
grad_step = 000079, loss = 0.007071
grad_step = 000080, loss = 0.006638
grad_step = 000081, loss = 0.006240
grad_step = 000082, loss = 0.005874
grad_step = 000083, loss = 0.005533
grad_step = 000084, loss = 0.005224
grad_step = 000085, loss = 0.004940
grad_step = 000086, loss = 0.004679
grad_step = 000087, loss = 0.004441
grad_step = 000088, loss = 0.004223
grad_step = 000089, loss = 0.004024
grad_step = 000090, loss = 0.003844
grad_step = 000091, loss = 0.003678
grad_step = 000092, loss = 0.003528
grad_step = 000093, loss = 0.003392
grad_step = 000094, loss = 0.003267
grad_step = 000095, loss = 0.003155
grad_step = 000096, loss = 0.003052
grad_step = 000097, loss = 0.002958
grad_step = 000098, loss = 0.002874
grad_step = 000099, loss = 0.002796
grad_step = 000100, loss = 0.002726
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002663
grad_step = 000102, loss = 0.002605
grad_step = 000103, loss = 0.002553
grad_step = 000104, loss = 0.002505
grad_step = 000105, loss = 0.002462
grad_step = 000106, loss = 0.002423
grad_step = 000107, loss = 0.002388
grad_step = 000108, loss = 0.002356
grad_step = 000109, loss = 0.002328
grad_step = 000110, loss = 0.002304
grad_step = 000111, loss = 0.002282
grad_step = 000112, loss = 0.002262
grad_step = 000113, loss = 0.002242
grad_step = 000114, loss = 0.002223
grad_step = 000115, loss = 0.002207
grad_step = 000116, loss = 0.002194
grad_step = 000117, loss = 0.002184
grad_step = 000118, loss = 0.002174
grad_step = 000119, loss = 0.002167
grad_step = 000120, loss = 0.002159
grad_step = 000121, loss = 0.002153
grad_step = 000122, loss = 0.002145
grad_step = 000123, loss = 0.002139
grad_step = 000124, loss = 0.002133
grad_step = 000125, loss = 0.002127
grad_step = 000126, loss = 0.002123
grad_step = 000127, loss = 0.002119
grad_step = 000128, loss = 0.002117
grad_step = 000129, loss = 0.002116
grad_step = 000130, loss = 0.002119
grad_step = 000131, loss = 0.002122
grad_step = 000132, loss = 0.002129
grad_step = 000133, loss = 0.002134
grad_step = 000134, loss = 0.002139
grad_step = 000135, loss = 0.002137
grad_step = 000136, loss = 0.002127
grad_step = 000137, loss = 0.002109
grad_step = 000138, loss = 0.002090
grad_step = 000139, loss = 0.002074
grad_step = 000140, loss = 0.002066
grad_step = 000141, loss = 0.002064
grad_step = 000142, loss = 0.002067
grad_step = 000143, loss = 0.002074
grad_step = 000144, loss = 0.002085
grad_step = 000145, loss = 0.002101
grad_step = 000146, loss = 0.002121
grad_step = 000147, loss = 0.002147
grad_step = 000148, loss = 0.002163
grad_step = 000149, loss = 0.002168
grad_step = 000150, loss = 0.002138
grad_step = 000151, loss = 0.002092
grad_step = 000152, loss = 0.002049
grad_step = 000153, loss = 0.002032
grad_step = 000154, loss = 0.002041
grad_step = 000155, loss = 0.002065
grad_step = 000156, loss = 0.002088
grad_step = 000157, loss = 0.002096
grad_step = 000158, loss = 0.002089
grad_step = 000159, loss = 0.002064
grad_step = 000160, loss = 0.002037
grad_step = 000161, loss = 0.002017
grad_step = 000162, loss = 0.002011
grad_step = 000163, loss = 0.002015
grad_step = 000164, loss = 0.002026
grad_step = 000165, loss = 0.002040
grad_step = 000166, loss = 0.002054
grad_step = 000167, loss = 0.002066
grad_step = 000168, loss = 0.002072
grad_step = 000169, loss = 0.002075
grad_step = 000170, loss = 0.002065
grad_step = 000171, loss = 0.002051
grad_step = 000172, loss = 0.002030
grad_step = 000173, loss = 0.002010
grad_step = 000174, loss = 0.001993
grad_step = 000175, loss = 0.001983
grad_step = 000176, loss = 0.001978
grad_step = 000177, loss = 0.001978
grad_step = 000178, loss = 0.001981
grad_step = 000179, loss = 0.001988
grad_step = 000180, loss = 0.002000
grad_step = 000181, loss = 0.002020
grad_step = 000182, loss = 0.002058
grad_step = 000183, loss = 0.002112
grad_step = 000184, loss = 0.002187
grad_step = 000185, loss = 0.002239
grad_step = 000186, loss = 0.002234
grad_step = 000187, loss = 0.002126
grad_step = 000188, loss = 0.002001
grad_step = 000189, loss = 0.001958
grad_step = 000190, loss = 0.002015
grad_step = 000191, loss = 0.002093
grad_step = 000192, loss = 0.002097
grad_step = 000193, loss = 0.002025
grad_step = 000194, loss = 0.001957
grad_step = 000195, loss = 0.001953
grad_step = 000196, loss = 0.001993
grad_step = 000197, loss = 0.002027
grad_step = 000198, loss = 0.002022
grad_step = 000199, loss = 0.001976
grad_step = 000200, loss = 0.001937
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001937
grad_step = 000202, loss = 0.001963
grad_step = 000203, loss = 0.001983
grad_step = 000204, loss = 0.001978
grad_step = 000205, loss = 0.001955
grad_step = 000206, loss = 0.001930
grad_step = 000207, loss = 0.001920
grad_step = 000208, loss = 0.001927
grad_step = 000209, loss = 0.001941
grad_step = 000210, loss = 0.001949
grad_step = 000211, loss = 0.001947
grad_step = 000212, loss = 0.001937
grad_step = 000213, loss = 0.001923
grad_step = 000214, loss = 0.001911
grad_step = 000215, loss = 0.001904
grad_step = 000216, loss = 0.001903
grad_step = 000217, loss = 0.001906
grad_step = 000218, loss = 0.001910
grad_step = 000219, loss = 0.001915
grad_step = 000220, loss = 0.001922
grad_step = 000221, loss = 0.001928
grad_step = 000222, loss = 0.001936
grad_step = 000223, loss = 0.001947
grad_step = 000224, loss = 0.001960
grad_step = 000225, loss = 0.001981
grad_step = 000226, loss = 0.002002
grad_step = 000227, loss = 0.002026
grad_step = 000228, loss = 0.002042
grad_step = 000229, loss = 0.002044
grad_step = 000230, loss = 0.002019
grad_step = 000231, loss = 0.001973
grad_step = 000232, loss = 0.001920
grad_step = 000233, loss = 0.001880
grad_step = 000234, loss = 0.001866
grad_step = 000235, loss = 0.001875
grad_step = 000236, loss = 0.001898
grad_step = 000237, loss = 0.001922
grad_step = 000238, loss = 0.001940
grad_step = 000239, loss = 0.001944
grad_step = 000240, loss = 0.001936
grad_step = 000241, loss = 0.001917
grad_step = 000242, loss = 0.001892
grad_step = 000243, loss = 0.001869
grad_step = 000244, loss = 0.001853
grad_step = 000245, loss = 0.001844
grad_step = 000246, loss = 0.001843
grad_step = 000247, loss = 0.001846
grad_step = 000248, loss = 0.001854
grad_step = 000249, loss = 0.001865
grad_step = 000250, loss = 0.001879
grad_step = 000251, loss = 0.001900
grad_step = 000252, loss = 0.001929
grad_step = 000253, loss = 0.001968
grad_step = 000254, loss = 0.002015
grad_step = 000255, loss = 0.002060
grad_step = 000256, loss = 0.002079
grad_step = 000257, loss = 0.002053
grad_step = 000258, loss = 0.001974
grad_step = 000259, loss = 0.001880
grad_step = 000260, loss = 0.001820
grad_step = 000261, loss = 0.001817
grad_step = 000262, loss = 0.001857
grad_step = 000263, loss = 0.001904
grad_step = 000264, loss = 0.001923
grad_step = 000265, loss = 0.001903
grad_step = 000266, loss = 0.001855
grad_step = 000267, loss = 0.001811
grad_step = 000268, loss = 0.001794
grad_step = 000269, loss = 0.001805
grad_step = 000270, loss = 0.001829
grad_step = 000271, loss = 0.001850
grad_step = 000272, loss = 0.001857
grad_step = 000273, loss = 0.001848
grad_step = 000274, loss = 0.001827
grad_step = 000275, loss = 0.001802
grad_step = 000276, loss = 0.001782
grad_step = 000277, loss = 0.001771
grad_step = 000278, loss = 0.001769
grad_step = 000279, loss = 0.001773
grad_step = 000280, loss = 0.001782
grad_step = 000281, loss = 0.001794
grad_step = 000282, loss = 0.001810
grad_step = 000283, loss = 0.001834
grad_step = 000284, loss = 0.001858
grad_step = 000285, loss = 0.001872
grad_step = 000286, loss = 0.001872
grad_step = 000287, loss = 0.001866
grad_step = 000288, loss = 0.001861
grad_step = 000289, loss = 0.001853
grad_step = 000290, loss = 0.001832
grad_step = 000291, loss = 0.001799
grad_step = 000292, loss = 0.001764
grad_step = 000293, loss = 0.001740
grad_step = 000294, loss = 0.001735
grad_step = 000295, loss = 0.001744
grad_step = 000296, loss = 0.001764
grad_step = 000297, loss = 0.001784
grad_step = 000298, loss = 0.001797
grad_step = 000299, loss = 0.001807
grad_step = 000300, loss = 0.001819
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001844
grad_step = 000302, loss = 0.001912
grad_step = 000303, loss = 0.001965
grad_step = 000304, loss = 0.001984
grad_step = 000305, loss = 0.001941
grad_step = 000306, loss = 0.001881
grad_step = 000307, loss = 0.001823
grad_step = 000308, loss = 0.001767
grad_step = 000309, loss = 0.001726
grad_step = 000310, loss = 0.001744
grad_step = 000311, loss = 0.001791
grad_step = 000312, loss = 0.001814
grad_step = 000313, loss = 0.001815
grad_step = 000314, loss = 0.001779
grad_step = 000315, loss = 0.001754
grad_step = 000316, loss = 0.001745
grad_step = 000317, loss = 0.001714
grad_step = 000318, loss = 0.001701
grad_step = 000319, loss = 0.001717
grad_step = 000320, loss = 0.001731
grad_step = 000321, loss = 0.001751
grad_step = 000322, loss = 0.001766
grad_step = 000323, loss = 0.001763
grad_step = 000324, loss = 0.001765
grad_step = 000325, loss = 0.001756
grad_step = 000326, loss = 0.001742
grad_step = 000327, loss = 0.001722
grad_step = 000328, loss = 0.001698
grad_step = 000329, loss = 0.001691
grad_step = 000330, loss = 0.001694
grad_step = 000331, loss = 0.001685
grad_step = 000332, loss = 0.001682
grad_step = 000333, loss = 0.001686
grad_step = 000334, loss = 0.001692
grad_step = 000335, loss = 0.001702
grad_step = 000336, loss = 0.001711
grad_step = 000337, loss = 0.001724
grad_step = 000338, loss = 0.001744
grad_step = 000339, loss = 0.001776
grad_step = 000340, loss = 0.001818
grad_step = 000341, loss = 0.001873
grad_step = 000342, loss = 0.001915
grad_step = 000343, loss = 0.001946
grad_step = 000344, loss = 0.001924
grad_step = 000345, loss = 0.001856
grad_step = 000346, loss = 0.001755
grad_step = 000347, loss = 0.001679
grad_step = 000348, loss = 0.001662
grad_step = 000349, loss = 0.001697
grad_step = 000350, loss = 0.001745
grad_step = 000351, loss = 0.001766
grad_step = 000352, loss = 0.001754
grad_step = 000353, loss = 0.001716
grad_step = 000354, loss = 0.001680
grad_step = 000355, loss = 0.001663
grad_step = 000356, loss = 0.001663
grad_step = 000357, loss = 0.001674
grad_step = 000358, loss = 0.001687
grad_step = 000359, loss = 0.001696
grad_step = 000360, loss = 0.001697
grad_step = 000361, loss = 0.001685
grad_step = 000362, loss = 0.001668
grad_step = 000363, loss = 0.001650
grad_step = 000364, loss = 0.001640
grad_step = 000365, loss = 0.001640
grad_step = 000366, loss = 0.001646
grad_step = 000367, loss = 0.001654
grad_step = 000368, loss = 0.001659
grad_step = 000369, loss = 0.001661
grad_step = 000370, loss = 0.001662
grad_step = 000371, loss = 0.001663
grad_step = 000372, loss = 0.001666
grad_step = 000373, loss = 0.001667
grad_step = 000374, loss = 0.001670
grad_step = 000375, loss = 0.001669
grad_step = 000376, loss = 0.001670
grad_step = 000377, loss = 0.001672
grad_step = 000378, loss = 0.001680
grad_step = 000379, loss = 0.001694
grad_step = 000380, loss = 0.001711
grad_step = 000381, loss = 0.001734
grad_step = 000382, loss = 0.001761
grad_step = 000383, loss = 0.001788
grad_step = 000384, loss = 0.001810
grad_step = 000385, loss = 0.001817
grad_step = 000386, loss = 0.001804
grad_step = 000387, loss = 0.001765
grad_step = 000388, loss = 0.001708
grad_step = 000389, loss = 0.001652
grad_step = 000390, loss = 0.001617
grad_step = 000391, loss = 0.001610
grad_step = 000392, loss = 0.001625
grad_step = 000393, loss = 0.001650
grad_step = 000394, loss = 0.001672
grad_step = 000395, loss = 0.001685
grad_step = 000396, loss = 0.001683
grad_step = 000397, loss = 0.001670
grad_step = 000398, loss = 0.001649
grad_step = 000399, loss = 0.001628
grad_step = 000400, loss = 0.001610
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001601
grad_step = 000402, loss = 0.001601
grad_step = 000403, loss = 0.001611
grad_step = 000404, loss = 0.001628
grad_step = 000405, loss = 0.001644
grad_step = 000406, loss = 0.001655
grad_step = 000407, loss = 0.001651
grad_step = 000408, loss = 0.001648
grad_step = 000409, loss = 0.001654
grad_step = 000410, loss = 0.001670
grad_step = 000411, loss = 0.001683
grad_step = 000412, loss = 0.001684
grad_step = 000413, loss = 0.001679
grad_step = 000414, loss = 0.001678
grad_step = 000415, loss = 0.001671
grad_step = 000416, loss = 0.001657
grad_step = 000417, loss = 0.001634
grad_step = 000418, loss = 0.001612
grad_step = 000419, loss = 0.001598
grad_step = 000420, loss = 0.001591
grad_step = 000421, loss = 0.001595
grad_step = 000422, loss = 0.001604
grad_step = 000423, loss = 0.001604
grad_step = 000424, loss = 0.001597
grad_step = 000425, loss = 0.001582
grad_step = 000426, loss = 0.001572
grad_step = 000427, loss = 0.001568
grad_step = 000428, loss = 0.001573
grad_step = 000429, loss = 0.001582
grad_step = 000430, loss = 0.001589
grad_step = 000431, loss = 0.001594
grad_step = 000432, loss = 0.001601
grad_step = 000433, loss = 0.001618
grad_step = 000434, loss = 0.001657
grad_step = 000435, loss = 0.001732
grad_step = 000436, loss = 0.001858
grad_step = 000437, loss = 0.002056
grad_step = 000438, loss = 0.002249
grad_step = 000439, loss = 0.002381
grad_step = 000440, loss = 0.002236
grad_step = 000441, loss = 0.001912
grad_step = 000442, loss = 0.001648
grad_step = 000443, loss = 0.001610
grad_step = 000444, loss = 0.001778
grad_step = 000445, loss = 0.001906
grad_step = 000446, loss = 0.001795
grad_step = 000447, loss = 0.001590
grad_step = 000448, loss = 0.001581
grad_step = 000449, loss = 0.001732
grad_step = 000450, loss = 0.001752
grad_step = 000451, loss = 0.001629
grad_step = 000452, loss = 0.001567
grad_step = 000453, loss = 0.001606
grad_step = 000454, loss = 0.001660
grad_step = 000455, loss = 0.001648
grad_step = 000456, loss = 0.001581
grad_step = 000457, loss = 0.001543
grad_step = 000458, loss = 0.001588
grad_step = 000459, loss = 0.001625
grad_step = 000460, loss = 0.001584
grad_step = 000461, loss = 0.001537
grad_step = 000462, loss = 0.001545
grad_step = 000463, loss = 0.001574
grad_step = 000464, loss = 0.001572
grad_step = 000465, loss = 0.001548
grad_step = 000466, loss = 0.001533
grad_step = 000467, loss = 0.001536
grad_step = 000468, loss = 0.001547
grad_step = 000469, loss = 0.001547
grad_step = 000470, loss = 0.001533
grad_step = 000471, loss = 0.001522
grad_step = 000472, loss = 0.001526
grad_step = 000473, loss = 0.001531
grad_step = 000474, loss = 0.001528
grad_step = 000475, loss = 0.001519
grad_step = 000476, loss = 0.001514
grad_step = 000477, loss = 0.001517
grad_step = 000478, loss = 0.001521
grad_step = 000479, loss = 0.001517
grad_step = 000480, loss = 0.001508
grad_step = 000481, loss = 0.001503
grad_step = 000482, loss = 0.001505
grad_step = 000483, loss = 0.001508
grad_step = 000484, loss = 0.001507
grad_step = 000485, loss = 0.001504
grad_step = 000486, loss = 0.001502
grad_step = 000487, loss = 0.001501
grad_step = 000488, loss = 0.001502
grad_step = 000489, loss = 0.001504
grad_step = 000490, loss = 0.001505
grad_step = 000491, loss = 0.001505
grad_step = 000492, loss = 0.001507
grad_step = 000493, loss = 0.001511
grad_step = 000494, loss = 0.001517
grad_step = 000495, loss = 0.001522
grad_step = 000496, loss = 0.001524
grad_step = 000497, loss = 0.001517
grad_step = 000498, loss = 0.001509
grad_step = 000499, loss = 0.001505
grad_step = 000500, loss = 0.001501
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001495
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

  date_run                              2020-05-11 03:12:49.407968
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.201773
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 03:12:49.414433
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.088503
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 03:12:49.423036
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.133609
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 03:12:49.429186
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.344834
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
0   2020-05-11 03:12:19.780572  ...    mean_absolute_error
1   2020-05-11 03:12:19.784415  ...     mean_squared_error
2   2020-05-11 03:12:19.787796  ...  median_absolute_error
3   2020-05-11 03:12:19.791561  ...               r2_score
4   2020-05-11 03:12:29.174023  ...    mean_absolute_error
5   2020-05-11 03:12:29.178704  ...     mean_squared_error
6   2020-05-11 03:12:29.183412  ...  median_absolute_error
7   2020-05-11 03:12:29.187169  ...               r2_score
8   2020-05-11 03:12:49.407968  ...    mean_absolute_error
9   2020-05-11 03:12:49.414433  ...     mean_squared_error
10  2020-05-11 03:12:49.423036  ...  median_absolute_error
11  2020-05-11 03:12:49.429186  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3588096/9912422 [00:00<00:00, 35739036.72it/s]9920512it [00:00, 34131639.80it/s]                             
0it [00:00, ?it/s]32768it [00:00, 590023.76it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157549.16it/s]1654784it [00:00, 11245938.57it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194693.70it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d4c2c8780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3ce9a11c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d4c27fe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3ce9a11da0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d4c27fe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cfec78cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d4c2c8e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cfec78cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3d4c2c8e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cfec78cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3ce9a0c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa2a2cc81d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d9c30d7d98635b1d979fa9a0221aaf9bd7f3c5cdf133add02dd1034b19da1334
  Stored in directory: /tmp/pip-ephem-wheel-cache-woltvwrg/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa29ab49f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  245760/17464789 [..............................] - ETA: 3s
  557056/17464789 [..............................] - ETA: 3s
  958464/17464789 [>.............................] - ETA: 2s
 1433600/17464789 [=>............................] - ETA: 2s
 1900544/17464789 [==>...........................] - ETA: 2s
 2473984/17464789 [===>..........................] - ETA: 1s
 3088384/17464789 [====>.........................] - ETA: 1s
 3801088/17464789 [=====>........................] - ETA: 1s
 4595712/17464789 [======>.......................] - ETA: 1s
 5398528/17464789 [========>.....................] - ETA: 1s
 6250496/17464789 [=========>....................] - ETA: 1s
 7192576/17464789 [===========>..................] - ETA: 0s
 8134656/17464789 [============>.................] - ETA: 0s
 8765440/17464789 [==============>...............] - ETA: 0s
 9560064/17464789 [===============>..............] - ETA: 0s
10674176/17464789 [=================>............] - ETA: 0s
11845632/17464789 [===================>..........] - ETA: 0s
13074432/17464789 [=====================>........] - ETA: 0s
14344192/17464789 [=======================>......] - ETA: 0s
15810560/17464789 [==========================>...] - ETA: 0s
17309696/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 03:14:17.555840: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 03:14:17.560591: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 03:14:17.560739: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55df88ec7160 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 03:14:17.560752: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6283 - accuracy: 0.5025 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7637 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6385 - accuracy: 0.5018
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7160 - accuracy: 0.4968
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7310 - accuracy: 0.4958
11000/25000 [============>.................] - ETA: 3s - loss: 7.7238 - accuracy: 0.4963
12000/25000 [=============>................] - ETA: 3s - loss: 7.7331 - accuracy: 0.4957
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7527 - accuracy: 0.4944
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7444 - accuracy: 0.4949
15000/25000 [=================>............] - ETA: 2s - loss: 7.7627 - accuracy: 0.4937
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7625 - accuracy: 0.4938
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7478 - accuracy: 0.4947
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7228 - accuracy: 0.4963
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7046 - accuracy: 0.4975
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6926 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
25000/25000 [==============================] - 7s 299us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:14:31.867125
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 03:14:31.867125  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 03:14:38.134728: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 03:14:38.140268: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 03:14:38.140429: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c9ce2342a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 03:14:38.140442: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc77d49dd68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0940 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 2.0221 - val_crf_viterbi_accuracy: 0.0267

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc799924fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3753 - accuracy: 0.5190
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5478 - accuracy: 0.5077
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 3s - loss: 7.6527 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 3s - loss: 7.6551 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6425 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6765 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6542 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 8s 301us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc738490470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:52:30, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:32:40, 15.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:56:04, 21.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:39:44, 31.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.59M/862M [00:01<5:21:01, 44.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.40M/862M [00:01<3:43:16, 63.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:35:48, 90.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<1:48:25, 130kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:01<1:15:28, 185kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<52:34, 264kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:02<36:39, 376kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<25:34, 535kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<17:52, 760kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<12:40, 1.07MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<09:15, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:37:33, 19.3kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<8:08:52, 27.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<5:41:26, 39.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<4:04:25, 54.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<2:53:53, 76.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<2:02:17, 109kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:06<1:25:26, 156kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:08<2:34:52, 85.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<1:49:44, 121kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<1:17:00, 172kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:10<56:52, 233kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<42:30, 311kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<30:18, 436kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:10<21:21, 618kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<20:44, 635kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<15:51, 831kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<11:24, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<11:02, 1.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<10:26, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:57, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<07:39, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<06:43, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<05:01, 2.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:18<06:33, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<05:42, 2.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<04:11, 3.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:20<07:49, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<06:48, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:02, 2.55MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:22<06:32, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<07:11, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<05:35, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<04:04, 3.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<09:57, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<08:16, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:14, 1.75MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:20, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:41, 2.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<06:17, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:59, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:32, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:25, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:07, 3.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:46, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:34, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:09, 2.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<03:48, 3.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:11, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:18, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:43, 2.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:11, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:39, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:17, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:43, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:36, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:49, 3.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<30:54, 395kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<22:53, 533kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<16:18, 747kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<14:14, 853kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<12:26, 976kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<09:18, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:05, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<7:30:11, 26.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<5:15:03, 38.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<3:41:51, 54.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<2:37:56, 76.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<1:50:59, 108kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<1:17:48, 154kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<57:28, 208kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<41:27, 288kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<29:15, 407kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<23:11, 512kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<18:38, 637kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<13:38, 869kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<09:40, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:31:04, 130kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:04:56, 182kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<45:37, 258kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<34:35, 339kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<25:23, 462kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<18:02, 649kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<15:21, 760kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<11:56, 976kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:38, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:46, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<07:19, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:30, 1.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:54, 1.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [00:59<03:59, 2.88MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<05:50, 1.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:15, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<03:58, 2.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:13, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<04:52, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:31, 3.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<15:19, 739kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<11:53, 952kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:33, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<08:36, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:10, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:17, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:20, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:34, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:07, 2.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:32, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:00, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:44, 2.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:14, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:37, 3.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:08, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:42, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:30, 3.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:04, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:28, 2.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:26, 3.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:17<02:33, 4.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<1:22:43, 131kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<58:58, 184kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<41:27, 261kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<31:29, 342kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<24:13, 445kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<17:28, 616kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<12:42, 844kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<6:47:41, 26.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<4:45:12, 37.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<3:19:01, 53.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<2:29:14, 71.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<1:47:02, 99.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<1:15:23, 141kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<52:47, 201kB/s]  .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<40:22, 262kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<29:21, 360kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<20:46, 508kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<16:57, 620kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<12:55, 813kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<09:17, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:57, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:19, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<05:22, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:24, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:02, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:15, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:47, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:34, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:50, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:26, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:22, 3.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:42, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:22, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:16, 2.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:36, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:16, 2.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:14, 3.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:36, 2.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:12, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<02:21, 4.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<47:22, 210kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<35:11, 283kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<25:06, 396kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<19:05, 518kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<15:27, 640kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:14, 878kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<07:57, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<12:27, 789kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<09:43, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:02, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:12, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<06:01, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:45, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:26, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:17, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:12, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:26, 2.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:39, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:13, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:10, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:05, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:03, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:22, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:01, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:01, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<02:50, 3.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<5:56:10, 26.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<4:09:05, 37.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<2:55:13, 53.0kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<2:04:53, 74.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<1:27:46, 106kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<1:01:20, 151kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<46:43, 197kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<33:41, 274kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<23:45, 387kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<18:36, 492kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<14:58, 611kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<10:52, 840kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<07:45, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<08:15, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:34, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:46, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:27, 2.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<19:27, 463kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<15:32, 580kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<11:16, 798kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<08:00, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<09:10, 975kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:20, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<05:21, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:49, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:53, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<03:17, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<34:09, 258kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<24:38, 357kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<17:27, 503kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<12:15, 713kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<43:47, 200kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<31:31, 277kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<22:11, 392kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<17:31, 495kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<13:08, 659kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:21, 923kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<08:34, 1.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:52, 1.25MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:00, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:31, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<05:37, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:21, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:24, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:48, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<02:50, 2.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:07, 3.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<22:21, 376kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<16:29, 509kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<11:42, 714kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<10:08, 821kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<08:47, 948kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:33, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:54, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:58, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:39, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:28, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:47, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:45, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:56, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:27, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:43, 2.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<01:59, 4.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<24:35, 327kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<18:50, 427kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<13:31, 594kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<09:32, 839kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<09:02, 884kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<4:55:52, 27.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<3:26:53, 38.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<2:25:23, 54.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<1:43:20, 76.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<1:12:36, 109kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<51:46, 152kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<37:02, 212kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<26:03, 300kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<19:56, 390kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<14:45, 527kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<10:29, 739kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<09:07, 846kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:10, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<05:10, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:25, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:34, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:21, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:08, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:45, 2.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:41, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:06, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:15, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:28, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:26, 3.04MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:59, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:10, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:17, 3.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<53:44, 136kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<38:20, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<26:55, 270kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<20:27, 354kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<14:56, 484kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<10:35, 680kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<09:04, 790kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<07:04, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:06, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:14, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:06, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:55, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:48, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<6:48:25, 17.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<4:46:15, 24.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<3:20:00, 35.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<2:19:18, 50.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<1:54:53, 60.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<1:21:47, 85.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<57:27, 121kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<40:03, 172kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<33:09, 208kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<23:53, 288kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<16:50, 408kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<13:19, 512kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<10:42, 637kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<07:49, 870kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:32, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:15, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<03:49, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:07, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<5:36:05, 19.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<3:53:39, 28.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<2:44:49, 40.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<1:55:28, 57.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<1:21:16, 80.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<57:31, 114kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<40:15, 162kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<29:31, 220kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<21:57, 296kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<15:37, 414kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<10:56, 588kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<12:55, 497kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<09:41, 663kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<06:53, 927kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:17, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<05:03, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:39, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:02, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<04:06, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:08, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:17, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:58, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:25, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:32, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:12, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:30, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:45, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:54, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:06, 2.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:42, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:13, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:34, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:51, 3.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<09:51, 603kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<07:30, 792kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:21, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<05:05, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:09, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:29, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:01, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:15, 2.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:56, 1.96MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:13, 1.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:29, 2.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:50, 3.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:17, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:09, 2.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:06, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:24, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:44, 3.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:16, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:33, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:37, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:44, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:40, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:25, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:49, 2.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:37, 3.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<3:29:17, 25.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<2:26:09, 36.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:42:20, 51.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:12:58, 72.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<51:14, 103kB/s]   .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<35:42, 146kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<26:52, 193kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<19:25, 267kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<13:41, 378kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<10:33, 486kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<08:28, 605kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<06:09, 831kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<04:19, 1.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<09:14, 547kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<06:59, 723kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<05:00, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:38, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:18, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:15, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<02:19, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<36:49, 134kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<26:15, 187kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<18:24, 266kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<13:55, 349kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<10:14, 474kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<07:14, 665kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:10, 776kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:17, 905kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:55, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:29, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:55, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:09, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:35, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:17, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:43, 2.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:16, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:33, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:01, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:27, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<33:15, 136kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<23:42, 190kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<16:36, 270kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<12:34, 354kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<09:41, 458kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<06:59, 634kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<05:32, 789kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<04:46, 917kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:31, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:30, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:37, 1.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:37, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:40, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:17, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:42, 2.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:09, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:27, 2.84MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:31<01:48, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:20, 3.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:43, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:17, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:12, 3.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:29:30, 26.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<1:44:15, 37.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<1:12:08, 54.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:11:18, 54.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<50:47, 76.8kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<35:37, 109kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<24:46, 156kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<18:44, 205kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<13:29, 284kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<09:27, 402kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<07:27, 506kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<06:00, 627kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:22, 856kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<03:04, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<28:38, 129kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<20:22, 181kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<14:15, 257kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:44, 338kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<08:15, 440kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:56, 609kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:40, 762kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:37, 980kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:35, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:37, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:32, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:56, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:22, 2.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<3:18:53, 17.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<2:19:19, 24.5kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<1:36:51, 35.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<1:07:52, 49.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<47:45, 70.1kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<33:14, 100kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<23:47, 138kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<17:18, 190kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<12:12, 268kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<08:29, 381kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<07:45, 415kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:45, 559kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<04:03, 785kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:33, 886kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:07, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:19, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:17, 939kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:36, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:53, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:00, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:57, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<11:36, 254kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<08:24, 350kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<05:54, 493kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<04:45, 604kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<03:37, 793kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:34, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:27, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:16, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:42, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:14, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:47, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:31, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:07, 2.39MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:58, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<1:42:46, 26.1kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<1:11:29, 37.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<49:37, 52.6kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<35:20, 73.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<24:46, 105kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<17:04, 150kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<13:57, 182kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<10:00, 253kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<06:59, 359kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<05:23, 459kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:14, 582kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:04, 798kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:30, 961kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:59, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:26, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:32, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:32, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:11, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:50, 2.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<16:48, 135kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<11:57, 189kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<08:19, 268kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<06:15, 352kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<04:48, 456kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:26, 633kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<02:23, 894kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<03:15, 654kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:29, 851kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:46, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:42, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:36, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:12, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:52, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:05, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:48, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:00, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:53, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:48, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:36, 3.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:50, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:56, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:32, 3.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:16, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:37, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:14, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:55, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:39, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<02:19, 710kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<02:07, 775kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:35, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:07, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:18, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:19, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:01, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:43, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:07, 1.35MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:07, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:51, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:36, 2.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:01, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:03, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:48, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:35, 2.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:47, 1.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:49, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:27, 2.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:54, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:58, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:44, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:31, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:44, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:48, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:36, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:27, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:34, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:21, 3.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:48, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:37, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:26, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:33, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:33, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:25, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:17, 3.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:07, 851kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:12, 791kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:55, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:39, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:30, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<31:04, 28.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<21:31, 41.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<14:16, 58.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<10:36, 78.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<07:46, 106kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<05:29, 150kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<03:44, 212kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<02:40, 285kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:58, 382kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:22, 536kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:56, 757kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:59, 695kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:58, 708kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:44, 929kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:30, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:29, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:22, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:15, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:22, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:16, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:17, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 3.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 1.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.58MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 892/400000 [00:00<00:44, 8919.35it/s]  0%|          | 1769/400000 [00:00<00:44, 8872.14it/s]  1%|          | 2647/400000 [00:00<00:44, 8842.66it/s]  1%|          | 3519/400000 [00:00<00:45, 8804.33it/s]  1%|          | 4407/400000 [00:00<00:44, 8825.10it/s]  1%|▏         | 5306/400000 [00:00<00:44, 8871.30it/s]  2%|▏         | 6184/400000 [00:00<00:44, 8843.10it/s]  2%|▏         | 7041/400000 [00:00<00:44, 8756.34it/s]  2%|▏         | 7866/400000 [00:00<00:45, 8525.67it/s]  2%|▏         | 8715/400000 [00:01<00:45, 8512.90it/s]  2%|▏         | 9565/400000 [00:01<00:45, 8507.06it/s]  3%|▎         | 10399/400000 [00:01<00:46, 8427.33it/s]  3%|▎         | 11278/400000 [00:01<00:45, 8531.98it/s]  3%|▎         | 12141/400000 [00:01<00:45, 8559.49it/s]  3%|▎         | 13011/400000 [00:01<00:45, 8599.58it/s]  3%|▎         | 13884/400000 [00:01<00:44, 8635.78it/s]  4%|▎         | 14751/400000 [00:01<00:44, 8643.29it/s]  4%|▍         | 15624/400000 [00:01<00:44, 8667.25it/s]  4%|▍         | 16490/400000 [00:01<00:44, 8554.36it/s]  4%|▍         | 17374/400000 [00:02<00:44, 8637.69it/s]  5%|▍         | 18238/400000 [00:02<00:44, 8560.44it/s]  5%|▍         | 19095/400000 [00:02<00:44, 8546.59it/s]  5%|▍         | 19950/400000 [00:02<00:45, 8427.44it/s]  5%|▌         | 20794/400000 [00:02<00:45, 8390.68it/s]  5%|▌         | 21669/400000 [00:02<00:44, 8493.62it/s]  6%|▌         | 22532/400000 [00:02<00:44, 8533.08it/s]  6%|▌         | 23431/400000 [00:02<00:43, 8662.40it/s]  6%|▌         | 24308/400000 [00:02<00:43, 8694.00it/s]  6%|▋         | 25178/400000 [00:02<00:43, 8583.84it/s]  7%|▋         | 26050/400000 [00:03<00:43, 8622.93it/s]  7%|▋         | 26913/400000 [00:03<00:43, 8609.70it/s]  7%|▋         | 27797/400000 [00:03<00:42, 8676.54it/s]  7%|▋         | 28666/400000 [00:03<00:43, 8613.57it/s]  7%|▋         | 29545/400000 [00:03<00:42, 8665.11it/s]  8%|▊         | 30412/400000 [00:03<00:42, 8651.85it/s]  8%|▊         | 31278/400000 [00:03<00:42, 8576.46it/s]  8%|▊         | 32136/400000 [00:03<00:43, 8540.01it/s]  8%|▊         | 33001/400000 [00:03<00:42, 8570.70it/s]  8%|▊         | 33859/400000 [00:03<00:42, 8530.45it/s]  9%|▊         | 34728/400000 [00:04<00:42, 8577.03it/s]  9%|▉         | 35586/400000 [00:04<00:42, 8575.03it/s]  9%|▉         | 36476/400000 [00:04<00:41, 8668.00it/s]  9%|▉         | 37349/400000 [00:04<00:41, 8685.17it/s] 10%|▉         | 38218/400000 [00:04<00:42, 8594.20it/s] 10%|▉         | 39106/400000 [00:04<00:41, 8676.41it/s] 10%|▉         | 39975/400000 [00:04<00:41, 8663.00it/s] 10%|█         | 40876/400000 [00:04<00:40, 8762.97it/s] 10%|█         | 41753/400000 [00:04<00:40, 8751.34it/s] 11%|█         | 42633/400000 [00:04<00:40, 8763.11it/s] 11%|█         | 43510/400000 [00:05<00:41, 8677.96it/s] 11%|█         | 44379/400000 [00:05<00:41, 8625.21it/s] 11%|█▏        | 45242/400000 [00:05<00:41, 8587.50it/s] 12%|█▏        | 46102/400000 [00:05<00:41, 8504.22it/s] 12%|█▏        | 46985/400000 [00:05<00:41, 8598.39it/s] 12%|█▏        | 47846/400000 [00:05<00:41, 8422.95it/s] 12%|█▏        | 48743/400000 [00:05<00:40, 8579.76it/s] 12%|█▏        | 49624/400000 [00:05<00:40, 8645.61it/s] 13%|█▎        | 50521/400000 [00:05<00:39, 8740.25it/s] 13%|█▎        | 51397/400000 [00:05<00:41, 8491.87it/s] 13%|█▎        | 52308/400000 [00:06<00:40, 8665.03it/s] 13%|█▎        | 53199/400000 [00:06<00:39, 8734.72it/s] 14%|█▎        | 54087/400000 [00:06<00:39, 8777.61it/s] 14%|█▎        | 54977/400000 [00:06<00:39, 8813.18it/s] 14%|█▍        | 55893/400000 [00:06<00:38, 8912.41it/s] 14%|█▍        | 56822/400000 [00:06<00:38, 9020.55it/s] 14%|█▍        | 57731/400000 [00:06<00:37, 9039.38it/s] 15%|█▍        | 58636/400000 [00:06<00:37, 9005.04it/s] 15%|█▍        | 59549/400000 [00:06<00:37, 9039.86it/s] 15%|█▌        | 60454/400000 [00:06<00:37, 9035.12it/s] 15%|█▌        | 61368/400000 [00:07<00:37, 9064.08it/s] 16%|█▌        | 62275/400000 [00:07<00:37, 8960.19it/s] 16%|█▌        | 63172/400000 [00:07<00:38, 8643.88it/s] 16%|█▌        | 64040/400000 [00:07<00:40, 8367.26it/s] 16%|█▌        | 64881/400000 [00:07<00:40, 8262.95it/s] 16%|█▋        | 65711/400000 [00:07<00:40, 8170.89it/s] 17%|█▋        | 66531/400000 [00:07<00:41, 8049.00it/s] 17%|█▋        | 67401/400000 [00:07<00:40, 8231.62it/s] 17%|█▋        | 68227/400000 [00:07<00:40, 8186.22it/s] 17%|█▋        | 69093/400000 [00:08<00:39, 8320.44it/s] 17%|█▋        | 69972/400000 [00:08<00:39, 8455.20it/s] 18%|█▊        | 70842/400000 [00:08<00:38, 8524.52it/s] 18%|█▊        | 71722/400000 [00:08<00:38, 8604.86it/s] 18%|█▊        | 72584/400000 [00:08<00:38, 8600.06it/s] 18%|█▊        | 73465/400000 [00:08<00:37, 8661.02it/s] 19%|█▊        | 74344/400000 [00:08<00:37, 8696.68it/s] 19%|█▉        | 75215/400000 [00:08<00:37, 8697.41it/s] 19%|█▉        | 76099/400000 [00:08<00:37, 8739.00it/s] 19%|█▉        | 76980/400000 [00:08<00:36, 8757.48it/s] 19%|█▉        | 77857/400000 [00:09<00:36, 8760.45it/s] 20%|█▉        | 78734/400000 [00:09<00:36, 8761.83it/s] 20%|█▉        | 79611/400000 [00:09<00:36, 8670.77it/s] 20%|██        | 80494/400000 [00:09<00:36, 8716.20it/s] 20%|██        | 81366/400000 [00:09<00:36, 8697.89it/s] 21%|██        | 82245/400000 [00:09<00:36, 8710.87it/s] 21%|██        | 83117/400000 [00:09<00:36, 8640.58it/s] 21%|██        | 83995/400000 [00:09<00:36, 8680.79it/s] 21%|██        | 84885/400000 [00:09<00:36, 8743.40it/s] 21%|██▏       | 85764/400000 [00:09<00:35, 8756.16it/s] 22%|██▏       | 86640/400000 [00:10<00:36, 8591.19it/s] 22%|██▏       | 87500/400000 [00:10<00:37, 8400.91it/s] 22%|██▏       | 88342/400000 [00:10<00:39, 7966.40it/s] 22%|██▏       | 89151/400000 [00:10<00:38, 8001.26it/s] 22%|██▏       | 89956/400000 [00:10<00:39, 7914.96it/s] 23%|██▎       | 90809/400000 [00:10<00:38, 8089.18it/s] 23%|██▎       | 91666/400000 [00:10<00:37, 8227.51it/s] 23%|██▎       | 92525/400000 [00:10<00:36, 8331.63it/s] 23%|██▎       | 93373/400000 [00:10<00:36, 8372.79it/s] 24%|██▎       | 94212/400000 [00:10<00:36, 8268.74it/s] 24%|██▍       | 95064/400000 [00:11<00:36, 8340.68it/s] 24%|██▍       | 95953/400000 [00:11<00:35, 8496.57it/s] 24%|██▍       | 96814/400000 [00:11<00:35, 8529.80it/s] 24%|██▍       | 97694/400000 [00:11<00:35, 8608.51it/s] 25%|██▍       | 98568/400000 [00:11<00:34, 8647.24it/s] 25%|██▍       | 99434/400000 [00:11<00:35, 8543.40it/s] 25%|██▌       | 100297/400000 [00:11<00:34, 8567.95it/s] 25%|██▌       | 101155/400000 [00:11<00:34, 8571.25it/s] 26%|██▌       | 102031/400000 [00:11<00:34, 8626.69it/s] 26%|██▌       | 102934/400000 [00:11<00:33, 8742.81it/s] 26%|██▌       | 103854/400000 [00:12<00:33, 8875.08it/s] 26%|██▌       | 104743/400000 [00:12<00:33, 8832.74it/s] 26%|██▋       | 105627/400000 [00:12<00:34, 8631.51it/s] 27%|██▋       | 106527/400000 [00:12<00:33, 8738.76it/s] 27%|██▋       | 107403/400000 [00:12<00:35, 8289.54it/s] 27%|██▋       | 108268/400000 [00:12<00:34, 8393.69it/s] 27%|██▋       | 109133/400000 [00:12<00:34, 8466.32it/s] 28%|██▊       | 110006/400000 [00:12<00:33, 8543.70it/s] 28%|██▊       | 110891/400000 [00:12<00:33, 8630.89it/s] 28%|██▊       | 111756/400000 [00:12<00:33, 8619.51it/s] 28%|██▊       | 112640/400000 [00:13<00:33, 8682.93it/s] 28%|██▊       | 113562/400000 [00:13<00:32, 8836.00it/s] 29%|██▊       | 114447/400000 [00:13<00:32, 8688.02it/s] 29%|██▉       | 115318/400000 [00:13<00:32, 8680.23it/s] 29%|██▉       | 116188/400000 [00:13<00:33, 8584.32it/s] 29%|██▉       | 117051/400000 [00:13<00:32, 8596.04it/s] 29%|██▉       | 117912/400000 [00:13<00:32, 8565.51it/s] 30%|██▉       | 118770/400000 [00:13<00:32, 8539.49it/s] 30%|██▉       | 119647/400000 [00:13<00:32, 8604.66it/s] 30%|███       | 120514/400000 [00:14<00:32, 8622.59it/s] 30%|███       | 121377/400000 [00:14<00:32, 8594.38it/s] 31%|███       | 122256/400000 [00:14<00:32, 8650.33it/s] 31%|███       | 123122/400000 [00:14<00:32, 8651.82it/s] 31%|███       | 123988/400000 [00:14<00:31, 8631.56it/s] 31%|███       | 124856/400000 [00:14<00:31, 8645.84it/s] 31%|███▏      | 125734/400000 [00:14<00:31, 8684.39it/s] 32%|███▏      | 126603/400000 [00:14<00:31, 8608.35it/s] 32%|███▏      | 127467/400000 [00:14<00:31, 8617.05it/s] 32%|███▏      | 128333/400000 [00:14<00:31, 8629.03it/s] 32%|███▏      | 129197/400000 [00:15<00:31, 8574.64it/s] 33%|███▎      | 130060/400000 [00:15<00:31, 8590.07it/s] 33%|███▎      | 130920/400000 [00:15<00:31, 8461.13it/s] 33%|███▎      | 131787/400000 [00:15<00:31, 8520.34it/s] 33%|███▎      | 132655/400000 [00:15<00:31, 8566.72it/s] 33%|███▎      | 133548/400000 [00:15<00:30, 8672.38it/s] 34%|███▎      | 134416/400000 [00:15<00:30, 8658.76it/s] 34%|███▍      | 135310/400000 [00:15<00:30, 8738.63it/s] 34%|███▍      | 136185/400000 [00:15<00:30, 8712.77it/s] 34%|███▍      | 137057/400000 [00:15<00:30, 8542.54it/s] 34%|███▍      | 137947/400000 [00:16<00:30, 8645.00it/s] 35%|███▍      | 138827/400000 [00:16<00:30, 8689.70it/s] 35%|███▍      | 139717/400000 [00:16<00:29, 8751.05it/s] 35%|███▌      | 140616/400000 [00:16<00:29, 8819.88it/s] 35%|███▌      | 141522/400000 [00:16<00:29, 8889.38it/s] 36%|███▌      | 142412/400000 [00:16<00:29, 8842.98it/s] 36%|███▌      | 143297/400000 [00:16<00:29, 8706.76it/s] 36%|███▌      | 144169/400000 [00:16<00:29, 8640.46it/s] 36%|███▋      | 145054/400000 [00:16<00:29, 8700.90it/s] 36%|███▋      | 145925/400000 [00:16<00:29, 8641.18it/s] 37%|███▋      | 146790/400000 [00:17<00:29, 8641.88it/s] 37%|███▋      | 147680/400000 [00:17<00:28, 8717.66it/s] 37%|███▋      | 148553/400000 [00:17<00:28, 8697.49it/s] 37%|███▋      | 149424/400000 [00:17<00:28, 8657.10it/s] 38%|███▊      | 150290/400000 [00:17<00:28, 8657.17it/s] 38%|███▊      | 151158/400000 [00:17<00:28, 8663.51it/s] 38%|███▊      | 152031/400000 [00:17<00:28, 8683.16it/s] 38%|███▊      | 152944/400000 [00:17<00:28, 8810.79it/s] 38%|███▊      | 153826/400000 [00:17<00:27, 8802.31it/s] 39%|███▊      | 154707/400000 [00:17<00:28, 8679.33it/s] 39%|███▉      | 155588/400000 [00:18<00:28, 8717.12it/s] 39%|███▉      | 156461/400000 [00:18<00:28, 8618.50it/s] 39%|███▉      | 157358/400000 [00:18<00:27, 8718.87it/s] 40%|███▉      | 158249/400000 [00:18<00:27, 8773.78it/s] 40%|███▉      | 159137/400000 [00:18<00:27, 8804.05it/s] 40%|████      | 160045/400000 [00:18<00:27, 8884.48it/s] 40%|████      | 160934/400000 [00:18<00:28, 8399.39it/s] 40%|████      | 161828/400000 [00:18<00:27, 8554.01it/s] 41%|████      | 162743/400000 [00:18<00:27, 8723.79it/s] 41%|████      | 163635/400000 [00:18<00:26, 8779.78it/s] 41%|████      | 164526/400000 [00:19<00:26, 8816.88it/s] 41%|████▏     | 165410/400000 [00:19<00:26, 8802.37it/s] 42%|████▏     | 166292/400000 [00:19<00:26, 8781.90it/s] 42%|████▏     | 167174/400000 [00:19<00:26, 8792.03it/s] 42%|████▏     | 168054/400000 [00:19<00:26, 8791.83it/s] 42%|████▏     | 168934/400000 [00:19<00:26, 8767.95it/s] 42%|████▏     | 169812/400000 [00:19<00:26, 8624.36it/s] 43%|████▎     | 170702/400000 [00:19<00:26, 8705.17it/s] 43%|████▎     | 171606/400000 [00:19<00:25, 8802.65it/s] 43%|████▎     | 172488/400000 [00:19<00:26, 8707.47it/s] 43%|████▎     | 173361/400000 [00:20<00:26, 8712.47it/s] 44%|████▎     | 174240/400000 [00:20<00:25, 8734.28it/s] 44%|████▍     | 175122/400000 [00:20<00:25, 8759.05it/s] 44%|████▍     | 175999/400000 [00:20<00:25, 8713.95it/s] 44%|████▍     | 176871/400000 [00:20<00:25, 8690.92it/s] 44%|████▍     | 177742/400000 [00:20<00:25, 8695.71it/s] 45%|████▍     | 178612/400000 [00:20<00:26, 8267.77it/s] 45%|████▍     | 179470/400000 [00:20<00:26, 8356.87it/s] 45%|████▌     | 180350/400000 [00:20<00:25, 8484.74it/s] 45%|████▌     | 181232/400000 [00:21<00:25, 8581.89it/s] 46%|████▌     | 182117/400000 [00:21<00:25, 8659.34it/s] 46%|████▌     | 182988/400000 [00:21<00:25, 8671.32it/s] 46%|████▌     | 183859/400000 [00:21<00:24, 8681.13it/s] 46%|████▌     | 184739/400000 [00:21<00:24, 8715.81it/s] 46%|████▋     | 185617/400000 [00:21<00:24, 8734.16it/s] 47%|████▋     | 186491/400000 [00:21<00:25, 8530.12it/s] 47%|████▋     | 187366/400000 [00:21<00:24, 8592.88it/s] 47%|████▋     | 188231/400000 [00:21<00:24, 8607.50it/s] 47%|████▋     | 189106/400000 [00:21<00:24, 8648.25it/s] 47%|████▋     | 189972/400000 [00:22<00:25, 8388.92it/s] 48%|████▊     | 190847/400000 [00:22<00:24, 8492.19it/s] 48%|████▊     | 191712/400000 [00:22<00:24, 8536.41it/s] 48%|████▊     | 192581/400000 [00:22<00:24, 8580.25it/s] 48%|████▊     | 193454/400000 [00:22<00:23, 8623.15it/s] 49%|████▊     | 194318/400000 [00:22<00:23, 8593.88it/s] 49%|████▉     | 195199/400000 [00:22<00:23, 8654.58it/s] 49%|████▉     | 196065/400000 [00:22<00:23, 8571.03it/s] 49%|████▉     | 196947/400000 [00:22<00:23, 8643.83it/s] 49%|████▉     | 197829/400000 [00:22<00:23, 8694.90it/s] 50%|████▉     | 198713/400000 [00:23<00:23, 8734.80it/s] 50%|████▉     | 199599/400000 [00:23<00:22, 8771.88it/s] 50%|█████     | 200477/400000 [00:23<00:22, 8732.83it/s] 50%|█████     | 201351/400000 [00:23<00:22, 8719.58it/s] 51%|█████     | 202227/400000 [00:23<00:22, 8729.70it/s] 51%|█████     | 203101/400000 [00:23<00:22, 8636.30it/s] 51%|█████     | 203977/400000 [00:23<00:22, 8671.57it/s] 51%|█████     | 204845/400000 [00:23<00:22, 8658.72it/s] 51%|█████▏    | 205712/400000 [00:23<00:22, 8633.40it/s] 52%|█████▏    | 206584/400000 [00:23<00:22, 8656.45it/s] 52%|█████▏    | 207450/400000 [00:24<00:22, 8602.27it/s] 52%|█████▏    | 208312/400000 [00:24<00:22, 8606.66it/s] 52%|█████▏    | 209173/400000 [00:24<00:22, 8468.04it/s] 53%|█████▎    | 210021/400000 [00:24<00:22, 8418.41it/s] 53%|█████▎    | 210864/400000 [00:24<00:22, 8326.72it/s] 53%|█████▎    | 211698/400000 [00:24<00:22, 8291.35it/s] 53%|█████▎    | 212528/400000 [00:24<00:22, 8291.39it/s] 53%|█████▎    | 213358/400000 [00:24<00:22, 8155.81it/s] 54%|█████▎    | 214222/400000 [00:24<00:22, 8292.90it/s] 54%|█████▍    | 215111/400000 [00:24<00:21, 8461.91it/s] 54%|█████▍    | 215987/400000 [00:25<00:21, 8547.54it/s] 54%|█████▍    | 216844/400000 [00:25<00:21, 8469.76it/s] 54%|█████▍    | 217693/400000 [00:25<00:21, 8352.06it/s] 55%|█████▍    | 218557/400000 [00:25<00:21, 8433.81it/s] 55%|█████▍    | 219439/400000 [00:25<00:21, 8543.61it/s] 55%|█████▌    | 220307/400000 [00:25<00:20, 8583.61it/s] 55%|█████▌    | 221167/400000 [00:25<00:21, 8330.80it/s] 56%|█████▌    | 222025/400000 [00:25<00:21, 8402.35it/s] 56%|█████▌    | 222883/400000 [00:25<00:20, 8454.39it/s] 56%|█████▌    | 223733/400000 [00:25<00:20, 8467.32it/s] 56%|█████▌    | 224611/400000 [00:26<00:20, 8556.55it/s] 56%|█████▋    | 225489/400000 [00:26<00:20, 8619.69it/s] 57%|█████▋    | 226365/400000 [00:26<00:20, 8659.34it/s] 57%|█████▋    | 227232/400000 [00:26<00:20, 8618.31it/s] 57%|█████▋    | 228095/400000 [00:26<00:19, 8609.64it/s] 57%|█████▋    | 228966/400000 [00:26<00:19, 8637.10it/s] 57%|█████▋    | 229847/400000 [00:26<00:19, 8688.01it/s] 58%|█████▊    | 230727/400000 [00:26<00:19, 8721.29it/s] 58%|█████▊    | 231607/400000 [00:26<00:19, 8742.38it/s] 58%|█████▊    | 232482/400000 [00:26<00:19, 8695.07it/s] 58%|█████▊    | 233359/400000 [00:27<00:19, 8716.75it/s] 59%|█████▊    | 234236/400000 [00:27<00:18, 8730.76it/s] 59%|█████▉    | 235110/400000 [00:27<00:19, 8458.53it/s] 59%|█████▉    | 235993/400000 [00:27<00:19, 8566.50it/s] 59%|█████▉    | 236875/400000 [00:27<00:18, 8638.73it/s] 59%|█████▉    | 237750/400000 [00:27<00:18, 8668.95it/s] 60%|█████▉    | 238619/400000 [00:27<00:18, 8673.60it/s] 60%|█████▉    | 239488/400000 [00:27<00:18, 8670.90it/s] 60%|██████    | 240356/400000 [00:27<00:18, 8668.21it/s] 60%|██████    | 241233/400000 [00:27<00:18, 8697.65it/s] 61%|██████    | 242121/400000 [00:28<00:18, 8749.80it/s] 61%|██████    | 243012/400000 [00:28<00:17, 8793.67it/s] 61%|██████    | 243892/400000 [00:28<00:17, 8769.37it/s] 61%|██████    | 244770/400000 [00:28<00:17, 8664.57it/s] 61%|██████▏   | 245644/400000 [00:28<00:17, 8685.28it/s] 62%|██████▏   | 246517/400000 [00:28<00:17, 8695.88it/s] 62%|██████▏   | 247387/400000 [00:28<00:17, 8652.54it/s] 62%|██████▏   | 248260/400000 [00:28<00:17, 8674.84it/s] 62%|██████▏   | 249128/400000 [00:28<00:18, 8153.62it/s] 63%|██████▎   | 250006/400000 [00:29<00:18, 8331.93it/s] 63%|██████▎   | 250892/400000 [00:29<00:17, 8479.46it/s] 63%|██████▎   | 251773/400000 [00:29<00:17, 8574.09it/s] 63%|██████▎   | 252647/400000 [00:29<00:17, 8621.03it/s] 63%|██████▎   | 253514/400000 [00:29<00:16, 8633.65it/s] 64%|██████▎   | 254380/400000 [00:29<00:17, 8557.14it/s] 64%|██████▍   | 255251/400000 [00:29<00:16, 8600.19it/s] 64%|██████▍   | 256113/400000 [00:29<00:16, 8583.65it/s] 64%|██████▍   | 256973/400000 [00:29<00:16, 8501.40it/s] 64%|██████▍   | 257824/400000 [00:29<00:16, 8373.40it/s] 65%|██████▍   | 258689/400000 [00:30<00:16, 8454.34it/s] 65%|██████▍   | 259565/400000 [00:30<00:16, 8541.57it/s] 65%|██████▌   | 260420/400000 [00:30<00:16, 8457.75it/s] 65%|██████▌   | 261282/400000 [00:30<00:16, 8504.42it/s] 66%|██████▌   | 262164/400000 [00:30<00:16, 8596.41it/s] 66%|██████▌   | 263051/400000 [00:30<00:15, 8676.18it/s] 66%|██████▌   | 263929/400000 [00:30<00:15, 8706.79it/s] 66%|██████▌   | 264801/400000 [00:30<00:15, 8587.03it/s] 66%|██████▋   | 265677/400000 [00:30<00:15, 8637.67it/s] 67%|██████▋   | 266542/400000 [00:30<00:15, 8534.38it/s] 67%|██████▋   | 267397/400000 [00:31<00:15, 8501.50it/s] 67%|██████▋   | 268248/400000 [00:31<00:15, 8465.18it/s] 67%|██████▋   | 269095/400000 [00:31<00:15, 8368.46it/s] 67%|██████▋   | 269972/400000 [00:31<00:15, 8483.34it/s] 68%|██████▊   | 270822/400000 [00:31<00:15, 8400.13it/s] 68%|██████▊   | 271672/400000 [00:31<00:15, 8429.75it/s] 68%|██████▊   | 272516/400000 [00:31<00:15, 8183.41it/s] 68%|██████▊   | 273365/400000 [00:31<00:15, 8272.23it/s] 69%|██████▊   | 274225/400000 [00:31<00:15, 8366.45it/s] 69%|██████▉   | 275087/400000 [00:31<00:14, 8440.26it/s] 69%|██████▉   | 275963/400000 [00:32<00:14, 8532.33it/s] 69%|██████▉   | 276844/400000 [00:32<00:14, 8613.61it/s] 69%|██████▉   | 277707/400000 [00:32<00:14, 8602.17it/s] 70%|██████▉   | 278581/400000 [00:32<00:14, 8640.48it/s] 70%|██████▉   | 279446/400000 [00:32<00:14, 8561.03it/s] 70%|███████   | 280303/400000 [00:32<00:13, 8561.56it/s] 70%|███████   | 281176/400000 [00:32<00:13, 8609.63it/s] 71%|███████   | 282038/400000 [00:32<00:14, 8346.87it/s] 71%|███████   | 282875/400000 [00:32<00:14, 8342.95it/s] 71%|███████   | 283711/400000 [00:32<00:14, 8220.36it/s] 71%|███████   | 284546/400000 [00:33<00:13, 8256.53it/s] 71%|███████▏  | 285428/400000 [00:33<00:13, 8417.58it/s] 72%|███████▏  | 286308/400000 [00:33<00:13, 8528.67it/s] 72%|███████▏  | 287183/400000 [00:33<00:13, 8592.61it/s] 72%|███████▏  | 288058/400000 [00:33<00:12, 8638.76it/s] 72%|███████▏  | 288926/400000 [00:33<00:12, 8649.61it/s] 72%|███████▏  | 289812/400000 [00:33<00:12, 8710.27it/s] 73%|███████▎  | 290689/400000 [00:33<00:12, 8727.04it/s] 73%|███████▎  | 291563/400000 [00:33<00:12, 8720.38it/s] 73%|███████▎  | 292436/400000 [00:33<00:12, 8663.74it/s] 73%|███████▎  | 293322/400000 [00:34<00:12, 8721.53it/s] 74%|███████▎  | 294195/400000 [00:34<00:12, 8696.07it/s] 74%|███████▍  | 295065/400000 [00:34<00:12, 8578.91it/s] 74%|███████▍  | 295924/400000 [00:34<00:12, 8560.57it/s] 74%|███████▍  | 296781/400000 [00:34<00:12, 8255.02it/s] 74%|███████▍  | 297650/400000 [00:34<00:12, 8380.49it/s] 75%|███████▍  | 298539/400000 [00:34<00:11, 8524.39it/s] 75%|███████▍  | 299409/400000 [00:34<00:11, 8576.10it/s] 75%|███████▌  | 300285/400000 [00:34<00:11, 8628.90it/s] 75%|███████▌  | 301163/400000 [00:35<00:11, 8672.47it/s] 76%|███████▌  | 302032/400000 [00:35<00:11, 8593.06it/s] 76%|███████▌  | 302893/400000 [00:35<00:11, 8513.74it/s] 76%|███████▌  | 303768/400000 [00:35<00:11, 8582.07it/s] 76%|███████▌  | 304651/400000 [00:35<00:11, 8654.55it/s] 76%|███████▋  | 305518/400000 [00:35<00:11, 8498.51it/s] 77%|███████▋  | 306388/400000 [00:35<00:10, 8556.52it/s] 77%|███████▋  | 307268/400000 [00:35<00:10, 8626.56it/s] 77%|███████▋  | 308158/400000 [00:35<00:10, 8705.40it/s] 77%|███████▋  | 309047/400000 [00:35<00:10, 8759.15it/s] 77%|███████▋  | 309924/400000 [00:36<00:10, 8705.80it/s] 78%|███████▊  | 310797/400000 [00:36<00:10, 8710.71it/s] 78%|███████▊  | 311669/400000 [00:36<00:10, 8611.06it/s] 78%|███████▊  | 312531/400000 [00:36<00:10, 8595.16it/s] 78%|███████▊  | 313416/400000 [00:36<00:09, 8669.96it/s] 79%|███████▊  | 314284/400000 [00:36<00:10, 8489.21it/s] 79%|███████▉  | 315163/400000 [00:36<00:09, 8577.08it/s] 79%|███████▉  | 316031/400000 [00:36<00:09, 8606.60it/s] 79%|███████▉  | 316902/400000 [00:36<00:09, 8636.05it/s] 79%|███████▉  | 317792/400000 [00:36<00:09, 8712.45it/s] 80%|███████▉  | 318664/400000 [00:37<00:09, 8618.32it/s] 80%|███████▉  | 319537/400000 [00:37<00:09, 8649.42it/s] 80%|████████  | 320404/400000 [00:37<00:09, 8654.73it/s] 80%|████████  | 321284/400000 [00:37<00:09, 8696.02it/s] 81%|████████  | 322172/400000 [00:37<00:08, 8748.74it/s] 81%|████████  | 323062/400000 [00:37<00:08, 8793.43it/s] 81%|████████  | 323943/400000 [00:37<00:08, 8798.15it/s] 81%|████████  | 324823/400000 [00:37<00:08, 8757.48it/s] 81%|████████▏ | 325715/400000 [00:37<00:08, 8804.63it/s] 82%|████████▏ | 326608/400000 [00:37<00:08, 8840.62it/s] 82%|████████▏ | 327493/400000 [00:38<00:08, 8784.61it/s] 82%|████████▏ | 328374/400000 [00:38<00:08, 8789.86it/s] 82%|████████▏ | 329267/400000 [00:38<00:08, 8829.20it/s] 83%|████████▎ | 330158/400000 [00:38<00:07, 8852.15it/s] 83%|████████▎ | 331050/400000 [00:38<00:07, 8869.57it/s] 83%|████████▎ | 331938/400000 [00:38<00:07, 8776.15it/s] 83%|████████▎ | 332829/400000 [00:38<00:07, 8814.42it/s] 83%|████████▎ | 333719/400000 [00:38<00:07, 8838.39it/s] 84%|████████▎ | 334612/400000 [00:38<00:07, 8865.45it/s] 84%|████████▍ | 335499/400000 [00:38<00:07, 8819.59it/s] 84%|████████▍ | 336382/400000 [00:39<00:07, 8592.97it/s] 84%|████████▍ | 337266/400000 [00:39<00:07, 8665.58it/s] 85%|████████▍ | 338134/400000 [00:39<00:07, 8645.01it/s] 85%|████████▍ | 339005/400000 [00:39<00:07, 8662.09it/s] 85%|████████▍ | 339874/400000 [00:39<00:06, 8670.15it/s] 85%|████████▌ | 340742/400000 [00:39<00:06, 8613.53it/s] 85%|████████▌ | 341632/400000 [00:39<00:06, 8694.67it/s] 86%|████████▌ | 342505/400000 [00:39<00:06, 8705.09it/s] 86%|████████▌ | 343389/400000 [00:39<00:06, 8744.74it/s] 86%|████████▌ | 344276/400000 [00:39<00:06, 8781.31it/s] 86%|████████▋ | 345155/400000 [00:40<00:06, 8704.25it/s] 87%|████████▋ | 346026/400000 [00:40<00:06, 8688.72it/s] 87%|████████▋ | 346896/400000 [00:40<00:06, 8558.50it/s] 87%|████████▋ | 347785/400000 [00:40<00:06, 8653.78it/s] 87%|████████▋ | 348670/400000 [00:40<00:05, 8711.40it/s] 87%|████████▋ | 349542/400000 [00:40<00:05, 8701.68it/s] 88%|████████▊ | 350430/400000 [00:40<00:05, 8751.50it/s] 88%|████████▊ | 351313/400000 [00:40<00:05, 8774.40it/s] 88%|████████▊ | 352191/400000 [00:40<00:05, 8742.57it/s] 88%|████████▊ | 353066/400000 [00:40<00:05, 8560.29it/s] 88%|████████▊ | 353934/400000 [00:41<00:05, 8595.40it/s] 89%|████████▊ | 354798/400000 [00:41<00:05, 8606.56it/s] 89%|████████▉ | 355660/400000 [00:41<00:05, 8608.22it/s] 89%|████████▉ | 356522/400000 [00:41<00:05, 8376.96it/s] 89%|████████▉ | 357362/400000 [00:41<00:05, 8303.79it/s] 90%|████████▉ | 358194/400000 [00:41<00:05, 8251.01it/s] 90%|████████▉ | 359083/400000 [00:41<00:04, 8432.00it/s] 90%|████████▉ | 359942/400000 [00:41<00:04, 8476.30it/s] 90%|█████████ | 360815/400000 [00:41<00:04, 8549.70it/s] 90%|█████████ | 361671/400000 [00:42<00:04, 8545.06it/s] 91%|█████████ | 362546/400000 [00:42<00:04, 8604.77it/s] 91%|█████████ | 363436/400000 [00:42<00:04, 8688.46it/s] 91%|█████████ | 364325/400000 [00:42<00:04, 8745.84it/s] 91%|█████████▏| 365213/400000 [00:42<00:03, 8784.58it/s] 92%|█████████▏| 366092/400000 [00:42<00:03, 8651.81it/s] 92%|█████████▏| 366958/400000 [00:42<00:03, 8541.86it/s] 92%|█████████▏| 367835/400000 [00:42<00:03, 8607.76it/s] 92%|█████████▏| 368704/400000 [00:42<00:03, 8631.46it/s] 92%|█████████▏| 369578/400000 [00:42<00:03, 8663.43it/s] 93%|█████████▎| 370445/400000 [00:43<00:03, 8641.08it/s] 93%|█████████▎| 371310/400000 [00:43<00:03, 8511.31it/s] 93%|█████████▎| 372193/400000 [00:43<00:03, 8604.41it/s] 93%|█████████▎| 373069/400000 [00:43<00:03, 8648.87it/s] 93%|█████████▎| 373955/400000 [00:43<00:02, 8710.00it/s] 94%|█████████▎| 374827/400000 [00:43<00:02, 8704.44it/s] 94%|█████████▍| 375698/400000 [00:43<00:02, 8695.31it/s] 94%|█████████▍| 376572/400000 [00:43<00:02, 8706.48it/s] 94%|█████████▍| 377443/400000 [00:43<00:02, 8693.32it/s] 95%|█████████▍| 378327/400000 [00:43<00:02, 8735.19it/s] 95%|█████████▍| 379201/400000 [00:44<00:02, 8725.28it/s] 95%|█████████▌| 380074/400000 [00:44<00:02, 8469.58it/s] 95%|█████████▌| 380923/400000 [00:44<00:02, 8382.32it/s] 95%|█████████▌| 381763/400000 [00:44<00:02, 8295.81it/s] 96%|█████████▌| 382594/400000 [00:44<00:02, 8273.21it/s] 96%|█████████▌| 383448/400000 [00:44<00:01, 8351.36it/s] 96%|█████████▌| 384298/400000 [00:44<00:01, 8395.09it/s] 96%|█████████▋| 385192/400000 [00:44<00:01, 8549.12it/s] 97%|█████████▋| 386070/400000 [00:44<00:01, 8615.45it/s] 97%|█████████▋| 386974/400000 [00:44<00:01, 8737.10it/s] 97%|█████████▋| 387859/400000 [00:45<00:01, 8768.98it/s] 97%|█████████▋| 388737/400000 [00:45<00:01, 8633.93it/s] 97%|█████████▋| 389613/400000 [00:45<00:01, 8671.28it/s] 98%|█████████▊| 390485/400000 [00:45<00:01, 8683.21it/s] 98%|█████████▊| 391362/400000 [00:45<00:00, 8708.50it/s] 98%|█████████▊| 392244/400000 [00:45<00:00, 8740.60it/s] 98%|█████████▊| 393119/400000 [00:45<00:00, 8715.40it/s] 98%|█████████▊| 393991/400000 [00:45<00:00, 8633.04it/s] 99%|█████████▊| 394871/400000 [00:45<00:00, 8681.09it/s] 99%|█████████▉| 395761/400000 [00:45<00:00, 8745.29it/s] 99%|█████████▉| 396652/400000 [00:46<00:00, 8791.31it/s] 99%|█████████▉| 397533/400000 [00:46<00:00, 8795.06it/s]100%|█████████▉| 398413/400000 [00:46<00:00, 8605.15it/s]100%|█████████▉| 399275/400000 [00:46<00:00, 8445.69it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8610.61it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc737ec8a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011515465587675997 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011006397148438521 	 Accuracy: 63

  model saves at 63% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15817 out of table with 15720 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15817 out of table with 15720 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

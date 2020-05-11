
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa740050fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 09:13:12.986166
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 09:13:12.991038
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 09:13:12.995140
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 09:13:12.998244
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa74c068470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356358.2188
Epoch 2/10

1/1 [==============================] - 0s 95ms/step - loss: 271713.5312
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 181772.2969
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 101208.0781
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 52974.6406
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 29432.6367
Epoch 7/10

1/1 [==============================] - 0s 89ms/step - loss: 18052.0684
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 12155.5645
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 8638.6182
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 6652.0103

  #### Inference Need return ypred, ytrue ######################### 
[[-2.99386233e-01 -2.10327840e+00 -2.20864356e-01  1.03439033e+00
   1.45799875e-01  1.06951833e+00  1.28431356e+00 -1.07309198e+00
   5.25553167e-01  1.85434133e-01  4.19870019e-01 -3.04169565e-01
   6.57357752e-01 -6.15229249e-01 -6.29880309e-01 -1.24476779e+00
  -1.25604725e+00 -1.42029476e+00  5.08281231e-01 -1.13498592e+00
   1.14054084e-01 -2.64193177e+00  4.40579653e-03 -7.86948502e-01
   1.48951685e+00  2.75708342e+00 -8.31459880e-01  1.71329379e-02
  -1.01709843e-01  1.45838761e+00 -3.37601960e-01  1.24874794e+00
   2.57496893e-01  8.65278840e-02 -7.20600963e-01 -2.81803489e-01
   1.32597387e-01 -1.11434484e+00 -1.87138259e-01 -1.59476876e-01
   2.03361559e+00 -2.06126618e+00 -3.59252095e-01  2.44952273e+00
   7.07912266e-01  1.97765887e+00 -5.51493704e-01  5.63688636e-01
   1.54102492e+00 -7.46173859e-01 -1.30519116e+00 -6.48298562e-01
  -2.75935054e-01 -1.23839819e+00 -1.00982189e-02  5.76960623e-01
  -1.42406583e+00 -2.17317581e-01 -1.91725850e-01 -1.09010994e-01
  -2.35992953e-01  9.23224163e+00  9.12757492e+00  1.01130409e+01
   8.23515511e+00  6.62777472e+00  7.86369896e+00  8.10881519e+00
   1.06849527e+01  7.81449604e+00  9.14280033e+00  6.64440346e+00
   7.66048861e+00  6.30168676e+00  7.89815807e+00  6.45978928e+00
   1.02684460e+01  8.77766800e+00  7.32770014e+00  8.54495811e+00
   6.15056324e+00  8.77455330e+00  8.99190712e+00  9.34380341e+00
   8.64775372e+00  7.61167574e+00  7.63008928e+00  9.73923492e+00
   7.27672625e+00  7.71601486e+00  8.69268990e+00  9.19765568e+00
   7.06984949e+00  8.25138760e+00  7.43602991e+00  8.29311657e+00
   1.04688635e+01  8.97470284e+00  8.87184334e+00  8.91262245e+00
   9.14228153e+00  8.77855778e+00  7.26075268e+00  7.17625809e+00
   7.67456388e+00  8.70477772e+00  6.65803766e+00  6.45465231e+00
   9.35439491e+00  7.49096394e+00  6.73928356e+00  7.96217108e+00
   9.70645237e+00  6.51996136e+00  8.25913906e+00  7.15341425e+00
   8.73868561e+00  1.02490273e+01  1.00865717e+01  9.06380558e+00
  -1.16644025e+00 -1.56870985e+00  5.67702055e-01  1.09593880e+00
  -1.86415946e+00  4.99822378e-01  1.03075302e+00 -4.18669492e-01
   7.45636046e-01 -1.17995346e+00  1.02191830e+00  6.81111753e-01
   1.72353065e+00  7.31353283e-01  8.90356183e-01 -1.22217107e+00
  -5.78607380e-01  7.97955990e-01 -1.76045036e+00  8.81217480e-01
  -1.09553325e+00  6.31395280e-01  1.99490941e+00  3.92664611e-01
  -7.14781582e-01  8.62246990e-01 -1.83993578e-03  1.86652434e+00
   1.09131694e+00 -1.75207484e+00  2.28175545e+00  9.28739607e-02
   4.32673514e-01 -3.14974129e-01  4.39708263e-01 -4.47186410e-01
   6.05830014e-01 -2.17687607e-01 -8.48749399e-01  6.90481365e-01
  -9.63657498e-02  7.56836355e-01 -1.81182289e+00  3.25842023e-01
   2.83836126e-02  1.92520988e+00  4.54300404e-01 -6.18305326e-01
   8.43950748e-01 -1.11554456e+00 -8.66366386e-01  2.30406380e+00
  -1.71085969e-01  3.70717049e-01  5.94959080e-01  2.14659601e-01
   4.69147772e-01 -9.22424912e-01  6.05468869e-01 -2.85696268e-01
   1.01633894e+00  2.54959321e+00  1.67469907e+00  6.80956542e-01
   2.37412786e+00  3.04922533e+00  7.43941367e-01  2.06416988e+00
   1.13493514e+00  3.45377207e+00  4.51635301e-01  5.62276363e-01
   2.09756804e+00  1.97828174e-01  1.61378217e+00  2.06487894e+00
   4.84315991e-01  8.37051749e-01  1.97402298e-01  4.19614077e-01
   2.10575914e+00  9.52522159e-01  2.09930754e+00  9.82014298e-01
   4.95029509e-01  1.47722983e+00  2.79689133e-01  6.11645103e-01
   1.49522924e+00  1.06610596e+00  1.01845264e+00  5.58163285e-01
   1.08535767e-01  4.97329235e-01  2.07912779e+00  1.71689737e+00
   6.96456552e-01  2.87344217e+00  6.59071922e-01  1.40231252e-01
   2.20572233e+00  1.05937290e+00  1.22716486e+00  1.15283453e+00
   3.79696131e-01  2.10900116e+00  1.33158708e+00  2.81380033e+00
   4.32843626e-01  5.32472491e-01  1.44332397e+00  2.68728256e-01
   1.01541567e+00  4.33450997e-01  8.57684851e-01  1.30129039e+00
   2.33698368e+00  1.26919043e+00  1.89837539e+00  1.85263371e+00
   3.71664762e-02  7.01385403e+00  9.81047344e+00  9.99963570e+00
   8.73045921e+00  9.15554333e+00  8.55858803e+00  6.85164928e+00
   9.11254501e+00  1.05298977e+01  7.55266571e+00  8.65165901e+00
   9.53829479e+00  8.13786507e+00  7.65496922e+00  7.51571417e+00
   8.99619770e+00  8.37463188e+00  7.76474333e+00  8.78965759e+00
   9.12155056e+00  8.81749249e+00  9.62992191e+00  8.74967098e+00
   7.62681675e+00  6.49281025e+00  9.38396072e+00  7.87591219e+00
   8.39994240e+00  8.13639641e+00  8.70080853e+00  9.08566380e+00
   9.16465759e+00  7.14907789e+00  7.99322033e+00  7.49265480e+00
   8.30498600e+00  7.06274176e+00  8.79409695e+00  9.09990597e+00
   9.57202435e+00  8.78919411e+00  7.79977798e+00  9.31212425e+00
   8.93785954e+00  7.31494331e+00  9.00309658e+00  5.53504086e+00
   8.09710693e+00  8.52460003e+00  9.17191410e+00  9.46952152e+00
   9.20928288e+00  1.01557980e+01  9.21596432e+00  8.15050793e+00
   7.25914383e+00  8.93900299e+00  8.85276890e+00  9.17474556e+00
   3.32307816e-01  1.66784286e+00  4.73038435e-01  9.32131648e-01
   2.43487263e+00  1.51357174e+00  2.38267660e+00  1.09150827e-01
   1.71834731e+00  1.79469430e+00  5.90818167e-01  1.64377928e+00
   7.64229178e-01  1.35679245e+00  1.61327767e+00  1.83243561e+00
   1.49190402e+00  1.72405648e+00  2.38555956e+00  2.10877395e+00
   2.57191777e-01  1.12934303e+00  3.51053476e+00  1.74908113e+00
   7.35243022e-01  9.51923847e-01  6.13374233e-01  5.25257170e-01
   1.51622963e+00  2.28656578e+00  5.68927884e-01  3.08157980e-01
   2.27371740e+00  1.33306503e+00  2.83769298e+00  4.45422769e-01
   8.30137253e-01  1.54558694e+00  1.10137129e+00  1.81050181e-01
   1.72532773e+00  6.28711104e-01  1.84404111e+00  2.95913601e+00
   1.72094667e+00  1.49200273e+00  2.50452757e-01  2.36726403e+00
   2.46750402e+00  1.62613177e+00  2.71507621e-01  1.33607531e+00
   3.35448623e-01  2.41195142e-01  1.42509580e+00  1.11165047e+00
   2.27618098e-01  1.10220528e+00  3.51033211e-01  1.89192152e+00
  -2.37232161e+00  5.46850634e+00 -5.79621840e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 09:13:22.610787
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    93.885
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 09:13:22.614057
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8836.28
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 09:13:22.617106
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.5197
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 09:13:22.620666
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -790.351
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140355970784672
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140354743659992
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140354743660496
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140354743235080
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140354743235584
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140354743236088

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa739aac3c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.530446
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.496061
grad_step = 000002, loss = 0.465926
grad_step = 000003, loss = 0.432669
grad_step = 000004, loss = 0.395816
grad_step = 000005, loss = 0.361594
grad_step = 000006, loss = 0.333130
grad_step = 000007, loss = 0.312513
grad_step = 000008, loss = 0.294366
grad_step = 000009, loss = 0.280269
grad_step = 000010, loss = 0.264070
grad_step = 000011, loss = 0.246653
grad_step = 000012, loss = 0.232734
grad_step = 000013, loss = 0.222169
grad_step = 000014, loss = 0.211913
grad_step = 000015, loss = 0.199965
grad_step = 000016, loss = 0.186456
grad_step = 000017, loss = 0.172830
grad_step = 000018, loss = 0.160450
grad_step = 000019, loss = 0.150389
grad_step = 000020, loss = 0.142117
grad_step = 000021, loss = 0.132563
grad_step = 000022, loss = 0.121456
grad_step = 000023, loss = 0.111311
grad_step = 000024, loss = 0.103159
grad_step = 000025, loss = 0.096101
grad_step = 000026, loss = 0.089152
grad_step = 000027, loss = 0.081787
grad_step = 000028, loss = 0.074107
grad_step = 000029, loss = 0.067029
grad_step = 000030, loss = 0.061157
grad_step = 000031, loss = 0.055984
grad_step = 000032, loss = 0.050825
grad_step = 000033, loss = 0.045541
grad_step = 000034, loss = 0.040759
grad_step = 000035, loss = 0.036959
grad_step = 000036, loss = 0.033659
grad_step = 000037, loss = 0.030407
grad_step = 000038, loss = 0.027124
grad_step = 000039, loss = 0.023964
grad_step = 000040, loss = 0.021310
grad_step = 000041, loss = 0.019179
grad_step = 000042, loss = 0.017271
grad_step = 000043, loss = 0.015383
grad_step = 000044, loss = 0.013606
grad_step = 000045, loss = 0.012200
grad_step = 000046, loss = 0.011050
grad_step = 000047, loss = 0.009966
grad_step = 000048, loss = 0.008893
grad_step = 000049, loss = 0.007964
grad_step = 000050, loss = 0.007262
grad_step = 000051, loss = 0.006644
grad_step = 000052, loss = 0.006048
grad_step = 000053, loss = 0.005485
grad_step = 000054, loss = 0.005058
grad_step = 000055, loss = 0.004735
grad_step = 000056, loss = 0.004452
grad_step = 000057, loss = 0.004158
grad_step = 000058, loss = 0.003890
grad_step = 000059, loss = 0.003696
grad_step = 000060, loss = 0.003547
grad_step = 000061, loss = 0.003392
grad_step = 000062, loss = 0.003237
grad_step = 000063, loss = 0.003132
grad_step = 000064, loss = 0.003060
grad_step = 000065, loss = 0.002987
grad_step = 000066, loss = 0.002900
grad_step = 000067, loss = 0.002830
grad_step = 000068, loss = 0.002784
grad_step = 000069, loss = 0.002747
grad_step = 000070, loss = 0.002697
grad_step = 000071, loss = 0.002648
grad_step = 000072, loss = 0.002618
grad_step = 000073, loss = 0.002601
grad_step = 000074, loss = 0.002575
grad_step = 000075, loss = 0.002542
grad_step = 000076, loss = 0.002517
grad_step = 000077, loss = 0.002502
grad_step = 000078, loss = 0.002485
grad_step = 000079, loss = 0.002462
grad_step = 000080, loss = 0.002443
grad_step = 000081, loss = 0.002431
grad_step = 000082, loss = 0.002422
grad_step = 000083, loss = 0.002408
grad_step = 000084, loss = 0.002393
grad_step = 000085, loss = 0.002382
grad_step = 000086, loss = 0.002374
grad_step = 000087, loss = 0.002363
grad_step = 000088, loss = 0.002352
grad_step = 000089, loss = 0.002343
grad_step = 000090, loss = 0.002336
grad_step = 000091, loss = 0.002327
grad_step = 000092, loss = 0.002318
grad_step = 000093, loss = 0.002310
grad_step = 000094, loss = 0.002303
grad_step = 000095, loss = 0.002296
grad_step = 000096, loss = 0.002288
grad_step = 000097, loss = 0.002282
grad_step = 000098, loss = 0.002276
grad_step = 000099, loss = 0.002270
grad_step = 000100, loss = 0.002264
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002259
grad_step = 000102, loss = 0.002254
grad_step = 000103, loss = 0.002249
grad_step = 000104, loss = 0.002244
grad_step = 000105, loss = 0.002240
grad_step = 000106, loss = 0.002236
grad_step = 000107, loss = 0.002232
grad_step = 000108, loss = 0.002229
grad_step = 000109, loss = 0.002225
grad_step = 000110, loss = 0.002222
grad_step = 000111, loss = 0.002219
grad_step = 000112, loss = 0.002215
grad_step = 000113, loss = 0.002213
grad_step = 000114, loss = 0.002210
grad_step = 000115, loss = 0.002207
grad_step = 000116, loss = 0.002204
grad_step = 000117, loss = 0.002201
grad_step = 000118, loss = 0.002199
grad_step = 000119, loss = 0.002196
grad_step = 000120, loss = 0.002193
grad_step = 000121, loss = 0.002191
grad_step = 000122, loss = 0.002188
grad_step = 000123, loss = 0.002186
grad_step = 000124, loss = 0.002183
grad_step = 000125, loss = 0.002181
grad_step = 000126, loss = 0.002178
grad_step = 000127, loss = 0.002176
grad_step = 000128, loss = 0.002174
grad_step = 000129, loss = 0.002171
grad_step = 000130, loss = 0.002169
grad_step = 000131, loss = 0.002166
grad_step = 000132, loss = 0.002164
grad_step = 000133, loss = 0.002161
grad_step = 000134, loss = 0.002159
grad_step = 000135, loss = 0.002156
grad_step = 000136, loss = 0.002154
grad_step = 000137, loss = 0.002151
grad_step = 000138, loss = 0.002149
grad_step = 000139, loss = 0.002146
grad_step = 000140, loss = 0.002144
grad_step = 000141, loss = 0.002141
grad_step = 000142, loss = 0.002139
grad_step = 000143, loss = 0.002136
grad_step = 000144, loss = 0.002134
grad_step = 000145, loss = 0.002131
grad_step = 000146, loss = 0.002129
grad_step = 000147, loss = 0.002126
grad_step = 000148, loss = 0.002123
grad_step = 000149, loss = 0.002121
grad_step = 000150, loss = 0.002118
grad_step = 000151, loss = 0.002116
grad_step = 000152, loss = 0.002113
grad_step = 000153, loss = 0.002111
grad_step = 000154, loss = 0.002108
grad_step = 000155, loss = 0.002106
grad_step = 000156, loss = 0.002103
grad_step = 000157, loss = 0.002100
grad_step = 000158, loss = 0.002098
grad_step = 000159, loss = 0.002095
grad_step = 000160, loss = 0.002093
grad_step = 000161, loss = 0.002090
grad_step = 000162, loss = 0.002087
grad_step = 000163, loss = 0.002085
grad_step = 000164, loss = 0.002082
grad_step = 000165, loss = 0.002080
grad_step = 000166, loss = 0.002077
grad_step = 000167, loss = 0.002074
grad_step = 000168, loss = 0.002072
grad_step = 000169, loss = 0.002069
grad_step = 000170, loss = 0.002067
grad_step = 000171, loss = 0.002064
grad_step = 000172, loss = 0.002061
grad_step = 000173, loss = 0.002059
grad_step = 000174, loss = 0.002056
grad_step = 000175, loss = 0.002053
grad_step = 000176, loss = 0.002051
grad_step = 000177, loss = 0.002048
grad_step = 000178, loss = 0.002045
grad_step = 000179, loss = 0.002043
grad_step = 000180, loss = 0.002041
grad_step = 000181, loss = 0.002039
grad_step = 000182, loss = 0.002041
grad_step = 000183, loss = 0.002054
grad_step = 000184, loss = 0.002101
grad_step = 000185, loss = 0.002213
grad_step = 000186, loss = 0.002261
grad_step = 000187, loss = 0.002187
grad_step = 000188, loss = 0.002032
grad_step = 000189, loss = 0.002095
grad_step = 000190, loss = 0.002183
grad_step = 000191, loss = 0.002070
grad_step = 000192, loss = 0.002038
grad_step = 000193, loss = 0.002112
grad_step = 000194, loss = 0.002079
grad_step = 000195, loss = 0.002020
grad_step = 000196, loss = 0.002054
grad_step = 000197, loss = 0.002070
grad_step = 000198, loss = 0.002011
grad_step = 000199, loss = 0.002018
grad_step = 000200, loss = 0.002050
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002005
grad_step = 000202, loss = 0.001999
grad_step = 000203, loss = 0.002028
grad_step = 000204, loss = 0.002004
grad_step = 000205, loss = 0.001986
grad_step = 000206, loss = 0.002003
grad_step = 000207, loss = 0.002000
grad_step = 000208, loss = 0.001980
grad_step = 000209, loss = 0.001980
grad_step = 000210, loss = 0.001989
grad_step = 000211, loss = 0.001978
grad_step = 000212, loss = 0.001965
grad_step = 000213, loss = 0.001972
grad_step = 000214, loss = 0.001973
grad_step = 000215, loss = 0.001959
grad_step = 000216, loss = 0.001957
grad_step = 000217, loss = 0.001962
grad_step = 000218, loss = 0.001956
grad_step = 000219, loss = 0.001948
grad_step = 000220, loss = 0.001948
grad_step = 000221, loss = 0.001948
grad_step = 000222, loss = 0.001944
grad_step = 000223, loss = 0.001939
grad_step = 000224, loss = 0.001936
grad_step = 000225, loss = 0.001936
grad_step = 000226, loss = 0.001934
grad_step = 000227, loss = 0.001928
grad_step = 000228, loss = 0.001924
grad_step = 000229, loss = 0.001924
grad_step = 000230, loss = 0.001922
grad_step = 000231, loss = 0.001918
grad_step = 000232, loss = 0.001914
grad_step = 000233, loss = 0.001912
grad_step = 000234, loss = 0.001910
grad_step = 000235, loss = 0.001908
grad_step = 000236, loss = 0.001905
grad_step = 000237, loss = 0.001902
grad_step = 000238, loss = 0.001898
grad_step = 000239, loss = 0.001896
grad_step = 000240, loss = 0.001894
grad_step = 000241, loss = 0.001892
grad_step = 000242, loss = 0.001889
grad_step = 000243, loss = 0.001886
grad_step = 000244, loss = 0.001883
grad_step = 000245, loss = 0.001880
grad_step = 000246, loss = 0.001876
grad_step = 000247, loss = 0.001873
grad_step = 000248, loss = 0.001870
grad_step = 000249, loss = 0.001866
grad_step = 000250, loss = 0.001862
grad_step = 000251, loss = 0.001859
grad_step = 000252, loss = 0.001855
grad_step = 000253, loss = 0.001851
grad_step = 000254, loss = 0.001848
grad_step = 000255, loss = 0.001845
grad_step = 000256, loss = 0.001847
grad_step = 000257, loss = 0.001861
grad_step = 000258, loss = 0.001916
grad_step = 000259, loss = 0.002063
grad_step = 000260, loss = 0.002329
grad_step = 000261, loss = 0.002553
grad_step = 000262, loss = 0.002376
grad_step = 000263, loss = 0.001916
grad_step = 000264, loss = 0.001937
grad_step = 000265, loss = 0.002199
grad_step = 000266, loss = 0.002019
grad_step = 000267, loss = 0.001824
grad_step = 000268, loss = 0.002003
grad_step = 000269, loss = 0.001986
grad_step = 000270, loss = 0.001814
grad_step = 000271, loss = 0.001895
grad_step = 000272, loss = 0.001930
grad_step = 000273, loss = 0.001810
grad_step = 000274, loss = 0.001837
grad_step = 000275, loss = 0.001880
grad_step = 000276, loss = 0.001815
grad_step = 000277, loss = 0.001796
grad_step = 000278, loss = 0.001838
grad_step = 000279, loss = 0.001813
grad_step = 000280, loss = 0.001774
grad_step = 000281, loss = 0.001802
grad_step = 000282, loss = 0.001805
grad_step = 000283, loss = 0.001762
grad_step = 000284, loss = 0.001775
grad_step = 000285, loss = 0.001788
grad_step = 000286, loss = 0.001758
grad_step = 000287, loss = 0.001753
grad_step = 000288, loss = 0.001767
grad_step = 000289, loss = 0.001757
grad_step = 000290, loss = 0.001740
grad_step = 000291, loss = 0.001741
grad_step = 000292, loss = 0.001747
grad_step = 000293, loss = 0.001736
grad_step = 000294, loss = 0.001726
grad_step = 000295, loss = 0.001730
grad_step = 000296, loss = 0.001730
grad_step = 000297, loss = 0.001719
grad_step = 000298, loss = 0.001714
grad_step = 000299, loss = 0.001718
grad_step = 000300, loss = 0.001714
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001706
grad_step = 000302, loss = 0.001703
grad_step = 000303, loss = 0.001705
grad_step = 000304, loss = 0.001702
grad_step = 000305, loss = 0.001695
grad_step = 000306, loss = 0.001692
grad_step = 000307, loss = 0.001692
grad_step = 000308, loss = 0.001691
grad_step = 000309, loss = 0.001686
grad_step = 000310, loss = 0.001682
grad_step = 000311, loss = 0.001681
grad_step = 000312, loss = 0.001680
grad_step = 000313, loss = 0.001677
grad_step = 000314, loss = 0.001673
grad_step = 000315, loss = 0.001671
grad_step = 000316, loss = 0.001669
grad_step = 000317, loss = 0.001667
grad_step = 000318, loss = 0.001665
grad_step = 000319, loss = 0.001663
grad_step = 000320, loss = 0.001661
grad_step = 000321, loss = 0.001658
grad_step = 000322, loss = 0.001655
grad_step = 000323, loss = 0.001653
grad_step = 000324, loss = 0.001652
grad_step = 000325, loss = 0.001650
grad_step = 000326, loss = 0.001648
grad_step = 000327, loss = 0.001646
grad_step = 000328, loss = 0.001644
grad_step = 000329, loss = 0.001641
grad_step = 000330, loss = 0.001639
grad_step = 000331, loss = 0.001637
grad_step = 000332, loss = 0.001634
grad_step = 000333, loss = 0.001632
grad_step = 000334, loss = 0.001630
grad_step = 000335, loss = 0.001628
grad_step = 000336, loss = 0.001625
grad_step = 000337, loss = 0.001623
grad_step = 000338, loss = 0.001621
grad_step = 000339, loss = 0.001619
grad_step = 000340, loss = 0.001618
grad_step = 000341, loss = 0.001617
grad_step = 000342, loss = 0.001618
grad_step = 000343, loss = 0.001620
grad_step = 000344, loss = 0.001626
grad_step = 000345, loss = 0.001638
grad_step = 000346, loss = 0.001659
grad_step = 000347, loss = 0.001690
grad_step = 000348, loss = 0.001728
grad_step = 000349, loss = 0.001763
grad_step = 000350, loss = 0.001765
grad_step = 000351, loss = 0.001725
grad_step = 000352, loss = 0.001653
grad_step = 000353, loss = 0.001598
grad_step = 000354, loss = 0.001593
grad_step = 000355, loss = 0.001625
grad_step = 000356, loss = 0.001653
grad_step = 000357, loss = 0.001644
grad_step = 000358, loss = 0.001608
grad_step = 000359, loss = 0.001579
grad_step = 000360, loss = 0.001578
grad_step = 000361, loss = 0.001597
grad_step = 000362, loss = 0.001611
grad_step = 000363, loss = 0.001603
grad_step = 000364, loss = 0.001582
grad_step = 000365, loss = 0.001565
grad_step = 000366, loss = 0.001563
grad_step = 000367, loss = 0.001570
grad_step = 000368, loss = 0.001576
grad_step = 000369, loss = 0.001575
grad_step = 000370, loss = 0.001568
grad_step = 000371, loss = 0.001560
grad_step = 000372, loss = 0.001553
grad_step = 000373, loss = 0.001549
grad_step = 000374, loss = 0.001546
grad_step = 000375, loss = 0.001546
grad_step = 000376, loss = 0.001549
grad_step = 000377, loss = 0.001552
grad_step = 000378, loss = 0.001557
grad_step = 000379, loss = 0.001559
grad_step = 000380, loss = 0.001562
grad_step = 000381, loss = 0.001561
grad_step = 000382, loss = 0.001560
grad_step = 000383, loss = 0.001559
grad_step = 000384, loss = 0.001557
grad_step = 000385, loss = 0.001555
grad_step = 000386, loss = 0.001549
grad_step = 000387, loss = 0.001543
grad_step = 000388, loss = 0.001538
grad_step = 000389, loss = 0.001536
grad_step = 000390, loss = 0.001540
grad_step = 000391, loss = 0.001545
grad_step = 000392, loss = 0.001556
grad_step = 000393, loss = 0.001562
grad_step = 000394, loss = 0.001569
grad_step = 000395, loss = 0.001573
grad_step = 000396, loss = 0.001574
grad_step = 000397, loss = 0.001568
grad_step = 000398, loss = 0.001551
grad_step = 000399, loss = 0.001529
grad_step = 000400, loss = 0.001513
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001508
grad_step = 000402, loss = 0.001510
grad_step = 000403, loss = 0.001509
grad_step = 000404, loss = 0.001501
grad_step = 000405, loss = 0.001492
grad_step = 000406, loss = 0.001488
grad_step = 000407, loss = 0.001491
grad_step = 000408, loss = 0.001495
grad_step = 000409, loss = 0.001498
grad_step = 000410, loss = 0.001499
grad_step = 000411, loss = 0.001507
grad_step = 000412, loss = 0.001529
grad_step = 000413, loss = 0.001560
grad_step = 000414, loss = 0.001599
grad_step = 000415, loss = 0.001611
grad_step = 000416, loss = 0.001609
grad_step = 000417, loss = 0.001604
grad_step = 000418, loss = 0.001597
grad_step = 000419, loss = 0.001565
grad_step = 000420, loss = 0.001506
grad_step = 000421, loss = 0.001493
grad_step = 000422, loss = 0.001505
grad_step = 000423, loss = 0.001505
grad_step = 000424, loss = 0.001489
grad_step = 000425, loss = 0.001501
grad_step = 000426, loss = 0.001517
grad_step = 000427, loss = 0.001494
grad_step = 000428, loss = 0.001471
grad_step = 000429, loss = 0.001464
grad_step = 000430, loss = 0.001464
grad_step = 000431, loss = 0.001459
grad_step = 000432, loss = 0.001460
grad_step = 000433, loss = 0.001473
grad_step = 000434, loss = 0.001476
grad_step = 000435, loss = 0.001470
grad_step = 000436, loss = 0.001469
grad_step = 000437, loss = 0.001475
grad_step = 000438, loss = 0.001480
grad_step = 000439, loss = 0.001477
grad_step = 000440, loss = 0.001473
grad_step = 000441, loss = 0.001471
grad_step = 000442, loss = 0.001462
grad_step = 000443, loss = 0.001452
grad_step = 000444, loss = 0.001445
grad_step = 000445, loss = 0.001442
grad_step = 000446, loss = 0.001437
grad_step = 000447, loss = 0.001431
grad_step = 000448, loss = 0.001427
grad_step = 000449, loss = 0.001426
grad_step = 000450, loss = 0.001425
grad_step = 000451, loss = 0.001424
grad_step = 000452, loss = 0.001425
grad_step = 000453, loss = 0.001428
grad_step = 000454, loss = 0.001435
grad_step = 000455, loss = 0.001446
grad_step = 000456, loss = 0.001468
grad_step = 000457, loss = 0.001507
grad_step = 000458, loss = 0.001566
grad_step = 000459, loss = 0.001625
grad_step = 000460, loss = 0.001668
grad_step = 000461, loss = 0.001628
grad_step = 000462, loss = 0.001539
grad_step = 000463, loss = 0.001439
grad_step = 000464, loss = 0.001418
grad_step = 000465, loss = 0.001473
grad_step = 000466, loss = 0.001513
grad_step = 000467, loss = 0.001486
grad_step = 000468, loss = 0.001425
grad_step = 000469, loss = 0.001412
grad_step = 000470, loss = 0.001447
grad_step = 000471, loss = 0.001467
grad_step = 000472, loss = 0.001444
grad_step = 000473, loss = 0.001410
grad_step = 000474, loss = 0.001404
grad_step = 000475, loss = 0.001425
grad_step = 000476, loss = 0.001441
grad_step = 000477, loss = 0.001434
grad_step = 000478, loss = 0.001412
grad_step = 000479, loss = 0.001396
grad_step = 000480, loss = 0.001397
grad_step = 000481, loss = 0.001409
grad_step = 000482, loss = 0.001418
grad_step = 000483, loss = 0.001416
grad_step = 000484, loss = 0.001406
grad_step = 000485, loss = 0.001394
grad_step = 000486, loss = 0.001388
grad_step = 000487, loss = 0.001388
grad_step = 000488, loss = 0.001393
grad_step = 000489, loss = 0.001397
grad_step = 000490, loss = 0.001398
grad_step = 000491, loss = 0.001395
grad_step = 000492, loss = 0.001389
grad_step = 000493, loss = 0.001384
grad_step = 000494, loss = 0.001381
grad_step = 000495, loss = 0.001380
grad_step = 000496, loss = 0.001381
grad_step = 000497, loss = 0.001382
grad_step = 000498, loss = 0.001383
grad_step = 000499, loss = 0.001383
grad_step = 000500, loss = 0.001381
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001379
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

  date_run                              2020-05-11 09:13:40.307294
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.244312
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 09:13:40.313304
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.155035
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 09:13:40.320574
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.14746
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 09:13:40.325615
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.35581
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
0   2020-05-11 09:13:12.986166  ...    mean_absolute_error
1   2020-05-11 09:13:12.991038  ...     mean_squared_error
2   2020-05-11 09:13:12.995140  ...  median_absolute_error
3   2020-05-11 09:13:12.998244  ...               r2_score
4   2020-05-11 09:13:22.610787  ...    mean_absolute_error
5   2020-05-11 09:13:22.614057  ...     mean_squared_error
6   2020-05-11 09:13:22.617106  ...  median_absolute_error
7   2020-05-11 09:13:22.620666  ...               r2_score
8   2020-05-11 09:13:40.307294  ...    mean_absolute_error
9   2020-05-11 09:13:40.313304  ...     mean_squared_error
10  2020-05-11 09:13:40.320574  ...  median_absolute_error
11  2020-05-11 09:13:40.325615  ...               r2_score

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

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 319064.02it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 412978.01it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 571265.84it/s] 26%|██▋       | 2613248/9912422 [00:00<00:09, 802497.43it/s] 46%|████▌     | 4538368/9912422 [00:00<00:04, 1121845.22it/s] 67%|██████▋   | 6643712/9912422 [00:01<00:02, 1559212.59it/s] 90%|█████████ | 8929280/9912422 [00:01<00:00, 2156616.49it/s]9920512it [00:01, 8707768.10it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 150125.13it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 309787.67it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 402027.06it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 556708.68it/s]1654784it [00:00, 2844948.47it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52050.12it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7db1cf3fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d4f40fda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7db1c7eef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d4c20a048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d64687d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d64677e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d4f40fb38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d64677e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d4f40fb38> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7d4f40fda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7db1cbbba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f872dcc9208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8d385bc7b49c6997db0cd05ae020e1dd5271f2bbcf8e9609062600b45ed9633f
  Stored in directory: /tmp/pip-ephem-wheel-cache-juueky33/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f86c69b6a20> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 43s
   57344/17464789 [..............................] - ETA: 37s
  106496/17464789 [..............................] - ETA: 29s
  229376/17464789 [..............................] - ETA: 18s
  475136/17464789 [..............................] - ETA: 10s
  999424/17464789 [>.............................] - ETA: 6s 
 1974272/17464789 [==>...........................] - ETA: 3s
 3956736/17464789 [=====>........................] - ETA: 1s
 6447104/17464789 [==========>...................] - ETA: 0s
 8986624/17464789 [==============>...............] - ETA: 0s
11526144/17464789 [==================>...........] - ETA: 0s
14065664/17464789 [=======================>......] - ETA: 0s
16523264/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 09:15:08.810037: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 09:15:08.814252: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 09:15:08.814407: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557a361e9b00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 09:15:08.814418: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7893 - accuracy: 0.4920 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6622 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 3s - loss: 7.6150 - accuracy: 0.5034
12000/25000 [=============>................] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6702 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
15000/25000 [=================>............] - ETA: 2s - loss: 7.7004 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7030 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7153 - accuracy: 0.4968
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7254 - accuracy: 0.4962
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7312 - accuracy: 0.4958
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7034 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7075 - accuracy: 0.4973
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7050 - accuracy: 0.4975
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
25000/25000 [==============================] - 7s 265us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 09:15:21.803657
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 09:15:21.803657  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 09:15:27.577393: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 09:15:27.583039: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 09:15:27.583594: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5648818d1f50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 09:15:27.583893: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fdbe2591da0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6990 - crf_viterbi_accuracy: 0.1467 - val_loss: 1.5829 - val_crf_viterbi_accuracy: 0.1600

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdc0424f128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4980 - accuracy: 0.5110 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6452 - accuracy: 0.5014
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6491 - accuracy: 0.5011
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 3s - loss: 7.6443 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 3s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6348 - accuracy: 0.5021
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6634 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 267us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fdba0f46400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:52:52, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 270k/862M [00:00<8:20:30, 28.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.15M/862M [00:00<5:49:20, 41.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.1M/862M [00:00<4:01:45, 58.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.8M/862M [00:00<2:47:19, 83.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:00<1:55:46, 119kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:01<1:20:23, 170kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:01<56:01, 243kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:01<38:59, 347kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:01<28:42, 470kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<20:21, 661kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<11:15:00, 19.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:03<7:52:02, 28.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<5:31:54, 40.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<3:53:18, 57.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:44:33, 80.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<1:55:57, 115kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:22:52, 160kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<58:48, 225kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<43:02, 306kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<30:56, 425kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<23:36, 555kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<17:21, 754kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<14:04, 925kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:15<10:42, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<09:28, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:17<07:20, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<05:51, 2.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<5:44:04, 37.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<4:01:24, 53.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<2:50:45, 75.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:20<1:59:05, 107kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<1:41:17, 126kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<1:12:01, 177kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<52:02, 244kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<37:40, 337kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<26:25, 479kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<24:57, 507kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<18:31, 683kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:26<13:02, 966kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<17:46, 707kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<13:21, 941kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<10:01, 1.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<5:20:53, 39.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<3:45:10, 55.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<2:38:24, 78.7kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:31<1:50:28, 112kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<2:01:54, 102kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<1:26:38, 143kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:02:07, 198kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<44:15, 278kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<32:44, 375kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<23:37, 519kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<18:20, 665kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<13:41, 890kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<10:08, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<5:35:38, 36.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:40<3:53:57, 51.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<2:53:04, 69.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<2:02:15, 98.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<1:26:53, 138kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<1:02:07, 193kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<44:59, 266kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<32:13, 370kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<24:18, 489kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<18:17, 649kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<12:54, 917kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<13:06, 901kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<09:48, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<07:29, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<5:14:53, 37.4kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:51<3:39:24, 53.4kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<2:45:01, 70.9kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<1:56:25, 100kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<1:22:48, 140kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<59:05, 197kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<42:50, 270kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<31:14, 370kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<23:24, 491kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<17:22, 661kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<13:47, 828kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<10:22, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<07:47, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<5:13:52, 36.3kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<3:40:00, 51.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<2:34:49, 73.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:04<1:47:55, 104kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<1:37:31, 115kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<1:08:50, 163kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<49:40, 225kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<35:29, 315kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<26:24, 421kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<19:10, 579kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<15:04, 733kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<11:07, 992kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<08:24, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<4:45:25, 38.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<3:20:06, 54.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<2:20:53, 77.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:15<1:38:12, 111kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<1:21:28, 133kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<57:40, 188kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<41:49, 258kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<29:56, 360kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<22:30, 476kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<16:25, 652kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<13:05, 814kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<09:57, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<08:31, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:45, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<05:17, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<4:29:59, 39.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<3:09:16, 55.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<2:13:48, 78.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:28<1:33:12, 112kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<1:29:16, 117kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<1:03:10, 165kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<45:31, 227kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<32:24, 319kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<24:09, 425kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<17:34, 584kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<13:48, 739kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<10:17, 990kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<07:41, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<4:40:01, 36.3kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<3:16:08, 51.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<2:18:02, 73.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<1:37:32, 103kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<1:08:50, 145kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<49:27, 201kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<35:11, 283kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<26:00, 380kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<18:58, 520kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<14:40, 669kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<11:07, 882kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<08:12, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<4:30:15, 36.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:48<3:08:04, 51.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<12:00:27, 13.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<8:24:26, 19.2kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<5:52:27, 27.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<4:07:07, 39.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<2:53:21, 55.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<2:01:50, 78.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<1:26:14, 110kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<1:00:54, 156kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<43:48, 215kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<31:14, 302kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<23:11, 404kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<16:48, 556kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<12:11, 765kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<4:16:51, 36.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:01<2:58:40, 51.8kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<11:26:31, 13.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<8:00:33, 19.2kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<5:35:41, 27.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<3:55:19, 39.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<2:45:05, 55.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<1:56:01, 78.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<1:22:05, 110kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<58:08, 155kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<41:45, 215kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<29:50, 301kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<22:05, 403kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<16:02, 555kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<11:37, 763kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<4:04:38, 36.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<2:51:10, 51.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<2:00:29, 73.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<1:25:03, 103kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<59:59, 145kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<43:02, 201kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<30:48, 281kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<22:41, 379kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<16:24, 523kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<12:44, 669kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<09:27, 900kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<07:04, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<3:40:16, 38.5kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<2:34:08, 54.6kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<1:48:41, 77.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<1:15:57, 110kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<55:10, 151kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<39:32, 211kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<28:39, 289kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<20:45, 398kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<15:38, 525kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<11:50, 693kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<09:24, 866kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<07:10, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<05:23, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<3:42:54, 36.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<2:35:52, 51.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<1:49:39, 73.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<1:17:22, 103kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<54:35, 146kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<39:09, 202kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<27:52, 283kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<20:35, 380kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<14:51, 526kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<11:31, 673kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<08:32, 906kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<06:19, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<3:33:22, 36.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:47<2:28:13, 51.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<1:53:52, 67.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<1:20:45, 94.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:49<56:08, 135kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<1:00:35, 125kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<43:08, 175kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<31:02, 242kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<22:12, 338kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<16:33, 449kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<12:26, 598kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<09:41, 760kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<07:23, 997kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<05:29, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<3:22:00, 36.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<2:21:07, 51.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<1:39:16, 73.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<1:09:59, 103kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<49:14, 146kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<35:23, 201kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<25:10, 283kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<18:34, 380kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<13:26, 524kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<10:24, 671kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<07:47, 895kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<05:45, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<3:10:46, 36.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<2:13:12, 51.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:33:55, 73.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:11<1:05:14, 104kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<58:58, 115kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<41:38, 163kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<29:57, 225kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<21:21, 315kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<15:51, 421kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<11:37, 573kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<09:02, 730kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:56, 949kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<05:10, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<2:49:24, 38.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:58:16, 54.9kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:23:14, 77.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<58:40, 109kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<41:24, 155kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<29:42, 214kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<21:10, 300kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<15:39, 401kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<11:38, 540kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<08:58, 693kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:57, 892kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<05:09, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<2:39:02, 38.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<1:50:57, 55.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<1:18:05, 78.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:33<54:10, 112kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<1:00:53, 99.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<42:54, 141kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<30:39, 195kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<22:04, 270kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<16:08, 366kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<11:40, 505kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<08:59, 649kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<06:50, 852kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<05:01, 1.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<2:39:32, 36.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<1:51:10, 51.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<1:18:08, 73.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:44<54:10, 104kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<1:23:38, 67.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<58:48, 96.0kB/s]  .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<41:34, 134kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<29:25, 189kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<21:14, 260kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<15:08, 364kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<11:20, 481kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<08:14, 660kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<06:00, 900kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<2:30:22, 36.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:44:41, 51.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:13:39, 72.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<51:44, 102kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<36:29, 144kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<26:04, 200kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<18:33, 280kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<13:38, 376kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<09:54, 517kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<07:36, 665kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<05:38, 896kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<04:10, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<2:18:34, 36.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<1:36:23, 51.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<1:07:57, 72.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:06<47:03, 104kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<39:55, 122kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<28:10, 173kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<20:13, 238kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<14:27, 333kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<10:43, 443kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<08:00, 592kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<06:12, 753kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<04:45, 982kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<03:33, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<2:00:31, 38.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<1:23:46, 54.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<58:55, 77.4kB/s]  .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:17<40:47, 110kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<35:54, 125kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<25:28, 176kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:19<17:39, 251kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<17:29, 253kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<12:51, 344kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<08:58, 489kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<07:56, 549kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<06:00, 726kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<04:12, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<04:45, 902kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<03:40, 1.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<02:45, 1.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:57:29, 36.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<1:21:46, 51.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<57:39, 72.6kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<40:34, 103kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:28<28:07, 147kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<22:12, 185kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<15:49, 260kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<10:59, 369kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<09:52, 410kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<07:29, 539kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<05:13, 764kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<05:41, 700kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<04:14, 935kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:34<02:59, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<04:03, 964kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:09, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<02:23, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<1:36:22, 40.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:37<1:06:40, 57.3kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<48:03, 79.1kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<33:55, 112kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<23:54, 156kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<17:02, 219kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<12:16, 298kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<08:52, 412kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<06:39, 540kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<05:09, 695kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<03:36, 982kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<03:56, 893kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<03:02, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<02:17, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<1:30:35, 38.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:48<1:02:30, 54.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<45:27, 75.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<32:02, 106kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:50<22:04, 152kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<26:41, 125kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<18:53, 177kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<13:05, 252kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<10:43, 306kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<07:45, 422kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:54<05:23, 598kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<05:23, 595kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<03:59, 802kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:56<02:47, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<03:26, 911kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:40, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:59, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<1:24:53, 36.5kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<58:49, 52.1kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<41:30, 73.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<29:13, 104kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:01<20:09, 148kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<16:00, 185kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<11:26, 258kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:03<07:51, 368kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:37:59, 29.5kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:08:47, 42.0kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:05<47:21, 60.0kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<34:40, 81.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<24:28, 115kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:07<16:52, 164kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<13:17, 207kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<09:30, 289kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<06:43, 404kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<1:14:02, 36.7kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:10<50:48, 52.4kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<37:12, 71.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<26:10, 101kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:12<17:59, 144kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<14:46, 175kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<10:31, 244kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<07:33, 332kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<05:31, 453kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:16<03:49, 643kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<04:18, 567kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<03:10, 767kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:18<02:12, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<02:42, 876kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<02:06, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<01:34, 1.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<59:41, 39.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<40:55, 55.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<29:26, 76.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<20:44, 109kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<14:26, 152kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<10:15, 213kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<07:17, 291kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<05:15, 402kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<03:53, 529kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:54, 706kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:29<02:00, 998kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<02:29, 797kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:54, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:24, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<48:17, 40.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:32<32:51, 57.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<24:12, 77.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<17:01, 110kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<11:48, 153kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<08:22, 215kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:36<05:42, 307kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<05:32, 314kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<04:06, 423kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<02:51, 597kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<02:20, 715kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:46, 943kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<01:15, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:12, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:58, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:44, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<40:23, 38.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<27:51, 55.2kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<19:15, 77.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<13:32, 110kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:45<09:12, 157kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<07:08, 199kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<05:05, 278kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:47<03:27, 396kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<03:33, 382kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<02:35, 521kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:49<01:45, 738kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<02:08, 603kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:35, 804kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:51<01:05, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<01:37, 749kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<01:13, 983kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:53, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<31:43, 37.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<21:45, 53.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<14:54, 74.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<10:01, 106kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<07:12, 144kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<05:05, 203kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:58<03:24, 289kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<03:13, 301kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<02:18, 417kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:00<01:33, 591kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<01:27, 621kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<01:06, 814kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:02<00:44, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:48, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:37, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<00:27, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<20:36, 38.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<13:53, 54.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<09:24, 77.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<06:34, 109kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:07<04:16, 156kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<03:35, 183kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:31, 256kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:09<01:40, 364kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:20, 435kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:58, 592kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:11<00:37, 838kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:51, 608kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:37, 818kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:13<00:24, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:28, 940kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:21, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:15, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<10:56, 37.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<07:07, 53.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<04:32, 74.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:08, 106kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:18<01:50, 151kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:35, 171kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:06, 239kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:20<00:37, 340kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:35, 341kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:25, 468kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:22<00:12, 664kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:16, 480kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:11, 647kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:24<00:05, 916kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:05, 702kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:03, 926kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:01, 1.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:37, 39.6kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 900/400000 [00:00<00:44, 8991.41it/s]  0%|          | 1811/400000 [00:00<00:44, 9023.96it/s]  1%|          | 2740/400000 [00:00<00:43, 9099.67it/s]  1%|          | 3647/400000 [00:00<00:43, 9088.45it/s]  1%|          | 4607/400000 [00:00<00:42, 9235.85it/s]  1%|▏         | 5442/400000 [00:00<00:44, 8950.15it/s]  2%|▏         | 6299/400000 [00:00<00:44, 8832.33it/s]  2%|▏         | 7118/400000 [00:00<00:45, 8628.50it/s]  2%|▏         | 8020/400000 [00:00<00:44, 8740.52it/s]  2%|▏         | 8919/400000 [00:01<00:44, 8811.06it/s]  2%|▏         | 9842/400000 [00:01<00:43, 8931.14it/s]  3%|▎         | 10775/400000 [00:01<00:43, 9044.60it/s]  3%|▎         | 11669/400000 [00:01<00:43, 8907.68it/s]  3%|▎         | 12553/400000 [00:01<00:43, 8858.79it/s]  3%|▎         | 13434/400000 [00:01<00:44, 8780.56it/s]  4%|▎         | 14309/400000 [00:01<00:44, 8664.41it/s]  4%|▍         | 15228/400000 [00:01<00:43, 8813.12it/s]  4%|▍         | 16142/400000 [00:01<00:43, 8907.94it/s]  4%|▍         | 17043/400000 [00:01<00:42, 8935.29it/s]  4%|▍         | 17940/400000 [00:02<00:42, 8942.84it/s]  5%|▍         | 18835/400000 [00:02<00:42, 8935.50it/s]  5%|▍         | 19729/400000 [00:02<00:44, 8544.24it/s]  5%|▌         | 20587/400000 [00:02<00:44, 8509.72it/s]  5%|▌         | 21449/400000 [00:02<00:44, 8541.51it/s]  6%|▌         | 22341/400000 [00:02<00:43, 8650.06it/s]  6%|▌         | 23208/400000 [00:02<00:44, 8493.64it/s]  6%|▌         | 24090/400000 [00:02<00:43, 8588.77it/s]  6%|▋         | 25005/400000 [00:02<00:42, 8747.79it/s]  6%|▋         | 25930/400000 [00:02<00:42, 8892.19it/s]  7%|▋         | 26828/400000 [00:03<00:41, 8917.43it/s]  7%|▋         | 27722/400000 [00:03<00:42, 8762.45it/s]  7%|▋         | 28600/400000 [00:03<00:42, 8688.78it/s]  7%|▋         | 29527/400000 [00:03<00:41, 8854.32it/s]  8%|▊         | 30453/400000 [00:03<00:41, 8970.68it/s]  8%|▊         | 31362/400000 [00:03<00:40, 9004.65it/s]  8%|▊         | 32270/400000 [00:03<00:40, 9024.40it/s]  8%|▊         | 33174/400000 [00:03<00:41, 8793.99it/s]  9%|▊         | 34056/400000 [00:03<00:42, 8619.61it/s]  9%|▊         | 34921/400000 [00:03<00:43, 8469.85it/s]  9%|▉         | 35811/400000 [00:04<00:42, 8592.36it/s]  9%|▉         | 36726/400000 [00:04<00:41, 8751.32it/s]  9%|▉         | 37611/400000 [00:04<00:41, 8780.01it/s] 10%|▉         | 38522/400000 [00:04<00:40, 8873.75it/s] 10%|▉         | 39411/400000 [00:04<00:40, 8869.46it/s] 10%|█         | 40351/400000 [00:04<00:39, 9022.12it/s] 10%|█         | 41279/400000 [00:04<00:39, 9097.18it/s] 11%|█         | 42198/400000 [00:04<00:39, 9124.41it/s] 11%|█         | 43112/400000 [00:04<00:39, 8991.02it/s] 11%|█         | 44013/400000 [00:04<00:39, 8966.51it/s] 11%|█         | 44911/400000 [00:05<00:41, 8618.88it/s] 11%|█▏        | 45872/400000 [00:05<00:39, 8893.27it/s] 12%|█▏        | 46777/400000 [00:05<00:39, 8937.55it/s] 12%|█▏        | 47676/400000 [00:05<00:39, 8951.03it/s] 12%|█▏        | 48585/400000 [00:05<00:39, 8989.88it/s] 12%|█▏        | 49486/400000 [00:05<00:39, 8871.43it/s] 13%|█▎        | 50375/400000 [00:05<00:39, 8758.24it/s] 13%|█▎        | 51367/400000 [00:05<00:38, 9075.36it/s] 13%|█▎        | 52308/400000 [00:05<00:37, 9172.14it/s] 13%|█▎        | 53229/400000 [00:05<00:37, 9162.85it/s] 14%|█▎        | 54149/400000 [00:06<00:37, 9173.47it/s] 14%|█▍        | 55090/400000 [00:06<00:37, 9241.02it/s] 14%|█▍        | 56026/400000 [00:06<00:37, 9273.47it/s] 14%|█▍        | 56993/400000 [00:06<00:36, 9386.77it/s] 14%|█▍        | 57933/400000 [00:06<00:36, 9329.86it/s] 15%|█▍        | 58917/400000 [00:06<00:35, 9475.37it/s] 15%|█▍        | 59866/400000 [00:06<00:36, 9356.54it/s] 15%|█▌        | 60803/400000 [00:06<00:36, 9265.46it/s] 15%|█▌        | 61731/400000 [00:06<00:36, 9257.45it/s] 16%|█▌        | 62658/400000 [00:07<00:36, 9251.62it/s] 16%|█▌        | 63584/400000 [00:07<00:36, 9182.91it/s] 16%|█▌        | 64505/400000 [00:07<00:36, 9189.16it/s] 16%|█▋        | 65425/400000 [00:07<00:36, 9138.85it/s] 17%|█▋        | 66369/400000 [00:07<00:36, 9225.69it/s] 17%|█▋        | 67292/400000 [00:07<00:37, 8963.45it/s] 17%|█▋        | 68223/400000 [00:07<00:36, 9062.68it/s] 17%|█▋        | 69133/400000 [00:07<00:36, 9068.92it/s] 18%|█▊        | 70042/400000 [00:07<00:36, 9054.32it/s] 18%|█▊        | 70963/400000 [00:07<00:36, 9098.55it/s] 18%|█▊        | 71874/400000 [00:08<00:36, 8927.19it/s] 18%|█▊        | 72800/400000 [00:08<00:36, 9023.45it/s] 18%|█▊        | 73704/400000 [00:08<00:36, 9016.66it/s] 19%|█▊        | 74635/400000 [00:08<00:35, 9100.32it/s] 19%|█▉        | 75559/400000 [00:08<00:35, 9139.14it/s] 19%|█▉        | 76474/400000 [00:08<00:35, 9132.48it/s] 19%|█▉        | 77428/400000 [00:08<00:34, 9250.36it/s] 20%|█▉        | 78359/400000 [00:08<00:34, 9266.98it/s] 20%|█▉        | 79289/400000 [00:08<00:34, 9275.34it/s] 20%|██        | 80217/400000 [00:08<00:35, 8974.68it/s] 20%|██        | 81117/400000 [00:09<00:35, 8968.83it/s] 21%|██        | 82051/400000 [00:09<00:35, 9075.10it/s] 21%|██        | 82982/400000 [00:09<00:34, 9141.66it/s] 21%|██        | 83908/400000 [00:09<00:34, 9174.89it/s] 21%|██        | 84860/400000 [00:09<00:33, 9274.55it/s] 21%|██▏       | 85789/400000 [00:09<00:33, 9259.48it/s] 22%|██▏       | 86716/400000 [00:09<00:33, 9251.31it/s] 22%|██▏       | 87642/400000 [00:09<00:33, 9202.69it/s] 22%|██▏       | 88563/400000 [00:09<00:34, 9149.13it/s] 22%|██▏       | 89479/400000 [00:09<00:33, 9142.39it/s] 23%|██▎       | 90438/400000 [00:10<00:33, 9270.39it/s] 23%|██▎       | 91366/400000 [00:10<00:33, 9218.78it/s] 23%|██▎       | 92289/400000 [00:10<00:33, 9217.43it/s] 23%|██▎       | 93217/400000 [00:10<00:33, 9235.56it/s] 24%|██▎       | 94141/400000 [00:10<00:33, 9231.00it/s] 24%|██▍       | 95071/400000 [00:10<00:32, 9250.21it/s] 24%|██▍       | 95997/400000 [00:10<00:33, 9083.32it/s] 24%|██▍       | 96947/400000 [00:10<00:32, 9203.56it/s] 24%|██▍       | 97921/400000 [00:10<00:32, 9357.61it/s] 25%|██▍       | 98894/400000 [00:10<00:31, 9464.13it/s] 25%|██▍       | 99842/400000 [00:11<00:31, 9443.04it/s] 25%|██▌       | 100795/400000 [00:11<00:31, 9466.54it/s] 25%|██▌       | 101767/400000 [00:11<00:31, 9540.20it/s] 26%|██▌       | 102722/400000 [00:11<00:31, 9437.03it/s] 26%|██▌       | 103667/400000 [00:11<00:31, 9401.31it/s] 26%|██▌       | 104619/400000 [00:11<00:31, 9435.92it/s] 26%|██▋       | 105578/400000 [00:11<00:31, 9476.85it/s] 27%|██▋       | 106526/400000 [00:11<00:31, 9430.39it/s] 27%|██▋       | 107470/400000 [00:11<00:31, 9328.48it/s] 27%|██▋       | 108414/400000 [00:11<00:31, 9358.74it/s] 27%|██▋       | 109351/400000 [00:12<00:31, 9285.04it/s] 28%|██▊       | 110280/400000 [00:12<00:32, 9039.78it/s] 28%|██▊       | 111186/400000 [00:12<00:32, 8930.55it/s] 28%|██▊       | 112081/400000 [00:12<00:33, 8642.40it/s] 28%|██▊       | 112949/400000 [00:12<00:33, 8646.95it/s] 28%|██▊       | 113849/400000 [00:12<00:32, 8749.86it/s] 29%|██▊       | 114787/400000 [00:12<00:31, 8928.84it/s] 29%|██▉       | 115736/400000 [00:12<00:31, 9089.92it/s] 29%|██▉       | 116648/400000 [00:12<00:31, 9021.95it/s] 29%|██▉       | 117552/400000 [00:13<00:31, 9005.34it/s] 30%|██▉       | 118454/400000 [00:13<00:32, 8730.96it/s] 30%|██▉       | 119373/400000 [00:13<00:31, 8861.59it/s] 30%|███       | 120268/400000 [00:13<00:31, 8886.38it/s] 30%|███       | 121170/400000 [00:13<00:31, 8923.40it/s] 31%|███       | 122080/400000 [00:13<00:30, 8973.50it/s] 31%|███       | 122979/400000 [00:13<00:31, 8921.92it/s] 31%|███       | 123875/400000 [00:13<00:30, 8931.87it/s] 31%|███       | 124769/400000 [00:13<00:30, 8931.84it/s] 31%|███▏      | 125665/400000 [00:13<00:30, 8935.49it/s] 32%|███▏      | 126571/400000 [00:14<00:30, 8970.39it/s] 32%|███▏      | 127469/400000 [00:14<00:30, 8921.55it/s] 32%|███▏      | 128369/400000 [00:14<00:30, 8944.14it/s] 32%|███▏      | 129300/400000 [00:14<00:29, 9046.38it/s] 33%|███▎      | 130211/400000 [00:14<00:29, 9064.98it/s] 33%|███▎      | 131124/400000 [00:14<00:29, 9084.27it/s] 33%|███▎      | 132033/400000 [00:14<00:29, 8982.37it/s] 33%|███▎      | 132932/400000 [00:14<00:30, 8846.13it/s] 33%|███▎      | 133822/400000 [00:14<00:30, 8861.46it/s] 34%|███▎      | 134709/400000 [00:14<00:29, 8852.45it/s] 34%|███▍      | 135604/400000 [00:15<00:29, 8879.81it/s] 34%|███▍      | 136537/400000 [00:15<00:29, 9009.73it/s] 34%|███▍      | 137439/400000 [00:15<00:29, 8990.54it/s] 35%|███▍      | 138409/400000 [00:15<00:28, 9190.44it/s] 35%|███▍      | 139330/400000 [00:15<00:28, 9031.10it/s] 35%|███▌      | 140267/400000 [00:15<00:28, 9129.87it/s] 35%|███▌      | 141182/400000 [00:15<00:29, 8801.69it/s] 36%|███▌      | 142078/400000 [00:15<00:29, 8847.55it/s] 36%|███▌      | 142966/400000 [00:15<00:29, 8644.63it/s] 36%|███▌      | 143915/400000 [00:15<00:28, 8880.62it/s] 36%|███▌      | 144821/400000 [00:16<00:28, 8931.36it/s] 36%|███▋      | 145717/400000 [00:16<00:28, 8925.08it/s] 37%|███▋      | 146643/400000 [00:16<00:28, 9021.70it/s] 37%|███▋      | 147588/400000 [00:16<00:27, 9144.62it/s] 37%|███▋      | 148534/400000 [00:16<00:27, 9235.38it/s] 37%|███▋      | 149472/400000 [00:16<00:27, 9276.33it/s] 38%|███▊      | 150401/400000 [00:16<00:27, 9199.98it/s] 38%|███▊      | 151322/400000 [00:16<00:27, 9165.40it/s] 38%|███▊      | 152240/400000 [00:16<00:27, 9126.03it/s] 38%|███▊      | 153184/400000 [00:16<00:26, 9217.55it/s] 39%|███▊      | 154107/400000 [00:17<00:26, 9204.40it/s] 39%|███▉      | 155028/400000 [00:17<00:26, 9151.91it/s] 39%|███▉      | 155980/400000 [00:17<00:26, 9258.38it/s] 39%|███▉      | 156907/400000 [00:17<00:26, 9122.21it/s] 39%|███▉      | 157821/400000 [00:17<00:26, 9123.51it/s] 40%|███▉      | 158734/400000 [00:17<00:26, 9110.05it/s] 40%|███▉      | 159646/400000 [00:17<00:26, 9091.76it/s] 40%|████      | 160556/400000 [00:17<00:26, 9050.53it/s] 40%|████      | 161462/400000 [00:17<00:26, 8951.77it/s] 41%|████      | 162358/400000 [00:17<00:27, 8759.42it/s] 41%|████      | 163236/400000 [00:18<00:27, 8722.52it/s] 41%|████      | 164113/400000 [00:18<00:27, 8736.03it/s] 41%|████▏     | 165047/400000 [00:18<00:26, 8906.82it/s] 41%|████▏     | 165967/400000 [00:18<00:26, 8992.49it/s] 42%|████▏     | 166921/400000 [00:18<00:25, 9148.90it/s] 42%|████▏     | 167838/400000 [00:18<00:25, 9087.33it/s] 42%|████▏     | 168752/400000 [00:18<00:25, 9101.93it/s] 42%|████▏     | 169676/400000 [00:18<00:25, 9141.32it/s] 43%|████▎     | 170601/400000 [00:18<00:25, 9172.51it/s] 43%|████▎     | 171519/400000 [00:18<00:25, 9101.91it/s] 43%|████▎     | 172430/400000 [00:19<00:25, 9092.60it/s] 43%|████▎     | 173340/400000 [00:19<00:25, 8969.97it/s] 44%|████▎     | 174267/400000 [00:19<00:24, 9055.93it/s] 44%|████▍     | 175208/400000 [00:19<00:24, 9158.71it/s] 44%|████▍     | 176152/400000 [00:19<00:24, 9241.29it/s] 44%|████▍     | 177124/400000 [00:19<00:23, 9379.15it/s] 45%|████▍     | 178063/400000 [00:19<00:23, 9290.58it/s] 45%|████▍     | 178993/400000 [00:19<00:24, 9159.73it/s] 45%|████▍     | 179910/400000 [00:19<00:24, 9086.67it/s] 45%|████▌     | 180838/400000 [00:20<00:23, 9142.20it/s] 45%|████▌     | 181765/400000 [00:20<00:23, 9179.09it/s] 46%|████▌     | 182684/400000 [00:20<00:23, 9093.29it/s] 46%|████▌     | 183600/400000 [00:20<00:23, 9110.74it/s] 46%|████▌     | 184512/400000 [00:20<00:23, 9075.79it/s] 46%|████▋     | 185436/400000 [00:20<00:23, 9124.11it/s] 47%|████▋     | 186349/400000 [00:20<00:23, 9121.08it/s] 47%|████▋     | 187262/400000 [00:20<00:23, 9092.85it/s] 47%|████▋     | 188172/400000 [00:20<00:23, 9011.12it/s] 47%|████▋     | 189095/400000 [00:20<00:23, 9073.69it/s] 48%|████▊     | 190018/400000 [00:21<00:23, 9118.68it/s] 48%|████▊     | 190931/400000 [00:21<00:23, 9059.63it/s] 48%|████▊     | 191853/400000 [00:21<00:22, 9104.92it/s] 48%|████▊     | 192764/400000 [00:21<00:22, 9079.31it/s] 48%|████▊     | 193673/400000 [00:21<00:23, 8839.26it/s] 49%|████▊     | 194618/400000 [00:21<00:22, 9011.89it/s] 49%|████▉     | 195541/400000 [00:21<00:22, 9073.76it/s] 49%|████▉     | 196450/400000 [00:21<00:22, 9017.60it/s] 49%|████▉     | 197353/400000 [00:21<00:22, 9019.56it/s] 50%|████▉     | 198262/400000 [00:21<00:22, 9039.63it/s] 50%|████▉     | 199198/400000 [00:22<00:21, 9131.81it/s] 50%|█████     | 200112/400000 [00:22<00:22, 8884.08it/s] 50%|█████     | 201030/400000 [00:22<00:22, 8970.22it/s] 50%|█████     | 201929/400000 [00:22<00:22, 8923.31it/s] 51%|█████     | 202823/400000 [00:22<00:22, 8786.65it/s] 51%|█████     | 203763/400000 [00:22<00:21, 8961.64it/s] 51%|█████     | 204661/400000 [00:22<00:21, 8918.76it/s] 51%|█████▏    | 205555/400000 [00:22<00:21, 8911.01it/s] 52%|█████▏    | 206447/400000 [00:22<00:22, 8773.70it/s] 52%|█████▏    | 207351/400000 [00:22<00:21, 8849.13it/s] 52%|█████▏    | 208312/400000 [00:23<00:21, 9064.02it/s] 52%|█████▏    | 209260/400000 [00:23<00:20, 9182.18it/s] 53%|█████▎    | 210180/400000 [00:23<00:20, 9111.01it/s] 53%|█████▎    | 211093/400000 [00:23<00:20, 9077.31it/s] 53%|█████▎    | 212027/400000 [00:23<00:20, 9154.18it/s] 53%|█████▎    | 212976/400000 [00:23<00:20, 9252.10it/s] 53%|█████▎    | 213912/400000 [00:23<00:20, 9281.53it/s] 54%|█████▎    | 214841/400000 [00:23<00:20, 9256.16it/s] 54%|█████▍    | 215785/400000 [00:23<00:19, 9309.62it/s] 54%|█████▍    | 216722/400000 [00:23<00:19, 9326.05it/s] 54%|█████▍    | 217655/400000 [00:24<00:19, 9280.92it/s] 55%|█████▍    | 218584/400000 [00:24<00:19, 9234.27it/s] 55%|█████▍    | 219510/400000 [00:24<00:19, 9239.51it/s] 55%|█████▌    | 220435/400000 [00:24<00:19, 9204.64it/s] 55%|█████▌    | 221377/400000 [00:24<00:19, 9267.33it/s] 56%|█████▌    | 222314/400000 [00:24<00:19, 9297.59it/s] 56%|█████▌    | 223244/400000 [00:24<00:19, 9182.85it/s] 56%|█████▌    | 224163/400000 [00:24<00:19, 8889.79it/s] 56%|█████▋    | 225075/400000 [00:24<00:19, 8956.50it/s] 56%|█████▋    | 225991/400000 [00:24<00:19, 9015.41it/s] 57%|█████▋    | 226931/400000 [00:25<00:18, 9124.83it/s] 57%|█████▋    | 227845/400000 [00:25<00:18, 9116.61it/s] 57%|█████▋    | 228759/400000 [00:25<00:18, 9123.41it/s] 57%|█████▋    | 229672/400000 [00:25<00:19, 8906.52it/s] 58%|█████▊    | 230583/400000 [00:25<00:18, 8963.17it/s] 58%|█████▊    | 231506/400000 [00:25<00:18, 9039.26it/s] 58%|█████▊    | 232426/400000 [00:25<00:18, 9086.50it/s] 58%|█████▊    | 233336/400000 [00:25<00:18, 9064.14it/s] 59%|█████▊    | 234243/400000 [00:25<00:18, 8974.17it/s] 59%|█████▉    | 235141/400000 [00:25<00:18, 8964.16it/s] 59%|█████▉    | 236048/400000 [00:26<00:18, 8993.18it/s] 59%|█████▉    | 236973/400000 [00:26<00:17, 9067.29it/s] 59%|█████▉    | 237881/400000 [00:26<00:17, 9028.90it/s] 60%|█████▉    | 238809/400000 [00:26<00:17, 9100.12it/s] 60%|█████▉    | 239740/400000 [00:26<00:17, 9159.06it/s] 60%|██████    | 240680/400000 [00:26<00:17, 9227.16it/s] 60%|██████    | 241604/400000 [00:26<00:17, 9160.00it/s] 61%|██████    | 242528/400000 [00:26<00:17, 9183.40it/s] 61%|██████    | 243459/400000 [00:26<00:16, 9220.98it/s] 61%|██████    | 244382/400000 [00:27<00:16, 9218.96it/s] 61%|██████▏   | 245305/400000 [00:27<00:17, 9010.50it/s] 62%|██████▏   | 246210/400000 [00:27<00:17, 9020.09it/s] 62%|██████▏   | 247113/400000 [00:27<00:16, 9008.04it/s] 62%|██████▏   | 248015/400000 [00:27<00:17, 8893.69it/s] 62%|██████▏   | 248906/400000 [00:27<00:17, 8671.97it/s] 62%|██████▏   | 249813/400000 [00:27<00:17, 8786.33it/s] 63%|██████▎   | 250734/400000 [00:27<00:16, 8906.57it/s] 63%|██████▎   | 251636/400000 [00:27<00:16, 8935.55it/s] 63%|██████▎   | 252550/400000 [00:27<00:16, 8995.08it/s] 63%|██████▎   | 253486/400000 [00:28<00:16, 9098.64it/s] 64%|██████▎   | 254397/400000 [00:28<00:16, 8968.89it/s] 64%|██████▍   | 255312/400000 [00:28<00:16, 9020.81it/s] 64%|██████▍   | 256215/400000 [00:28<00:16, 8920.76it/s] 64%|██████▍   | 257108/400000 [00:28<00:16, 8771.66it/s] 65%|██████▍   | 258023/400000 [00:28<00:15, 8880.37it/s] 65%|██████▍   | 258977/400000 [00:28<00:15, 9067.07it/s] 65%|██████▍   | 259886/400000 [00:28<00:15, 8981.58it/s] 65%|██████▌   | 260818/400000 [00:28<00:15, 9079.27it/s] 65%|██████▌   | 261778/400000 [00:28<00:14, 9226.72it/s] 66%|██████▌   | 262703/400000 [00:29<00:14, 9204.65it/s] 66%|██████▌   | 263625/400000 [00:29<00:15, 8903.20it/s] 66%|██████▌   | 264519/400000 [00:29<00:15, 8866.14it/s] 66%|██████▋   | 265420/400000 [00:29<00:15, 8906.24it/s] 67%|██████▋   | 266322/400000 [00:29<00:14, 8938.33it/s] 67%|██████▋   | 267217/400000 [00:29<00:14, 8891.51it/s] 67%|██████▋   | 268132/400000 [00:29<00:14, 8966.96it/s] 67%|██████▋   | 269030/400000 [00:29<00:14, 8895.32it/s] 67%|██████▋   | 269927/400000 [00:29<00:14, 8915.71it/s] 68%|██████▊   | 270839/400000 [00:29<00:14, 8975.74it/s] 68%|██████▊   | 271769/400000 [00:30<00:14, 9070.11it/s] 68%|██████▊   | 272693/400000 [00:30<00:13, 9119.88it/s] 68%|██████▊   | 273622/400000 [00:30<00:13, 9167.18it/s] 69%|██████▊   | 274578/400000 [00:30<00:13, 9280.10it/s] 69%|██████▉   | 275507/400000 [00:30<00:13, 9261.76it/s] 69%|██████▉   | 276446/400000 [00:30<00:13, 9298.19it/s] 69%|██████▉   | 277377/400000 [00:30<00:13, 9250.20it/s] 70%|██████▉   | 278303/400000 [00:30<00:13, 8954.87it/s] 70%|██████▉   | 279201/400000 [00:30<00:13, 8813.79it/s] 70%|███████   | 280147/400000 [00:30<00:13, 8997.58it/s] 70%|███████   | 281084/400000 [00:31<00:13, 9104.74it/s] 71%|███████   | 282008/400000 [00:31<00:12, 9144.44it/s] 71%|███████   | 282939/400000 [00:31<00:12, 9192.45it/s] 71%|███████   | 283860/400000 [00:31<00:12, 9139.98it/s] 71%|███████   | 284775/400000 [00:31<00:12, 9069.91it/s] 71%|███████▏  | 285763/400000 [00:31<00:12, 9296.51it/s] 72%|███████▏  | 286751/400000 [00:31<00:11, 9463.39it/s] 72%|███████▏  | 287700/400000 [00:31<00:12, 9248.91it/s] 72%|███████▏  | 288628/400000 [00:31<00:12, 9136.13it/s] 72%|███████▏  | 289544/400000 [00:32<00:12, 9027.94it/s] 73%|███████▎  | 290449/400000 [00:32<00:12, 8889.42it/s] 73%|███████▎  | 291388/400000 [00:32<00:12, 9033.07it/s] 73%|███████▎  | 292354/400000 [00:32<00:11, 9210.29it/s] 73%|███████▎  | 293278/400000 [00:32<00:11, 9007.34it/s] 74%|███████▎  | 294182/400000 [00:32<00:11, 8932.84it/s] 74%|███████▍  | 295078/400000 [00:32<00:11, 8846.16it/s] 74%|███████▍  | 296002/400000 [00:32<00:11, 8958.70it/s] 74%|███████▍  | 296911/400000 [00:32<00:11, 8997.54it/s] 74%|███████▍  | 297828/400000 [00:32<00:11, 9046.81it/s] 75%|███████▍  | 298754/400000 [00:33<00:11, 9107.49it/s] 75%|███████▍  | 299682/400000 [00:33<00:10, 9158.47it/s] 75%|███████▌  | 300599/400000 [00:33<00:10, 9133.02it/s] 75%|███████▌  | 301530/400000 [00:33<00:10, 9184.73it/s] 76%|███████▌  | 302469/400000 [00:33<00:10, 9242.26it/s] 76%|███████▌  | 303394/400000 [00:33<00:10, 9242.41it/s] 76%|███████▌  | 304319/400000 [00:33<00:10, 9119.01it/s] 76%|███████▋  | 305232/400000 [00:33<00:10, 9114.74it/s] 77%|███████▋  | 306158/400000 [00:33<00:10, 9157.48it/s] 77%|███████▋  | 307096/400000 [00:33<00:10, 9221.91it/s] 77%|███████▋  | 308042/400000 [00:34<00:09, 9288.67it/s] 77%|███████▋  | 308972/400000 [00:34<00:09, 9286.10it/s] 77%|███████▋  | 309901/400000 [00:34<00:09, 9183.23it/s] 78%|███████▊  | 310820/400000 [00:34<00:09, 9140.28it/s] 78%|███████▊  | 311735/400000 [00:34<00:09, 9003.10it/s] 78%|███████▊  | 312673/400000 [00:34<00:09, 9112.09it/s] 78%|███████▊  | 313586/400000 [00:34<00:09, 9086.27it/s] 79%|███████▊  | 314499/400000 [00:34<00:09, 9096.56it/s] 79%|███████▉  | 315410/400000 [00:34<00:09, 9056.78it/s] 79%|███████▉  | 316317/400000 [00:34<00:09, 8994.33it/s] 79%|███████▉  | 317247/400000 [00:35<00:09, 9082.22it/s] 80%|███████▉  | 318156/400000 [00:35<00:09, 8918.37it/s] 80%|███████▉  | 319086/400000 [00:35<00:08, 9028.90it/s] 80%|████████  | 320044/400000 [00:35<00:08, 9184.91it/s] 80%|████████  | 320964/400000 [00:35<00:08, 9052.47it/s] 80%|████████  | 321871/400000 [00:35<00:08, 9042.84it/s] 81%|████████  | 322800/400000 [00:35<00:08, 9114.35it/s] 81%|████████  | 323713/400000 [00:35<00:08, 9078.88it/s] 81%|████████  | 324622/400000 [00:35<00:08, 9031.32it/s] 81%|████████▏ | 325539/400000 [00:35<00:08, 9072.25it/s] 82%|████████▏ | 326455/400000 [00:36<00:08, 9096.06it/s] 82%|████████▏ | 327403/400000 [00:36<00:07, 9207.84it/s] 82%|████████▏ | 328325/400000 [00:36<00:07, 9165.17it/s] 82%|████████▏ | 329258/400000 [00:36<00:07, 9211.44it/s] 83%|████████▎ | 330180/400000 [00:36<00:07, 9177.08it/s] 83%|████████▎ | 331110/400000 [00:36<00:07, 9211.91it/s] 83%|████████▎ | 332032/400000 [00:36<00:07, 9139.95it/s] 83%|████████▎ | 332947/400000 [00:36<00:07, 9098.00it/s] 83%|████████▎ | 333891/400000 [00:36<00:07, 9197.55it/s] 84%|████████▎ | 334812/400000 [00:36<00:07, 9162.83it/s] 84%|████████▍ | 335742/400000 [00:37<00:06, 9200.97it/s] 84%|████████▍ | 336684/400000 [00:37<00:06, 9263.72it/s] 84%|████████▍ | 337611/400000 [00:37<00:06, 9154.11it/s] 85%|████████▍ | 338552/400000 [00:37<00:06, 9228.43it/s] 85%|████████▍ | 339476/400000 [00:37<00:06, 8901.33it/s] 85%|████████▌ | 340381/400000 [00:37<00:06, 8943.11it/s] 85%|████████▌ | 341278/400000 [00:37<00:06, 8918.40it/s] 86%|████████▌ | 342172/400000 [00:37<00:06, 8907.67it/s] 86%|████████▌ | 343103/400000 [00:37<00:06, 9022.02it/s] 86%|████████▌ | 344007/400000 [00:37<00:06, 8924.13it/s] 86%|████████▌ | 344964/400000 [00:38<00:06, 9108.44it/s] 86%|████████▋ | 345894/400000 [00:38<00:05, 9164.51it/s] 87%|████████▋ | 346812/400000 [00:38<00:05, 9150.59it/s] 87%|████████▋ | 347728/400000 [00:38<00:05, 9079.60it/s] 87%|████████▋ | 348637/400000 [00:38<00:05, 9061.82it/s] 87%|████████▋ | 349561/400000 [00:38<00:05, 9111.84it/s] 88%|████████▊ | 350473/400000 [00:38<00:05, 9087.85it/s] 88%|████████▊ | 351383/400000 [00:38<00:05, 9025.92it/s] 88%|████████▊ | 352308/400000 [00:38<00:05, 9091.00it/s] 88%|████████▊ | 353218/400000 [00:39<00:05, 9081.68it/s] 89%|████████▊ | 354149/400000 [00:39<00:05, 9147.44it/s] 89%|████████▉ | 355086/400000 [00:39<00:04, 9211.09it/s] 89%|████████▉ | 356011/400000 [00:39<00:04, 9221.22it/s] 89%|████████▉ | 356936/400000 [00:39<00:04, 9228.87it/s] 89%|████████▉ | 357860/400000 [00:39<00:04, 9168.02it/s] 90%|████████▉ | 358818/400000 [00:39<00:04, 9287.58it/s] 90%|████████▉ | 359748/400000 [00:39<00:04, 9189.21it/s] 90%|█████████ | 360668/400000 [00:39<00:04, 9093.53it/s] 90%|█████████ | 361578/400000 [00:39<00:04, 8897.98it/s] 91%|█████████ | 362476/400000 [00:40<00:04, 8921.08it/s] 91%|█████████ | 363382/400000 [00:40<00:04, 8959.86it/s] 91%|█████████ | 364279/400000 [00:40<00:04, 8930.19it/s] 91%|█████████▏| 365193/400000 [00:40<00:03, 8991.24it/s] 92%|█████████▏| 366093/400000 [00:40<00:03, 8974.03it/s] 92%|█████████▏| 366991/400000 [00:40<00:03, 8851.93it/s] 92%|█████████▏| 367881/400000 [00:40<00:03, 8864.26it/s] 92%|█████████▏| 368786/400000 [00:40<00:03, 8918.84it/s] 92%|█████████▏| 369696/400000 [00:40<00:03, 8971.73it/s] 93%|█████████▎| 370594/400000 [00:40<00:03, 8783.68it/s] 93%|█████████▎| 371474/400000 [00:41<00:03, 8787.76it/s] 93%|█████████▎| 372354/400000 [00:41<00:03, 8784.66it/s] 93%|█████████▎| 373257/400000 [00:41<00:03, 8854.97it/s] 94%|█████████▎| 374159/400000 [00:41<00:02, 8901.07it/s] 94%|█████████▍| 375050/400000 [00:41<00:02, 8856.87it/s] 94%|█████████▍| 375937/400000 [00:41<00:02, 8842.27it/s] 94%|█████████▍| 376822/400000 [00:41<00:02, 8780.61it/s] 94%|█████████▍| 377767/400000 [00:41<00:02, 8968.74it/s] 95%|█████████▍| 378682/400000 [00:41<00:02, 9020.42it/s] 95%|█████████▍| 379603/400000 [00:41<00:02, 9073.32it/s] 95%|█████████▌| 380512/400000 [00:42<00:02, 9048.82it/s] 95%|█████████▌| 381471/400000 [00:42<00:02, 9203.37it/s] 96%|█████████▌| 382393/400000 [00:42<00:01, 9179.12it/s] 96%|█████████▌| 383312/400000 [00:42<00:01, 9039.23it/s] 96%|█████████▌| 384245/400000 [00:42<00:01, 9124.41it/s] 96%|█████████▋| 385159/400000 [00:42<00:01, 8987.08it/s] 97%|█████████▋| 386059/400000 [00:42<00:01, 8693.26it/s] 97%|█████████▋| 386933/400000 [00:42<00:01, 8705.06it/s] 97%|█████████▋| 387858/400000 [00:42<00:01, 8859.46it/s] 97%|█████████▋| 388794/400000 [00:42<00:01, 9001.76it/s] 97%|█████████▋| 389746/400000 [00:43<00:01, 9149.35it/s] 98%|█████████▊| 390663/400000 [00:43<00:01, 9111.20it/s] 98%|█████████▊| 391576/400000 [00:43<00:00, 9075.02it/s] 98%|█████████▊| 392494/400000 [00:43<00:00, 9105.50it/s] 98%|█████████▊| 393440/400000 [00:43<00:00, 9208.07it/s] 99%|█████████▊| 394362/400000 [00:43<00:00, 9199.77it/s] 99%|█████████▉| 395283/400000 [00:43<00:00, 9133.01it/s] 99%|█████████▉| 396223/400000 [00:43<00:00, 9208.70it/s] 99%|█████████▉| 397145/400000 [00:43<00:00, 9124.03it/s]100%|█████████▉| 398058/400000 [00:43<00:00, 8802.90it/s]100%|█████████▉| 398951/400000 [00:44<00:00, 8840.37it/s]100%|█████████▉| 399859/400000 [00:44<00:00, 8909.37it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9049.57it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdbd4164f60> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010959549096861964 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010997160820657992 	 Accuracy: 62

  model saves at 62% accuracy 

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

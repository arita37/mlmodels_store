
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4f73672f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 10:13:45.409574
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 10:13:45.413777
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 10:13:45.417520
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 10:13:45.421098
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4f7f4363c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357514.4062
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 268720.8438
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 179660.5312
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 106421.2656
Epoch 5/10

1/1 [==============================] - 0s 102ms/step - loss: 63026.7656
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 38880.1289
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 25297.9375
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 17301.4395
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 12372.6982
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 9276.9766

  #### Inference Need return ypred, ytrue ######################### 
[[-0.37991238  0.7583639   0.13189207 -0.41447893  1.0306572  -0.62814003
   1.1703871   1.4083571   0.439373    0.5566942   1.058195    1.1485676
  -0.74233216 -0.7693842   0.4747073  -1.1623358   0.24878244 -0.8296983
   0.5320151   0.9105769   0.6391276  -1.1296862  -0.2827922   1.5730866
   0.39698216  0.04933004  0.01553384  0.8119558  -0.0349261  -0.36001617
   0.43386662 -0.2598183  -0.9136723  -0.3150939  -1.4554695  -1.3058908
   0.45671067 -0.22165757 -0.21620998 -1.2900747   0.3632441   0.03577872
  -0.33854488 -0.19748306  0.14519599 -0.3879825  -0.13669592 -0.2345077
  -0.934938    0.31400403 -0.9900429   0.29424408 -0.5498528   0.25869408
   0.92360455  0.37693882  0.03065559  0.254313    0.6592275   1.6463366
   0.07242751  6.7381005   5.770223    6.5833697   8.123425    6.691076
   6.0873146   6.0020905   5.005992    5.1083345   5.4929223   7.6951246
   5.885074    6.6236258   7.0805683   7.1715035   8.085086    6.813533
   6.103081    7.5267024   5.6370983   5.5030923   5.9392266   6.0138526
   6.583618    7.4030185   6.8544803   5.898954    8.388402    7.240919
   6.2036266   6.68163     5.115488    6.7866883   5.2338305   5.472616
   5.995626    7.59239     6.435051    5.682654    4.270024    5.4172177
   7.687685    5.232747    6.474368    5.579557    6.2325807   6.521331
   5.6344357   7.578066    5.739608    6.197313    7.24432     7.598963
   5.7461386   5.5182      6.6283236   5.453932    7.0542006   5.282586
  -0.61387473 -0.7246986  -0.6286677  -1.1054463   0.3745461   1.791959
   0.5174709  -0.18714745  1.547731   -0.78180575  0.14898789  1.0038689
   0.36771834  0.35807997  0.3052033   0.44425526 -0.14018022  1.9405366
   1.1381702  -0.8158153   1.1236684   0.2517055  -0.48873442  0.3242523
  -1.0247555  -0.565633   -0.7490669  -0.40148535  0.37148854 -0.8394042
   2.2298234   1.2061538   0.6866715   0.01220226 -0.10243621 -0.04317536
  -1.1034516  -0.59455425 -1.1334107   0.42348424  0.94117737  0.6014411
   0.6996921   0.3457435   1.0275255  -0.33436325 -1.1231898   0.55395263
  -1.3494462   1.2700084  -1.3260044  -0.81715333 -0.7532615  -0.6830555
   0.13885662 -1.4575268  -0.8387681   0.17319581  0.72468     1.4707611
   1.4416937   2.0732198   0.4513986   0.9708957   1.9648714   1.7114493
   1.3576647   0.4680264   0.56872344  0.5528044   0.5955825   1.81932
   0.43233842  0.39890796  1.6873795   0.6345653   1.2315483   1.3771135
   0.39562243  1.5703194   1.4475493   1.6464329   1.3339288   0.95088917
   0.46772158  0.4966861   1.4519272   1.2898844   1.2832834   0.62473035
   0.91506827  1.5787961   0.8114489   0.47237992  0.4500925   2.2026505
   2.0702844   0.2953444   1.5505203   1.4533403   0.9116573   0.7483929
   1.6331719   1.4234245   1.6574047   1.3252863   2.3886075   1.016756
   0.7187625   0.7819443   0.40017998  0.6391471   0.83781457  1.4466047
   0.5568073   0.51609546  0.9668217   0.3061788   2.1017046   1.364584
   0.05398452  8.999416    6.8622847   7.5850487   8.000551    7.9566145
   7.0524015   5.890205    7.344244    7.72108     7.2706747   6.8384867
   7.746591    7.3748207   7.808466    7.165255    8.538366    6.9019384
   7.329052    7.444719    7.05866     7.927599    6.947831    5.7278247
   7.196592    6.769547    8.9333515   6.8618298   7.3870726   7.7235465
   7.0333886   7.514216    8.164426    6.404379    6.849466    7.4792614
   8.041443    6.8514996   5.9881406   5.930957    7.085309    8.2122555
   8.739595    5.9723      5.8533144   6.924585    6.962678    8.88567
   6.1790385   6.612796    6.4235315   8.673595    7.0004835   7.796349
   5.216537    6.2030044   6.141132    5.682293    7.3401895   8.41804
   0.22947246  0.47837806  1.7166274   1.4266229   0.37515116  1.0044976
   0.5772942   1.8971064   1.0083454   1.0635698   0.9134952   0.49058318
   2.2931993   1.1758459   1.0747504   0.7853383   1.6787271   1.8122097
   0.7555821   3.2273097   1.2266898   0.55243707  0.19017202  0.6784551
   0.28143966  1.4980073   1.475117    2.0786815   0.28290474  0.80695313
   0.62960243  0.34560555  1.5524415   1.0720128   1.3355665   1.9127392
   0.5201169   2.5330324   0.4217764   0.8477284   1.4678547   1.277556
   1.5359378   0.9192627   0.27136564  1.375195    2.3406272   2.0546565
   1.0654567   0.76277196  0.14949906  0.5528384   0.9716941   0.5499319
   1.6649065   0.43905056  0.8058073   0.836416    1.9030132   0.990615
  -8.652864    5.66815    -5.8550167 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 10:13:54.740627
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.3767
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 10:13:54.745339
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9119.2
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 10:13:54.749158
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.1123
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 10:13:54.752765
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -815.689
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139978856145304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139977914834336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139977914834840
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139977914417616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139977914418120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139977914418624

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4f6cebe550> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.578627
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.542720
grad_step = 000002, loss = 0.516417
grad_step = 000003, loss = 0.488890
grad_step = 000004, loss = 0.464051
grad_step = 000005, loss = 0.445572
grad_step = 000006, loss = 0.426631
grad_step = 000007, loss = 0.405367
grad_step = 000008, loss = 0.387136
grad_step = 000009, loss = 0.368658
grad_step = 000010, loss = 0.351973
grad_step = 000011, loss = 0.340217
grad_step = 000012, loss = 0.329418
grad_step = 000013, loss = 0.314424
grad_step = 000014, loss = 0.299281
grad_step = 000015, loss = 0.287789
grad_step = 000016, loss = 0.278193
grad_step = 000017, loss = 0.267283
grad_step = 000018, loss = 0.253654
grad_step = 000019, loss = 0.239624
grad_step = 000020, loss = 0.229362
grad_step = 000021, loss = 0.221791
grad_step = 000022, loss = 0.211848
grad_step = 000023, loss = 0.199978
grad_step = 000024, loss = 0.189292
grad_step = 000025, loss = 0.180103
grad_step = 000026, loss = 0.171028
grad_step = 000027, loss = 0.161166
grad_step = 000028, loss = 0.151541
grad_step = 000029, loss = 0.143511
grad_step = 000030, loss = 0.135877
grad_step = 000031, loss = 0.127310
grad_step = 000032, loss = 0.119037
grad_step = 000033, loss = 0.112074
grad_step = 000034, loss = 0.105598
grad_step = 000035, loss = 0.098758
grad_step = 000036, loss = 0.092020
grad_step = 000037, loss = 0.086016
grad_step = 000038, loss = 0.080392
grad_step = 000039, loss = 0.074770
grad_step = 000040, loss = 0.069553
grad_step = 000041, loss = 0.064955
grad_step = 000042, loss = 0.060354
grad_step = 000043, loss = 0.055784
grad_step = 000044, loss = 0.051817
grad_step = 000045, loss = 0.048218
grad_step = 000046, loss = 0.044633
grad_step = 000047, loss = 0.041210
grad_step = 000048, loss = 0.038154
grad_step = 000049, loss = 0.035241
grad_step = 000050, loss = 0.032471
grad_step = 000051, loss = 0.030063
grad_step = 000052, loss = 0.027743
grad_step = 000053, loss = 0.025455
grad_step = 000054, loss = 0.023510
grad_step = 000055, loss = 0.021731
grad_step = 000056, loss = 0.019911
grad_step = 000057, loss = 0.018309
grad_step = 000058, loss = 0.016914
grad_step = 000059, loss = 0.015498
grad_step = 000060, loss = 0.014226
grad_step = 000061, loss = 0.013126
grad_step = 000062, loss = 0.012017
grad_step = 000063, loss = 0.011054
grad_step = 000064, loss = 0.010232
grad_step = 000065, loss = 0.009388
grad_step = 000066, loss = 0.008629
grad_step = 000067, loss = 0.007992
grad_step = 000068, loss = 0.007373
grad_step = 000069, loss = 0.006825
grad_step = 000070, loss = 0.006345
grad_step = 000071, loss = 0.005884
grad_step = 000072, loss = 0.005488
grad_step = 000073, loss = 0.005141
grad_step = 000074, loss = 0.004807
grad_step = 000075, loss = 0.004519
grad_step = 000076, loss = 0.004269
grad_step = 000077, loss = 0.004036
grad_step = 000078, loss = 0.003831
grad_step = 000079, loss = 0.003655
grad_step = 000080, loss = 0.003484
grad_step = 000081, loss = 0.003351
grad_step = 000082, loss = 0.003217
grad_step = 000083, loss = 0.003101
grad_step = 000084, loss = 0.003008
grad_step = 000085, loss = 0.002914
grad_step = 000086, loss = 0.002838
grad_step = 000087, loss = 0.002768
grad_step = 000088, loss = 0.002705
grad_step = 000089, loss = 0.002653
grad_step = 000090, loss = 0.002603
grad_step = 000091, loss = 0.002528
grad_step = 000092, loss = 0.002511
grad_step = 000093, loss = 0.002459
grad_step = 000094, loss = 0.002456
grad_step = 000095, loss = 0.002407
grad_step = 000096, loss = 0.002396
grad_step = 000097, loss = 0.002381
grad_step = 000098, loss = 0.002351
grad_step = 000099, loss = 0.002344
grad_step = 000100, loss = 0.002331
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002315
grad_step = 000102, loss = 0.002311
grad_step = 000103, loss = 0.002303
grad_step = 000104, loss = 0.002293
grad_step = 000105, loss = 0.002286
grad_step = 000106, loss = 0.002283
grad_step = 000107, loss = 0.002275
grad_step = 000108, loss = 0.002270
grad_step = 000109, loss = 0.002269
grad_step = 000110, loss = 0.002260
grad_step = 000111, loss = 0.002258
grad_step = 000112, loss = 0.002254
grad_step = 000113, loss = 0.002249
grad_step = 000114, loss = 0.002247
grad_step = 000115, loss = 0.002243
grad_step = 000116, loss = 0.002239
grad_step = 000117, loss = 0.002236
grad_step = 000118, loss = 0.002234
grad_step = 000119, loss = 0.002230
grad_step = 000120, loss = 0.002228
grad_step = 000121, loss = 0.002226
grad_step = 000122, loss = 0.002223
grad_step = 000123, loss = 0.002221
grad_step = 000124, loss = 0.002219
grad_step = 000125, loss = 0.002216
grad_step = 000126, loss = 0.002214
grad_step = 000127, loss = 0.002211
grad_step = 000128, loss = 0.002209
grad_step = 000129, loss = 0.002207
grad_step = 000130, loss = 0.002205
grad_step = 000131, loss = 0.002202
grad_step = 000132, loss = 0.002199
grad_step = 000133, loss = 0.002197
grad_step = 000134, loss = 0.002194
grad_step = 000135, loss = 0.002192
grad_step = 000136, loss = 0.002189
grad_step = 000137, loss = 0.002186
grad_step = 000138, loss = 0.002183
grad_step = 000139, loss = 0.002180
grad_step = 000140, loss = 0.002177
grad_step = 000141, loss = 0.002175
grad_step = 000142, loss = 0.002172
grad_step = 000143, loss = 0.002169
grad_step = 000144, loss = 0.002167
grad_step = 000145, loss = 0.002165
grad_step = 000146, loss = 0.002161
grad_step = 000147, loss = 0.002157
grad_step = 000148, loss = 0.002155
grad_step = 000149, loss = 0.002153
grad_step = 000150, loss = 0.002150
grad_step = 000151, loss = 0.002146
grad_step = 000152, loss = 0.002144
grad_step = 000153, loss = 0.002142
grad_step = 000154, loss = 0.002140
grad_step = 000155, loss = 0.002136
grad_step = 000156, loss = 0.002133
grad_step = 000157, loss = 0.002130
grad_step = 000158, loss = 0.002128
grad_step = 000159, loss = 0.002126
grad_step = 000160, loss = 0.002124
grad_step = 000161, loss = 0.002121
grad_step = 000162, loss = 0.002118
grad_step = 000163, loss = 0.002114
grad_step = 000164, loss = 0.002111
grad_step = 000165, loss = 0.002108
grad_step = 000166, loss = 0.002106
grad_step = 000167, loss = 0.002105
grad_step = 000168, loss = 0.002105
grad_step = 000169, loss = 0.002106
grad_step = 000170, loss = 0.002105
grad_step = 000171, loss = 0.002100
grad_step = 000172, loss = 0.002092
grad_step = 000173, loss = 0.002088
grad_step = 000174, loss = 0.002089
grad_step = 000175, loss = 0.002090
grad_step = 000176, loss = 0.002089
grad_step = 000177, loss = 0.002084
grad_step = 000178, loss = 0.002078
grad_step = 000179, loss = 0.002075
grad_step = 000180, loss = 0.002074
grad_step = 000181, loss = 0.002075
grad_step = 000182, loss = 0.002075
grad_step = 000183, loss = 0.002074
grad_step = 000184, loss = 0.002071
grad_step = 000185, loss = 0.002066
grad_step = 000186, loss = 0.002061
grad_step = 000187, loss = 0.002058
grad_step = 000188, loss = 0.002056
grad_step = 000189, loss = 0.002055
grad_step = 000190, loss = 0.002055
grad_step = 000191, loss = 0.002058
grad_step = 000192, loss = 0.002063
grad_step = 000193, loss = 0.002071
grad_step = 000194, loss = 0.002077
grad_step = 000195, loss = 0.002075
grad_step = 000196, loss = 0.002058
grad_step = 000197, loss = 0.002042
grad_step = 000198, loss = 0.002037
grad_step = 000199, loss = 0.002045
grad_step = 000200, loss = 0.002053
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002051
grad_step = 000202, loss = 0.002041
grad_step = 000203, loss = 0.002029
grad_step = 000204, loss = 0.002026
grad_step = 000205, loss = 0.002030
grad_step = 000206, loss = 0.002035
grad_step = 000207, loss = 0.002035
grad_step = 000208, loss = 0.002029
grad_step = 000209, loss = 0.002021
grad_step = 000210, loss = 0.002015
grad_step = 000211, loss = 0.002014
grad_step = 000212, loss = 0.002015
grad_step = 000213, loss = 0.002017
grad_step = 000214, loss = 0.002019
grad_step = 000215, loss = 0.002018
grad_step = 000216, loss = 0.002015
grad_step = 000217, loss = 0.002011
grad_step = 000218, loss = 0.002006
grad_step = 000219, loss = 0.002002
grad_step = 000220, loss = 0.001998
grad_step = 000221, loss = 0.001995
grad_step = 000222, loss = 0.001993
grad_step = 000223, loss = 0.001991
grad_step = 000224, loss = 0.001989
grad_step = 000225, loss = 0.001988
grad_step = 000226, loss = 0.001989
grad_step = 000227, loss = 0.001992
grad_step = 000228, loss = 0.001999
grad_step = 000229, loss = 0.002018
grad_step = 000230, loss = 0.002052
grad_step = 000231, loss = 0.002102
grad_step = 000232, loss = 0.002114
grad_step = 000233, loss = 0.002072
grad_step = 000234, loss = 0.001989
grad_step = 000235, loss = 0.001977
grad_step = 000236, loss = 0.002028
grad_step = 000237, loss = 0.002040
grad_step = 000238, loss = 0.001997
grad_step = 000239, loss = 0.001962
grad_step = 000240, loss = 0.001982
grad_step = 000241, loss = 0.002011
grad_step = 000242, loss = 0.001993
grad_step = 000243, loss = 0.001959
grad_step = 000244, loss = 0.001956
grad_step = 000245, loss = 0.001977
grad_step = 000246, loss = 0.001981
grad_step = 000247, loss = 0.001958
grad_step = 000248, loss = 0.001944
grad_step = 000249, loss = 0.001952
grad_step = 000250, loss = 0.001961
grad_step = 000251, loss = 0.001954
grad_step = 000252, loss = 0.001938
grad_step = 000253, loss = 0.001934
grad_step = 000254, loss = 0.001941
grad_step = 000255, loss = 0.001943
grad_step = 000256, loss = 0.001936
grad_step = 000257, loss = 0.001926
grad_step = 000258, loss = 0.001923
grad_step = 000259, loss = 0.001926
grad_step = 000260, loss = 0.001927
grad_step = 000261, loss = 0.001922
grad_step = 000262, loss = 0.001915
grad_step = 000263, loss = 0.001910
grad_step = 000264, loss = 0.001910
grad_step = 000265, loss = 0.001911
grad_step = 000266, loss = 0.001909
grad_step = 000267, loss = 0.001905
grad_step = 000268, loss = 0.001900
grad_step = 000269, loss = 0.001896
grad_step = 000270, loss = 0.001893
grad_step = 000271, loss = 0.001892
grad_step = 000272, loss = 0.001891
grad_step = 000273, loss = 0.001889
grad_step = 000274, loss = 0.001887
grad_step = 000275, loss = 0.001883
grad_step = 000276, loss = 0.001880
grad_step = 000277, loss = 0.001876
grad_step = 000278, loss = 0.001872
grad_step = 000279, loss = 0.001869
grad_step = 000280, loss = 0.001865
grad_step = 000281, loss = 0.001862
grad_step = 000282, loss = 0.001859
grad_step = 000283, loss = 0.001856
grad_step = 000284, loss = 0.001853
grad_step = 000285, loss = 0.001849
grad_step = 000286, loss = 0.001846
grad_step = 000287, loss = 0.001843
grad_step = 000288, loss = 0.001840
grad_step = 000289, loss = 0.001836
grad_step = 000290, loss = 0.001834
grad_step = 000291, loss = 0.001832
grad_step = 000292, loss = 0.001833
grad_step = 000293, loss = 0.001841
grad_step = 000294, loss = 0.001872
grad_step = 000295, loss = 0.001959
grad_step = 000296, loss = 0.002172
grad_step = 000297, loss = 0.002442
grad_step = 000298, loss = 0.002624
grad_step = 000299, loss = 0.002186
grad_step = 000300, loss = 0.001812
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002025
grad_step = 000302, loss = 0.002239
grad_step = 000303, loss = 0.001988
grad_step = 000304, loss = 0.001807
grad_step = 000305, loss = 0.002033
grad_step = 000306, loss = 0.002060
grad_step = 000307, loss = 0.001806
grad_step = 000308, loss = 0.001873
grad_step = 000309, loss = 0.002004
grad_step = 000310, loss = 0.001847
grad_step = 000311, loss = 0.001782
grad_step = 000312, loss = 0.001917
grad_step = 000313, loss = 0.001875
grad_step = 000314, loss = 0.001754
grad_step = 000315, loss = 0.001814
grad_step = 000316, loss = 0.001859
grad_step = 000317, loss = 0.001776
grad_step = 000318, loss = 0.001739
grad_step = 000319, loss = 0.001806
grad_step = 000320, loss = 0.001786
grad_step = 000321, loss = 0.001723
grad_step = 000322, loss = 0.001738
grad_step = 000323, loss = 0.001769
grad_step = 000324, loss = 0.001736
grad_step = 000325, loss = 0.001695
grad_step = 000326, loss = 0.001718
grad_step = 000327, loss = 0.001729
grad_step = 000328, loss = 0.001694
grad_step = 000329, loss = 0.001669
grad_step = 000330, loss = 0.001684
grad_step = 000331, loss = 0.001690
grad_step = 000332, loss = 0.001662
grad_step = 000333, loss = 0.001640
grad_step = 000334, loss = 0.001640
grad_step = 000335, loss = 0.001649
grad_step = 000336, loss = 0.001639
grad_step = 000337, loss = 0.001616
grad_step = 000338, loss = 0.001597
grad_step = 000339, loss = 0.001588
grad_step = 000340, loss = 0.001589
grad_step = 000341, loss = 0.001590
grad_step = 000342, loss = 0.001590
grad_step = 000343, loss = 0.001687
grad_step = 000344, loss = 0.001835
grad_step = 000345, loss = 0.002002
grad_step = 000346, loss = 0.001964
grad_step = 000347, loss = 0.001737
grad_step = 000348, loss = 0.001593
grad_step = 000349, loss = 0.001649
grad_step = 000350, loss = 0.001743
grad_step = 000351, loss = 0.001686
grad_step = 000352, loss = 0.001694
grad_step = 000353, loss = 0.001573
grad_step = 000354, loss = 0.001512
grad_step = 000355, loss = 0.001628
grad_step = 000356, loss = 0.001640
grad_step = 000357, loss = 0.001457
grad_step = 000358, loss = 0.001462
grad_step = 000359, loss = 0.001552
grad_step = 000360, loss = 0.001483
grad_step = 000361, loss = 0.001419
grad_step = 000362, loss = 0.001438
grad_step = 000363, loss = 0.001452
grad_step = 000364, loss = 0.001366
grad_step = 000365, loss = 0.001352
grad_step = 000366, loss = 0.001408
grad_step = 000367, loss = 0.001599
grad_step = 000368, loss = 0.001790
grad_step = 000369, loss = 0.001725
grad_step = 000370, loss = 0.001604
grad_step = 000371, loss = 0.001516
grad_step = 000372, loss = 0.001356
grad_step = 000373, loss = 0.001266
grad_step = 000374, loss = 0.001319
grad_step = 000375, loss = 0.001422
grad_step = 000376, loss = 0.001410
grad_step = 000377, loss = 0.001258
grad_step = 000378, loss = 0.001182
grad_step = 000379, loss = 0.001218
grad_step = 000380, loss = 0.001230
grad_step = 000381, loss = 0.001165
grad_step = 000382, loss = 0.001163
grad_step = 000383, loss = 0.001568
grad_step = 000384, loss = 0.002511
grad_step = 000385, loss = 0.001254
grad_step = 000386, loss = 0.002268
grad_step = 000387, loss = 0.003527
grad_step = 000388, loss = 0.002834
grad_step = 000389, loss = 0.002976
grad_step = 000390, loss = 0.002551
grad_step = 000391, loss = 0.002597
grad_step = 000392, loss = 0.002487
grad_step = 000393, loss = 0.002120
grad_step = 000394, loss = 0.002460
grad_step = 000395, loss = 0.001914
grad_step = 000396, loss = 0.002280
grad_step = 000397, loss = 0.002005
grad_step = 000398, loss = 0.002034
grad_step = 000399, loss = 0.002038
grad_step = 000400, loss = 0.001983
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001921
grad_step = 000402, loss = 0.001942
grad_step = 000403, loss = 0.001981
grad_step = 000404, loss = 0.001782
grad_step = 000405, loss = 0.001945
grad_step = 000406, loss = 0.001808
grad_step = 000407, loss = 0.001824
grad_step = 000408, loss = 0.001828
grad_step = 000409, loss = 0.001778
grad_step = 000410, loss = 0.001788
grad_step = 000411, loss = 0.001758
grad_step = 000412, loss = 0.001770
grad_step = 000413, loss = 0.001713
grad_step = 000414, loss = 0.001757
grad_step = 000415, loss = 0.001716
grad_step = 000416, loss = 0.001706
grad_step = 000417, loss = 0.001728
grad_step = 000418, loss = 0.001697
grad_step = 000419, loss = 0.001703
grad_step = 000420, loss = 0.001695
grad_step = 000421, loss = 0.001696
grad_step = 000422, loss = 0.001676
grad_step = 000423, loss = 0.001686
grad_step = 000424, loss = 0.001676
grad_step = 000425, loss = 0.001667
grad_step = 000426, loss = 0.001675
grad_step = 000427, loss = 0.001656
grad_step = 000428, loss = 0.001666
grad_step = 000429, loss = 0.001655
grad_step = 000430, loss = 0.001650
grad_step = 000431, loss = 0.001654
grad_step = 000432, loss = 0.001643
grad_step = 000433, loss = 0.001646
grad_step = 000434, loss = 0.001640
grad_step = 000435, loss = 0.001639
grad_step = 000436, loss = 0.001635
grad_step = 000437, loss = 0.001634
grad_step = 000438, loss = 0.001632
grad_step = 000439, loss = 0.001628
grad_step = 000440, loss = 0.001629
grad_step = 000441, loss = 0.001623
grad_step = 000442, loss = 0.001624
grad_step = 000443, loss = 0.001621
grad_step = 000444, loss = 0.001618
grad_step = 000445, loss = 0.001620
grad_step = 000446, loss = 0.001617
grad_step = 000447, loss = 0.001621
grad_step = 000448, loss = 0.001630
grad_step = 000449, loss = 0.001649
grad_step = 000450, loss = 0.001699
grad_step = 000451, loss = 0.001808
grad_step = 000452, loss = 0.002001
grad_step = 000453, loss = 0.002319
grad_step = 000454, loss = 0.002432
grad_step = 000455, loss = 0.002379
grad_step = 000456, loss = 0.001796
grad_step = 000457, loss = 0.001634
grad_step = 000458, loss = 0.001960
grad_step = 000459, loss = 0.001969
grad_step = 000460, loss = 0.001676
grad_step = 000461, loss = 0.001676
grad_step = 000462, loss = 0.001871
grad_step = 000463, loss = 0.001781
grad_step = 000464, loss = 0.001607
grad_step = 000465, loss = 0.001699
grad_step = 000466, loss = 0.001800
grad_step = 000467, loss = 0.001670
grad_step = 000468, loss = 0.001596
grad_step = 000469, loss = 0.001692
grad_step = 000470, loss = 0.001707
grad_step = 000471, loss = 0.001607
grad_step = 000472, loss = 0.001596
grad_step = 000473, loss = 0.001665
grad_step = 000474, loss = 0.001654
grad_step = 000475, loss = 0.001587
grad_step = 000476, loss = 0.001588
grad_step = 000477, loss = 0.001635
grad_step = 000478, loss = 0.001626
grad_step = 000479, loss = 0.001580
grad_step = 000480, loss = 0.001573
grad_step = 000481, loss = 0.001603
grad_step = 000482, loss = 0.001609
grad_step = 000483, loss = 0.001580
grad_step = 000484, loss = 0.001561
grad_step = 000485, loss = 0.001573
grad_step = 000486, loss = 0.001589
grad_step = 000487, loss = 0.001581
grad_step = 000488, loss = 0.001561
grad_step = 000489, loss = 0.001554
grad_step = 000490, loss = 0.001562
grad_step = 000491, loss = 0.001570
grad_step = 000492, loss = 0.001564
grad_step = 000493, loss = 0.001552
grad_step = 000494, loss = 0.001546
grad_step = 000495, loss = 0.001549
grad_step = 000496, loss = 0.001554
grad_step = 000497, loss = 0.001552
grad_step = 000498, loss = 0.001545
grad_step = 000499, loss = 0.001538
grad_step = 000500, loss = 0.001537
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001539
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

  date_run                              2020-05-11 10:14:18.819144
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.20809
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 10:14:18.825230
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.10138
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 10:14:18.832064
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.128964
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 10:14:18.838172
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.540506
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
0   2020-05-11 10:13:45.409574  ...    mean_absolute_error
1   2020-05-11 10:13:45.413777  ...     mean_squared_error
2   2020-05-11 10:13:45.417520  ...  median_absolute_error
3   2020-05-11 10:13:45.421098  ...               r2_score
4   2020-05-11 10:13:54.740627  ...    mean_absolute_error
5   2020-05-11 10:13:54.745339  ...     mean_squared_error
6   2020-05-11 10:13:54.749158  ...  median_absolute_error
7   2020-05-11 10:13:54.752765  ...               r2_score
8   2020-05-11 10:14:18.819144  ...    mean_absolute_error
9   2020-05-11 10:14:18.825230  ...     mean_squared_error
10  2020-05-11 10:14:18.832064  ...  median_absolute_error
11  2020-05-11 10:14:18.838172  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147076.27it/s] 75%|███████▌  | 7454720/9912422 [00:00<00:11, 209931.06it/s]9920512it [00:00, 44468952.33it/s]                           
0it [00:00, ?it/s]32768it [00:00, 650993.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163422.86it/s]1654784it [00:00, 10014005.49it/s]                         
0it [00:00, ?it/s]8192it [00:00, 183038.15it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cabfeffd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c4970cef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cabf7aef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c491e30f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cabfeffd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c5e974e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cabf7aef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c52a24748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cabfb7ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c52a24748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c491e3438> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f05605131d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a1368bd9ea20a79421cdcc235ae4851c00751b57509ba3a834ec8070343cf11d
  Stored in directory: /tmp/pip-ephem-wheel-cache-dxmivztj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0500240dd8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3457024/17464789 [====>.........................] - ETA: 0s
 9584640/17464789 [===============>..............] - ETA: 0s
15228928/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 10:15:47.126135: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 10:15:47.131345: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 10:15:47.131543: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55672ca5cad0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 10:15:47.131560: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7464 - accuracy: 0.4948
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7510 - accuracy: 0.4945
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7345 - accuracy: 0.4956
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7299 - accuracy: 0.4959
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7092 - accuracy: 0.4972
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7050 - accuracy: 0.4975
11000/25000 [============>.................] - ETA: 4s - loss: 7.7210 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 4s - loss: 7.7216 - accuracy: 0.4964
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7291 - accuracy: 0.4959
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7269 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 3s - loss: 7.6758 - accuracy: 0.4994
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6567 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6881 - accuracy: 0.4986
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 10:16:04.464034
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 10:16:04.464034  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 10:16:11.140197: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 10:16:11.145417: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 10:16:11.145649: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d235090420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 10:16:11.145667: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f140ad35be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3043 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.2533 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f13e6a75860> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8200 - accuracy: 0.4900
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7382 - accuracy: 0.4953 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6145 - accuracy: 0.5034
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5420 - accuracy: 0.5081
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5542 - accuracy: 0.5073
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
11000/25000 [============>.................] - ETA: 4s - loss: 7.6332 - accuracy: 0.5022
12000/25000 [=============>................] - ETA: 4s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6396 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6351 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6255 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6390 - accuracy: 0.5018
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f13e41005f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<100:59:00, 2.37kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<70:55:02, 3.38kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<49:41:15, 4.82kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<34:46:04, 6.88kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<24:15:48, 9.83kB/s].vector_cache/glove.6B.zip:   1%|          | 8.24M/862M [00:04<16:53:42, 14.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 12.9M/862M [00:04<11:45:47, 20.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:04<8:11:34, 28.6kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:04<5:42:27, 40.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:04<3:58:26, 58.4kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:04<2:46:15, 83.4kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:04<1:55:47, 119kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 38.8M/862M [00:04<1:20:48, 170kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:04<56:18, 242kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:05<39:20, 345kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:05<27:46, 486kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:07<21:15, 632kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:07<16:45, 802kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<12:07, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:09<11:01, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:09<08:53, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:09<06:33, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<04:45, 2.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:11<18:29, 719kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:11<14:15, 932kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:11<10:17, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:13<10:21, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:13<09:56, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:13<07:37, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:15<07:27, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:15<06:33, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:15<04:52, 2.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:17<06:29, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:17<05:51, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:17<04:22, 2.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:19<06:09, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:19<05:38, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:19<04:16, 3.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:21<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:21<06:51, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:21<05:27, 2.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:23<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:23<05:26, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:23<04:07, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:25<05:52, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:25<05:26, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:25<04:04, 3.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:27<05:51, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:27<06:40, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:27<05:19, 2.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<03:51, 3.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<12:15:39, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<8:35:59, 24.6kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<6:00:46, 35.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<4:14:47, 49.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<3:00:51, 69.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<2:07:07, 99.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:30:39, 138kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<1:05:59, 190kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<46:42, 268kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:33<32:46, 381kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<29:22, 425kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<21:49, 571kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<15:33, 799kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<13:46, 901kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<12:09, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<09:04, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:27, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<21:17, 579kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<15:26, 798kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<10:54, 1.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<52:43, 233kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<38:08, 322kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<26:56, 454kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<21:42, 562kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<16:24, 743kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<11:43, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<11:04, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<08:59, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<06:34, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<07:27, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:26, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<04:46, 2.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<06:11, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<04:10, 2.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:44, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<06:28, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:07, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:52<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<03:47, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<06:11, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<04:56, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<03:41, 3.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<05:19, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<04:55, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<03:43, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<05:20, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<04:55, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<03:43, 3.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<04:53, 2.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<03:42, 3.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<05:18, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<06:02, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<04:43, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<03:26, 3.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<08:19, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<05:03, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<06:11, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<06:39, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<05:07, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<03:44, 3.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<07:16, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:15, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<04:39, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:50, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<06:22, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<05:01, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<03:37, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<10:42:08, 17.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<7:30:23, 24.6kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<5:14:46, 35.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<3:42:13, 49.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<2:36:35, 70.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<1:49:37, 100kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<1:19:05, 139kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<57:34, 190kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<40:43, 269kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<28:32, 382kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<27:07, 402kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<20:06, 541kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<14:16, 761kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<12:30, 865kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<10:58, 987kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<08:08, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<05:51, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<08:14, 1.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<06:59, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<05:08, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<05:54, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<06:36, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<05:09, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<03:43, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<10:11, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<08:20, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<06:04, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<06:54, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<03:51, 2.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<11:18, 927kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<09:05, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<06:38, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:33<06:53, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<07:07, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<05:30, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<03:57, 2.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<11:30, 899kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<09:12, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<06:40, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<06:52, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<07:05, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<05:28, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<03:59, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<06:26, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<05:26, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<04:03, 2.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<02:59, 3.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<10:10, 996kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<09:29, 1.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<07:08, 1.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<07:39, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<06:18, 1.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<03:25, 2.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<11:59, 833kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<10:43, 932kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<08:03, 1.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<05:44, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<13:16, 748kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<10:24, 953kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<07:32, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<07:23, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<07:30, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<05:41, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<04:18, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<05:02, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<04:47, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<03:36, 2.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<04:26, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<05:22, 1.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<04:18, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<03:07, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<11:10, 864kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<08:54, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<06:27, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<06:34, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<06:49, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<05:14, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:58<03:47, 2.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:54, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<05:54, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<04:20, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<05:04, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<05:38, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<04:28, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<03:14, 2.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<10:59, 852kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<08:34, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:03<06:12, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<04:30, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<11:57, 778kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<10:25, 892kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<07:49, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<05:35, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<12:58, 711kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<10:06, 913kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<07:16, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:09<07:02, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<07:02, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<05:23, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<06:48, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<05:48, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<04:18, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<04:56, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<05:33, 1.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<04:19, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<03:08, 2.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<05:47, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<05:03, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<03:47, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<04:33, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<04:01, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<03:00, 2.94MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<02:14, 3.94MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<09:18, 948kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<08:27, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<06:20, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:32, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<07:06, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<05:56, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<04:21, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<04:55, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<04:11, 2.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:01, 2.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<07:07, 1.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<05:56, 1.45MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<04:21, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<04:53, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<05:18, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<04:08, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<02:59, 2.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<06:24, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<05:24, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<03:59, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<04:36, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<04:07, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<03:06, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<04:00, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<03:45, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<02:48, 2.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<03:45, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<04:27, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<03:29, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<02:33, 3.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<05:00, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:25, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<03:16, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<04:02, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<03:40, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<02:41, 3.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<04:51, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<03:13, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<03:57, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<03:40, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<02:45, 2.89MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<03:36, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<04:17, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<03:23, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<02:26, 3.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<07:24, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<06:04, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<04:27, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<04:47, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<05:04, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<03:55, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<02:51, 2.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<03:58, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<02:59, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<03:24, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<02:33, 2.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<03:29, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<04:11, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<03:17, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<02:25, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<04:01, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<02:43, 2.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<03:31, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<04:07, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<03:15, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<02:21, 3.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<05:51, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:53, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<03:36, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<04:08, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<04:26, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:26, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<02:32, 2.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:52, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:32, 2.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<02:38, 2.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<03:23, 2.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:57, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:08, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:16, 3.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<07:53, 898kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<06:19, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<04:37, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<04:43, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:11<04:48, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:40, 1.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<02:41, 2.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:30, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<02:37, 2.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:18, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:55, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<03:04, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<02:15, 3.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<04:14, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<02:46, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<03:23, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<03:55, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:19<03:04, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<02:14, 2.99MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<04:09, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<03:38, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<02:41, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<03:20, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<03:47, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<02:58, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:23<02:09, 3.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<04:07, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<03:36, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<02:40, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<03:15, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<03:46, 1.71MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<02:08, 3.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<04:34, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<03:52, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:50, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<03:23, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<03:43, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<02:53, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:05, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:33<05:04, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<04:07, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<03:01, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<02:13, 2.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<07:29, 825kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<05:53, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<04:15, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<04:19, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<04:27, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:24, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:27, 2.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<04:13, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<03:38, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<02:42, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<03:10, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<03:33, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<02:00, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<03:49, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:25, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<02:56, 1.98MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<02:42, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<02:01, 2.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<02:39, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<03:11, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<02:31, 2.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<01:49, 3.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<04:31, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<03:46, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<02:45, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<03:10, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:05, 2.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<01:31, 3.65MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<07:43, 719kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<06:41, 829kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<04:56, 1.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<03:33, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<03:21, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:29, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<02:56, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<03:18, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:37, 2.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<01:53, 2.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<06:35, 811kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<05:12, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<03:45, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<03:44, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<03:46, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:54, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<02:04, 2.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<05:30, 945kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<04:26, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<03:14, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<03:21, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<03:29, 1.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:41, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<01:56, 2.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<03:51, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<02:23, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<03:04, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<02:23, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<01:43, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<04:06, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<03:24, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<02:31, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:04, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:13<02:23, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<01:43, 2.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<05:28, 875kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<04:21, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<03:10, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<03:13, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<02:27, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<01:37, 2.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:28, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<01:55, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<01:24, 3.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:35, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:19, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<01:43, 2.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:29, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:57, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:25<01:24, 3.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<03:46, 1.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<03:02, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:12, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:36, 2.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<07:11, 598kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<06:01, 713kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<04:26, 966kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:29<03:08, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<05:40, 746kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<04:26, 950kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<03:12, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<03:07, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<03:08, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:23, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<01:45, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:17, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:04, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<01:33, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<01:56, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:15, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<01:46, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:16, 3.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<02:39, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<02:12, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<02:23, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<01:51, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:20, 2.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<02:31, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<02:05, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<01:33, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<01:55, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<01:44, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:14, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<03:50, 956kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<03:06, 1.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<02:14, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<02:19, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<02:00, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<01:29, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<01:50, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<02:03, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<01:37, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:10, 2.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<03:43, 929kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<02:59, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<02:09, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<02:14, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<01:56, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<01:26, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<01:42, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<01:58, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<01:32, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<01:06, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<02:02, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<01:47, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:58<01:19, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<01:36, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<01:29, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<01:06, 2.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:26, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:42, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<01:20, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<00:58, 3.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:04<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<01:35, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<01:10, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<00:51, 3.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<03:55, 758kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<03:26, 864kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<02:32, 1.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:47, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<02:45, 1.05MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<02:15, 1.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<01:44, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<01:32, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:09, 2.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<01:23, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<01:35, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<01:14, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<00:54, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<01:23, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:01, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<01:16, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<01:28, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<00:49, 3.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<01:50, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:08, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<01:21, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<01:10, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<00:59, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<00:43, 3.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<01:20, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<01:12, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<00:54, 2.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:07, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:20, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:24<01:02, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<00:45, 3.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:36, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:22, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:00, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<01:10, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<01:19, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<00:45, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<01:10, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<01:02, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<00:46, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<00:59, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<01:08, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<00:53, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:37, 3.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<02:08, 941kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<01:42, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:13, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<01:16, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<01:45, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<01:26, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:03, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<01:06, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<01:10, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<00:55, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:39, 2.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<03:12, 561kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<02:37, 684kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:55, 930kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:19, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<06:45, 257kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<05:06, 339kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<03:37, 474kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<02:30, 670kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<02:25, 684kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<02:01, 820kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<01:29, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:01, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<13:42, 116kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<09:50, 162kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<06:54, 228kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<04:45, 325kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<03:57, 385kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<03:05, 493kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:48<02:13, 678kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<01:31, 958kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<03:03, 476kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<02:24, 603kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:43, 832kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<01:10, 1.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<21:20, 65.0kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<15:31, 89.2kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<10:58, 126kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<07:35, 179kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<05:26, 242kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<04:02, 326kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<02:50, 456kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<02:07, 587kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<01:44, 712kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<01:16, 969kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:52, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<01:30, 786kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<01:17, 906kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:57, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:39, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:53, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:48, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:25, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:59, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<01:09, 901kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:55, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:39, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:38, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:38, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:20, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<02:58, 304kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<02:16, 397kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<01:36, 552kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<01:06, 780kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:07<01:04, 780kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:07<00:55, 902kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:40, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:27, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:09<00:43, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:09<00:39, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:29, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:20, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:42, 984kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:38, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:28, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:26, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:26, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:20, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<01:37, 345kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<01:15, 447kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:53, 617kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:34, 873kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<01:38, 302kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<01:13, 403kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:50, 562kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:36, 704kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:35, 714kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:27, 919kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:18, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:16, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:14, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:10, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:09, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:10, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:07, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:04, 2.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:25<00:23, 564kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:25<00:18, 688kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:12, 942kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:07, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:10, 849kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:09, 965kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:06, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 1.80MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:03, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:29<00:03, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:02, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.51MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 1.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 1.41MB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 775/400000 [00:00<00:51, 7746.17it/s]  0%|          | 1485/400000 [00:00<00:52, 7527.01it/s]  1%|          | 2197/400000 [00:00<00:53, 7397.74it/s]  1%|          | 2987/400000 [00:00<00:52, 7539.20it/s]  1%|          | 3779/400000 [00:00<00:51, 7647.30it/s]  1%|          | 4556/400000 [00:00<00:51, 7682.45it/s]  1%|▏         | 5312/400000 [00:00<00:51, 7644.24it/s]  2%|▏         | 6082/400000 [00:00<00:51, 7659.43it/s]  2%|▏         | 6865/400000 [00:00<00:51, 7707.63it/s]  2%|▏         | 7633/400000 [00:01<00:50, 7697.04it/s]  2%|▏         | 8418/400000 [00:01<00:50, 7741.69it/s]  2%|▏         | 9178/400000 [00:01<00:51, 7548.34it/s]  2%|▏         | 9937/400000 [00:01<00:51, 7558.85it/s]  3%|▎         | 10687/400000 [00:01<00:52, 7482.52it/s]  3%|▎         | 11431/400000 [00:01<00:52, 7448.45it/s]  3%|▎         | 12224/400000 [00:01<00:51, 7585.99it/s]  3%|▎         | 12982/400000 [00:01<00:51, 7545.99it/s]  3%|▎         | 13736/400000 [00:01<00:52, 7383.77it/s]  4%|▎         | 14475/400000 [00:01<00:52, 7298.82it/s]  4%|▍         | 15240/400000 [00:02<00:51, 7399.81it/s]  4%|▍         | 15999/400000 [00:02<00:51, 7455.58it/s]  4%|▍         | 16746/400000 [00:02<00:51, 7453.51it/s]  4%|▍         | 17492/400000 [00:02<00:51, 7381.81it/s]  5%|▍         | 18268/400000 [00:02<00:50, 7489.95it/s]  5%|▍         | 19018/400000 [00:02<00:51, 7374.62it/s]  5%|▍         | 19793/400000 [00:02<00:50, 7479.53it/s]  5%|▌         | 20542/400000 [00:02<00:50, 7455.88it/s]  5%|▌         | 21336/400000 [00:02<00:49, 7592.52it/s]  6%|▌         | 22133/400000 [00:02<00:49, 7701.57it/s]  6%|▌         | 22907/400000 [00:03<00:48, 7711.00it/s]  6%|▌         | 23679/400000 [00:03<00:49, 7647.85it/s]  6%|▌         | 24445/400000 [00:03<00:49, 7574.83it/s]  6%|▋         | 25204/400000 [00:03<00:49, 7520.80it/s]  6%|▋         | 25958/400000 [00:03<00:49, 7525.17it/s]  7%|▋         | 26711/400000 [00:03<00:50, 7362.10it/s]  7%|▋         | 27523/400000 [00:03<00:49, 7572.14it/s]  7%|▋         | 28283/400000 [00:03<00:49, 7513.68it/s]  7%|▋         | 29060/400000 [00:03<00:48, 7588.48it/s]  7%|▋         | 29821/400000 [00:03<00:50, 7387.91it/s]  8%|▊         | 30562/400000 [00:04<00:50, 7346.51it/s]  8%|▊         | 31299/400000 [00:04<00:50, 7317.24it/s]  8%|▊         | 32032/400000 [00:04<00:50, 7215.18it/s]  8%|▊         | 32770/400000 [00:04<00:50, 7262.37it/s]  8%|▊         | 33521/400000 [00:04<00:49, 7332.81it/s]  9%|▊         | 34297/400000 [00:04<00:49, 7455.60it/s]  9%|▉         | 35079/400000 [00:04<00:48, 7560.80it/s]  9%|▉         | 35843/400000 [00:04<00:48, 7582.32it/s]  9%|▉         | 36602/400000 [00:04<00:47, 7575.37it/s]  9%|▉         | 37361/400000 [00:04<00:48, 7538.47it/s] 10%|▉         | 38116/400000 [00:05<00:48, 7445.94it/s] 10%|▉         | 38862/400000 [00:05<00:49, 7331.35it/s] 10%|▉         | 39596/400000 [00:05<00:49, 7328.77it/s] 10%|█         | 40330/400000 [00:05<00:49, 7237.80it/s] 10%|█         | 41117/400000 [00:05<00:48, 7415.49it/s] 10%|█         | 41891/400000 [00:05<00:47, 7507.64it/s] 11%|█         | 42686/400000 [00:05<00:46, 7632.89it/s] 11%|█         | 43451/400000 [00:05<00:47, 7475.02it/s] 11%|█         | 44201/400000 [00:05<00:48, 7371.48it/s] 11%|█         | 44950/400000 [00:06<00:47, 7403.93it/s] 11%|█▏        | 45703/400000 [00:06<00:47, 7437.51it/s] 12%|█▏        | 46464/400000 [00:06<00:47, 7487.10it/s] 12%|█▏        | 47214/400000 [00:06<00:47, 7487.20it/s] 12%|█▏        | 47964/400000 [00:06<00:47, 7344.10it/s] 12%|█▏        | 48700/400000 [00:06<00:48, 7228.00it/s] 12%|█▏        | 49445/400000 [00:06<00:48, 7291.56it/s] 13%|█▎        | 50188/400000 [00:06<00:47, 7331.23it/s] 13%|█▎        | 50922/400000 [00:06<00:47, 7277.15it/s] 13%|█▎        | 51660/400000 [00:06<00:47, 7306.89it/s] 13%|█▎        | 52396/400000 [00:07<00:47, 7321.18it/s] 13%|█▎        | 53129/400000 [00:07<00:47, 7236.82it/s] 13%|█▎        | 53854/400000 [00:07<00:48, 7193.73it/s] 14%|█▎        | 54592/400000 [00:07<00:47, 7248.36it/s] 14%|█▍        | 55352/400000 [00:07<00:46, 7349.59it/s] 14%|█▍        | 56100/400000 [00:07<00:46, 7386.49it/s] 14%|█▍        | 56840/400000 [00:07<00:46, 7314.31it/s] 14%|█▍        | 57573/400000 [00:07<00:46, 7318.70it/s] 15%|█▍        | 58306/400000 [00:07<00:46, 7315.50it/s] 15%|█▍        | 59060/400000 [00:07<00:46, 7381.03it/s] 15%|█▍        | 59799/400000 [00:08<00:46, 7259.98it/s] 15%|█▌        | 60526/400000 [00:08<00:47, 7179.67it/s] 15%|█▌        | 61247/400000 [00:08<00:47, 7188.53it/s] 15%|█▌        | 61973/400000 [00:08<00:46, 7207.31it/s] 16%|█▌        | 62695/400000 [00:08<00:47, 7175.32it/s] 16%|█▌        | 63451/400000 [00:08<00:46, 7284.78it/s] 16%|█▌        | 64208/400000 [00:08<00:45, 7365.99it/s] 16%|█▌        | 64980/400000 [00:08<00:44, 7466.88it/s] 16%|█▋        | 65728/400000 [00:08<00:44, 7461.46it/s] 17%|█▋        | 66513/400000 [00:08<00:44, 7573.71it/s] 17%|█▋        | 67272/400000 [00:09<00:44, 7475.78it/s] 17%|█▋        | 68021/400000 [00:09<00:44, 7457.50it/s] 17%|█▋        | 68779/400000 [00:09<00:44, 7491.68it/s] 17%|█▋        | 69536/400000 [00:09<00:43, 7513.62it/s] 18%|█▊        | 70292/400000 [00:09<00:43, 7526.19it/s] 18%|█▊        | 71045/400000 [00:09<00:44, 7443.88it/s] 18%|█▊        | 71790/400000 [00:09<00:44, 7432.50it/s] 18%|█▊        | 72570/400000 [00:09<00:43, 7537.59it/s] 18%|█▊        | 73325/400000 [00:09<00:44, 7419.63it/s] 19%|█▊        | 74073/400000 [00:09<00:43, 7437.12it/s] 19%|█▊        | 74819/400000 [00:10<00:43, 7442.51it/s] 19%|█▉        | 75564/400000 [00:10<00:43, 7428.83it/s] 19%|█▉        | 76345/400000 [00:10<00:42, 7537.41it/s] 19%|█▉        | 77100/400000 [00:10<00:43, 7485.03it/s] 19%|█▉        | 77891/400000 [00:10<00:42, 7606.79it/s] 20%|█▉        | 78720/400000 [00:10<00:41, 7798.70it/s] 20%|█▉        | 79532/400000 [00:10<00:40, 7892.36it/s] 20%|██        | 80323/400000 [00:10<00:40, 7887.38it/s] 20%|██        | 81113/400000 [00:10<00:41, 7729.24it/s] 20%|██        | 81888/400000 [00:10<00:42, 7558.37it/s] 21%|██        | 82646/400000 [00:11<00:42, 7507.42it/s] 21%|██        | 83399/400000 [00:11<00:43, 7273.92it/s] 21%|██        | 84129/400000 [00:11<00:44, 7123.17it/s] 21%|██        | 84844/400000 [00:11<00:44, 7130.49it/s] 21%|██▏       | 85589/400000 [00:11<00:43, 7220.88it/s] 22%|██▏       | 86340/400000 [00:11<00:42, 7303.67it/s] 22%|██▏       | 87072/400000 [00:11<00:43, 7274.37it/s] 22%|██▏       | 87801/400000 [00:11<00:42, 7275.30it/s] 22%|██▏       | 88548/400000 [00:11<00:42, 7332.08it/s] 22%|██▏       | 89335/400000 [00:11<00:41, 7485.49it/s] 23%|██▎       | 90105/400000 [00:12<00:41, 7546.65it/s] 23%|██▎       | 90873/400000 [00:12<00:40, 7583.11it/s] 23%|██▎       | 91633/400000 [00:12<00:41, 7444.43it/s] 23%|██▎       | 92379/400000 [00:12<00:42, 7312.70it/s] 23%|██▎       | 93112/400000 [00:12<00:42, 7306.14it/s] 23%|██▎       | 93844/400000 [00:12<00:41, 7297.72it/s] 24%|██▎       | 94599/400000 [00:12<00:41, 7369.42it/s] 24%|██▍       | 95353/400000 [00:12<00:41, 7417.26it/s] 24%|██▍       | 96096/400000 [00:12<00:41, 7377.87it/s] 24%|██▍       | 96836/400000 [00:13<00:41, 7383.96it/s] 24%|██▍       | 97575/400000 [00:13<00:41, 7241.07it/s] 25%|██▍       | 98309/400000 [00:13<00:41, 7269.08it/s] 25%|██▍       | 99037/400000 [00:13<00:41, 7195.97it/s] 25%|██▍       | 99758/400000 [00:13<00:43, 6831.42it/s] 25%|██▌       | 100473/400000 [00:13<00:43, 6923.38it/s] 25%|██▌       | 101214/400000 [00:13<00:42, 7058.65it/s] 26%|██▌       | 102000/400000 [00:13<00:40, 7281.14it/s] 26%|██▌       | 102743/400000 [00:13<00:40, 7323.30it/s] 26%|██▌       | 103479/400000 [00:13<00:41, 7222.00it/s] 26%|██▌       | 104223/400000 [00:14<00:40, 7284.29it/s] 26%|██▌       | 104997/400000 [00:14<00:39, 7414.83it/s] 26%|██▋       | 105756/400000 [00:14<00:39, 7464.81it/s] 27%|██▋       | 106504/400000 [00:14<00:39, 7396.82it/s] 27%|██▋       | 107245/400000 [00:14<00:40, 7179.54it/s] 27%|██▋       | 107966/400000 [00:14<00:40, 7187.06it/s] 27%|██▋       | 108687/400000 [00:14<00:41, 7081.02it/s] 27%|██▋       | 109442/400000 [00:14<00:40, 7210.15it/s] 28%|██▊       | 110182/400000 [00:14<00:39, 7265.86it/s] 28%|██▊       | 110917/400000 [00:14<00:39, 7288.31it/s] 28%|██▊       | 111675/400000 [00:15<00:39, 7372.29it/s] 28%|██▊       | 112414/400000 [00:15<00:39, 7370.78it/s] 28%|██▊       | 113152/400000 [00:15<00:39, 7344.83it/s] 28%|██▊       | 113907/400000 [00:15<00:38, 7404.32it/s] 29%|██▊       | 114648/400000 [00:15<00:39, 7312.18it/s] 29%|██▉       | 115380/400000 [00:15<00:39, 7267.96it/s] 29%|██▉       | 116147/400000 [00:15<00:38, 7382.96it/s] 29%|██▉       | 116910/400000 [00:15<00:37, 7454.17it/s] 29%|██▉       | 117671/400000 [00:15<00:37, 7497.15it/s] 30%|██▉       | 118422/400000 [00:15<00:38, 7385.86it/s] 30%|██▉       | 119178/400000 [00:16<00:37, 7437.10it/s] 30%|██▉       | 119932/400000 [00:16<00:37, 7467.17it/s] 30%|███       | 120680/400000 [00:16<00:39, 7146.01it/s] 30%|███       | 121398/400000 [00:16<00:39, 7102.26it/s] 31%|███       | 122111/400000 [00:16<00:39, 7044.94it/s] 31%|███       | 122833/400000 [00:16<00:39, 7094.14it/s] 31%|███       | 123544/400000 [00:16<00:39, 7021.95it/s] 31%|███       | 124248/400000 [00:16<00:40, 6833.33it/s] 31%|███       | 124934/400000 [00:16<00:41, 6626.77it/s] 31%|███▏      | 125618/400000 [00:17<00:41, 6689.00it/s] 32%|███▏      | 126339/400000 [00:17<00:40, 6834.53it/s] 32%|███▏      | 127047/400000 [00:17<00:39, 6904.98it/s] 32%|███▏      | 127802/400000 [00:17<00:38, 7085.90it/s] 32%|███▏      | 128543/400000 [00:17<00:37, 7178.60it/s] 32%|███▏      | 129276/400000 [00:17<00:37, 7223.22it/s] 33%|███▎      | 130028/400000 [00:17<00:36, 7309.53it/s] 33%|███▎      | 130797/400000 [00:17<00:36, 7418.21it/s] 33%|███▎      | 131541/400000 [00:17<00:36, 7336.38it/s] 33%|███▎      | 132276/400000 [00:17<00:36, 7308.80it/s] 33%|███▎      | 133048/400000 [00:18<00:35, 7425.36it/s] 33%|███▎      | 133792/400000 [00:18<00:37, 7042.03it/s] 34%|███▎      | 134533/400000 [00:18<00:37, 7148.20it/s] 34%|███▍      | 135252/400000 [00:18<00:37, 7086.08it/s] 34%|███▍      | 135964/400000 [00:18<00:37, 7044.33it/s] 34%|███▍      | 136756/400000 [00:18<00:36, 7283.28it/s] 34%|███▍      | 137508/400000 [00:18<00:35, 7351.85it/s] 35%|███▍      | 138266/400000 [00:18<00:35, 7416.98it/s] 35%|███▍      | 139010/400000 [00:18<00:35, 7410.04it/s] 35%|███▍      | 139753/400000 [00:18<00:35, 7411.28it/s] 35%|███▌      | 140496/400000 [00:19<00:35, 7291.17it/s] 35%|███▌      | 141240/400000 [00:19<00:35, 7334.20it/s] 36%|███▌      | 142004/400000 [00:19<00:34, 7420.29it/s] 36%|███▌      | 142780/400000 [00:19<00:34, 7518.25it/s] 36%|███▌      | 143533/400000 [00:19<00:34, 7418.11it/s] 36%|███▌      | 144276/400000 [00:19<00:34, 7395.54it/s] 36%|███▋      | 145017/400000 [00:19<00:34, 7369.76it/s] 36%|███▋      | 145762/400000 [00:19<00:34, 7391.93it/s] 37%|███▋      | 146506/400000 [00:19<00:34, 7405.41it/s] 37%|███▋      | 147247/400000 [00:19<00:35, 7122.87it/s] 37%|███▋      | 147970/400000 [00:20<00:35, 7153.39it/s] 37%|███▋      | 148689/400000 [00:20<00:35, 7162.31it/s] 37%|███▋      | 149450/400000 [00:20<00:34, 7288.83it/s] 38%|███▊      | 150189/400000 [00:20<00:34, 7318.85it/s] 38%|███▊      | 150922/400000 [00:20<00:34, 7251.51it/s] 38%|███▊      | 151660/400000 [00:20<00:34, 7287.08it/s] 38%|███▊      | 152411/400000 [00:20<00:33, 7352.40it/s] 38%|███▊      | 153205/400000 [00:20<00:32, 7517.04it/s] 38%|███▊      | 153959/400000 [00:20<00:32, 7500.70it/s] 39%|███▊      | 154711/400000 [00:20<00:33, 7425.06it/s] 39%|███▉      | 155469/400000 [00:21<00:32, 7469.96it/s] 39%|███▉      | 156217/400000 [00:21<00:32, 7471.09it/s] 39%|███▉      | 156965/400000 [00:21<00:32, 7383.36it/s] 39%|███▉      | 157708/400000 [00:21<00:32, 7397.28it/s] 40%|███▉      | 158449/400000 [00:21<00:33, 7274.02it/s] 40%|███▉      | 159215/400000 [00:21<00:32, 7381.96it/s] 40%|███▉      | 159955/400000 [00:21<00:33, 7213.94it/s] 40%|████      | 160678/400000 [00:21<00:33, 7202.46it/s] 40%|████      | 161405/400000 [00:21<00:33, 7220.49it/s] 41%|████      | 162128/400000 [00:22<00:33, 7203.99it/s] 41%|████      | 162884/400000 [00:22<00:32, 7307.06it/s] 41%|████      | 163616/400000 [00:22<00:32, 7277.42it/s] 41%|████      | 164345/400000 [00:22<00:32, 7265.64it/s] 41%|████▏     | 165072/400000 [00:22<00:32, 7261.15it/s] 41%|████▏     | 165799/400000 [00:22<00:32, 7201.21it/s] 42%|████▏     | 166528/400000 [00:22<00:32, 7226.22it/s] 42%|████▏     | 167269/400000 [00:22<00:31, 7278.48it/s] 42%|████▏     | 168038/400000 [00:22<00:31, 7395.99it/s] 42%|████▏     | 168829/400000 [00:22<00:30, 7541.46it/s] 42%|████▏     | 169585/400000 [00:23<00:30, 7529.29it/s] 43%|████▎     | 170351/400000 [00:23<00:30, 7567.07it/s] 43%|████▎     | 171109/400000 [00:23<00:30, 7466.31it/s] 43%|████▎     | 171857/400000 [00:23<00:30, 7387.30it/s] 43%|████▎     | 172597/400000 [00:23<00:32, 7034.80it/s] 43%|████▎     | 173305/400000 [00:23<00:32, 6998.91it/s] 44%|████▎     | 174017/400000 [00:23<00:32, 7032.89it/s] 44%|████▎     | 174751/400000 [00:23<00:31, 7120.94it/s] 44%|████▍     | 175500/400000 [00:23<00:31, 7225.82it/s] 44%|████▍     | 176246/400000 [00:23<00:30, 7290.86it/s] 44%|████▍     | 176977/400000 [00:24<00:30, 7252.02it/s] 44%|████▍     | 177704/400000 [00:24<00:31, 7154.82it/s] 45%|████▍     | 178457/400000 [00:24<00:30, 7261.86it/s] 45%|████▍     | 179208/400000 [00:24<00:30, 7334.39it/s] 45%|████▌     | 180013/400000 [00:24<00:29, 7535.00it/s] 45%|████▌     | 180775/400000 [00:24<00:29, 7558.57it/s] 45%|████▌     | 181567/400000 [00:24<00:28, 7661.90it/s] 46%|████▌     | 182360/400000 [00:24<00:28, 7739.57it/s] 46%|████▌     | 183195/400000 [00:24<00:27, 7910.91it/s] 46%|████▌     | 183995/400000 [00:24<00:27, 7935.58it/s] 46%|████▌     | 184790/400000 [00:25<00:27, 7793.71it/s] 46%|████▋     | 185611/400000 [00:25<00:27, 7912.80it/s] 47%|████▋     | 186412/400000 [00:25<00:26, 7940.88it/s] 47%|████▋     | 187208/400000 [00:25<00:27, 7881.07it/s] 47%|████▋     | 187997/400000 [00:25<00:26, 7869.06it/s] 47%|████▋     | 188785/400000 [00:25<00:26, 7837.78it/s] 47%|████▋     | 189616/400000 [00:25<00:26, 7971.01it/s] 48%|████▊     | 190414/400000 [00:25<00:26, 7929.68it/s] 48%|████▊     | 191234/400000 [00:25<00:26, 8006.95it/s] 48%|████▊     | 192055/400000 [00:25<00:25, 8064.88it/s] 48%|████▊     | 192863/400000 [00:26<00:26, 7813.07it/s] 48%|████▊     | 193659/400000 [00:26<00:26, 7855.24it/s] 49%|████▊     | 194447/400000 [00:26<00:26, 7818.84it/s] 49%|████▉     | 195231/400000 [00:26<00:26, 7824.05it/s] 49%|████▉     | 196015/400000 [00:26<00:26, 7745.30it/s] 49%|████▉     | 196791/400000 [00:26<00:26, 7543.85it/s] 49%|████▉     | 197548/400000 [00:26<00:27, 7445.82it/s] 50%|████▉     | 198301/400000 [00:26<00:27, 7469.57it/s] 50%|████▉     | 199049/400000 [00:26<00:26, 7471.28it/s] 50%|████▉     | 199823/400000 [00:26<00:26, 7549.06it/s] 50%|█████     | 200591/400000 [00:27<00:26, 7586.79it/s] 50%|█████     | 201364/400000 [00:27<00:26, 7628.50it/s] 51%|█████     | 202128/400000 [00:27<00:27, 7311.34it/s] 51%|█████     | 202863/400000 [00:27<00:27, 7216.99it/s] 51%|█████     | 203612/400000 [00:27<00:26, 7295.34it/s] 51%|█████     | 204344/400000 [00:27<00:26, 7254.43it/s] 51%|█████▏    | 205080/400000 [00:27<00:26, 7284.95it/s] 51%|█████▏    | 205820/400000 [00:27<00:26, 7317.57it/s] 52%|█████▏    | 206553/400000 [00:27<00:26, 7284.04it/s] 52%|█████▏    | 207282/400000 [00:28<00:26, 7256.63it/s] 52%|█████▏    | 208009/400000 [00:28<00:27, 7109.66it/s] 52%|█████▏    | 208727/400000 [00:28<00:26, 7129.02it/s] 52%|█████▏    | 209441/400000 [00:28<00:26, 7062.00it/s] 53%|█████▎    | 210164/400000 [00:28<00:26, 7109.60it/s] 53%|█████▎    | 210883/400000 [00:28<00:26, 7132.49it/s] 53%|█████▎    | 211606/400000 [00:28<00:26, 7154.74it/s] 53%|█████▎    | 212336/400000 [00:28<00:26, 7195.67it/s] 53%|█████▎    | 213083/400000 [00:28<00:25, 7275.24it/s] 53%|█████▎    | 213811/400000 [00:28<00:25, 7208.60it/s] 54%|█████▎    | 214584/400000 [00:29<00:25, 7355.86it/s] 54%|█████▍    | 215321/400000 [00:29<00:25, 7354.35it/s] 54%|█████▍    | 216084/400000 [00:29<00:24, 7433.03it/s] 54%|█████▍    | 216881/400000 [00:29<00:24, 7583.47it/s] 54%|█████▍    | 217641/400000 [00:29<00:24, 7539.84it/s] 55%|█████▍    | 218396/400000 [00:29<00:24, 7379.13it/s] 55%|█████▍    | 219136/400000 [00:29<00:24, 7290.47it/s] 55%|█████▍    | 219867/400000 [00:29<00:24, 7286.19it/s] 55%|█████▌    | 220597/400000 [00:29<00:24, 7256.31it/s] 55%|█████▌    | 221324/400000 [00:29<00:24, 7222.19it/s] 56%|█████▌    | 222047/400000 [00:30<00:25, 7031.38it/s] 56%|█████▌    | 222808/400000 [00:30<00:24, 7193.81it/s] 56%|█████▌    | 223553/400000 [00:30<00:24, 7268.75it/s] 56%|█████▌    | 224292/400000 [00:30<00:24, 7300.71it/s] 56%|█████▋    | 225047/400000 [00:30<00:23, 7370.58it/s] 56%|█████▋    | 225788/400000 [00:30<00:23, 7381.97it/s] 57%|█████▋    | 226527/400000 [00:30<00:23, 7362.01it/s] 57%|█████▋    | 227264/400000 [00:30<00:23, 7354.63it/s] 57%|█████▋    | 228040/400000 [00:30<00:23, 7470.75it/s] 57%|█████▋    | 228791/400000 [00:30<00:22, 7481.07it/s] 57%|█████▋    | 229540/400000 [00:31<00:23, 7406.25it/s] 58%|█████▊    | 230282/400000 [00:31<00:23, 7222.24it/s] 58%|█████▊    | 231054/400000 [00:31<00:22, 7363.59it/s] 58%|█████▊    | 231865/400000 [00:31<00:22, 7571.08it/s] 58%|█████▊    | 232651/400000 [00:31<00:21, 7653.27it/s] 58%|█████▊    | 233419/400000 [00:31<00:21, 7630.02it/s] 59%|█████▊    | 234184/400000 [00:31<00:21, 7597.90it/s] 59%|█████▊    | 234988/400000 [00:31<00:21, 7723.33it/s] 59%|█████▉    | 235762/400000 [00:31<00:22, 7406.78it/s] 59%|█████▉    | 236529/400000 [00:31<00:21, 7482.39it/s] 59%|█████▉    | 237280/400000 [00:32<00:21, 7398.60it/s] 60%|█████▉    | 238043/400000 [00:32<00:21, 7465.03it/s] 60%|█████▉    | 238798/400000 [00:32<00:21, 7489.10it/s] 60%|█████▉    | 239549/400000 [00:32<00:21, 7487.22it/s] 60%|██████    | 240299/400000 [00:32<00:21, 7484.46it/s] 60%|██████    | 241095/400000 [00:32<00:20, 7620.86it/s] 60%|██████    | 241859/400000 [00:32<00:20, 7592.78it/s] 61%|██████    | 242620/400000 [00:32<00:20, 7565.75it/s] 61%|██████    | 243378/400000 [00:32<00:20, 7528.17it/s] 61%|██████    | 244148/400000 [00:33<00:20, 7578.48it/s] 61%|██████    | 244915/400000 [00:33<00:20, 7603.68it/s] 61%|██████▏   | 245676/400000 [00:33<00:20, 7506.11it/s] 62%|██████▏   | 246428/400000 [00:33<00:20, 7483.30it/s] 62%|██████▏   | 247177/400000 [00:33<00:20, 7469.09it/s] 62%|██████▏   | 247963/400000 [00:33<00:20, 7580.04it/s] 62%|██████▏   | 248722/400000 [00:33<00:20, 7544.05it/s] 62%|██████▏   | 249510/400000 [00:33<00:19, 7639.71it/s] 63%|██████▎   | 250281/400000 [00:33<00:19, 7658.03it/s] 63%|██████▎   | 251087/400000 [00:33<00:19, 7771.12it/s] 63%|██████▎   | 251865/400000 [00:34<00:19, 7722.45it/s] 63%|██████▎   | 252638/400000 [00:34<00:19, 7572.81it/s] 63%|██████▎   | 253418/400000 [00:34<00:19, 7638.63it/s] 64%|██████▎   | 254183/400000 [00:34<00:19, 7612.69it/s] 64%|██████▎   | 254973/400000 [00:34<00:18, 7692.64it/s] 64%|██████▍   | 255743/400000 [00:34<00:18, 7689.34it/s] 64%|██████▍   | 256513/400000 [00:34<00:18, 7562.18it/s] 64%|██████▍   | 257295/400000 [00:34<00:18, 7635.77it/s] 65%|██████▍   | 258109/400000 [00:34<00:18, 7779.05it/s] 65%|██████▍   | 258889/400000 [00:34<00:18, 7694.03it/s] 65%|██████▍   | 259660/400000 [00:35<00:18, 7568.56it/s] 65%|██████▌   | 260419/400000 [00:35<00:18, 7524.06it/s] 65%|██████▌   | 261223/400000 [00:35<00:18, 7669.28it/s] 65%|██████▌   | 261992/400000 [00:35<00:18, 7621.23it/s] 66%|██████▌   | 262781/400000 [00:35<00:17, 7695.53it/s] 66%|██████▌   | 263552/400000 [00:35<00:17, 7630.04it/s] 66%|██████▌   | 264319/400000 [00:35<00:17, 7641.50it/s] 66%|██████▋   | 265084/400000 [00:35<00:17, 7562.37it/s] 66%|██████▋   | 265898/400000 [00:35<00:17, 7724.46it/s] 67%|██████▋   | 266672/400000 [00:35<00:17, 7712.08it/s] 67%|██████▋   | 267445/400000 [00:36<00:17, 7662.14it/s] 67%|██████▋   | 268228/400000 [00:36<00:17, 7710.08it/s] 67%|██████▋   | 269008/400000 [00:36<00:16, 7735.97it/s] 67%|██████▋   | 269799/400000 [00:36<00:16, 7786.71it/s] 68%|██████▊   | 270579/400000 [00:36<00:16, 7734.57it/s] 68%|██████▊   | 271353/400000 [00:36<00:16, 7705.08it/s] 68%|██████▊   | 272124/400000 [00:36<00:17, 7472.87it/s] 68%|██████▊   | 272891/400000 [00:36<00:16, 7530.37it/s] 68%|██████▊   | 273646/400000 [00:36<00:17, 7415.38it/s] 69%|██████▊   | 274430/400000 [00:36<00:16, 7537.66it/s] 69%|██████▉   | 275208/400000 [00:37<00:16, 7608.42it/s] 69%|██████▉   | 275976/400000 [00:37<00:16, 7628.07it/s] 69%|██████▉   | 276751/400000 [00:37<00:16, 7661.87it/s] 69%|██████▉   | 277530/400000 [00:37<00:15, 7699.15it/s] 70%|██████▉   | 278337/400000 [00:37<00:15, 7803.70it/s] 70%|██████▉   | 279119/400000 [00:37<00:15, 7760.34it/s] 70%|██████▉   | 279896/400000 [00:37<00:15, 7678.19it/s] 70%|███████   | 280677/400000 [00:37<00:15, 7715.34it/s] 70%|███████   | 281479/400000 [00:37<00:15, 7803.98it/s] 71%|███████   | 282269/400000 [00:37<00:15, 7831.37it/s] 71%|███████   | 283079/400000 [00:38<00:14, 7908.74it/s] 71%|███████   | 283871/400000 [00:38<00:14, 7875.16it/s] 71%|███████   | 284659/400000 [00:38<00:14, 7828.35it/s] 71%|███████▏  | 285443/400000 [00:38<00:15, 7623.12it/s] 72%|███████▏  | 286207/400000 [00:38<00:15, 7580.35it/s] 72%|███████▏  | 287020/400000 [00:38<00:14, 7735.22it/s] 72%|███████▏  | 287806/400000 [00:38<00:14, 7768.91it/s] 72%|███████▏  | 288584/400000 [00:38<00:14, 7592.31it/s] 72%|███████▏  | 289357/400000 [00:38<00:14, 7631.44it/s] 73%|███████▎  | 290133/400000 [00:39<00:14, 7669.47it/s] 73%|███████▎  | 290901/400000 [00:39<00:14, 7622.35it/s] 73%|███████▎  | 291686/400000 [00:39<00:14, 7687.13it/s] 73%|███████▎  | 292458/400000 [00:39<00:13, 7693.90it/s] 73%|███████▎  | 293228/400000 [00:39<00:13, 7659.77it/s] 73%|███████▎  | 293995/400000 [00:39<00:13, 7626.52it/s] 74%|███████▎  | 294758/400000 [00:39<00:13, 7522.41it/s] 74%|███████▍  | 295511/400000 [00:39<00:14, 7381.40it/s] 74%|███████▍  | 296251/400000 [00:39<00:14, 7386.63it/s] 74%|███████▍  | 296991/400000 [00:39<00:14, 7344.66it/s] 74%|███████▍  | 297727/400000 [00:40<00:13, 7345.12it/s] 75%|███████▍  | 298462/400000 [00:40<00:13, 7308.48it/s] 75%|███████▍  | 299194/400000 [00:40<00:13, 7253.42it/s] 75%|███████▍  | 299979/400000 [00:40<00:13, 7421.45it/s] 75%|███████▌  | 300743/400000 [00:40<00:13, 7485.37it/s] 75%|███████▌  | 301565/400000 [00:40<00:12, 7688.23it/s] 76%|███████▌  | 302336/400000 [00:40<00:12, 7690.85it/s] 76%|███████▌  | 303115/400000 [00:40<00:12, 7719.01it/s] 76%|███████▌  | 303935/400000 [00:40<00:12, 7854.56it/s] 76%|███████▌  | 304736/400000 [00:40<00:12, 7900.05it/s] 76%|███████▋  | 305548/400000 [00:41<00:11, 7961.37it/s] 77%|███████▋  | 306345/400000 [00:41<00:11, 7887.80it/s] 77%|███████▋  | 307135/400000 [00:41<00:11, 7776.91it/s] 77%|███████▋  | 307933/400000 [00:41<00:11, 7835.55it/s] 77%|███████▋  | 308718/400000 [00:41<00:12, 7481.72it/s] 77%|███████▋  | 309471/400000 [00:41<00:12, 7339.81it/s] 78%|███████▊  | 310209/400000 [00:41<00:12, 7302.77it/s] 78%|███████▊  | 310942/400000 [00:41<00:13, 6839.42it/s] 78%|███████▊  | 311669/400000 [00:41<00:12, 6961.63it/s] 78%|███████▊  | 312399/400000 [00:41<00:12, 7056.99it/s] 78%|███████▊  | 313110/400000 [00:42<00:12, 6926.50it/s] 78%|███████▊  | 313825/400000 [00:42<00:12, 6990.72it/s] 79%|███████▊  | 314531/400000 [00:42<00:12, 7009.54it/s] 79%|███████▉  | 315259/400000 [00:42<00:11, 7087.68it/s] 79%|███████▉  | 316001/400000 [00:42<00:11, 7183.34it/s] 79%|███████▉  | 316728/400000 [00:42<00:11, 7206.87it/s] 79%|███████▉  | 317484/400000 [00:42<00:11, 7308.01it/s] 80%|███████▉  | 318216/400000 [00:42<00:11, 7024.37it/s] 80%|███████▉  | 318993/400000 [00:42<00:11, 7230.71it/s] 80%|███████▉  | 319782/400000 [00:43<00:10, 7415.37it/s] 80%|████████  | 320552/400000 [00:43<00:10, 7496.72it/s] 80%|████████  | 321305/400000 [00:43<00:10, 7499.53it/s] 81%|████████  | 322057/400000 [00:43<00:10, 7386.12it/s] 81%|████████  | 322798/400000 [00:43<00:10, 7331.93it/s] 81%|████████  | 323533/400000 [00:43<00:10, 7282.87it/s] 81%|████████  | 324290/400000 [00:43<00:10, 7366.26it/s] 81%|████████▏ | 325042/400000 [00:43<00:10, 7411.46it/s] 81%|████████▏ | 325784/400000 [00:43<00:10, 7375.51it/s] 82%|████████▏ | 326523/400000 [00:43<00:10, 7327.23it/s] 82%|████████▏ | 327261/400000 [00:44<00:09, 7342.12it/s] 82%|████████▏ | 328000/400000 [00:44<00:09, 7355.74it/s] 82%|████████▏ | 328736/400000 [00:44<00:09, 7338.82it/s] 82%|████████▏ | 329471/400000 [00:44<00:09, 7252.22it/s] 83%|████████▎ | 330217/400000 [00:44<00:09, 7312.47it/s] 83%|████████▎ | 330960/400000 [00:44<00:09, 7344.76it/s] 83%|████████▎ | 331726/400000 [00:44<00:09, 7434.86it/s] 83%|████████▎ | 332502/400000 [00:44<00:08, 7528.73it/s] 83%|████████▎ | 333256/400000 [00:44<00:09, 7355.63it/s] 83%|████████▎ | 333993/400000 [00:44<00:09, 7230.23it/s] 84%|████████▎ | 334718/400000 [00:45<00:09, 7170.13it/s] 84%|████████▍ | 335449/400000 [00:45<00:08, 7211.22it/s] 84%|████████▍ | 336171/400000 [00:45<00:08, 7209.30it/s] 84%|████████▍ | 336893/400000 [00:45<00:08, 7142.20it/s] 84%|████████▍ | 337619/400000 [00:45<00:08, 7175.97it/s] 85%|████████▍ | 338349/400000 [00:45<00:08, 7210.35it/s] 85%|████████▍ | 339082/400000 [00:45<00:08, 7244.15it/s] 85%|████████▍ | 339817/400000 [00:45<00:08, 7275.38it/s] 85%|████████▌ | 340545/400000 [00:45<00:08, 7235.76it/s] 85%|████████▌ | 341291/400000 [00:45<00:08, 7300.84it/s] 86%|████████▌ | 342053/400000 [00:46<00:07, 7391.72it/s] 86%|████████▌ | 342793/400000 [00:46<00:07, 7217.47it/s] 86%|████████▌ | 343532/400000 [00:46<00:07, 7267.15it/s] 86%|████████▌ | 344263/400000 [00:46<00:07, 7279.08it/s] 86%|████████▋ | 345014/400000 [00:46<00:07, 7345.47it/s] 86%|████████▋ | 345765/400000 [00:46<00:07, 7394.06it/s] 87%|████████▋ | 346512/400000 [00:46<00:07, 7414.97it/s] 87%|████████▋ | 347254/400000 [00:46<00:07, 7319.10it/s] 87%|████████▋ | 347987/400000 [00:46<00:07, 7155.30it/s] 87%|████████▋ | 348704/400000 [00:46<00:07, 7030.54it/s] 87%|████████▋ | 349449/400000 [00:47<00:07, 7150.77it/s] 88%|████████▊ | 350215/400000 [00:47<00:06, 7293.93it/s] 88%|████████▊ | 350973/400000 [00:47<00:06, 7376.34it/s] 88%|████████▊ | 351713/400000 [00:47<00:06, 7338.31it/s] 88%|████████▊ | 352473/400000 [00:47<00:06, 7412.90it/s] 88%|████████▊ | 353229/400000 [00:47<00:06, 7456.33it/s] 88%|████████▊ | 353976/400000 [00:47<00:06, 7389.04it/s] 89%|████████▊ | 354717/400000 [00:47<00:06, 7393.76it/s] 89%|████████▉ | 355457/400000 [00:47<00:06, 7378.31it/s] 89%|████████▉ | 356196/400000 [00:47<00:05, 7363.33it/s] 89%|████████▉ | 356987/400000 [00:48<00:05, 7517.49it/s] 89%|████████▉ | 357749/400000 [00:48<00:05, 7547.03it/s] 90%|████████▉ | 358505/400000 [00:48<00:05, 7481.96it/s] 90%|████████▉ | 359254/400000 [00:48<00:05, 7357.59it/s] 90%|████████▉ | 359991/400000 [00:48<00:05, 7261.84it/s] 90%|█████████ | 360725/400000 [00:48<00:05, 7282.86it/s] 90%|█████████ | 361504/400000 [00:48<00:05, 7426.42it/s] 91%|█████████ | 362283/400000 [00:48<00:05, 7530.51it/s] 91%|█████████ | 363038/400000 [00:48<00:04, 7429.03it/s] 91%|█████████ | 363786/400000 [00:49<00:04, 7443.81it/s] 91%|█████████ | 364571/400000 [00:49<00:04, 7559.33it/s] 91%|█████████▏| 365328/400000 [00:49<00:04, 7558.08it/s] 92%|█████████▏| 366114/400000 [00:49<00:04, 7645.00it/s] 92%|█████████▏| 366890/400000 [00:49<00:04, 7675.84it/s] 92%|█████████▏| 367659/400000 [00:49<00:04, 7619.13it/s] 92%|█████████▏| 368422/400000 [00:49<00:04, 7591.14it/s] 92%|█████████▏| 369182/400000 [00:49<00:04, 7506.42it/s] 92%|█████████▏| 369934/400000 [00:49<00:04, 7400.19it/s] 93%|█████████▎| 370675/400000 [00:49<00:04, 7221.35it/s] 93%|█████████▎| 371399/400000 [00:50<00:03, 7164.43it/s] 93%|█████████▎| 372117/400000 [00:50<00:03, 7063.62it/s] 93%|█████████▎| 372868/400000 [00:50<00:03, 7189.34it/s] 93%|█████████▎| 373597/400000 [00:50<00:03, 7217.99it/s] 94%|█████████▎| 374320/400000 [00:50<00:03, 7204.36it/s] 94%|█████████▍| 375070/400000 [00:50<00:03, 7289.01it/s] 94%|█████████▍| 375820/400000 [00:50<00:03, 7350.32it/s] 94%|█████████▍| 376598/400000 [00:50<00:03, 7473.41it/s] 94%|█████████▍| 377354/400000 [00:50<00:03, 7497.43it/s] 95%|█████████▍| 378105/400000 [00:50<00:02, 7453.72it/s] 95%|█████████▍| 378895/400000 [00:51<00:02, 7582.03it/s] 95%|█████████▍| 379680/400000 [00:51<00:02, 7660.19it/s] 95%|█████████▌| 380457/400000 [00:51<00:02, 7690.43it/s] 95%|█████████▌| 381235/400000 [00:51<00:02, 7714.30it/s] 96%|█████████▌| 382007/400000 [00:51<00:02, 7579.74it/s] 96%|█████████▌| 382776/400000 [00:51<00:02, 7610.23it/s] 96%|█████████▌| 383538/400000 [00:51<00:02, 7587.89it/s] 96%|█████████▌| 384298/400000 [00:51<00:02, 7534.50it/s] 96%|█████████▋| 385052/400000 [00:51<00:02, 7428.22it/s] 96%|█████████▋| 385796/400000 [00:51<00:01, 7385.46it/s] 97%|█████████▋| 386559/400000 [00:52<00:01, 7455.02it/s] 97%|█████████▋| 387316/400000 [00:52<00:01, 7488.82it/s] 97%|█████████▋| 388066/400000 [00:52<00:01, 7485.69it/s] 97%|█████████▋| 388815/400000 [00:52<00:01, 7485.33it/s] 97%|█████████▋| 389572/400000 [00:52<00:01, 7510.17it/s] 98%|█████████▊| 390324/400000 [00:52<00:01, 7507.18it/s] 98%|█████████▊| 391088/400000 [00:52<00:01, 7546.03it/s] 98%|█████████▊| 391856/400000 [00:52<00:01, 7585.45it/s] 98%|█████████▊| 392615/400000 [00:52<00:00, 7482.34it/s] 98%|█████████▊| 393364/400000 [00:52<00:00, 7304.19it/s] 99%|█████████▊| 394116/400000 [00:53<00:00, 7367.47it/s] 99%|█████████▊| 394854/400000 [00:53<00:00, 7359.58it/s] 99%|█████████▉| 395613/400000 [00:53<00:00, 7424.75it/s] 99%|█████████▉| 396357/400000 [00:53<00:00, 7414.63it/s] 99%|█████████▉| 397099/400000 [00:53<00:00, 7360.02it/s] 99%|█████████▉| 397859/400000 [00:53<00:00, 7427.85it/s]100%|█████████▉| 398621/400000 [00:53<00:00, 7483.03it/s]100%|█████████▉| 399392/400000 [00:53<00:00, 7549.08it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7427.34it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f13ab1c9b70> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011287162829195821 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.01097116621840359 	 Accuracy: 71

  model saves at 71% accuracy 

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

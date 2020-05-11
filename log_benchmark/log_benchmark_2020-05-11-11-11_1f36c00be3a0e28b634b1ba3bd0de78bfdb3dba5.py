
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f00b529ffd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 11:11:30.136716
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 11:11:30.139388
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 11:11:30.142152
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 11:11:30.144883
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f00c12b7470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355871.8750
Epoch 2/10

1/1 [==============================] - 0s 82ms/step - loss: 289289.5625
Epoch 3/10

1/1 [==============================] - 0s 84ms/step - loss: 209917.9375
Epoch 4/10

1/1 [==============================] - 0s 86ms/step - loss: 136353.2344
Epoch 5/10

1/1 [==============================] - 0s 83ms/step - loss: 83457.0469
Epoch 6/10

1/1 [==============================] - 0s 84ms/step - loss: 51029.8281
Epoch 7/10

1/1 [==============================] - 0s 87ms/step - loss: 31735.7578
Epoch 8/10

1/1 [==============================] - 0s 84ms/step - loss: 20332.1328
Epoch 9/10

1/1 [==============================] - 0s 85ms/step - loss: 13720.1943
Epoch 10/10

1/1 [==============================] - 0s 84ms/step - loss: 9806.6133

  #### Inference Need return ypred, ytrue ######################### 
[[-8.38539600e-02  8.23955345e+00  6.55373669e+00  6.89686441e+00
   5.84840202e+00  7.67427492e+00  6.12823486e+00  6.31864500e+00
   6.08683777e+00  5.91544151e+00  5.91917133e+00  6.68274546e+00
   6.18699551e+00  4.51738167e+00  6.28273344e+00  5.87257338e+00
   6.43967295e+00  5.97473526e+00  6.41180420e+00  5.60163879e+00
   7.53960848e+00  7.75560141e+00  7.79055548e+00  7.10042477e+00
   5.19200802e+00  5.05242205e+00  6.85064888e+00  6.45256042e+00
   5.22617388e+00  6.00181818e+00  5.64234591e+00  7.75948334e+00
   8.08426857e+00  7.54006481e+00  5.76448822e+00  5.25796080e+00
   5.45734644e+00  5.72225094e+00  7.54896688e+00  6.95594501e+00
   4.79610205e+00  7.55810738e+00  6.25398588e+00  5.33776426e+00
   5.25491714e+00  6.83048487e+00  6.12029791e+00  7.15249443e+00
   6.00621080e+00  5.66162157e+00  5.68677855e+00  6.67312050e+00
   6.14793539e+00  6.96733284e+00  5.73320723e+00  5.99618292e+00
   6.10960484e+00  7.35222864e+00  7.63643217e+00  7.26941729e+00
  -5.74000895e-01 -6.20183051e-02 -6.83217883e-01  3.30060184e-01
   7.36491203e-01  2.21289134e+00  2.29471922e-01  1.23386502e+00
   1.30406559e+00  8.69041920e-01  1.26731753e-01 -1.21121955e+00
  -8.51231933e-01 -1.43231332e-01 -1.07355928e+00  4.41289008e-01
   8.03325772e-02 -9.71189916e-01  1.01823092e+00 -6.70545697e-01
   1.06118426e-01 -2.77349025e-01  1.69026881e-01 -9.82771337e-01
   1.04339027e+00  6.63359761e-01 -8.11671257e-01  5.21419883e-01
  -2.66252100e-01 -2.45530665e-01 -7.29223251e-01  2.98317373e-01
  -1.23161745e+00  1.27796650e-01 -6.77616894e-01 -2.13890409e+00
  -1.15465403e+00 -1.22817338e-01 -1.33562922e+00  2.24915698e-01
  -1.51206398e+00 -1.63204598e+00 -2.86710382e-01 -8.81065190e-01
  -1.66075528e+00  4.46664453e-01  4.23071086e-01  3.65098685e-01
  -1.13546097e+00  8.91885221e-01  1.09841257e-01  1.58543110e+00
   6.14461005e-01 -2.35631913e-01 -1.25976861e+00  1.17335331e+00
  -7.58969307e-01  1.04530811e+00  2.60689914e-01 -6.66043162e-03
   4.50087488e-01 -2.71919072e-01 -6.65923774e-01 -1.00830960e+00
   1.40398359e+00  2.58067161e-01 -3.20914596e-01  4.99439895e-01
   4.12374049e-01 -2.91235507e-01 -7.17528537e-02  7.75948167e-01
   1.20771900e-01  7.66135931e-01  2.77606457e-01 -1.18416071e+00
   1.03395975e+00  2.48878002e-02 -6.19105399e-02  5.20609617e-02
   1.67066741e+00  9.18055594e-01 -1.35720551e+00 -7.25202203e-01
   7.77143002e-01  1.35469818e+00 -1.16319478e-01  6.88744009e-01
   4.86733019e-02 -1.23070538e+00  7.67997146e-01  1.84279287e+00
  -1.36281538e+00  4.57477391e-01 -3.87599170e-02 -1.57592702e+00
   1.00479579e+00  4.09073353e-01 -6.95266068e-01 -3.34464908e-01
  -1.21312606e+00  1.15804076e-02 -4.15345222e-01 -6.66488886e-01
   9.76962447e-02  5.78106761e-01  6.70813918e-01 -6.68384671e-01
   2.31922731e-01 -5.52437425e-01  1.82500750e-01  8.91084552e-01
   5.30232668e-01 -1.87742436e+00  4.70543802e-02 -8.81592333e-02
   1.90010333e+00 -2.64809012e-01  8.29863310e-01  2.41823703e-01
   6.18200898e-02  6.70532465e+00  8.08181381e+00  5.64810991e+00
   7.71350050e+00  6.32745838e+00  6.65211201e+00  7.03836393e+00
   6.10721588e+00  7.04543829e+00  6.21716309e+00  5.86000538e+00
   6.63872290e+00  7.77419424e+00  6.87490273e+00  7.15939951e+00
   6.36840010e+00  7.15402699e+00  6.72651148e+00  6.82755184e+00
   7.10658312e+00  8.68860531e+00  7.10363913e+00  6.30669212e+00
   6.62087297e+00  8.28966999e+00  6.47444677e+00  6.94622803e+00
   7.75547552e+00  6.91621733e+00  7.86545944e+00  6.18718004e+00
   6.27883196e+00  7.42615128e+00  7.82261562e+00  7.25422096e+00
   7.83346844e+00  5.75480175e+00  5.72404194e+00  8.36427784e+00
   6.73894739e+00  7.62290382e+00  8.11081600e+00  7.70565796e+00
   7.33619452e+00  7.00101852e+00  8.37462330e+00  7.67924070e+00
   6.92524529e+00  8.10952568e+00  6.23203564e+00  7.28025770e+00
   6.20363045e+00  6.98392105e+00  6.80916500e+00  7.17570305e+00
   8.73382473e+00  7.09016228e+00  5.68462801e+00  7.57790947e+00
   1.41850972e+00  5.09275377e-01  1.80347085e+00  1.89836240e+00
   5.67395031e-01  1.97401881e+00  2.08037519e+00  2.46906328e+00
   1.23334730e+00  1.20028114e+00  1.82693517e+00  9.00461376e-01
   3.48390222e-01  1.60917759e+00  1.74160182e+00  8.24085832e-01
   1.31192994e+00  1.65855300e+00  4.33002591e-01  2.38869429e+00
   4.08860326e-01  1.34741855e+00  1.49896741e-01  3.64898324e-01
   6.83759451e-01  1.52519143e+00  2.66198254e+00  2.44656754e+00
   3.45942438e-01  2.48054326e-01  1.33291733e+00  3.11270857e+00
   2.43116856e-01  1.85491526e+00  1.25722170e+00  2.80093193e-01
   7.40514636e-01  1.67883158e+00  2.79177856e+00  1.86084664e+00
   3.07347918e+00  2.56885886e-01  9.17726934e-01  9.03832138e-01
   7.13987410e-01  3.76874864e-01  1.84988999e+00  2.63309097e+00
   8.92772913e-01  1.77080464e+00  2.42731214e+00  1.49797165e+00
   5.58770299e-01  5.71252346e-01  2.77825356e-01  1.55863225e+00
   8.44814062e-01  3.02915382e+00  1.79771090e+00  4.50916588e-01
   1.21761882e+00  7.53865123e-01  8.80131602e-01  3.93126965e-01
   1.40928781e+00  1.80262601e+00  2.43294907e+00  1.35658455e+00
   1.00203216e+00  1.64489877e+00  1.52007818e+00  4.18660760e-01
   3.22333217e-01  2.34270000e+00  1.34643269e+00  5.84692836e-01
   1.21370935e+00  1.08474207e+00  1.12537217e+00  4.57192361e-01
   6.15585387e-01  6.79666221e-01  1.14253724e+00  4.38537121e-01
   2.27882814e+00  2.33803892e+00  2.60363460e+00  9.97551799e-01
   5.96089900e-01  2.04414296e+00  2.03396440e+00  2.19854403e+00
   8.05840075e-01  1.39629233e+00  3.52236390e-01  5.42870343e-01
   2.39441633e+00  1.43073237e+00  9.64274585e-01  7.34807372e-01
   9.71903265e-01  4.48568344e-01  6.22747004e-01  2.14479160e+00
   5.94979823e-01  1.84300363e-01  7.70686865e-01  1.76573431e+00
   3.39913487e-01  4.24604058e-01  8.07333231e-01  1.96191335e+00
   1.08766246e+00  2.61773729e+00  8.13418627e-01  1.68478358e+00
   1.88749349e+00  1.08628941e+00  1.43153453e+00  2.33411694e+00
   2.84892035e+00 -8.69342327e+00 -7.83410120e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 11:11:37.225588
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.2919
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 11:11:37.229425
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9293.62
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 11:11:37.232187
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.6135
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 11:11:37.234730
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -831.31
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139640676601304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139638146985488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139638146564168
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139638146564672
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139638146565176
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139638146565680

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f00bd138f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.661391
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.628646
grad_step = 000002, loss = 0.602992
grad_step = 000003, loss = 0.574509
grad_step = 000004, loss = 0.543675
grad_step = 000005, loss = 0.514646
grad_step = 000006, loss = 0.490109
grad_step = 000007, loss = 0.472982
grad_step = 000008, loss = 0.458856
grad_step = 000009, loss = 0.437764
grad_step = 000010, loss = 0.417526
grad_step = 000011, loss = 0.402951
grad_step = 000012, loss = 0.390778
grad_step = 000013, loss = 0.378150
grad_step = 000014, loss = 0.364975
grad_step = 000015, loss = 0.351417
grad_step = 000016, loss = 0.337603
grad_step = 000017, loss = 0.324822
grad_step = 000018, loss = 0.312651
grad_step = 000019, loss = 0.300927
grad_step = 000020, loss = 0.289446
grad_step = 000021, loss = 0.277334
grad_step = 000022, loss = 0.265252
grad_step = 000023, loss = 0.253479
grad_step = 000024, loss = 0.241799
grad_step = 000025, loss = 0.230406
grad_step = 000026, loss = 0.219867
grad_step = 000027, loss = 0.209996
grad_step = 000028, loss = 0.199899
grad_step = 000029, loss = 0.189717
grad_step = 000030, loss = 0.180342
grad_step = 000031, loss = 0.171759
grad_step = 000032, loss = 0.163198
grad_step = 000033, loss = 0.154506
grad_step = 000034, loss = 0.145868
grad_step = 000035, loss = 0.137689
grad_step = 000036, loss = 0.130111
grad_step = 000037, loss = 0.122558
grad_step = 000038, loss = 0.115060
grad_step = 000039, loss = 0.108011
grad_step = 000040, loss = 0.101148
grad_step = 000041, loss = 0.094464
grad_step = 000042, loss = 0.088138
grad_step = 000043, loss = 0.082175
grad_step = 000044, loss = 0.076409
grad_step = 000045, loss = 0.070766
grad_step = 000046, loss = 0.065546
grad_step = 000047, loss = 0.060581
grad_step = 000048, loss = 0.055860
grad_step = 000049, loss = 0.051360
grad_step = 000050, loss = 0.047134
grad_step = 000051, loss = 0.043128
grad_step = 000052, loss = 0.039403
grad_step = 000053, loss = 0.035875
grad_step = 000054, loss = 0.032622
grad_step = 000055, loss = 0.029589
grad_step = 000056, loss = 0.026788
grad_step = 000057, loss = 0.024232
grad_step = 000058, loss = 0.021851
grad_step = 000059, loss = 0.019639
grad_step = 000060, loss = 0.017629
grad_step = 000061, loss = 0.015791
grad_step = 000062, loss = 0.014139
grad_step = 000063, loss = 0.012639
grad_step = 000064, loss = 0.011279
grad_step = 000065, loss = 0.010072
grad_step = 000066, loss = 0.009004
grad_step = 000067, loss = 0.008041
grad_step = 000068, loss = 0.007195
grad_step = 000069, loss = 0.006457
grad_step = 000070, loss = 0.005793
grad_step = 000071, loss = 0.005237
grad_step = 000072, loss = 0.004742
grad_step = 000073, loss = 0.004316
grad_step = 000074, loss = 0.003951
grad_step = 000075, loss = 0.003643
grad_step = 000076, loss = 0.003380
grad_step = 000077, loss = 0.003157
grad_step = 000078, loss = 0.002969
grad_step = 000079, loss = 0.002818
grad_step = 000080, loss = 0.002691
grad_step = 000081, loss = 0.002584
grad_step = 000082, loss = 0.002494
grad_step = 000083, loss = 0.002424
grad_step = 000084, loss = 0.002369
grad_step = 000085, loss = 0.002323
grad_step = 000086, loss = 0.002290
grad_step = 000087, loss = 0.002258
grad_step = 000088, loss = 0.002233
grad_step = 000089, loss = 0.002213
grad_step = 000090, loss = 0.002196
grad_step = 000091, loss = 0.002184
grad_step = 000092, loss = 0.002172
grad_step = 000093, loss = 0.002162
grad_step = 000094, loss = 0.002152
grad_step = 000095, loss = 0.002144
grad_step = 000096, loss = 0.002138
grad_step = 000097, loss = 0.002139
grad_step = 000098, loss = 0.002153
grad_step = 000099, loss = 0.002192
grad_step = 000100, loss = 0.002240
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002274
grad_step = 000102, loss = 0.002201
grad_step = 000103, loss = 0.002079
grad_step = 000104, loss = 0.002026
grad_step = 000105, loss = 0.002069
grad_step = 000106, loss = 0.002113
grad_step = 000107, loss = 0.002068
grad_step = 000108, loss = 0.001985
grad_step = 000109, loss = 0.001955
grad_step = 000110, loss = 0.001985
grad_step = 000111, loss = 0.002003
grad_step = 000112, loss = 0.001961
grad_step = 000113, loss = 0.001907
grad_step = 000114, loss = 0.001895
grad_step = 000115, loss = 0.001915
grad_step = 000116, loss = 0.001922
grad_step = 000117, loss = 0.001893
grad_step = 000118, loss = 0.001857
grad_step = 000119, loss = 0.001843
grad_step = 000120, loss = 0.001852
grad_step = 000121, loss = 0.001863
grad_step = 000122, loss = 0.001855
grad_step = 000123, loss = 0.001835
grad_step = 000124, loss = 0.001812
grad_step = 000125, loss = 0.001801
grad_step = 000126, loss = 0.001801
grad_step = 000127, loss = 0.001806
grad_step = 000128, loss = 0.001812
grad_step = 000129, loss = 0.001813
grad_step = 000130, loss = 0.001811
grad_step = 000131, loss = 0.001804
grad_step = 000132, loss = 0.001796
grad_step = 000133, loss = 0.001786
grad_step = 000134, loss = 0.001777
grad_step = 000135, loss = 0.001769
grad_step = 000136, loss = 0.001764
grad_step = 000137, loss = 0.001761
grad_step = 000138, loss = 0.001763
grad_step = 000139, loss = 0.001771
grad_step = 000140, loss = 0.001794
grad_step = 000141, loss = 0.001832
grad_step = 000142, loss = 0.001909
grad_step = 000143, loss = 0.001970
grad_step = 000144, loss = 0.002015
grad_step = 000145, loss = 0.001910
grad_step = 000146, loss = 0.001769
grad_step = 000147, loss = 0.001697
grad_step = 000148, loss = 0.001748
grad_step = 000149, loss = 0.001842
grad_step = 000150, loss = 0.001863
grad_step = 000151, loss = 0.001810
grad_step = 000152, loss = 0.001710
grad_step = 000153, loss = 0.001672
grad_step = 000154, loss = 0.001710
grad_step = 000155, loss = 0.001764
grad_step = 000156, loss = 0.001783
grad_step = 000157, loss = 0.001736
grad_step = 000158, loss = 0.001678
grad_step = 000159, loss = 0.001651
grad_step = 000160, loss = 0.001667
grad_step = 000161, loss = 0.001702
grad_step = 000162, loss = 0.001717
grad_step = 000163, loss = 0.001706
grad_step = 000164, loss = 0.001670
grad_step = 000165, loss = 0.001640
grad_step = 000166, loss = 0.001628
grad_step = 000167, loss = 0.001635
grad_step = 000168, loss = 0.001652
grad_step = 000169, loss = 0.001667
grad_step = 000170, loss = 0.001677
grad_step = 000171, loss = 0.001671
grad_step = 000172, loss = 0.001660
grad_step = 000173, loss = 0.001641
grad_step = 000174, loss = 0.001623
grad_step = 000175, loss = 0.001607
grad_step = 000176, loss = 0.001597
grad_step = 000177, loss = 0.001592
grad_step = 000178, loss = 0.001590
grad_step = 000179, loss = 0.001591
grad_step = 000180, loss = 0.001595
grad_step = 000181, loss = 0.001605
grad_step = 000182, loss = 0.001622
grad_step = 000183, loss = 0.001660
grad_step = 000184, loss = 0.001723
grad_step = 000185, loss = 0.001849
grad_step = 000186, loss = 0.001970
grad_step = 000187, loss = 0.002100
grad_step = 000188, loss = 0.001961
grad_step = 000189, loss = 0.001715
grad_step = 000190, loss = 0.001561
grad_step = 000191, loss = 0.001649
grad_step = 000192, loss = 0.001809
grad_step = 000193, loss = 0.001778
grad_step = 000194, loss = 0.001624
grad_step = 000195, loss = 0.001547
grad_step = 000196, loss = 0.001626
grad_step = 000197, loss = 0.001714
grad_step = 000198, loss = 0.001664
grad_step = 000199, loss = 0.001564
grad_step = 000200, loss = 0.001531
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001584
grad_step = 000202, loss = 0.001642
grad_step = 000203, loss = 0.001626
grad_step = 000204, loss = 0.001567
grad_step = 000205, loss = 0.001521
grad_step = 000206, loss = 0.001526
grad_step = 000207, loss = 0.001560
grad_step = 000208, loss = 0.001576
grad_step = 000209, loss = 0.001563
grad_step = 000210, loss = 0.001529
grad_step = 000211, loss = 0.001506
grad_step = 000212, loss = 0.001506
grad_step = 000213, loss = 0.001520
grad_step = 000214, loss = 0.001530
grad_step = 000215, loss = 0.001524
grad_step = 000216, loss = 0.001509
grad_step = 000217, loss = 0.001494
grad_step = 000218, loss = 0.001486
grad_step = 000219, loss = 0.001487
grad_step = 000220, loss = 0.001491
grad_step = 000221, loss = 0.001494
grad_step = 000222, loss = 0.001492
grad_step = 000223, loss = 0.001486
grad_step = 000224, loss = 0.001479
grad_step = 000225, loss = 0.001473
grad_step = 000226, loss = 0.001468
grad_step = 000227, loss = 0.001466
grad_step = 000228, loss = 0.001466
grad_step = 000229, loss = 0.001468
grad_step = 000230, loss = 0.001471
grad_step = 000231, loss = 0.001475
grad_step = 000232, loss = 0.001482
grad_step = 000233, loss = 0.001490
grad_step = 000234, loss = 0.001504
grad_step = 000235, loss = 0.001520
grad_step = 000236, loss = 0.001543
grad_step = 000237, loss = 0.001561
grad_step = 000238, loss = 0.001581
grad_step = 000239, loss = 0.001586
grad_step = 000240, loss = 0.001596
grad_step = 000241, loss = 0.001613
grad_step = 000242, loss = 0.001677
grad_step = 000243, loss = 0.001799
grad_step = 000244, loss = 0.001909
grad_step = 000245, loss = 0.001960
grad_step = 000246, loss = 0.001774
grad_step = 000247, loss = 0.001605
grad_step = 000248, loss = 0.001561
grad_step = 000249, loss = 0.001622
grad_step = 000250, loss = 0.001611
grad_step = 000251, loss = 0.001512
grad_step = 000252, loss = 0.001486
grad_step = 000253, loss = 0.001566
grad_step = 000254, loss = 0.001605
grad_step = 000255, loss = 0.001504
grad_step = 000256, loss = 0.001402
grad_step = 000257, loss = 0.001432
grad_step = 000258, loss = 0.001518
grad_step = 000259, loss = 0.001524
grad_step = 000260, loss = 0.001443
grad_step = 000261, loss = 0.001395
grad_step = 000262, loss = 0.001421
grad_step = 000263, loss = 0.001457
grad_step = 000264, loss = 0.001444
grad_step = 000265, loss = 0.001405
grad_step = 000266, loss = 0.001393
grad_step = 000267, loss = 0.001413
grad_step = 000268, loss = 0.001431
grad_step = 000269, loss = 0.001415
grad_step = 000270, loss = 0.001383
grad_step = 000271, loss = 0.001363
grad_step = 000272, loss = 0.001371
grad_step = 000273, loss = 0.001390
grad_step = 000274, loss = 0.001399
grad_step = 000275, loss = 0.001390
grad_step = 000276, loss = 0.001371
grad_step = 000277, loss = 0.001360
grad_step = 000278, loss = 0.001361
grad_step = 000279, loss = 0.001373
grad_step = 000280, loss = 0.001384
grad_step = 000281, loss = 0.001390
grad_step = 000282, loss = 0.001393
grad_step = 000283, loss = 0.001399
grad_step = 000284, loss = 0.001418
grad_step = 000285, loss = 0.001457
grad_step = 000286, loss = 0.001516
grad_step = 000287, loss = 0.001596
grad_step = 000288, loss = 0.001665
grad_step = 000289, loss = 0.001705
grad_step = 000290, loss = 0.001646
grad_step = 000291, loss = 0.001508
grad_step = 000292, loss = 0.001372
grad_step = 000293, loss = 0.001335
grad_step = 000294, loss = 0.001396
grad_step = 000295, loss = 0.001471
grad_step = 000296, loss = 0.001486
grad_step = 000297, loss = 0.001420
grad_step = 000298, loss = 0.001342
grad_step = 000299, loss = 0.001316
grad_step = 000300, loss = 0.001349
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001400
grad_step = 000302, loss = 0.001420
grad_step = 000303, loss = 0.001399
grad_step = 000304, loss = 0.001350
grad_step = 000305, loss = 0.001318
grad_step = 000306, loss = 0.001324
grad_step = 000307, loss = 0.001362
grad_step = 000308, loss = 0.001415
grad_step = 000309, loss = 0.001469
grad_step = 000310, loss = 0.001542
grad_step = 000311, loss = 0.001631
grad_step = 000312, loss = 0.001766
grad_step = 000313, loss = 0.001839
grad_step = 000314, loss = 0.001728
grad_step = 000315, loss = 0.001504
grad_step = 000316, loss = 0.001360
grad_step = 000317, loss = 0.001442
grad_step = 000318, loss = 0.001541
grad_step = 000319, loss = 0.001463
grad_step = 000320, loss = 0.001329
grad_step = 000321, loss = 0.001350
grad_step = 000322, loss = 0.001464
grad_step = 000323, loss = 0.001460
grad_step = 000324, loss = 0.001347
grad_step = 000325, loss = 0.001285
grad_step = 000326, loss = 0.001341
grad_step = 000327, loss = 0.001403
grad_step = 000328, loss = 0.001375
grad_step = 000329, loss = 0.001314
grad_step = 000330, loss = 0.001301
grad_step = 000331, loss = 0.001333
grad_step = 000332, loss = 0.001343
grad_step = 000333, loss = 0.001313
grad_step = 000334, loss = 0.001281
grad_step = 000335, loss = 0.001286
grad_step = 000336, loss = 0.001317
grad_step = 000337, loss = 0.001326
grad_step = 000338, loss = 0.001307
grad_step = 000339, loss = 0.001276
grad_step = 000340, loss = 0.001265
grad_step = 000341, loss = 0.001276
grad_step = 000342, loss = 0.001289
grad_step = 000343, loss = 0.001286
grad_step = 000344, loss = 0.001270
grad_step = 000345, loss = 0.001258
grad_step = 000346, loss = 0.001259
grad_step = 000347, loss = 0.001269
grad_step = 000348, loss = 0.001276
grad_step = 000349, loss = 0.001274
grad_step = 000350, loss = 0.001267
grad_step = 000351, loss = 0.001260
grad_step = 000352, loss = 0.001259
grad_step = 000353, loss = 0.001263
grad_step = 000354, loss = 0.001269
grad_step = 000355, loss = 0.001275
grad_step = 000356, loss = 0.001278
grad_step = 000357, loss = 0.001283
grad_step = 000358, loss = 0.001290
grad_step = 000359, loss = 0.001305
grad_step = 000360, loss = 0.001332
grad_step = 000361, loss = 0.001374
grad_step = 000362, loss = 0.001430
grad_step = 000363, loss = 0.001488
grad_step = 000364, loss = 0.001533
grad_step = 000365, loss = 0.001533
grad_step = 000366, loss = 0.001478
grad_step = 000367, loss = 0.001376
grad_step = 000368, loss = 0.001280
grad_step = 000369, loss = 0.001229
grad_step = 000370, loss = 0.001238
grad_step = 000371, loss = 0.001284
grad_step = 000372, loss = 0.001330
grad_step = 000373, loss = 0.001349
grad_step = 000374, loss = 0.001327
grad_step = 000375, loss = 0.001284
grad_step = 000376, loss = 0.001242
grad_step = 000377, loss = 0.001224
grad_step = 000378, loss = 0.001229
grad_step = 000379, loss = 0.001250
grad_step = 000380, loss = 0.001270
grad_step = 000381, loss = 0.001280
grad_step = 000382, loss = 0.001275
grad_step = 000383, loss = 0.001258
grad_step = 000384, loss = 0.001238
grad_step = 000385, loss = 0.001221
grad_step = 000386, loss = 0.001215
grad_step = 000387, loss = 0.001218
grad_step = 000388, loss = 0.001231
grad_step = 000389, loss = 0.001250
grad_step = 000390, loss = 0.001277
grad_step = 000391, loss = 0.001307
grad_step = 000392, loss = 0.001357
grad_step = 000393, loss = 0.001408
grad_step = 000394, loss = 0.001490
grad_step = 000395, loss = 0.001551
grad_step = 000396, loss = 0.001584
grad_step = 000397, loss = 0.001521
grad_step = 000398, loss = 0.001390
grad_step = 000399, loss = 0.001285
grad_step = 000400, loss = 0.001286
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001361
grad_step = 000402, loss = 0.001389
grad_step = 000403, loss = 0.001337
grad_step = 000404, loss = 0.001234
grad_step = 000405, loss = 0.001189
grad_step = 000406, loss = 0.001230
grad_step = 000407, loss = 0.001291
grad_step = 000408, loss = 0.001309
grad_step = 000409, loss = 0.001260
grad_step = 000410, loss = 0.001206
grad_step = 000411, loss = 0.001190
grad_step = 000412, loss = 0.001213
grad_step = 000413, loss = 0.001240
grad_step = 000414, loss = 0.001239
grad_step = 000415, loss = 0.001215
grad_step = 000416, loss = 0.001187
grad_step = 000417, loss = 0.001176
grad_step = 000418, loss = 0.001186
grad_step = 000419, loss = 0.001205
grad_step = 000420, loss = 0.001219
grad_step = 000421, loss = 0.001216
grad_step = 000422, loss = 0.001202
grad_step = 000423, loss = 0.001185
grad_step = 000424, loss = 0.001175
grad_step = 000425, loss = 0.001176
grad_step = 000426, loss = 0.001184
grad_step = 000427, loss = 0.001193
grad_step = 000428, loss = 0.001197
grad_step = 000429, loss = 0.001197
grad_step = 000430, loss = 0.001195
grad_step = 000431, loss = 0.001197
grad_step = 000432, loss = 0.001209
grad_step = 000433, loss = 0.001233
grad_step = 000434, loss = 0.001273
grad_step = 000435, loss = 0.001326
grad_step = 000436, loss = 0.001386
grad_step = 000437, loss = 0.001445
grad_step = 000438, loss = 0.001477
grad_step = 000439, loss = 0.001464
grad_step = 000440, loss = 0.001396
grad_step = 000441, loss = 0.001293
grad_step = 000442, loss = 0.001203
grad_step = 000443, loss = 0.001162
grad_step = 000444, loss = 0.001176
grad_step = 000445, loss = 0.001220
grad_step = 000446, loss = 0.001253
grad_step = 000447, loss = 0.001250
grad_step = 000448, loss = 0.001214
grad_step = 000449, loss = 0.001172
grad_step = 000450, loss = 0.001151
grad_step = 000451, loss = 0.001156
grad_step = 000452, loss = 0.001178
grad_step = 000453, loss = 0.001196
grad_step = 000454, loss = 0.001200
grad_step = 000455, loss = 0.001186
grad_step = 000456, loss = 0.001164
grad_step = 000457, loss = 0.001143
grad_step = 000458, loss = 0.001132
grad_step = 000459, loss = 0.001132
grad_step = 000460, loss = 0.001140
grad_step = 000461, loss = 0.001152
grad_step = 000462, loss = 0.001162
grad_step = 000463, loss = 0.001168
grad_step = 000464, loss = 0.001168
grad_step = 000465, loss = 0.001163
grad_step = 000466, loss = 0.001155
grad_step = 000467, loss = 0.001147
grad_step = 000468, loss = 0.001141
grad_step = 000469, loss = 0.001137
grad_step = 000470, loss = 0.001136
grad_step = 000471, loss = 0.001138
grad_step = 000472, loss = 0.001142
grad_step = 000473, loss = 0.001148
grad_step = 000474, loss = 0.001154
grad_step = 000475, loss = 0.001164
grad_step = 000476, loss = 0.001172
grad_step = 000477, loss = 0.001184
grad_step = 000478, loss = 0.001193
grad_step = 000479, loss = 0.001206
grad_step = 000480, loss = 0.001212
grad_step = 000481, loss = 0.001222
grad_step = 000482, loss = 0.001218
grad_step = 000483, loss = 0.001216
grad_step = 000484, loss = 0.001201
grad_step = 000485, loss = 0.001187
grad_step = 000486, loss = 0.001173
grad_step = 000487, loss = 0.001167
grad_step = 000488, loss = 0.001172
grad_step = 000489, loss = 0.001189
grad_step = 000490, loss = 0.001219
grad_step = 000491, loss = 0.001255
grad_step = 000492, loss = 0.001289
grad_step = 000493, loss = 0.001308
grad_step = 000494, loss = 0.001303
grad_step = 000495, loss = 0.001271
grad_step = 000496, loss = 0.001217
grad_step = 000497, loss = 0.001160
grad_step = 000498, loss = 0.001116
grad_step = 000499, loss = 0.001099
grad_step = 000500, loss = 0.001106
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001129
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

  date_run                              2020-05-11 11:11:53.654318
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.274083
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 11:11:53.659168
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.190305
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 11:11:53.665168
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.152606
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 11:11:53.669663
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.89175
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
0   2020-05-11 11:11:30.136716  ...    mean_absolute_error
1   2020-05-11 11:11:30.139388  ...     mean_squared_error
2   2020-05-11 11:11:30.142152  ...  median_absolute_error
3   2020-05-11 11:11:30.144883  ...               r2_score
4   2020-05-11 11:11:37.225588  ...    mean_absolute_error
5   2020-05-11 11:11:37.229425  ...     mean_squared_error
6   2020-05-11 11:11:37.232187  ...  median_absolute_error
7   2020-05-11 11:11:37.234730  ...               r2_score
8   2020-05-11 11:11:53.654318  ...    mean_absolute_error
9   2020-05-11 11:11:53.659168  ...     mean_squared_error
10  2020-05-11 11:11:53.665168  ...  median_absolute_error
11  2020-05-11 11:11:53.669663  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 11%|█         | 1105920/9912422 [00:00<00:00, 11027214.46it/s] 36%|███▋      | 3604480/9912422 [00:00<00:00, 13240502.04it/s] 61%|██████    | 6029312/9912422 [00:00<00:00, 15319242.48it/s] 87%|████████▋ | 8667136/9912422 [00:00<00:00, 17446754.65it/s]9920512it [00:00, 17618192.65it/s]                             
0it [00:00, ?it/s]32768it [00:00, 633050.00it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 471810.25it/s]1654784it [00:00, 11951933.65it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200499.14it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6d72dfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f0ae49ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6d6b8ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f0a921048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f0ae460b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f200b2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f0ae49fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f200c1f28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f0ae460b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f200b2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3f6d6b8ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2a140e4208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=75f64dc3ab9989c98ee75444a8d3fbaa9ca50ec4c73b9843e4cf38e3786c2c68
  Stored in directory: /tmp/pip-ephem-wheel-cache-6yeci5hg/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f29abccc208> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3760128/17464789 [=====>........................] - ETA: 0s
11714560/17464789 [===================>..........] - ETA: 0s
16801792/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 11:13:16.329279: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 11:13:16.333500: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095250000 Hz
2020-05-11 11:13:16.333669: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55afb2c5d970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 11:13:16.333708: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 8.0040 - accuracy: 0.4780
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6381 - accuracy: 0.5019
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.5842 - accuracy: 0.5054
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6121 - accuracy: 0.5036
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5961 - accuracy: 0.5046
11000/25000 [============>.................] - ETA: 3s - loss: 7.5900 - accuracy: 0.5050
12000/25000 [=============>................] - ETA: 2s - loss: 7.5797 - accuracy: 0.5057
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6159 - accuracy: 0.5033
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6400 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6216 - accuracy: 0.5029
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5999 - accuracy: 0.5044
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6375 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6257 - accuracy: 0.5027
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6276 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6446 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 6s 250us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 11:13:28.121146
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 11:13:28.121146  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 11:13:33.449311: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 11:13:33.455076: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095250000 Hz
2020-05-11 11:13:33.455667: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ed9c6619c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 11:13:33.455987: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ffab0cf0dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1750 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.1660 - val_crf_viterbi_accuracy: 0.6800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ffaa6f928d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8097 - accuracy: 0.4907
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7855 - accuracy: 0.4922
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.7617 - accuracy: 0.4938
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7535 - accuracy: 0.4943
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7652 - accuracy: 0.4936
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.7855 - accuracy: 0.4922
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7654 - accuracy: 0.4936
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7540 - accuracy: 0.4943
11000/25000 [============>.................] - ETA: 3s - loss: 7.7391 - accuracy: 0.4953
12000/25000 [=============>................] - ETA: 2s - loss: 7.7101 - accuracy: 0.4972
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6772 - accuracy: 0.4993
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6350 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6161 - accuracy: 0.5033
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6155 - accuracy: 0.5033
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6126 - accuracy: 0.5035
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6107 - accuracy: 0.5037
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6024 - accuracy: 0.5042
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6199 - accuracy: 0.5030
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6473 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 6s 248us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ffaa40ee5f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:52:29, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:32:51, 15.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:56:18, 21.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 852k/862M [00:01<7:39:46, 31.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.58M/862M [00:01<5:22:09, 44.5kB/s].vector_cache/glove.6B.zip:   1%|          | 4.88M/862M [00:01<3:44:46, 63.6kB/s].vector_cache/glove.6B.zip:   1%|          | 8.86M/862M [00:01<2:36:43, 90.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.5M/862M [00:01<1:49:12, 130kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:16:09, 185kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<53:08, 263kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<37:05, 375kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:01<25:56, 534kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<18:08, 759kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.6M/862M [00:02<12:43, 1.08MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<08:56, 1.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<06:21, 2.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<04:50, 2.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<05:18, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<07:34, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:07, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<04:38, 2.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:02, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:24, 2.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:16, 3.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<03:08, 4.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<26:31, 501kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<21:16, 625kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<15:34, 853kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<13:00, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<10:28, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<07:38, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<05:30, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<52:48, 249kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<39:37, 332kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<28:19, 464kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:12<20:01, 655kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<18:16, 716kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<14:07, 926kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:14<10:12, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<10:10, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<09:47, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<07:21, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<07:00, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<04:39, 2.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<06:16, 2.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:41, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<04:15, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<06:51, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<03:55, 3.25MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<1:25:08, 150kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<1:00:53, 209kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<42:52, 297kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<32:51, 386kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<24:17, 522kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<17:14, 733kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<15:02, 838kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<11:49, 1.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<08:34, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:57, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:32, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:32, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:29, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:38, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:05, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:33, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:48, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:19, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:59, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:27, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:08, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:33, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:10, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:55, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:32, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:08, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:53, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:21, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<04:58, 2.42MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<03:38, 3.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:10, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:57, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:07, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:22, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:53, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:25, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<03:55, 3.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<11:23:35, 17.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<7:59:30, 24.7kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<5:35:14, 35.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<3:56:40, 49.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<2:48:01, 70.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:58:05, 99.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:24:12, 139kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<59:54, 196kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<42:19, 276kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<29:39, 393kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:30:03, 129kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:05:25, 178kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<46:15, 251kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<32:30, 357kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<26:35, 436kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<19:48, 584kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:05, 820kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<12:32, 918kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<09:44, 1.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<07:09, 1.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:07, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<49:46, 230kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<37:08, 308kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<26:30, 431kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<18:39, 611kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<18:40, 609kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<14:13, 799kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<10:10, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:03, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:51, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<04:54, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<13:18, 844kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<10:28, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<07:36, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:55, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:49, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:02, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<04:20, 2.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<14:48, 750kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<11:33, 961kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<08:21, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:23, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:00, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:10, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:12, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:01, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:43, 2.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:54, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:57, 2.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:18, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:00, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:45, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:27, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:19:13, 136kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<56:31, 190kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<39:43, 270kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<30:13, 354kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<23:20, 458kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<16:51, 633kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<13:28, 789kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:32, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:38, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:47, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:36, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:51, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:46, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:09, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:49, 2.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:44, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:32, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:15, 3.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<52:31, 197kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<37:48, 273kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<26:40, 387kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<21:01, 489kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<15:45, 652kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<11:15, 909kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:16, 994kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<09:15, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:55, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:57, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<09:04, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:23, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:25, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:09, 1.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:19, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:58, 2.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:07, 1.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:35, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:27, 2.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:46, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:23, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:16, 2.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:34, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:15, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:10, 3.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:30, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:03, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<02:57, 3.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:16, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:04, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:29, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:44, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:31, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:15, 2.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<39:15, 244kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<28:27, 337kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<20:07, 475kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<16:15, 586kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<13:14, 719kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<09:44, 976kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:18, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:47, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:59, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:39, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:46, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:34, 2.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:42, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:10, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:01, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<02:55, 3.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:41, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:37, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:08, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:26, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:20, 2.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:36, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:04, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:58, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<02:51, 3.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<30:02, 301kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<21:55, 412kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<15:32, 580kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<12:56, 693kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<10:53, 824kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<07:59, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<05:41, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:43, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<07:00, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<05:07, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:38, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:44, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:24, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<03:09, 2.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<21:22, 410kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<15:52, 552kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<11:17, 773kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<09:55, 877kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<07:49, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:41, 1.52MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:00, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:03, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:43, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:38, 1.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:59, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:06, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:35, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:43, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<01:59, 4.22MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<35:24, 238kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<25:37, 329kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<18:03, 465kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<14:32, 575kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<11:47, 708kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:36, 969kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<06:06, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<09:15, 894kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<07:20, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:20, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:38, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:37, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:17, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:07, 2.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:04, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:22, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:15, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:10, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<04:33, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:32, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:34, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:33, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:43, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:30, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:17, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:37, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:35, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:35, 3.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:37, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:07, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:28, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:57, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:03, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:56, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:49, 2.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<31:46, 244kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<23:01, 336kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<16:15, 474kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<13:07, 584kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<09:57, 770kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<07:07, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:44, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:15, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:43, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:23, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<7:16:47, 17.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<5:06:16, 24.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<3:33:50, 35.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<2:30:43, 49.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:46:10, 70.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<1:14:15, 100kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<53:27, 138kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<38:08, 194kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<26:45, 275kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<20:23, 359kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<15:00, 488kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<10:39, 685kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<09:08, 794kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:52, 921kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<05:52, 1.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:15, 1.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:25, 1.62MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:16, 2.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:56, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:14, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:27, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:08, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:21, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:46, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:56, 2.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:07, 3.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:14, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:05, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:43, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:11, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:20, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:20, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:26, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:06, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:38, 2.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:42, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:52, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:11, 3.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:50, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:09, 3.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<01:34, 4.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<33:39, 195kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<24:53, 264kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<17:41, 371kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<12:24, 525kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<11:52, 548kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<08:35, 755kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<06:05, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<06:44, 956kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:22, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:54, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:04, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:19, 2.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:05, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:48, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:07, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:56, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:35, 2.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<01:56, 3.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:26, 4.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<25:50, 238kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<19:20, 318kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<13:46, 446kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<09:41, 631kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<09:07, 667kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<07:00, 868kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:01, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:52, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:35, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:30, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:30, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<09:01, 660kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:56, 856kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:58, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:48, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:33, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:29, 1.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:21, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:57, 1.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:11, 2.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:11, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:28, 2.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:49, 3.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:59, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:35, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:55, 2.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:26, 3.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<18:37, 301kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<13:35, 412kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<09:36, 580kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:59, 693kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<06:09, 900kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<04:25, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:22, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:10, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:11, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<02:16, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:04:38, 83.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<45:44, 118kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<32:00, 168kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<23:30, 227kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<16:52, 316kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<11:52, 447kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<08:19, 633kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<25:44, 205kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<19:04, 276kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<13:33, 388kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<09:31, 549kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:28, 613kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:27, 803kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:37, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:24, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:08, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:08, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:00, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:36, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:55, 2.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:31, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:08, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:32, 3.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<4:50:49, 16.9kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<3:23:49, 24.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<2:22:02, 34.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<1:39:50, 48.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<1:10:49, 68.5kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<49:40, 97.4kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<34:30, 139kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<29:13, 164kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<20:55, 229kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<14:40, 324kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<11:18, 417kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<08:52, 531kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<06:24, 735kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<04:30, 1.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<05:24, 861kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:14, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:04, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:12, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:43, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:00, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:27, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:11, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:38, 2.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:10, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:26, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:55, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:22<01:23, 3.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<32:13, 136kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<22:58, 190kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<16:05, 270kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<12:11, 354kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<09:24, 458kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<06:45, 635kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<04:43, 898kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<07:23, 573kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<05:36, 754kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:00, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:45, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:31, 1.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:38, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:52, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:44, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:02, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:12, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:28, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:32, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:57, 2.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:24, 2.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:55, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:48, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:09, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:54, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:24, 2.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:53, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:06, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:38, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:10, 3.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:09, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:36, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:54, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:16, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:44, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:17, 2.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:01, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:20, 2.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:31, 2.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:05, 3.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:55, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:24, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:44, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:00, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:04, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:35, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:08, 2.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<05:43, 584kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<04:21, 768kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:05, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:55, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:42, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:01, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:38, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:10, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:34, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:50, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:36, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:10, 2.64MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:32, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:42, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:19, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<00:56, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:31, 665kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:32, 846kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:31, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:26, 1.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:17, 1.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:44, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:14, 2.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<21:38, 133kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<15:24, 186kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<10:45, 264kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<08:05, 346kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<06:13, 449kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:27, 624kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<03:05, 883kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<05:40, 482kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:14, 642kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:00, 896kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:42, 984kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:27, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:50, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<19:27, 133kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<13:50, 187kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<09:39, 265kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<07:15, 348kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<05:35, 451kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<04:01, 624kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<02:47, 883kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<22:19, 110kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<15:49, 155kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<11:01, 220kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<08:09, 293kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<05:54, 404kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<04:08, 569kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<02:53, 803kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<13:33, 171kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<09:56, 233kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<07:02, 327kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<05:10, 434kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:50, 584kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:43, 816kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:23, 915kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:53, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:21, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:26, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:12, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:05, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:11, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:38, 3.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<02:41, 733kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:05, 946kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:28, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:28, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:13, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:58, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:52, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:38, 2.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:57, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:32, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:40, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:28, 3.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:11, 302kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:46, 413kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:38, 581kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:09, 695kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:48, 825kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:19, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:07, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:54, 1.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:42, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:48<00:29, 2.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<05:48, 234kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<04:20, 313kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<03:03, 438kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<02:05, 621kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:14, 574kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:42, 755kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<01:11, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:05, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:53, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:38, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:42, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:43, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:33, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:23, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:24, 774kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:05, 994kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:45, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:45, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:31, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:26, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:19, 2.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:13, 3.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<03:29, 253kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:36, 336kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:50, 470kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:15, 662kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:06, 732kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:51, 946kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:35, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:34, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:28, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:20, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:24, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:14, 2.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:17, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:15, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:10, 2.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:11, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:04, 4.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:21, 243kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:58, 335kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:37, 474kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:27, 584kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:21, 714kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:15, 969kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:10, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:05, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.90MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 917/400000 [00:00<00:43, 9165.35it/s]  0%|          | 1826/400000 [00:00<00:43, 9140.37it/s]  1%|          | 2773/400000 [00:00<00:43, 9234.70it/s]  1%|          | 3734/400000 [00:00<00:42, 9343.87it/s]  1%|          | 4688/400000 [00:00<00:42, 9400.88it/s]  1%|▏         | 5600/400000 [00:00<00:42, 9314.73it/s]  2%|▏         | 6579/400000 [00:00<00:41, 9450.51it/s]  2%|▏         | 7575/400000 [00:00<00:40, 9597.10it/s]  2%|▏         | 8483/400000 [00:00<00:41, 9396.33it/s]  2%|▏         | 9388/400000 [00:01<00:42, 9288.43it/s]  3%|▎         | 10358/400000 [00:01<00:41, 9406.45it/s]  3%|▎         | 11330/400000 [00:01<00:40, 9496.77it/s]  3%|▎         | 12366/400000 [00:01<00:39, 9738.56it/s]  3%|▎         | 13334/400000 [00:01<00:40, 9518.94it/s]  4%|▎         | 14283/400000 [00:01<00:41, 9225.71it/s]  4%|▍         | 15292/400000 [00:01<00:40, 9466.81it/s]  4%|▍         | 16273/400000 [00:01<00:40, 9562.89it/s]  4%|▍         | 17242/400000 [00:01<00:39, 9600.24it/s]  5%|▍         | 18243/400000 [00:01<00:39, 9717.30it/s]  5%|▍         | 19216/400000 [00:02<00:40, 9484.27it/s]  5%|▌         | 20224/400000 [00:02<00:39, 9654.86it/s]  5%|▌         | 21236/400000 [00:02<00:38, 9787.76it/s]  6%|▌         | 22263/400000 [00:02<00:38, 9925.84it/s]  6%|▌         | 23258/400000 [00:02<00:38, 9888.62it/s]  6%|▌         | 24249/400000 [00:02<00:38, 9811.62it/s]  6%|▋         | 25289/400000 [00:02<00:37, 9978.81it/s]  7%|▋         | 26316/400000 [00:02<00:37, 10064.22it/s]  7%|▋         | 27369/400000 [00:02<00:36, 10198.46it/s]  7%|▋         | 28391/400000 [00:02<00:36, 10181.55it/s]  7%|▋         | 29411/400000 [00:03<00:37, 9972.28it/s]   8%|▊         | 30410/400000 [00:03<00:37, 9943.33it/s]  8%|▊         | 31406/400000 [00:03<00:37, 9706.40it/s]  8%|▊         | 32437/400000 [00:03<00:37, 9877.77it/s]  8%|▊         | 33449/400000 [00:03<00:36, 9947.83it/s]  9%|▊         | 34446/400000 [00:03<00:37, 9763.65it/s]  9%|▉         | 35425/400000 [00:03<00:37, 9763.61it/s]  9%|▉         | 36403/400000 [00:03<00:37, 9758.66it/s]  9%|▉         | 37380/400000 [00:03<00:38, 9530.99it/s] 10%|▉         | 38335/400000 [00:03<00:38, 9423.05it/s] 10%|▉         | 39279/400000 [00:04<00:38, 9259.76it/s] 10%|█         | 40220/400000 [00:04<00:38, 9303.82it/s] 10%|█         | 41152/400000 [00:04<00:39, 9190.51it/s] 11%|█         | 42101/400000 [00:04<00:38, 9276.95it/s] 11%|█         | 43032/400000 [00:04<00:38, 9284.06it/s] 11%|█         | 44022/400000 [00:04<00:37, 9459.39it/s] 11%|█▏        | 45095/400000 [00:04<00:36, 9806.67it/s] 12%|█▏        | 46094/400000 [00:04<00:35, 9859.75it/s] 12%|█▏        | 47083/400000 [00:04<00:36, 9787.91it/s] 12%|█▏        | 48067/400000 [00:04<00:35, 9800.46it/s] 12%|█▏        | 49050/400000 [00:05<00:35, 9807.37it/s] 13%|█▎        | 50066/400000 [00:05<00:35, 9908.75it/s] 13%|█▎        | 51058/400000 [00:05<00:35, 9910.09it/s] 13%|█▎        | 52057/400000 [00:05<00:35, 9931.54it/s] 13%|█▎        | 53065/400000 [00:05<00:34, 9974.51it/s] 14%|█▎        | 54063/400000 [00:05<00:34, 9973.71it/s] 14%|█▍        | 55061/400000 [00:05<00:34, 9918.05it/s] 14%|█▍        | 56054/400000 [00:05<00:35, 9752.19it/s] 14%|█▍        | 57077/400000 [00:05<00:34, 9887.56it/s] 15%|█▍        | 58146/400000 [00:05<00:33, 10114.24it/s] 15%|█▍        | 59172/400000 [00:06<00:33, 10156.46it/s] 15%|█▌        | 60190/400000 [00:06<00:33, 10135.95it/s] 15%|█▌        | 61251/400000 [00:06<00:32, 10273.17it/s] 16%|█▌        | 62362/400000 [00:06<00:32, 10510.20it/s] 16%|█▌        | 63417/400000 [00:06<00:31, 10521.05it/s] 16%|█▌        | 64471/400000 [00:06<00:32, 10335.15it/s] 16%|█▋        | 65540/400000 [00:06<00:32, 10438.22it/s] 17%|█▋        | 66588/400000 [00:06<00:31, 10449.91it/s] 17%|█▋        | 67635/400000 [00:06<00:31, 10397.41it/s] 17%|█▋        | 68676/400000 [00:07<00:32, 10051.49it/s] 17%|█▋        | 69685/400000 [00:07<00:32, 10043.72it/s] 18%|█▊        | 70802/400000 [00:07<00:31, 10356.49it/s] 18%|█▊        | 71910/400000 [00:07<00:31, 10562.46it/s] 18%|█▊        | 72971/400000 [00:07<00:31, 10393.53it/s] 19%|█▊        | 74057/400000 [00:07<00:30, 10527.49it/s] 19%|█▉        | 75113/400000 [00:07<00:30, 10480.42it/s] 19%|█▉        | 76212/400000 [00:07<00:30, 10628.08it/s] 19%|█▉        | 77277/400000 [00:07<00:30, 10557.26it/s] 20%|█▉        | 78335/400000 [00:07<00:31, 10314.93it/s] 20%|█▉        | 79369/400000 [00:08<00:31, 10066.53it/s] 20%|██        | 80379/400000 [00:08<00:32, 9963.91it/s]  20%|██        | 81430/400000 [00:08<00:31, 10120.89it/s] 21%|██        | 82445/400000 [00:08<00:31, 10023.55it/s] 21%|██        | 83468/400000 [00:08<00:31, 10082.49it/s] 21%|██        | 84478/400000 [00:08<00:31, 10081.88it/s] 21%|██▏       | 85488/400000 [00:08<00:32, 9733.78it/s]  22%|██▏       | 86465/400000 [00:08<00:33, 9440.49it/s] 22%|██▏       | 87414/400000 [00:08<00:33, 9295.80it/s] 22%|██▏       | 88348/400000 [00:08<00:34, 9157.57it/s] 22%|██▏       | 89267/400000 [00:09<00:34, 9009.72it/s] 23%|██▎       | 90171/400000 [00:09<00:34, 8957.36it/s] 23%|██▎       | 91080/400000 [00:09<00:34, 8995.28it/s] 23%|██▎       | 92007/400000 [00:09<00:33, 9075.37it/s] 23%|██▎       | 92967/400000 [00:09<00:33, 9223.69it/s] 23%|██▎       | 93891/400000 [00:09<00:33, 9132.06it/s] 24%|██▎       | 94806/400000 [00:09<00:33, 9051.69it/s] 24%|██▍       | 95713/400000 [00:09<00:33, 8984.46it/s] 24%|██▍       | 96620/400000 [00:09<00:33, 9008.87it/s] 24%|██▍       | 97522/400000 [00:10<00:33, 8943.07it/s] 25%|██▍       | 98474/400000 [00:10<00:33, 9106.19it/s] 25%|██▍       | 99429/400000 [00:10<00:32, 9232.80it/s] 25%|██▌       | 100354/400000 [00:10<00:32, 9104.62it/s] 25%|██▌       | 101266/400000 [00:10<00:33, 9038.19it/s] 26%|██▌       | 102234/400000 [00:10<00:32, 9221.52it/s] 26%|██▌       | 103158/400000 [00:10<00:32, 9179.96it/s] 26%|██▌       | 104078/400000 [00:10<00:32, 9121.00it/s] 26%|██▋       | 105101/400000 [00:10<00:31, 9425.63it/s] 27%|██▋       | 106047/400000 [00:10<00:31, 9316.25it/s] 27%|██▋       | 106999/400000 [00:11<00:31, 9375.77it/s] 27%|██▋       | 107977/400000 [00:11<00:30, 9491.24it/s] 27%|██▋       | 108928/400000 [00:11<00:31, 9333.74it/s] 27%|██▋       | 109879/400000 [00:11<00:30, 9384.61it/s] 28%|██▊       | 110851/400000 [00:11<00:30, 9481.48it/s] 28%|██▊       | 111801/400000 [00:11<00:30, 9407.43it/s] 28%|██▊       | 112750/400000 [00:11<00:30, 9429.79it/s] 28%|██▊       | 113694/400000 [00:11<00:30, 9285.57it/s] 29%|██▊       | 114638/400000 [00:11<00:30, 9327.34it/s] 29%|██▉       | 115572/400000 [00:11<00:30, 9210.71it/s] 29%|██▉       | 116497/400000 [00:12<00:30, 9222.27it/s] 29%|██▉       | 117420/400000 [00:12<00:30, 9186.17it/s] 30%|██▉       | 118340/400000 [00:12<00:30, 9156.75it/s] 30%|██▉       | 119315/400000 [00:12<00:30, 9325.24it/s] 30%|███       | 120280/400000 [00:12<00:29, 9419.71it/s] 30%|███       | 121223/400000 [00:12<00:30, 9278.21it/s] 31%|███       | 122152/400000 [00:12<00:30, 9185.83it/s] 31%|███       | 123072/400000 [00:12<00:30, 9174.25it/s] 31%|███       | 124028/400000 [00:12<00:29, 9286.23it/s] 31%|███▏      | 125003/400000 [00:12<00:29, 9419.55it/s] 31%|███▏      | 125949/400000 [00:13<00:29, 9428.62it/s] 32%|███▏      | 126962/400000 [00:13<00:28, 9627.24it/s] 32%|███▏      | 127927/400000 [00:13<00:28, 9472.42it/s] 32%|███▏      | 128876/400000 [00:13<00:28, 9388.06it/s] 32%|███▏      | 129817/400000 [00:13<00:28, 9369.36it/s] 33%|███▎      | 130775/400000 [00:13<00:28, 9430.76it/s] 33%|███▎      | 131719/400000 [00:13<00:28, 9289.06it/s] 33%|███▎      | 132699/400000 [00:13<00:28, 9433.92it/s] 33%|███▎      | 133669/400000 [00:13<00:28, 9511.59it/s] 34%|███▎      | 134647/400000 [00:13<00:27, 9588.60it/s] 34%|███▍      | 135609/400000 [00:14<00:27, 9595.83it/s] 34%|███▍      | 136570/400000 [00:14<00:27, 9432.78it/s] 34%|███▍      | 137515/400000 [00:14<00:28, 9241.02it/s] 35%|███▍      | 138485/400000 [00:14<00:27, 9373.66it/s] 35%|███▍      | 139436/400000 [00:14<00:27, 9413.64it/s] 35%|███▌      | 140400/400000 [00:14<00:27, 9479.34it/s] 35%|███▌      | 141408/400000 [00:14<00:26, 9651.81it/s] 36%|███▌      | 142375/400000 [00:14<00:26, 9586.29it/s] 36%|███▌      | 143335/400000 [00:14<00:26, 9513.82it/s] 36%|███▌      | 144366/400000 [00:14<00:26, 9739.35it/s] 36%|███▋      | 145400/400000 [00:15<00:25, 9909.95it/s] 37%|███▋      | 146394/400000 [00:15<00:25, 9796.53it/s] 37%|███▋      | 147376/400000 [00:15<00:25, 9761.09it/s] 37%|███▋      | 148424/400000 [00:15<00:25, 9965.46it/s] 37%|███▋      | 149451/400000 [00:15<00:24, 10053.60it/s] 38%|███▊      | 150458/400000 [00:15<00:24, 10043.89it/s] 38%|███▊      | 151508/400000 [00:15<00:24, 10174.75it/s] 38%|███▊      | 152527/400000 [00:15<00:24, 10150.14it/s] 38%|███▊      | 153543/400000 [00:15<00:24, 9883.63it/s]  39%|███▊      | 154565/400000 [00:15<00:24, 9981.68it/s] 39%|███▉      | 155574/400000 [00:16<00:24, 10012.84it/s] 39%|███▉      | 156577/400000 [00:16<00:24, 9835.97it/s]  39%|███▉      | 157563/400000 [00:16<00:25, 9654.63it/s] 40%|███▉      | 158531/400000 [00:16<00:25, 9530.54it/s] 40%|███▉      | 159551/400000 [00:16<00:24, 9720.39it/s] 40%|████      | 160598/400000 [00:16<00:24, 9933.52it/s] 40%|████      | 161594/400000 [00:16<00:24, 9870.69it/s] 41%|████      | 162583/400000 [00:16<00:24, 9553.52it/s] 41%|████      | 163542/400000 [00:16<00:25, 9353.94it/s] 41%|████      | 164546/400000 [00:17<00:24, 9548.74it/s] 41%|████▏     | 165543/400000 [00:17<00:24, 9670.26it/s] 42%|████▏     | 166513/400000 [00:17<00:24, 9630.51it/s] 42%|████▏     | 167479/400000 [00:17<00:24, 9582.77it/s] 42%|████▏     | 168446/400000 [00:17<00:24, 9608.31it/s] 42%|████▏     | 169442/400000 [00:17<00:23, 9709.61it/s] 43%|████▎     | 170525/400000 [00:17<00:22, 10019.49it/s] 43%|████▎     | 171547/400000 [00:17<00:22, 10075.87it/s] 43%|████▎     | 172557/400000 [00:17<00:22, 10082.38it/s] 43%|████▎     | 173593/400000 [00:17<00:22, 10161.71it/s] 44%|████▎     | 174611/400000 [00:18<00:22, 10014.31it/s] 44%|████▍     | 175653/400000 [00:18<00:22, 10130.02it/s] 44%|████▍     | 176709/400000 [00:18<00:21, 10254.14it/s] 44%|████▍     | 177756/400000 [00:18<00:21, 10315.84it/s] 45%|████▍     | 178839/400000 [00:18<00:21, 10462.66it/s] 45%|████▍     | 179887/400000 [00:18<00:21, 10456.75it/s] 45%|████▌     | 180934/400000 [00:18<00:21, 10229.82it/s] 45%|████▌     | 181965/400000 [00:18<00:21, 10251.51it/s] 46%|████▌     | 182992/400000 [00:18<00:21, 10164.28it/s] 46%|████▌     | 184010/400000 [00:18<00:21, 10137.65it/s] 46%|████▋     | 185025/400000 [00:19<00:21, 9999.37it/s]  47%|████▋     | 186086/400000 [00:19<00:21, 10172.29it/s] 47%|████▋     | 187105/400000 [00:19<00:21, 9941.49it/s]  47%|████▋     | 188147/400000 [00:19<00:21, 10078.31it/s] 47%|████▋     | 189157/400000 [00:19<00:20, 10078.22it/s] 48%|████▊     | 190167/400000 [00:19<00:21, 9920.82it/s]  48%|████▊     | 191161/400000 [00:19<00:21, 9840.29it/s] 48%|████▊     | 192147/400000 [00:19<00:21, 9557.68it/s] 48%|████▊     | 193106/400000 [00:19<00:21, 9434.27it/s] 49%|████▊     | 194066/400000 [00:19<00:21, 9480.96it/s] 49%|████▉     | 195021/400000 [00:20<00:21, 9500.95it/s] 49%|████▉     | 195973/400000 [00:20<00:21, 9503.91it/s] 49%|████▉     | 196925/400000 [00:20<00:21, 9485.07it/s] 49%|████▉     | 197894/400000 [00:20<00:21, 9544.18it/s] 50%|████▉     | 198951/400000 [00:20<00:20, 9828.52it/s] 50%|█████     | 200006/400000 [00:20<00:19, 10034.31it/s] 50%|█████     | 201013/400000 [00:20<00:19, 9953.00it/s]  51%|█████     | 202058/400000 [00:20<00:19, 10096.82it/s] 51%|█████     | 203070/400000 [00:20<00:19, 10069.98it/s] 51%|█████     | 204079/400000 [00:21<00:19, 9982.27it/s]  51%|█████▏    | 205134/400000 [00:21<00:19, 10144.18it/s] 52%|█████▏    | 206150/400000 [00:21<00:19, 10114.12it/s] 52%|█████▏    | 207163/400000 [00:21<00:19, 10012.77it/s] 52%|█████▏    | 208166/400000 [00:21<00:19, 9629.50it/s]  52%|█████▏    | 209151/400000 [00:21<00:19, 9693.51it/s] 53%|█████▎    | 210180/400000 [00:21<00:19, 9864.05it/s] 53%|█████▎    | 211170/400000 [00:21<00:19, 9712.09it/s] 53%|█████▎    | 212162/400000 [00:21<00:19, 9771.90it/s] 53%|█████▎    | 213214/400000 [00:21<00:18, 9983.95it/s] 54%|█████▎    | 214300/400000 [00:22<00:18, 10229.43it/s] 54%|█████▍    | 215345/400000 [00:22<00:17, 10292.88it/s] 54%|█████▍    | 216420/400000 [00:22<00:17, 10423.99it/s] 54%|█████▍    | 217465/400000 [00:22<00:17, 10292.35it/s] 55%|█████▍    | 218497/400000 [00:22<00:17, 10275.70it/s] 55%|█████▍    | 219526/400000 [00:22<00:17, 10158.95it/s] 55%|█████▌    | 220555/400000 [00:22<00:17, 10197.13it/s] 55%|█████▌    | 221576/400000 [00:22<00:17, 10090.53it/s] 56%|█████▌    | 222586/400000 [00:22<00:17, 10058.06it/s] 56%|█████▌    | 223618/400000 [00:22<00:17, 10133.78it/s] 56%|█████▌    | 224652/400000 [00:23<00:17, 10192.03it/s] 56%|█████▋    | 225672/400000 [00:23<00:17, 9960.11it/s]  57%|█████▋    | 226686/400000 [00:23<00:17, 10011.53it/s] 57%|█████▋    | 227757/400000 [00:23<00:16, 10210.38it/s] 57%|█████▋    | 228889/400000 [00:23<00:16, 10518.32it/s] 57%|█████▋    | 229987/400000 [00:23<00:15, 10651.14it/s] 58%|█████▊    | 231122/400000 [00:23<00:15, 10849.30it/s] 58%|█████▊    | 232228/400000 [00:23<00:15, 10911.59it/s] 58%|█████▊    | 233322/400000 [00:23<00:15, 10842.42it/s] 59%|█████▊    | 234431/400000 [00:23<00:15, 10911.35it/s] 59%|█████▉    | 235551/400000 [00:24<00:14, 10994.57it/s] 59%|█████▉    | 236686/400000 [00:24<00:14, 11097.69it/s] 59%|█████▉    | 237797/400000 [00:24<00:14, 11010.11it/s] 60%|█████▉    | 238899/400000 [00:24<00:14, 10929.69it/s] 60%|█████▉    | 239993/400000 [00:24<00:14, 10781.94it/s] 60%|██████    | 241073/400000 [00:24<00:14, 10729.47it/s] 61%|██████    | 242168/400000 [00:24<00:14, 10793.93it/s] 61%|██████    | 243248/400000 [00:24<00:14, 10511.70it/s] 61%|██████    | 244302/400000 [00:24<00:15, 10198.02it/s] 61%|██████▏   | 245326/400000 [00:24<00:15, 9966.67it/s]  62%|██████▏   | 246327/400000 [00:25<00:15, 9946.94it/s] 62%|██████▏   | 247441/400000 [00:25<00:14, 10274.57it/s] 62%|██████▏   | 248473/400000 [00:25<00:14, 10117.37it/s] 62%|██████▏   | 249498/400000 [00:25<00:14, 10156.25it/s] 63%|██████▎   | 250522/400000 [00:25<00:14, 10180.52it/s] 63%|██████▎   | 251542/400000 [00:25<00:14, 10135.46it/s] 63%|██████▎   | 252557/400000 [00:25<00:14, 10014.95it/s] 63%|██████▎   | 253581/400000 [00:25<00:14, 10081.39it/s] 64%|██████▎   | 254591/400000 [00:25<00:14, 9994.25it/s]  64%|██████▍   | 255609/400000 [00:26<00:14, 10047.77it/s] 64%|██████▍   | 256621/400000 [00:26<00:14, 10068.38it/s] 64%|██████▍   | 257629/400000 [00:26<00:14, 9968.02it/s]  65%|██████▍   | 258627/400000 [00:26<00:14, 9868.97it/s] 65%|██████▍   | 259615/400000 [00:26<00:14, 9754.04it/s] 65%|██████▌   | 260618/400000 [00:26<00:14, 9832.68it/s] 65%|██████▌   | 261602/400000 [00:26<00:14, 9778.41it/s] 66%|██████▌   | 262581/400000 [00:26<00:14, 9655.26it/s] 66%|██████▌   | 263573/400000 [00:26<00:14, 9731.48it/s] 66%|██████▌   | 264547/400000 [00:26<00:13, 9709.75it/s] 66%|██████▋   | 265551/400000 [00:27<00:13, 9805.61it/s] 67%|██████▋   | 266565/400000 [00:27<00:13, 9900.75it/s] 67%|██████▋   | 267584/400000 [00:27<00:13, 9985.73it/s] 67%|██████▋   | 268584/400000 [00:27<00:13, 9853.09it/s] 67%|██████▋   | 269638/400000 [00:27<00:12, 10048.34it/s] 68%|██████▊   | 270734/400000 [00:27<00:12, 10303.05it/s] 68%|██████▊   | 271809/400000 [00:27<00:12, 10432.03it/s] 68%|██████▊   | 272855/400000 [00:27<00:12, 10429.82it/s] 68%|██████▊   | 273967/400000 [00:27<00:11, 10626.88it/s] 69%|██████▉   | 275032/400000 [00:27<00:11, 10516.75it/s] 69%|██████▉   | 276093/400000 [00:28<00:11, 10542.58it/s] 69%|██████▉   | 277152/400000 [00:28<00:11, 10554.87it/s] 70%|██████▉   | 278211/400000 [00:28<00:11, 10563.80it/s] 70%|██████▉   | 279268/400000 [00:28<00:11, 10513.94it/s] 70%|███████   | 280320/400000 [00:28<00:11, 10325.19it/s] 70%|███████   | 281354/400000 [00:28<00:11, 10309.42it/s] 71%|███████   | 282386/400000 [00:28<00:11, 10239.07it/s] 71%|███████   | 283427/400000 [00:28<00:11, 10289.16it/s] 71%|███████   | 284497/400000 [00:28<00:11, 10408.73it/s] 71%|███████▏  | 285539/400000 [00:28<00:11, 10384.51it/s] 72%|███████▏  | 286578/400000 [00:29<00:11, 10300.71it/s] 72%|███████▏  | 287652/400000 [00:29<00:10, 10426.03it/s] 72%|███████▏  | 288723/400000 [00:29<00:10, 10506.61it/s] 72%|███████▏  | 289775/400000 [00:29<00:10, 10418.55it/s] 73%|███████▎  | 290818/400000 [00:29<00:10, 10285.69it/s] 73%|███████▎  | 291848/400000 [00:29<00:10, 10053.68it/s] 73%|███████▎  | 292856/400000 [00:29<00:10, 9869.23it/s]  73%|███████▎  | 293845/400000 [00:29<00:10, 9762.69it/s] 74%|███████▎  | 294829/400000 [00:29<00:10, 9784.88it/s] 74%|███████▍  | 295810/400000 [00:29<00:10, 9789.97it/s] 74%|███████▍  | 296790/400000 [00:30<00:10, 9670.20it/s] 74%|███████▍  | 297758/400000 [00:30<00:10, 9625.61it/s] 75%|███████▍  | 298722/400000 [00:30<00:10, 9627.87it/s] 75%|███████▍  | 299707/400000 [00:30<00:10, 9692.68it/s] 75%|███████▌  | 300677/400000 [00:30<00:10, 9642.57it/s] 75%|███████▌  | 301642/400000 [00:30<00:10, 9497.25it/s] 76%|███████▌  | 302593/400000 [00:30<00:10, 9310.13it/s] 76%|███████▌  | 303532/400000 [00:30<00:10, 9331.08it/s] 76%|███████▌  | 304467/400000 [00:30<00:10, 9334.44it/s] 76%|███████▋  | 305410/400000 [00:30<00:10, 9361.12it/s] 77%|███████▋  | 306485/400000 [00:31<00:09, 9738.53it/s] 77%|███████▋  | 307568/400000 [00:31<00:09, 10039.98it/s] 77%|███████▋  | 308578/400000 [00:31<00:09, 9997.42it/s]  77%|███████▋  | 309612/400000 [00:31<00:08, 10096.82it/s] 78%|███████▊  | 310625/400000 [00:31<00:09, 9890.14it/s]  78%|███████▊  | 311708/400000 [00:31<00:08, 10152.00it/s] 78%|███████▊  | 312750/400000 [00:31<00:08, 10230.04it/s] 78%|███████▊  | 313809/400000 [00:31<00:08, 10333.51it/s] 79%|███████▊  | 314919/400000 [00:31<00:08, 10548.69it/s] 79%|███████▉  | 315977/400000 [00:32<00:08, 10497.84it/s] 79%|███████▉  | 317029/400000 [00:32<00:08, 10298.88it/s] 80%|███████▉  | 318062/400000 [00:32<00:08, 10240.38it/s] 80%|███████▉  | 319088/400000 [00:32<00:07, 10181.20it/s] 80%|████████  | 320112/400000 [00:32<00:07, 10197.02it/s] 80%|████████  | 321133/400000 [00:32<00:07, 10107.33it/s] 81%|████████  | 322145/400000 [00:32<00:07, 10043.54it/s] 81%|████████  | 323150/400000 [00:32<00:07, 9939.00it/s]  81%|████████  | 324145/400000 [00:32<00:07, 9858.20it/s] 81%|████████▏ | 325132/400000 [00:32<00:07, 9602.91it/s] 82%|████████▏ | 326095/400000 [00:33<00:07, 9538.10it/s] 82%|████████▏ | 327079/400000 [00:33<00:07, 9624.68it/s] 82%|████████▏ | 328063/400000 [00:33<00:07, 9685.46it/s] 82%|████████▏ | 329058/400000 [00:33<00:07, 9761.94it/s] 83%|████████▎ | 330062/400000 [00:33<00:07, 9842.65it/s] 83%|████████▎ | 331047/400000 [00:33<00:07, 9744.22it/s] 83%|████████▎ | 332030/400000 [00:33<00:06, 9767.67it/s] 83%|████████▎ | 333081/400000 [00:33<00:06, 9977.63it/s] 84%|████████▎ | 334153/400000 [00:33<00:06, 10186.96it/s] 84%|████████▍ | 335216/400000 [00:33<00:06, 10315.89it/s] 84%|████████▍ | 336250/400000 [00:34<00:06, 10137.97it/s] 84%|████████▍ | 337303/400000 [00:34<00:06, 10250.94it/s] 85%|████████▍ | 338330/400000 [00:34<00:06, 10086.22it/s] 85%|████████▍ | 339341/400000 [00:34<00:06, 9978.80it/s]  85%|████████▌ | 340341/400000 [00:34<00:06, 9926.28it/s] 85%|████████▌ | 341335/400000 [00:34<00:05, 9906.06it/s] 86%|████████▌ | 342382/400000 [00:34<00:05, 10068.52it/s] 86%|████████▌ | 343447/400000 [00:34<00:05, 10235.78it/s] 86%|████████▌ | 344532/400000 [00:34<00:05, 10410.27it/s] 86%|████████▋ | 345636/400000 [00:34<00:05, 10588.35it/s] 87%|████████▋ | 346697/400000 [00:35<00:05, 10413.50it/s] 87%|████████▋ | 347741/400000 [00:35<00:05, 10218.03it/s] 87%|████████▋ | 348766/400000 [00:35<00:05, 10189.44it/s] 87%|████████▋ | 349795/400000 [00:35<00:04, 10219.08it/s] 88%|████████▊ | 350820/400000 [00:35<00:04, 10227.90it/s] 88%|████████▊ | 351844/400000 [00:35<00:04, 10159.95it/s] 88%|████████▊ | 352861/400000 [00:35<00:04, 9852.71it/s]  88%|████████▊ | 353897/400000 [00:35<00:04, 9997.29it/s] 89%|████████▊ | 354921/400000 [00:35<00:04, 10067.92it/s] 89%|████████▉ | 355930/400000 [00:35<00:04, 9919.35it/s]  89%|████████▉ | 356924/400000 [00:36<00:04, 9890.80it/s] 89%|████████▉ | 357978/400000 [00:36<00:04, 10076.20it/s] 90%|████████▉ | 358988/400000 [00:36<00:04, 9968.59it/s]  90%|█████████ | 360048/400000 [00:36<00:03, 10149.61it/s] 90%|█████████ | 361065/400000 [00:36<00:03, 10115.51it/s] 91%|█████████ | 362078/400000 [00:36<00:03, 10067.04it/s] 91%|█████████ | 363145/400000 [00:36<00:03, 10240.47it/s] 91%|█████████ | 364277/400000 [00:36<00:03, 10540.52it/s] 91%|█████████▏| 365335/400000 [00:36<00:03, 10310.27it/s] 92%|█████████▏| 366370/400000 [00:37<00:03, 10243.25it/s] 92%|█████████▏| 367429/400000 [00:37<00:03, 10344.24it/s] 92%|█████████▏| 368491/400000 [00:37<00:03, 10424.50it/s] 92%|█████████▏| 369576/400000 [00:37<00:02, 10548.38it/s] 93%|█████████▎| 370667/400000 [00:37<00:02, 10651.73it/s] 93%|█████████▎| 371734/400000 [00:37<00:02, 10428.42it/s] 93%|█████████▎| 372779/400000 [00:37<00:02, 10320.11it/s] 93%|█████████▎| 373813/400000 [00:37<00:02, 10184.36it/s] 94%|█████████▎| 374976/400000 [00:37<00:02, 10577.59it/s] 94%|█████████▍| 376102/400000 [00:37<00:02, 10771.09it/s] 94%|█████████▍| 377202/400000 [00:38<00:02, 10835.39it/s] 95%|█████████▍| 378289/400000 [00:38<00:02, 10447.07it/s] 95%|█████████▍| 379339/400000 [00:38<00:02, 10293.81it/s] 95%|█████████▌| 380373/400000 [00:38<00:01, 10300.61it/s] 95%|█████████▌| 381455/400000 [00:38<00:01, 10447.96it/s] 96%|█████████▌| 382503/400000 [00:38<00:01, 10363.90it/s] 96%|█████████▌| 383586/400000 [00:38<00:01, 10496.69it/s] 96%|█████████▌| 384645/400000 [00:38<00:01, 10524.48it/s] 96%|█████████▋| 385731/400000 [00:38<00:01, 10621.87it/s] 97%|█████████▋| 386832/400000 [00:38<00:01, 10732.01it/s] 97%|█████████▋| 387907/400000 [00:39<00:01, 10578.26it/s] 97%|█████████▋| 388967/400000 [00:39<00:01, 10582.18it/s] 98%|█████████▊| 390027/400000 [00:39<00:00, 10290.86it/s] 98%|█████████▊| 391097/400000 [00:39<00:00, 10409.13it/s] 98%|█████████▊| 392150/400000 [00:39<00:00, 10445.05it/s] 98%|█████████▊| 393196/400000 [00:39<00:00, 10355.66it/s] 99%|█████████▊| 394233/400000 [00:39<00:00, 10320.27it/s] 99%|█████████▉| 395266/400000 [00:39<00:00, 10038.07it/s] 99%|█████████▉| 396319/400000 [00:39<00:00, 10180.18it/s] 99%|█████████▉| 397340/400000 [00:39<00:00, 10147.20it/s]100%|█████████▉| 398357/400000 [00:40<00:00, 10072.97it/s]100%|█████████▉| 399394/400000 [00:40<00:00, 10160.14it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9939.53it/s] >>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ffaa7180358> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011056838458529188 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010916420049890626 	 Accuracy: 67

  model saves at 67% accuracy 

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7f43845f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 12:13:02.429342
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 12:13:02.434950
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 12:13:02.439203
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 12:13:02.443425
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7f4f609400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355467.8438
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 266204.5625
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 160305.6094
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 84671.7109
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 43703.3320
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 24464.2129
Epoch 7/10

1/1 [==============================] - 0s 96ms/step - loss: 15129.9893
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 10235.6826
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 7400.5449
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 5672.9287

  #### Inference Need return ypred, ytrue ######################### 
[[ -1.8283557    0.4374864   -0.14186412   1.2870545   -0.93944514
    0.38761118  -0.2733522   -1.8284962    2.6407704   -0.36545348
   -0.16835642  -1.3640524    2.1313314    0.5108247   -0.26403934
    1.0105217   -1.1875603    0.34005338   2.1920946    1.8865235
   -2.1734242    0.8145746   -0.31064773  -0.89493537   0.5706345
   -1.6236955    0.46815795  -1.7289346   -1.1355563   -0.49667
    0.22585732  -1.2211004    2.119146     0.7292495    0.18656313
    1.5405593    0.10187465  -0.47848392   0.17787457   0.54264003
    1.5853138   -1.7297819   -1.9881008    0.35767376  -0.2701902
    1.3602682   -0.46903688   1.150888    -0.58875024  -0.15129572
    0.18123984  -0.6691676    0.6044018    0.02346826  -0.40600815
   -0.42511755   0.33324307  -0.1295278   -0.14098263  -0.8013366
   -0.4231025    7.810162     7.940768     9.773297     7.186947
    7.9982653   11.384833     9.641868     8.285033     7.2353086
    9.319517     9.789464    10.099964     9.141718     7.922773
   10.0918255    7.5976377    9.059075     7.2842484    8.511045
    9.351987     6.890112     8.745305     8.248781     9.561339
    9.407078     9.214051     9.266201     8.020221     8.926078
    9.23654      8.32473     10.494975     6.7839828    8.738641
    9.8025255    8.143394     9.85184      8.066781     6.474059
    8.604348     9.18925     10.8277645   11.1897955    9.518584
   10.103056    10.79695      7.5048976    8.679749     9.025381
    7.400644     9.432655    11.900374     8.587883     8.885116
    8.895493     9.123838    10.498343     8.352945    10.420059
    0.25098994  -1.0782752   -2.7968006   -0.30905068  -0.05803388
   -0.33460164  -2.5109596    0.43862346  -1.6189344   -0.62284124
   -0.9923591   -0.2793913   -1.7566293   -0.87296206   0.16753957
    0.87397087  -0.7581856   -0.06050146  -0.06808096  -1.3841228
    1.0506849    0.06367889  -0.24414796  -0.8382639    0.1843738
    1.1715751   -0.9487145    0.19689977   0.41088647   0.19032758
   -0.11090043   1.6780282   -0.7273407   -0.7115114   -1.5638565
   -1.1954105    2.4175563   -2.4280696    0.10949919   0.8671958
    0.9889698   -0.29637578  -1.486235    -0.5300434    2.5314097
   -0.08322209  -1.1085634   -0.3330844   -1.1148901    1.6044497
   -0.2395829    0.100283     0.30618855  -0.14712709  -1.5137081
    0.01502657  -0.88892585   1.1135349    1.9670476   -1.0165489
    2.1260705    0.4583038    0.41564506   1.8851461    0.8119947
    2.9397073    0.26460725   1.2231482    0.32662266   0.78932315
    1.7278125    0.12137872   0.622252     0.58247614   1.7153735
    1.1127926    2.3277237    1.6674507    0.6464655    1.3853539
    0.61996514   0.43089056   1.9713755    1.2152615    0.31535852
    0.5196688    0.34414983   2.6415741    0.5855084    1.0059122
    0.4777242    0.87660116   1.0540448    0.26918685   2.0073075
    0.6326986    1.2049923    0.959497     0.7737954    1.2810799
    1.8350834    0.67164034   0.84852123   2.2150588    2.1423697
    0.06810075   2.5151982    0.6470499    2.8003526    1.5563747
    1.6366441    0.26225376   1.1681597    2.8180995    0.12227857
    0.60882765   0.31744933   1.1938796    0.9850427    0.24365836
    0.18390393   9.668266     9.822935     8.651296     9.7242
   10.727959     9.291199    10.237699     9.016876     7.386544
    8.638683     9.874335     8.723833     8.377946     8.523591
    7.5423923    8.468232    10.449518    10.046943     9.923159
   10.000423     8.955892     7.356018    10.11557      9.091749
   10.760312     8.468764     7.948357     7.954522     9.382654
    7.72778      8.648046    10.141296     7.99706      6.4076886
    7.6869884    8.319511    10.217206     7.706053     9.840151
    9.359861    10.242966     8.877626    10.014909     8.012553
    8.261212     9.20494      9.953297     8.267387     8.679563
    9.120552    10.235595     8.962874     9.1081915   10.8993
    8.450233     8.65158      9.11731      9.137681     8.362843
    2.1963587    1.5984061    0.84157825   0.90017945   2.0307798
    2.2212877    1.8196129    0.4946093    0.7158152    1.9205743
    1.7197912    0.3307833    0.56287503   1.0343274    3.436543
    0.42720306   0.31645525   0.15740001   1.319768     0.5420194
    0.07278806   0.61825764   0.693695     1.5549943    0.40378243
    2.2370715    1.4208205    0.22025508   0.54604137   1.5790571
    0.14455402   1.1110886    0.43135154   1.5613081    3.5847278
    1.1511039    1.8745859    2.9069903    1.2877753    2.4457502
    0.0880574    0.24295026   1.6248646    1.7296898    0.26568532
    1.1597347    2.3216996    0.585987     2.7352068    0.8475566
    2.039577     1.9564519    0.23113513   0.26086485   1.0275668
    1.101413     1.0510782    3.911191     0.34144455   1.3874581
   -5.7781024   10.97274    -10.690836  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 12:13:11.332587
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2667
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 12:13:11.336936
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8720.94
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 12:13:11.340707
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.3219
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 12:13:11.344108
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -780.022
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140184231188296
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140183001218576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140183000797256
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140183000797760
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140183000798264
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140183000798768

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7f2fc57358> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.582899
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.551450
grad_step = 000002, loss = 0.528125
grad_step = 000003, loss = 0.503096
grad_step = 000004, loss = 0.475955
grad_step = 000005, loss = 0.447916
grad_step = 000006, loss = 0.424215
grad_step = 000007, loss = 0.408883
grad_step = 000008, loss = 0.389515
grad_step = 000009, loss = 0.367518
grad_step = 000010, loss = 0.347509
grad_step = 000011, loss = 0.331051
grad_step = 000012, loss = 0.315988
grad_step = 000013, loss = 0.302598
grad_step = 000014, loss = 0.290618
grad_step = 000015, loss = 0.280339
grad_step = 000016, loss = 0.270478
grad_step = 000017, loss = 0.259626
grad_step = 000018, loss = 0.248322
grad_step = 000019, loss = 0.237514
grad_step = 000020, loss = 0.227053
grad_step = 000021, loss = 0.216549
grad_step = 000022, loss = 0.206479
grad_step = 000023, loss = 0.197394
grad_step = 000024, loss = 0.188894
grad_step = 000025, loss = 0.180220
grad_step = 000026, loss = 0.171592
grad_step = 000027, loss = 0.163521
grad_step = 000028, loss = 0.155859
grad_step = 000029, loss = 0.148332
grad_step = 000030, loss = 0.140966
grad_step = 000031, loss = 0.133793
grad_step = 000032, loss = 0.126845
grad_step = 000033, loss = 0.120235
grad_step = 000034, loss = 0.113946
grad_step = 000035, loss = 0.107899
grad_step = 000036, loss = 0.102108
grad_step = 000037, loss = 0.096612
grad_step = 000038, loss = 0.091361
grad_step = 000039, loss = 0.086373
grad_step = 000040, loss = 0.081640
grad_step = 000041, loss = 0.077024
grad_step = 000042, loss = 0.072580
grad_step = 000043, loss = 0.068396
grad_step = 000044, loss = 0.064387
grad_step = 000045, loss = 0.060553
grad_step = 000046, loss = 0.056947
grad_step = 000047, loss = 0.053522
grad_step = 000048, loss = 0.050261
grad_step = 000049, loss = 0.047196
grad_step = 000050, loss = 0.044288
grad_step = 000051, loss = 0.041539
grad_step = 000052, loss = 0.038926
grad_step = 000053, loss = 0.036435
grad_step = 000054, loss = 0.034078
grad_step = 000055, loss = 0.031845
grad_step = 000056, loss = 0.029738
grad_step = 000057, loss = 0.027759
grad_step = 000058, loss = 0.025878
grad_step = 000059, loss = 0.024127
grad_step = 000060, loss = 0.022495
grad_step = 000061, loss = 0.020943
grad_step = 000062, loss = 0.019485
grad_step = 000063, loss = 0.018129
grad_step = 000064, loss = 0.016858
grad_step = 000065, loss = 0.015659
grad_step = 000066, loss = 0.014547
grad_step = 000067, loss = 0.013509
grad_step = 000068, loss = 0.012544
grad_step = 000069, loss = 0.011651
grad_step = 000070, loss = 0.010822
grad_step = 000071, loss = 0.010062
grad_step = 000072, loss = 0.009358
grad_step = 000073, loss = 0.008704
grad_step = 000074, loss = 0.008103
grad_step = 000075, loss = 0.007556
grad_step = 000076, loss = 0.007054
grad_step = 000077, loss = 0.006598
grad_step = 000078, loss = 0.006183
grad_step = 000079, loss = 0.005805
grad_step = 000080, loss = 0.005460
grad_step = 000081, loss = 0.005146
grad_step = 000082, loss = 0.004859
grad_step = 000083, loss = 0.004599
grad_step = 000084, loss = 0.004360
grad_step = 000085, loss = 0.004142
grad_step = 000086, loss = 0.003945
grad_step = 000087, loss = 0.003766
grad_step = 000088, loss = 0.003604
grad_step = 000089, loss = 0.003458
grad_step = 000090, loss = 0.003325
grad_step = 000091, loss = 0.003205
grad_step = 000092, loss = 0.003098
grad_step = 000093, loss = 0.003001
grad_step = 000094, loss = 0.002914
grad_step = 000095, loss = 0.002836
grad_step = 000096, loss = 0.002767
grad_step = 000097, loss = 0.002705
grad_step = 000098, loss = 0.002651
grad_step = 000099, loss = 0.002604
grad_step = 000100, loss = 0.002562
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002525
grad_step = 000102, loss = 0.002492
grad_step = 000103, loss = 0.002463
grad_step = 000104, loss = 0.002437
grad_step = 000105, loss = 0.002416
grad_step = 000106, loss = 0.002398
grad_step = 000107, loss = 0.002382
grad_step = 000108, loss = 0.002369
grad_step = 000109, loss = 0.002356
grad_step = 000110, loss = 0.002345
grad_step = 000111, loss = 0.002335
grad_step = 000112, loss = 0.002327
grad_step = 000113, loss = 0.002319
grad_step = 000114, loss = 0.002313
grad_step = 000115, loss = 0.002307
grad_step = 000116, loss = 0.002302
grad_step = 000117, loss = 0.002297
grad_step = 000118, loss = 0.002292
grad_step = 000119, loss = 0.002288
grad_step = 000120, loss = 0.002285
grad_step = 000121, loss = 0.002282
grad_step = 000122, loss = 0.002281
grad_step = 000123, loss = 0.002282
grad_step = 000124, loss = 0.002288
grad_step = 000125, loss = 0.002300
grad_step = 000126, loss = 0.002318
grad_step = 000127, loss = 0.002344
grad_step = 000128, loss = 0.002351
grad_step = 000129, loss = 0.002334
grad_step = 000130, loss = 0.002281
grad_step = 000131, loss = 0.002246
grad_step = 000132, loss = 0.002254
grad_step = 000133, loss = 0.002280
grad_step = 000134, loss = 0.002288
grad_step = 000135, loss = 0.002260
grad_step = 000136, loss = 0.002237
grad_step = 000137, loss = 0.002240
grad_step = 000138, loss = 0.002253
grad_step = 000139, loss = 0.002249
grad_step = 000140, loss = 0.002229
grad_step = 000141, loss = 0.002218
grad_step = 000142, loss = 0.002225
grad_step = 000143, loss = 0.002233
grad_step = 000144, loss = 0.002227
grad_step = 000145, loss = 0.002211
grad_step = 000146, loss = 0.002204
grad_step = 000147, loss = 0.002208
grad_step = 000148, loss = 0.002212
grad_step = 000149, loss = 0.002208
grad_step = 000150, loss = 0.002198
grad_step = 000151, loss = 0.002193
grad_step = 000152, loss = 0.002193
grad_step = 000153, loss = 0.002195
grad_step = 000154, loss = 0.002192
grad_step = 000155, loss = 0.002185
grad_step = 000156, loss = 0.002179
grad_step = 000157, loss = 0.002176
grad_step = 000158, loss = 0.002176
grad_step = 000159, loss = 0.002176
grad_step = 000160, loss = 0.002173
grad_step = 000161, loss = 0.002169
grad_step = 000162, loss = 0.002166
grad_step = 000163, loss = 0.002169
grad_step = 000164, loss = 0.002182
grad_step = 000165, loss = 0.002215
grad_step = 000166, loss = 0.002282
grad_step = 000167, loss = 0.002332
grad_step = 000168, loss = 0.002324
grad_step = 000169, loss = 0.002242
grad_step = 000170, loss = 0.002208
grad_step = 000171, loss = 0.002207
grad_step = 000172, loss = 0.002183
grad_step = 000173, loss = 0.002168
grad_step = 000174, loss = 0.002202
grad_step = 000175, loss = 0.002222
grad_step = 000176, loss = 0.002169
grad_step = 000177, loss = 0.002131
grad_step = 000178, loss = 0.002155
grad_step = 000179, loss = 0.002178
grad_step = 000180, loss = 0.002160
grad_step = 000181, loss = 0.002136
grad_step = 000182, loss = 0.002139
grad_step = 000183, loss = 0.002144
grad_step = 000184, loss = 0.002132
grad_step = 000185, loss = 0.002125
grad_step = 000186, loss = 0.002133
grad_step = 000187, loss = 0.002133
grad_step = 000188, loss = 0.002118
grad_step = 000189, loss = 0.002106
grad_step = 000190, loss = 0.002111
grad_step = 000191, loss = 0.002120
grad_step = 000192, loss = 0.002117
grad_step = 000193, loss = 0.002105
grad_step = 000194, loss = 0.002098
grad_step = 000195, loss = 0.002099
grad_step = 000196, loss = 0.002100
grad_step = 000197, loss = 0.002097
grad_step = 000198, loss = 0.002094
grad_step = 000199, loss = 0.002093
grad_step = 000200, loss = 0.002092
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002089
grad_step = 000202, loss = 0.002083
grad_step = 000203, loss = 0.002078
grad_step = 000204, loss = 0.002075
grad_step = 000205, loss = 0.002075
grad_step = 000206, loss = 0.002076
grad_step = 000207, loss = 0.002076
grad_step = 000208, loss = 0.002075
grad_step = 000209, loss = 0.002072
grad_step = 000210, loss = 0.002072
grad_step = 000211, loss = 0.002078
grad_step = 000212, loss = 0.002104
grad_step = 000213, loss = 0.002171
grad_step = 000214, loss = 0.002192
grad_step = 000215, loss = 0.002182
grad_step = 000216, loss = 0.002138
grad_step = 000217, loss = 0.002189
grad_step = 000218, loss = 0.002225
grad_step = 000219, loss = 0.002127
grad_step = 000220, loss = 0.002145
grad_step = 000221, loss = 0.002134
grad_step = 000222, loss = 0.002050
grad_step = 000223, loss = 0.002093
grad_step = 000224, loss = 0.002145
grad_step = 000225, loss = 0.002095
grad_step = 000226, loss = 0.002093
grad_step = 000227, loss = 0.002124
grad_step = 000228, loss = 0.002077
grad_step = 000229, loss = 0.002041
grad_step = 000230, loss = 0.002074
grad_step = 000231, loss = 0.002081
grad_step = 000232, loss = 0.002055
grad_step = 000233, loss = 0.002054
grad_step = 000234, loss = 0.002070
grad_step = 000235, loss = 0.002057
grad_step = 000236, loss = 0.002029
grad_step = 000237, loss = 0.002038
grad_step = 000238, loss = 0.002053
grad_step = 000239, loss = 0.002039
grad_step = 000240, loss = 0.002030
grad_step = 000241, loss = 0.002041
grad_step = 000242, loss = 0.002042
grad_step = 000243, loss = 0.002028
grad_step = 000244, loss = 0.002020
grad_step = 000245, loss = 0.002025
grad_step = 000246, loss = 0.002027
grad_step = 000247, loss = 0.002018
grad_step = 000248, loss = 0.002012
grad_step = 000249, loss = 0.002017
grad_step = 000250, loss = 0.002020
grad_step = 000251, loss = 0.002018
grad_step = 000252, loss = 0.002013
grad_step = 000253, loss = 0.002014
grad_step = 000254, loss = 0.002025
grad_step = 000255, loss = 0.002046
grad_step = 000256, loss = 0.002101
grad_step = 000257, loss = 0.002164
grad_step = 000258, loss = 0.002261
grad_step = 000259, loss = 0.002109
grad_step = 000260, loss = 0.002037
grad_step = 000261, loss = 0.002090
grad_step = 000262, loss = 0.002057
grad_step = 000263, loss = 0.002062
grad_step = 000264, loss = 0.002090
grad_step = 000265, loss = 0.002015
grad_step = 000266, loss = 0.002029
grad_step = 000267, loss = 0.002090
grad_step = 000268, loss = 0.002023
grad_step = 000269, loss = 0.002001
grad_step = 000270, loss = 0.002035
grad_step = 000271, loss = 0.002040
grad_step = 000272, loss = 0.002011
grad_step = 000273, loss = 0.001995
grad_step = 000274, loss = 0.002025
grad_step = 000275, loss = 0.002017
grad_step = 000276, loss = 0.002004
grad_step = 000277, loss = 0.002026
grad_step = 000278, loss = 0.002054
grad_step = 000279, loss = 0.002072
grad_step = 000280, loss = 0.002105
grad_step = 000281, loss = 0.002177
grad_step = 000282, loss = 0.002276
grad_step = 000283, loss = 0.002356
grad_step = 000284, loss = 0.002278
grad_step = 000285, loss = 0.002134
grad_step = 000286, loss = 0.002021
grad_step = 000287, loss = 0.001998
grad_step = 000288, loss = 0.002062
grad_step = 000289, loss = 0.002126
grad_step = 000290, loss = 0.002078
grad_step = 000291, loss = 0.002013
grad_step = 000292, loss = 0.002001
grad_step = 000293, loss = 0.002021
grad_step = 000294, loss = 0.002023
grad_step = 000295, loss = 0.002012
grad_step = 000296, loss = 0.002009
grad_step = 000297, loss = 0.002001
grad_step = 000298, loss = 0.001991
grad_step = 000299, loss = 0.001995
grad_step = 000300, loss = 0.001997
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001978
grad_step = 000302, loss = 0.001963
grad_step = 000303, loss = 0.001978
grad_step = 000304, loss = 0.001994
grad_step = 000305, loss = 0.001983
grad_step = 000306, loss = 0.001961
grad_step = 000307, loss = 0.001957
grad_step = 000308, loss = 0.001962
grad_step = 000309, loss = 0.001961
grad_step = 000310, loss = 0.001958
grad_step = 000311, loss = 0.001958
grad_step = 000312, loss = 0.001958
grad_step = 000313, loss = 0.001957
grad_step = 000314, loss = 0.001965
grad_step = 000315, loss = 0.001977
grad_step = 000316, loss = 0.001994
grad_step = 000317, loss = 0.002020
grad_step = 000318, loss = 0.002073
grad_step = 000319, loss = 0.002115
grad_step = 000320, loss = 0.002108
grad_step = 000321, loss = 0.002004
grad_step = 000322, loss = 0.001942
grad_step = 000323, loss = 0.001989
grad_step = 000324, loss = 0.002001
grad_step = 000325, loss = 0.001950
grad_step = 000326, loss = 0.001959
grad_step = 000327, loss = 0.001986
grad_step = 000328, loss = 0.001954
grad_step = 000329, loss = 0.001938
grad_step = 000330, loss = 0.001965
grad_step = 000331, loss = 0.001957
grad_step = 000332, loss = 0.001933
grad_step = 000333, loss = 0.001942
grad_step = 000334, loss = 0.001954
grad_step = 000335, loss = 0.001942
grad_step = 000336, loss = 0.001929
grad_step = 000337, loss = 0.001933
grad_step = 000338, loss = 0.001937
grad_step = 000339, loss = 0.001932
grad_step = 000340, loss = 0.001930
grad_step = 000341, loss = 0.001927
grad_step = 000342, loss = 0.001923
grad_step = 000343, loss = 0.001922
grad_step = 000344, loss = 0.001925
grad_step = 000345, loss = 0.001925
grad_step = 000346, loss = 0.001918
grad_step = 000347, loss = 0.001914
grad_step = 000348, loss = 0.001917
grad_step = 000349, loss = 0.001918
grad_step = 000350, loss = 0.001916
grad_step = 000351, loss = 0.001913
grad_step = 000352, loss = 0.001913
grad_step = 000353, loss = 0.001914
grad_step = 000354, loss = 0.001913
grad_step = 000355, loss = 0.001913
grad_step = 000356, loss = 0.001917
grad_step = 000357, loss = 0.001925
grad_step = 000358, loss = 0.001937
grad_step = 000359, loss = 0.001960
grad_step = 000360, loss = 0.001997
grad_step = 000361, loss = 0.002057
grad_step = 000362, loss = 0.002090
grad_step = 000363, loss = 0.002100
grad_step = 000364, loss = 0.002007
grad_step = 000365, loss = 0.001923
grad_step = 000366, loss = 0.001906
grad_step = 000367, loss = 0.001950
grad_step = 000368, loss = 0.001976
grad_step = 000369, loss = 0.001931
grad_step = 000370, loss = 0.001906
grad_step = 000371, loss = 0.001928
grad_step = 000372, loss = 0.001935
grad_step = 000373, loss = 0.001916
grad_step = 000374, loss = 0.001903
grad_step = 000375, loss = 0.001911
grad_step = 000376, loss = 0.001915
grad_step = 000377, loss = 0.001905
grad_step = 000378, loss = 0.001901
grad_step = 000379, loss = 0.001901
grad_step = 000380, loss = 0.001899
grad_step = 000381, loss = 0.001898
grad_step = 000382, loss = 0.001897
grad_step = 000383, loss = 0.001891
grad_step = 000384, loss = 0.001889
grad_step = 000385, loss = 0.001891
grad_step = 000386, loss = 0.001892
grad_step = 000387, loss = 0.001885
grad_step = 000388, loss = 0.001881
grad_step = 000389, loss = 0.001885
grad_step = 000390, loss = 0.001887
grad_step = 000391, loss = 0.001883
grad_step = 000392, loss = 0.001879
grad_step = 000393, loss = 0.001881
grad_step = 000394, loss = 0.001887
grad_step = 000395, loss = 0.001893
grad_step = 000396, loss = 0.001902
grad_step = 000397, loss = 0.001925
grad_step = 000398, loss = 0.001974
grad_step = 000399, loss = 0.002069
grad_step = 000400, loss = 0.002218
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.002454
grad_step = 000402, loss = 0.002599
grad_step = 000403, loss = 0.002542
grad_step = 000404, loss = 0.002151
grad_step = 000405, loss = 0.001888
grad_step = 000406, loss = 0.002008
grad_step = 000407, loss = 0.002173
grad_step = 000408, loss = 0.002102
grad_step = 000409, loss = 0.001975
grad_step = 000410, loss = 0.001979
grad_step = 000411, loss = 0.002005
grad_step = 000412, loss = 0.001953
grad_step = 000413, loss = 0.001958
grad_step = 000414, loss = 0.002013
grad_step = 000415, loss = 0.001948
grad_step = 000416, loss = 0.001882
grad_step = 000417, loss = 0.001922
grad_step = 000418, loss = 0.001956
grad_step = 000419, loss = 0.001904
grad_step = 000420, loss = 0.001862
grad_step = 000421, loss = 0.001910
grad_step = 000422, loss = 0.001922
grad_step = 000423, loss = 0.001865
grad_step = 000424, loss = 0.001870
grad_step = 000425, loss = 0.001904
grad_step = 000426, loss = 0.001873
grad_step = 000427, loss = 0.001851
grad_step = 000428, loss = 0.001876
grad_step = 000429, loss = 0.001876
grad_step = 000430, loss = 0.001851
grad_step = 000431, loss = 0.001852
grad_step = 000432, loss = 0.001865
grad_step = 000433, loss = 0.001855
grad_step = 000434, loss = 0.001846
grad_step = 000435, loss = 0.001852
grad_step = 000436, loss = 0.001850
grad_step = 000437, loss = 0.001842
grad_step = 000438, loss = 0.001845
grad_step = 000439, loss = 0.001847
grad_step = 000440, loss = 0.001838
grad_step = 000441, loss = 0.001834
grad_step = 000442, loss = 0.001840
grad_step = 000443, loss = 0.001839
grad_step = 000444, loss = 0.001832
grad_step = 000445, loss = 0.001832
grad_step = 000446, loss = 0.001834
grad_step = 000447, loss = 0.001830
grad_step = 000448, loss = 0.001826
grad_step = 000449, loss = 0.001827
grad_step = 000450, loss = 0.001828
grad_step = 000451, loss = 0.001825
grad_step = 000452, loss = 0.001824
grad_step = 000453, loss = 0.001825
grad_step = 000454, loss = 0.001825
grad_step = 000455, loss = 0.001824
grad_step = 000456, loss = 0.001824
grad_step = 000457, loss = 0.001828
grad_step = 000458, loss = 0.001835
grad_step = 000459, loss = 0.001846
grad_step = 000460, loss = 0.001871
grad_step = 000461, loss = 0.001911
grad_step = 000462, loss = 0.001972
grad_step = 000463, loss = 0.002013
grad_step = 000464, loss = 0.002013
grad_step = 000465, loss = 0.001921
grad_step = 000466, loss = 0.001833
grad_step = 000467, loss = 0.001826
grad_step = 000468, loss = 0.001874
grad_step = 000469, loss = 0.001880
grad_step = 000470, loss = 0.001834
grad_step = 000471, loss = 0.001825
grad_step = 000472, loss = 0.001849
grad_step = 000473, loss = 0.001838
grad_step = 000474, loss = 0.001811
grad_step = 000475, loss = 0.001817
grad_step = 000476, loss = 0.001836
grad_step = 000477, loss = 0.001825
grad_step = 000478, loss = 0.001802
grad_step = 000479, loss = 0.001804
grad_step = 000480, loss = 0.001818
grad_step = 000481, loss = 0.001816
grad_step = 000482, loss = 0.001802
grad_step = 000483, loss = 0.001796
grad_step = 000484, loss = 0.001801
grad_step = 000485, loss = 0.001806
grad_step = 000486, loss = 0.001799
grad_step = 000487, loss = 0.001790
grad_step = 000488, loss = 0.001789
grad_step = 000489, loss = 0.001794
grad_step = 000490, loss = 0.001795
grad_step = 000491, loss = 0.001790
grad_step = 000492, loss = 0.001783
grad_step = 000493, loss = 0.001781
grad_step = 000494, loss = 0.001782
grad_step = 000495, loss = 0.001784
grad_step = 000496, loss = 0.001783
grad_step = 000497, loss = 0.001781
grad_step = 000498, loss = 0.001780
grad_step = 000499, loss = 0.001779
grad_step = 000500, loss = 0.001778
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001776
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

  date_run                              2020-05-11 12:13:34.302469
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.236916
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 12:13:34.308774
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.123047
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 12:13:34.315685
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.16067
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 12:13:34.321827
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -0.86974
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
0   2020-05-11 12:13:02.429342  ...    mean_absolute_error
1   2020-05-11 12:13:02.434950  ...     mean_squared_error
2   2020-05-11 12:13:02.439203  ...  median_absolute_error
3   2020-05-11 12:13:02.443425  ...               r2_score
4   2020-05-11 12:13:11.332587  ...    mean_absolute_error
5   2020-05-11 12:13:11.336936  ...     mean_squared_error
6   2020-05-11 12:13:11.340707  ...  median_absolute_error
7   2020-05-11 12:13:11.344108  ...               r2_score
8   2020-05-11 12:13:34.302469  ...    mean_absolute_error
9   2020-05-11 12:13:34.308774  ...     mean_squared_error
10  2020-05-11 12:13:34.315685  ...  median_absolute_error
11  2020-05-11 12:13:34.321827  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 468911.69it/s] 81%|████████  | 7995392/9912422 [00:00<00:02, 668173.73it/s]9920512it [00:00, 45065685.52it/s]                           
0it [00:00, ?it/s]32768it [00:00, 631195.13it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 134854.86it/s]1654784it [00:00, 10369171.49it/s]                         
0it [00:00, ?it/s]8192it [00:00, 197400.56it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a3c49b0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a17b10eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a380e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a1490c048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a3c49b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a2cd79e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a3c49b0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a20e296d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a3c8f98> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a20e296d8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7a7a3c8710> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5149c4a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2dd963552603b042e18c35b0258548dba62177e1b3bf7211d14ded4eb0af9552
  Stored in directory: /tmp/pip-ephem-wheel-cache-z_6meugf/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f50e18321d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 10s
 2424832/17464789 [===>..........................] - ETA: 0s 
 7659520/17464789 [============>.................] - ETA: 0s
15015936/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 12:15:00.391867: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 12:15:00.396315: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 12:15:00.396456: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5624d55d4750 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 12:15:00.396469: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8200 - accuracy: 0.4900 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7249 - accuracy: 0.4962
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6532 - accuracy: 0.5009
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 4s - loss: 7.5858 - accuracy: 0.5053
12000/25000 [=============>................] - ETA: 4s - loss: 7.6117 - accuracy: 0.5036
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6006 - accuracy: 0.5043
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6020 - accuracy: 0.5042
15000/25000 [=================>............] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6043 - accuracy: 0.5041
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6134 - accuracy: 0.5035
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6335 - accuracy: 0.5022
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6337 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6338 - accuracy: 0.5021
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6220 - accuracy: 0.5029
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6300 - accuracy: 0.5024
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 9s 380us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 12:15:16.822410
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 12:15:16.822410  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 12:15:22.767094: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 12:15:22.772596: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 12:15:22.772781: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d0b77a75d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 12:15:22.772797: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ff01f42cdd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 966ms/step - loss: 1.4054 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.3561 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff0156cd898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7561 - accuracy: 0.4942
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7165 - accuracy: 0.4967
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
11000/25000 [============>.................] - ETA: 4s - loss: 7.7461 - accuracy: 0.4948
12000/25000 [=============>................] - ETA: 4s - loss: 7.7050 - accuracy: 0.4975
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6491 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 3s - loss: 7.6779 - accuracy: 0.4993
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6922 - accuracy: 0.4983
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6924 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6807 - accuracy: 0.4991
25000/25000 [==============================] - 10s 381us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fefddd47400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<50:08:27, 4.78kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:19:45, 6.78kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<24:46:51, 9.66kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:20:49, 13.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<12:06:27, 19.7kB/s].vector_cache/glove.6B.zip:   1%|          | 8.12M/862M [00:02<8:25:58, 28.1kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:02<5:52:35, 40.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:02<4:05:58, 57.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.2M/862M [00:02<2:51:33, 81.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:02<1:59:36, 117kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:02<1:23:23, 167kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.4M/862M [00:02<58:13, 238kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:03<40:38, 339kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:03<28:24, 482kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:03<19:53, 686kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:03<13:58, 971kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:03<09:51, 1.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:03<08:02, 1.68MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<07:31, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<07:14, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:05<05:29, 2.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<04:00, 3.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:07<18:42, 715kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<14:56, 894kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<10:54, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<10:08, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:09<10:30, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<08:05, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<05:54, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:11<07:42, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<06:43, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<05:02, 2.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:13<06:38, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<05:58, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<04:30, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<06:15, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<05:40, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<06:05, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<05:34, 2.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:17<04:13, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<06:50, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<05:20, 2.42MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:19<03:54, 3.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:21<08:56, 1.44MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<07:33, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<05:36, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:23<06:55, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<04:36, 2.78MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:25<06:12, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<06:56, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<05:24, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<03:55, 3.24MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<12:21, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<09:56, 1.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<07:15, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<08:02, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<08:12, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:22, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:28, 1.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<04:22, 2.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:57, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:41, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:13, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<03:47, 3.28MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<14:08, 878kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<11:09, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:06, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<08:33, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<08:36, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:47, 2.56MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<1:31:08, 135kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<1:04:59, 189kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<45:42, 268kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<34:46, 351kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<26:48, 455kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<19:17, 632kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<13:35, 894kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<23:51, 509kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<17:55, 677kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<12:49, 943kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:46, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<10:44, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:07, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<07:36, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<06:35, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<04:55, 2.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:13, 1.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<04:11, 2.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<03:04, 3.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<1:06:28, 178kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<48:56, 242kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<34:43, 341kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<24:21, 484kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<27:44, 425kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<20:38, 571kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<14:40, 802kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<13:00, 902kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<11:30, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<08:38, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<11:15:29, 17.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<7:53:45, 24.6kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<5:31:11, 35.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<3:53:50, 49.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<2:46:00, 69.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<1:56:39, 99.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:21:26, 141kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<3:08:58, 60.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<2:13:22, 86.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:33:26, 123kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<1:07:53, 169kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<49:51, 230kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<35:21, 323kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<24:51, 459kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<21:48, 522kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<16:25, 692kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<11:45, 965kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<10:52, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<08:44, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:21, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:07, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:15, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:35, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:03, 2.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<08:45, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<07:17, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:22, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<06:21, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<06:42, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:09, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<03:46, 2.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<07:01, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:53, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:24, 2.50MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<05:37, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:09, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<04:51, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<05:09, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<04:42, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<03:34, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<04:36, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<03:27, 3.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<04:58, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:39, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<04:30, 2.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<04:52, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:26, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<04:51, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<05:35, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<04:26, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<04:48, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<04:26, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<03:21, 3.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:48, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:24, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:18, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:06, 3.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<27:31, 378kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<20:20, 512kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<14:26, 720kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<12:28, 829kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<10:49, 955kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<08:05, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:45, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<9:59:32, 17.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<7:00:27, 24.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<4:53:49, 34.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<3:27:20, 49.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<2:27:10, 69.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<1:43:24, 98.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:12:08, 141kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<2:03:54, 81.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<1:27:43, 116kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<1:01:28, 165kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<45:13, 223kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<32:39, 308kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<23:03, 436kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<18:27, 542kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<13:55, 718kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:58, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<09:18, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<07:31, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<06:10, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:10, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:48, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<02:49, 3.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<46:57, 209kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<33:50, 289kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<23:52, 409kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<18:57, 513kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<15:15, 638kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<11:09, 871kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<09:20, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:30, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:27, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<06:03, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:12, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:45, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:27, 2.76MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<07:03, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<05:55, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:22, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:16, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:36, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:24, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:11, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<1:09:20, 135kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<49:27, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<34:46, 269kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<26:26, 353kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<20:24, 457kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<14:43, 632kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<11:45, 787kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<09:10, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:37, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<06:45, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<06:35, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:37, 2.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<1:08:12, 134kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<48:40, 187kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<34:12, 266kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<25:58, 348kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<20:00, 452kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<14:26, 625kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<10:09, 884kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:13<1:11:58, 125kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<51:15, 175kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<36:00, 248kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<27:11, 328kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<20:50, 427kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<14:57, 595kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<10:32, 840kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<13:02, 678kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<10:02, 880kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:12, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<07:05, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<06:43, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:05, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:39, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<08:32, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<06:51, 1.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:00, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<05:31, 1.56MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:44, 3.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<04:43, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<04:22, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<03:16, 2.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:01, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<04:41, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:45, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<02:43, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<09:54, 851kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<07:46, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<05:38, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:54, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<04:59, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:41, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:32, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:48, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<02:45, 2.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<08:30, 967kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:48, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:55, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:19, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:23, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<04:06, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<02:58, 2.73MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<06:46, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<05:34, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:05, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:44, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:07, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:05, 2.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:02, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:25, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:29, 2.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<03:43, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:24, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<02:34, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:37, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:11, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:19, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:24, 3.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<56:59, 136kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<40:31, 191kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<28:39, 270kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<20:02, 383kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<32:24, 237kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<24:15, 316kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<17:17, 443kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<12:09, 627kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<12:18, 618kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<09:24, 808kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<06:43, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<06:27, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<06:03, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:36, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<04:25, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:50, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<02:52, 2.59MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<03:44, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:22, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<02:31, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:29, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:56, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:07, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<03:21, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:04, 2.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<02:18, 3.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:18, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:46, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<02:56, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:09, 3.32MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<05:08, 1.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<04:18, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:11, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:52, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<04:08, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:21, 2.96MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<6:40:05, 17.5kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<4:40:31, 24.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<3:15:47, 35.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<2:17:58, 50.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<1:37:56, 70.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<1:08:43, 100kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<48:04, 143kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<35:33, 193kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<25:33, 268kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<18:00, 379kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<14:07, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<10:27, 648kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:29, 903kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:18, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<53:40, 125kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<38:12, 176kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<26:47, 249kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<20:14, 328kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<15:30, 428kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<11:10, 594kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<08:49, 745kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<06:50, 961kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<04:56, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:57, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:38, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:34, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:09, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:22, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:08, 2.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:29, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<01:58, 3.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:32, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:49, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:49, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:25, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:39, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:50, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:03, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:05, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:28, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:33, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:13, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:28, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:42, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<01:56, 3.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<23:35, 256kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<17:07, 352kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<12:05, 496kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<09:48, 608kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<07:27, 798kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:20, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<05:06, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<04:46, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:37, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<03:27, 1.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<03:39, 1.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:51, 2.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:03, 2.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<42:33, 135kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<30:21, 189kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<21:16, 269kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<16:08, 352kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<12:26, 457kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<08:56, 634kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<06:16, 896kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<09:39, 582kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<07:13, 777kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:08, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:40, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<30:09, 184kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<21:39, 256kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<15:13, 362kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<11:53, 461kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<08:51, 617kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<07:00, 779kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<05:39, 956kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<04:30, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:16, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<03:31, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<03:30, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:42, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:44, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<03:02, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:21, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:42, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<04:04, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<03:23, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:28, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:56, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:06, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:25, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<02:30, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:15, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<01:42, 2.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<02:40, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:05, 2.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:30, 3.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<06:15, 788kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<04:53, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:30, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<03:33, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:59, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:11, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:39, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:20, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<01:45, 2.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:20, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<01:35, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:12, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<02:29, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<01:56, 2.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:23, 3.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<12:44, 360kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<09:22, 488kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<06:38, 685kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:40, 796kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:55, 916kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<03:40, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:35, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<30:21, 147kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<21:35, 206kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<15:08, 292kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<10:35, 414kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<18:18, 239kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<13:41, 320kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<09:46, 447kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<07:27, 578kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<06:05, 708kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:27, 965kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:07, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<07:23, 574kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<05:36, 757kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<04:00, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:45, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:02, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:30, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:35, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:01, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:03, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:23, 2.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:53, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:43, 2.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:17, 3.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:04, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:37, 2.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:09, 3.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<09:10, 418kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<06:48, 562kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:49, 787kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<04:14, 889kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<03:43, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:45, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:58, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<03:06, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:32, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:51, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:08, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:51, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:22, 2.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:48, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:37, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:12, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<01:40, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<01:52, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:29, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<01:34, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<01:27, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:05, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:32, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:25, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:03, 3.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<01:31, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:20, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<00:58, 3.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<02:37, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:09, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<01:49, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:55, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:29, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:34, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:14, 2.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<00:56, 3.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:20, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:15, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<00:56, 3.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:19, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:13, 2.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<00:55, 3.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:19, 2.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<01:12, 2.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<00:54, 3.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:17, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<01:11, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<00:53, 3.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:16, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:09, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<00:51, 3.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:14, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:07, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<00:51, 3.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:12, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:22, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:04, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:46, 3.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<2:26:23, 17.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<1:42:28, 24.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<1:11:03, 35.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<49:36, 49.6kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<35:10, 69.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<24:36, 99.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<17:15, 139kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<12:17, 194kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<08:34, 276kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<06:27, 360kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<04:58, 466kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<03:35, 644kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:49, 800kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:11, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:34, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:35, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:33, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:10, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:51, 2.52MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:17, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:07, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:49, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:03, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:08, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:54, 2.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:38, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<14:37, 135kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<11:24, 174kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<07:45, 247kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<12:55, 148kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<09:12, 207kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<06:24, 294kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<04:50, 381kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<03:45, 491kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:42, 677kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:51, 957kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<1:44:04, 17.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<1:12:46, 24.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<50:14, 34.7kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<34:50, 49.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<24:41, 69.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<17:13, 98.1kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<11:57, 137kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<08:30, 192kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<05:53, 272kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<04:24, 357kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<03:22, 463kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<02:25, 643kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:52, 798kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:27, 1.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:02, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:38, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<00:48, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:37, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:38, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:34, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:25, 3.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:34, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<00:39, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:30, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:29, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:21, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:26, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:18, 3.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:51, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:42, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:30, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:21, 2.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:26, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<01:16, 689kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:54, 958kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:44, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:35, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:25, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:27, 1.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:28, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:14, 2.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<05:03, 134kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<03:33, 189kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<02:25, 268kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:36, 381kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<04:45, 128kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<03:21, 180kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<02:15, 255kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<01:36, 336kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<01:10, 457kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:47, 643kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:37, 754kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:28, 971kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:19, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:18, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:17, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:08, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:12, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:10, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:08, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 2.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:01, 3.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 3.14MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 738/400000 [00:00<00:54, 7370.47it/s]  0%|          | 1518/400000 [00:00<00:53, 7492.94it/s]  1%|          | 2299/400000 [00:00<00:52, 7584.57it/s]  1%|          | 3096/400000 [00:00<00:51, 7693.46it/s]  1%|          | 3888/400000 [00:00<00:51, 7759.70it/s]  1%|          | 4658/400000 [00:00<00:51, 7741.37it/s]  1%|▏         | 5415/400000 [00:00<00:51, 7687.38it/s]  2%|▏         | 6217/400000 [00:00<00:50, 7782.31it/s]  2%|▏         | 7007/400000 [00:00<00:50, 7813.89it/s]  2%|▏         | 7772/400000 [00:01<00:50, 7762.27it/s]  2%|▏         | 8528/400000 [00:01<00:51, 7624.54it/s]  2%|▏         | 9317/400000 [00:01<00:50, 7701.55it/s]  3%|▎         | 10120/400000 [00:01<00:50, 7796.80it/s]  3%|▎         | 10923/400000 [00:01<00:49, 7864.64it/s]  3%|▎         | 11718/400000 [00:01<00:49, 7888.47it/s]  3%|▎         | 12504/400000 [00:01<00:50, 7707.12it/s]  3%|▎         | 13274/400000 [00:01<00:50, 7695.68it/s]  4%|▎         | 14063/400000 [00:01<00:49, 7750.74it/s]  4%|▎         | 14838/400000 [00:01<00:49, 7750.04it/s]  4%|▍         | 15624/400000 [00:02<00:49, 7781.97it/s]  4%|▍         | 16402/400000 [00:02<00:49, 7777.20it/s]  4%|▍         | 17197/400000 [00:02<00:48, 7825.90it/s]  4%|▍         | 17998/400000 [00:02<00:48, 7878.78it/s]  5%|▍         | 18786/400000 [00:02<00:48, 7856.40it/s]  5%|▍         | 19572/400000 [00:02<00:48, 7807.04it/s]  5%|▌         | 20353/400000 [00:02<00:48, 7751.57it/s]  5%|▌         | 21131/400000 [00:02<00:48, 7759.34it/s]  5%|▌         | 21910/400000 [00:02<00:48, 7766.80it/s]  6%|▌         | 22698/400000 [00:02<00:48, 7798.18it/s]  6%|▌         | 23478/400000 [00:03<00:48, 7756.93it/s]  6%|▌         | 24254/400000 [00:03<00:48, 7750.61it/s]  6%|▋         | 25030/400000 [00:03<00:48, 7747.82it/s]  6%|▋         | 25842/400000 [00:03<00:47, 7855.19it/s]  7%|▋         | 26641/400000 [00:03<00:47, 7894.09it/s]  7%|▋         | 27450/400000 [00:03<00:46, 7949.72it/s]  7%|▋         | 28246/400000 [00:03<00:47, 7808.15it/s]  7%|▋         | 29028/400000 [00:03<00:47, 7762.54it/s]  7%|▋         | 29817/400000 [00:03<00:47, 7799.98it/s]  8%|▊         | 30611/400000 [00:03<00:47, 7840.08it/s]  8%|▊         | 31400/400000 [00:04<00:46, 7854.87it/s]  8%|▊         | 32186/400000 [00:04<00:47, 7699.27it/s]  8%|▊         | 32971/400000 [00:04<00:47, 7742.60it/s]  8%|▊         | 33746/400000 [00:04<00:48, 7512.61it/s]  9%|▊         | 34500/400000 [00:04<00:48, 7474.39it/s]  9%|▉         | 35257/400000 [00:04<00:48, 7502.57it/s]  9%|▉         | 36009/400000 [00:04<00:48, 7481.51it/s]  9%|▉         | 36758/400000 [00:04<00:49, 7376.22it/s]  9%|▉         | 37517/400000 [00:04<00:48, 7439.03it/s] 10%|▉         | 38262/400000 [00:04<00:48, 7433.54it/s] 10%|▉         | 39051/400000 [00:05<00:47, 7561.29it/s] 10%|▉         | 39825/400000 [00:05<00:47, 7613.96it/s] 10%|█         | 40606/400000 [00:05<00:46, 7669.01it/s] 10%|█         | 41402/400000 [00:05<00:46, 7753.20it/s] 11%|█         | 42178/400000 [00:05<00:46, 7730.49it/s] 11%|█         | 42964/400000 [00:05<00:45, 7766.81it/s] 11%|█         | 43742/400000 [00:05<00:46, 7739.12it/s] 11%|█         | 44535/400000 [00:05<00:45, 7794.51it/s] 11%|█▏        | 45346/400000 [00:05<00:44, 7884.01it/s] 12%|█▏        | 46164/400000 [00:05<00:44, 7969.88it/s] 12%|█▏        | 46962/400000 [00:06<00:45, 7794.23it/s] 12%|█▏        | 47743/400000 [00:06<00:46, 7616.51it/s] 12%|█▏        | 48525/400000 [00:06<00:45, 7675.24it/s] 12%|█▏        | 49345/400000 [00:06<00:44, 7824.47it/s] 13%|█▎        | 50152/400000 [00:06<00:44, 7895.38it/s] 13%|█▎        | 50943/400000 [00:06<00:44, 7889.86it/s] 13%|█▎        | 51733/400000 [00:06<00:44, 7849.34it/s] 13%|█▎        | 52536/400000 [00:06<00:43, 7901.82it/s] 13%|█▎        | 53344/400000 [00:06<00:43, 7954.37it/s] 14%|█▎        | 54168/400000 [00:06<00:43, 8036.28it/s] 14%|█▎        | 54977/400000 [00:07<00:42, 8051.18it/s] 14%|█▍        | 55783/400000 [00:07<00:43, 7921.49it/s] 14%|█▍        | 56576/400000 [00:07<00:43, 7843.03it/s] 14%|█▍        | 57366/400000 [00:07<00:43, 7857.25it/s] 15%|█▍        | 58161/400000 [00:07<00:43, 7884.77it/s] 15%|█▍        | 58977/400000 [00:07<00:42, 7961.56it/s] 15%|█▍        | 59774/400000 [00:07<00:43, 7870.42it/s] 15%|█▌        | 60562/400000 [00:07<00:43, 7849.46it/s] 15%|█▌        | 61348/400000 [00:07<00:43, 7850.55it/s] 16%|█▌        | 62134/400000 [00:07<00:43, 7752.06it/s] 16%|█▌        | 62915/400000 [00:08<00:43, 7767.83it/s] 16%|█▌        | 63723/400000 [00:08<00:42, 7855.69it/s] 16%|█▌        | 64510/400000 [00:08<00:43, 7736.76it/s] 16%|█▋        | 65310/400000 [00:08<00:42, 7813.17it/s] 17%|█▋        | 66117/400000 [00:08<00:42, 7888.30it/s] 17%|█▋        | 66934/400000 [00:08<00:41, 7968.33it/s] 17%|█▋        | 67732/400000 [00:08<00:41, 7970.32it/s] 17%|█▋        | 68530/400000 [00:08<00:41, 7948.43it/s] 17%|█▋        | 69326/400000 [00:08<00:41, 7946.71it/s] 18%|█▊        | 70121/400000 [00:09<00:41, 7941.91it/s] 18%|█▊        | 70924/400000 [00:09<00:41, 7965.06it/s] 18%|█▊        | 71721/400000 [00:09<00:41, 7923.14it/s] 18%|█▊        | 72514/400000 [00:09<00:41, 7852.25it/s] 18%|█▊        | 73300/400000 [00:09<00:42, 7748.90it/s] 19%|█▊        | 74076/400000 [00:09<00:42, 7713.07it/s] 19%|█▊        | 74884/400000 [00:09<00:41, 7818.36it/s] 19%|█▉        | 75669/400000 [00:09<00:41, 7827.54it/s] 19%|█▉        | 76453/400000 [00:09<00:42, 7554.23it/s] 19%|█▉        | 77223/400000 [00:09<00:42, 7596.41it/s] 20%|█▉        | 78017/400000 [00:10<00:41, 7694.47it/s] 20%|█▉        | 78790/400000 [00:10<00:41, 7703.67it/s] 20%|█▉        | 79569/400000 [00:10<00:41, 7727.84it/s] 20%|██        | 80343/400000 [00:10<00:41, 7683.74it/s] 20%|██        | 81125/400000 [00:10<00:41, 7722.69it/s] 20%|██        | 81903/400000 [00:10<00:41, 7737.55it/s] 21%|██        | 82678/400000 [00:10<00:41, 7709.27it/s] 21%|██        | 83481/400000 [00:10<00:40, 7800.15it/s] 21%|██        | 84262/400000 [00:10<00:40, 7785.59it/s] 21%|██▏       | 85042/400000 [00:10<00:40, 7788.40it/s] 21%|██▏       | 85822/400000 [00:11<00:41, 7635.80it/s] 22%|██▏       | 86587/400000 [00:11<00:41, 7529.15it/s] 22%|██▏       | 87350/400000 [00:11<00:41, 7559.05it/s] 22%|██▏       | 88107/400000 [00:11<00:41, 7551.37it/s] 22%|██▏       | 88891/400000 [00:11<00:40, 7634.51it/s] 22%|██▏       | 89655/400000 [00:11<00:40, 7629.12it/s] 23%|██▎       | 90443/400000 [00:11<00:40, 7701.45it/s] 23%|██▎       | 91224/400000 [00:11<00:39, 7732.57it/s] 23%|██▎       | 91998/400000 [00:11<00:40, 7595.32it/s] 23%|██▎       | 92759/400000 [00:11<00:40, 7576.31it/s] 23%|██▎       | 93518/400000 [00:12<00:40, 7571.58it/s] 24%|██▎       | 94305/400000 [00:12<00:39, 7656.78it/s] 24%|██▍       | 95084/400000 [00:12<00:39, 7696.10it/s] 24%|██▍       | 95855/400000 [00:12<00:39, 7608.52it/s] 24%|██▍       | 96658/400000 [00:12<00:39, 7728.10it/s] 24%|██▍       | 97466/400000 [00:12<00:38, 7827.43it/s] 25%|██▍       | 98280/400000 [00:12<00:38, 7916.77it/s] 25%|██▍       | 99073/400000 [00:12<00:38, 7777.58it/s] 25%|██▍       | 99852/400000 [00:12<00:39, 7658.74it/s] 25%|██▌       | 100639/400000 [00:12<00:38, 7720.79it/s] 25%|██▌       | 101421/400000 [00:13<00:38, 7749.46it/s] 26%|██▌       | 102218/400000 [00:13<00:38, 7813.71it/s] 26%|██▌       | 103008/400000 [00:13<00:37, 7838.14it/s] 26%|██▌       | 103793/400000 [00:13<00:38, 7791.32it/s] 26%|██▌       | 104580/400000 [00:13<00:37, 7813.30it/s] 26%|██▋       | 105376/400000 [00:13<00:37, 7855.78it/s] 27%|██▋       | 106162/400000 [00:13<00:37, 7764.85it/s] 27%|██▋       | 106962/400000 [00:13<00:37, 7833.64it/s] 27%|██▋       | 107746/400000 [00:13<00:37, 7767.98it/s] 27%|██▋       | 108525/400000 [00:13<00:37, 7774.05it/s] 27%|██▋       | 109303/400000 [00:14<00:37, 7767.04it/s] 28%|██▊       | 110091/400000 [00:14<00:37, 7799.23it/s] 28%|██▊       | 110872/400000 [00:14<00:37, 7801.88it/s] 28%|██▊       | 111653/400000 [00:14<00:37, 7708.72it/s] 28%|██▊       | 112425/400000 [00:14<00:37, 7669.00it/s] 28%|██▊       | 113193/400000 [00:14<00:37, 7664.07it/s] 28%|██▊       | 113960/400000 [00:14<00:37, 7656.75it/s] 29%|██▊       | 114740/400000 [00:14<00:37, 7697.04it/s] 29%|██▉       | 115510/400000 [00:14<00:37, 7645.56it/s] 29%|██▉       | 116275/400000 [00:14<00:38, 7370.90it/s] 29%|██▉       | 117016/400000 [00:15<00:38, 7380.45it/s] 29%|██▉       | 117794/400000 [00:15<00:37, 7494.52it/s] 30%|██▉       | 118583/400000 [00:15<00:36, 7606.67it/s] 30%|██▉       | 119346/400000 [00:15<00:36, 7612.47it/s] 30%|███       | 120158/400000 [00:15<00:36, 7757.55it/s] 30%|███       | 120974/400000 [00:15<00:35, 7870.19it/s] 30%|███       | 121774/400000 [00:15<00:35, 7907.02it/s] 31%|███       | 122596/400000 [00:15<00:34, 7998.24it/s] 31%|███       | 123397/400000 [00:15<00:34, 7934.75it/s] 31%|███       | 124192/400000 [00:16<00:35, 7856.57it/s] 31%|███       | 124979/400000 [00:16<00:35, 7804.99it/s] 31%|███▏      | 125761/400000 [00:16<00:35, 7758.02it/s] 32%|███▏      | 126539/400000 [00:16<00:35, 7762.42it/s] 32%|███▏      | 127316/400000 [00:16<00:35, 7764.57it/s] 32%|███▏      | 128127/400000 [00:16<00:34, 7864.41it/s] 32%|███▏      | 128914/400000 [00:16<00:34, 7846.31it/s] 32%|███▏      | 129701/400000 [00:16<00:34, 7851.90it/s] 33%|███▎      | 130507/400000 [00:16<00:34, 7911.26it/s] 33%|███▎      | 131299/400000 [00:16<00:33, 7904.88it/s] 33%|███▎      | 132096/400000 [00:17<00:33, 7922.60it/s] 33%|███▎      | 132901/400000 [00:17<00:33, 7959.82it/s] 33%|███▎      | 133698/400000 [00:17<00:33, 7949.00it/s] 34%|███▎      | 134502/400000 [00:17<00:33, 7971.54it/s] 34%|███▍      | 135300/400000 [00:17<00:33, 7861.53it/s] 34%|███▍      | 136105/400000 [00:17<00:33, 7915.68it/s] 34%|███▍      | 136915/400000 [00:17<00:33, 7968.58it/s] 34%|███▍      | 137720/400000 [00:17<00:32, 7991.60it/s] 35%|███▍      | 138526/400000 [00:17<00:32, 8010.83it/s] 35%|███▍      | 139328/400000 [00:17<00:32, 7954.84it/s] 35%|███▌      | 140124/400000 [00:18<00:32, 7954.86it/s] 35%|███▌      | 140933/400000 [00:18<00:32, 7993.17it/s] 35%|███▌      | 141736/400000 [00:18<00:32, 8001.99it/s] 36%|███▌      | 142537/400000 [00:18<00:32, 7894.06it/s] 36%|███▌      | 143327/400000 [00:18<00:32, 7805.62it/s] 36%|███▌      | 144118/400000 [00:18<00:32, 7836.51it/s] 36%|███▌      | 144903/400000 [00:18<00:32, 7798.54it/s] 36%|███▋      | 145701/400000 [00:18<00:32, 7851.50it/s] 37%|███▋      | 146515/400000 [00:18<00:31, 7935.34it/s] 37%|███▋      | 147312/400000 [00:18<00:31, 7945.40it/s] 37%|███▋      | 148107/400000 [00:19<00:32, 7844.25it/s] 37%|███▋      | 148906/400000 [00:19<00:31, 7885.81it/s] 37%|███▋      | 149705/400000 [00:19<00:31, 7913.61it/s] 38%|███▊      | 150518/400000 [00:19<00:31, 7975.23it/s] 38%|███▊      | 151316/400000 [00:19<00:31, 7901.26it/s] 38%|███▊      | 152107/400000 [00:19<00:31, 7849.01it/s] 38%|███▊      | 152893/400000 [00:19<00:31, 7806.74it/s] 38%|███▊      | 153683/400000 [00:19<00:31, 7834.25it/s] 39%|███▊      | 154478/400000 [00:19<00:31, 7867.00it/s] 39%|███▉      | 155265/400000 [00:19<00:31, 7825.24it/s] 39%|███▉      | 156049/400000 [00:20<00:31, 7827.70it/s] 39%|███▉      | 156832/400000 [00:20<00:31, 7779.35it/s] 39%|███▉      | 157611/400000 [00:20<00:31, 7708.76it/s] 40%|███▉      | 158411/400000 [00:20<00:31, 7793.08it/s] 40%|███▉      | 159191/400000 [00:20<00:30, 7787.01it/s] 40%|███▉      | 159998/400000 [00:20<00:30, 7865.82it/s] 40%|████      | 160812/400000 [00:20<00:30, 7944.38it/s] 40%|████      | 161613/400000 [00:20<00:29, 7962.04it/s] 41%|████      | 162440/400000 [00:20<00:29, 8051.55it/s] 41%|████      | 163246/400000 [00:20<00:29, 8008.11it/s] 41%|████      | 164062/400000 [00:21<00:29, 8050.97it/s] 41%|████      | 164868/400000 [00:21<00:29, 8002.81it/s] 41%|████▏     | 165669/400000 [00:21<00:29, 7962.63it/s] 42%|████▏     | 166466/400000 [00:21<00:29, 7945.01it/s] 42%|████▏     | 167261/400000 [00:21<00:29, 7927.44it/s] 42%|████▏     | 168056/400000 [00:21<00:29, 7932.18it/s] 42%|████▏     | 168866/400000 [00:21<00:28, 7979.55it/s] 42%|████▏     | 169675/400000 [00:21<00:28, 8011.25it/s] 43%|████▎     | 170491/400000 [00:21<00:28, 8051.95it/s] 43%|████▎     | 171297/400000 [00:21<00:28, 7971.79it/s] 43%|████▎     | 172095/400000 [00:22<00:28, 7907.27it/s] 43%|████▎     | 172887/400000 [00:22<00:28, 7872.85it/s] 43%|████▎     | 173675/400000 [00:22<00:28, 7815.14it/s] 44%|████▎     | 174457/400000 [00:22<00:28, 7805.73it/s] 44%|████▍     | 175238/400000 [00:22<00:28, 7755.52it/s] 44%|████▍     | 176033/400000 [00:22<00:28, 7811.90it/s] 44%|████▍     | 176827/400000 [00:22<00:28, 7848.69it/s] 44%|████▍     | 177613/400000 [00:22<00:28, 7676.13it/s] 45%|████▍     | 178382/400000 [00:22<00:28, 7666.32it/s] 45%|████▍     | 179151/400000 [00:22<00:28, 7672.69it/s] 45%|████▍     | 179919/400000 [00:23<00:28, 7645.86it/s] 45%|████▌     | 180723/400000 [00:23<00:28, 7759.61it/s] 45%|████▌     | 181500/400000 [00:23<00:28, 7663.92it/s] 46%|████▌     | 182268/400000 [00:23<00:28, 7657.10it/s] 46%|████▌     | 183035/400000 [00:23<00:28, 7659.83it/s] 46%|████▌     | 183810/400000 [00:23<00:28, 7686.09it/s] 46%|████▌     | 184592/400000 [00:23<00:27, 7723.63it/s] 46%|████▋     | 185366/400000 [00:23<00:27, 7727.60it/s] 47%|████▋     | 186142/400000 [00:23<00:27, 7735.82it/s] 47%|████▋     | 186916/400000 [00:23<00:27, 7707.92it/s] 47%|████▋     | 187689/400000 [00:24<00:27, 7714.35it/s] 47%|████▋     | 188461/400000 [00:24<00:28, 7431.83it/s] 47%|████▋     | 189227/400000 [00:24<00:28, 7497.00it/s] 48%|████▊     | 190005/400000 [00:24<00:27, 7577.41it/s] 48%|████▊     | 190765/400000 [00:24<00:27, 7553.12it/s] 48%|████▊     | 191551/400000 [00:24<00:27, 7640.13it/s] 48%|████▊     | 192341/400000 [00:24<00:26, 7715.37it/s] 48%|████▊     | 193114/400000 [00:24<00:26, 7716.58it/s] 48%|████▊     | 193887/400000 [00:24<00:26, 7697.37it/s] 49%|████▊     | 194658/400000 [00:25<00:26, 7607.05it/s] 49%|████▉     | 195420/400000 [00:25<00:27, 7548.85it/s] 49%|████▉     | 196177/400000 [00:25<00:26, 7553.38it/s] 49%|████▉     | 196933/400000 [00:25<00:26, 7538.09it/s] 49%|████▉     | 197688/400000 [00:25<00:27, 7333.71it/s] 50%|████▉     | 198423/400000 [00:25<00:28, 6992.92it/s] 50%|████▉     | 199162/400000 [00:25<00:28, 7106.57it/s] 50%|████▉     | 199947/400000 [00:25<00:27, 7313.19it/s] 50%|█████     | 200720/400000 [00:25<00:26, 7432.73it/s] 50%|█████     | 201502/400000 [00:25<00:26, 7544.36it/s] 51%|█████     | 202260/400000 [00:26<00:26, 7480.08it/s] 51%|█████     | 203014/400000 [00:26<00:26, 7493.97it/s] 51%|█████     | 203780/400000 [00:26<00:26, 7541.19it/s] 51%|█████     | 204552/400000 [00:26<00:25, 7592.66it/s] 51%|█████▏    | 205321/400000 [00:26<00:25, 7620.46it/s] 52%|█████▏    | 206084/400000 [00:26<00:25, 7580.35it/s] 52%|█████▏    | 206844/400000 [00:26<00:25, 7583.71it/s] 52%|█████▏    | 207612/400000 [00:26<00:25, 7611.18it/s] 52%|█████▏    | 208394/400000 [00:26<00:24, 7670.84it/s] 52%|█████▏    | 209166/400000 [00:26<00:24, 7685.31it/s] 52%|█████▏    | 209936/400000 [00:27<00:24, 7689.67it/s] 53%|█████▎    | 210739/400000 [00:27<00:24, 7786.47it/s] 53%|█████▎    | 211537/400000 [00:27<00:24, 7841.08it/s] 53%|█████▎    | 212322/400000 [00:27<00:24, 7764.16it/s] 53%|█████▎    | 213105/400000 [00:27<00:24, 7783.54it/s] 53%|█████▎    | 213884/400000 [00:27<00:23, 7772.53it/s] 54%|█████▎    | 214682/400000 [00:27<00:23, 7831.02it/s] 54%|█████▍    | 215466/400000 [00:27<00:23, 7821.29it/s] 54%|█████▍    | 216250/400000 [00:27<00:23, 7825.15it/s] 54%|█████▍    | 217044/400000 [00:27<00:23, 7858.18it/s] 54%|█████▍    | 217830/400000 [00:28<00:23, 7801.77it/s] 55%|█████▍    | 218621/400000 [00:28<00:23, 7832.18it/s] 55%|█████▍    | 219425/400000 [00:28<00:22, 7892.96it/s] 55%|█████▌    | 220215/400000 [00:28<00:22, 7832.40it/s] 55%|█████▌    | 221011/400000 [00:28<00:22, 7869.80it/s] 55%|█████▌    | 221799/400000 [00:28<00:22, 7848.46it/s] 56%|█████▌    | 222588/400000 [00:28<00:22, 7860.81it/s] 56%|█████▌    | 223384/400000 [00:28<00:22, 7888.31it/s] 56%|█████▌    | 224179/400000 [00:28<00:22, 7904.95it/s] 56%|█████▌    | 224970/400000 [00:28<00:22, 7812.78it/s] 56%|█████▋    | 225752/400000 [00:29<00:22, 7744.77it/s] 57%|█████▋    | 226538/400000 [00:29<00:22, 7777.55it/s] 57%|█████▋    | 227317/400000 [00:29<00:22, 7731.83it/s] 57%|█████▋    | 228091/400000 [00:29<00:22, 7724.78it/s] 57%|█████▋    | 228874/400000 [00:29<00:22, 7754.42it/s] 57%|█████▋    | 229652/400000 [00:29<00:21, 7760.90it/s] 58%|█████▊    | 230429/400000 [00:29<00:21, 7763.22it/s] 58%|█████▊    | 231212/400000 [00:29<00:21, 7782.19it/s] 58%|█████▊    | 231991/400000 [00:29<00:21, 7778.46it/s] 58%|█████▊    | 232773/400000 [00:29<00:21, 7790.38it/s] 58%|█████▊    | 233553/400000 [00:30<00:21, 7694.65it/s] 59%|█████▊    | 234323/400000 [00:30<00:21, 7614.45it/s] 59%|█████▉    | 235085/400000 [00:30<00:22, 7469.98it/s] 59%|█████▉    | 235833/400000 [00:30<00:22, 7373.98it/s] 59%|█████▉    | 236607/400000 [00:30<00:21, 7478.48it/s] 59%|█████▉    | 237362/400000 [00:30<00:21, 7498.14it/s] 60%|█████▉    | 238136/400000 [00:30<00:21, 7568.18it/s] 60%|█████▉    | 238894/400000 [00:30<00:21, 7508.00it/s] 60%|█████▉    | 239682/400000 [00:30<00:21, 7615.74it/s] 60%|██████    | 240482/400000 [00:30<00:20, 7725.17it/s] 60%|██████    | 241256/400000 [00:31<00:20, 7720.40it/s] 61%|██████    | 242031/400000 [00:31<00:20, 7726.24it/s] 61%|██████    | 242807/400000 [00:31<00:20, 7733.16it/s] 61%|██████    | 243604/400000 [00:31<00:20, 7801.08it/s] 61%|██████    | 244385/400000 [00:31<00:20, 7718.66it/s] 61%|██████▏   | 245158/400000 [00:31<00:20, 7711.73it/s] 61%|██████▏   | 245943/400000 [00:31<00:19, 7752.00it/s] 62%|██████▏   | 246726/400000 [00:31<00:19, 7773.25it/s] 62%|██████▏   | 247517/400000 [00:31<00:19, 7812.86it/s] 62%|██████▏   | 248319/400000 [00:31<00:19, 7871.74it/s] 62%|██████▏   | 249107/400000 [00:32<00:19, 7757.53it/s] 62%|██████▏   | 249898/400000 [00:32<00:19, 7800.21it/s] 63%|██████▎   | 250695/400000 [00:32<00:19, 7850.37it/s] 63%|██████▎   | 251481/400000 [00:32<00:19, 7759.97it/s] 63%|██████▎   | 252258/400000 [00:32<00:19, 7681.10it/s] 63%|██████▎   | 253033/400000 [00:32<00:19, 7699.66it/s] 63%|██████▎   | 253810/400000 [00:32<00:18, 7718.18it/s] 64%|██████▎   | 254585/400000 [00:32<00:18, 7726.62it/s] 64%|██████▍   | 255377/400000 [00:32<00:18, 7781.11it/s] 64%|██████▍   | 256166/400000 [00:33<00:18, 7810.98it/s] 64%|██████▍   | 256961/400000 [00:33<00:18, 7851.80it/s] 64%|██████▍   | 257747/400000 [00:33<00:18, 7806.50it/s] 65%|██████▍   | 258528/400000 [00:33<00:18, 7807.21it/s] 65%|██████▍   | 259309/400000 [00:33<00:18, 7766.89it/s] 65%|██████▌   | 260108/400000 [00:33<00:17, 7831.82it/s] 65%|██████▌   | 260892/400000 [00:33<00:17, 7790.50it/s] 65%|██████▌   | 261672/400000 [00:33<00:17, 7694.35it/s] 66%|██████▌   | 262468/400000 [00:33<00:17, 7770.20it/s] 66%|██████▌   | 263265/400000 [00:33<00:17, 7826.99it/s] 66%|██████▌   | 264055/400000 [00:34<00:17, 7847.98it/s] 66%|██████▌   | 264841/400000 [00:34<00:17, 7773.69it/s] 66%|██████▋   | 265619/400000 [00:34<00:17, 7711.27it/s] 67%|██████▋   | 266391/400000 [00:34<00:17, 7649.91it/s] 67%|██████▋   | 267157/400000 [00:34<00:17, 7611.86it/s] 67%|██████▋   | 267922/400000 [00:34<00:17, 7621.88it/s] 67%|██████▋   | 268685/400000 [00:34<00:17, 7500.27it/s] 67%|██████▋   | 269436/400000 [00:34<00:17, 7412.72it/s] 68%|██████▊   | 270178/400000 [00:34<00:17, 7350.69it/s] 68%|██████▊   | 270927/400000 [00:34<00:17, 7390.12it/s] 68%|██████▊   | 271669/400000 [00:35<00:17, 7399.02it/s] 68%|██████▊   | 272440/400000 [00:35<00:17, 7488.81it/s] 68%|██████▊   | 273229/400000 [00:35<00:16, 7604.21it/s] 69%|██████▊   | 274017/400000 [00:35<00:16, 7684.24it/s] 69%|██████▊   | 274790/400000 [00:35<00:16, 7695.68it/s] 69%|██████▉   | 275561/400000 [00:35<00:16, 7660.78it/s] 69%|██████▉   | 276338/400000 [00:35<00:16, 7690.78it/s] 69%|██████▉   | 277108/400000 [00:35<00:16, 7585.90it/s] 69%|██████▉   | 277868/400000 [00:35<00:16, 7421.87it/s] 70%|██████▉   | 278638/400000 [00:35<00:16, 7502.52it/s] 70%|██████▉   | 279399/400000 [00:36<00:16, 7532.94it/s] 70%|███████   | 280167/400000 [00:36<00:15, 7574.99it/s] 70%|███████   | 280928/400000 [00:36<00:15, 7584.51it/s] 70%|███████   | 281687/400000 [00:36<00:15, 7577.71it/s] 71%|███████   | 282460/400000 [00:36<00:15, 7621.83it/s] 71%|███████   | 283230/400000 [00:36<00:15, 7643.14it/s] 71%|███████   | 284004/400000 [00:36<00:15, 7669.72it/s] 71%|███████   | 284772/400000 [00:36<00:15, 7616.19it/s] 71%|███████▏  | 285562/400000 [00:36<00:14, 7698.20it/s] 72%|███████▏  | 286333/400000 [00:36<00:15, 7456.36it/s] 72%|███████▏  | 287109/400000 [00:37<00:14, 7542.80it/s] 72%|███████▏  | 287872/400000 [00:37<00:14, 7566.41it/s] 72%|███████▏  | 288630/400000 [00:37<00:14, 7499.60it/s] 72%|███████▏  | 289381/400000 [00:37<00:14, 7464.91it/s] 73%|███████▎  | 290129/400000 [00:37<00:14, 7425.12it/s] 73%|███████▎  | 290929/400000 [00:37<00:14, 7586.43it/s] 73%|███████▎  | 291714/400000 [00:37<00:14, 7661.08it/s] 73%|███████▎  | 292482/400000 [00:37<00:14, 7583.67it/s] 73%|███████▎  | 293242/400000 [00:37<00:14, 7456.29it/s] 74%|███████▎  | 294008/400000 [00:37<00:14, 7513.93it/s] 74%|███████▎  | 294761/400000 [00:38<00:14, 7512.78it/s] 74%|███████▍  | 295559/400000 [00:38<00:13, 7647.08it/s] 74%|███████▍  | 296325/400000 [00:38<00:13, 7598.94it/s] 74%|███████▍  | 297086/400000 [00:38<00:13, 7440.59it/s] 74%|███████▍  | 297870/400000 [00:38<00:13, 7554.47it/s] 75%|███████▍  | 298640/400000 [00:38<00:13, 7588.59it/s] 75%|███████▍  | 299434/400000 [00:38<00:13, 7687.90it/s] 75%|███████▌  | 300204/400000 [00:38<00:13, 7668.88it/s] 75%|███████▌  | 300972/400000 [00:38<00:12, 7617.89it/s] 75%|███████▌  | 301735/400000 [00:38<00:12, 7577.15it/s] 76%|███████▌  | 302494/400000 [00:39<00:12, 7552.34it/s] 76%|███████▌  | 303280/400000 [00:39<00:12, 7639.81it/s] 76%|███████▌  | 304045/400000 [00:39<00:12, 7610.85it/s] 76%|███████▌  | 304807/400000 [00:39<00:12, 7584.59it/s] 76%|███████▋  | 305566/400000 [00:39<00:12, 7534.78it/s] 77%|███████▋  | 306348/400000 [00:39<00:12, 7616.21it/s] 77%|███████▋  | 307150/400000 [00:39<00:12, 7732.30it/s] 77%|███████▋  | 307924/400000 [00:39<00:12, 7670.11it/s] 77%|███████▋  | 308698/400000 [00:39<00:11, 7688.19it/s] 77%|███████▋  | 309485/400000 [00:40<00:11, 7741.59it/s] 78%|███████▊  | 310260/400000 [00:40<00:11, 7702.34it/s] 78%|███████▊  | 311031/400000 [00:40<00:11, 7676.99it/s] 78%|███████▊  | 311799/400000 [00:40<00:11, 7674.20it/s] 78%|███████▊  | 312592/400000 [00:40<00:11, 7749.07it/s] 78%|███████▊  | 313385/400000 [00:40<00:11, 7802.32it/s] 79%|███████▊  | 314171/400000 [00:40<00:10, 7816.89it/s] 79%|███████▊  | 314953/400000 [00:40<00:10, 7745.28it/s] 79%|███████▉  | 315728/400000 [00:40<00:11, 7605.98it/s] 79%|███████▉  | 316504/400000 [00:40<00:10, 7649.60it/s] 79%|███████▉  | 317270/400000 [00:41<00:11, 7459.17it/s] 80%|███████▉  | 318018/400000 [00:41<00:11, 7233.43it/s] 80%|███████▉  | 318778/400000 [00:41<00:11, 7339.29it/s] 80%|███████▉  | 319556/400000 [00:41<00:10, 7464.74it/s] 80%|████████  | 320354/400000 [00:41<00:10, 7610.55it/s] 80%|████████  | 321145/400000 [00:41<00:10, 7695.69it/s] 80%|████████  | 321927/400000 [00:41<00:10, 7731.42it/s] 81%|████████  | 322709/400000 [00:41<00:09, 7757.70it/s] 81%|████████  | 323486/400000 [00:41<00:09, 7731.05it/s] 81%|████████  | 324276/400000 [00:41<00:09, 7780.60it/s] 81%|████████▏ | 325066/400000 [00:42<00:09, 7814.93it/s] 81%|████████▏ | 325848/400000 [00:42<00:09, 7803.22it/s] 82%|████████▏ | 326635/400000 [00:42<00:09, 7821.14it/s] 82%|████████▏ | 327418/400000 [00:42<00:09, 7737.78it/s] 82%|████████▏ | 328193/400000 [00:42<00:09, 7694.92it/s] 82%|████████▏ | 328992/400000 [00:42<00:09, 7779.91it/s] 82%|████████▏ | 329774/400000 [00:42<00:09, 7791.33it/s] 83%|████████▎ | 330568/400000 [00:42<00:08, 7835.03it/s] 83%|████████▎ | 331352/400000 [00:42<00:08, 7789.69it/s] 83%|████████▎ | 332142/400000 [00:42<00:08, 7820.66it/s] 83%|████████▎ | 332940/400000 [00:43<00:08, 7866.84it/s] 83%|████████▎ | 333729/400000 [00:43<00:08, 7873.23it/s] 84%|████████▎ | 334531/400000 [00:43<00:08, 7915.58it/s] 84%|████████▍ | 335323/400000 [00:43<00:08, 7794.97it/s] 84%|████████▍ | 336112/400000 [00:43<00:08, 7822.41it/s] 84%|████████▍ | 336908/400000 [00:43<00:08, 7863.13it/s] 84%|████████▍ | 337707/400000 [00:43<00:07, 7898.96it/s] 85%|████████▍ | 338498/400000 [00:43<00:07, 7871.55it/s] 85%|████████▍ | 339286/400000 [00:43<00:07, 7814.38it/s] 85%|████████▌ | 340073/400000 [00:43<00:07, 7827.25it/s] 85%|████████▌ | 340863/400000 [00:44<00:07, 7847.65it/s] 85%|████████▌ | 341655/400000 [00:44<00:07, 7866.74it/s] 86%|████████▌ | 342454/400000 [00:44<00:07, 7900.72it/s] 86%|████████▌ | 343245/400000 [00:44<00:07, 7787.19it/s] 86%|████████▌ | 344025/400000 [00:44<00:07, 7605.50it/s] 86%|████████▌ | 344802/400000 [00:44<00:07, 7653.41it/s] 86%|████████▋ | 345570/400000 [00:44<00:07, 7660.83it/s] 87%|████████▋ | 346337/400000 [00:44<00:07, 7558.37it/s] 87%|████████▋ | 347097/400000 [00:44<00:06, 7568.64it/s] 87%|████████▋ | 347855/400000 [00:44<00:07, 7446.71it/s] 87%|████████▋ | 348607/400000 [00:45<00:06, 7466.76it/s] 87%|████████▋ | 349355/400000 [00:45<00:06, 7445.93it/s] 88%|████████▊ | 350123/400000 [00:45<00:06, 7512.60it/s] 88%|████████▊ | 350876/400000 [00:45<00:06, 7516.94it/s] 88%|████████▊ | 351629/400000 [00:45<00:06, 7408.51it/s] 88%|████████▊ | 352404/400000 [00:45<00:06, 7507.22it/s] 88%|████████▊ | 353156/400000 [00:45<00:06, 7504.41it/s] 88%|████████▊ | 353928/400000 [00:45<00:06, 7566.81it/s] 89%|████████▊ | 354686/400000 [00:45<00:05, 7552.61it/s] 89%|████████▉ | 355461/400000 [00:45<00:05, 7608.72it/s] 89%|████████▉ | 356223/400000 [00:46<00:05, 7374.20it/s] 89%|████████▉ | 356963/400000 [00:46<00:06, 7145.61it/s] 89%|████████▉ | 357681/400000 [00:46<00:05, 7106.05it/s] 90%|████████▉ | 358418/400000 [00:46<00:05, 7182.63it/s] 90%|████████▉ | 359184/400000 [00:46<00:05, 7318.74it/s] 90%|████████▉ | 359954/400000 [00:46<00:05, 7426.44it/s] 90%|█████████ | 360706/400000 [00:46<00:05, 7453.09it/s] 90%|█████████ | 361466/400000 [00:46<00:05, 7492.87it/s] 91%|█████████ | 362217/400000 [00:46<00:05, 7451.67it/s] 91%|█████████ | 363003/400000 [00:47<00:04, 7567.81it/s] 91%|█████████ | 363761/400000 [00:47<00:04, 7415.93it/s] 91%|█████████ | 364504/400000 [00:47<00:04, 7362.66it/s] 91%|█████████▏| 365244/400000 [00:47<00:04, 7367.47it/s] 91%|█████████▏| 365982/400000 [00:47<00:04, 7309.76it/s] 92%|█████████▏| 366756/400000 [00:47<00:04, 7431.09it/s] 92%|█████████▏| 367540/400000 [00:47<00:04, 7548.63it/s] 92%|█████████▏| 368301/400000 [00:47<00:04, 7564.78it/s] 92%|█████████▏| 369070/400000 [00:47<00:04, 7595.97it/s] 92%|█████████▏| 369831/400000 [00:47<00:03, 7593.08it/s] 93%|█████████▎| 370600/400000 [00:48<00:03, 7619.84it/s] 93%|█████████▎| 371383/400000 [00:48<00:03, 7680.06it/s] 93%|█████████▎| 372165/400000 [00:48<00:03, 7720.45it/s] 93%|█████████▎| 372938/400000 [00:48<00:03, 7712.43it/s] 93%|█████████▎| 373710/400000 [00:48<00:03, 7666.75it/s] 94%|█████████▎| 374480/400000 [00:48<00:03, 7675.40it/s] 94%|█████████▍| 375248/400000 [00:48<00:03, 7674.46it/s] 94%|█████████▍| 376021/400000 [00:48<00:03, 7690.60it/s] 94%|█████████▍| 376791/400000 [00:48<00:03, 7642.47it/s] 94%|█████████▍| 377556/400000 [00:48<00:03, 7425.34it/s] 95%|█████████▍| 378300/400000 [00:49<00:02, 7283.82it/s] 95%|█████████▍| 379075/400000 [00:49<00:02, 7417.04it/s] 95%|█████████▍| 379820/400000 [00:49<00:02, 7424.30it/s] 95%|█████████▌| 380592/400000 [00:49<00:02, 7510.07it/s] 95%|█████████▌| 381345/400000 [00:49<00:02, 7492.92it/s] 96%|█████████▌| 382136/400000 [00:49<00:02, 7613.04it/s] 96%|█████████▌| 382934/400000 [00:49<00:02, 7718.29it/s] 96%|█████████▌| 383736/400000 [00:49<00:02, 7805.28it/s] 96%|█████████▌| 384518/400000 [00:49<00:01, 7779.53it/s] 96%|█████████▋| 385297/400000 [00:49<00:01, 7659.32it/s] 97%|█████████▋| 386074/400000 [00:50<00:01, 7689.52it/s] 97%|█████████▋| 386859/400000 [00:50<00:01, 7736.51it/s] 97%|█████████▋| 387645/400000 [00:50<00:01, 7772.06it/s] 97%|█████████▋| 388430/400000 [00:50<00:01, 7794.30it/s] 97%|█████████▋| 389210/400000 [00:50<00:01, 7717.81it/s] 97%|█████████▋| 389983/400000 [00:50<00:01, 7708.78it/s] 98%|█████████▊| 390771/400000 [00:50<00:01, 7757.54it/s] 98%|█████████▊| 391549/400000 [00:50<00:01, 7762.62it/s] 98%|█████████▊| 392326/400000 [00:50<00:01, 7639.13it/s] 98%|█████████▊| 393091/400000 [00:50<00:00, 7606.83it/s] 98%|█████████▊| 393864/400000 [00:51<00:00, 7640.62it/s] 99%|█████████▊| 394631/400000 [00:51<00:00, 7646.84it/s] 99%|█████████▉| 395412/400000 [00:51<00:00, 7694.90it/s] 99%|█████████▉| 396182/400000 [00:51<00:00, 7453.57it/s] 99%|█████████▉| 396930/400000 [00:51<00:00, 7436.96it/s] 99%|█████████▉| 397707/400000 [00:51<00:00, 7532.76it/s]100%|█████████▉| 398462/400000 [00:51<00:00, 7490.47it/s]100%|█████████▉| 399222/400000 [00:51<00:00, 7521.26it/s]100%|█████████▉| 399992/400000 [00:51<00:00, 7573.85it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7708.69it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fefd9e07b70> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0113495340163104 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011004018943046647 	 Accuracy: 67

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

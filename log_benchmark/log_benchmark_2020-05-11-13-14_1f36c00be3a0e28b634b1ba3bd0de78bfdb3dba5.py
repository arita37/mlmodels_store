
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1cfde3ffd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 13:15:03.763971
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 13:15:03.768161
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 13:15:03.771358
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 13:15:03.774663
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1d09e57470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354059.5312
Epoch 2/10

1/1 [==============================] - 0s 112ms/step - loss: 242075.2344
Epoch 3/10

1/1 [==============================] - 0s 100ms/step - loss: 142623.9219
Epoch 4/10

1/1 [==============================] - 0s 95ms/step - loss: 80894.3672
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 46628.3867
Epoch 6/10

1/1 [==============================] - 0s 97ms/step - loss: 27801.7344
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 17621.8809
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 12115.5654
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 8888.4111
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 6859.7656

  #### Inference Need return ypred, ytrue ######################### 
[[-1.9642367   0.6720263  -1.6826833  -1.4512225   1.2307112   0.696471
  -1.3126819  -2.683458   -0.50062394 -0.66298556  0.11662354  1.5696629
   0.71190244 -0.12688023  0.2518084   1.1504178   1.3373867  -0.37530306
  -0.84843165 -1.1916724  -0.7318491  -0.165274    1.7430024   0.8994763
   1.5260439   1.3832513  -0.17488927  0.0339385  -0.15961665 -1.2756509
  -0.66289043  1.1710571   0.93876445  1.0128809   0.65598863 -1.5413269
   1.2200651   0.46214986  1.5048866  -0.62945557 -1.0199178  -0.16629952
  -0.8335874  -0.49841163  1.9340303   0.7662567   1.6064613   0.50436443
  -0.11611345  0.95608085 -0.6012728  -1.9061466   1.108664   -0.07911831
   0.74149394 -1.0347152   1.0161594  -0.7291906   1.331388    0.53754514
  -0.09152973  5.714702    8.79347     7.1743083   8.817155    9.148714
   8.191377    8.0287895   7.5669675   7.261765    7.0843754   6.5291085
   6.657582    9.521244    7.1251736   7.3611403   7.8687797   8.553507
   7.448487    8.677538    7.608678    8.885778    7.45133     7.283041
   8.90628     8.822769    6.472726    8.509682    8.962562    8.639594
   8.763342    8.762159    9.129367    9.683325    7.895888    7.8096514
   6.559364    6.924664    8.716361    6.6369357   8.31947     8.296721
   6.663921    7.875407    8.906317    7.011707    8.227058    7.6203737
   6.7319674   7.182396    6.2281594   8.530875    7.353556    6.396783
   8.320502    7.886521   10.322552    8.760187    7.144272    6.310951
   0.20376635  0.8536033   0.9292433  -0.18000695 -0.7102063  -1.0021465
  -0.14905444  0.42466855 -1.1348109  -0.20019333  0.49727342  1.1640159
  -0.41240048 -1.3972495   1.0732677   0.5922333   1.3630195  -0.90579367
   1.4702023  -0.01782861 -0.3421556   0.90166545  0.5031904   1.7910749
   1.8641475   0.13928601 -0.69142574 -1.6957235  -0.2708668   0.48205858
   1.2699764   0.6259965  -0.63847774  0.5344113  -1.8383638   1.700727
   0.5770639  -0.9363326  -0.64168006 -0.44263402  0.26736438 -0.48006597
   1.1096637   1.3864712  -1.1036447   0.9839237   0.52321243 -0.20778745
   0.62724346 -1.5233576  -1.3847328  -0.41018867  0.05357844 -0.06143028
   2.2722406  -0.68998927 -0.6739909   0.62171626  0.03126106 -2.0502467
   1.7790453   0.5162669   0.14287752  1.2012157   0.37268543  2.0305848
   0.7394387   2.326788    2.4929686   0.6079953   0.6862373   0.3238451
   2.6609159   0.79596186  0.15034044  0.14139265  1.2709363   1.0029184
   1.2771308   0.82527804  0.86688155  1.2945647   0.74378544  0.20457363
   1.0342095   2.835876    0.98544526  0.25997412  0.69733506  2.8733077
   2.2358732   1.6937712   0.38084984  2.131549    1.5221175   0.8319072
   0.49354625  0.29571247  1.6836991   2.3367      1.8491263   1.4366809
   0.14994448  0.85255307  1.0239985   1.3924224   0.9495401   1.7717493
   1.3686972   2.044517    0.9840995   0.3175996   1.8553532   1.6239158
   1.6628625   0.9036323   1.6262708   1.4895054   1.8606957   0.18233943
   0.19928122 10.080399    8.720315    7.374869    7.2477      8.839206
   7.051599    6.8882313   9.617338    7.558959    8.627186    8.11851
   9.9285345   7.850584    7.510628    7.4957037   8.251927    6.9448023
   8.1826105   7.584515    8.552761    8.69221     9.357032    7.707141
   8.01488     7.7372293   9.245501    6.867822    6.7217174   7.022999
   8.037991    7.3482194   6.66839     8.848683    8.47039     7.9347773
   9.303016    7.7076283   7.545847    8.895848    8.945724    8.949327
   6.8569975   8.874573    8.499953    7.671394    9.142871    9.16039
   8.663719    8.595695    7.3822875   6.9573216   9.629828    8.847316
   8.958499    8.479045    9.251975    9.38849     8.192096    6.7067256
   0.68574154  0.6759205   1.724491    1.6064525   0.35835892  0.3932588
   0.8974325   1.3022765   0.94517094  2.0557442   3.4208941   0.47764176
   0.5392766   1.6665481   1.6384234   0.33983165  0.6450535   0.8026353
   0.5274959   0.51895916  1.9870305   2.3839912   1.1411859   1.1607894
   0.6538531   0.5895059   0.46378624  1.6163621   1.0589972   0.5330204
   0.27401465  1.5346518   1.7623498   2.7117608   0.83415073  0.40771413
   0.8201138   0.8713807   1.3661062   0.7197993   1.1461195   0.6017901
   2.3386438   1.5065295   0.67575634  0.6726639   2.5829377   2.2359872
   1.484943    2.9111013   0.19944394  0.5409098   0.25027347  1.0799122
   0.22973543  0.87110263  0.7450288   1.9659194   1.2476465   0.1779437
  -9.783666    8.964722   -4.028069  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 13:15:13.496179
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.5316
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 13:15:13.499950
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8958.07
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 13:15:13.503441
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.3118
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 13:15:13.508548
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -801.259
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139762138690560
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139761197092368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139761197191240
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139761197191744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139761197192248
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139761197192752

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1d05cdaef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.622704
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.589712
grad_step = 000002, loss = 0.560970
grad_step = 000003, loss = 0.528294
grad_step = 000004, loss = 0.486952
grad_step = 000005, loss = 0.440715
grad_step = 000006, loss = 0.394683
grad_step = 000007, loss = 0.371904
grad_step = 000008, loss = 0.369473
grad_step = 000009, loss = 0.350555
grad_step = 000010, loss = 0.321829
grad_step = 000011, loss = 0.300155
grad_step = 000012, loss = 0.287486
grad_step = 000013, loss = 0.278657
grad_step = 000014, loss = 0.268972
grad_step = 000015, loss = 0.256631
grad_step = 000016, loss = 0.242310
grad_step = 000017, loss = 0.228370
grad_step = 000018, loss = 0.216935
grad_step = 000019, loss = 0.207575
grad_step = 000020, loss = 0.197594
grad_step = 000021, loss = 0.185800
grad_step = 000022, loss = 0.173933
grad_step = 000023, loss = 0.163626
grad_step = 000024, loss = 0.154675
grad_step = 000025, loss = 0.145723
grad_step = 000026, loss = 0.135950
grad_step = 000027, loss = 0.125872
grad_step = 000028, loss = 0.116721
grad_step = 000029, loss = 0.109006
grad_step = 000030, loss = 0.101831
grad_step = 000031, loss = 0.094291
grad_step = 000032, loss = 0.086701
grad_step = 000033, loss = 0.079837
grad_step = 000034, loss = 0.073727
grad_step = 000035, loss = 0.067873
grad_step = 000036, loss = 0.062038
grad_step = 000037, loss = 0.056501
grad_step = 000038, loss = 0.051705
grad_step = 000039, loss = 0.047525
grad_step = 000040, loss = 0.043418
grad_step = 000041, loss = 0.039360
grad_step = 000042, loss = 0.035667
grad_step = 000043, loss = 0.032460
grad_step = 000044, loss = 0.029575
grad_step = 000045, loss = 0.026780
grad_step = 000046, loss = 0.024134
grad_step = 000047, loss = 0.021795
grad_step = 000048, loss = 0.019709
grad_step = 000049, loss = 0.017759
grad_step = 000050, loss = 0.015910
grad_step = 000051, loss = 0.014252
grad_step = 000052, loss = 0.012804
grad_step = 000053, loss = 0.011494
grad_step = 000054, loss = 0.010312
grad_step = 000055, loss = 0.009278
grad_step = 000056, loss = 0.008356
grad_step = 000057, loss = 0.007507
grad_step = 000058, loss = 0.006748
grad_step = 000059, loss = 0.006100
grad_step = 000060, loss = 0.005528
grad_step = 000061, loss = 0.005005
grad_step = 000062, loss = 0.004560
grad_step = 000063, loss = 0.004198
grad_step = 000064, loss = 0.003891
grad_step = 000065, loss = 0.003618
grad_step = 000066, loss = 0.003392
grad_step = 000067, loss = 0.003204
grad_step = 000068, loss = 0.003037
grad_step = 000069, loss = 0.002903
grad_step = 000070, loss = 0.002807
grad_step = 000071, loss = 0.002726
grad_step = 000072, loss = 0.002645
grad_step = 000073, loss = 0.002588
grad_step = 000074, loss = 0.002555
grad_step = 000075, loss = 0.002523
grad_step = 000076, loss = 0.002487
grad_step = 000077, loss = 0.002463
grad_step = 000078, loss = 0.002448
grad_step = 000079, loss = 0.002432
grad_step = 000080, loss = 0.002416
grad_step = 000081, loss = 0.002406
grad_step = 000082, loss = 0.002394
grad_step = 000083, loss = 0.002379
grad_step = 000084, loss = 0.002369
grad_step = 000085, loss = 0.002359
grad_step = 000086, loss = 0.002346
grad_step = 000087, loss = 0.002334
grad_step = 000088, loss = 0.002324
grad_step = 000089, loss = 0.002313
grad_step = 000090, loss = 0.002299
grad_step = 000091, loss = 0.002288
grad_step = 000092, loss = 0.002279
grad_step = 000093, loss = 0.002267
grad_step = 000094, loss = 0.002255
grad_step = 000095, loss = 0.002245
grad_step = 000096, loss = 0.002235
grad_step = 000097, loss = 0.002226
grad_step = 000098, loss = 0.002217
grad_step = 000099, loss = 0.002208
grad_step = 000100, loss = 0.002200
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002193
grad_step = 000102, loss = 0.002185
grad_step = 000103, loss = 0.002177
grad_step = 000104, loss = 0.002171
grad_step = 000105, loss = 0.002165
grad_step = 000106, loss = 0.002158
grad_step = 000107, loss = 0.002153
grad_step = 000108, loss = 0.002147
grad_step = 000109, loss = 0.002142
grad_step = 000110, loss = 0.002137
grad_step = 000111, loss = 0.002132
grad_step = 000112, loss = 0.002128
grad_step = 000113, loss = 0.002124
grad_step = 000114, loss = 0.002119
grad_step = 000115, loss = 0.002115
grad_step = 000116, loss = 0.002111
grad_step = 000117, loss = 0.002107
grad_step = 000118, loss = 0.002104
grad_step = 000119, loss = 0.002100
grad_step = 000120, loss = 0.002096
grad_step = 000121, loss = 0.002093
grad_step = 000122, loss = 0.002089
grad_step = 000123, loss = 0.002086
grad_step = 000124, loss = 0.002083
grad_step = 000125, loss = 0.002079
grad_step = 000126, loss = 0.002076
grad_step = 000127, loss = 0.002072
grad_step = 000128, loss = 0.002069
grad_step = 000129, loss = 0.002066
grad_step = 000130, loss = 0.002062
grad_step = 000131, loss = 0.002059
grad_step = 000132, loss = 0.002055
grad_step = 000133, loss = 0.002052
grad_step = 000134, loss = 0.002049
grad_step = 000135, loss = 0.002045
grad_step = 000136, loss = 0.002042
grad_step = 000137, loss = 0.002038
grad_step = 000138, loss = 0.002035
grad_step = 000139, loss = 0.002031
grad_step = 000140, loss = 0.002028
grad_step = 000141, loss = 0.002025
grad_step = 000142, loss = 0.002021
grad_step = 000143, loss = 0.002018
grad_step = 000144, loss = 0.002015
grad_step = 000145, loss = 0.002011
grad_step = 000146, loss = 0.002008
grad_step = 000147, loss = 0.002005
grad_step = 000148, loss = 0.002002
grad_step = 000149, loss = 0.001998
grad_step = 000150, loss = 0.001995
grad_step = 000151, loss = 0.001992
grad_step = 000152, loss = 0.001989
grad_step = 000153, loss = 0.001988
grad_step = 000154, loss = 0.001990
grad_step = 000155, loss = 0.001992
grad_step = 000156, loss = 0.001987
grad_step = 000157, loss = 0.001978
grad_step = 000158, loss = 0.001972
grad_step = 000159, loss = 0.001971
grad_step = 000160, loss = 0.001973
grad_step = 000161, loss = 0.001971
grad_step = 000162, loss = 0.001965
grad_step = 000163, loss = 0.001959
grad_step = 000164, loss = 0.001956
grad_step = 000165, loss = 0.001955
grad_step = 000166, loss = 0.001955
grad_step = 000167, loss = 0.001954
grad_step = 000168, loss = 0.001951
grad_step = 000169, loss = 0.001945
grad_step = 000170, loss = 0.001941
grad_step = 000171, loss = 0.001938
grad_step = 000172, loss = 0.001936
grad_step = 000173, loss = 0.001935
grad_step = 000174, loss = 0.001934
grad_step = 000175, loss = 0.001933
grad_step = 000176, loss = 0.001931
grad_step = 000177, loss = 0.001929
grad_step = 000178, loss = 0.001926
grad_step = 000179, loss = 0.001923
grad_step = 000180, loss = 0.001920
grad_step = 000181, loss = 0.001917
grad_step = 000182, loss = 0.001914
grad_step = 000183, loss = 0.001911
grad_step = 000184, loss = 0.001909
grad_step = 000185, loss = 0.001906
grad_step = 000186, loss = 0.001904
grad_step = 000187, loss = 0.001902
grad_step = 000188, loss = 0.001901
grad_step = 000189, loss = 0.001900
grad_step = 000190, loss = 0.001902
grad_step = 000191, loss = 0.001906
grad_step = 000192, loss = 0.001916
grad_step = 000193, loss = 0.001930
grad_step = 000194, loss = 0.001950
grad_step = 000195, loss = 0.001957
grad_step = 000196, loss = 0.001945
grad_step = 000197, loss = 0.001912
grad_step = 000198, loss = 0.001882
grad_step = 000199, loss = 0.001872
grad_step = 000200, loss = 0.001883
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001903
grad_step = 000202, loss = 0.001917
grad_step = 000203, loss = 0.001918
grad_step = 000204, loss = 0.001900
grad_step = 000205, loss = 0.001876
grad_step = 000206, loss = 0.001859
grad_step = 000207, loss = 0.001855
grad_step = 000208, loss = 0.001863
grad_step = 000209, loss = 0.001873
grad_step = 000210, loss = 0.001879
grad_step = 000211, loss = 0.001876
grad_step = 000212, loss = 0.001866
grad_step = 000213, loss = 0.001854
grad_step = 000214, loss = 0.001843
grad_step = 000215, loss = 0.001837
grad_step = 000216, loss = 0.001836
grad_step = 000217, loss = 0.001839
grad_step = 000218, loss = 0.001843
grad_step = 000219, loss = 0.001848
grad_step = 000220, loss = 0.001852
grad_step = 000221, loss = 0.001855
grad_step = 000222, loss = 0.001857
grad_step = 000223, loss = 0.001857
grad_step = 000224, loss = 0.001854
grad_step = 000225, loss = 0.001848
grad_step = 000226, loss = 0.001839
grad_step = 000227, loss = 0.001829
grad_step = 000228, loss = 0.001820
grad_step = 000229, loss = 0.001812
grad_step = 000230, loss = 0.001806
grad_step = 000231, loss = 0.001803
grad_step = 000232, loss = 0.001801
grad_step = 000233, loss = 0.001800
grad_step = 000234, loss = 0.001801
grad_step = 000235, loss = 0.001804
grad_step = 000236, loss = 0.001812
grad_step = 000237, loss = 0.001828
grad_step = 000238, loss = 0.001860
grad_step = 000239, loss = 0.001913
grad_step = 000240, loss = 0.001985
grad_step = 000241, loss = 0.002029
grad_step = 000242, loss = 0.001982
grad_step = 000243, loss = 0.001859
grad_step = 000244, loss = 0.001780
grad_step = 000245, loss = 0.001817
grad_step = 000246, loss = 0.001889
grad_step = 000247, loss = 0.001881
grad_step = 000248, loss = 0.001804
grad_step = 000249, loss = 0.001764
grad_step = 000250, loss = 0.001796
grad_step = 000251, loss = 0.001836
grad_step = 000252, loss = 0.001824
grad_step = 000253, loss = 0.001777
grad_step = 000254, loss = 0.001751
grad_step = 000255, loss = 0.001762
grad_step = 000256, loss = 0.001782
grad_step = 000257, loss = 0.001778
grad_step = 000258, loss = 0.001757
grad_step = 000259, loss = 0.001757
grad_step = 000260, loss = 0.001740
grad_step = 000261, loss = 0.001742
grad_step = 000262, loss = 0.001741
grad_step = 000263, loss = 0.001755
grad_step = 000264, loss = 0.001797
grad_step = 000265, loss = 0.001746
grad_step = 000266, loss = 0.001736
grad_step = 000267, loss = 0.001751
grad_step = 000268, loss = 0.001738
grad_step = 000269, loss = 0.001733
grad_step = 000270, loss = 0.001757
grad_step = 000271, loss = 0.001750
grad_step = 000272, loss = 0.001721
grad_step = 000273, loss = 0.001713
grad_step = 000274, loss = 0.001707
grad_step = 000275, loss = 0.001715
grad_step = 000276, loss = 0.001716
grad_step = 000277, loss = 0.001714
grad_step = 000278, loss = 0.001712
grad_step = 000279, loss = 0.001709
grad_step = 000280, loss = 0.001699
grad_step = 000281, loss = 0.001686
grad_step = 000282, loss = 0.001705
grad_step = 000283, loss = 0.001753
grad_step = 000284, loss = 0.001781
grad_step = 000285, loss = 0.001732
grad_step = 000286, loss = 0.001680
grad_step = 000287, loss = 0.001701
grad_step = 000288, loss = 0.001714
grad_step = 000289, loss = 0.001678
grad_step = 000290, loss = 0.001693
grad_step = 000291, loss = 0.001693
grad_step = 000292, loss = 0.001664
grad_step = 000293, loss = 0.001687
grad_step = 000294, loss = 0.001677
grad_step = 000295, loss = 0.001660
grad_step = 000296, loss = 0.001688
grad_step = 000297, loss = 0.001673
grad_step = 000298, loss = 0.001658
grad_step = 000299, loss = 0.001668
grad_step = 000300, loss = 0.001657
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001646
grad_step = 000302, loss = 0.001650
grad_step = 000303, loss = 0.001651
grad_step = 000304, loss = 0.001638
grad_step = 000305, loss = 0.001637
grad_step = 000306, loss = 0.001642
grad_step = 000307, loss = 0.001635
grad_step = 000308, loss = 0.001627
grad_step = 000309, loss = 0.001625
grad_step = 000310, loss = 0.001628
grad_step = 000311, loss = 0.001628
grad_step = 000312, loss = 0.001627
grad_step = 000313, loss = 0.001622
grad_step = 000314, loss = 0.001617
grad_step = 000315, loss = 0.001614
grad_step = 000316, loss = 0.001614
grad_step = 000317, loss = 0.001618
grad_step = 000318, loss = 0.001624
grad_step = 000319, loss = 0.001636
grad_step = 000320, loss = 0.001658
grad_step = 000321, loss = 0.001692
grad_step = 000322, loss = 0.001737
grad_step = 000323, loss = 0.001798
grad_step = 000324, loss = 0.001835
grad_step = 000325, loss = 0.001802
grad_step = 000326, loss = 0.001663
grad_step = 000327, loss = 0.001595
grad_step = 000328, loss = 0.001664
grad_step = 000329, loss = 0.001675
grad_step = 000330, loss = 0.001629
grad_step = 000331, loss = 0.001672
grad_step = 000332, loss = 0.001716
grad_step = 000333, loss = 0.001752
grad_step = 000334, loss = 0.001854
grad_step = 000335, loss = 0.001779
grad_step = 000336, loss = 0.001717
grad_step = 000337, loss = 0.001601
grad_step = 000338, loss = 0.001572
grad_step = 000339, loss = 0.001653
grad_step = 000340, loss = 0.001625
grad_step = 000341, loss = 0.001599
grad_step = 000342, loss = 0.001605
grad_step = 000343, loss = 0.001544
grad_step = 000344, loss = 0.001603
grad_step = 000345, loss = 0.001603
grad_step = 000346, loss = 0.001552
grad_step = 000347, loss = 0.001590
grad_step = 000348, loss = 0.001540
grad_step = 000349, loss = 0.001522
grad_step = 000350, loss = 0.001574
grad_step = 000351, loss = 0.001537
grad_step = 000352, loss = 0.001511
grad_step = 000353, loss = 0.001506
grad_step = 000354, loss = 0.001486
grad_step = 000355, loss = 0.001508
grad_step = 000356, loss = 0.001496
grad_step = 000357, loss = 0.001455
grad_step = 000358, loss = 0.001460
grad_step = 000359, loss = 0.001469
grad_step = 000360, loss = 0.001445
grad_step = 000361, loss = 0.001434
grad_step = 000362, loss = 0.001443
grad_step = 000363, loss = 0.001454
grad_step = 000364, loss = 0.001477
grad_step = 000365, loss = 0.001587
grad_step = 000366, loss = 0.001716
grad_step = 000367, loss = 0.001602
grad_step = 000368, loss = 0.001512
grad_step = 000369, loss = 0.001557
grad_step = 000370, loss = 0.001458
grad_step = 000371, loss = 0.001437
grad_step = 000372, loss = 0.001533
grad_step = 000373, loss = 0.001459
grad_step = 000374, loss = 0.001464
grad_step = 000375, loss = 0.001440
grad_step = 000376, loss = 0.001399
grad_step = 000377, loss = 0.001469
grad_step = 000378, loss = 0.001434
grad_step = 000379, loss = 0.001404
grad_step = 000380, loss = 0.001401
grad_step = 000381, loss = 0.001388
grad_step = 000382, loss = 0.001419
grad_step = 000383, loss = 0.001403
grad_step = 000384, loss = 0.001371
grad_step = 000385, loss = 0.001378
grad_step = 000386, loss = 0.001379
grad_step = 000387, loss = 0.001372
grad_step = 000388, loss = 0.001370
grad_step = 000389, loss = 0.001360
grad_step = 000390, loss = 0.001348
grad_step = 000391, loss = 0.001354
grad_step = 000392, loss = 0.001352
grad_step = 000393, loss = 0.001341
grad_step = 000394, loss = 0.001334
grad_step = 000395, loss = 0.001323
grad_step = 000396, loss = 0.001319
grad_step = 000397, loss = 0.001326
grad_step = 000398, loss = 0.001321
grad_step = 000399, loss = 0.001304
grad_step = 000400, loss = 0.001299
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001304
grad_step = 000402, loss = 0.001311
grad_step = 000403, loss = 0.001313
grad_step = 000404, loss = 0.001325
grad_step = 000405, loss = 0.001333
grad_step = 000406, loss = 0.001324
grad_step = 000407, loss = 0.001301
grad_step = 000408, loss = 0.001286
grad_step = 000409, loss = 0.001281
grad_step = 000410, loss = 0.001282
grad_step = 000411, loss = 0.001285
grad_step = 000412, loss = 0.001284
grad_step = 000413, loss = 0.001283
grad_step = 000414, loss = 0.001277
grad_step = 000415, loss = 0.001273
grad_step = 000416, loss = 0.001262
grad_step = 000417, loss = 0.001252
grad_step = 000418, loss = 0.001246
grad_step = 000419, loss = 0.001247
grad_step = 000420, loss = 0.001251
grad_step = 000421, loss = 0.001254
grad_step = 000422, loss = 0.001255
grad_step = 000423, loss = 0.001255
grad_step = 000424, loss = 0.001264
grad_step = 000425, loss = 0.001286
grad_step = 000426, loss = 0.001319
grad_step = 000427, loss = 0.001357
grad_step = 000428, loss = 0.001388
grad_step = 000429, loss = 0.001344
grad_step = 000430, loss = 0.001269
grad_step = 000431, loss = 0.001233
grad_step = 000432, loss = 0.001256
grad_step = 000433, loss = 0.001271
grad_step = 000434, loss = 0.001249
grad_step = 000435, loss = 0.001225
grad_step = 000436, loss = 0.001235
grad_step = 000437, loss = 0.001253
grad_step = 000438, loss = 0.001244
grad_step = 000439, loss = 0.001225
grad_step = 000440, loss = 0.001211
grad_step = 000441, loss = 0.001216
grad_step = 000442, loss = 0.001229
grad_step = 000443, loss = 0.001229
grad_step = 000444, loss = 0.001222
grad_step = 000445, loss = 0.001213
grad_step = 000446, loss = 0.001209
grad_step = 000447, loss = 0.001207
grad_step = 000448, loss = 0.001203
grad_step = 000449, loss = 0.001200
grad_step = 000450, loss = 0.001194
grad_step = 000451, loss = 0.001190
grad_step = 000452, loss = 0.001190
grad_step = 000453, loss = 0.001194
grad_step = 000454, loss = 0.001200
grad_step = 000455, loss = 0.001201
grad_step = 000456, loss = 0.001201
grad_step = 000457, loss = 0.001199
grad_step = 000458, loss = 0.001203
grad_step = 000459, loss = 0.001203
grad_step = 000460, loss = 0.001201
grad_step = 000461, loss = 0.001192
grad_step = 000462, loss = 0.001179
grad_step = 000463, loss = 0.001171
grad_step = 000464, loss = 0.001170
grad_step = 000465, loss = 0.001172
grad_step = 000466, loss = 0.001173
grad_step = 000467, loss = 0.001175
grad_step = 000468, loss = 0.001172
grad_step = 000469, loss = 0.001170
grad_step = 000470, loss = 0.001164
grad_step = 000471, loss = 0.001158
grad_step = 000472, loss = 0.001152
grad_step = 000473, loss = 0.001148
grad_step = 000474, loss = 0.001146
grad_step = 000475, loss = 0.001145
grad_step = 000476, loss = 0.001144
grad_step = 000477, loss = 0.001145
grad_step = 000478, loss = 0.001148
grad_step = 000479, loss = 0.001152
grad_step = 000480, loss = 0.001164
grad_step = 000481, loss = 0.001182
grad_step = 000482, loss = 0.001214
grad_step = 000483, loss = 0.001250
grad_step = 000484, loss = 0.001267
grad_step = 000485, loss = 0.001199
grad_step = 000486, loss = 0.001145
grad_step = 000487, loss = 0.001136
grad_step = 000488, loss = 0.001171
grad_step = 000489, loss = 0.001198
grad_step = 000490, loss = 0.001169
grad_step = 000491, loss = 0.001141
grad_step = 000492, loss = 0.001154
grad_step = 000493, loss = 0.001159
grad_step = 000494, loss = 0.001146
grad_step = 000495, loss = 0.001133
grad_step = 000496, loss = 0.001134
grad_step = 000497, loss = 0.001127
grad_step = 000498, loss = 0.001120
grad_step = 000499, loss = 0.001126
grad_step = 000500, loss = 0.001131
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001119
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

  date_run                              2020-05-11 13:15:32.550644
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.224815
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 13:15:32.556771
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.124375
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 13:15:32.565121
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.132641
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 13:15:32.570931
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.889925
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
0   2020-05-11 13:15:03.763971  ...    mean_absolute_error
1   2020-05-11 13:15:03.768161  ...     mean_squared_error
2   2020-05-11 13:15:03.771358  ...  median_absolute_error
3   2020-05-11 13:15:03.774663  ...               r2_score
4   2020-05-11 13:15:13.496179  ...    mean_absolute_error
5   2020-05-11 13:15:13.499950  ...     mean_squared_error
6   2020-05-11 13:15:13.503441  ...  median_absolute_error
7   2020-05-11 13:15:13.508548  ...               r2_score
8   2020-05-11 13:15:32.550644  ...    mean_absolute_error
9   2020-05-11 13:15:32.556771  ...     mean_squared_error
10  2020-05-11 13:15:32.565121  ...  median_absolute_error
11  2020-05-11 13:15:32.570931  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3751936/9912422 [00:00<00:00, 37143646.57it/s]9920512it [00:00, 36441891.42it/s]                             
0it [00:00, ?it/s]32768it [00:00, 533126.53it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7711295.46it/s]          
0it [00:00, ?it/s]8192it [00:00, 201574.23it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8fc58e0b38> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f63034eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8fc58a4e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f62b0d080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8fc58e0b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f7829de10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8fc58a4e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f630310b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f63031fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f630310b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8f63031080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe9da7c9208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0815f477c624cdb32eaa598fd4f9b2b79f1d6eae3de85ff921fc512762691a55
  Stored in directory: /tmp/pip-ephem-wheel-cache-ir3ylvum/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe9d0938048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1761280/17464789 [==>...........................] - ETA: 0s
 7307264/17464789 [===========>..................] - ETA: 0s
12935168/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 13:16:59.758679: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 13:16:59.763129: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-11 13:16:59.763256: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557822286550 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 13:16:59.763270: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0193 - accuracy: 0.4770
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7586 - accuracy: 0.4940 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7740 - accuracy: 0.4930
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8046 - accuracy: 0.4910
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7637 - accuracy: 0.4937
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7696 - accuracy: 0.4933
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7143 - accuracy: 0.4969
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 3s - loss: 7.6931 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 3s - loss: 7.7050 - accuracy: 0.4975
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6996 - accuracy: 0.4978
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6885 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 2s - loss: 7.7096 - accuracy: 0.4972
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7174 - accuracy: 0.4967
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7171 - accuracy: 0.4967
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7194 - accuracy: 0.4966
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7086 - accuracy: 0.4973
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6914 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6994 - accuracy: 0.4979
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7073 - accuracy: 0.4973
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6909 - accuracy: 0.4984
25000/25000 [==============================] - 7s 290us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 13:17:13.771221
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 13:17:13.771221  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 13:17:20.081308: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 13:17:20.086951: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-11 13:17:20.087156: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559ecb059eb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 13:17:20.087171: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd98c6b4dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2271 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.1889 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd982956898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.3260 - accuracy: 0.4570
 2000/25000 [=>............................] - ETA: 8s - loss: 8.0423 - accuracy: 0.4755 
 3000/25000 [==>...........................] - ETA: 7s - loss: 8.0704 - accuracy: 0.4737
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8928 - accuracy: 0.4852
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.9150 - accuracy: 0.4838
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8557 - accuracy: 0.4877
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8046 - accuracy: 0.4910
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8123 - accuracy: 0.4905
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8472 - accuracy: 0.4882
10000/25000 [===========>..................] - ETA: 3s - loss: 7.8460 - accuracy: 0.4883
11000/25000 [============>.................] - ETA: 3s - loss: 7.7963 - accuracy: 0.4915
12000/25000 [=============>................] - ETA: 3s - loss: 7.7612 - accuracy: 0.4938
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7170 - accuracy: 0.4967
15000/25000 [=================>............] - ETA: 2s - loss: 7.7331 - accuracy: 0.4957
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7040 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6928 - accuracy: 0.4983
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 7s 292us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd93d90b400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<43:18:40, 5.53kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<30:32:54, 7.84kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:26:18, 11.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:00:33, 15.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.08M/862M [00:01<10:28:59, 22.8kB/s].vector_cache/glove.6B.zip:   1%|          | 5.89M/862M [00:02<7:19:00, 32.5kB/s] .vector_cache/glove.6B.zip:   1%|          | 10.0M/862M [00:02<5:05:55, 46.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.5M/862M [00:02<3:33:23, 66.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.3M/862M [00:02<2:28:49, 94.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:02<1:43:45, 135kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:02<1:12:25, 193kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:02<50:29, 275kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.6M/862M [00:02<35:18, 391kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<24:40, 557kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:02<17:20, 789kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:03<12:10, 1.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:03<08:33, 1.58MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:03<06:30, 2.08MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:05<06:27, 2.08MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<08:09, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<06:38, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:05<04:49, 2.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<08:59, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<08:09, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:10, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:09<06:50, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:09<06:35, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<05:04, 2.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:11<06:03, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:11<07:50, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<06:24, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<04:42, 2.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<08:55, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<08:02, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<06:05, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:15<06:44, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:15<06:30, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<04:55, 2.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<05:56, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<05:55, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<04:31, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<03:18, 3.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<50:43, 255kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:19<37:15, 348kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<26:24, 490kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:19<18:35, 694kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:21<39:35, 325kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<29:27, 437kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<21:01, 612kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:23<17:06, 749kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<13:42, 934kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:23<09:57, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:23<07:07, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:25<18:19, 695kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<14:33, 875kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:25<10:36, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<09:49, 1.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<08:35, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:23, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:37, 2.73MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<25:57, 486kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<19:52, 634kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<14:14, 883kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<10:06, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<26:47, 468kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<20:30, 611kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<14:42, 851kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<12:38, 986kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<10:35, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<07:45, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:36, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<12:09, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<10:13, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<07:32, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<07:36, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<07:03, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:57, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:35, 2.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:31, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:34, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:16, 2.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:17, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<04:11, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:20, 2.26MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<04:04, 2.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<02:58, 4.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<53:13, 225kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<38:54, 308kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<27:32, 434kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<21:29, 554kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<16:42, 713kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<12:01, 990kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<10:39, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:07, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<06:46, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:58, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:31, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:57, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:41, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:36, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<04:15, 2.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:12, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:16, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<04:05, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:02, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:08, 2.25MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:57, 2.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:57, 2.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:49, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:46, 3.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<02:46, 4.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<15:05, 758kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<12:09, 940kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:53, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<08:21, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:26, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:35, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:03, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:31, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:54, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:18, 2.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<06:23, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:52, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:23, 2.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:20, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:15, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:03, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<04:56, 2.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:27, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<03:52, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:26, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:42, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:00, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:36, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:53, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:27, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:57, 2.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:51, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:16, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:41, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:21, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:13, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:00, 2.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:51, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:54, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<03:44, 2.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:39, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:44, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:37, 2.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<02:38, 4.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<3:33:02, 49.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<2:30:32, 70.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<1:45:29, 100kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<1:13:39, 143kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<6:06:06, 28.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<4:17:23, 40.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<3:00:04, 58.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<2:07:46, 81.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<1:32:17, 113kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<1:05:09, 160kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<45:36, 228kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<35:35, 292kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<26:20, 394kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<18:43, 554kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<15:01, 687kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<13:18, 776kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<09:54, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:07, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:40, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<06:47, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:03, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:28, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:16, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:58, 2.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<02:53, 3.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<45:09, 224kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<33:00, 306kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<23:22, 432kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<18:13, 552kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<14:09, 710kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<10:13, 980kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<09:01, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<07:42, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:43, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:53, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<06:48, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:21, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:54, 2.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<06:54, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<06:13, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:37, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:20, 2.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<38:47, 252kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<28:29, 343kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<20:14, 482kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<15:57, 608kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<13:48, 703kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<10:12, 950kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:17, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<07:47, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<06:48, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:02, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<05:04, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:52, 2.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:30, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<05:44, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<04:33, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:20, 2.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:06, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<04:53, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<03:44, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<04:23, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:22, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<03:22, 2.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:07, 2.25MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:11, 2.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<03:14, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<04:01, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<04:05, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<03:07, 2.94MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:56, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<04:00, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<03:06, 2.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:53, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<03:58, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<03:02, 2.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:51, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<03:55, 2.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<02:59, 3.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:48, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<03:53, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<02:58, 3.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:10, 4.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<16:40, 533kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<12:41, 699kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<09:05, 974kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<08:11, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<06:56, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:06, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:14, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:50, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<03:38, 2.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:37, 3.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<59:28, 146kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<42:47, 203kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<30:08, 287kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<22:38, 380kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<18:18, 470kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<13:23, 641kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:27, 903kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<10:18, 827kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<08:22, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<06:08, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:53, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:17, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:56, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:50, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<23:49, 352kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<17:50, 470kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<12:44, 656kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<10:27, 796kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:29, 981kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:12, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:53, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:14, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:54, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:18, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:09, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:08, 2.60MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<03:44, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<03:45, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<02:53, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<03:33, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:35, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<02:44, 2.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<01:59, 4.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<1:00:58, 131kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<43:44, 182kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<30:49, 258kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<22:57, 345kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<17:08, 461kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<12:11, 646kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<10:00, 784kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<08:03, 972kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:51, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<04:10, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<21:27, 362kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<16:04, 483kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<11:26, 677kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<08:03, 957kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<1:22:25, 93.4kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<58:44, 131kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<41:14, 186kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<30:09, 253kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<22:08, 345kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<15:41, 485kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<12:22, 611kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<10:42, 706kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<07:56, 951kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<05:38, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<06:49, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:48, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:15, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:03, 2.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<29:53, 248kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<21:57, 338kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<15:32, 476kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<10:54, 675kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<25:54, 284kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<19:07, 384kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<13:33, 541kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<09:32, 765kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<17:18, 421kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<13:05, 556kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<09:21, 776kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<07:54, 914kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<06:30, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:45, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:23, 2.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<16:08, 443kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<12:15, 582kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:46, 812kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<07:28, 948kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<06:11, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:33, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<04:31, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<04:06, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:13, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<16:21, 424kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<12:23, 560kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<08:51, 781kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<06:14, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<55:40, 123kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<39:52, 172kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<28:01, 244kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<19:35, 348kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<29:58, 227kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<21:53, 311kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<15:30, 437kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<12:04, 558kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<09:22, 717kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:44, 995kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<05:58, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<05:06, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:45, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<03:52, 1.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:38, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:45, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:08, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:57, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:10, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:18, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<04:19, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<03:54, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:01, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:11, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:18, 2.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:41, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:27, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:37, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:58, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:17, 2.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:46, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:48, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:08, 2.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<01:33, 3.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:35<41:28, 147kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<29:52, 204kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<21:02, 289kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<15:46, 383kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<12:40, 476kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<09:12, 655kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:29, 923kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<06:46, 882kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<05:33, 1.07MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:05, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<03:57, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<03:35, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:41, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<02:58, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<02:54, 2.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:13, 2.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<02:38, 2.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<02:37, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:01, 2.83MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:29, 2.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<03:17, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:41, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:57, 2.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<03:47, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<03:26, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:33, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:50, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<22:02, 252kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<16:11, 343kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<11:29, 481kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<09:01, 608kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<07:03, 777kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<05:04, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:34, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<03:55, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:53, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:04, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<10:05, 530kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<07:46, 686kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<05:36, 948kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:54, 1.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<04:08, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:03, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:07, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<03:35, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:48, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:02, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:00, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:48, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:06, 2.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:36, 3.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:11, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:13, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:30, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:33, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:12, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:20, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:38, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:05, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:14, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:31, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<02:00, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:10, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:29, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:58, 2.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:08, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:27, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:57, 2.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:06, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:23, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:54, 2.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:05, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:53, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:25, 3.21MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:04, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:19, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:51, 2.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:00, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:16, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:48, 2.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<01:57, 2.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<03:09, 1.39MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:35, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:53, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:22, 3.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<31:12, 138kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<22:42, 190kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<16:03, 268kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<11:47, 360kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<09:04, 467kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<06:32, 646kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<05:12, 802kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:29, 929kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:20, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:58, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:53, 1.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:13, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:34, 2.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<1:20:30, 50.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<57:58, 69.6kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<40:55, 98.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<28:32, 140kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<20:43, 192kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<15:16, 260kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<10:50, 364kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<08:07, 480kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<06:29, 600kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<04:43, 823kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:53, 986kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:27, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:36, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:52, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<03:17, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<03:43, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:56, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:07, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:24, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<03:04, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:28, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:18, 2.77MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<11:23, 318kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<09:14, 392kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<06:43, 538kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:45, 755kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<04:14, 840kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<04:12, 845kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<03:14, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:19, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:40, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<1:58:15, 29.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<1:21:55, 42.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<57:40, 59.2kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<41:32, 82.2kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<29:15, 116kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<20:24, 166kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<14:13, 236kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<13:07, 255kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<10:16, 325kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<07:27, 448kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<05:13, 632kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<04:32, 723kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<04:14, 773kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:11, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:16, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:39, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<04:30, 711kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<04:12, 762kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<03:11, 1.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:17, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<02:27, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<02:44, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<02:07, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:32, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:55, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<02:16, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:48, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:18, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:45, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<02:08, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:43, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:15, 2.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:42, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:05, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:38, 1.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:11, 2.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<00:53, 3.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<04:17, 667kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:55, 728kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:55, 974kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:05, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:29, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:37, 604kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<04:05, 682kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<03:02, 915kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:09, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:33, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:20, 627kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:52, 704kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:52, 943kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:03, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:10, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<02:26, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:55, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:23, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:39, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:57, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:33, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:07, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:30, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:30, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:05, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:27, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:45, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:23, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:00, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:22, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:41, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:20, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<00:58, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:19, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:38, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:19, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:57, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:17, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:38, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:18, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:57, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:16, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:33, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:14, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:53, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:12, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:29, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:11, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:51, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:10, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:26, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:07, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:50, 2.39MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:37, 3.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:53, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:23, 818kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:55, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:23, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:00, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:38, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:38, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:14, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:41, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:27, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:54, 951kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:33, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:08, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:49, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:37, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:00, 876kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:36, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:10, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:50, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:40, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:59, 843kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:35, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<01:09, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:49, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:46, 913kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:57, 826kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:33, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:07, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:48, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:44, 888kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:54, 811kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:30, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:05, 1.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:46, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<01:43, 855kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:52, 791kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:28, 1.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:03, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:45, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:48, 781kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:53, 746kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:28, 952kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:03, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:37, 829kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:43, 775kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:21, 985kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:58, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:41, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:19, 961kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:29, 852kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:10, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:51, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:36, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:25, 842kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:32, 782kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:12, 994kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:51, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:36, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:25, 794kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:30, 751kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:10, 958kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:50, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:35, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<01:24, 753kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<01:27, 728kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:08, 932kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:48, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<01:18, 765kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:21, 735kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:03, 940kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:31, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:10, 785kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:14, 748kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:57, 955kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:40, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:28, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:03, 811kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:09, 739kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:54, 945kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:38, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:26, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:54, 872kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:59, 799kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:46, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:32, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:53, 802kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:56, 757kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:44, 964kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:31, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:51, 759kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:53, 731kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:41, 933kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:28, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:19, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:38, 913kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:40, 852kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:31, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:22, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:45, 674kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:44, 690kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:33, 902kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:23, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:15, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:45, 585kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:42, 621kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:31, 818kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:21, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:14, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:33, 673kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:32, 688kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:24, 893kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:16, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:10, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:37, 485kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:33, 540kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:24, 715kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:16, 991kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:10, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:44, 318kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:36, 388kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:25, 526kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:16, 737kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:09, 1.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<01:06, 151kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:49, 200kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:34, 279kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:20, 394kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:10, 558kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:20, 289kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:16, 358kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:10, 489kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:05, 685kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 802kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 819kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 1.07MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 859/400000 [00:00<00:46, 8584.46it/s]  0%|          | 1700/400000 [00:00<00:46, 8530.62it/s]  1%|          | 2563/400000 [00:00<00:46, 8559.83it/s]  1%|          | 3389/400000 [00:00<00:46, 8466.19it/s]  1%|          | 4224/400000 [00:00<00:46, 8430.74it/s]  1%|          | 4989/400000 [00:00<00:48, 8178.13it/s]  1%|▏         | 5820/400000 [00:00<00:47, 8213.74it/s]  2%|▏         | 6663/400000 [00:00<00:47, 8274.80it/s]  2%|▏         | 7533/400000 [00:00<00:46, 8397.41it/s]  2%|▏         | 8389/400000 [00:01<00:46, 8443.04it/s]  2%|▏         | 9244/400000 [00:01<00:46, 8472.47it/s]  3%|▎         | 10078/400000 [00:01<00:46, 8430.42it/s]  3%|▎         | 10924/400000 [00:01<00:46, 8438.75it/s]  3%|▎         | 11780/400000 [00:01<00:45, 8472.72it/s]  3%|▎         | 12645/400000 [00:01<00:45, 8524.64it/s]  3%|▎         | 13507/400000 [00:01<00:45, 8551.83it/s]  4%|▎         | 14372/400000 [00:01<00:44, 8580.63it/s]  4%|▍         | 15234/400000 [00:01<00:44, 8591.45it/s]  4%|▍         | 16092/400000 [00:01<00:44, 8567.28it/s]  4%|▍         | 16961/400000 [00:02<00:44, 8602.17it/s]  4%|▍         | 17821/400000 [00:02<00:44, 8582.22it/s]  5%|▍         | 18679/400000 [00:02<00:44, 8503.27it/s]  5%|▍         | 19530/400000 [00:02<00:45, 8425.92it/s]  5%|▌         | 20394/400000 [00:02<00:44, 8487.87it/s]  5%|▌         | 21252/400000 [00:02<00:44, 8515.25it/s]  6%|▌         | 22114/400000 [00:02<00:44, 8545.60it/s]  6%|▌         | 22981/400000 [00:02<00:43, 8579.79it/s]  6%|▌         | 23840/400000 [00:02<00:43, 8575.04it/s]  6%|▌         | 24706/400000 [00:02<00:43, 8597.65it/s]  6%|▋         | 25566/400000 [00:03<00:43, 8595.18it/s]  7%|▋         | 26426/400000 [00:03<00:43, 8559.67it/s]  7%|▋         | 27283/400000 [00:03<00:43, 8533.77it/s]  7%|▋         | 28137/400000 [00:03<00:43, 8511.04it/s]  7%|▋         | 28991/400000 [00:03<00:43, 8519.57it/s]  7%|▋         | 29857/400000 [00:03<00:43, 8559.92it/s]  8%|▊         | 30714/400000 [00:03<00:43, 8559.07it/s]  8%|▊         | 31576/400000 [00:03<00:42, 8574.49it/s]  8%|▊         | 32434/400000 [00:03<00:43, 8516.67it/s]  8%|▊         | 33287/400000 [00:03<00:43, 8519.55it/s]  9%|▊         | 34149/400000 [00:04<00:42, 8548.04it/s]  9%|▉         | 35016/400000 [00:04<00:42, 8582.32it/s]  9%|▉         | 35875/400000 [00:04<00:42, 8534.28it/s]  9%|▉         | 36729/400000 [00:04<00:42, 8502.47it/s]  9%|▉         | 37596/400000 [00:04<00:42, 8549.58it/s] 10%|▉         | 38452/400000 [00:04<00:42, 8543.36it/s] 10%|▉         | 39314/400000 [00:04<00:42, 8564.41it/s] 10%|█         | 40171/400000 [00:04<00:42, 8513.32it/s] 10%|█         | 41023/400000 [00:04<00:42, 8507.39it/s] 10%|█         | 41888/400000 [00:04<00:41, 8546.91it/s] 11%|█         | 42747/400000 [00:05<00:41, 8546.90it/s] 11%|█         | 43613/400000 [00:05<00:41, 8577.76it/s] 11%|█         | 44471/400000 [00:05<00:41, 8568.81it/s] 11%|█▏        | 45328/400000 [00:05<00:41, 8501.05it/s] 12%|█▏        | 46188/400000 [00:05<00:41, 8527.90it/s] 12%|█▏        | 47046/400000 [00:05<00:41, 8542.04it/s] 12%|█▏        | 47905/400000 [00:05<00:41, 8555.47it/s] 12%|█▏        | 48772/400000 [00:05<00:40, 8582.36it/s] 12%|█▏        | 49631/400000 [00:05<00:40, 8551.51it/s] 13%|█▎        | 50487/400000 [00:05<00:41, 8504.56it/s] 13%|█▎        | 51338/400000 [00:06<00:41, 8417.09it/s] 13%|█▎        | 52196/400000 [00:06<00:41, 8465.11it/s] 13%|█▎        | 53059/400000 [00:06<00:40, 8512.48it/s] 13%|█▎        | 53911/400000 [00:06<00:40, 8447.85it/s] 14%|█▎        | 54771/400000 [00:06<00:40, 8492.19it/s] 14%|█▍        | 55646/400000 [00:06<00:40, 8565.50it/s] 14%|█▍        | 56518/400000 [00:06<00:39, 8609.09it/s] 14%|█▍        | 57391/400000 [00:06<00:39, 8643.68it/s] 15%|█▍        | 58256/400000 [00:06<00:39, 8635.32it/s] 15%|█▍        | 59120/400000 [00:06<00:39, 8630.19it/s] 15%|█▍        | 59984/400000 [00:07<00:39, 8588.28it/s] 15%|█▌        | 60854/400000 [00:07<00:39, 8620.49it/s] 15%|█▌        | 61730/400000 [00:07<00:39, 8660.31it/s] 16%|█▌        | 62597/400000 [00:07<00:39, 8646.22it/s] 16%|█▌        | 63472/400000 [00:07<00:38, 8676.44it/s] 16%|█▌        | 64346/400000 [00:07<00:38, 8694.80it/s] 16%|█▋        | 65216/400000 [00:07<00:38, 8689.81it/s] 17%|█▋        | 66086/400000 [00:07<00:38, 8680.38it/s] 17%|█▋        | 66955/400000 [00:07<00:38, 8665.19it/s] 17%|█▋        | 67829/400000 [00:07<00:38, 8687.32it/s] 17%|█▋        | 68699/400000 [00:08<00:38, 8688.71it/s] 17%|█▋        | 69568/400000 [00:08<00:38, 8686.41it/s] 18%|█▊        | 70438/400000 [00:08<00:37, 8688.31it/s] 18%|█▊        | 71307/400000 [00:08<00:38, 8546.62it/s] 18%|█▊        | 72170/400000 [00:08<00:38, 8570.64it/s] 18%|█▊        | 73033/400000 [00:08<00:38, 8586.65it/s] 18%|█▊        | 73894/400000 [00:08<00:37, 8592.03it/s] 19%|█▊        | 74754/400000 [00:08<00:38, 8541.73it/s] 19%|█▉        | 75613/400000 [00:08<00:37, 8555.52it/s] 19%|█▉        | 76486/400000 [00:08<00:37, 8606.78it/s] 19%|█▉        | 77347/400000 [00:09<00:37, 8594.74it/s] 20%|█▉        | 78207/400000 [00:09<00:37, 8555.27it/s] 20%|█▉        | 79074/400000 [00:09<00:37, 8587.52it/s] 20%|█▉        | 79939/400000 [00:09<00:37, 8604.43it/s] 20%|██        | 80807/400000 [00:09<00:37, 8626.15it/s] 20%|██        | 81678/400000 [00:09<00:36, 8650.47it/s] 21%|██        | 82547/400000 [00:09<00:36, 8661.11it/s] 21%|██        | 83423/400000 [00:09<00:36, 8688.80it/s] 21%|██        | 84294/400000 [00:09<00:36, 8693.51it/s] 21%|██▏       | 85164/400000 [00:09<00:36, 8664.82it/s] 22%|██▏       | 86031/400000 [00:10<00:36, 8658.78it/s] 22%|██▏       | 86897/400000 [00:10<00:36, 8654.77it/s] 22%|██▏       | 87763/400000 [00:10<00:36, 8654.90it/s] 22%|██▏       | 88629/400000 [00:10<00:36, 8611.32it/s] 22%|██▏       | 89500/400000 [00:10<00:35, 8640.46it/s] 23%|██▎       | 90365/400000 [00:10<00:35, 8618.01it/s] 23%|██▎       | 91227/400000 [00:10<00:35, 8587.74it/s] 23%|██▎       | 92094/400000 [00:10<00:35, 8610.50it/s] 23%|██▎       | 92963/400000 [00:10<00:35, 8633.76it/s] 23%|██▎       | 93842/400000 [00:10<00:35, 8677.34it/s] 24%|██▎       | 94720/400000 [00:11<00:35, 8706.11it/s] 24%|██▍       | 95594/400000 [00:11<00:34, 8714.15it/s] 24%|██▍       | 96466/400000 [00:11<00:35, 8612.09it/s] 24%|██▍       | 97335/400000 [00:11<00:35, 8632.46it/s] 25%|██▍       | 98211/400000 [00:11<00:34, 8669.86it/s] 25%|██▍       | 99082/400000 [00:11<00:34, 8680.52it/s] 25%|██▍       | 99951/400000 [00:11<00:34, 8646.12it/s] 25%|██▌       | 100816/400000 [00:11<00:34, 8634.53it/s] 25%|██▌       | 101680/400000 [00:11<00:35, 8520.60it/s] 26%|██▌       | 102537/400000 [00:11<00:34, 8532.85it/s] 26%|██▌       | 103401/400000 [00:12<00:34, 8564.42it/s] 26%|██▌       | 104258/400000 [00:12<00:34, 8562.96it/s] 26%|██▋       | 105122/400000 [00:12<00:34, 8585.06it/s] 26%|██▋       | 105981/400000 [00:12<00:34, 8571.61it/s] 27%|██▋       | 106847/400000 [00:12<00:34, 8597.92it/s] 27%|██▋       | 107707/400000 [00:12<00:34, 8564.08it/s] 27%|██▋       | 108580/400000 [00:12<00:33, 8611.12it/s] 27%|██▋       | 109442/400000 [00:12<00:33, 8596.97it/s] 28%|██▊       | 110302/400000 [00:12<00:33, 8580.19it/s] 28%|██▊       | 111161/400000 [00:12<00:33, 8571.17it/s] 28%|██▊       | 112021/400000 [00:13<00:33, 8577.52it/s] 28%|██▊       | 112879/400000 [00:13<00:33, 8480.08it/s] 28%|██▊       | 113728/400000 [00:13<00:34, 8370.82it/s] 29%|██▊       | 114575/400000 [00:13<00:33, 8399.57it/s] 29%|██▉       | 115416/400000 [00:13<00:34, 8214.70it/s] 29%|██▉       | 116278/400000 [00:13<00:34, 8329.67it/s] 29%|██▉       | 117136/400000 [00:13<00:33, 8390.37it/s] 29%|██▉       | 117976/400000 [00:13<00:33, 8375.28it/s] 30%|██▉       | 118815/400000 [00:13<00:33, 8343.20it/s] 30%|██▉       | 119668/400000 [00:13<00:33, 8397.05it/s] 30%|███       | 120532/400000 [00:14<00:33, 8466.86it/s] 30%|███       | 121380/400000 [00:14<00:33, 8260.18it/s] 31%|███       | 122240/400000 [00:14<00:33, 8358.21it/s] 31%|███       | 123103/400000 [00:14<00:32, 8437.86it/s] 31%|███       | 123948/400000 [00:14<00:32, 8430.49it/s] 31%|███       | 124812/400000 [00:14<00:32, 8491.17it/s] 31%|███▏      | 125680/400000 [00:14<00:32, 8544.51it/s] 32%|███▏      | 126539/400000 [00:14<00:31, 8554.87it/s] 32%|███▏      | 127395/400000 [00:14<00:31, 8552.61it/s] 32%|███▏      | 128251/400000 [00:15<00:31, 8535.95it/s] 32%|███▏      | 129105/400000 [00:15<00:31, 8531.26it/s] 32%|███▏      | 129959/400000 [00:15<00:31, 8470.53it/s] 33%|███▎      | 130815/400000 [00:15<00:31, 8494.64it/s] 33%|███▎      | 131665/400000 [00:15<00:31, 8489.00it/s] 33%|███▎      | 132529/400000 [00:15<00:31, 8531.74it/s] 33%|███▎      | 133401/400000 [00:15<00:31, 8585.85it/s] 34%|███▎      | 134260/400000 [00:15<00:30, 8586.69it/s] 34%|███▍      | 135125/400000 [00:15<00:30, 8604.82it/s] 34%|███▍      | 135986/400000 [00:15<00:30, 8575.84it/s] 34%|███▍      | 136852/400000 [00:16<00:30, 8598.60it/s] 34%|███▍      | 137719/400000 [00:16<00:30, 8617.18it/s] 35%|███▍      | 138585/400000 [00:16<00:30, 8628.81it/s] 35%|███▍      | 139448/400000 [00:16<00:30, 8556.09it/s] 35%|███▌      | 140304/400000 [00:16<00:30, 8531.43it/s] 35%|███▌      | 141175/400000 [00:16<00:30, 8583.52it/s] 36%|███▌      | 142034/400000 [00:16<00:30, 8574.00it/s] 36%|███▌      | 142892/400000 [00:16<00:30, 8494.64it/s] 36%|███▌      | 143742/400000 [00:16<00:30, 8494.57it/s] 36%|███▌      | 144596/400000 [00:16<00:30, 8507.42it/s] 36%|███▋      | 145461/400000 [00:17<00:29, 8546.85it/s] 37%|███▋      | 146324/400000 [00:17<00:29, 8571.34it/s] 37%|███▋      | 147189/400000 [00:17<00:29, 8592.96it/s] 37%|███▋      | 148058/400000 [00:17<00:29, 8619.46it/s] 37%|███▋      | 148921/400000 [00:17<00:29, 8594.11it/s] 37%|███▋      | 149781/400000 [00:17<00:29, 8437.44it/s] 38%|███▊      | 150647/400000 [00:17<00:29, 8502.54it/s] 38%|███▊      | 151515/400000 [00:17<00:29, 8552.59it/s] 38%|███▊      | 152376/400000 [00:17<00:28, 8567.35it/s] 38%|███▊      | 153234/400000 [00:17<00:28, 8524.99it/s] 39%|███▊      | 154090/400000 [00:18<00:28, 8533.97it/s] 39%|███▊      | 154957/400000 [00:18<00:28, 8573.32it/s] 39%|███▉      | 155821/400000 [00:18<00:28, 8590.95it/s] 39%|███▉      | 156681/400000 [00:18<00:28, 8579.83it/s] 39%|███▉      | 157556/400000 [00:18<00:28, 8627.80it/s] 40%|███▉      | 158430/400000 [00:18<00:27, 8659.25it/s] 40%|███▉      | 159297/400000 [00:18<00:27, 8638.66it/s] 40%|████      | 160173/400000 [00:18<00:27, 8672.10it/s] 40%|████      | 161049/400000 [00:18<00:27, 8696.62it/s] 40%|████      | 161921/400000 [00:18<00:27, 8701.98it/s] 41%|████      | 162798/400000 [00:19<00:27, 8721.36it/s] 41%|████      | 163678/400000 [00:19<00:27, 8742.65it/s] 41%|████      | 164553/400000 [00:19<00:27, 8625.30it/s] 41%|████▏     | 165424/400000 [00:19<00:27, 8647.87it/s] 42%|████▏     | 166290/400000 [00:19<00:27, 8614.07it/s] 42%|████▏     | 167161/400000 [00:19<00:26, 8640.64it/s] 42%|████▏     | 168036/400000 [00:19<00:26, 8671.45it/s] 42%|████▏     | 168904/400000 [00:19<00:26, 8645.95it/s] 42%|████▏     | 169780/400000 [00:19<00:26, 8678.14it/s] 43%|████▎     | 170648/400000 [00:19<00:26, 8641.06it/s] 43%|████▎     | 171513/400000 [00:20<00:26, 8631.50it/s] 43%|████▎     | 172377/400000 [00:20<00:26, 8558.21it/s] 43%|████▎     | 173242/400000 [00:20<00:26, 8582.93it/s] 44%|████▎     | 174114/400000 [00:20<00:26, 8622.21it/s] 44%|████▎     | 174987/400000 [00:20<00:26, 8653.39it/s] 44%|████▍     | 175862/400000 [00:20<00:25, 8681.51it/s] 44%|████▍     | 176731/400000 [00:20<00:25, 8664.43it/s] 44%|████▍     | 177605/400000 [00:20<00:25, 8686.35it/s] 45%|████▍     | 178481/400000 [00:20<00:25, 8707.12it/s] 45%|████▍     | 179357/400000 [00:20<00:25, 8721.24it/s] 45%|████▌     | 180231/400000 [00:21<00:25, 8725.92it/s] 45%|████▌     | 181107/400000 [00:21<00:25, 8735.35it/s] 45%|████▌     | 181986/400000 [00:21<00:24, 8750.25it/s] 46%|████▌     | 182862/400000 [00:21<00:24, 8751.53it/s] 46%|████▌     | 183739/400000 [00:21<00:24, 8755.91it/s] 46%|████▌     | 184615/400000 [00:21<00:24, 8757.11it/s] 46%|████▋     | 185491/400000 [00:21<00:24, 8745.09it/s] 47%|████▋     | 186366/400000 [00:21<00:24, 8643.63it/s] 47%|████▋     | 187231/400000 [00:21<00:24, 8611.01it/s] 47%|████▋     | 188094/400000 [00:21<00:24, 8615.36it/s] 47%|████▋     | 188962/400000 [00:22<00:24, 8633.27it/s] 47%|████▋     | 189837/400000 [00:22<00:24, 8667.85it/s] 48%|████▊     | 190715/400000 [00:22<00:24, 8700.67it/s] 48%|████▊     | 191591/400000 [00:22<00:23, 8715.62it/s] 48%|████▊     | 192463/400000 [00:22<00:23, 8710.06it/s] 48%|████▊     | 193335/400000 [00:22<00:23, 8690.94it/s] 49%|████▊     | 194205/400000 [00:22<00:23, 8679.20it/s] 49%|████▉     | 195081/400000 [00:22<00:23, 8702.75it/s] 49%|████▉     | 195959/400000 [00:22<00:23, 8725.09it/s] 49%|████▉     | 196832/400000 [00:22<00:23, 8716.57it/s] 49%|████▉     | 197704/400000 [00:23<00:23, 8680.23it/s] 50%|████▉     | 198573/400000 [00:23<00:23, 8610.27it/s] 50%|████▉     | 199449/400000 [00:23<00:23, 8654.21it/s] 50%|█████     | 200328/400000 [00:23<00:22, 8694.17it/s] 50%|█████     | 201198/400000 [00:23<00:23, 8629.22it/s] 51%|█████     | 202062/400000 [00:23<00:23, 8581.17it/s] 51%|█████     | 202937/400000 [00:23<00:22, 8629.37it/s] 51%|█████     | 203816/400000 [00:23<00:22, 8676.93it/s] 51%|█████     | 204696/400000 [00:23<00:22, 8711.36it/s] 51%|█████▏    | 205574/400000 [00:23<00:22, 8729.01it/s] 52%|█████▏    | 206448/400000 [00:24<00:22, 8724.20it/s] 52%|█████▏    | 207325/400000 [00:24<00:22, 8735.42it/s] 52%|█████▏    | 208201/400000 [00:24<00:21, 8741.21it/s] 52%|█████▏    | 209076/400000 [00:24<00:21, 8741.29it/s] 52%|█████▏    | 209952/400000 [00:24<00:21, 8745.52it/s] 53%|█████▎    | 210827/400000 [00:24<00:21, 8737.53it/s] 53%|█████▎    | 211704/400000 [00:24<00:21, 8745.49it/s] 53%|█████▎    | 212580/400000 [00:24<00:21, 8747.90it/s] 53%|█████▎    | 213458/400000 [00:24<00:21, 8756.30it/s] 54%|█████▎    | 214334/400000 [00:24<00:21, 8756.03it/s] 54%|█████▍    | 215210/400000 [00:25<00:21, 8738.75it/s] 54%|█████▍    | 216086/400000 [00:25<00:21, 8744.15it/s] 54%|█████▍    | 216961/400000 [00:25<00:21, 8682.66it/s] 54%|█████▍    | 217839/400000 [00:25<00:20, 8709.57it/s] 55%|█████▍    | 218715/400000 [00:25<00:20, 8723.07it/s] 55%|█████▍    | 219588/400000 [00:25<00:20, 8686.23it/s] 55%|█████▌    | 220463/400000 [00:25<00:20, 8704.60it/s] 55%|█████▌    | 221334/400000 [00:25<00:20, 8570.96it/s] 56%|█████▌    | 222211/400000 [00:25<00:20, 8627.99it/s] 56%|█████▌    | 223089/400000 [00:25<00:20, 8671.82it/s] 56%|█████▌    | 223959/400000 [00:26<00:20, 8677.59it/s] 56%|█████▌    | 224831/400000 [00:26<00:20, 8689.33it/s] 56%|█████▋    | 225705/400000 [00:26<00:20, 8701.90it/s] 57%|█████▋    | 226578/400000 [00:26<00:19, 8710.15it/s] 57%|█████▋    | 227455/400000 [00:26<00:19, 8727.10it/s] 57%|█████▋    | 228328/400000 [00:26<00:19, 8725.84it/s] 57%|█████▋    | 229201/400000 [00:26<00:19, 8644.83it/s] 58%|█████▊    | 230078/400000 [00:26<00:19, 8681.59it/s] 58%|█████▊    | 230952/400000 [00:26<00:19, 8699.02it/s] 58%|█████▊    | 231828/400000 [00:26<00:19, 8715.95it/s] 58%|█████▊    | 232701/400000 [00:27<00:19, 8718.75it/s] 58%|█████▊    | 233573/400000 [00:27<00:19, 8696.03it/s] 59%|█████▊    | 234443/400000 [00:27<00:19, 8695.57it/s] 59%|█████▉    | 235314/400000 [00:27<00:18, 8697.50it/s] 59%|█████▉    | 236189/400000 [00:27<00:18, 8712.01it/s] 59%|█████▉    | 237061/400000 [00:27<00:18, 8703.23it/s] 59%|█████▉    | 237932/400000 [00:27<00:18, 8672.01it/s] 60%|█████▉    | 238806/400000 [00:27<00:18, 8692.05it/s] 60%|█████▉    | 239676/400000 [00:27<00:18, 8621.35it/s] 60%|██████    | 240549/400000 [00:27<00:18, 8651.35it/s] 60%|██████    | 241421/400000 [00:28<00:18, 8671.23it/s] 61%|██████    | 242289/400000 [00:28<00:18, 8671.80it/s] 61%|██████    | 243157/400000 [00:28<00:18, 8598.86it/s] 61%|██████    | 244020/400000 [00:28<00:18, 8605.09it/s] 61%|██████    | 244897/400000 [00:28<00:17, 8650.96it/s] 61%|██████▏   | 245767/400000 [00:28<00:17, 8663.26it/s] 62%|██████▏   | 246634/400000 [00:28<00:17, 8654.36it/s] 62%|██████▏   | 247513/400000 [00:28<00:17, 8692.79it/s] 62%|██████▏   | 248388/400000 [00:28<00:17, 8709.28it/s] 62%|██████▏   | 249263/400000 [00:28<00:17, 8719.19it/s] 63%|██████▎   | 250138/400000 [00:29<00:17, 8727.89it/s] 63%|██████▎   | 251011/400000 [00:29<00:17, 8680.74it/s] 63%|██████▎   | 251883/400000 [00:29<00:17, 8690.24it/s] 63%|██████▎   | 252755/400000 [00:29<00:16, 8699.06it/s] 63%|██████▎   | 253630/400000 [00:29<00:16, 8713.77it/s] 64%|██████▎   | 254502/400000 [00:29<00:16, 8710.99it/s] 64%|██████▍   | 255374/400000 [00:29<00:16, 8706.56it/s] 64%|██████▍   | 256251/400000 [00:29<00:16, 8724.03it/s] 64%|██████▍   | 257126/400000 [00:29<00:16, 8730.70it/s] 65%|██████▍   | 258004/400000 [00:29<00:16, 8745.23it/s] 65%|██████▍   | 258880/400000 [00:30<00:16, 8747.04it/s] 65%|██████▍   | 259755/400000 [00:30<00:16, 8624.21it/s] 65%|██████▌   | 260626/400000 [00:30<00:16, 8647.68it/s] 65%|██████▌   | 261500/400000 [00:30<00:15, 8673.06it/s] 66%|██████▌   | 262378/400000 [00:30<00:15, 8702.39it/s] 66%|██████▌   | 263255/400000 [00:30<00:15, 8719.80it/s] 66%|██████▌   | 264128/400000 [00:30<00:15, 8674.64it/s] 66%|██████▌   | 264996/400000 [00:30<00:15, 8674.79it/s] 66%|██████▋   | 265864/400000 [00:30<00:15, 8672.58it/s] 67%|██████▋   | 266735/400000 [00:30<00:15, 8682.92it/s] 67%|██████▋   | 267609/400000 [00:31<00:15, 8697.20it/s] 67%|██████▋   | 268479/400000 [00:31<00:15, 8619.77it/s] 67%|██████▋   | 269357/400000 [00:31<00:15, 8664.77it/s] 68%|██████▊   | 270231/400000 [00:31<00:14, 8684.68it/s] 68%|██████▊   | 271105/400000 [00:31<00:14, 8700.65it/s] 68%|██████▊   | 271982/400000 [00:31<00:14, 8718.87it/s] 68%|██████▊   | 272854/400000 [00:31<00:14, 8698.91it/s] 68%|██████▊   | 273727/400000 [00:31<00:14, 8706.94it/s] 69%|██████▊   | 274607/400000 [00:31<00:14, 8732.71it/s] 69%|██████▉   | 275481/400000 [00:31<00:14, 8730.62it/s] 69%|██████▉   | 276358/400000 [00:32<00:14, 8740.69it/s] 69%|██████▉   | 277233/400000 [00:32<00:14, 8640.86it/s] 70%|██████▉   | 278110/400000 [00:32<00:14, 8677.66it/s] 70%|██████▉   | 278987/400000 [00:32<00:13, 8703.72it/s] 70%|██████▉   | 279859/400000 [00:32<00:13, 8706.73it/s] 70%|███████   | 280737/400000 [00:32<00:13, 8727.36it/s] 70%|███████   | 281610/400000 [00:32<00:13, 8688.99it/s] 71%|███████   | 282482/400000 [00:32<00:13, 8696.87it/s] 71%|███████   | 283352/400000 [00:32<00:13, 8673.66it/s] 71%|███████   | 284220/400000 [00:33<00:13, 8462.33it/s] 71%|███████▏  | 285098/400000 [00:33<00:13, 8554.64it/s] 71%|███████▏  | 285955/400000 [00:33<00:13, 8544.08it/s] 72%|███████▏  | 286833/400000 [00:33<00:13, 8610.63it/s] 72%|███████▏  | 287712/400000 [00:33<00:12, 8663.21it/s] 72%|███████▏  | 288585/400000 [00:33<00:12, 8682.86it/s] 72%|███████▏  | 289461/400000 [00:33<00:12, 8703.28it/s] 73%|███████▎  | 290332/400000 [00:33<00:12, 8650.72it/s] 73%|███████▎  | 291208/400000 [00:33<00:12, 8680.93it/s] 73%|███████▎  | 292085/400000 [00:33<00:12, 8706.11it/s] 73%|███████▎  | 292956/400000 [00:34<00:12, 8688.99it/s] 73%|███████▎  | 293831/400000 [00:34<00:12, 8706.23it/s] 74%|███████▎  | 294702/400000 [00:34<00:12, 8659.84it/s] 74%|███████▍  | 295575/400000 [00:34<00:12, 8678.53it/s] 74%|███████▍  | 296446/400000 [00:34<00:11, 8687.77it/s] 74%|███████▍  | 297315/400000 [00:34<00:11, 8671.24it/s] 75%|███████▍  | 298183/400000 [00:34<00:11, 8577.22it/s] 75%|███████▍  | 299041/400000 [00:34<00:11, 8515.58it/s] 75%|███████▍  | 299917/400000 [00:34<00:11, 8587.10it/s] 75%|███████▌  | 300793/400000 [00:34<00:11, 8637.32it/s] 75%|███████▌  | 301659/400000 [00:35<00:11, 8643.57it/s] 76%|███████▌  | 302538/400000 [00:35<00:11, 8684.05it/s] 76%|███████▌  | 303407/400000 [00:35<00:11, 8433.14it/s] 76%|███████▌  | 304280/400000 [00:35<00:11, 8519.89it/s] 76%|███████▋  | 305150/400000 [00:35<00:11, 8571.29it/s] 77%|███████▋  | 306021/400000 [00:35<00:10, 8612.26it/s] 77%|███████▋  | 306896/400000 [00:35<00:10, 8651.92it/s] 77%|███████▋  | 307762/400000 [00:35<00:10, 8636.63it/s] 77%|███████▋  | 308636/400000 [00:35<00:10, 8667.36it/s] 77%|███████▋  | 309506/400000 [00:35<00:10, 8676.69it/s] 78%|███████▊  | 310374/400000 [00:36<00:10, 8617.02it/s] 78%|███████▊  | 311240/400000 [00:36<00:10, 8629.75it/s] 78%|███████▊  | 312104/400000 [00:36<00:10, 8627.61it/s] 78%|███████▊  | 312967/400000 [00:36<00:10, 8463.02it/s] 78%|███████▊  | 313837/400000 [00:36<00:10, 8532.23it/s] 79%|███████▊  | 314713/400000 [00:36<00:09, 8598.76it/s] 79%|███████▉  | 315585/400000 [00:36<00:09, 8632.54it/s] 79%|███████▉  | 316459/400000 [00:36<00:09, 8662.84it/s] 79%|███████▉  | 317335/400000 [00:36<00:09, 8689.47it/s] 80%|███████▉  | 318205/400000 [00:36<00:09, 8690.59it/s] 80%|███████▉  | 319081/400000 [00:37<00:09, 8709.56it/s] 80%|███████▉  | 319960/400000 [00:37<00:09, 8730.96it/s] 80%|████████  | 320839/400000 [00:37<00:09, 8747.79it/s] 80%|████████  | 321714/400000 [00:37<00:08, 8734.43it/s] 81%|████████  | 322592/400000 [00:37<00:08, 8747.14it/s] 81%|████████  | 323469/400000 [00:37<00:08, 8752.45it/s] 81%|████████  | 324348/400000 [00:37<00:08, 8762.43it/s] 81%|████████▏ | 325225/400000 [00:37<00:08, 8761.77it/s] 82%|████████▏ | 326102/400000 [00:37<00:08, 8737.50it/s] 82%|████████▏ | 326976/400000 [00:37<00:08, 8625.93it/s] 82%|████████▏ | 327851/400000 [00:38<00:08, 8660.29it/s] 82%|████████▏ | 328718/400000 [00:38<00:08, 8607.36it/s] 82%|████████▏ | 329595/400000 [00:38<00:08, 8653.92it/s] 83%|████████▎ | 330462/400000 [00:38<00:08, 8657.34it/s] 83%|████████▎ | 331338/400000 [00:38<00:07, 8685.28it/s] 83%|████████▎ | 332218/400000 [00:38<00:07, 8717.78it/s] 83%|████████▎ | 333094/400000 [00:38<00:07, 8728.81it/s] 83%|████████▎ | 333971/400000 [00:38<00:07, 8738.86it/s] 84%|████████▎ | 334845/400000 [00:38<00:07, 8710.31it/s] 84%|████████▍ | 335721/400000 [00:38<00:07, 8724.93it/s] 84%|████████▍ | 336598/400000 [00:39<00:07, 8736.75it/s] 84%|████████▍ | 337474/400000 [00:39<00:07, 8741.53it/s] 85%|████████▍ | 338349/400000 [00:39<00:07, 8734.53it/s] 85%|████████▍ | 339223/400000 [00:39<00:06, 8704.52it/s] 85%|████████▌ | 340097/400000 [00:39<00:06, 8712.18it/s] 85%|████████▌ | 340972/400000 [00:39<00:06, 8720.75it/s] 85%|████████▌ | 341845/400000 [00:39<00:06, 8653.24it/s] 86%|████████▌ | 342711/400000 [00:39<00:06, 8453.75it/s] 86%|████████▌ | 343558/400000 [00:39<00:06, 8386.62it/s] 86%|████████▌ | 344432/400000 [00:39<00:06, 8488.67it/s] 86%|████████▋ | 345303/400000 [00:40<00:06, 8553.80it/s] 87%|████████▋ | 346169/400000 [00:40<00:06, 8584.53it/s] 87%|████████▋ | 347042/400000 [00:40<00:06, 8626.01it/s] 87%|████████▋ | 347906/400000 [00:40<00:06, 8620.36it/s] 87%|████████▋ | 348773/400000 [00:40<00:05, 8632.83it/s] 87%|████████▋ | 349645/400000 [00:40<00:05, 8656.12it/s] 88%|████████▊ | 350519/400000 [00:40<00:05, 8678.85it/s] 88%|████████▊ | 351396/400000 [00:40<00:05, 8704.79it/s] 88%|████████▊ | 352267/400000 [00:40<00:05, 8480.45it/s] 88%|████████▊ | 353137/400000 [00:40<00:05, 8544.92it/s] 89%|████████▊ | 354014/400000 [00:41<00:05, 8609.54it/s] 89%|████████▊ | 354890/400000 [00:41<00:05, 8651.55it/s] 89%|████████▉ | 355765/400000 [00:41<00:05, 8678.21it/s] 89%|████████▉ | 356634/400000 [00:41<00:05, 8582.72it/s] 89%|████████▉ | 357493/400000 [00:41<00:05, 8433.64it/s] 90%|████████▉ | 358355/400000 [00:41<00:04, 8487.71it/s] 90%|████████▉ | 359221/400000 [00:41<00:04, 8536.68it/s] 90%|█████████ | 360096/400000 [00:41<00:04, 8598.76it/s] 90%|█████████ | 360959/400000 [00:41<00:04, 8605.43it/s] 90%|█████████ | 361838/400000 [00:41<00:04, 8657.36it/s] 91%|█████████ | 362714/400000 [00:42<00:04, 8687.17it/s] 91%|█████████ | 363588/400000 [00:42<00:04, 8700.62it/s] 91%|█████████ | 364464/400000 [00:42<00:04, 8717.17it/s] 91%|█████████▏| 365336/400000 [00:42<00:03, 8713.17it/s] 92%|█████████▏| 366211/400000 [00:42<00:03, 8722.27it/s] 92%|█████████▏| 367084/400000 [00:42<00:03, 8720.36it/s] 92%|█████████▏| 367957/400000 [00:42<00:03, 8712.90it/s] 92%|█████████▏| 368833/400000 [00:42<00:03, 8726.14it/s] 92%|█████████▏| 369706/400000 [00:42<00:03, 8629.77it/s] 93%|█████████▎| 370579/400000 [00:42<00:03, 8659.26it/s] 93%|█████████▎| 371446/400000 [00:43<00:03, 8495.20it/s] 93%|█████████▎| 372314/400000 [00:43<00:03, 8549.39it/s] 93%|█████████▎| 373185/400000 [00:43<00:03, 8596.88it/s] 94%|█████████▎| 374055/400000 [00:43<00:03, 8625.70it/s] 94%|█████████▎| 374918/400000 [00:43<00:02, 8624.41it/s] 94%|█████████▍| 375792/400000 [00:43<00:02, 8656.32it/s] 94%|█████████▍| 376666/400000 [00:43<00:02, 8678.98it/s] 94%|█████████▍| 377541/400000 [00:43<00:02, 8698.59it/s] 95%|█████████▍| 378411/400000 [00:43<00:02, 8666.37it/s] 95%|█████████▍| 379288/400000 [00:44<00:02, 8695.18it/s] 95%|█████████▌| 380162/400000 [00:44<00:02, 8706.24it/s] 95%|█████████▌| 381033/400000 [00:44<00:02, 8519.04it/s] 95%|█████████▌| 381886/400000 [00:44<00:02, 8471.28it/s] 96%|█████████▌| 382758/400000 [00:44<00:02, 8541.37it/s] 96%|█████████▌| 383636/400000 [00:44<00:01, 8609.84it/s] 96%|█████████▌| 384514/400000 [00:44<00:01, 8659.65it/s] 96%|█████████▋| 385381/400000 [00:44<00:01, 8566.72it/s] 97%|█████████▋| 386252/400000 [00:44<00:01, 8606.22it/s] 97%|█████████▋| 387127/400000 [00:44<00:01, 8648.01it/s] 97%|█████████▋| 388003/400000 [00:45<00:01, 8679.97it/s] 97%|█████████▋| 388878/400000 [00:45<00:01, 8699.71it/s] 97%|█████████▋| 389751/400000 [00:45<00:01, 8707.64it/s] 98%|█████████▊| 390625/400000 [00:45<00:01, 8714.92it/s] 98%|█████████▊| 391499/400000 [00:45<00:00, 8722.31it/s] 98%|█████████▊| 392372/400000 [00:45<00:00, 8679.20it/s] 98%|█████████▊| 393243/400000 [00:45<00:00, 8687.02it/s] 99%|█████████▊| 394115/400000 [00:45<00:00, 8693.92it/s] 99%|█████████▊| 394988/400000 [00:45<00:00, 8703.05it/s] 99%|█████████▉| 395859/400000 [00:45<00:00, 8666.95it/s] 99%|█████████▉| 396736/400000 [00:46<00:00, 8695.58it/s] 99%|█████████▉| 397606/400000 [00:46<00:00, 8603.26it/s]100%|█████████▉| 398475/400000 [00:46<00:00, 8628.51it/s]100%|█████████▉| 399339/400000 [00:46<00:00, 8610.88it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8619.80it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd9476a0cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011089248206994053 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011158671466801877 	 Accuracy: 55

  model saves at 55% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15853 out of table with 15798 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15853 out of table with 15798 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

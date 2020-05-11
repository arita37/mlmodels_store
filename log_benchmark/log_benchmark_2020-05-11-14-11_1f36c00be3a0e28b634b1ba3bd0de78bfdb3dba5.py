
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5dc6e38f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 14:12:14.599109
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 14:12:14.602929
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 14:12:14.606257
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 14:12:14.609474
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5dd2bfc400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353102.7500
Epoch 2/10

1/1 [==============================] - 0s 99ms/step - loss: 213190.1875
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 109438.5078
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 54157.5312
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 30359.9199
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 19105.3691
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 13104.6553
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 9603.5225
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 7417.3091
Epoch 10/10

1/1 [==============================] - 0s 110ms/step - loss: 5985.9736

  #### Inference Need return ypred, ytrue ######################### 
[[-0.61921036 -0.46088687 -0.74771523 -0.38624477 -1.4550862  -0.07320666
  -0.34908932 -0.9021932  -0.89320004 -1.0091058   1.3735778   0.7554565
   1.0331112  -0.6971381   0.6062193  -1.6641399  -0.29726812 -0.8308284
  -1.1583724   1.2785703  -0.18330348  0.62723035 -0.21564293  0.16393805
  -0.26603252 -0.30773038 -0.21608356 -1.5202461   1.2745869   0.0189994
  -0.9538658   1.3813561   0.16305429 -0.28390723  0.8882846  -0.05120236
  -0.81128514  0.9451567  -0.2738045   0.5483977   0.6887877  -0.35889965
  -0.61638707 -0.68606913 -0.03148399  0.1041047  -0.92991406 -1.3518167
   0.08949512 -1.1978323  -0.34725583  2.0270617   0.48184842  1.6270796
  -0.63649267 -0.1490857   0.11077151 -0.04024673  1.3761754   1.6897483
  -0.29896477  9.517302    7.3208833   8.78082     6.8541      8.694327
   9.672948    6.310332    8.710857    7.749713    6.507025   10.403139
   8.752154    8.928356    9.47923     7.404144    7.6654806  10.119331
   7.565447   10.24218     7.4561124   9.358673    9.424216    7.0275583
   8.807618    7.9210143   8.065238    7.477369    8.051577    8.350525
  10.959303   10.4379225   8.93477    10.586614    8.546581   10.470285
   8.911812    8.376398    7.8459024   7.1326537   8.170875    5.8061028
   8.031436   10.250346    8.358545    9.6164055  10.65059     7.748518
   7.4834404   8.646919   10.025396    7.71373     7.6344585   8.201776
   9.053088    9.7725      8.885789    9.049821    9.389811    7.8373847
   0.33588043 -0.46945158  1.4811587   0.752231   -2.559391    0.9881528
  -1.6257149   0.23079967 -0.17446285 -1.0540621   0.50105715  0.92243695
   0.22147286  1.5504305  -0.83296025  1.2344565   0.29941526 -1.9571266
  -2.285735   -0.50000185  1.3570476   0.28356707 -1.1169121   0.56411844
   0.7662654  -0.31476235  0.97586876 -0.589555   -1.7076627   0.08092475
   0.6811397  -0.15282616  0.7779134  -0.7642908  -0.9285136   0.24840519
   2.0414157   0.49602443  1.5820242   1.4236071   2.4327216  -1.136837
   0.58671427  0.16832873  0.18519521  0.80600786  0.49973288  0.18078291
  -0.37463045 -0.04347223 -0.02218613  0.19203818  0.4887193  -0.74142605
  -0.27436477  1.2887981   0.7561309   1.8934264   0.7335534  -1.4947463
   1.1529431   2.341628    0.42018598  0.83857435  2.9660373   1.4516873
   0.19361985  2.019648    2.014089    1.2855475   0.9890063   1.7842333
   0.33494925  1.4861134   1.0036263   0.3399791   0.36938882  1.5495213
   0.907967    0.6453086   2.085562    0.6528136   0.4974283   0.41791803
   1.3984976   1.205943    0.48922914  0.21816659  2.6799428   1.1791914
   3.4435568   0.51097274  0.25663215  0.6089591   2.690825    0.3412767
   1.2558098   1.9758022   0.75788486  1.8591921   2.8072476   0.4855007
   0.40462238  0.27834952  2.9630747   0.8713553   2.462584    1.0717949
   0.2847185   1.4807677   0.45074904  0.9059689   0.47051167  0.12789053
   1.0893717   0.6770402   2.0340276   0.15597963  0.18030477  1.8100147
   0.6147169   9.088016    8.784445    7.408897    8.6034355  10.3260145
   8.89928     9.0690365   9.409549    9.333962    7.9082127   8.193755
   9.031714    7.7720933   8.107303    8.994238    7.28405     9.503084
   9.584363    8.949902    9.131939    8.375027    8.482453   10.466183
   7.361651    8.288028    7.7010646   7.974819    8.650857    9.347001
  10.573946    9.231349    8.6102295   8.27393     8.762155    9.401578
   9.214334    9.5837      6.843559    7.5233965   9.869358    6.953307
   8.580724    9.374909    7.7735314  10.261656    9.718267    8.191908
  10.166513    8.410497    8.479889    7.4236965   8.199684    7.670541
   9.102401    9.599272    8.582251    9.005479    9.034383    8.559224
   0.77679574  1.0889168   2.5651155   0.4489677   1.2503898   0.607073
   1.8129965   1.1096096   1.7156181   0.25623447  1.4096078   0.89968395
   1.8051057   0.452205    1.4695528   1.4010296   0.28153414  2.8504162
   0.83372074  0.5473572   1.1559086   1.2614346   0.9205372   1.6291653
   1.058853    2.4138403   2.081092    1.9407758   1.3377606   1.723956
   0.1953497   1.5909874   0.33916837  0.91999984  0.31813908  1.039965
   0.51919484  1.1655538   0.27327013  0.43634617  0.09396708  1.4465032
   0.2332133   0.30614728  4.0828953   2.435647    1.0331877   0.11030477
   0.99670607  2.3464031   0.9413351   0.7877738   0.5530682   0.17425048
   0.69422454  1.3750606   1.6259999   0.33954692  1.5265716   0.48982787
  -5.065992    4.688688   -6.5160813 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 14:12:24.118723
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2081
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 14:12:24.122620
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8712.12
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 14:12:24.126858
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9608
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 14:12:24.130580
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -779.232
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140040406353624
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140039444745632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140039444746136
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140039444324816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140039444325320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140039444325824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5dcea7feb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.661563
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.623049
grad_step = 000002, loss = 0.587038
grad_step = 000003, loss = 0.551727
grad_step = 000004, loss = 0.511429
grad_step = 000005, loss = 0.467493
grad_step = 000006, loss = 0.428833
grad_step = 000007, loss = 0.406504
grad_step = 000008, loss = 0.389210
grad_step = 000009, loss = 0.355792
grad_step = 000010, loss = 0.322407
grad_step = 000011, loss = 0.298205
grad_step = 000012, loss = 0.280211
grad_step = 000013, loss = 0.263729
grad_step = 000014, loss = 0.245109
grad_step = 000015, loss = 0.225103
grad_step = 000016, loss = 0.208457
grad_step = 000017, loss = 0.197297
grad_step = 000018, loss = 0.184616
grad_step = 000019, loss = 0.169441
grad_step = 000020, loss = 0.157065
grad_step = 000021, loss = 0.145722
grad_step = 000022, loss = 0.134431
grad_step = 000023, loss = 0.124715
grad_step = 000024, loss = 0.116579
grad_step = 000025, loss = 0.108434
grad_step = 000026, loss = 0.099825
grad_step = 000027, loss = 0.091716
grad_step = 000028, loss = 0.084852
grad_step = 000029, loss = 0.078447
grad_step = 000030, loss = 0.071925
grad_step = 000031, loss = 0.066049
grad_step = 000032, loss = 0.060980
grad_step = 000033, loss = 0.056042
grad_step = 000034, loss = 0.051163
grad_step = 000035, loss = 0.046864
grad_step = 000036, loss = 0.043202
grad_step = 000037, loss = 0.039774
grad_step = 000038, loss = 0.036520
grad_step = 000039, loss = 0.033584
grad_step = 000040, loss = 0.030855
grad_step = 000041, loss = 0.028210
grad_step = 000042, loss = 0.025789
grad_step = 000043, loss = 0.023610
grad_step = 000044, loss = 0.021485
grad_step = 000045, loss = 0.019477
grad_step = 000046, loss = 0.017740
grad_step = 000047, loss = 0.016161
grad_step = 000048, loss = 0.014657
grad_step = 000049, loss = 0.013330
grad_step = 000050, loss = 0.012102
grad_step = 000051, loss = 0.010902
grad_step = 000052, loss = 0.009872
grad_step = 000053, loss = 0.009003
grad_step = 000054, loss = 0.008180
grad_step = 000055, loss = 0.007446
grad_step = 000056, loss = 0.006800
grad_step = 000057, loss = 0.006177
grad_step = 000058, loss = 0.005635
grad_step = 000059, loss = 0.005168
grad_step = 000060, loss = 0.004738
grad_step = 000061, loss = 0.004396
grad_step = 000062, loss = 0.004113
grad_step = 000063, loss = 0.003835
grad_step = 000064, loss = 0.003606
grad_step = 000065, loss = 0.003405
grad_step = 000066, loss = 0.003211
grad_step = 000067, loss = 0.003064
grad_step = 000068, loss = 0.002940
grad_step = 000069, loss = 0.002823
grad_step = 000070, loss = 0.002730
grad_step = 000071, loss = 0.002640
grad_step = 000072, loss = 0.002562
grad_step = 000073, loss = 0.002508
grad_step = 000074, loss = 0.002449
grad_step = 000075, loss = 0.002400
grad_step = 000076, loss = 0.002366
grad_step = 000077, loss = 0.002331
grad_step = 000078, loss = 0.002301
grad_step = 000079, loss = 0.002277
grad_step = 000080, loss = 0.002255
grad_step = 000081, loss = 0.002238
grad_step = 000082, loss = 0.002220
grad_step = 000083, loss = 0.002206
grad_step = 000084, loss = 0.002195
grad_step = 000085, loss = 0.002182
grad_step = 000086, loss = 0.002171
grad_step = 000087, loss = 0.002161
grad_step = 000088, loss = 0.002151
grad_step = 000089, loss = 0.002143
grad_step = 000090, loss = 0.002136
grad_step = 000091, loss = 0.002129
grad_step = 000092, loss = 0.002122
grad_step = 000093, loss = 0.002115
grad_step = 000094, loss = 0.002109
grad_step = 000095, loss = 0.002102
grad_step = 000096, loss = 0.002096
grad_step = 000097, loss = 0.002091
grad_step = 000098, loss = 0.002085
grad_step = 000099, loss = 0.002080
grad_step = 000100, loss = 0.002074
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002069
grad_step = 000102, loss = 0.002063
grad_step = 000103, loss = 0.002058
grad_step = 000104, loss = 0.002053
grad_step = 000105, loss = 0.002047
grad_step = 000106, loss = 0.002042
grad_step = 000107, loss = 0.002036
grad_step = 000108, loss = 0.002031
grad_step = 000109, loss = 0.002025
grad_step = 000110, loss = 0.002020
grad_step = 000111, loss = 0.002015
grad_step = 000112, loss = 0.002010
grad_step = 000113, loss = 0.002005
grad_step = 000114, loss = 0.002000
grad_step = 000115, loss = 0.001995
grad_step = 000116, loss = 0.001990
grad_step = 000117, loss = 0.001985
grad_step = 000118, loss = 0.001980
grad_step = 000119, loss = 0.001975
grad_step = 000120, loss = 0.001971
grad_step = 000121, loss = 0.001966
grad_step = 000122, loss = 0.001961
grad_step = 000123, loss = 0.001956
grad_step = 000124, loss = 0.001952
grad_step = 000125, loss = 0.001947
grad_step = 000126, loss = 0.001943
grad_step = 000127, loss = 0.001938
grad_step = 000128, loss = 0.001934
grad_step = 000129, loss = 0.001929
grad_step = 000130, loss = 0.001924
grad_step = 000131, loss = 0.001920
grad_step = 000132, loss = 0.001915
grad_step = 000133, loss = 0.001910
grad_step = 000134, loss = 0.001905
grad_step = 000135, loss = 0.001900
grad_step = 000136, loss = 0.001895
grad_step = 000137, loss = 0.001890
grad_step = 000138, loss = 0.001885
grad_step = 000139, loss = 0.001880
grad_step = 000140, loss = 0.001875
grad_step = 000141, loss = 0.001876
grad_step = 000142, loss = 0.001885
grad_step = 000143, loss = 0.001868
grad_step = 000144, loss = 0.001858
grad_step = 000145, loss = 0.001863
grad_step = 000146, loss = 0.001852
grad_step = 000147, loss = 0.001843
grad_step = 000148, loss = 0.001847
grad_step = 000149, loss = 0.001842
grad_step = 000150, loss = 0.001828
grad_step = 000151, loss = 0.001826
grad_step = 000152, loss = 0.001829
grad_step = 000153, loss = 0.001821
grad_step = 000154, loss = 0.001809
grad_step = 000155, loss = 0.001803
grad_step = 000156, loss = 0.001803
grad_step = 000157, loss = 0.001806
grad_step = 000158, loss = 0.001814
grad_step = 000159, loss = 0.001819
grad_step = 000160, loss = 0.001808
grad_step = 000161, loss = 0.001786
grad_step = 000162, loss = 0.001767
grad_step = 000163, loss = 0.001767
grad_step = 000164, loss = 0.001774
grad_step = 000165, loss = 0.001786
grad_step = 000166, loss = 0.001792
grad_step = 000167, loss = 0.001781
grad_step = 000168, loss = 0.001756
grad_step = 000169, loss = 0.001735
grad_step = 000170, loss = 0.001727
grad_step = 000171, loss = 0.001732
grad_step = 000172, loss = 0.001750
grad_step = 000173, loss = 0.001777
grad_step = 000174, loss = 0.001789
grad_step = 000175, loss = 0.001789
grad_step = 000176, loss = 0.001729
grad_step = 000177, loss = 0.001695
grad_step = 000178, loss = 0.001703
grad_step = 000179, loss = 0.001732
grad_step = 000180, loss = 0.001755
grad_step = 000181, loss = 0.001732
grad_step = 000182, loss = 0.001705
grad_step = 000183, loss = 0.001680
grad_step = 000184, loss = 0.001669
grad_step = 000185, loss = 0.001678
grad_step = 000186, loss = 0.001713
grad_step = 000187, loss = 0.001758
grad_step = 000188, loss = 0.001769
grad_step = 000189, loss = 0.001750
grad_step = 000190, loss = 0.001673
grad_step = 000191, loss = 0.001642
grad_step = 000192, loss = 0.001649
grad_step = 000193, loss = 0.001682
grad_step = 000194, loss = 0.001724
grad_step = 000195, loss = 0.001744
grad_step = 000196, loss = 0.001737
grad_step = 000197, loss = 0.001654
grad_step = 000198, loss = 0.001626
grad_step = 000199, loss = 0.001664
grad_step = 000200, loss = 0.001722
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001775
grad_step = 000202, loss = 0.001749
grad_step = 000203, loss = 0.001709
grad_step = 000204, loss = 0.001648
grad_step = 000205, loss = 0.001621
grad_step = 000206, loss = 0.001660
grad_step = 000207, loss = 0.001729
grad_step = 000208, loss = 0.001748
grad_step = 000209, loss = 0.001691
grad_step = 000210, loss = 0.001627
grad_step = 000211, loss = 0.001602
grad_step = 000212, loss = 0.001620
grad_step = 000213, loss = 0.001657
grad_step = 000214, loss = 0.001687
grad_step = 000215, loss = 0.001671
grad_step = 000216, loss = 0.001629
grad_step = 000217, loss = 0.001591
grad_step = 000218, loss = 0.001582
grad_step = 000219, loss = 0.001601
grad_step = 000220, loss = 0.001628
grad_step = 000221, loss = 0.001641
grad_step = 000222, loss = 0.001630
grad_step = 000223, loss = 0.001611
grad_step = 000224, loss = 0.001588
grad_step = 000225, loss = 0.001572
grad_step = 000226, loss = 0.001566
grad_step = 000227, loss = 0.001570
grad_step = 000228, loss = 0.001578
grad_step = 000229, loss = 0.001588
grad_step = 000230, loss = 0.001606
grad_step = 000231, loss = 0.001630
grad_step = 000232, loss = 0.001644
grad_step = 000233, loss = 0.001651
grad_step = 000234, loss = 0.001624
grad_step = 000235, loss = 0.001604
grad_step = 000236, loss = 0.001584
grad_step = 000237, loss = 0.001566
grad_step = 000238, loss = 0.001553
grad_step = 000239, loss = 0.001551
grad_step = 000240, loss = 0.001554
grad_step = 000241, loss = 0.001553
grad_step = 000242, loss = 0.001547
grad_step = 000243, loss = 0.001544
grad_step = 000244, loss = 0.001551
grad_step = 000245, loss = 0.001561
grad_step = 000246, loss = 0.001577
grad_step = 000247, loss = 0.001600
grad_step = 000248, loss = 0.001654
grad_step = 000249, loss = 0.001709
grad_step = 000250, loss = 0.001773
grad_step = 000251, loss = 0.001751
grad_step = 000252, loss = 0.001684
grad_step = 000253, loss = 0.001576
grad_step = 000254, loss = 0.001531
grad_step = 000255, loss = 0.001562
grad_step = 000256, loss = 0.001619
grad_step = 000257, loss = 0.001642
grad_step = 000258, loss = 0.001594
grad_step = 000259, loss = 0.001543
grad_step = 000260, loss = 0.001531
grad_step = 000261, loss = 0.001566
grad_step = 000262, loss = 0.001604
grad_step = 000263, loss = 0.001591
grad_step = 000264, loss = 0.001549
grad_step = 000265, loss = 0.001521
grad_step = 000266, loss = 0.001531
grad_step = 000267, loss = 0.001557
grad_step = 000268, loss = 0.001565
grad_step = 000269, loss = 0.001551
grad_step = 000270, loss = 0.001526
grad_step = 000271, loss = 0.001512
grad_step = 000272, loss = 0.001516
grad_step = 000273, loss = 0.001527
grad_step = 000274, loss = 0.001536
grad_step = 000275, loss = 0.001533
grad_step = 000276, loss = 0.001525
grad_step = 000277, loss = 0.001513
grad_step = 000278, loss = 0.001505
grad_step = 000279, loss = 0.001502
grad_step = 000280, loss = 0.001502
grad_step = 000281, loss = 0.001504
grad_step = 000282, loss = 0.001507
grad_step = 000283, loss = 0.001514
grad_step = 000284, loss = 0.001522
grad_step = 000285, loss = 0.001538
grad_step = 000286, loss = 0.001551
grad_step = 000287, loss = 0.001573
grad_step = 000288, loss = 0.001576
grad_step = 000289, loss = 0.001590
grad_step = 000290, loss = 0.001577
grad_step = 000291, loss = 0.001563
grad_step = 000292, loss = 0.001535
grad_step = 000293, loss = 0.001509
grad_step = 000294, loss = 0.001490
grad_step = 000295, loss = 0.001485
grad_step = 000296, loss = 0.001491
grad_step = 000297, loss = 0.001501
grad_step = 000298, loss = 0.001514
grad_step = 000299, loss = 0.001526
grad_step = 000300, loss = 0.001544
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001550
grad_step = 000302, loss = 0.001559
grad_step = 000303, loss = 0.001541
grad_step = 000304, loss = 0.001523
grad_step = 000305, loss = 0.001497
grad_step = 000306, loss = 0.001481
grad_step = 000307, loss = 0.001475
grad_step = 000308, loss = 0.001475
grad_step = 000309, loss = 0.001480
grad_step = 000310, loss = 0.001487
grad_step = 000311, loss = 0.001497
grad_step = 000312, loss = 0.001502
grad_step = 000313, loss = 0.001509
grad_step = 000314, loss = 0.001507
grad_step = 000315, loss = 0.001505
grad_step = 000316, loss = 0.001496
grad_step = 000317, loss = 0.001485
grad_step = 000318, loss = 0.001472
grad_step = 000319, loss = 0.001464
grad_step = 000320, loss = 0.001460
grad_step = 000321, loss = 0.001460
grad_step = 000322, loss = 0.001461
grad_step = 000323, loss = 0.001461
grad_step = 000324, loss = 0.001461
grad_step = 000325, loss = 0.001461
grad_step = 000326, loss = 0.001464
grad_step = 000327, loss = 0.001471
grad_step = 000328, loss = 0.001494
grad_step = 000329, loss = 0.001527
grad_step = 000330, loss = 0.001597
grad_step = 000331, loss = 0.001649
grad_step = 000332, loss = 0.001708
grad_step = 000333, loss = 0.001631
grad_step = 000334, loss = 0.001560
grad_step = 000335, loss = 0.001535
grad_step = 000336, loss = 0.001538
grad_step = 000337, loss = 0.001523
grad_step = 000338, loss = 0.001552
grad_step = 000339, loss = 0.001608
grad_step = 000340, loss = 0.001558
grad_step = 000341, loss = 0.001470
grad_step = 000342, loss = 0.001455
grad_step = 000343, loss = 0.001488
grad_step = 000344, loss = 0.001504
grad_step = 000345, loss = 0.001513
grad_step = 000346, loss = 0.001542
grad_step = 000347, loss = 0.001521
grad_step = 000348, loss = 0.001466
grad_step = 000349, loss = 0.001442
grad_step = 000350, loss = 0.001463
grad_step = 000351, loss = 0.001483
grad_step = 000352, loss = 0.001483
grad_step = 000353, loss = 0.001492
grad_step = 000354, loss = 0.001496
grad_step = 000355, loss = 0.001483
grad_step = 000356, loss = 0.001452
grad_step = 000357, loss = 0.001428
grad_step = 000358, loss = 0.001433
grad_step = 000359, loss = 0.001453
grad_step = 000360, loss = 0.001461
grad_step = 000361, loss = 0.001460
grad_step = 000362, loss = 0.001463
grad_step = 000363, loss = 0.001463
grad_step = 000364, loss = 0.001464
grad_step = 000365, loss = 0.001449
grad_step = 000366, loss = 0.001436
grad_step = 000367, loss = 0.001427
grad_step = 000368, loss = 0.001425
grad_step = 000369, loss = 0.001421
grad_step = 000370, loss = 0.001414
grad_step = 000371, loss = 0.001413
grad_step = 000372, loss = 0.001417
grad_step = 000373, loss = 0.001421
grad_step = 000374, loss = 0.001423
grad_step = 000375, loss = 0.001428
grad_step = 000376, loss = 0.001441
grad_step = 000377, loss = 0.001472
grad_step = 000378, loss = 0.001506
grad_step = 000379, loss = 0.001569
grad_step = 000380, loss = 0.001608
grad_step = 000381, loss = 0.001651
grad_step = 000382, loss = 0.001635
grad_step = 000383, loss = 0.001564
grad_step = 000384, loss = 0.001468
grad_step = 000385, loss = 0.001411
grad_step = 000386, loss = 0.001429
grad_step = 000387, loss = 0.001492
grad_step = 000388, loss = 0.001547
grad_step = 000389, loss = 0.001542
grad_step = 000390, loss = 0.001487
grad_step = 000391, loss = 0.001418
grad_step = 000392, loss = 0.001400
grad_step = 000393, loss = 0.001434
grad_step = 000394, loss = 0.001481
grad_step = 000395, loss = 0.001512
grad_step = 000396, loss = 0.001500
grad_step = 000397, loss = 0.001458
grad_step = 000398, loss = 0.001404
grad_step = 000399, loss = 0.001398
grad_step = 000400, loss = 0.001438
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001476
grad_step = 000402, loss = 0.001490
grad_step = 000403, loss = 0.001452
grad_step = 000404, loss = 0.001406
grad_step = 000405, loss = 0.001388
grad_step = 000406, loss = 0.001414
grad_step = 000407, loss = 0.001449
grad_step = 000408, loss = 0.001449
grad_step = 000409, loss = 0.001428
grad_step = 000410, loss = 0.001403
grad_step = 000411, loss = 0.001393
grad_step = 000412, loss = 0.001394
grad_step = 000413, loss = 0.001398
grad_step = 000414, loss = 0.001410
grad_step = 000415, loss = 0.001415
grad_step = 000416, loss = 0.001406
grad_step = 000417, loss = 0.001388
grad_step = 000418, loss = 0.001374
grad_step = 000419, loss = 0.001372
grad_step = 000420, loss = 0.001378
grad_step = 000421, loss = 0.001383
grad_step = 000422, loss = 0.001383
grad_step = 000423, loss = 0.001383
grad_step = 000424, loss = 0.001389
grad_step = 000425, loss = 0.001399
grad_step = 000426, loss = 0.001402
grad_step = 000427, loss = 0.001406
grad_step = 000428, loss = 0.001398
grad_step = 000429, loss = 0.001395
grad_step = 000430, loss = 0.001391
grad_step = 000431, loss = 0.001386
grad_step = 000432, loss = 0.001376
grad_step = 000433, loss = 0.001365
grad_step = 000434, loss = 0.001358
grad_step = 000435, loss = 0.001356
grad_step = 000436, loss = 0.001357
grad_step = 000437, loss = 0.001357
grad_step = 000438, loss = 0.001356
grad_step = 000439, loss = 0.001355
grad_step = 000440, loss = 0.001359
grad_step = 000441, loss = 0.001368
grad_step = 000442, loss = 0.001393
grad_step = 000443, loss = 0.001436
grad_step = 000444, loss = 0.001540
grad_step = 000445, loss = 0.001693
grad_step = 000446, loss = 0.001896
grad_step = 000447, loss = 0.001897
grad_step = 000448, loss = 0.001742
grad_step = 000449, loss = 0.001574
grad_step = 000450, loss = 0.001475
grad_step = 000451, loss = 0.001453
grad_step = 000452, loss = 0.001550
grad_step = 000453, loss = 0.001644
grad_step = 000454, loss = 0.001510
grad_step = 000455, loss = 0.001359
grad_step = 000456, loss = 0.001402
grad_step = 000457, loss = 0.001477
grad_step = 000458, loss = 0.001447
grad_step = 000459, loss = 0.001421
grad_step = 000460, loss = 0.001495
grad_step = 000461, loss = 0.001488
grad_step = 000462, loss = 0.001389
grad_step = 000463, loss = 0.001367
grad_step = 000464, loss = 0.001398
grad_step = 000465, loss = 0.001373
grad_step = 000466, loss = 0.001361
grad_step = 000467, loss = 0.001390
grad_step = 000468, loss = 0.001397
grad_step = 000469, loss = 0.001368
grad_step = 000470, loss = 0.001345
grad_step = 000471, loss = 0.001347
grad_step = 000472, loss = 0.001345
grad_step = 000473, loss = 0.001332
grad_step = 000474, loss = 0.001339
grad_step = 000475, loss = 0.001352
grad_step = 000476, loss = 0.001344
grad_step = 000477, loss = 0.001325
grad_step = 000478, loss = 0.001327
grad_step = 000479, loss = 0.001331
grad_step = 000480, loss = 0.001319
grad_step = 000481, loss = 0.001314
grad_step = 000482, loss = 0.001325
grad_step = 000483, loss = 0.001328
grad_step = 000484, loss = 0.001322
grad_step = 000485, loss = 0.001323
grad_step = 000486, loss = 0.001329
grad_step = 000487, loss = 0.001332
grad_step = 000488, loss = 0.001325
grad_step = 000489, loss = 0.001325
grad_step = 000490, loss = 0.001324
grad_step = 000491, loss = 0.001320
grad_step = 000492, loss = 0.001313
grad_step = 000493, loss = 0.001311
grad_step = 000494, loss = 0.001309
grad_step = 000495, loss = 0.001308
grad_step = 000496, loss = 0.001302
grad_step = 000497, loss = 0.001298
grad_step = 000498, loss = 0.001296
grad_step = 000499, loss = 0.001295
grad_step = 000500, loss = 0.001291
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001289
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

  date_run                              2020-05-11 14:12:43.549858
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.239447
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 14:12:43.555447
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.131348
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 14:12:43.563284
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.158039
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 14:12:43.568574
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.995876
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
0   2020-05-11 14:12:14.599109  ...    mean_absolute_error
1   2020-05-11 14:12:14.602929  ...     mean_squared_error
2   2020-05-11 14:12:14.606257  ...  median_absolute_error
3   2020-05-11 14:12:14.609474  ...               r2_score
4   2020-05-11 14:12:24.118723  ...    mean_absolute_error
5   2020-05-11 14:12:24.122620  ...     mean_squared_error
6   2020-05-11 14:12:24.126858  ...  median_absolute_error
7   2020-05-11 14:12:24.130580  ...               r2_score
8   2020-05-11 14:12:43.549858  ...    mean_absolute_error
9   2020-05-11 14:12:43.555447  ...     mean_squared_error
10  2020-05-11 14:12:43.563284  ...  median_absolute_error
11  2020-05-11 14:12:43.568574  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132526.72it/s] 87%|████████▋ | 8617984/9912422 [00:00<00:06, 189198.94it/s]9920512it [00:00, 42219246.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 648878.49it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 926103.94it/s]1654784it [00:00, 9382398.95it/s]                           
0it [00:00, ?it/s]8192it [00:00, 271923.73it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b41efd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f43e8b3af28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b3aaef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f43e8612048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b41efd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f43fdda4e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b41efd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f43fd361e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b3aaef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f43e8b3af28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f444b41efd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc181e981d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=116f6a6d73e75a43a95bb13c88b95f83cc09cab8b3dc6edb1f2b1ef44514cf18
  Stored in directory: /tmp/pip-ephem-wheel-cache-mb94sx75/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc178006048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 2670592/17464789 [===>..........................] - ETA: 0s
 6537216/17464789 [==========>...................] - ETA: 0s
10117120/17464789 [================>.............] - ETA: 0s
15736832/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 14:14:14.314495: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 14:14:14.318408: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 14:14:14.318537: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564c55220330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 14:14:14.318549: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6022 - accuracy: 0.5042
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 3s - loss: 7.6736 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6807 - accuracy: 0.4991
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6492 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6423 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6643 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 7s 289us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 14:14:28.375425
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 14:14:28.375425  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 14:14:34.679220: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 14:14:34.684526: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 14:14:34.684659: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55edf6106ee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 14:14:34.684673: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f15f7e78c18> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1131 - crf_viterbi_accuracy: 0.6800 - val_loss: 0.9936 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f15d3bb8898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6590 - accuracy: 0.5005 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6390 - accuracy: 0.5018
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6579 - accuracy: 0.5006
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6452 - accuracy: 0.5014
11000/25000 [============>.................] - ETA: 3s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7008 - accuracy: 0.4978
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6841 - accuracy: 0.4989
15000/25000 [=================>............] - ETA: 2s - loss: 7.7014 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6459 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f158eb4b400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<35:04:09, 6.83kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<24:46:55, 9.66kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<17:23:59, 13.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<12:11:04, 19.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<8:30:20, 28.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.59M/862M [00:01<5:55:15, 40.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<4:07:31, 57.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.6M/862M [00:02<2:52:38, 81.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.6M/862M [00:02<2:00:12, 117kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.0M/862M [00:02<1:23:56, 166kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:02<58:31, 237kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:02<40:54, 338kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<28:33, 481kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<20:00, 683kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<13:59, 971kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:02<09:52, 1.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:03<07:49, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<07:22, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<08:49, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:57, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:15, 2.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<06:22, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<05:51, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<04:23, 3.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:09<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<05:41, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:09<04:15, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:11<06:08, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<05:38, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<04:13, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:13<06:05, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<05:36, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<04:12, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:15<06:03, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<04:11, 3.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<06:00, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:17<05:20, 2.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<04:17, 3.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<03:37, 3.57MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<9:35:26, 22.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<6:43:00, 32.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<4:41:36, 45.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<3:21:23, 64.0kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<2:23:42, 89.7kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:21<1:41:11, 127kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:21<1:10:43, 181kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<1:30:05, 142kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<1:04:22, 199kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:23<45:15, 283kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<34:33, 369kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<26:47, 476kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<19:16, 661kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:25<13:40, 929kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<14:21, 883kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:22, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:13, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:42, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:11, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:20, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:43, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:59, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:30, 2.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:49, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:08, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:49, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:21, 2.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:13, 3.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:44, 784kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:35, 907kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:08, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:03, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:24, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:06, 2.97MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<52:04, 234kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<38:56, 313kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<27:51, 437kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<19:32, 621kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<50:50, 239kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<36:50, 329kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<26:03, 464kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<20:58, 575kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<17:18, 697kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<12:40, 950kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<09:00, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<13:08, 913kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:27, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:37, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:04, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:12, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:16, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:36, 2.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<06:49, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:01, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:31, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:21, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:02, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:32, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:23, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:59, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:38, 3.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:00, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<06:37, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:53, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:34, 3.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<24:22, 475kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<18:15, 634kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<13:00, 888kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<11:45, 979kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:26, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:53, 1.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:27, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:40, 1.49MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:15, 2.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<58:18, 195kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<41:59, 271kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<29:34, 384kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<23:15, 486kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<18:41, 605kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<13:34, 832kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:41, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<07:39, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<8:12:06, 22.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<5:44:42, 32.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<4:00:22, 46.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<3:03:34, 60.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<2:10:51, 85.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<1:32:04, 121kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<1:04:22, 173kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<51:17, 217kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<37:02, 300kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<26:07, 424kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<20:46, 531kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<16:54, 652kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:24, 888kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<08:47, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:24:16, 130kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:00:07, 182kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<42:15, 259kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<32:00, 341kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<23:33, 463kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<16:43, 650kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<14:10, 764kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<11:03, 980kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:00, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:04, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:47, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:01, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:59, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:30, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:00, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:41, 2.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:11, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:26, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:04, 2.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:17, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:48, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:35, 2.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:56, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:32, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:27, 3.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:27, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:21, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:32, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:24, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:12, 3.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<1:15:25, 136kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<53:40, 192kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<37:40, 272kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<26:26, 386kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<44:11, 231kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<33:01, 309kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<23:32, 433kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<16:39, 611kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<14:24, 704kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<10:57, 925kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:00, 1.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<05:41, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<22:25, 449kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<16:45, 601kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<11:57, 840kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<10:39, 939kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<08:30, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:12, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<06:37, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:03, 2.44MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:58, 3.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<26:55, 367kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<20:58, 470kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<15:06, 653kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<10:41, 919kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<11:42, 837kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<09:13, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:40, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<06:55, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<05:51, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:18, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:16, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<04:41, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:31, 2.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:01, 3.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<7:14:01, 22.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<5:03:55, 31.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<3:31:44, 45.0kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<2:42:41, 58.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<1:55:53, 82.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<1:21:28, 117kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<56:54, 167kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<45:49, 207kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<32:52, 288kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<23:12, 407kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<16:18, 577kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<35:02, 268kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<25:30, 368kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<18:03, 519kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<14:45, 632kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<11:17, 825kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:07, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:50, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:27, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:40, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:04, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:14, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:01, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:26, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:13, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:36, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:25, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:00, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:56, 2.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:52, 3.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:51, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<04:50, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<11:21, 785kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:51, 1.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:25, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:33, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:53, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:20, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:13, 2.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:15, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:52, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<02:55, 2.96MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:03, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:44, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:42, 3.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<1:03:20, 135kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<45:12, 189kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<31:46, 269kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<24:07, 353kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<17:45, 478kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<12:37, 672kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<10:45, 785kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<08:24, 1.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:03, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<06:10, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<06:06, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:39, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:21, 2.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<06:00, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:04, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:45, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:31, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:01, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:01, 2.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<04:00, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:39, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<02:45, 2.94MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<03:48, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:23, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:25, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:31, 3.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:39, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:05, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:02, 2.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:58, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:29, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:29, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<02:33, 3.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:35, 3.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<5:55:33, 22.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<4:08:41, 31.7kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<2:55:17, 44.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<2:03:15, 63.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<1:27:04, 89.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<1:00:58, 128kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<44:11, 175kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<32:35, 238kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<23:08, 334kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<16:13, 474kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<15:06, 509kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<11:22, 675kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<08:07, 941kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:25, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:49, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:41, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<56:21, 134kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<40:12, 188kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<28:12, 266kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<21:23, 349kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<15:44, 474kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<11:09, 667kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:29, 780kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:13, 900kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:05, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<04:20, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:43, 950kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:02, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:30, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<03:12, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<20:32, 354kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<14:59, 485kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<10:37, 682kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<07:29, 961kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<32:28, 222kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<24:11, 298kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<17:13, 417kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<12:07, 590kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<11:08, 640kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:32, 834kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:07, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:54, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:52, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:34, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:07, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:20, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:27, 2.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<51:03, 136kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<36:26, 190kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<25:35, 270kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<19:24, 353kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<15:00, 457kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<10:50, 631kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<07:37, 891kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<53:40, 127kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<38:15, 177kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<26:51, 252kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<20:15, 332kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<14:51, 452kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<10:30, 637kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<08:52, 749kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<06:54, 963kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:59, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:01, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<04:05, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:07, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:14, 2.90MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<13:29, 483kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<10:07, 643kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:12, 900kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<06:30, 990kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:13, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:48, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:08, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:34, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:39, 2.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:19, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:37, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:49, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:02, 3.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:01, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<04:03, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:10, 2.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<13:32, 456kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<10:47, 572kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<07:49, 787kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:35, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<04:19, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<4:30:32, 22.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<3:09:15, 32.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<2:11:50, 46.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<1:34:19, 64.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<1:07:16, 89.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<47:16, 127kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<33:01, 181kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<25:06, 238kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<18:10, 328kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<12:49, 463kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<10:17, 573kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:27, 697kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<06:12, 947kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:23, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<44:40, 131kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<31:50, 183kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<22:20, 260kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<16:52, 341kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<13:02, 442kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<09:22, 613kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:36, 864kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<07:01, 811kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:24, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:54, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:47, 2.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<12:35, 447kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<10:00, 562kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<07:14, 774kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<05:06, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:27, 745kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:42, 972kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:07, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<02:56, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<16:22, 335kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<11:38, 470kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<08:15, 660kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<07:04, 766kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:30, 982kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:58, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:00, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:00, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:10, 2.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:35, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:03, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:16, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:47, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:06, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:23, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:43, 2.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:36, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:40, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:39, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:55, 2.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<14:06, 360kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<10:23, 488kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<07:21, 685kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:16, 797kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:54, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:33, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:37, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:03, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:14, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:41, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:22, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:46, 2.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:21, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:08, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:36, 2.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:11, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:29, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<01:58, 2.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:25, 3.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<07:48, 597kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<05:52, 794kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:17, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<03:01, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<13:41, 336kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<10:02, 457kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:06, 642kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<06:00, 755kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:07, 882kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:47, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:44, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:03, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:35, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:55, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:21, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:01, 2.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:32, 2.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:07, 3.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<08:56, 483kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<06:37, 651kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:43, 909kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<03:20, 1.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<04:59, 851kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<3:22:23, 21.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<2:21:27, 30.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<1:37:52, 42.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<1:18:02, 53.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<55:29, 75.4kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<38:56, 107kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<26:58, 153kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<48:14, 85.3kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<34:09, 120kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<23:51, 171kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<17:29, 231kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<12:39, 319kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<08:54, 451kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<07:06, 560kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:23, 737kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:50, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:35, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:18, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:46, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:46, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:20, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:02, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:50, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:22, 2.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:49, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:11, 3.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<00:52, 4.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:09, 358kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<07:50, 463kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:37, 643kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<03:57, 906kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:18, 828kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:23, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:26, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:29, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:28, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:52, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:21, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:09, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:52, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:22, 2.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:44, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:33, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:10, 2.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:26, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:05, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:30, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:23, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:02, 3.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:27, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:21, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:01, 3.06MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:25, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:38, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:16, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<00:55, 3.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:05, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:46, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:18, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:35, 1.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<01:22, 2.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:01, 2.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:44, 3.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<05:47, 498kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<04:20, 662kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:05, 922kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:47, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<02:11, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:36, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:08, 2.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<07:26, 369kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<05:28, 500kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:51, 703kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:17, 812kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:34, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:51, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:53, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:35, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:09, 2.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:57, 2.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<1:57:26, 21.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<1:21:55, 30.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<56:13, 44.1kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<44:03, 56.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<31:20, 78.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<21:56, 112kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<15:10, 160kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<11:39, 206kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<08:23, 286kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<05:52, 404kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<04:35, 509kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<03:41, 633kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:40, 868kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:54, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:55, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:35, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:09, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:17, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:08, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:50, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:04, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:43, 2.91MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:53, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:40, 3.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:51, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:38, 3.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:49, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:37, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:51, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:47, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:35, 3.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:49, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:56, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:44, 2.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:32, 3.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:52, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:48, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:35, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:47, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:50, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:40, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:29, 3.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:55, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:47, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:34, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:24, 3.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<04:15, 356kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<03:07, 482kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<02:11, 678kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:49, 790kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:34, 913kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<10:20, 133kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<07:21, 186kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<05:05, 264kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<03:45, 347kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:45, 471kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:55, 661kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:35, 774kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:14, 992kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:52, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:52, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:10, 989kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:52, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:36, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:59, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:48, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:38, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:39, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:21, 2.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:33, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:21, 2.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:26, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:31, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:27, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:19, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:23, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:21, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:15, 2.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<31:44, 21.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<21:50, 30.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<14:09, 44.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<10:47, 57.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<07:38, 80.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<05:16, 115kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<03:24, 163kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<03:17, 168kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<02:19, 234kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:34, 331kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:07, 427kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:49, 572kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:33, 800kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:27, 904kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:21, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:15, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:45, 454kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:33, 607kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:22, 848kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:17, 946kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 4.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 471kB/s] .vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 847/400000 [00:00<00:47, 8463.67it/s]  0%|          | 1668/400000 [00:00<00:47, 8383.55it/s]  1%|          | 2500/400000 [00:00<00:47, 8364.03it/s]  1%|          | 3287/400000 [00:00<00:48, 8207.61it/s]  1%|          | 4086/400000 [00:00<00:48, 8139.74it/s]  1%|          | 4880/400000 [00:00<00:48, 8077.64it/s]  1%|▏         | 5704/400000 [00:00<00:48, 8124.41it/s]  2%|▏         | 6542/400000 [00:00<00:47, 8198.78it/s]  2%|▏         | 7364/400000 [00:00<00:47, 8203.52it/s]  2%|▏         | 8210/400000 [00:01<00:47, 8278.80it/s]  2%|▏         | 9049/400000 [00:01<00:47, 8310.41it/s]  2%|▏         | 9869/400000 [00:01<00:47, 8275.63it/s]  3%|▎         | 10686/400000 [00:01<00:47, 8208.61it/s]  3%|▎         | 11499/400000 [00:01<00:48, 8080.97it/s]  3%|▎         | 12303/400000 [00:01<00:48, 8042.89it/s]  3%|▎         | 13113/400000 [00:01<00:48, 8058.19it/s]  3%|▎         | 13964/400000 [00:01<00:47, 8187.07it/s]  4%|▎         | 14800/400000 [00:01<00:46, 8238.18it/s]  4%|▍         | 15624/400000 [00:01<00:46, 8188.55it/s]  4%|▍         | 16454/400000 [00:02<00:46, 8220.60it/s]  4%|▍         | 17294/400000 [00:02<00:46, 8269.03it/s]  5%|▍         | 18141/400000 [00:02<00:45, 8327.54it/s]  5%|▍         | 19005/400000 [00:02<00:45, 8417.23it/s]  5%|▍         | 19848/400000 [00:02<00:45, 8420.69it/s]  5%|▌         | 20691/400000 [00:02<00:46, 8162.47it/s]  5%|▌         | 21513/400000 [00:02<00:46, 8179.55it/s]  6%|▌         | 22333/400000 [00:02<00:46, 8181.19it/s]  6%|▌         | 23155/400000 [00:02<00:46, 8191.18it/s]  6%|▌         | 23975/400000 [00:02<00:46, 8141.50it/s]  6%|▌         | 24790/400000 [00:03<00:46, 8049.12it/s]  6%|▋         | 25650/400000 [00:03<00:45, 8204.55it/s]  7%|▋         | 26515/400000 [00:03<00:44, 8332.15it/s]  7%|▋         | 27385/400000 [00:03<00:44, 8436.91it/s]  7%|▋         | 28230/400000 [00:03<00:44, 8398.54it/s]  7%|▋         | 29071/400000 [00:03<00:44, 8347.40it/s]  7%|▋         | 29907/400000 [00:03<00:44, 8272.32it/s]  8%|▊         | 30765/400000 [00:03<00:44, 8361.07it/s]  8%|▊         | 31622/400000 [00:03<00:43, 8421.87it/s]  8%|▊         | 32465/400000 [00:03<00:44, 8340.03it/s]  8%|▊         | 33300/400000 [00:04<00:44, 8264.19it/s]  9%|▊         | 34127/400000 [00:04<00:44, 8248.47it/s]  9%|▊         | 34953/400000 [00:04<00:44, 8213.06it/s]  9%|▉         | 35801/400000 [00:04<00:43, 8290.84it/s]  9%|▉         | 36639/400000 [00:04<00:43, 8314.93it/s]  9%|▉         | 37471/400000 [00:04<00:43, 8242.29it/s] 10%|▉         | 38296/400000 [00:04<00:45, 8014.90it/s] 10%|▉         | 39119/400000 [00:04<00:44, 8075.85it/s] 10%|▉         | 39928/400000 [00:04<00:44, 8074.62it/s] 10%|█         | 40755/400000 [00:04<00:44, 8130.58it/s] 10%|█         | 41569/400000 [00:05<00:44, 8017.29it/s] 11%|█         | 42424/400000 [00:05<00:43, 8168.57it/s] 11%|█         | 43258/400000 [00:05<00:43, 8215.42it/s] 11%|█         | 44086/400000 [00:05<00:43, 8233.75it/s] 11%|█         | 44911/400000 [00:05<00:44, 7998.86it/s] 11%|█▏        | 45713/400000 [00:05<00:44, 7967.24it/s] 12%|█▏        | 46516/400000 [00:05<00:44, 7983.68it/s] 12%|█▏        | 47329/400000 [00:05<00:43, 8026.46it/s] 12%|█▏        | 48150/400000 [00:05<00:43, 8078.67it/s] 12%|█▏        | 48996/400000 [00:05<00:42, 8188.04it/s] 12%|█▏        | 49816/400000 [00:06<00:42, 8144.00it/s] 13%|█▎        | 50631/400000 [00:06<00:42, 8137.04it/s] 13%|█▎        | 51450/400000 [00:06<00:42, 8152.84it/s] 13%|█▎        | 52266/400000 [00:06<00:42, 8098.79it/s] 13%|█▎        | 53077/400000 [00:06<00:42, 8096.53it/s] 13%|█▎        | 53887/400000 [00:06<00:42, 8084.56it/s] 14%|█▎        | 54750/400000 [00:06<00:41, 8240.16it/s] 14%|█▍        | 55609/400000 [00:06<00:41, 8340.80it/s] 14%|█▍        | 56444/400000 [00:06<00:41, 8330.05it/s] 14%|█▍        | 57278/400000 [00:06<00:42, 8149.96it/s] 15%|█▍        | 58121/400000 [00:07<00:41, 8231.13it/s] 15%|█▍        | 58985/400000 [00:07<00:40, 8347.31it/s] 15%|█▍        | 59847/400000 [00:07<00:40, 8424.63it/s] 15%|█▌        | 60704/400000 [00:07<00:40, 8465.90it/s] 15%|█▌        | 61563/400000 [00:07<00:39, 8500.30it/s] 16%|█▌        | 62414/400000 [00:07<00:40, 8390.89it/s] 16%|█▌        | 63254/400000 [00:07<00:41, 8066.40it/s] 16%|█▌        | 64100/400000 [00:07<00:41, 8180.02it/s] 16%|█▌        | 64937/400000 [00:07<00:40, 8233.68it/s] 16%|█▋        | 65798/400000 [00:08<00:40, 8341.44it/s] 17%|█▋        | 66634/400000 [00:08<00:40, 8330.71it/s] 17%|█▋        | 67493/400000 [00:08<00:39, 8403.84it/s] 17%|█▋        | 68335/400000 [00:08<00:39, 8374.72it/s] 17%|█▋        | 69174/400000 [00:08<00:39, 8373.94it/s] 18%|█▊        | 70023/400000 [00:08<00:39, 8406.17it/s] 18%|█▊        | 70864/400000 [00:08<00:39, 8322.92it/s] 18%|█▊        | 71709/400000 [00:08<00:39, 8358.90it/s] 18%|█▊        | 72568/400000 [00:08<00:38, 8425.84it/s] 18%|█▊        | 73434/400000 [00:08<00:38, 8493.92it/s] 19%|█▊        | 74284/400000 [00:09<00:38, 8485.44it/s] 19%|█▉        | 75133/400000 [00:09<00:38, 8408.83it/s] 19%|█▉        | 75975/400000 [00:09<00:38, 8405.82it/s] 19%|█▉        | 76831/400000 [00:09<00:38, 8450.63it/s] 19%|█▉        | 77702/400000 [00:09<00:37, 8526.33it/s] 20%|█▉        | 78555/400000 [00:09<00:38, 8426.47it/s] 20%|█▉        | 79399/400000 [00:09<00:38, 8429.69it/s] 20%|██        | 80247/400000 [00:09<00:37, 8441.88it/s] 20%|██        | 81092/400000 [00:09<00:38, 8289.47it/s] 20%|██        | 81957/400000 [00:09<00:37, 8394.34it/s] 21%|██        | 82823/400000 [00:10<00:37, 8472.22it/s] 21%|██        | 83672/400000 [00:10<00:37, 8429.45it/s] 21%|██        | 84539/400000 [00:10<00:37, 8499.86it/s] 21%|██▏       | 85406/400000 [00:10<00:36, 8550.06it/s] 22%|██▏       | 86262/400000 [00:10<00:36, 8523.00it/s] 22%|██▏       | 87115/400000 [00:10<00:37, 8355.20it/s] 22%|██▏       | 87952/400000 [00:10<00:37, 8310.12it/s] 22%|██▏       | 88805/400000 [00:10<00:37, 8374.37it/s] 22%|██▏       | 89677/400000 [00:10<00:36, 8474.82it/s] 23%|██▎       | 90526/400000 [00:10<00:36, 8407.40it/s] 23%|██▎       | 91389/400000 [00:11<00:36, 8472.18it/s] 23%|██▎       | 92254/400000 [00:11<00:36, 8522.87it/s] 23%|██▎       | 93129/400000 [00:11<00:35, 8588.72it/s] 23%|██▎       | 93989/400000 [00:11<00:36, 8458.41it/s] 24%|██▎       | 94837/400000 [00:11<00:36, 8463.71it/s] 24%|██▍       | 95684/400000 [00:11<00:35, 8454.55it/s] 24%|██▍       | 96530/400000 [00:11<00:36, 8378.62it/s] 24%|██▍       | 97369/400000 [00:11<00:36, 8336.21it/s] 25%|██▍       | 98203/400000 [00:11<00:36, 8320.77it/s] 25%|██▍       | 99050/400000 [00:11<00:35, 8364.59it/s] 25%|██▍       | 99889/400000 [00:12<00:35, 8371.50it/s] 25%|██▌       | 100727/400000 [00:12<00:35, 8349.64it/s] 25%|██▌       | 101588/400000 [00:12<00:35, 8424.65it/s] 26%|██▌       | 102431/400000 [00:12<00:35, 8414.40it/s] 26%|██▌       | 103280/400000 [00:12<00:35, 8434.26it/s] 26%|██▌       | 104129/400000 [00:12<00:35, 8449.73it/s] 26%|██▌       | 104975/400000 [00:12<00:35, 8339.70it/s] 26%|██▋       | 105815/400000 [00:12<00:35, 8355.20it/s] 27%|██▋       | 106651/400000 [00:12<00:35, 8349.73it/s] 27%|██▋       | 107491/400000 [00:12<00:34, 8362.92it/s] 27%|██▋       | 108345/400000 [00:13<00:34, 8413.00it/s] 27%|██▋       | 109187/400000 [00:13<00:34, 8342.73it/s] 28%|██▊       | 110022/400000 [00:13<00:34, 8338.39it/s] 28%|██▊       | 110861/400000 [00:13<00:34, 8352.82it/s] 28%|██▊       | 111697/400000 [00:13<00:34, 8324.39it/s] 28%|██▊       | 112530/400000 [00:13<00:34, 8298.42it/s] 28%|██▊       | 113363/400000 [00:13<00:34, 8305.41it/s] 29%|██▊       | 114216/400000 [00:13<00:34, 8368.79it/s] 29%|██▉       | 115073/400000 [00:13<00:33, 8427.20it/s] 29%|██▉       | 115932/400000 [00:13<00:33, 8473.04it/s] 29%|██▉       | 116792/400000 [00:14<00:33, 8507.95it/s] 29%|██▉       | 117653/400000 [00:14<00:33, 8537.12it/s] 30%|██▉       | 118507/400000 [00:14<00:33, 8517.53it/s] 30%|██▉       | 119362/400000 [00:14<00:32, 8526.12it/s] 30%|███       | 120225/400000 [00:14<00:32, 8556.91it/s] 30%|███       | 121081/400000 [00:14<00:32, 8464.58it/s] 30%|███       | 121928/400000 [00:14<00:33, 8419.01it/s] 31%|███       | 122771/400000 [00:14<00:33, 8358.12it/s] 31%|███       | 123608/400000 [00:14<00:33, 8330.66it/s] 31%|███       | 124442/400000 [00:14<00:33, 8316.98it/s] 31%|███▏      | 125303/400000 [00:15<00:32, 8402.23it/s] 32%|███▏      | 126154/400000 [00:15<00:32, 8432.48it/s] 32%|███▏      | 126998/400000 [00:15<00:32, 8325.81it/s] 32%|███▏      | 127858/400000 [00:15<00:32, 8403.70it/s] 32%|███▏      | 128731/400000 [00:15<00:31, 8498.64it/s] 32%|███▏      | 129593/400000 [00:15<00:31, 8534.57it/s] 33%|███▎      | 130447/400000 [00:15<00:31, 8523.60it/s] 33%|███▎      | 131310/400000 [00:15<00:31, 8553.06it/s] 33%|███▎      | 132166/400000 [00:15<00:31, 8519.02it/s] 33%|███▎      | 133019/400000 [00:15<00:31, 8510.47it/s] 33%|███▎      | 133887/400000 [00:16<00:31, 8559.62it/s] 34%|███▎      | 134744/400000 [00:16<00:31, 8544.11it/s] 34%|███▍      | 135599/400000 [00:16<00:31, 8471.72it/s] 34%|███▍      | 136447/400000 [00:16<00:31, 8408.74it/s] 34%|███▍      | 137289/400000 [00:16<00:31, 8381.04it/s] 35%|███▍      | 138128/400000 [00:16<00:31, 8352.86it/s] 35%|███▍      | 138970/400000 [00:16<00:31, 8372.70it/s] 35%|███▍      | 139808/400000 [00:16<00:31, 8334.08it/s] 35%|███▌      | 140653/400000 [00:16<00:31, 8365.92it/s] 35%|███▌      | 141490/400000 [00:16<00:30, 8349.75it/s] 36%|███▌      | 142326/400000 [00:17<00:31, 8231.73it/s] 36%|███▌      | 143176/400000 [00:17<00:30, 8309.89it/s] 36%|███▌      | 144008/400000 [00:17<00:31, 8229.58it/s] 36%|███▌      | 144832/400000 [00:17<00:31, 8204.41it/s] 36%|███▋      | 145678/400000 [00:17<00:30, 8277.15it/s] 37%|███▋      | 146539/400000 [00:17<00:30, 8374.02it/s] 37%|███▋      | 147395/400000 [00:17<00:29, 8427.71it/s] 37%|███▋      | 148239/400000 [00:17<00:30, 8337.48it/s] 37%|███▋      | 149083/400000 [00:17<00:29, 8367.79it/s] 37%|███▋      | 149921/400000 [00:18<00:29, 8355.71it/s] 38%|███▊      | 150757/400000 [00:18<00:29, 8349.75it/s] 38%|███▊      | 151616/400000 [00:18<00:29, 8418.37it/s] 38%|███▊      | 152459/400000 [00:18<00:29, 8413.02it/s] 38%|███▊      | 153301/400000 [00:18<00:29, 8255.10it/s] 39%|███▊      | 154144/400000 [00:18<00:29, 8304.69it/s] 39%|███▊      | 154997/400000 [00:18<00:29, 8370.22it/s] 39%|███▉      | 155854/400000 [00:18<00:28, 8428.64it/s] 39%|███▉      | 156698/400000 [00:18<00:29, 8361.34it/s] 39%|███▉      | 157548/400000 [00:18<00:28, 8400.35it/s] 40%|███▉      | 158402/400000 [00:19<00:28, 8440.39it/s] 40%|███▉      | 159258/400000 [00:19<00:28, 8475.74it/s] 40%|████      | 160121/400000 [00:19<00:28, 8520.52it/s] 40%|████      | 160974/400000 [00:19<00:28, 8487.59it/s] 40%|████      | 161830/400000 [00:19<00:27, 8507.20it/s] 41%|████      | 162684/400000 [00:19<00:27, 8515.70it/s] 41%|████      | 163549/400000 [00:19<00:27, 8554.61it/s] 41%|████      | 164405/400000 [00:19<00:27, 8535.92it/s] 41%|████▏     | 165259/400000 [00:19<00:27, 8483.02it/s] 42%|████▏     | 166108/400000 [00:19<00:27, 8465.16it/s] 42%|████▏     | 166962/400000 [00:20<00:27, 8485.01it/s] 42%|████▏     | 167811/400000 [00:20<00:27, 8314.83it/s] 42%|████▏     | 168644/400000 [00:20<00:27, 8295.00it/s] 42%|████▏     | 169478/400000 [00:20<00:27, 8306.12it/s] 43%|████▎     | 170340/400000 [00:20<00:27, 8397.78it/s] 43%|████▎     | 171186/400000 [00:20<00:27, 8416.17it/s] 43%|████▎     | 172049/400000 [00:20<00:26, 8477.73it/s] 43%|████▎     | 172905/400000 [00:20<00:26, 8501.19it/s] 43%|████▎     | 173756/400000 [00:20<00:26, 8471.67it/s] 44%|████▎     | 174615/400000 [00:20<00:26, 8506.34it/s] 44%|████▍     | 175466/400000 [00:21<00:26, 8506.53it/s] 44%|████▍     | 176325/400000 [00:21<00:26, 8529.40it/s] 44%|████▍     | 177185/400000 [00:21<00:26, 8549.89it/s] 45%|████▍     | 178041/400000 [00:21<00:26, 8495.12it/s] 45%|████▍     | 178894/400000 [00:21<00:26, 8504.02it/s] 45%|████▍     | 179745/400000 [00:21<00:25, 8499.56it/s] 45%|████▌     | 180600/400000 [00:21<00:25, 8513.99it/s] 45%|████▌     | 181453/400000 [00:21<00:25, 8516.80it/s] 46%|████▌     | 182305/400000 [00:21<00:25, 8480.74it/s] 46%|████▌     | 183154/400000 [00:21<00:25, 8393.74it/s] 46%|████▌     | 184010/400000 [00:22<00:25, 8441.56it/s] 46%|████▌     | 184855/400000 [00:22<00:25, 8292.41it/s] 46%|████▋     | 185685/400000 [00:22<00:26, 8166.12it/s] 47%|████▋     | 186503/400000 [00:22<00:26, 8098.78it/s] 47%|████▋     | 187314/400000 [00:22<00:26, 8075.04it/s] 47%|████▋     | 188163/400000 [00:22<00:25, 8195.16it/s] 47%|████▋     | 189020/400000 [00:22<00:25, 8301.56it/s] 47%|████▋     | 189852/400000 [00:22<00:25, 8234.26it/s] 48%|████▊     | 190677/400000 [00:22<00:25, 8180.44it/s] 48%|████▊     | 191496/400000 [00:22<00:25, 8104.41it/s] 48%|████▊     | 192354/400000 [00:23<00:25, 8239.03it/s] 48%|████▊     | 193198/400000 [00:23<00:24, 8297.33it/s] 49%|████▊     | 194029/400000 [00:23<00:24, 8297.36it/s] 49%|████▊     | 194860/400000 [00:23<00:24, 8241.50it/s] 49%|████▉     | 195702/400000 [00:23<00:24, 8294.03it/s] 49%|████▉     | 196556/400000 [00:23<00:24, 8364.70it/s] 49%|████▉     | 197399/400000 [00:23<00:24, 8381.62it/s] 50%|████▉     | 198254/400000 [00:23<00:23, 8429.51it/s] 50%|████▉     | 199103/400000 [00:23<00:23, 8447.42it/s] 50%|████▉     | 199962/400000 [00:23<00:23, 8489.54it/s] 50%|█████     | 200812/400000 [00:24<00:23, 8476.56it/s] 50%|█████     | 201660/400000 [00:24<00:23, 8441.02it/s] 51%|█████     | 202505/400000 [00:24<00:23, 8382.34it/s] 51%|█████     | 203360/400000 [00:24<00:23, 8430.93it/s] 51%|█████     | 204209/400000 [00:24<00:23, 8448.01it/s] 51%|█████▏    | 205054/400000 [00:24<00:23, 8400.58it/s] 51%|█████▏    | 205895/400000 [00:24<00:23, 8318.15it/s] 52%|█████▏    | 206749/400000 [00:24<00:23, 8382.15it/s] 52%|█████▏    | 207599/400000 [00:24<00:22, 8415.74it/s] 52%|█████▏    | 208441/400000 [00:24<00:22, 8413.66it/s] 52%|█████▏    | 209283/400000 [00:25<00:22, 8404.41it/s] 53%|█████▎    | 210124/400000 [00:25<00:22, 8320.94it/s] 53%|█████▎    | 210989/400000 [00:25<00:22, 8416.44it/s] 53%|█████▎    | 211838/400000 [00:25<00:22, 8437.07it/s] 53%|█████▎    | 212683/400000 [00:25<00:22, 8388.43it/s] 53%|█████▎    | 213530/400000 [00:25<00:22, 8412.19it/s] 54%|█████▎    | 214390/400000 [00:25<00:21, 8467.25it/s] 54%|█████▍    | 215237/400000 [00:25<00:21, 8402.06it/s] 54%|█████▍    | 216078/400000 [00:25<00:21, 8368.85it/s] 54%|█████▍    | 216916/400000 [00:25<00:22, 8270.48it/s] 54%|█████▍    | 217764/400000 [00:26<00:21, 8331.24it/s] 55%|█████▍    | 218612/400000 [00:26<00:21, 8374.30it/s] 55%|█████▍    | 219457/400000 [00:26<00:21, 8394.63it/s] 55%|█████▌    | 220297/400000 [00:26<00:21, 8308.70it/s] 55%|█████▌    | 221162/400000 [00:26<00:21, 8406.85it/s] 56%|█████▌    | 222024/400000 [00:26<00:21, 8466.99it/s] 56%|█████▌    | 222872/400000 [00:26<00:21, 8392.11it/s] 56%|█████▌    | 223721/400000 [00:26<00:20, 8418.72it/s] 56%|█████▌    | 224577/400000 [00:26<00:20, 8459.34it/s] 56%|█████▋    | 225424/400000 [00:27<00:20, 8393.33it/s] 57%|█████▋    | 226278/400000 [00:27<00:20, 8436.23it/s] 57%|█████▋    | 227122/400000 [00:27<00:20, 8336.08it/s] 57%|█████▋    | 227980/400000 [00:27<00:20, 8406.55it/s] 57%|█████▋    | 228822/400000 [00:27<00:20, 8377.36it/s] 57%|█████▋    | 229661/400000 [00:27<00:20, 8283.29it/s] 58%|█████▊    | 230490/400000 [00:27<00:20, 8213.17it/s] 58%|█████▊    | 231319/400000 [00:27<00:20, 8234.39it/s] 58%|█████▊    | 232166/400000 [00:27<00:20, 8302.88it/s] 58%|█████▊    | 233030/400000 [00:27<00:19, 8400.06it/s] 58%|█████▊    | 233897/400000 [00:28<00:19, 8476.73it/s] 59%|█████▊    | 234762/400000 [00:28<00:19, 8526.97it/s] 59%|█████▉    | 235616/400000 [00:28<00:19, 8513.33it/s] 59%|█████▉    | 236469/400000 [00:28<00:19, 8517.29it/s] 59%|█████▉    | 237321/400000 [00:28<00:19, 8433.49it/s] 60%|█████▉    | 238165/400000 [00:28<00:19, 8301.77it/s] 60%|█████▉    | 239002/400000 [00:28<00:19, 8321.09it/s] 60%|█████▉    | 239835/400000 [00:28<00:19, 8309.46it/s] 60%|██████    | 240670/400000 [00:28<00:19, 8319.79it/s] 60%|██████    | 241503/400000 [00:28<00:19, 8231.56it/s] 61%|██████    | 242327/400000 [00:29<00:19, 8189.07it/s] 61%|██████    | 243170/400000 [00:29<00:18, 8257.46it/s] 61%|██████    | 244016/400000 [00:29<00:18, 8315.31it/s] 61%|██████    | 244848/400000 [00:29<00:18, 8276.24it/s] 61%|██████▏   | 245696/400000 [00:29<00:18, 8333.90it/s] 62%|██████▏   | 246541/400000 [00:29<00:18, 8368.01it/s] 62%|██████▏   | 247386/400000 [00:29<00:18, 8390.88it/s] 62%|██████▏   | 248226/400000 [00:29<00:18, 8387.88it/s] 62%|██████▏   | 249065/400000 [00:29<00:18, 8227.39it/s] 62%|██████▏   | 249889/400000 [00:29<00:18, 8201.25it/s] 63%|██████▎   | 250719/400000 [00:30<00:18, 8228.38it/s] 63%|██████▎   | 251543/400000 [00:30<00:18, 8172.48it/s] 63%|██████▎   | 252398/400000 [00:30<00:17, 8281.51it/s] 63%|██████▎   | 253250/400000 [00:30<00:17, 8350.19it/s] 64%|██████▎   | 254086/400000 [00:30<00:17, 8347.13it/s] 64%|██████▎   | 254922/400000 [00:30<00:17, 8282.34it/s] 64%|██████▍   | 255751/400000 [00:30<00:17, 8232.80it/s] 64%|██████▍   | 256583/400000 [00:30<00:17, 8256.71it/s] 64%|██████▍   | 257422/400000 [00:30<00:17, 8295.82it/s] 65%|██████▍   | 258273/400000 [00:30<00:16, 8356.40it/s] 65%|██████▍   | 259109/400000 [00:31<00:16, 8313.39it/s] 65%|██████▍   | 259941/400000 [00:31<00:16, 8246.58it/s] 65%|██████▌   | 260775/400000 [00:31<00:16, 8273.14it/s] 65%|██████▌   | 261636/400000 [00:31<00:16, 8370.37it/s] 66%|██████▌   | 262474/400000 [00:31<00:16, 8258.19it/s] 66%|██████▌   | 263301/400000 [00:31<00:16, 8175.63it/s] 66%|██████▌   | 264120/400000 [00:31<00:16, 8118.30it/s] 66%|██████▌   | 264933/400000 [00:31<00:16, 8074.03it/s] 66%|██████▋   | 265768/400000 [00:31<00:16, 8154.66it/s] 67%|██████▋   | 266585/400000 [00:31<00:16, 8158.59it/s] 67%|██████▋   | 267419/400000 [00:32<00:16, 8210.68it/s] 67%|██████▋   | 268283/400000 [00:32<00:15, 8332.48it/s] 67%|██████▋   | 269136/400000 [00:32<00:15, 8388.16it/s] 67%|██████▋   | 269976/400000 [00:32<00:15, 8391.28it/s] 68%|██████▊   | 270816/400000 [00:32<00:15, 8379.61it/s] 68%|██████▊   | 271655/400000 [00:32<00:15, 8351.74it/s] 68%|██████▊   | 272527/400000 [00:32<00:15, 8457.46it/s] 68%|██████▊   | 273398/400000 [00:32<00:14, 8530.64it/s] 69%|██████▊   | 274252/400000 [00:32<00:14, 8488.63it/s] 69%|██████▉   | 275102/400000 [00:32<00:15, 8300.76it/s] 69%|██████▉   | 275962/400000 [00:33<00:14, 8387.15it/s] 69%|██████▉   | 276842/400000 [00:33<00:14, 8506.69it/s] 69%|██████▉   | 277713/400000 [00:33<00:14, 8566.39it/s] 70%|██████▉   | 278571/400000 [00:33<00:14, 8518.51it/s] 70%|██████▉   | 279424/400000 [00:33<00:14, 8487.37it/s] 70%|███████   | 280274/400000 [00:33<00:14, 8447.35it/s] 70%|███████   | 281120/400000 [00:33<00:14, 8414.71it/s] 70%|███████   | 281980/400000 [00:33<00:13, 8469.35it/s] 71%|███████   | 282828/400000 [00:33<00:13, 8451.46it/s] 71%|███████   | 283674/400000 [00:33<00:13, 8391.02it/s] 71%|███████   | 284514/400000 [00:34<00:13, 8373.66it/s] 71%|███████▏  | 285352/400000 [00:34<00:13, 8328.71it/s] 72%|███████▏  | 286213/400000 [00:34<00:13, 8411.06it/s] 72%|███████▏  | 287058/400000 [00:34<00:13, 8422.11it/s] 72%|███████▏  | 287901/400000 [00:34<00:13, 8358.58it/s] 72%|███████▏  | 288738/400000 [00:34<00:13, 8313.49it/s] 72%|███████▏  | 289594/400000 [00:34<00:13, 8385.64it/s] 73%|███████▎  | 290453/400000 [00:34<00:12, 8444.25it/s] 73%|███████▎  | 291298/400000 [00:34<00:12, 8440.26it/s] 73%|███████▎  | 292143/400000 [00:35<00:13, 8253.87it/s] 73%|███████▎  | 292970/400000 [00:35<00:13, 8159.37it/s] 73%|███████▎  | 293787/400000 [00:35<00:13, 8150.70it/s] 74%|███████▎  | 294604/400000 [00:35<00:12, 8154.40it/s] 74%|███████▍  | 295420/400000 [00:35<00:13, 7952.04it/s] 74%|███████▍  | 296258/400000 [00:35<00:12, 8074.73it/s] 74%|███████▍  | 297107/400000 [00:35<00:12, 8192.75it/s] 74%|███████▍  | 297960/400000 [00:35<00:12, 8288.69it/s] 75%|███████▍  | 298814/400000 [00:35<00:12, 8361.26it/s] 75%|███████▍  | 299654/400000 [00:35<00:11, 8372.58it/s] 75%|███████▌  | 300492/400000 [00:36<00:11, 8337.41it/s] 75%|███████▌  | 301334/400000 [00:36<00:11, 8361.02it/s] 76%|███████▌  | 302171/400000 [00:36<00:11, 8362.73it/s] 76%|███████▌  | 303018/400000 [00:36<00:11, 8393.35it/s] 76%|███████▌  | 303876/400000 [00:36<00:11, 8447.82it/s] 76%|███████▌  | 304728/400000 [00:36<00:11, 8469.30it/s] 76%|███████▋  | 305576/400000 [00:36<00:11, 8331.96it/s] 77%|███████▋  | 306410/400000 [00:36<00:11, 8268.52it/s] 77%|███████▋  | 307239/400000 [00:36<00:11, 8272.45it/s] 77%|███████▋  | 308092/400000 [00:36<00:11, 8346.59it/s] 77%|███████▋  | 308928/400000 [00:37<00:11, 8184.00it/s] 77%|███████▋  | 309748/400000 [00:37<00:11, 8160.51it/s] 78%|███████▊  | 310565/400000 [00:37<00:11, 8069.29it/s] 78%|███████▊  | 311405/400000 [00:37<00:10, 8163.51it/s] 78%|███████▊  | 312223/400000 [00:37<00:10, 8143.96it/s] 78%|███████▊  | 313055/400000 [00:37<00:10, 8193.66it/s] 78%|███████▊  | 313899/400000 [00:37<00:10, 8264.38it/s] 79%|███████▊  | 314726/400000 [00:37<00:10, 8184.29it/s] 79%|███████▉  | 315582/400000 [00:37<00:10, 8290.90it/s] 79%|███████▉  | 316437/400000 [00:37<00:09, 8364.48it/s] 79%|███████▉  | 317275/400000 [00:38<00:10, 8188.89it/s] 80%|███████▉  | 318096/400000 [00:38<00:10, 8071.18it/s] 80%|███████▉  | 318923/400000 [00:38<00:09, 8128.89it/s] 80%|███████▉  | 319785/400000 [00:38<00:09, 8269.35it/s] 80%|████████  | 320625/400000 [00:38<00:09, 8306.04it/s] 80%|████████  | 321479/400000 [00:38<00:09, 8373.71it/s] 81%|████████  | 322325/400000 [00:38<00:09, 8396.97it/s] 81%|████████  | 323181/400000 [00:38<00:09, 8443.53it/s] 81%|████████  | 324030/400000 [00:38<00:08, 8455.09it/s] 81%|████████  | 324885/400000 [00:38<00:08, 8480.94it/s] 81%|████████▏ | 325742/400000 [00:39<00:08, 8505.23it/s] 82%|████████▏ | 326593/400000 [00:39<00:08, 8441.96it/s] 82%|████████▏ | 327456/400000 [00:39<00:08, 8496.78it/s] 82%|████████▏ | 328306/400000 [00:39<00:08, 8431.76it/s] 82%|████████▏ | 329166/400000 [00:39<00:08, 8480.78it/s] 83%|████████▎ | 330017/400000 [00:39<00:08, 8487.66it/s] 83%|████████▎ | 330866/400000 [00:39<00:08, 8436.00it/s] 83%|████████▎ | 331715/400000 [00:39<00:08, 8450.94it/s] 83%|████████▎ | 332561/400000 [00:39<00:08, 8425.23it/s] 83%|████████▎ | 333417/400000 [00:39<00:07, 8464.41it/s] 84%|████████▎ | 334264/400000 [00:40<00:07, 8312.37it/s] 84%|████████▍ | 335096/400000 [00:40<00:07, 8123.71it/s] 84%|████████▍ | 335956/400000 [00:40<00:07, 8260.22it/s] 84%|████████▍ | 336809/400000 [00:40<00:07, 8337.93it/s] 84%|████████▍ | 337658/400000 [00:40<00:07, 8382.57it/s] 85%|████████▍ | 338498/400000 [00:40<00:07, 8340.87it/s] 85%|████████▍ | 339333/400000 [00:40<00:07, 7972.59it/s] 85%|████████▌ | 340135/400000 [00:40<00:07, 7971.18it/s] 85%|████████▌ | 340986/400000 [00:40<00:07, 8118.89it/s] 85%|████████▌ | 341820/400000 [00:40<00:07, 8181.48it/s] 86%|████████▌ | 342685/400000 [00:41<00:06, 8315.46it/s] 86%|████████▌ | 343519/400000 [00:41<00:06, 8087.92it/s] 86%|████████▌ | 344381/400000 [00:41<00:06, 8239.11it/s] 86%|████████▋ | 345243/400000 [00:41<00:06, 8349.18it/s] 87%|████████▋ | 346102/400000 [00:41<00:06, 8418.55it/s] 87%|████████▋ | 346965/400000 [00:41<00:06, 8476.93it/s] 87%|████████▋ | 347814/400000 [00:41<00:06, 8465.83it/s] 87%|████████▋ | 348662/400000 [00:41<00:06, 8425.20it/s] 87%|████████▋ | 349506/400000 [00:41<00:06, 8312.68it/s] 88%|████████▊ | 350354/400000 [00:42<00:05, 8360.93it/s] 88%|████████▊ | 351207/400000 [00:42<00:05, 8408.94it/s] 88%|████████▊ | 352049/400000 [00:42<00:05, 8381.64it/s] 88%|████████▊ | 352898/400000 [00:42<00:05, 8412.30it/s] 88%|████████▊ | 353745/400000 [00:42<00:05, 8427.01it/s] 89%|████████▊ | 354593/400000 [00:42<00:05, 8442.08it/s] 89%|████████▉ | 355454/400000 [00:42<00:05, 8490.57it/s] 89%|████████▉ | 356304/400000 [00:42<00:05, 8457.32it/s] 89%|████████▉ | 357150/400000 [00:42<00:05, 8310.63it/s] 90%|████████▉ | 358013/400000 [00:42<00:04, 8401.85it/s] 90%|████████▉ | 358870/400000 [00:43<00:04, 8449.54it/s] 90%|████████▉ | 359729/400000 [00:43<00:04, 8488.37it/s] 90%|█████████ | 360579/400000 [00:43<00:04, 8407.67it/s] 90%|█████████ | 361421/400000 [00:43<00:04, 8268.99it/s] 91%|█████████ | 362249/400000 [00:43<00:04, 8167.00it/s] 91%|█████████ | 363084/400000 [00:43<00:04, 8218.34it/s] 91%|█████████ | 363934/400000 [00:43<00:04, 8298.76it/s] 91%|█████████ | 364768/400000 [00:43<00:04, 8310.28it/s] 91%|█████████▏| 365621/400000 [00:43<00:04, 8373.52it/s] 92%|█████████▏| 366471/400000 [00:43<00:03, 8409.97it/s] 92%|█████████▏| 367316/400000 [00:44<00:03, 8419.39it/s] 92%|█████████▏| 368159/400000 [00:44<00:03, 8318.44it/s] 92%|█████████▏| 369024/400000 [00:44<00:03, 8412.63it/s] 92%|█████████▏| 369892/400000 [00:44<00:03, 8487.98it/s] 93%|█████████▎| 370744/400000 [00:44<00:03, 8496.67it/s] 93%|█████████▎| 371595/400000 [00:44<00:03, 8464.37it/s] 93%|█████████▎| 372458/400000 [00:44<00:03, 8513.09it/s] 93%|█████████▎| 373310/400000 [00:44<00:03, 8505.49it/s] 94%|█████████▎| 374175/400000 [00:44<00:03, 8547.75it/s] 94%|█████████▍| 375030/400000 [00:44<00:02, 8508.47it/s] 94%|█████████▍| 375882/400000 [00:45<00:02, 8465.83it/s] 94%|█████████▍| 376743/400000 [00:45<00:02, 8506.84it/s] 94%|█████████▍| 377601/400000 [00:45<00:02, 8528.34it/s] 95%|█████████▍| 378458/400000 [00:45<00:02, 8540.46it/s] 95%|█████████▍| 379313/400000 [00:45<00:02, 8520.15it/s] 95%|█████████▌| 380166/400000 [00:45<00:02, 8398.07it/s] 95%|█████████▌| 381007/400000 [00:45<00:02, 8401.11it/s] 95%|█████████▌| 381848/400000 [00:45<00:02, 8212.33it/s] 96%|█████████▌| 382671/400000 [00:45<00:02, 7963.07it/s] 96%|█████████▌| 383515/400000 [00:45<00:02, 8099.45it/s] 96%|█████████▌| 384357/400000 [00:46<00:01, 8168.66it/s] 96%|█████████▋| 385211/400000 [00:46<00:01, 8275.68it/s] 97%|█████████▋| 386041/400000 [00:46<00:01, 8255.42it/s] 97%|█████████▋| 386891/400000 [00:46<00:01, 8325.99it/s] 97%|█████████▋| 387725/400000 [00:46<00:01, 8172.06it/s] 97%|█████████▋| 388544/400000 [00:46<00:01, 8130.08it/s] 97%|█████████▋| 389384/400000 [00:46<00:01, 8207.54it/s] 98%|█████████▊| 390216/400000 [00:46<00:01, 8239.66it/s] 98%|█████████▊| 391041/400000 [00:46<00:01, 8238.97it/s] 98%|█████████▊| 391893/400000 [00:46<00:00, 8320.62it/s] 98%|█████████▊| 392738/400000 [00:47<00:00, 8356.55it/s] 98%|█████████▊| 393595/400000 [00:47<00:00, 8418.46it/s] 99%|█████████▊| 394438/400000 [00:47<00:00, 8384.52it/s] 99%|█████████▉| 395277/400000 [00:47<00:00, 8380.90it/s] 99%|█████████▉| 396116/400000 [00:47<00:00, 8360.10it/s] 99%|█████████▉| 396962/400000 [00:47<00:00, 8388.82it/s] 99%|█████████▉| 397808/400000 [00:47<00:00, 8407.53it/s]100%|█████████▉| 398649/400000 [00:47<00:00, 8289.44it/s]100%|█████████▉| 399479/400000 [00:47<00:00, 8232.72it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8341.59it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1598309b70> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011178917212888748 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010788613139187612 	 Accuracy: 61

  model saves at 61% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16135 out of table with 15929 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16135 out of table with 15929 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

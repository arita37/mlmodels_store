
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb51e2befd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 15:11:51.634791
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 15:11:51.638061
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 15:11:51.640870
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 15:11:51.643798
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb52a2d64a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 353722.0938
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 262322.5625
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 158994.6719
Epoch 4/10

1/1 [==============================] - 0s 101ms/step - loss: 85134.7969
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 43682.9609
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 24051.1660
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 14460.0146
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 9514.8818
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 6660.0752
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 5061.3354

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.19719522 -0.19073129  1.359824    0.64623034  0.4629847   0.68330824
  -1.6001729   0.80122006  1.0525473   0.5742843  -1.0249004   0.99537003
  -2.0956273   1.9848266   0.3475849   0.53206336 -0.923948   -0.33470285
   0.29490238  1.3763081   0.09893553 -0.2733704   0.32806522 -0.45263717
   1.8700631  -0.26454988 -1.102665    2.7350233   0.88260114  0.5321753
   1.9762981  -0.14268705  1.7699534   0.0808831  -0.84636056 -1.8610795
  -0.7965156  -1.2584804  -0.5842141   1.3103919  -0.9699646   1.7227657
  -1.7097032   0.76228964 -0.5759367   0.33331192 -1.6622528  -0.8191136
  -0.5122479   0.36322352  0.28273737 -0.42540944 -0.7380637   0.82587767
   0.6121341   0.34164608  0.12580681  0.46822578 -0.6275441   1.4245789
   0.36099336 10.952392    9.577154   10.071593    9.607407    8.188347
   8.766591    8.658109   10.389565   10.069079    9.080609    9.452868
  10.8629265   9.005996    8.67066     9.789391    9.286948   11.994111
   8.326627    8.756407   10.001338   10.3841915   9.838988    9.751947
   9.573105    9.566746    8.724915    7.358133   10.430889   10.004056
   9.655526   10.225675    8.335973   11.4411545   9.26564     9.756666
   9.482522    9.052284   11.273938    9.815839    9.376942    9.868929
   9.38253    11.611168    8.409725   10.242014   10.177473   10.164043
   9.69225    10.8487625  11.376604    7.7702656   9.321028    9.126296
  10.098546   10.425342    9.000727    8.635417    8.737393    8.663035
   0.88768417 -0.05125996 -0.9200669   1.1114113  -1.641402   -0.32966202
   0.50672036  1.0441277  -0.5951852  -1.321079    0.04805329  0.9068722
  -0.42726743  1.0593011   1.5160255   0.3001576  -0.14322023  0.3310834
  -0.88613695 -0.16757694  0.04693256  0.8451404  -1.7158889   0.0894815
   0.21639761 -0.45931762  1.497829    0.22568032  0.21290635  0.7506225
   0.7139195   0.54067326 -0.63817143 -0.01751673  1.3018578   2.1419034
  -0.759865    1.028234    1.5532558  -0.4195341  -0.05308262  2.856748
   1.2510855   0.6339244  -0.67460704  0.3801447  -0.8448572   1.3072655
  -1.6057768   1.8874317  -0.9878085  -0.20112476  0.01525962  2.1831565
  -1.037454   -1.0891662   0.12556495 -0.841178   -0.43857118  0.83140635
   1.503825    1.2585571   0.7147168   0.8445965   1.1111778   1.1445681
   0.66759825  0.24214733  1.2637869   1.7390132   0.8500806   1.3912187
   2.4937687   0.3529986   0.92850554  0.20664865  2.2278068   0.5457751
   1.3456895   0.36374593  0.7070092   0.75253326  0.6473618   2.7985682
   0.42913842  1.2040608   0.27951348  0.63875675  1.1259053   1.7009058
   0.66298103  0.1791755   1.2963666   1.837949    2.1553736   1.146395
   0.97486335  0.656399    0.5059818   2.014264    0.9341988   1.0570316
   1.4365242   1.4455006   0.551158    0.31187397  0.71939975  2.689895
   2.0329242   0.6829816   0.8477073   0.8024429   0.8886335   0.18561965
   1.920701    1.7694722   0.5877516   0.24236739  1.8783419   1.2863655
   0.06943184 10.425071    7.5185056  10.129189    9.7716255   9.574309
  10.509534    9.634721    9.849694    8.14778    10.25154    10.53143
  11.082644   10.672192    9.6733465   9.862617   11.893176   11.184461
   9.77541     9.383975   10.252102   10.774534   10.276516   10.247571
  10.793525   10.482327   10.830496    9.87039    10.515917    9.478111
   8.740525   10.317614   10.226139    8.339065    9.153936    8.1622
   8.874269   11.71528    11.325509    9.262452   10.401698    8.561724
   8.23433     9.125723    9.980145    9.026304   10.85516    10.724339
   9.593925    9.659493    8.596817    9.913076    8.758094    9.147905
   8.409069    9.151034    8.611106    8.717437   10.445209   11.135907
   1.1366382   0.72544277  1.0109485   1.7653867   0.5071503   1.5542064
   1.4369049   0.21646619  0.3970616   2.5725117   0.4794681   0.9280537
   0.4417435   0.40153134  2.164886    2.3890266   1.5350852   2.3006902
   0.2995906   0.44908595  0.62870073  0.3228724   0.4945097   1.4649075
   0.7863745   0.19968045  0.9241394   1.2817519   1.3505278   0.4169575
   1.1757517   0.82446414  1.6709427   0.6169219   1.5477633   1.6909015
   1.89256     2.1299906   0.7145042   0.41585267  0.62925905  0.3463266
   1.8915238   0.56371975  0.61208767  0.3979876   1.607051    0.29333186
   0.40555912  2.5056329   1.3443022   1.8115809   1.0050044   1.115892
   2.0672426   0.19674075  1.0160414   1.5595653   1.4953513   0.31235838
  -6.237096   14.214344   -6.6774807 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 15:12:01.639022
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5947
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 15:12:01.642755
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8595.05
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 15:12:01.646500
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4482
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 15:12:01.650966
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -768.747
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140415535310664
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140414573858376
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140414573944960
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414573945464
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414573945968
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414573946472

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb526159ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.502263
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.475305
grad_step = 000002, loss = 0.458966
grad_step = 000003, loss = 0.445115
grad_step = 000004, loss = 0.428064
grad_step = 000005, loss = 0.412978
grad_step = 000006, loss = 0.402220
grad_step = 000007, loss = 0.393443
grad_step = 000008, loss = 0.381582
grad_step = 000009, loss = 0.369432
grad_step = 000010, loss = 0.359231
grad_step = 000011, loss = 0.349536
grad_step = 000012, loss = 0.339484
grad_step = 000013, loss = 0.328695
grad_step = 000014, loss = 0.317674
grad_step = 000015, loss = 0.307124
grad_step = 000016, loss = 0.297160
grad_step = 000017, loss = 0.287204
grad_step = 000018, loss = 0.274040
grad_step = 000019, loss = 0.259559
grad_step = 000020, loss = 0.244889
grad_step = 000021, loss = 0.231307
grad_step = 000022, loss = 0.220620
grad_step = 000023, loss = 0.213364
grad_step = 000024, loss = 0.204474
grad_step = 000025, loss = 0.193942
grad_step = 000026, loss = 0.184674
grad_step = 000027, loss = 0.176937
grad_step = 000028, loss = 0.169593
grad_step = 000029, loss = 0.161840
grad_step = 000030, loss = 0.153701
grad_step = 000031, loss = 0.145789
grad_step = 000032, loss = 0.138565
grad_step = 000033, loss = 0.132010
grad_step = 000034, loss = 0.125045
grad_step = 000035, loss = 0.117712
grad_step = 000036, loss = 0.110448
grad_step = 000037, loss = 0.103811
grad_step = 000038, loss = 0.097602
grad_step = 000039, loss = 0.091490
grad_step = 000040, loss = 0.085477
grad_step = 000041, loss = 0.079642
grad_step = 000042, loss = 0.074353
grad_step = 000043, loss = 0.069821
grad_step = 000044, loss = 0.065599
grad_step = 000045, loss = 0.061127
grad_step = 000046, loss = 0.056841
grad_step = 000047, loss = 0.053322
grad_step = 000048, loss = 0.050365
grad_step = 000049, loss = 0.047350
grad_step = 000050, loss = 0.044163
grad_step = 000051, loss = 0.041134
grad_step = 000052, loss = 0.038439
grad_step = 000053, loss = 0.035928
grad_step = 000054, loss = 0.033415
grad_step = 000055, loss = 0.031104
grad_step = 000056, loss = 0.029096
grad_step = 000057, loss = 0.027225
grad_step = 000058, loss = 0.025374
grad_step = 000059, loss = 0.023611
grad_step = 000060, loss = 0.022027
grad_step = 000061, loss = 0.020592
grad_step = 000062, loss = 0.019210
grad_step = 000063, loss = 0.017893
grad_step = 000064, loss = 0.016723
grad_step = 000065, loss = 0.015683
grad_step = 000066, loss = 0.014698
grad_step = 000067, loss = 0.013750
grad_step = 000068, loss = 0.012881
grad_step = 000069, loss = 0.012078
grad_step = 000070, loss = 0.011273
grad_step = 000071, loss = 0.010510
grad_step = 000072, loss = 0.009847
grad_step = 000073, loss = 0.009237
grad_step = 000074, loss = 0.008649
grad_step = 000075, loss = 0.008117
grad_step = 000076, loss = 0.007644
grad_step = 000077, loss = 0.007184
grad_step = 000078, loss = 0.006743
grad_step = 000079, loss = 0.006355
grad_step = 000080, loss = 0.005990
grad_step = 000081, loss = 0.005635
grad_step = 000082, loss = 0.005318
grad_step = 000083, loss = 0.005032
grad_step = 000084, loss = 0.004757
grad_step = 000085, loss = 0.004508
grad_step = 000086, loss = 0.004289
grad_step = 000087, loss = 0.004084
grad_step = 000088, loss = 0.003898
grad_step = 000089, loss = 0.003734
grad_step = 000090, loss = 0.003581
grad_step = 000091, loss = 0.003442
grad_step = 000092, loss = 0.003320
grad_step = 000093, loss = 0.003208
grad_step = 000094, loss = 0.003110
grad_step = 000095, loss = 0.003019
grad_step = 000096, loss = 0.002936
grad_step = 000097, loss = 0.002862
grad_step = 000098, loss = 0.002793
grad_step = 000099, loss = 0.002731
grad_step = 000100, loss = 0.002678
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002630
grad_step = 000102, loss = 0.002586
grad_step = 000103, loss = 0.002549
grad_step = 000104, loss = 0.002513
grad_step = 000105, loss = 0.002481
grad_step = 000106, loss = 0.002453
grad_step = 000107, loss = 0.002426
grad_step = 000108, loss = 0.002403
grad_step = 000109, loss = 0.002382
grad_step = 000110, loss = 0.002363
grad_step = 000111, loss = 0.002347
grad_step = 000112, loss = 0.002331
grad_step = 000113, loss = 0.002318
grad_step = 000114, loss = 0.002306
grad_step = 000115, loss = 0.002295
grad_step = 000116, loss = 0.002285
grad_step = 000117, loss = 0.002276
grad_step = 000118, loss = 0.002268
grad_step = 000119, loss = 0.002261
grad_step = 000120, loss = 0.002254
grad_step = 000121, loss = 0.002248
grad_step = 000122, loss = 0.002242
grad_step = 000123, loss = 0.002236
grad_step = 000124, loss = 0.002231
grad_step = 000125, loss = 0.002226
grad_step = 000126, loss = 0.002221
grad_step = 000127, loss = 0.002216
grad_step = 000128, loss = 0.002212
grad_step = 000129, loss = 0.002207
grad_step = 000130, loss = 0.002202
grad_step = 000131, loss = 0.002197
grad_step = 000132, loss = 0.002192
grad_step = 000133, loss = 0.002187
grad_step = 000134, loss = 0.002181
grad_step = 000135, loss = 0.002176
grad_step = 000136, loss = 0.002171
grad_step = 000137, loss = 0.002165
grad_step = 000138, loss = 0.002159
grad_step = 000139, loss = 0.002154
grad_step = 000140, loss = 0.002147
grad_step = 000141, loss = 0.002141
grad_step = 000142, loss = 0.002135
grad_step = 000143, loss = 0.002128
grad_step = 000144, loss = 0.002122
grad_step = 000145, loss = 0.002115
grad_step = 000146, loss = 0.002108
grad_step = 000147, loss = 0.002101
grad_step = 000148, loss = 0.002094
grad_step = 000149, loss = 0.002091
grad_step = 000150, loss = 0.002098
grad_step = 000151, loss = 0.002130
grad_step = 000152, loss = 0.002166
grad_step = 000153, loss = 0.002169
grad_step = 000154, loss = 0.002100
grad_step = 000155, loss = 0.002054
grad_step = 000156, loss = 0.002100
grad_step = 000157, loss = 0.002120
grad_step = 000158, loss = 0.002069
grad_step = 000159, loss = 0.002046
grad_step = 000160, loss = 0.002057
grad_step = 000161, loss = 0.002077
grad_step = 000162, loss = 0.002035
grad_step = 000163, loss = 0.002020
grad_step = 000164, loss = 0.002047
grad_step = 000165, loss = 0.002023
grad_step = 000166, loss = 0.001999
grad_step = 000167, loss = 0.002003
grad_step = 000168, loss = 0.002008
grad_step = 000169, loss = 0.002012
grad_step = 000170, loss = 0.001995
grad_step = 000171, loss = 0.001980
grad_step = 000172, loss = 0.001975
grad_step = 000173, loss = 0.001980
grad_step = 000174, loss = 0.001983
grad_step = 000175, loss = 0.001977
grad_step = 000176, loss = 0.001975
grad_step = 000177, loss = 0.001962
grad_step = 000178, loss = 0.001951
grad_step = 000179, loss = 0.001947
grad_step = 000180, loss = 0.001946
grad_step = 000181, loss = 0.001948
grad_step = 000182, loss = 0.001950
grad_step = 000183, loss = 0.001956
grad_step = 000184, loss = 0.001951
grad_step = 000185, loss = 0.001949
grad_step = 000186, loss = 0.001936
grad_step = 000187, loss = 0.001926
grad_step = 000188, loss = 0.001916
grad_step = 000189, loss = 0.001910
grad_step = 000190, loss = 0.001907
grad_step = 000191, loss = 0.001906
grad_step = 000192, loss = 0.001909
grad_step = 000193, loss = 0.001917
grad_step = 000194, loss = 0.001945
grad_step = 000195, loss = 0.001942
grad_step = 000196, loss = 0.001958
grad_step = 000197, loss = 0.001905
grad_step = 000198, loss = 0.001887
grad_step = 000199, loss = 0.001908
grad_step = 000200, loss = 0.001912
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001908
grad_step = 000202, loss = 0.001878
grad_step = 000203, loss = 0.001868
grad_step = 000204, loss = 0.001869
grad_step = 000205, loss = 0.001884
grad_step = 000206, loss = 0.001914
grad_step = 000207, loss = 0.001907
grad_step = 000208, loss = 0.001917
grad_step = 000209, loss = 0.001909
grad_step = 000210, loss = 0.001934
grad_step = 000211, loss = 0.001923
grad_step = 000212, loss = 0.001880
grad_step = 000213, loss = 0.001870
grad_step = 000214, loss = 0.001876
grad_step = 000215, loss = 0.001889
grad_step = 000216, loss = 0.001861
grad_step = 000217, loss = 0.001850
grad_step = 000218, loss = 0.001873
grad_step = 000219, loss = 0.001867
grad_step = 000220, loss = 0.001842
grad_step = 000221, loss = 0.001827
grad_step = 000222, loss = 0.001840
grad_step = 000223, loss = 0.001840
grad_step = 000224, loss = 0.001825
grad_step = 000225, loss = 0.001831
grad_step = 000226, loss = 0.001841
grad_step = 000227, loss = 0.001840
grad_step = 000228, loss = 0.001819
grad_step = 000229, loss = 0.001817
grad_step = 000230, loss = 0.001821
grad_step = 000231, loss = 0.001818
grad_step = 000232, loss = 0.001806
grad_step = 000233, loss = 0.001799
grad_step = 000234, loss = 0.001804
grad_step = 000235, loss = 0.001808
grad_step = 000236, loss = 0.001809
grad_step = 000237, loss = 0.001820
grad_step = 000238, loss = 0.001873
grad_step = 000239, loss = 0.001889
grad_step = 000240, loss = 0.001933
grad_step = 000241, loss = 0.001837
grad_step = 000242, loss = 0.001799
grad_step = 000243, loss = 0.001816
grad_step = 000244, loss = 0.001826
grad_step = 000245, loss = 0.001801
grad_step = 000246, loss = 0.001790
grad_step = 000247, loss = 0.001805
grad_step = 000248, loss = 0.001813
grad_step = 000249, loss = 0.001797
grad_step = 000250, loss = 0.001805
grad_step = 000251, loss = 0.001808
grad_step = 000252, loss = 0.001803
grad_step = 000253, loss = 0.001772
grad_step = 000254, loss = 0.001760
grad_step = 000255, loss = 0.001770
grad_step = 000256, loss = 0.001777
grad_step = 000257, loss = 0.001767
grad_step = 000258, loss = 0.001759
grad_step = 000259, loss = 0.001767
grad_step = 000260, loss = 0.001765
grad_step = 000261, loss = 0.001756
grad_step = 000262, loss = 0.001743
grad_step = 000263, loss = 0.001743
grad_step = 000264, loss = 0.001745
grad_step = 000265, loss = 0.001741
grad_step = 000266, loss = 0.001733
grad_step = 000267, loss = 0.001730
grad_step = 000268, loss = 0.001734
grad_step = 000269, loss = 0.001738
grad_step = 000270, loss = 0.001743
grad_step = 000271, loss = 0.001752
grad_step = 000272, loss = 0.001787
grad_step = 000273, loss = 0.001789
grad_step = 000274, loss = 0.001815
grad_step = 000275, loss = 0.001757
grad_step = 000276, loss = 0.001724
grad_step = 000277, loss = 0.001722
grad_step = 000278, loss = 0.001736
grad_step = 000279, loss = 0.001735
grad_step = 000280, loss = 0.001721
grad_step = 000281, loss = 0.001718
grad_step = 000282, loss = 0.001709
grad_step = 000283, loss = 0.001700
grad_step = 000284, loss = 0.001701
grad_step = 000285, loss = 0.001713
grad_step = 000286, loss = 0.001736
grad_step = 000287, loss = 0.001748
grad_step = 000288, loss = 0.001781
grad_step = 000289, loss = 0.001774
grad_step = 000290, loss = 0.001780
grad_step = 000291, loss = 0.001721
grad_step = 000292, loss = 0.001680
grad_step = 000293, loss = 0.001674
grad_step = 000294, loss = 0.001700
grad_step = 000295, loss = 0.001732
grad_step = 000296, loss = 0.001693
grad_step = 000297, loss = 0.001670
grad_step = 000298, loss = 0.001673
grad_step = 000299, loss = 0.001677
grad_step = 000300, loss = 0.001667
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001654
grad_step = 000302, loss = 0.001657
grad_step = 000303, loss = 0.001666
grad_step = 000304, loss = 0.001662
grad_step = 000305, loss = 0.001655
grad_step = 000306, loss = 0.001649
grad_step = 000307, loss = 0.001656
grad_step = 000308, loss = 0.001668
grad_step = 000309, loss = 0.001678
grad_step = 000310, loss = 0.001674
grad_step = 000311, loss = 0.001683
grad_step = 000312, loss = 0.001675
grad_step = 000313, loss = 0.001674
grad_step = 000314, loss = 0.001639
grad_step = 000315, loss = 0.001616
grad_step = 000316, loss = 0.001609
grad_step = 000317, loss = 0.001617
grad_step = 000318, loss = 0.001624
grad_step = 000319, loss = 0.001613
grad_step = 000320, loss = 0.001608
grad_step = 000321, loss = 0.001612
grad_step = 000322, loss = 0.001623
grad_step = 000323, loss = 0.001624
grad_step = 000324, loss = 0.001627
grad_step = 000325, loss = 0.001629
grad_step = 000326, loss = 0.001659
grad_step = 000327, loss = 0.001658
grad_step = 000328, loss = 0.001668
grad_step = 000329, loss = 0.001612
grad_step = 000330, loss = 0.001581
grad_step = 000331, loss = 0.001571
grad_step = 000332, loss = 0.001574
grad_step = 000333, loss = 0.001578
grad_step = 000334, loss = 0.001578
grad_step = 000335, loss = 0.001598
grad_step = 000336, loss = 0.001612
grad_step = 000337, loss = 0.001626
grad_step = 000338, loss = 0.001598
grad_step = 000339, loss = 0.001582
grad_step = 000340, loss = 0.001562
grad_step = 000341, loss = 0.001555
grad_step = 000342, loss = 0.001539
grad_step = 000343, loss = 0.001529
grad_step = 000344, loss = 0.001533
grad_step = 000345, loss = 0.001544
grad_step = 000346, loss = 0.001560
grad_step = 000347, loss = 0.001557
grad_step = 000348, loss = 0.001573
grad_step = 000349, loss = 0.001580
grad_step = 000350, loss = 0.001606
grad_step = 000351, loss = 0.001579
grad_step = 000352, loss = 0.001561
grad_step = 000353, loss = 0.001523
grad_step = 000354, loss = 0.001507
grad_step = 000355, loss = 0.001504
grad_step = 000356, loss = 0.001514
grad_step = 000357, loss = 0.001530
grad_step = 000358, loss = 0.001529
grad_step = 000359, loss = 0.001534
grad_step = 000360, loss = 0.001510
grad_step = 000361, loss = 0.001494
grad_step = 000362, loss = 0.001483
grad_step = 000363, loss = 0.001477
grad_step = 000364, loss = 0.001473
grad_step = 000365, loss = 0.001471
grad_step = 000366, loss = 0.001476
grad_step = 000367, loss = 0.001483
grad_step = 000368, loss = 0.001497
grad_step = 000369, loss = 0.001502
grad_step = 000370, loss = 0.001520
grad_step = 000371, loss = 0.001515
grad_step = 000372, loss = 0.001524
grad_step = 000373, loss = 0.001490
grad_step = 000374, loss = 0.001468
grad_step = 000375, loss = 0.001444
grad_step = 000376, loss = 0.001439
grad_step = 000377, loss = 0.001449
grad_step = 000378, loss = 0.001456
grad_step = 000379, loss = 0.001469
grad_step = 000380, loss = 0.001463
grad_step = 000381, loss = 0.001463
grad_step = 000382, loss = 0.001446
grad_step = 000383, loss = 0.001432
grad_step = 000384, loss = 0.001419
grad_step = 000385, loss = 0.001412
grad_step = 000386, loss = 0.001410
grad_step = 000387, loss = 0.001412
grad_step = 000388, loss = 0.001418
grad_step = 000389, loss = 0.001424
grad_step = 000390, loss = 0.001440
grad_step = 000391, loss = 0.001446
grad_step = 000392, loss = 0.001469
grad_step = 000393, loss = 0.001462
grad_step = 000394, loss = 0.001470
grad_step = 000395, loss = 0.001437
grad_step = 000396, loss = 0.001410
grad_step = 000397, loss = 0.001384
grad_step = 000398, loss = 0.001379
grad_step = 000399, loss = 0.001392
grad_step = 000400, loss = 0.001402
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001408
grad_step = 000402, loss = 0.001393
grad_step = 000403, loss = 0.001383
grad_step = 000404, loss = 0.001368
grad_step = 000405, loss = 0.001359
grad_step = 000406, loss = 0.001353
grad_step = 000407, loss = 0.001351
grad_step = 000408, loss = 0.001352
grad_step = 000409, loss = 0.001356
grad_step = 000410, loss = 0.001367
grad_step = 000411, loss = 0.001374
grad_step = 000412, loss = 0.001398
grad_step = 000413, loss = 0.001407
grad_step = 000414, loss = 0.001444
grad_step = 000415, loss = 0.001427
grad_step = 000416, loss = 0.001428
grad_step = 000417, loss = 0.001371
grad_step = 000418, loss = 0.001331
grad_step = 000419, loss = 0.001319
grad_step = 000420, loss = 0.001336
grad_step = 000421, loss = 0.001355
grad_step = 000422, loss = 0.001347
grad_step = 000423, loss = 0.001335
grad_step = 000424, loss = 0.001313
grad_step = 000425, loss = 0.001301
grad_step = 000426, loss = 0.001298
grad_step = 000427, loss = 0.001300
grad_step = 000428, loss = 0.001307
grad_step = 000429, loss = 0.001313
grad_step = 000430, loss = 0.001327
grad_step = 000431, loss = 0.001328
grad_step = 000432, loss = 0.001343
grad_step = 000433, loss = 0.001329
grad_step = 000434, loss = 0.001328
grad_step = 000435, loss = 0.001304
grad_step = 000436, loss = 0.001286
grad_step = 000437, loss = 0.001272
grad_step = 000438, loss = 0.001268
grad_step = 000439, loss = 0.001268
grad_step = 000440, loss = 0.001269
grad_step = 000441, loss = 0.001275
grad_step = 000442, loss = 0.001278
grad_step = 000443, loss = 0.001284
grad_step = 000444, loss = 0.001280
grad_step = 000445, loss = 0.001282
grad_step = 000446, loss = 0.001274
grad_step = 000447, loss = 0.001278
grad_step = 000448, loss = 0.001274
grad_step = 000449, loss = 0.001273
grad_step = 000450, loss = 0.001257
grad_step = 000451, loss = 0.001247
grad_step = 000452, loss = 0.001231
grad_step = 000453, loss = 0.001223
grad_step = 000454, loss = 0.001219
grad_step = 000455, loss = 0.001218
grad_step = 000456, loss = 0.001216
grad_step = 000457, loss = 0.001213
grad_step = 000458, loss = 0.001217
grad_step = 000459, loss = 0.001226
grad_step = 000460, loss = 0.001256
grad_step = 000461, loss = 0.001285
grad_step = 000462, loss = 0.001370
grad_step = 000463, loss = 0.001380
grad_step = 000464, loss = 0.001470
grad_step = 000465, loss = 0.001310
grad_step = 000466, loss = 0.001237
grad_step = 000467, loss = 0.001202
grad_step = 000468, loss = 0.001231
grad_step = 000469, loss = 0.001264
grad_step = 000470, loss = 0.001214
grad_step = 000471, loss = 0.001186
grad_step = 000472, loss = 0.001199
grad_step = 000473, loss = 0.001218
grad_step = 000474, loss = 0.001238
grad_step = 000475, loss = 0.001202
grad_step = 000476, loss = 0.001192
grad_step = 000477, loss = 0.001177
grad_step = 000478, loss = 0.001175
grad_step = 000479, loss = 0.001175
grad_step = 000480, loss = 0.001165
grad_step = 000481, loss = 0.001173
grad_step = 000482, loss = 0.001166
grad_step = 000483, loss = 0.001161
grad_step = 000484, loss = 0.001142
grad_step = 000485, loss = 0.001128
grad_step = 000486, loss = 0.001129
grad_step = 000487, loss = 0.001134
grad_step = 000488, loss = 0.001143
grad_step = 000489, loss = 0.001132
grad_step = 000490, loss = 0.001120
grad_step = 000491, loss = 0.001109
grad_step = 000492, loss = 0.001107
grad_step = 000493, loss = 0.001109
grad_step = 000494, loss = 0.001109
grad_step = 000495, loss = 0.001101
grad_step = 000496, loss = 0.001093
grad_step = 000497, loss = 0.001086
grad_step = 000498, loss = 0.001082
grad_step = 000499, loss = 0.001083
grad_step = 000500, loss = 0.001083
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001080
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

  date_run                              2020-05-11 15:12:24.308924
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.260921
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 15:12:24.314083
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.178163
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 15:12:24.320262
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140609
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 15:12:24.325256
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.70725
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
0   2020-05-11 15:11:51.634791  ...    mean_absolute_error
1   2020-05-11 15:11:51.638061  ...     mean_squared_error
2   2020-05-11 15:11:51.640870  ...  median_absolute_error
3   2020-05-11 15:11:51.643798  ...               r2_score
4   2020-05-11 15:12:01.639022  ...    mean_absolute_error
5   2020-05-11 15:12:01.642755  ...     mean_squared_error
6   2020-05-11 15:12:01.646500  ...  median_absolute_error
7   2020-05-11 15:12:01.650966  ...               r2_score
8   2020-05-11 15:12:24.308924  ...    mean_absolute_error
9   2020-05-11 15:12:24.314083  ...     mean_squared_error
10  2020-05-11 15:12:24.320262  ...  median_absolute_error
11  2020-05-11 15:12:24.325256  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 154410.52it/s] 63%|██████▎   | 6209536/9912422 [00:00<00:16, 220350.51it/s]9920512it [00:00, 41150648.83it/s]                           
0it [00:00, ?it/s]32768it [00:00, 579984.61it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162962.07it/s]1654784it [00:00, 11306080.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 174357.39it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db1b4fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f238db38e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db17cba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db17ccc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23788ce0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db17ccc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f238d1131d0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db17ccc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23db17cba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f238db38e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f238db48eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f49752f4208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=619f5cdd359a818e2f412ea8e1dccf3f75f1e982b89f1e5f268c7ac6fc7568cf
  Stored in directory: /tmp/pip-ephem-wheel-cache-3670aj_x/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f490cedc240> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1867776/17464789 [==>...........................] - ETA: 0s
 7061504/17464789 [===========>..................] - ETA: 0s
14114816/17464789 [=======================>......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 15:13:48.009446: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 15:13:48.013991: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 15:13:48.014148: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5567edb51fd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 15:13:48.014162: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2986 - accuracy: 0.5240
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4903 - accuracy: 0.5115 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6462 - accuracy: 0.5013
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5409 - accuracy: 0.5082
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5133 - accuracy: 0.5100
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5418 - accuracy: 0.5081
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5823 - accuracy: 0.5055
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6068 - accuracy: 0.5039
11000/25000 [============>.................] - ETA: 4s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 3s - loss: 7.6091 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6338 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 2s - loss: 7.6431 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6331 - accuracy: 0.5022
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6269 - accuracy: 0.5026
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 9s 342us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 15:14:03.014630
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 15:14:03.014630  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 15:14:08.686927: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 15:14:08.691105: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 15:14:08.691252: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55688a8a6450 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 15:14:08.691265: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0e64c96d68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 916ms/step - loss: 1.4828 - crf_viterbi_accuracy: 0.2800 - val_loss: 1.3642 - val_crf_viterbi_accuracy: 0.3333

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0e8111cf60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4290 - accuracy: 0.5155 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5797 - accuracy: 0.5057
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5838 - accuracy: 0.5054
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6615 - accuracy: 0.5003
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5797 - accuracy: 0.5057
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5808 - accuracy: 0.5056
11000/25000 [============>.................] - ETA: 4s - loss: 7.5788 - accuracy: 0.5057
12000/25000 [=============>................] - ETA: 3s - loss: 7.5644 - accuracy: 0.5067
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5522 - accuracy: 0.5075
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
15000/25000 [=================>............] - ETA: 2s - loss: 7.5644 - accuracy: 0.5067
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5583 - accuracy: 0.5071
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5845 - accuracy: 0.5054
18000/25000 [====================>.........] - ETA: 2s - loss: 7.5908 - accuracy: 0.5049
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6013 - accuracy: 0.5043
20000/25000 [=======================>......] - ETA: 1s - loss: 7.5976 - accuracy: 0.5045
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6024 - accuracy: 0.5042
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6332 - accuracy: 0.5022
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6413 - accuracy: 0.5017
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 9s 346us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0e23032550> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<101:57:26, 2.35kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<71:36:54, 3.34kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<50:10:56, 4.77kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<35:06:52, 6.81kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:04<24:30:20, 9.73kB/s].vector_cache/glove.6B.zip:   1%|          | 8.68M/862M [00:04<17:03:17, 13.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:04<11:53:32, 19.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:04<8:16:17, 28.4kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:04<5:46:21, 40.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:04<4:01:08, 57.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:04<2:48:11, 82.5kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:04<1:57:05, 118kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:04<1:21:45, 168kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:05<56:58, 240kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:05<39:49, 341kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:05<27:47, 486kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:05<21:19, 633kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:07<16:46, 801kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:07<13:58, 961kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:07<10:42, 1.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:07<07:45, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:09<08:46, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:09<07:40, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:09<05:43, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:11<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:11<06:18, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:11<04:42, 2.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:13<06:27, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:13<05:55, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:13<04:29, 2.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:15<06:14, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:15<07:12, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:15<05:39, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:15<04:08, 3.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:17<08:58, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:17<07:42, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:17<05:44, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:19<07:02, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:19<06:19, 2.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:19<04:46, 2.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:21<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:21<05:50, 2.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:21<04:21, 2.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:23<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:23<05:38, 2.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:23<04:14, 3.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:25<05:59, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:25<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:25<04:10, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<03:06, 4.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:27<1:35:49, 133kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:27<1:08:25, 186kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:27<48:05, 264kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<36:31, 347kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<26:55, 471kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<19:09, 660kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<16:19, 772kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<12:47, 986kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:31<09:16, 1.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<09:25, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<07:55, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<05:52, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<07:01, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:15, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<04:39, 2.67MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<06:11, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<05:39, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<04:13, 2.92MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<03:07, 3.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<1:22:44, 149kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<59:14, 208kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<41:42, 295kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<31:59, 383kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<23:42, 517kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<16:53, 724kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<14:39, 832kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<11:34, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<08:24, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<08:42, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<07:24, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<05:30, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:40, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:59, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<04:30, 2.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<05:27, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<04:08, 2.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:41, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:16, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:51<04:00, 2.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<05:34, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<05:10, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<03:52, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<05:30, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<05:09, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<03:53, 3.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<05:28, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<05:06, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<03:52, 3.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<05:27, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<05:02, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<03:50, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<04:48, 2.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<03:52, 2.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<02:49, 4.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<20:14, 569kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<15:25, 746kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<11:02, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:51, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<1:20:16, 143kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<57:23, 199kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<40:23, 282kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<30:51, 369kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<22:48, 498kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<16:15, 698kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<13:59, 808kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<11:00, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<07:56, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:42, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<45:12, 249kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<32:38, 344kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<23:26, 479kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<16:27, 679kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<25:34, 437kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<19:06, 584kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<13:38, 816kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<12:05, 917kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<09:39, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<06:59, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<1:08:20, 161kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<48:58, 225kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<34:29, 319kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<26:38, 412kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<19:48, 553kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<14:08, 773kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<12:25, 877kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<09:52, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<07:11, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<07:33, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<06:28, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<04:48, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<06:03, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<06:33, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<05:05, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<03:54, 2.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<05:05, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<07:07, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<05:43, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<04:27, 2.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<03:14, 3.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<13:38, 778kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<13:00, 816kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<09:55, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<07:08, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<07:37, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<08:38, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<06:41, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<05:07, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<05:25, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<07:01, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<04:17, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<04:53, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<05:47, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<05:40, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<04:16, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<03:12, 3.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<06:46, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<06:18, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<04:42, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<03:30, 2.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<06:30, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<07:38, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<06:03, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:38<04:27, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<05:26, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<06:36, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<05:18, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<03:54, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<06:58, 1.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:42<07:32, 1.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<05:48, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<04:24, 2.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<05:03, 1.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:44<06:10, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<04:50, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<03:43, 2.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:46<04:34, 2.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:46<05:49, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<04:42, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<03:27, 2.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<07:33, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<07:44, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<05:56, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<04:26, 2.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<06:06, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<04:53, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<03:35, 2.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<07:14, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<07:28, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<05:50, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<04:13, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<07:32, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<08:05, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<06:13, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<04:36, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<05:15, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<06:27, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<05:12, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<03:48, 2.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<06:23, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<06:50, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<05:21, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:58<03:53, 2.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<07:30, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<07:51, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<06:08, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<06:59, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<07:28, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<05:45, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<04:17, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<06:10, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<04:53, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:04<03:34, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<05:04, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<05:50, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<04:34, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<03:20, 2.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<05:33, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<06:24, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<04:59, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<03:40, 2.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:09<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<06:03, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<04:44, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<03:35, 2.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<04:21, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<05:12, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<04:05, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<03:02, 2.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<04:34, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<05:19, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<04:11, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<03:05, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<04:40, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<05:22, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<04:12, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<03:10, 2.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<04:12, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<05:07, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<02:58, 2.97MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<05:05, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:20<05:38, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<04:23, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<03:12, 2.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<05:07, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<05:39, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<04:28, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<03:16, 2.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<06:46, 1.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<06:59, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<05:28, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<06:31, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<06:36, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<05:07, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<03:43, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<07:15, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<07:05, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<05:24, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<03:59, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<04:43, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<05:18, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<04:12, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<03:03, 2.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<06:41, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<06:58, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<05:26, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<03:55, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<06:17, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<06:22, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<04:52, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<03:37, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<04:22, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<04:55, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<03:51, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<02:49, 2.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:49, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<05:30, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<04:22, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<03:10, 2.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<05:57, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<06:04, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:40<03:24, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<04:30, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<05:15, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<04:06, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<03:03, 2.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:01, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:43<04:52, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<03:50, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:44<02:54, 2.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<03:42, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<04:23, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<03:30, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<02:34, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<08:05, 968kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<07:26, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<05:33, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<04:04, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<04:46, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<05:05, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<03:59, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<02:53, 2.66MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<08:10, 942kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<07:23, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:51<05:34, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<04:00, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<10:04, 758kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<09:01, 845kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<06:44, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<04:50, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<05:30, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<05:25, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<02:59, 2.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<15:30, 483kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<12:42, 589kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<09:22, 798kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<06:38, 1.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<07:30, 990kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<07:00, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<05:15, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<03:54, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:26, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<02:28, 2.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<05:39, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<05:50, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<04:29, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<03:19, 2.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:57, 1.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<04:03, 1.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:07, 2.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<02:18, 3.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<04:12, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<04:42, 1.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<02:43, 2.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<03:32, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<04:04, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<02:20, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<04:18, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<04:13, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:15, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:28, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:59, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:07, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:48, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<04:12, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<03:17, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<02:23, 2.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<04:47, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<04:53, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<03:47, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<02:43, 2.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<28:55, 233kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<21:45, 310kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:19<15:31, 433kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<10:55, 612kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<10:09, 657kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<08:36, 774kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<06:20, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<04:30, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<06:23, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<05:57, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<04:28, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<03:14, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<04:17, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<04:28, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<03:26, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<02:29, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:27<05:01, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<04:58, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<03:49, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<02:50, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<03:26, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<03:50, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<02:59, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:13, 2.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<03:13, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<03:41, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<02:52, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<02:07, 2.96MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<03:27, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<03:20, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<02:31, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<01:50, 3.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<06:27, 960kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<05:45, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<03:07, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<03:51, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<03:01, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<03:05, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<03:22, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<02:39, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<02:50, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<03:10, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<02:30, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:43, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<02:26, 2.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<02:39, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<02:21, 2.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<01:46, 3.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<02:47, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<03:05, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<02:24, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<01:48, 3.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<03:06, 1.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<02:27, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:57, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<02:18, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<01:42, 3.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:52<03:23, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<03:00, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<02:20, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<01:43, 3.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<03:16, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<03:20, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<02:35, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<02:41, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<02:56, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:19, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<02:29, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<02:12, 2.38MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<02:23, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<02:43, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:02<02:09, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:20, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:40, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<02:07, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<02:18, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<02:37, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<02:05, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:16, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:35, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<02:03, 2.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:14, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:33, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<02:01, 2.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:12, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:31, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<02:00, 2.43MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:10, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:14<02:29, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<01:58, 2.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<02:09, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:24, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<01:55, 2.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:06, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<02:22, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<01:51, 2.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<01:20, 3.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<04:38, 994kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<04:07, 1.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<03:06, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<02:54, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<02:55, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<02:13, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<01:38, 2.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:32, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<02:10, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<02:24, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<01:53, 2.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:01, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:17, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<01:48, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<01:57, 2.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<02:13, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<01:45, 2.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<01:54, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<02:10, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<01:43, 2.43MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:33<01:52, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<02:08, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<01:41, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<01:50, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<02:06, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<01:48, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<02:03, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:36, 2.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<01:11, 3.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<02:03, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<02:11, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<01:43, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<01:49, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<02:01, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<01:34, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<01:08, 3.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<03:01, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<03:31, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:44<02:46, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<02:02, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<02:08, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:12, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:42, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<01:13, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<05:07, 712kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<04:18, 846kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<03:11, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<07:01, 510kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<05:37, 636kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<04:05, 870kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<03:24, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<03:04, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<02:19, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:10, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:12, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:41, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<01:13, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:09, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:12, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:42, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:43, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:27, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<01:32, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:59<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<00:58, 3.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:37, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:38, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:44, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:22, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:26, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:25, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:05, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:19, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:28, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:09, 2.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:26, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:08, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:15, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:24, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<01:06, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:13, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:22, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:03, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<00:47, 3.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:30, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:10, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:14, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:21, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<01:03, 2.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:09, 2.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:17, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:00, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<00:44, 3.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:26, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<01:07, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<00:49, 2.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<01:24, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<01:25, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:06, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<00:47, 2.99MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<03:06, 752kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<02:36, 896kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:54, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:20, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<02:26, 934kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:57, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:27, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:49, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:41, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:13, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:15, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<00:58, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:05, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:51, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:55, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:01, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:47, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<00:34, 3.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<04:15, 454kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<03:18, 581kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<02:23, 799kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<01:56, 956kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:42, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:39<00:54, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:06, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:06, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<00:51, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:45, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:33, 3.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<00:49, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<00:53, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:41, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:44, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:45, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:37, 2.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:28, 3.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:44, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:37, 2.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:39, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<00:34, 2.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:51<00:25, 3.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:40, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:52<00:33, 2.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:24, 3.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:38, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:41, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:32, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:34, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:37, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:29, 2.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:31, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:35, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:27, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:29, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:25, 2.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:27, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:30, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:23, 2.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:17, 3.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:31, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:23, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:17, 3.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:26, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:28, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:22, 2.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:15, 3.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:43, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:39, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:19, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:49, 921kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:42, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:31, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:21, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<01:02, 660kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:51, 807kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:36, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:30, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:28, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:20, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:14, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:21, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:21, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:15, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:15, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:15, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:12, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:11, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:12, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:09, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:06, 3.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:14, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:13, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:09, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:08, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:09, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:06, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:04, 3.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:06, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:06, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:03, 3.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:04, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:04, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:03, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:02, 2.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:02, 1.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 2.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 2.00MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 962/400000 [00:00<00:41, 9612.15it/s]  0%|          | 1869/400000 [00:00<00:42, 9442.76it/s]  1%|          | 2838/400000 [00:00<00:41, 9514.47it/s]  1%|          | 3772/400000 [00:00<00:41, 9459.96it/s]  1%|          | 4531/400000 [00:00<00:44, 8806.98it/s]  1%|▏         | 5321/400000 [00:00<00:46, 8513.21it/s]  2%|▏         | 6098/400000 [00:00<00:47, 8275.58it/s]  2%|▏         | 6883/400000 [00:00<00:48, 8141.10it/s]  2%|▏         | 7677/400000 [00:00<00:48, 8078.05it/s]  2%|▏         | 8451/400000 [00:01<00:49, 7911.79it/s]  2%|▏         | 9219/400000 [00:01<00:49, 7821.69it/s]  2%|▏         | 9985/400000 [00:01<00:50, 7761.76it/s]  3%|▎         | 10761/400000 [00:01<00:50, 7761.12it/s]  3%|▎         | 11530/400000 [00:01<00:50, 7711.70it/s]  3%|▎         | 12296/400000 [00:01<00:50, 7673.10it/s]  3%|▎         | 13060/400000 [00:01<00:51, 7559.64it/s]  3%|▎         | 13814/400000 [00:01<00:51, 7552.02it/s]  4%|▎         | 14568/400000 [00:01<00:51, 7502.66it/s]  4%|▍         | 15328/400000 [00:01<00:51, 7531.19it/s]  4%|▍         | 16098/400000 [00:02<00:50, 7579.27it/s]  4%|▍         | 16856/400000 [00:02<00:51, 7387.08it/s]  4%|▍         | 17596/400000 [00:02<00:53, 7176.82it/s]  5%|▍         | 18333/400000 [00:02<00:52, 7233.19it/s]  5%|▍         | 19058/400000 [00:02<00:52, 7200.83it/s]  5%|▍         | 19840/400000 [00:02<00:51, 7375.42it/s]  5%|▌         | 20629/400000 [00:02<00:50, 7519.74it/s]  5%|▌         | 21383/400000 [00:02<00:51, 7373.37it/s]  6%|▌         | 22123/400000 [00:02<00:51, 7298.39it/s]  6%|▌         | 22855/400000 [00:02<00:52, 7249.17it/s]  6%|▌         | 23609/400000 [00:03<00:51, 7332.40it/s]  6%|▌         | 24344/400000 [00:03<00:51, 7257.74it/s]  6%|▋         | 25071/400000 [00:03<00:52, 7174.99it/s]  6%|▋         | 25830/400000 [00:03<00:51, 7294.63it/s]  7%|▋         | 26566/400000 [00:03<00:51, 7312.97it/s]  7%|▋         | 27313/400000 [00:03<00:50, 7357.77it/s]  7%|▋         | 28076/400000 [00:03<00:50, 7437.05it/s]  7%|▋         | 28836/400000 [00:03<00:49, 7484.33it/s]  7%|▋         | 29614/400000 [00:03<00:48, 7570.02it/s]  8%|▊         | 30392/400000 [00:03<00:48, 7630.83it/s]  8%|▊         | 31185/400000 [00:04<00:47, 7717.28it/s]  8%|▊         | 31958/400000 [00:04<00:47, 7702.79it/s]  8%|▊         | 32729/400000 [00:04<00:47, 7665.91it/s]  8%|▊         | 33496/400000 [00:04<00:48, 7583.27it/s]  9%|▊         | 34255/400000 [00:04<00:48, 7576.07it/s]  9%|▉         | 35013/400000 [00:04<00:49, 7350.79it/s]  9%|▉         | 35750/400000 [00:04<00:50, 7265.79it/s]  9%|▉         | 36478/400000 [00:04<00:50, 7184.90it/s]  9%|▉         | 37233/400000 [00:04<00:49, 7289.78it/s]  9%|▉         | 37982/400000 [00:04<00:49, 7348.35it/s] 10%|▉         | 38741/400000 [00:05<00:48, 7415.46it/s] 10%|▉         | 39494/400000 [00:05<00:48, 7446.78it/s] 10%|█         | 40240/400000 [00:05<00:48, 7443.65it/s] 10%|█         | 40988/400000 [00:05<00:48, 7452.81it/s] 10%|█         | 41743/400000 [00:05<00:47, 7480.92it/s] 11%|█         | 42580/400000 [00:05<00:46, 7725.46it/s] 11%|█         | 43443/400000 [00:05<00:44, 7973.65it/s] 11%|█         | 44251/400000 [00:05<00:44, 8003.06it/s] 11%|█▏        | 45087/400000 [00:05<00:43, 8104.15it/s] 11%|█▏        | 45943/400000 [00:05<00:43, 8233.19it/s] 12%|█▏        | 46790/400000 [00:06<00:42, 8300.75it/s] 12%|█▏        | 47654/400000 [00:06<00:41, 8397.74it/s] 12%|█▏        | 48506/400000 [00:06<00:41, 8432.54it/s] 12%|█▏        | 49453/400000 [00:06<00:40, 8718.35it/s] 13%|█▎        | 50371/400000 [00:06<00:39, 8850.75it/s] 13%|█▎        | 51259/400000 [00:06<00:39, 8801.71it/s] 13%|█▎        | 52191/400000 [00:06<00:38, 8949.39it/s] 13%|█▎        | 53088/400000 [00:06<00:40, 8659.39it/s] 14%|█▎        | 54000/400000 [00:06<00:39, 8792.48it/s] 14%|█▎        | 54963/400000 [00:06<00:38, 9027.66it/s] 14%|█▍        | 55917/400000 [00:07<00:37, 9172.60it/s] 14%|█▍        | 56838/400000 [00:07<00:37, 9068.79it/s] 14%|█▍        | 57748/400000 [00:07<00:37, 9057.44it/s] 15%|█▍        | 58685/400000 [00:07<00:37, 9147.21it/s] 15%|█▍        | 59602/400000 [00:07<00:37, 9122.53it/s] 15%|█▌        | 60523/400000 [00:07<00:37, 9148.15it/s] 15%|█▌        | 61439/400000 [00:07<00:39, 8664.78it/s] 16%|█▌        | 62312/400000 [00:07<00:40, 8356.50it/s] 16%|█▌        | 63208/400000 [00:07<00:39, 8527.75it/s] 16%|█▌        | 64195/400000 [00:08<00:37, 8889.42it/s] 16%|█▋        | 65112/400000 [00:08<00:37, 8968.87it/s] 17%|█▋        | 66015/400000 [00:08<00:38, 8771.02it/s] 17%|█▋        | 66900/400000 [00:08<00:37, 8793.14it/s] 17%|█▋        | 67850/400000 [00:08<00:36, 8992.54it/s] 17%|█▋        | 68794/400000 [00:08<00:36, 9119.07it/s] 17%|█▋        | 69739/400000 [00:08<00:35, 9214.32it/s] 18%|█▊        | 70663/400000 [00:08<00:35, 9185.19it/s] 18%|█▊        | 71584/400000 [00:08<00:36, 9063.09it/s] 18%|█▊        | 72492/400000 [00:08<00:36, 9018.20it/s] 18%|█▊        | 73450/400000 [00:09<00:35, 9178.60it/s] 19%|█▊        | 74370/400000 [00:09<00:35, 9146.91it/s] 19%|█▉        | 75286/400000 [00:09<00:36, 8931.46it/s] 19%|█▉        | 76182/400000 [00:09<00:36, 8862.96it/s] 19%|█▉        | 77088/400000 [00:09<00:36, 8919.43it/s] 20%|█▉        | 78008/400000 [00:09<00:35, 9000.92it/s] 20%|█▉        | 78910/400000 [00:09<00:36, 8850.34it/s] 20%|█▉        | 79797/400000 [00:09<00:37, 8568.26it/s] 20%|██        | 80657/400000 [00:09<00:37, 8555.98it/s] 20%|██        | 81553/400000 [00:09<00:36, 8670.80it/s] 21%|██        | 82422/400000 [00:10<00:36, 8615.52it/s] 21%|██        | 83285/400000 [00:10<00:37, 8500.51it/s] 21%|██        | 84137/400000 [00:10<00:37, 8422.15it/s] 21%|██        | 84981/400000 [00:10<00:38, 8288.89it/s] 21%|██▏       | 85812/400000 [00:10<00:38, 8229.35it/s] 22%|██▏       | 86656/400000 [00:10<00:37, 8289.61it/s] 22%|██▏       | 87608/400000 [00:10<00:36, 8622.00it/s] 22%|██▏       | 88475/400000 [00:10<00:36, 8589.60it/s] 22%|██▏       | 89403/400000 [00:10<00:35, 8784.28it/s] 23%|██▎       | 90324/400000 [00:11<00:34, 8906.99it/s] 23%|██▎       | 91218/400000 [00:11<00:34, 8843.31it/s] 23%|██▎       | 92105/400000 [00:11<00:35, 8766.16it/s] 23%|██▎       | 92984/400000 [00:11<00:35, 8742.37it/s] 23%|██▎       | 93860/400000 [00:11<00:35, 8680.96it/s] 24%|██▎       | 94729/400000 [00:11<00:36, 8442.84it/s] 24%|██▍       | 95660/400000 [00:11<00:35, 8685.37it/s] 24%|██▍       | 96648/400000 [00:11<00:33, 9004.59it/s] 24%|██▍       | 97554/400000 [00:11<00:33, 8990.56it/s] 25%|██▍       | 98457/400000 [00:11<00:33, 8988.78it/s] 25%|██▍       | 99366/400000 [00:12<00:33, 9017.80it/s] 25%|██▌       | 100303/400000 [00:12<00:32, 9116.84it/s] 25%|██▌       | 101217/400000 [00:12<00:34, 8728.62it/s] 26%|██▌       | 102095/400000 [00:12<00:34, 8715.74it/s] 26%|██▌       | 103021/400000 [00:12<00:33, 8868.79it/s] 26%|██▌       | 103911/400000 [00:12<00:33, 8869.23it/s] 26%|██▌       | 104800/400000 [00:12<00:33, 8856.64it/s] 26%|██▋       | 105719/400000 [00:12<00:32, 8953.63it/s] 27%|██▋       | 106662/400000 [00:12<00:32, 9089.52it/s] 27%|██▋       | 107629/400000 [00:12<00:31, 9256.05it/s] 27%|██▋       | 108557/400000 [00:13<00:31, 9239.16it/s] 27%|██▋       | 109483/400000 [00:13<00:31, 9175.26it/s] 28%|██▊       | 110402/400000 [00:13<00:31, 9117.62it/s] 28%|██▊       | 111315/400000 [00:13<00:31, 9112.65it/s] 28%|██▊       | 112253/400000 [00:13<00:31, 9190.01it/s] 28%|██▊       | 113189/400000 [00:13<00:31, 9239.90it/s] 29%|██▊       | 114162/400000 [00:13<00:30, 9379.40it/s] 29%|██▉       | 115101/400000 [00:13<00:30, 9367.33it/s] 29%|██▉       | 116039/400000 [00:13<00:30, 9363.94it/s] 29%|██▉       | 117000/400000 [00:13<00:29, 9435.64it/s] 29%|██▉       | 117981/400000 [00:14<00:29, 9544.08it/s] 30%|██▉       | 118972/400000 [00:14<00:29, 9650.11it/s] 30%|██▉       | 119938/400000 [00:14<00:29, 9535.61it/s] 30%|███       | 120893/400000 [00:14<00:29, 9444.08it/s] 30%|███       | 121847/400000 [00:14<00:29, 9470.10it/s] 31%|███       | 122832/400000 [00:14<00:28, 9578.39it/s] 31%|███       | 123791/400000 [00:14<00:28, 9541.64it/s] 31%|███       | 124746/400000 [00:14<00:29, 9201.35it/s] 31%|███▏      | 125670/400000 [00:14<00:29, 9203.66it/s] 32%|███▏      | 126635/400000 [00:14<00:29, 9331.23it/s] 32%|███▏      | 127612/400000 [00:15<00:28, 9455.90it/s] 32%|███▏      | 128597/400000 [00:15<00:28, 9569.79it/s] 32%|███▏      | 129556/400000 [00:15<00:28, 9413.80it/s] 33%|███▎      | 130504/400000 [00:15<00:28, 9431.06it/s] 33%|███▎      | 131449/400000 [00:15<00:29, 9053.61it/s] 33%|███▎      | 132359/400000 [00:15<00:29, 8970.83it/s] 33%|███▎      | 133260/400000 [00:15<00:29, 8967.07it/s] 34%|███▎      | 134159/400000 [00:15<00:29, 8905.05it/s] 34%|███▍      | 135152/400000 [00:15<00:28, 9187.09it/s] 34%|███▍      | 136152/400000 [00:15<00:28, 9415.24it/s] 34%|███▍      | 137157/400000 [00:16<00:27, 9596.90it/s] 35%|███▍      | 138164/400000 [00:16<00:26, 9733.69it/s] 35%|███▍      | 139141/400000 [00:16<00:27, 9486.54it/s] 35%|███▌      | 140094/400000 [00:16<00:28, 9251.00it/s] 35%|███▌      | 141036/400000 [00:16<00:27, 9299.26it/s] 36%|███▌      | 142006/400000 [00:16<00:27, 9413.94it/s] 36%|███▌      | 143009/400000 [00:16<00:26, 9588.99it/s] 36%|███▌      | 143971/400000 [00:16<00:27, 9424.85it/s] 36%|███▌      | 144916/400000 [00:16<00:27, 9298.58it/s] 36%|███▋      | 145888/400000 [00:17<00:26, 9419.90it/s] 37%|███▋      | 146888/400000 [00:17<00:26, 9586.22it/s] 37%|███▋      | 147875/400000 [00:17<00:26, 9668.97it/s] 37%|███▋      | 148844/400000 [00:17<00:27, 9286.67it/s] 37%|███▋      | 149792/400000 [00:17<00:26, 9342.31it/s] 38%|███▊      | 150775/400000 [00:17<00:26, 9482.52it/s] 38%|███▊      | 151741/400000 [00:17<00:26, 9533.33it/s] 38%|███▊      | 152697/400000 [00:17<00:26, 9434.89it/s] 38%|███▊      | 153643/400000 [00:17<00:27, 8988.67it/s] 39%|███▊      | 154551/400000 [00:17<00:27, 9014.49it/s] 39%|███▉      | 155458/400000 [00:18<00:27, 9030.05it/s] 39%|███▉      | 156384/400000 [00:18<00:26, 9095.59it/s] 39%|███▉      | 157296/400000 [00:18<00:27, 8957.56it/s] 40%|███▉      | 158251/400000 [00:18<00:26, 9125.73it/s] 40%|███▉      | 159250/400000 [00:18<00:25, 9367.78it/s] 40%|████      | 160205/400000 [00:18<00:25, 9420.34it/s] 40%|████      | 161207/400000 [00:18<00:24, 9591.78it/s] 41%|████      | 162169/400000 [00:18<00:24, 9527.62it/s] 41%|████      | 163161/400000 [00:18<00:24, 9641.06it/s] 41%|████      | 164127/400000 [00:18<00:24, 9627.60it/s] 41%|████▏     | 165114/400000 [00:19<00:24, 9697.97it/s] 42%|████▏     | 166121/400000 [00:19<00:23, 9804.29it/s] 42%|████▏     | 167103/400000 [00:19<00:24, 9677.14it/s] 42%|████▏     | 168072/400000 [00:19<00:24, 9635.38it/s] 42%|████▏     | 169048/400000 [00:19<00:23, 9670.06it/s] 43%|████▎     | 170016/400000 [00:19<00:23, 9590.42it/s] 43%|████▎     | 170976/400000 [00:19<00:24, 9339.09it/s] 43%|████▎     | 171912/400000 [00:19<00:27, 8330.83it/s] 43%|████▎     | 172860/400000 [00:19<00:26, 8644.55it/s] 43%|████▎     | 173813/400000 [00:20<00:25, 8891.72it/s] 44%|████▎     | 174749/400000 [00:20<00:24, 9024.89it/s] 44%|████▍     | 175734/400000 [00:20<00:24, 9256.99it/s] 44%|████▍     | 176669/400000 [00:20<00:24, 9176.85it/s] 44%|████▍     | 177653/400000 [00:20<00:23, 9364.05it/s] 45%|████▍     | 178639/400000 [00:20<00:23, 9505.02it/s] 45%|████▍     | 179603/400000 [00:20<00:23, 9545.06it/s] 45%|████▌     | 180613/400000 [00:20<00:22, 9704.71it/s] 45%|████▌     | 181587/400000 [00:20<00:22, 9568.36it/s] 46%|████▌     | 182582/400000 [00:20<00:22, 9678.88it/s] 46%|████▌     | 183571/400000 [00:21<00:22, 9739.65it/s] 46%|████▌     | 184574/400000 [00:21<00:21, 9822.56it/s] 46%|████▋     | 185576/400000 [00:21<00:21, 9880.74it/s] 47%|████▋     | 186565/400000 [00:21<00:22, 9564.86it/s] 47%|████▋     | 187526/400000 [00:21<00:22, 9576.63it/s] 47%|████▋     | 188486/400000 [00:21<00:22, 9531.98it/s] 47%|████▋     | 189441/400000 [00:21<00:22, 9533.73it/s] 48%|████▊     | 190407/400000 [00:21<00:21, 9568.98it/s] 48%|████▊     | 191365/400000 [00:21<00:22, 9345.45it/s] 48%|████▊     | 192353/400000 [00:21<00:21, 9499.29it/s] 48%|████▊     | 193342/400000 [00:22<00:21, 9611.15it/s] 49%|████▊     | 194339/400000 [00:22<00:21, 9715.99it/s] 49%|████▉     | 195312/400000 [00:22<00:21, 9664.61it/s] 49%|████▉     | 196280/400000 [00:22<00:21, 9280.44it/s] 49%|████▉     | 197215/400000 [00:22<00:21, 9300.96it/s] 50%|████▉     | 198181/400000 [00:22<00:21, 9403.16it/s] 50%|████▉     | 199166/400000 [00:22<00:21, 9531.64it/s] 50%|█████     | 200160/400000 [00:22<00:20, 9647.80it/s] 50%|█████     | 201127/400000 [00:22<00:20, 9490.19it/s] 51%|█████     | 202078/400000 [00:22<00:21, 9391.13it/s] 51%|█████     | 203036/400000 [00:23<00:20, 9447.00it/s] 51%|█████     | 204042/400000 [00:23<00:20, 9621.65it/s] 51%|█████▏    | 205008/400000 [00:23<00:20, 9632.00it/s] 51%|█████▏    | 205973/400000 [00:23<00:21, 9194.71it/s] 52%|█████▏    | 206955/400000 [00:23<00:20, 9372.87it/s] 52%|█████▏    | 207925/400000 [00:23<00:20, 9467.61it/s] 52%|█████▏    | 208942/400000 [00:23<00:19, 9666.47it/s] 52%|█████▏    | 209935/400000 [00:23<00:19, 9741.79it/s] 53%|█████▎    | 210912/400000 [00:23<00:19, 9609.14it/s] 53%|█████▎    | 211907/400000 [00:24<00:19, 9706.64it/s] 53%|█████▎    | 212895/400000 [00:24<00:19, 9756.20it/s] 53%|█████▎    | 213883/400000 [00:24<00:19, 9791.51it/s] 54%|█████▎    | 214882/400000 [00:24<00:18, 9849.53it/s] 54%|█████▍    | 215868/400000 [00:24<00:19, 9590.10it/s] 54%|█████▍    | 216830/400000 [00:24<00:19, 9533.89it/s] 54%|█████▍    | 217785/400000 [00:24<00:19, 9279.30it/s] 55%|█████▍    | 218738/400000 [00:24<00:19, 9352.56it/s] 55%|█████▍    | 219735/400000 [00:24<00:18, 9529.57it/s] 55%|█████▌    | 220691/400000 [00:24<00:20, 8873.68it/s] 55%|█████▌    | 221643/400000 [00:25<00:19, 9056.65it/s] 56%|█████▌    | 222634/400000 [00:25<00:19, 9296.37it/s] 56%|█████▌    | 223637/400000 [00:25<00:18, 9504.39it/s] 56%|█████▌    | 224594/400000 [00:25<00:18, 9468.98it/s] 56%|█████▋    | 225546/400000 [00:25<00:18, 9259.14it/s] 57%|█████▋    | 226540/400000 [00:25<00:18, 9452.11it/s] 57%|█████▋    | 227531/400000 [00:25<00:17, 9584.29it/s] 57%|█████▋    | 228521/400000 [00:25<00:17, 9674.81it/s] 57%|█████▋    | 229499/400000 [00:25<00:17, 9704.58it/s] 58%|█████▊    | 230472/400000 [00:25<00:17, 9421.47it/s] 58%|█████▊    | 231418/400000 [00:26<00:18, 9210.69it/s] 58%|█████▊    | 232343/400000 [00:26<00:18, 9045.78it/s] 58%|█████▊    | 233251/400000 [00:26<00:18, 8981.91it/s] 59%|█████▊    | 234152/400000 [00:26<00:18, 8975.95it/s] 59%|█████▉    | 235086/400000 [00:26<00:18, 9080.45it/s] 59%|█████▉    | 235996/400000 [00:26<00:18, 8932.40it/s] 59%|█████▉    | 236941/400000 [00:26<00:17, 9079.89it/s] 59%|█████▉    | 237884/400000 [00:26<00:17, 9180.08it/s] 60%|█████▉    | 238823/400000 [00:26<00:17, 9238.31it/s] 60%|█████▉    | 239748/400000 [00:27<00:17, 8911.31it/s] 60%|██████    | 240650/400000 [00:27<00:17, 8942.72it/s] 60%|██████    | 241572/400000 [00:27<00:17, 9023.02it/s] 61%|██████    | 242532/400000 [00:27<00:17, 9187.51it/s] 61%|██████    | 243453/400000 [00:27<00:17, 9103.24it/s] 61%|██████    | 244393/400000 [00:27<00:16, 9187.82it/s] 61%|██████▏   | 245314/400000 [00:27<00:16, 9158.89it/s] 62%|██████▏   | 246231/400000 [00:27<00:16, 9074.89it/s] 62%|██████▏   | 247174/400000 [00:27<00:16, 9176.05it/s] 62%|██████▏   | 248093/400000 [00:27<00:16, 9065.36it/s] 62%|██████▏   | 249001/400000 [00:28<00:17, 8715.68it/s] 62%|██████▏   | 249949/400000 [00:28<00:16, 8929.71it/s] 63%|██████▎   | 250881/400000 [00:28<00:16, 9042.87it/s] 63%|██████▎   | 251837/400000 [00:28<00:16, 9189.98it/s] 63%|██████▎   | 252759/400000 [00:28<00:16, 8746.49it/s] 63%|██████▎   | 253641/400000 [00:28<00:16, 8628.26it/s] 64%|██████▎   | 254562/400000 [00:28<00:16, 8793.37it/s] 64%|██████▍   | 255560/400000 [00:28<00:15, 9117.28it/s] 64%|██████▍   | 256542/400000 [00:28<00:15, 9314.73it/s] 64%|██████▍   | 257479/400000 [00:28<00:15, 9238.40it/s] 65%|██████▍   | 258407/400000 [00:29<00:17, 7882.03it/s] 65%|██████▍   | 259316/400000 [00:29<00:17, 8207.67it/s] 65%|██████▌   | 260293/400000 [00:29<00:16, 8620.25it/s] 65%|██████▌   | 261259/400000 [00:29<00:15, 8907.77it/s] 66%|██████▌   | 262171/400000 [00:29<00:16, 8546.44it/s] 66%|██████▌   | 263044/400000 [00:29<00:16, 8473.71it/s] 66%|██████▌   | 263966/400000 [00:29<00:15, 8684.46it/s] 66%|██████▌   | 264955/400000 [00:29<00:14, 9013.72it/s] 66%|██████▋   | 265956/400000 [00:29<00:14, 9288.24it/s] 67%|██████▋   | 266895/400000 [00:30<00:14, 9182.27it/s] 67%|██████▋   | 267850/400000 [00:30<00:14, 9287.33it/s] 67%|██████▋   | 268855/400000 [00:30<00:13, 9501.71it/s] 67%|██████▋   | 269858/400000 [00:30<00:13, 9652.94it/s] 68%|██████▊   | 270837/400000 [00:30<00:13, 9691.73it/s] 68%|██████▊   | 271809/400000 [00:30<00:13, 9378.82it/s] 68%|██████▊   | 272752/400000 [00:30<00:13, 9295.90it/s] 68%|██████▊   | 273748/400000 [00:30<00:13, 9483.04it/s] 69%|██████▊   | 274731/400000 [00:30<00:13, 9581.81it/s] 69%|██████▉   | 275720/400000 [00:30<00:12, 9671.80it/s] 69%|██████▉   | 276690/400000 [00:31<00:13, 9436.95it/s] 69%|██████▉   | 277637/400000 [00:31<00:12, 9420.78it/s] 70%|██████▉   | 278581/400000 [00:31<00:13, 9293.34it/s] 70%|██████▉   | 279561/400000 [00:31<00:12, 9438.64it/s] 70%|███████   | 280537/400000 [00:31<00:12, 9532.53it/s] 70%|███████   | 281492/400000 [00:31<00:12, 9459.92it/s] 71%|███████   | 282474/400000 [00:31<00:12, 9562.70it/s] 71%|███████   | 283481/400000 [00:31<00:12, 9708.21it/s] 71%|███████   | 284457/400000 [00:31<00:11, 9723.41it/s] 71%|███████▏  | 285454/400000 [00:31<00:11, 9793.95it/s] 72%|███████▏  | 286435/400000 [00:32<00:12, 9416.09it/s] 72%|███████▏  | 287381/400000 [00:32<00:12, 9372.46it/s] 72%|███████▏  | 288321/400000 [00:32<00:12, 9271.15it/s] 72%|███████▏  | 289251/400000 [00:32<00:12, 8981.61it/s] 73%|███████▎  | 290158/400000 [00:32<00:12, 9007.36it/s] 73%|███████▎  | 291124/400000 [00:32<00:11, 9192.85it/s] 73%|███████▎  | 292096/400000 [00:32<00:11, 9342.98it/s] 73%|███████▎  | 293061/400000 [00:32<00:11, 9431.20it/s] 74%|███████▎  | 294007/400000 [00:32<00:11, 9432.52it/s] 74%|███████▎  | 294992/400000 [00:33<00:10, 9551.49it/s] 74%|███████▍  | 295949/400000 [00:33<00:10, 9483.17it/s] 74%|███████▍  | 296917/400000 [00:33<00:10, 9539.05it/s] 74%|███████▍  | 297917/400000 [00:33<00:10, 9671.32it/s] 75%|███████▍  | 298886/400000 [00:33<00:10, 9232.66it/s] 75%|███████▍  | 299826/400000 [00:33<00:10, 9281.49it/s] 75%|███████▌  | 300758/400000 [00:33<00:10, 9121.72it/s] 75%|███████▌  | 301702/400000 [00:33<00:10, 9211.60it/s] 76%|███████▌  | 302706/400000 [00:33<00:10, 9443.50it/s] 76%|███████▌  | 303698/400000 [00:33<00:10, 9580.52it/s] 76%|███████▌  | 304679/400000 [00:34<00:09, 9647.79it/s] 76%|███████▋  | 305646/400000 [00:34<00:09, 9519.59it/s] 77%|███████▋  | 306600/400000 [00:34<00:10, 9262.80it/s] 77%|███████▋  | 307542/400000 [00:34<00:09, 9308.26it/s] 77%|███████▋  | 308475/400000 [00:34<00:09, 9299.74it/s] 77%|███████▋  | 309407/400000 [00:34<00:10, 8967.98it/s] 78%|███████▊  | 310308/400000 [00:34<00:10, 8796.25it/s] 78%|███████▊  | 311284/400000 [00:34<00:09, 9064.21it/s] 78%|███████▊  | 312253/400000 [00:34<00:09, 9240.59it/s] 78%|███████▊  | 313233/400000 [00:34<00:09, 9401.33it/s] 79%|███████▊  | 314193/400000 [00:35<00:09, 9459.21it/s] 79%|███████▉  | 315142/400000 [00:35<00:09, 9203.77it/s] 79%|███████▉  | 316134/400000 [00:35<00:08, 9407.45it/s] 79%|███████▉  | 317130/400000 [00:35<00:08, 9564.36it/s] 80%|███████▉  | 318090/400000 [00:35<00:08, 9542.17it/s] 80%|███████▉  | 319095/400000 [00:35<00:08, 9689.01it/s] 80%|████████  | 320066/400000 [00:35<00:08, 9571.42it/s] 80%|████████  | 321066/400000 [00:35<00:08, 9694.46it/s] 81%|████████  | 322067/400000 [00:35<00:07, 9784.35it/s] 81%|████████  | 323047/400000 [00:35<00:07, 9625.32it/s] 81%|████████  | 324050/400000 [00:36<00:07, 9740.52it/s] 81%|████████▏ | 325026/400000 [00:36<00:08, 9185.88it/s] 81%|████████▏ | 325953/400000 [00:36<00:08, 9013.61it/s] 82%|████████▏ | 326927/400000 [00:36<00:07, 9219.76it/s] 82%|████████▏ | 327908/400000 [00:36<00:07, 9388.65it/s] 82%|████████▏ | 328854/400000 [00:36<00:07, 9409.59it/s] 82%|████████▏ | 329825/400000 [00:36<00:07, 9495.89it/s] 83%|████████▎ | 330821/400000 [00:36<00:07, 9630.22it/s] 83%|████████▎ | 331787/400000 [00:36<00:07, 9400.29it/s] 83%|████████▎ | 332776/400000 [00:37<00:07, 9539.81it/s] 83%|████████▎ | 333733/400000 [00:37<00:07, 9323.03it/s] 84%|████████▎ | 334695/400000 [00:37<00:06, 9409.87it/s] 84%|████████▍ | 335701/400000 [00:37<00:06, 9593.83it/s] 84%|████████▍ | 336663/400000 [00:37<00:06, 9517.33it/s] 84%|████████▍ | 337617/400000 [00:37<00:06, 9438.68it/s] 85%|████████▍ | 338563/400000 [00:37<00:06, 9189.91it/s] 85%|████████▍ | 339518/400000 [00:37<00:06, 9294.34it/s] 85%|████████▌ | 340450/400000 [00:37<00:06, 9255.76it/s] 85%|████████▌ | 341385/400000 [00:37<00:06, 9281.65it/s] 86%|████████▌ | 342319/400000 [00:38<00:06, 9296.14it/s] 86%|████████▌ | 343250/400000 [00:38<00:06, 8858.73it/s] 86%|████████▌ | 344146/400000 [00:38<00:06, 8886.77it/s] 86%|████████▋ | 345061/400000 [00:38<00:06, 8962.96it/s] 86%|████████▋ | 345960/400000 [00:38<00:06, 8909.13it/s] 87%|████████▋ | 346941/400000 [00:38<00:05, 9159.82it/s] 87%|████████▋ | 347861/400000 [00:38<00:05, 8936.95it/s] 87%|████████▋ | 348759/400000 [00:38<00:05, 8851.33it/s] 87%|████████▋ | 349706/400000 [00:38<00:05, 9026.32it/s] 88%|████████▊ | 350650/400000 [00:38<00:05, 9145.98it/s] 88%|████████▊ | 351567/400000 [00:39<00:05, 8605.93it/s] 88%|████████▊ | 352456/400000 [00:39<00:05, 8687.78it/s] 88%|████████▊ | 353438/400000 [00:39<00:05, 8997.07it/s] 89%|████████▊ | 354419/400000 [00:39<00:04, 9223.89it/s] 89%|████████▉ | 355348/400000 [00:39<00:04, 9097.44it/s] 89%|████████▉ | 356286/400000 [00:39<00:04, 9179.75it/s] 89%|████████▉ | 357208/400000 [00:39<00:04, 9166.47it/s] 90%|████████▉ | 358193/400000 [00:39<00:04, 9359.21it/s] 90%|████████▉ | 359166/400000 [00:39<00:04, 9465.34it/s] 90%|█████████ | 360174/400000 [00:40<00:04, 9639.32it/s] 90%|█████████ | 361154/400000 [00:40<00:04, 9686.85it/s] 91%|█████████ | 362125/400000 [00:40<00:03, 9481.94it/s] 91%|█████████ | 363130/400000 [00:40<00:03, 9643.22it/s] 91%|█████████ | 364140/400000 [00:40<00:03, 9775.53it/s] 91%|█████████▏| 365120/400000 [00:40<00:03, 9728.54it/s] 92%|█████████▏| 366113/400000 [00:40<00:03, 9785.48it/s] 92%|█████████▏| 367093/400000 [00:40<00:03, 9600.75it/s] 92%|█████████▏| 368078/400000 [00:40<00:03, 9672.22it/s] 92%|█████████▏| 369078/400000 [00:40<00:03, 9768.10it/s] 93%|█████████▎| 370056/400000 [00:41<00:03, 9533.16it/s] 93%|█████████▎| 371012/400000 [00:41<00:03, 9034.81it/s] 93%|█████████▎| 371924/400000 [00:41<00:03, 9059.96it/s] 93%|█████████▎| 372925/400000 [00:41<00:02, 9323.84it/s] 93%|█████████▎| 373915/400000 [00:41<00:02, 9488.45it/s] 94%|█████████▎| 374899/400000 [00:41<00:02, 9589.86it/s] 94%|█████████▍| 375865/400000 [00:41<00:02, 9605.91it/s] 94%|█████████▍| 376828/400000 [00:41<00:02, 9609.66it/s] 94%|█████████▍| 377831/400000 [00:41<00:02, 9730.18it/s] 95%|█████████▍| 378816/400000 [00:41<00:02, 9764.44it/s] 95%|█████████▍| 379794/400000 [00:42<00:02, 9756.01it/s] 95%|█████████▌| 380771/400000 [00:42<00:01, 9639.00it/s] 95%|█████████▌| 381736/400000 [00:42<00:01, 9497.77it/s] 96%|█████████▌| 382701/400000 [00:42<00:01, 9542.14it/s] 96%|█████████▌| 383657/400000 [00:42<00:01, 9463.52it/s] 96%|█████████▌| 384605/400000 [00:42<00:01, 9249.14it/s] 96%|█████████▋| 385532/400000 [00:42<00:01, 9148.86it/s] 97%|█████████▋| 386449/400000 [00:42<00:01, 9080.96it/s] 97%|█████████▋| 387380/400000 [00:42<00:01, 9147.96it/s] 97%|█████████▋| 388362/400000 [00:42<00:01, 9338.62it/s] 97%|█████████▋| 389355/400000 [00:43<00:01, 9507.66it/s] 98%|█████████▊| 390325/400000 [00:43<00:01, 9560.88it/s] 98%|█████████▊| 391283/400000 [00:43<00:00, 9540.94it/s] 98%|█████████▊| 392239/400000 [00:43<00:00, 9147.33it/s] 98%|█████████▊| 393158/400000 [00:43<00:00, 9098.02it/s] 99%|█████████▊| 394119/400000 [00:43<00:00, 9243.35it/s] 99%|█████████▉| 395046/400000 [00:43<00:00, 9203.44it/s] 99%|█████████▉| 395969/400000 [00:43<00:00, 9160.14it/s] 99%|█████████▉| 396930/400000 [00:43<00:00, 9288.46it/s] 99%|█████████▉| 397861/400000 [00:44<00:00, 9272.22it/s]100%|█████████▉| 398840/400000 [00:44<00:00, 9421.32it/s]100%|█████████▉| 399831/400000 [00:44<00:00, 9560.26it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9043.76it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0e1f6c7b38> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011588598951931846 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011224798135534179 	 Accuracy: 57

  model saves at 57% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16099 out of table with 15975 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16099 out of table with 15975 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

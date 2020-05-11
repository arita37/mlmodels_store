
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5af77c8fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 16:15:36.865470
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 16:15:36.869006
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 16:15:36.872167
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 16:15:36.875449
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5b0358c470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 354158.6562
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 248941.1094
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 147459.8594
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 76164.6250
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 38762.9414
Epoch 6/10

1/1 [==============================] - 0s 108ms/step - loss: 21249.4453
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 12859.1631
Epoch 8/10

1/1 [==============================] - 0s 103ms/step - loss: 8587.7002
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 6209.1436
Epoch 10/10

1/1 [==============================] - 0s 113ms/step - loss: 4773.4912

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.7517322  10.491846    9.613557    9.314539   10.343801   10.7436905
  11.343842   11.82059    10.137157   11.160946   11.508552   10.544595
   8.394585   11.0882225  10.224395   10.785498   10.466436   10.101893
  11.072774   10.812623   10.961293   10.014618   10.896737    9.262092
   9.230131   10.657762   10.315666    8.348698   11.015863    9.633652
  11.444411    8.670194   10.659404    9.236283   10.494119    9.6586075
  10.779824    9.574228    9.605344   11.796547    9.221077    8.877507
  11.346148   10.344006    9.740131    9.595075    9.833929    8.59294
  10.005647    9.40727     8.53539    10.816499   11.435455   10.09884
  10.233575   10.353838    8.863385    9.141373    8.077329    8.190893
   0.77832097  0.90794754  0.80583674 -2.0068069  -0.098334   -0.14273643
  -1.2921491  -0.4505368  -0.82323635 -0.38688394 -0.37975362  0.15010345
   0.65932703 -0.35588074  0.21249658 -1.1517843   0.16777882 -2.42979
  -1.685175   -1.9317675  -1.1665349   0.3036862  -0.6708864  -0.5530205
   1.533472    0.75617725  1.530957   -0.12096336 -0.2519884  -0.51725435
  -0.11926901 -1.528719   -0.08193433 -0.37848198  1.1404531   0.54912746
  -1.7149224   1.8386028  -0.9368973  -0.2615304   0.5904565  -0.92915887
   0.6246431  -0.662899    0.7055676   0.6273584  -0.09827834 -1.995327
  -1.0595958  -0.9592394  -2.0181937  -0.12414142 -0.43614432 -1.4713817
   0.9604043   1.8708143   2.2308462   0.6288438   0.85857284  1.7712679
   1.4502484  -0.21741974 -0.8703269  -1.1491466  -0.9398985   0.70830894
   2.0756154  -0.35768506 -0.564733    0.2444225   0.8767719   1.7789414
  -0.40638113 -0.35778713  0.34121287  0.82159185 -0.73137826  1.370854
  -0.22560664 -1.2547487  -0.20143867  0.64970237  1.1158297   0.43881392
  -0.61099154  0.64278245  1.811025    0.13266048 -0.2949393   1.0745418
  -0.1812616  -0.73390085  1.4861535  -0.9690658  -1.826641   -0.9568764
  -1.1508836   1.3124975   0.5730938  -0.4451623  -1.1852981  -0.4656142
   0.50976056 -0.9385038  -0.7166196   0.39893553 -0.4350932  -0.09036076
  -1.356291   -0.13047487  1.743011    0.01884116  0.8235829  -0.3285948
   0.80261576  1.9922965   0.31599885 -1.3087927   0.0568015   1.2034982
   0.5278659  10.958176   10.5933485   9.852974    9.840244    8.647021
  10.87094    10.239062    9.0780735   9.119074    8.138813   10.58058
   9.976943    9.688412   10.827151    9.903912    8.76016    11.354215
  10.540386   11.234712   11.8800955  11.371497   10.527268   11.882917
  10.921414    8.5976      9.7581415  10.383828   11.033109   10.383906
   8.522049    8.709363    8.904979   11.069109   11.030304    8.847338
   9.638878   11.301963    9.599169    9.505815   10.668526    8.096378
  11.353751    7.628516    9.187219    8.05959     9.602299   11.708844
  10.324908    9.83881    11.560963    8.086071    9.396312    9.889965
   8.485147   10.047379   11.108085    9.490755   11.511494    9.369909
   1.4284837   1.3081646   0.7258849   0.4770738   0.4732898   0.7689525
   0.19693398  1.1549569   0.81784844  0.09012991  0.18952334  0.889982
   0.12607902  1.1257476   0.31335324  1.462579    2.6814742   0.18163955
   3.9400105   0.50330806  1.544142    0.7874757   0.1356293   1.5849
   2.0330234   2.8909993   0.9299106   0.2784952   0.43113506  2.2798095
   1.6090072   2.0539205   0.11888063  0.4422438   2.2002835   1.688027
   0.92853266  0.48610198  1.8365442   1.2125857   1.6813039   0.59157
   0.10036337  1.0902615   0.5683942   0.70653194  0.8223295   1.8548287
   2.609404    0.18291551  0.30707014  0.52609813  2.1635137   1.2664886
   0.35680246  0.34427178  0.79571146  0.89936113  0.48797393  1.9255133
   1.3661063   0.23377037  0.41829342  1.2516634   0.845678    0.36980486
   1.317197    0.1656707   0.6557821   2.4161441   0.59207433  0.61037517
   0.6600147   0.268566    3.0396109   3.066729    0.23216283  0.38738716
   0.48757935  0.99661183  0.5906055   0.6004963   0.7166921   0.35160214
   0.8486069   0.30488688  0.31613111  0.5911957   0.24545848  0.34474838
   1.7141607   2.5855246   0.48089665  0.47771943  0.90042555  2.2146091
   1.0978862   2.0040827   0.37603796  2.7916842   0.49531937  1.4190767
   0.5358215   1.0084176   1.2115273   0.13957226  0.2363894   0.8470168
   1.9403584   0.27212036  1.9421654   0.16250682  0.7856261   2.1810575
   1.465738    1.5631802   0.25619435  1.1104038   1.8180649   0.40346217
  11.307006   -5.5269685  -5.5882382 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 16:15:45.981987
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.9679
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 16:15:45.985816
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8482.45
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 16:15:45.989854
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.3369
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 16:15:45.993423
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -758.663
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140028333964088
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140027106815504
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140027106902088
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140027106902592
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140027106903096
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140027106903600

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5aff40fef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.575324
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.542068
grad_step = 000002, loss = 0.516114
grad_step = 000003, loss = 0.488432
grad_step = 000004, loss = 0.460295
grad_step = 000005, loss = 0.438975
grad_step = 000006, loss = 0.424392
grad_step = 000007, loss = 0.406661
grad_step = 000008, loss = 0.389169
grad_step = 000009, loss = 0.369952
grad_step = 000010, loss = 0.353440
grad_step = 000011, loss = 0.338423
grad_step = 000012, loss = 0.326526
grad_step = 000013, loss = 0.315938
grad_step = 000014, loss = 0.303982
grad_step = 000015, loss = 0.290493
grad_step = 000016, loss = 0.277302
grad_step = 000017, loss = 0.265155
grad_step = 000018, loss = 0.253638
grad_step = 000019, loss = 0.242004
grad_step = 000020, loss = 0.230098
grad_step = 000021, loss = 0.218810
grad_step = 000022, loss = 0.208088
grad_step = 000023, loss = 0.197494
grad_step = 000024, loss = 0.187410
grad_step = 000025, loss = 0.177738
grad_step = 000026, loss = 0.167810
grad_step = 000027, loss = 0.157747
grad_step = 000028, loss = 0.148266
grad_step = 000029, loss = 0.139572
grad_step = 000030, loss = 0.131417
grad_step = 000031, loss = 0.123297
grad_step = 000032, loss = 0.115455
grad_step = 000033, loss = 0.108159
grad_step = 000034, loss = 0.101088
grad_step = 000035, loss = 0.094338
grad_step = 000036, loss = 0.087973
grad_step = 000037, loss = 0.082016
grad_step = 000038, loss = 0.076369
grad_step = 000039, loss = 0.071018
grad_step = 000040, loss = 0.066014
grad_step = 000041, loss = 0.061181
grad_step = 000042, loss = 0.056679
grad_step = 000043, loss = 0.052577
grad_step = 000044, loss = 0.048775
grad_step = 000045, loss = 0.045097
grad_step = 000046, loss = 0.041678
grad_step = 000047, loss = 0.038543
grad_step = 000048, loss = 0.035595
grad_step = 000049, loss = 0.032761
grad_step = 000050, loss = 0.030136
grad_step = 000051, loss = 0.027720
grad_step = 000052, loss = 0.025514
grad_step = 000053, loss = 0.023457
grad_step = 000054, loss = 0.021495
grad_step = 000055, loss = 0.019628
grad_step = 000056, loss = 0.017942
grad_step = 000057, loss = 0.016398
grad_step = 000058, loss = 0.014940
grad_step = 000059, loss = 0.013585
grad_step = 000060, loss = 0.012355
grad_step = 000061, loss = 0.011227
grad_step = 000062, loss = 0.010181
grad_step = 000063, loss = 0.009233
grad_step = 000064, loss = 0.008369
grad_step = 000065, loss = 0.007585
grad_step = 000066, loss = 0.006881
grad_step = 000067, loss = 0.006243
grad_step = 000068, loss = 0.005674
grad_step = 000069, loss = 0.005169
grad_step = 000070, loss = 0.004722
grad_step = 000071, loss = 0.004333
grad_step = 000072, loss = 0.003993
grad_step = 000073, loss = 0.003690
grad_step = 000074, loss = 0.003426
grad_step = 000075, loss = 0.003207
grad_step = 000076, loss = 0.003020
grad_step = 000077, loss = 0.002858
grad_step = 000078, loss = 0.002720
grad_step = 000079, loss = 0.002607
grad_step = 000080, loss = 0.002512
grad_step = 000081, loss = 0.002433
grad_step = 000082, loss = 0.002368
grad_step = 000083, loss = 0.002316
grad_step = 000084, loss = 0.002273
grad_step = 000085, loss = 0.002237
grad_step = 000086, loss = 0.002209
grad_step = 000087, loss = 0.002187
grad_step = 000088, loss = 0.002170
grad_step = 000089, loss = 0.002156
grad_step = 000090, loss = 0.002144
grad_step = 000091, loss = 0.002134
grad_step = 000092, loss = 0.002127
grad_step = 000093, loss = 0.002122
grad_step = 000094, loss = 0.002116
grad_step = 000095, loss = 0.002111
grad_step = 000096, loss = 0.002107
grad_step = 000097, loss = 0.002103
grad_step = 000098, loss = 0.002099
grad_step = 000099, loss = 0.002094
grad_step = 000100, loss = 0.002090
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002085
grad_step = 000102, loss = 0.002079
grad_step = 000103, loss = 0.002074
grad_step = 000104, loss = 0.002068
grad_step = 000105, loss = 0.002062
grad_step = 000106, loss = 0.002055
grad_step = 000107, loss = 0.002048
grad_step = 000108, loss = 0.002041
grad_step = 000109, loss = 0.002034
grad_step = 000110, loss = 0.002027
grad_step = 000111, loss = 0.002020
grad_step = 000112, loss = 0.002014
grad_step = 000113, loss = 0.002013
grad_step = 000114, loss = 0.002019
grad_step = 000115, loss = 0.002032
grad_step = 000116, loss = 0.002035
grad_step = 000117, loss = 0.002006
grad_step = 000118, loss = 0.001985
grad_step = 000119, loss = 0.002000
grad_step = 000120, loss = 0.002024
grad_step = 000121, loss = 0.002024
grad_step = 000122, loss = 0.001980
grad_step = 000123, loss = 0.001954
grad_step = 000124, loss = 0.001963
grad_step = 000125, loss = 0.001969
grad_step = 000126, loss = 0.001947
grad_step = 000127, loss = 0.001918
grad_step = 000128, loss = 0.001918
grad_step = 000129, loss = 0.001938
grad_step = 000130, loss = 0.001942
grad_step = 000131, loss = 0.001927
grad_step = 000132, loss = 0.001902
grad_step = 000133, loss = 0.001896
grad_step = 000134, loss = 0.001903
grad_step = 000135, loss = 0.001905
grad_step = 000136, loss = 0.001898
grad_step = 000137, loss = 0.001883
grad_step = 000138, loss = 0.001875
grad_step = 000139, loss = 0.001878
grad_step = 000140, loss = 0.001896
grad_step = 000141, loss = 0.001927
grad_step = 000142, loss = 0.001969
grad_step = 000143, loss = 0.002037
grad_step = 000144, loss = 0.002152
grad_step = 000145, loss = 0.002105
grad_step = 000146, loss = 0.001964
grad_step = 000147, loss = 0.001832
grad_step = 000148, loss = 0.001929
grad_step = 000149, loss = 0.002019
grad_step = 000150, loss = 0.001884
grad_step = 000151, loss = 0.001863
grad_step = 000152, loss = 0.001942
grad_step = 000153, loss = 0.001872
grad_step = 000154, loss = 0.001820
grad_step = 000155, loss = 0.001880
grad_step = 000156, loss = 0.001889
grad_step = 000157, loss = 0.001830
grad_step = 000158, loss = 0.001812
grad_step = 000159, loss = 0.001847
grad_step = 000160, loss = 0.001853
grad_step = 000161, loss = 0.001813
grad_step = 000162, loss = 0.001810
grad_step = 000163, loss = 0.001835
grad_step = 000164, loss = 0.001827
grad_step = 000165, loss = 0.001798
grad_step = 000166, loss = 0.001787
grad_step = 000167, loss = 0.001805
grad_step = 000168, loss = 0.001819
grad_step = 000169, loss = 0.001804
grad_step = 000170, loss = 0.001784
grad_step = 000171, loss = 0.001777
grad_step = 000172, loss = 0.001785
grad_step = 000173, loss = 0.001795
grad_step = 000174, loss = 0.001794
grad_step = 000175, loss = 0.001786
grad_step = 000176, loss = 0.001773
grad_step = 000177, loss = 0.001764
grad_step = 000178, loss = 0.001761
grad_step = 000179, loss = 0.001763
grad_step = 000180, loss = 0.001769
grad_step = 000181, loss = 0.001776
grad_step = 000182, loss = 0.001788
grad_step = 000183, loss = 0.001804
grad_step = 000184, loss = 0.001839
grad_step = 000185, loss = 0.001883
grad_step = 000186, loss = 0.001958
grad_step = 000187, loss = 0.002005
grad_step = 000188, loss = 0.002084
grad_step = 000189, loss = 0.001992
grad_step = 000190, loss = 0.001874
grad_step = 000191, loss = 0.001792
grad_step = 000192, loss = 0.001873
grad_step = 000193, loss = 0.001937
grad_step = 000194, loss = 0.001801
grad_step = 000195, loss = 0.001755
grad_step = 000196, loss = 0.001853
grad_step = 000197, loss = 0.001835
grad_step = 000198, loss = 0.001747
grad_step = 000199, loss = 0.001755
grad_step = 000200, loss = 0.001796
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001768
grad_step = 000202, loss = 0.001742
grad_step = 000203, loss = 0.001769
grad_step = 000204, loss = 0.001760
grad_step = 000205, loss = 0.001724
grad_step = 000206, loss = 0.001745
grad_step = 000207, loss = 0.001765
grad_step = 000208, loss = 0.001735
grad_step = 000209, loss = 0.001729
grad_step = 000210, loss = 0.001748
grad_step = 000211, loss = 0.001733
grad_step = 000212, loss = 0.001714
grad_step = 000213, loss = 0.001723
grad_step = 000214, loss = 0.001726
grad_step = 000215, loss = 0.001711
grad_step = 000216, loss = 0.001709
grad_step = 000217, loss = 0.001719
grad_step = 000218, loss = 0.001722
grad_step = 000219, loss = 0.001728
grad_step = 000220, loss = 0.001771
grad_step = 000221, loss = 0.001848
grad_step = 000222, loss = 0.001992
grad_step = 000223, loss = 0.001967
grad_step = 000224, loss = 0.001881
grad_step = 000225, loss = 0.001711
grad_step = 000226, loss = 0.001797
grad_step = 000227, loss = 0.001873
grad_step = 000228, loss = 0.001729
grad_step = 000229, loss = 0.001760
grad_step = 000230, loss = 0.001822
grad_step = 000231, loss = 0.001723
grad_step = 000232, loss = 0.001742
grad_step = 000233, loss = 0.001789
grad_step = 000234, loss = 0.001718
grad_step = 000235, loss = 0.001716
grad_step = 000236, loss = 0.001758
grad_step = 000237, loss = 0.001721
grad_step = 000238, loss = 0.001690
grad_step = 000239, loss = 0.001726
grad_step = 000240, loss = 0.001722
grad_step = 000241, loss = 0.001680
grad_step = 000242, loss = 0.001698
grad_step = 000243, loss = 0.001710
grad_step = 000244, loss = 0.001688
grad_step = 000245, loss = 0.001681
grad_step = 000246, loss = 0.001693
grad_step = 000247, loss = 0.001688
grad_step = 000248, loss = 0.001671
grad_step = 000249, loss = 0.001681
grad_step = 000250, loss = 0.001683
grad_step = 000251, loss = 0.001668
grad_step = 000252, loss = 0.001666
grad_step = 000253, loss = 0.001671
grad_step = 000254, loss = 0.001673
grad_step = 000255, loss = 0.001662
grad_step = 000256, loss = 0.001655
grad_step = 000257, loss = 0.001662
grad_step = 000258, loss = 0.001661
grad_step = 000259, loss = 0.001655
grad_step = 000260, loss = 0.001651
grad_step = 000261, loss = 0.001650
grad_step = 000262, loss = 0.001652
grad_step = 000263, loss = 0.001650
grad_step = 000264, loss = 0.001645
grad_step = 000265, loss = 0.001644
grad_step = 000266, loss = 0.001643
grad_step = 000267, loss = 0.001642
grad_step = 000268, loss = 0.001640
grad_step = 000269, loss = 0.001637
grad_step = 000270, loss = 0.001636
grad_step = 000271, loss = 0.001636
grad_step = 000272, loss = 0.001634
grad_step = 000273, loss = 0.001632
grad_step = 000274, loss = 0.001630
grad_step = 000275, loss = 0.001628
grad_step = 000276, loss = 0.001627
grad_step = 000277, loss = 0.001626
grad_step = 000278, loss = 0.001625
grad_step = 000279, loss = 0.001624
grad_step = 000280, loss = 0.001623
grad_step = 000281, loss = 0.001621
grad_step = 000282, loss = 0.001620
grad_step = 000283, loss = 0.001620
grad_step = 000284, loss = 0.001621
grad_step = 000285, loss = 0.001627
grad_step = 000286, loss = 0.001641
grad_step = 000287, loss = 0.001674
grad_step = 000288, loss = 0.001730
grad_step = 000289, loss = 0.001823
grad_step = 000290, loss = 0.001880
grad_step = 000291, loss = 0.001894
grad_step = 000292, loss = 0.001742
grad_step = 000293, loss = 0.001631
grad_step = 000294, loss = 0.001673
grad_step = 000295, loss = 0.001764
grad_step = 000296, loss = 0.001756
grad_step = 000297, loss = 0.001663
grad_step = 000298, loss = 0.001677
grad_step = 000299, loss = 0.001730
grad_step = 000300, loss = 0.001675
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001613
grad_step = 000302, loss = 0.001627
grad_step = 000303, loss = 0.001646
grad_step = 000304, loss = 0.001621
grad_step = 000305, loss = 0.001603
grad_step = 000306, loss = 0.001629
grad_step = 000307, loss = 0.001646
grad_step = 000308, loss = 0.001617
grad_step = 000309, loss = 0.001598
grad_step = 000310, loss = 0.001608
grad_step = 000311, loss = 0.001613
grad_step = 000312, loss = 0.001597
grad_step = 000313, loss = 0.001581
grad_step = 000314, loss = 0.001585
grad_step = 000315, loss = 0.001600
grad_step = 000316, loss = 0.001603
grad_step = 000317, loss = 0.001594
grad_step = 000318, loss = 0.001585
grad_step = 000319, loss = 0.001586
grad_step = 000320, loss = 0.001592
grad_step = 000321, loss = 0.001592
grad_step = 000322, loss = 0.001587
grad_step = 000323, loss = 0.001577
grad_step = 000324, loss = 0.001569
grad_step = 000325, loss = 0.001566
grad_step = 000326, loss = 0.001568
grad_step = 000327, loss = 0.001571
grad_step = 000328, loss = 0.001572
grad_step = 000329, loss = 0.001571
grad_step = 000330, loss = 0.001569
grad_step = 000331, loss = 0.001567
grad_step = 000332, loss = 0.001565
grad_step = 000333, loss = 0.001565
grad_step = 000334, loss = 0.001567
grad_step = 000335, loss = 0.001575
grad_step = 000336, loss = 0.001592
grad_step = 000337, loss = 0.001618
grad_step = 000338, loss = 0.001665
grad_step = 000339, loss = 0.001716
grad_step = 000340, loss = 0.001780
grad_step = 000341, loss = 0.001791
grad_step = 000342, loss = 0.001764
grad_step = 000343, loss = 0.001711
grad_step = 000344, loss = 0.001665
grad_step = 000345, loss = 0.001667
grad_step = 000346, loss = 0.001666
grad_step = 000347, loss = 0.001657
grad_step = 000348, loss = 0.001602
grad_step = 000349, loss = 0.001572
grad_step = 000350, loss = 0.001584
grad_step = 000351, loss = 0.001603
grad_step = 000352, loss = 0.001600
grad_step = 000353, loss = 0.001565
grad_step = 000354, loss = 0.001549
grad_step = 000355, loss = 0.001563
grad_step = 000356, loss = 0.001576
grad_step = 000357, loss = 0.001565
grad_step = 000358, loss = 0.001539
grad_step = 000359, loss = 0.001533
grad_step = 000360, loss = 0.001550
grad_step = 000361, loss = 0.001562
grad_step = 000362, loss = 0.001554
grad_step = 000363, loss = 0.001531
grad_step = 000364, loss = 0.001519
grad_step = 000365, loss = 0.001525
grad_step = 000366, loss = 0.001536
grad_step = 000367, loss = 0.001540
grad_step = 000368, loss = 0.001535
grad_step = 000369, loss = 0.001529
grad_step = 000370, loss = 0.001527
grad_step = 000371, loss = 0.001527
grad_step = 000372, loss = 0.001522
grad_step = 000373, loss = 0.001514
grad_step = 000374, loss = 0.001508
grad_step = 000375, loss = 0.001506
grad_step = 000376, loss = 0.001507
grad_step = 000377, loss = 0.001509
grad_step = 000378, loss = 0.001509
grad_step = 000379, loss = 0.001507
grad_step = 000380, loss = 0.001506
grad_step = 000381, loss = 0.001510
grad_step = 000382, loss = 0.001521
grad_step = 000383, loss = 0.001543
grad_step = 000384, loss = 0.001585
grad_step = 000385, loss = 0.001650
grad_step = 000386, loss = 0.001742
grad_step = 000387, loss = 0.001784
grad_step = 000388, loss = 0.001750
grad_step = 000389, loss = 0.001604
grad_step = 000390, loss = 0.001520
grad_step = 000391, loss = 0.001562
grad_step = 000392, loss = 0.001602
grad_step = 000393, loss = 0.001558
grad_step = 000394, loss = 0.001503
grad_step = 000395, loss = 0.001542
grad_step = 000396, loss = 0.001582
grad_step = 000397, loss = 0.001525
grad_step = 000398, loss = 0.001489
grad_step = 000399, loss = 0.001521
grad_step = 000400, loss = 0.001529
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001498
grad_step = 000402, loss = 0.001489
grad_step = 000403, loss = 0.001511
grad_step = 000404, loss = 0.001512
grad_step = 000405, loss = 0.001486
grad_step = 000406, loss = 0.001473
grad_step = 000407, loss = 0.001484
grad_step = 000408, loss = 0.001495
grad_step = 000409, loss = 0.001489
grad_step = 000410, loss = 0.001475
grad_step = 000411, loss = 0.001470
grad_step = 000412, loss = 0.001476
grad_step = 000413, loss = 0.001478
grad_step = 000414, loss = 0.001470
grad_step = 000415, loss = 0.001461
grad_step = 000416, loss = 0.001459
grad_step = 000417, loss = 0.001462
grad_step = 000418, loss = 0.001466
grad_step = 000419, loss = 0.001466
grad_step = 000420, loss = 0.001460
grad_step = 000421, loss = 0.001454
grad_step = 000422, loss = 0.001450
grad_step = 000423, loss = 0.001449
grad_step = 000424, loss = 0.001450
grad_step = 000425, loss = 0.001451
grad_step = 000426, loss = 0.001450
grad_step = 000427, loss = 0.001447
grad_step = 000428, loss = 0.001444
grad_step = 000429, loss = 0.001440
grad_step = 000430, loss = 0.001438
grad_step = 000431, loss = 0.001436
grad_step = 000432, loss = 0.001434
grad_step = 000433, loss = 0.001434
grad_step = 000434, loss = 0.001434
grad_step = 000435, loss = 0.001435
grad_step = 000436, loss = 0.001437
grad_step = 000437, loss = 0.001442
grad_step = 000438, loss = 0.001451
grad_step = 000439, loss = 0.001468
grad_step = 000440, loss = 0.001496
grad_step = 000441, loss = 0.001541
grad_step = 000442, loss = 0.001588
grad_step = 000443, loss = 0.001624
grad_step = 000444, loss = 0.001591
grad_step = 000445, loss = 0.001520
grad_step = 000446, loss = 0.001449
grad_step = 000447, loss = 0.001453
grad_step = 000448, loss = 0.001522
grad_step = 000449, loss = 0.001558
grad_step = 000450, loss = 0.001559
grad_step = 000451, loss = 0.001528
grad_step = 000452, loss = 0.001560
grad_step = 000453, loss = 0.001594
grad_step = 000454, loss = 0.001549
grad_step = 000455, loss = 0.001471
grad_step = 000456, loss = 0.001426
grad_step = 000457, loss = 0.001431
grad_step = 000458, loss = 0.001445
grad_step = 000459, loss = 0.001451
grad_step = 000460, loss = 0.001461
grad_step = 000461, loss = 0.001469
grad_step = 000462, loss = 0.001440
grad_step = 000463, loss = 0.001406
grad_step = 000464, loss = 0.001396
grad_step = 000465, loss = 0.001412
grad_step = 000466, loss = 0.001429
grad_step = 000467, loss = 0.001422
grad_step = 000468, loss = 0.001406
grad_step = 000469, loss = 0.001398
grad_step = 000470, loss = 0.001402
grad_step = 000471, loss = 0.001408
grad_step = 000472, loss = 0.001404
grad_step = 000473, loss = 0.001391
grad_step = 000474, loss = 0.001382
grad_step = 000475, loss = 0.001381
grad_step = 000476, loss = 0.001386
grad_step = 000477, loss = 0.001392
grad_step = 000478, loss = 0.001393
grad_step = 000479, loss = 0.001389
grad_step = 000480, loss = 0.001383
grad_step = 000481, loss = 0.001378
grad_step = 000482, loss = 0.001376
grad_step = 000483, loss = 0.001375
grad_step = 000484, loss = 0.001375
grad_step = 000485, loss = 0.001373
grad_step = 000486, loss = 0.001370
grad_step = 000487, loss = 0.001366
grad_step = 000488, loss = 0.001363
grad_step = 000489, loss = 0.001361
grad_step = 000490, loss = 0.001360
grad_step = 000491, loss = 0.001361
grad_step = 000492, loss = 0.001362
grad_step = 000493, loss = 0.001364
grad_step = 000494, loss = 0.001367
grad_step = 000495, loss = 0.001373
grad_step = 000496, loss = 0.001383
grad_step = 000497, loss = 0.001402
grad_step = 000498, loss = 0.001433
grad_step = 000499, loss = 0.001486
grad_step = 000500, loss = 0.001545
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001598
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

  date_run                              2020-05-11 16:16:09.327865
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.262058
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 16:16:09.334847
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.171153
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 16:16:09.342904
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149937
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 16:16:09.348885
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.60073
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
0   2020-05-11 16:15:36.865470  ...    mean_absolute_error
1   2020-05-11 16:15:36.869006  ...     mean_squared_error
2   2020-05-11 16:15:36.872167  ...  median_absolute_error
3   2020-05-11 16:15:36.875449  ...               r2_score
4   2020-05-11 16:15:45.981987  ...    mean_absolute_error
5   2020-05-11 16:15:45.985816  ...     mean_squared_error
6   2020-05-11 16:15:45.989854  ...  median_absolute_error
7   2020-05-11 16:15:45.993423  ...               r2_score
8   2020-05-11 16:16:09.327865  ...    mean_absolute_error
9   2020-05-11 16:16:09.334847  ...     mean_squared_error
10  2020-05-11 16:16:09.342904  ...  median_absolute_error
11  2020-05-11 16:16:09.348885  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3391488/9912422 [00:00<00:00, 33873078.77it/s]9920512it [00:00, 34132731.74it/s]                             
0it [00:00, ?it/s]32768it [00:00, 494205.18it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461290.20it/s]1654784it [00:00, 11084052.08it/s]                         
0it [00:00, ?it/s]8192it [00:00, 190105.89it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd14d198ba8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ea8ece80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd14d15ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ea3c5080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd14d15ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ffb55e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ea3a6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ffb55e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd14d1a4fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd0ffb55e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd14d1a4fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3b97eee1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9aa1157392c643863a1cf6138576cf3bde0e5c3ccc606aad7c3cbabfa0f57307
  Stored in directory: /tmp/pip-ephem-wheel-cache-wc26r4em/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3b30bcbc88> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2686976/17464789 [===>..........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
16023552/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 16:17:34.240647: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 16:17:34.244501: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 16:17:34.244670: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f06813ecf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 16:17:34.244685: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6104 - accuracy: 0.5037
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6421 - accuracy: 0.5016
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6019 - accuracy: 0.5042
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5854 - accuracy: 0.5053
11000/25000 [============>.................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
12000/25000 [=============>................] - ETA: 4s - loss: 7.5682 - accuracy: 0.5064
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5711 - accuracy: 0.5062
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5921 - accuracy: 0.5049
15000/25000 [=================>............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6187 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6233 - accuracy: 0.5028
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6644 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 16:17:50.099578
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 16:17:50.099578  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 16:17:56.012077: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 16:17:56.017146: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 16:17:56.017426: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559df7a6d2b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 16:17:56.017530: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f750c623cf8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 977ms/step - loss: 1.8176 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.7158 - val_crf_viterbi_accuracy: 0.1867

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f74e8363898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6022 - accuracy: 0.5042
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6551 - accuracy: 0.5008
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6958 - accuracy: 0.4981
11000/25000 [============>.................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
12000/25000 [=============>................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 3s - loss: 7.6993 - accuracy: 0.4979
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6751 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f74dc0c15f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:52:24, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:14:43, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:25:31, 21.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:00:20, 29.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.10M/862M [00:01<5:35:38, 42.7kB/s].vector_cache/glove.6B.zip:   1%|          | 8.86M/862M [00:01<3:53:28, 60.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.4M/862M [00:01<2:42:39, 87.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:53:26, 124kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:01<1:19:04, 177kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:01<55:11, 253kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:01<38:30, 360kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:02<26:56, 512kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<18:50, 728kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:02<13:13, 1.03MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<09:23, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:44, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:37, 2.03MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<06:32, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<04:57, 2.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<05:51, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<04:27, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<05:52, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<06:49, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<05:22, 2.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:08<03:54, 3.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<13:17, 995kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<10:40, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<07:44, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<08:29, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<08:39, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:12<06:44, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:12<04:50, 2.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<1:38:04, 134kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<1:09:58, 187kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<49:13, 265kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<37:25, 348kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<28:49, 452kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<20:49, 625kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<16:37, 780kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<12:56, 1.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<09:22, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<09:34, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<08:00, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<07:05, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<06:09, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:39, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:22<03:25, 3.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<42:50, 298kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<32:48, 389kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<23:32, 541kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<16:34, 766kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<19:37, 646kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<15:06, 839kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<10:51, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<10:21, 1.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:01, 1.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:42, 1.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<05:31, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<14:30, 864kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:30, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:23, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:36, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:39, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:38, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:52, 2.55MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:56, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:00, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:32, 2.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:17, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:41, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:25, 3.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<17:06, 717kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<13:19, 920kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<09:38, 1.27MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:24, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:16, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:05, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:14, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:35, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:57, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:27, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:26, 3.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:14, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:53, 2.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:55, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:26, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:07, 2.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:30, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:12, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<03:46, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<15:21, 772kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<12:03, 981kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:45, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:41, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:22, 1.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:26, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:22, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:44, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:19, 2.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:06, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<03:43, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<20:19, 569kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<15:29, 746kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<11:06, 1.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:15, 1.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:43, 1.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:27, 1.54MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<05:19, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<16:01, 713kB/s] .vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<12:28, 916kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:59, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<08:46, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:40, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:41, 1.69MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:49, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<21:06, 535kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<15:59, 706kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<11:25, 985kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:28, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:48, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:25, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<05:18, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<12:10, 916kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<09:43, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:05, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:24, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:31, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:52, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:13, 2.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<19:10, 574kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<14:36, 754kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<10:29, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<09:46, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:19, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:07, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:06, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<20:39, 527kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<15:38, 695kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<11:14, 965kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<10:12, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:35, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:12, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:16, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:25, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:40, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:15, 2.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:21, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:06, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:45, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:30, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<05:42, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:10, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:51, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:01, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:52, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:35, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:23, 3.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:31, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:58, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:42, 2.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:55, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:36, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:28, 2.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:39, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:28, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:19, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:07, 3.29MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<13:22, 767kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<10:20, 991kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:48, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:36, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:31, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<07:35, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:48, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:12, 2.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:48, 1.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:53, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:21, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:14, 1.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:46, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:34, 2.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:41, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:23, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:20, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:28, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:20, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:15, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:06, 3.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<12:38, 778kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<09:44, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:08, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:07, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<14:21, 680kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<11:06, 878kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<08:00, 1.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:44, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:36, 1.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:51, 1.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:12, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<15:34, 618kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<13:01, 739kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:38, 996kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<06:50, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<16:23, 583kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<12:31, 763kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<09:00, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:21, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<06:53, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:04, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:36, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:59, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:21, 2.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:25, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:30, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:06, 2.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:54, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:29, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<04:16, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:06, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:47, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:04, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:45, 2.45MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:14, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:10, 2.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<04:57, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<03:53, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:51, 3.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:05, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:23, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:19, 2.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:17, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<03:58, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:01, 2.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:03, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:47, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:45, 3.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:34, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:50, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:37, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:26, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:56, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:50, 3.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<10:41, 810kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<08:25, 1.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:05, 1.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:07, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:01, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:41, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:43, 3.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<20:42, 411kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<16:17, 523kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<11:50, 718kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<08:20, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<23:56, 353kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<17:40, 477kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<12:32, 671kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<10:35, 791kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<09:11, 911kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<06:51, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:53, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<28:25, 292kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<20:47, 399kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<14:42, 563kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<12:03, 683kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<10:16, 802kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:38, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<05:25, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<13:42, 596kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:17, 793kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<07:24, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:05, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:40, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:39, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<26:41, 301kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<19:31, 411kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<13:50, 578kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<11:26, 696kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:46, 815kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:12, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<05:07, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<08:03, 980kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:30, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:43, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:01, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:22, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:15, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:58, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:29, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:34, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<02:34, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<10:03, 763kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:52, 976kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:42, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:38, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:33, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:13, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:02, 2.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:45, 1.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:51, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:08, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:33, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:35, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:36, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<10:48, 685kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<08:21, 885kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<06:02, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:49, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:37, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:16, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<03:03, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<08:12, 886kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:30, 1.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:42, 1.54MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:56, 1.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:02, 1.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:55, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:49, 2.53MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<09:35, 744kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:20, 970kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:19, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:15, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:10, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:50, 2.47MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:22, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:31, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:51, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:13, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:20, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:24, 2.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<08:11, 837kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<06:26, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:40, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:04, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:00, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:38, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:02, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:09, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:17, 2.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<07:54, 840kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:13, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:30, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:39, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:42, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:39, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:36, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<08:49, 738kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:53, 944kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:57, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:52, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:49, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:43, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:39, 2.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<09:50, 647kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<07:34, 841kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:27, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<05:10, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<05:00, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:51, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:45, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<08:36, 723kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<06:35, 943kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:46, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:40, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:41, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:37, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:35, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<08:19, 732kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<06:27, 942kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:39, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:37, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:34, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:31, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:31, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<08:09, 729kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<06:21, 936kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:34, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<04:28, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<04:22, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<02:24, 2.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<14:07, 412kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<10:30, 552kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:29, 772kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<06:28, 888kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<05:47, 991kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:04, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<06:18, 900kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<05:01, 1.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:38, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:45, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:52, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:58, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:08, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:38, 724kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<05:57, 928kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:16, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<04:13, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<04:06, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:06, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:14, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:34, 1.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:47, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:46, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:05, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:43, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:02, 2.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:37, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:20, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<01:42, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:17, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:52, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:07, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:36, 1.97MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:23, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:46, 2.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:21, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:46, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:35, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:47, 861kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:34, 1.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:18, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:22, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:24, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:38, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:53, 2.57MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<11:41, 414kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<08:41, 557kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<06:10, 779kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:22, 889kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:16, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:05, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:10, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:15, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:32, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:49, 2.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:13, 744kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:52, 950kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:30, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:25, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:24, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:37, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:52, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<06:07, 734kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:46, 940kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:26, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:21, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:48, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:03, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:26, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:11, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:37, 2.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:03, 2.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:23, 1.80MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:22, 3.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:28, 771kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:15, 989kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:04, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:05, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:02, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:20, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:40, 2.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<13:26, 303kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<09:51, 413kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<06:56, 583kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<05:42, 703kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:52, 822kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:37, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:33, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:01, 655kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:37, 850kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:18, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:09, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:37, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:54, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:09, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:22, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:52, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:20, 2.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:54, 760kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:50, 970kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:45, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:42, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:18, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:41, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:57, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:12, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:42, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:14, 2.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<04:50, 728kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:46, 932kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:43, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:38, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:36, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:00, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:25, 2.37MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:38, 730kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:36, 937kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:35, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:31, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:29, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:54, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<03:26, 943kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:46, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:00, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:05, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:09, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:41, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:11, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:10, 745kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:15, 951kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:20, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:17, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:16, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:14, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:42, 799kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:55, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:05, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:05, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:06, 1.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:37, 1.77MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:09, 2.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<03:50, 737kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:59, 943kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<02:09, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:05, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:04, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:34, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:07, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:20, 804kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:37, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:53, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:52, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:51, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:26, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:01, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<04:37, 551kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:30, 724kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:29, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:15, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:50, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:20, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:27, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:17, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:57, 2.49MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:10, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:20, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:45, 3.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:17, 691kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:32, 891kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:48, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:44, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:41, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:17, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:54, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<03:59, 535kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:00, 705kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:08, 981kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:55, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:48, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:22, 1.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:57, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:48, 709kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:10, 913kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:32, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:29, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:28, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:47, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:32, 731kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:58, 937kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:24, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:21, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:20, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:01, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:20, 734kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:49, 940kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:17, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:14, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:14, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:56, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:39, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:09, 733kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:39, 943kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:10, 1.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:08, 1.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:07, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:50, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:36, 2.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:54, 1.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:47, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:34, 2.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:42, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:26, 2.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:41, 765kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:19, 977kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:56, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:54, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:54, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:29, 2.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:34, 735kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:13, 941kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:52, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:25, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:50, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:41, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:29, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:27, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:18, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:02, 432kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:30, 580kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:03, 810kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:53, 917kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:47, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:35, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:24, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:05, 685kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:50, 884kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:35, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:32, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:20, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:22, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:29, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:23, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:09, 2.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:05, 3.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:23, 848kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:20, 955kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:12, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:03, 2.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:09, 801kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:07, 913kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 674kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:03, 871kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.20MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 777/400000 [00:00<00:51, 7754.92it/s]  0%|          | 1548/400000 [00:00<00:51, 7740.84it/s]  1%|          | 2302/400000 [00:00<00:51, 7677.38it/s]  1%|          | 3074/400000 [00:00<00:51, 7686.80it/s]  1%|          | 3792/400000 [00:00<00:52, 7525.42it/s]  1%|          | 4520/400000 [00:00<00:53, 7449.25it/s]  1%|▏         | 5258/400000 [00:00<00:53, 7427.55it/s]  2%|▏         | 6035/400000 [00:00<00:52, 7525.37it/s]  2%|▏         | 6850/400000 [00:00<00:51, 7700.04it/s]  2%|▏         | 7629/400000 [00:01<00:50, 7725.37it/s]  2%|▏         | 8473/400000 [00:01<00:49, 7925.68it/s]  2%|▏         | 9348/400000 [00:01<00:47, 8154.28it/s]  3%|▎         | 10236/400000 [00:01<00:46, 8358.15it/s]  3%|▎         | 11103/400000 [00:01<00:46, 8447.75it/s]  3%|▎         | 11946/400000 [00:01<00:46, 8412.79it/s]  3%|▎         | 12786/400000 [00:01<00:48, 8022.22it/s]  3%|▎         | 13591/400000 [00:01<00:50, 7725.15it/s]  4%|▎         | 14368/400000 [00:01<00:51, 7448.39it/s]  4%|▍         | 15118/400000 [00:01<00:51, 7452.28it/s]  4%|▍         | 15875/400000 [00:02<00:51, 7486.64it/s]  4%|▍         | 16640/400000 [00:02<00:50, 7532.62it/s]  4%|▍         | 17415/400000 [00:02<00:50, 7595.17it/s]  5%|▍         | 18188/400000 [00:02<00:50, 7634.36it/s]  5%|▍         | 18996/400000 [00:02<00:49, 7760.83it/s]  5%|▍         | 19780/400000 [00:02<00:48, 7782.59it/s]  5%|▌         | 20641/400000 [00:02<00:47, 8012.39it/s]  5%|▌         | 21445/400000 [00:02<00:47, 7955.66it/s]  6%|▌         | 22248/400000 [00:02<00:47, 7975.89it/s]  6%|▌         | 23047/400000 [00:02<00:47, 7969.15it/s]  6%|▌         | 23870/400000 [00:03<00:46, 8043.16it/s]  6%|▌         | 24725/400000 [00:03<00:45, 8186.41it/s]  6%|▋         | 25585/400000 [00:03<00:45, 8303.47it/s]  7%|▋         | 26446/400000 [00:03<00:44, 8390.14it/s]  7%|▋         | 27326/400000 [00:03<00:43, 8508.36it/s]  7%|▋         | 28214/400000 [00:03<00:43, 8614.48it/s]  7%|▋         | 29167/400000 [00:03<00:41, 8869.45it/s]  8%|▊         | 30057/400000 [00:03<00:42, 8756.60it/s]  8%|▊         | 30935/400000 [00:03<00:43, 8529.65it/s]  8%|▊         | 31791/400000 [00:03<00:44, 8234.30it/s]  8%|▊         | 32622/400000 [00:04<00:44, 8255.14it/s]  8%|▊         | 33514/400000 [00:04<00:43, 8442.47it/s]  9%|▊         | 34391/400000 [00:04<00:42, 8536.15it/s]  9%|▉         | 35302/400000 [00:04<00:41, 8698.57it/s]  9%|▉         | 36232/400000 [00:04<00:41, 8867.87it/s]  9%|▉         | 37122/400000 [00:04<00:41, 8743.60it/s] 10%|▉         | 38067/400000 [00:04<00:40, 8943.96it/s] 10%|▉         | 38965/400000 [00:04<00:41, 8674.01it/s] 10%|▉         | 39837/400000 [00:04<00:41, 8590.26it/s] 10%|█         | 40699/400000 [00:04<00:42, 8494.37it/s] 10%|█         | 41567/400000 [00:05<00:41, 8548.11it/s] 11%|█         | 42436/400000 [00:05<00:41, 8588.17it/s] 11%|█         | 43296/400000 [00:05<00:41, 8520.98it/s] 11%|█         | 44234/400000 [00:05<00:40, 8760.84it/s] 11%|█▏        | 45128/400000 [00:05<00:40, 8813.45it/s] 12%|█▏        | 46012/400000 [00:05<00:41, 8443.83it/s] 12%|█▏        | 46861/400000 [00:05<00:41, 8456.19it/s] 12%|█▏        | 47710/400000 [00:05<00:42, 8256.50it/s] 12%|█▏        | 48539/400000 [00:05<00:43, 8135.38it/s] 12%|█▏        | 49370/400000 [00:06<00:42, 8186.65it/s] 13%|█▎        | 50191/400000 [00:06<00:43, 8076.80it/s] 13%|█▎        | 51001/400000 [00:06<00:44, 7844.70it/s] 13%|█▎        | 51789/400000 [00:06<00:44, 7854.15it/s] 13%|█▎        | 52577/400000 [00:06<00:44, 7751.45it/s] 13%|█▎        | 53354/400000 [00:06<00:45, 7645.85it/s] 14%|█▎        | 54137/400000 [00:06<00:44, 7699.80it/s] 14%|█▎        | 54933/400000 [00:06<00:44, 7775.12it/s] 14%|█▍        | 55712/400000 [00:06<00:44, 7668.03it/s] 14%|█▍        | 56480/400000 [00:06<00:45, 7468.49it/s] 14%|█▍        | 57229/400000 [00:07<00:46, 7373.72it/s] 15%|█▍        | 58063/400000 [00:07<00:44, 7637.71it/s] 15%|█▍        | 58849/400000 [00:07<00:44, 7702.77it/s] 15%|█▍        | 59622/400000 [00:07<00:44, 7606.47it/s] 15%|█▌        | 60390/400000 [00:07<00:44, 7626.92it/s] 15%|█▌        | 61162/400000 [00:07<00:44, 7652.02it/s] 15%|█▌        | 61929/400000 [00:07<00:44, 7561.31it/s] 16%|█▌        | 62705/400000 [00:07<00:44, 7617.94it/s] 16%|█▌        | 63475/400000 [00:07<00:44, 7640.85it/s] 16%|█▌        | 64240/400000 [00:07<00:44, 7517.01it/s] 16%|█▌        | 64993/400000 [00:08<00:45, 7417.94it/s] 16%|█▋        | 65736/400000 [00:08<00:45, 7337.74it/s] 17%|█▋        | 66549/400000 [00:08<00:44, 7557.48it/s] 17%|█▋        | 67339/400000 [00:08<00:43, 7655.97it/s] 17%|█▋        | 68109/400000 [00:08<00:43, 7668.34it/s] 17%|█▋        | 68925/400000 [00:08<00:42, 7807.68it/s] 17%|█▋        | 69728/400000 [00:08<00:41, 7872.21it/s] 18%|█▊        | 70584/400000 [00:08<00:40, 8066.39it/s] 18%|█▊        | 71397/400000 [00:08<00:40, 8084.76it/s] 18%|█▊        | 72207/400000 [00:09<00:41, 7912.71it/s] 18%|█▊        | 73001/400000 [00:09<00:41, 7867.33it/s] 18%|█▊        | 73790/400000 [00:09<00:41, 7797.30it/s] 19%|█▊        | 74599/400000 [00:09<00:41, 7881.59it/s] 19%|█▉        | 75439/400000 [00:09<00:40, 8029.04it/s] 19%|█▉        | 76252/400000 [00:09<00:40, 8057.64it/s] 19%|█▉        | 77059/400000 [00:09<00:40, 8057.76it/s] 19%|█▉        | 77866/400000 [00:09<00:40, 8020.70it/s] 20%|█▉        | 78669/400000 [00:09<00:41, 7802.84it/s] 20%|█▉        | 79478/400000 [00:09<00:40, 7884.84it/s] 20%|██        | 80268/400000 [00:10<00:40, 7879.65it/s] 20%|██        | 81057/400000 [00:10<00:41, 7626.86it/s] 20%|██        | 81823/400000 [00:10<00:42, 7481.46it/s] 21%|██        | 82613/400000 [00:10<00:41, 7601.21it/s] 21%|██        | 83399/400000 [00:10<00:41, 7675.03it/s] 21%|██        | 84169/400000 [00:10<00:41, 7628.79it/s] 21%|██        | 84934/400000 [00:10<00:41, 7518.10it/s] 21%|██▏       | 85804/400000 [00:10<00:40, 7836.92it/s] 22%|██▏       | 86684/400000 [00:10<00:38, 8102.19it/s] 22%|██▏       | 87585/400000 [00:10<00:37, 8352.64it/s] 22%|██▏       | 88426/400000 [00:11<00:37, 8328.48it/s] 22%|██▏       | 89263/400000 [00:11<00:38, 8106.78it/s] 23%|██▎       | 90078/400000 [00:11<00:38, 8045.83it/s] 23%|██▎       | 90886/400000 [00:11<00:39, 7912.88it/s] 23%|██▎       | 91681/400000 [00:11<00:38, 7922.07it/s] 23%|██▎       | 92476/400000 [00:11<00:39, 7878.95it/s] 23%|██▎       | 93266/400000 [00:11<00:39, 7802.18it/s] 24%|██▎       | 94076/400000 [00:11<00:38, 7886.31it/s] 24%|██▎       | 94960/400000 [00:11<00:37, 8148.61it/s] 24%|██▍       | 95880/400000 [00:11<00:36, 8437.25it/s] 24%|██▍       | 96755/400000 [00:12<00:35, 8526.07it/s] 24%|██▍       | 97612/400000 [00:12<00:36, 8239.19it/s] 25%|██▍       | 98441/400000 [00:12<00:37, 8044.13it/s] 25%|██▍       | 99250/400000 [00:12<00:37, 7919.83it/s] 25%|██▌       | 100100/400000 [00:12<00:37, 8082.84it/s] 25%|██▌       | 100912/400000 [00:12<00:37, 8063.23it/s] 25%|██▌       | 101721/400000 [00:12<00:37, 7891.74it/s] 26%|██▌       | 102553/400000 [00:12<00:37, 8014.16it/s] 26%|██▌       | 103357/400000 [00:12<00:37, 7819.86it/s] 26%|██▌       | 104150/400000 [00:13<00:37, 7848.99it/s] 26%|██▌       | 104937/400000 [00:13<00:37, 7786.70it/s] 26%|██▋       | 105718/400000 [00:13<00:38, 7682.21it/s] 27%|██▋       | 106515/400000 [00:13<00:37, 7764.83it/s] 27%|██▋       | 107293/400000 [00:13<00:37, 7726.77it/s] 27%|██▋       | 108150/400000 [00:13<00:36, 7960.13it/s] 27%|██▋       | 108987/400000 [00:13<00:36, 8075.77it/s] 27%|██▋       | 109797/400000 [00:13<00:36, 8053.39it/s] 28%|██▊       | 110664/400000 [00:13<00:35, 8227.11it/s] 28%|██▊       | 111520/400000 [00:13<00:34, 8322.16it/s] 28%|██▊       | 112354/400000 [00:14<00:34, 8300.99it/s] 28%|██▊       | 113193/400000 [00:14<00:34, 8324.87it/s] 29%|██▊       | 114027/400000 [00:14<00:34, 8213.54it/s] 29%|██▊       | 114850/400000 [00:14<00:36, 7822.43it/s] 29%|██▉       | 115637/400000 [00:14<00:37, 7563.41it/s] 29%|██▉       | 116428/400000 [00:14<00:37, 7663.34it/s] 29%|██▉       | 117217/400000 [00:14<00:36, 7729.32it/s] 29%|██▉       | 117993/400000 [00:14<00:37, 7518.40it/s] 30%|██▉       | 118775/400000 [00:14<00:36, 7606.30it/s] 30%|██▉       | 119650/400000 [00:14<00:35, 7914.71it/s] 30%|███       | 120557/400000 [00:15<00:33, 8229.16it/s] 30%|███       | 121387/400000 [00:15<00:33, 8245.35it/s] 31%|███       | 122217/400000 [00:15<00:34, 8129.36it/s] 31%|███       | 123034/400000 [00:15<00:34, 7961.26it/s] 31%|███       | 123834/400000 [00:15<00:34, 7937.31it/s] 31%|███       | 124631/400000 [00:15<00:34, 7880.07it/s] 31%|███▏      | 125471/400000 [00:15<00:34, 8028.61it/s] 32%|███▏      | 126276/400000 [00:15<00:34, 7984.10it/s] 32%|███▏      | 127101/400000 [00:15<00:33, 8061.92it/s] 32%|███▏      | 127969/400000 [00:15<00:33, 8236.88it/s] 32%|███▏      | 128879/400000 [00:16<00:31, 8477.62it/s] 32%|███▏      | 129762/400000 [00:16<00:31, 8579.79it/s] 33%|███▎      | 130623/400000 [00:16<00:31, 8564.16it/s] 33%|███▎      | 131482/400000 [00:16<00:32, 8350.69it/s] 33%|███▎      | 132320/400000 [00:16<00:32, 8181.20it/s] 33%|███▎      | 133141/400000 [00:16<00:33, 8077.74it/s] 34%|███▎      | 134053/400000 [00:16<00:31, 8361.95it/s] 34%|███▎      | 134894/400000 [00:16<00:32, 8240.22it/s] 34%|███▍      | 135727/400000 [00:16<00:31, 8266.65it/s] 34%|███▍      | 136569/400000 [00:17<00:31, 8311.86it/s] 34%|███▍      | 137459/400000 [00:17<00:30, 8477.94it/s] 35%|███▍      | 138345/400000 [00:17<00:30, 8588.14it/s] 35%|███▍      | 139206/400000 [00:17<00:30, 8428.24it/s] 35%|███▌      | 140051/400000 [00:17<00:30, 8403.19it/s] 35%|███▌      | 140893/400000 [00:17<00:31, 8259.29it/s] 35%|███▌      | 141721/400000 [00:17<00:33, 7697.87it/s] 36%|███▌      | 142521/400000 [00:17<00:33, 7783.69it/s] 36%|███▌      | 143365/400000 [00:17<00:32, 7968.43it/s] 36%|███▌      | 144236/400000 [00:17<00:31, 8177.27it/s] 36%|███▋      | 145143/400000 [00:18<00:30, 8423.98it/s] 37%|███▋      | 146055/400000 [00:18<00:29, 8620.00it/s] 37%|███▋      | 146964/400000 [00:18<00:28, 8753.52it/s] 37%|███▋      | 147844/400000 [00:18<00:29, 8507.24it/s] 37%|███▋      | 148700/400000 [00:18<00:30, 8341.56it/s] 37%|███▋      | 149538/400000 [00:18<00:31, 8009.35it/s] 38%|███▊      | 150345/400000 [00:18<00:31, 7866.76it/s] 38%|███▊      | 151137/400000 [00:18<00:32, 7770.38it/s] 38%|███▊      | 151918/400000 [00:18<00:32, 7646.19it/s] 38%|███▊      | 152733/400000 [00:19<00:31, 7790.33it/s] 38%|███▊      | 153608/400000 [00:19<00:30, 8055.10it/s] 39%|███▊      | 154522/400000 [00:19<00:29, 8351.05it/s] 39%|███▉      | 155389/400000 [00:19<00:28, 8444.00it/s] 39%|███▉      | 156258/400000 [00:19<00:28, 8513.93it/s] 39%|███▉      | 157162/400000 [00:19<00:28, 8665.13it/s] 40%|███▉      | 158032/400000 [00:19<00:29, 8132.60it/s] 40%|███▉      | 158854/400000 [00:19<00:30, 7887.16it/s] 40%|███▉      | 159651/400000 [00:19<00:31, 7652.95it/s] 40%|████      | 160439/400000 [00:19<00:31, 7719.13it/s] 40%|████      | 161235/400000 [00:20<00:30, 7787.98it/s] 41%|████      | 162041/400000 [00:20<00:30, 7866.86it/s] 41%|████      | 162842/400000 [00:20<00:29, 7908.68it/s] 41%|████      | 163638/400000 [00:20<00:29, 7922.67it/s] 41%|████      | 164491/400000 [00:20<00:29, 8092.59it/s] 41%|████▏     | 165303/400000 [00:20<00:29, 8048.30it/s] 42%|████▏     | 166110/400000 [00:20<00:29, 7870.61it/s] 42%|████▏     | 166899/400000 [00:20<00:30, 7698.63it/s] 42%|████▏     | 167671/400000 [00:20<00:30, 7538.17it/s] 42%|████▏     | 168493/400000 [00:20<00:29, 7728.17it/s] 42%|████▏     | 169293/400000 [00:21<00:29, 7806.02it/s] 43%|████▎     | 170076/400000 [00:21<00:29, 7799.42it/s] 43%|████▎     | 170897/400000 [00:21<00:28, 7916.93it/s] 43%|████▎     | 171702/400000 [00:21<00:28, 7953.85it/s] 43%|████▎     | 172604/400000 [00:21<00:27, 8245.22it/s] 43%|████▎     | 173490/400000 [00:21<00:26, 8420.35it/s] 44%|████▎     | 174388/400000 [00:21<00:26, 8578.19it/s] 44%|████▍     | 175249/400000 [00:21<00:26, 8340.35it/s] 44%|████▍     | 176087/400000 [00:21<00:27, 8105.41it/s] 44%|████▍     | 176902/400000 [00:22<00:27, 8117.09it/s] 44%|████▍     | 177741/400000 [00:22<00:27, 8195.71it/s] 45%|████▍     | 178563/400000 [00:22<00:27, 7967.12it/s] 45%|████▍     | 179363/400000 [00:22<00:27, 7892.18it/s] 45%|████▌     | 180155/400000 [00:22<00:27, 7888.95it/s] 45%|████▌     | 181043/400000 [00:22<00:26, 8160.06it/s] 45%|████▌     | 181925/400000 [00:22<00:26, 8346.07it/s] 46%|████▌     | 182793/400000 [00:22<00:25, 8442.72it/s] 46%|████▌     | 183640/400000 [00:22<00:25, 8343.53it/s] 46%|████▌     | 184477/400000 [00:22<00:26, 8145.40it/s] 46%|████▋     | 185295/400000 [00:23<00:26, 8131.95it/s] 47%|████▋     | 186152/400000 [00:23<00:25, 8256.90it/s] 47%|████▋     | 186980/400000 [00:23<00:25, 8261.93it/s] 47%|████▋     | 187823/400000 [00:23<00:25, 8309.24it/s] 47%|████▋     | 188655/400000 [00:23<00:25, 8207.73it/s] 47%|████▋     | 189538/400000 [00:23<00:25, 8384.09it/s] 48%|████▊     | 190408/400000 [00:23<00:24, 8474.24it/s] 48%|████▊     | 191333/400000 [00:23<00:24, 8692.54it/s] 48%|████▊     | 192220/400000 [00:23<00:23, 8744.92it/s] 48%|████▊     | 193097/400000 [00:23<00:25, 7988.28it/s] 48%|████▊     | 193910/400000 [00:24<00:25, 7938.28it/s] 49%|████▊     | 194720/400000 [00:24<00:25, 7984.85it/s] 49%|████▉     | 195611/400000 [00:24<00:24, 8239.27it/s] 49%|████▉     | 196447/400000 [00:24<00:24, 8272.50it/s] 49%|████▉     | 197292/400000 [00:24<00:24, 8322.68it/s] 50%|████▉     | 198165/400000 [00:24<00:23, 8438.43it/s] 50%|████▉     | 199056/400000 [00:24<00:23, 8572.23it/s] 50%|████▉     | 199944/400000 [00:24<00:23, 8659.81it/s] 50%|█████     | 200814/400000 [00:24<00:22, 8670.83it/s] 50%|█████     | 201683/400000 [00:24<00:23, 8470.18it/s] 51%|█████     | 202533/400000 [00:25<00:23, 8314.15it/s] 51%|█████     | 203367/400000 [00:25<00:23, 8201.48it/s] 51%|█████     | 204258/400000 [00:25<00:23, 8401.68it/s] 51%|█████▏    | 205101/400000 [00:25<00:23, 8390.52it/s] 51%|█████▏    | 205943/400000 [00:25<00:23, 8396.80it/s] 52%|█████▏    | 206784/400000 [00:25<00:23, 8311.39it/s] 52%|█████▏    | 207670/400000 [00:25<00:22, 8466.04it/s] 52%|█████▏    | 208563/400000 [00:25<00:22, 8599.48it/s] 52%|█████▏    | 209446/400000 [00:25<00:21, 8666.64it/s] 53%|█████▎    | 210314/400000 [00:26<00:22, 8582.25it/s] 53%|█████▎    | 211174/400000 [00:26<00:22, 8445.09it/s] 53%|█████▎    | 212020/400000 [00:26<00:22, 8270.89it/s] 53%|█████▎    | 212871/400000 [00:26<00:22, 8340.50it/s] 53%|█████▎    | 213717/400000 [00:26<00:22, 8373.80it/s] 54%|█████▎    | 214556/400000 [00:26<00:22, 8273.26it/s] 54%|█████▍    | 215394/400000 [00:26<00:22, 8304.94it/s] 54%|█████▍    | 216318/400000 [00:26<00:21, 8563.68it/s] 54%|█████▍    | 217239/400000 [00:26<00:20, 8745.90it/s] 55%|█████▍    | 218141/400000 [00:26<00:20, 8825.63it/s] 55%|█████▍    | 219026/400000 [00:27<00:20, 8635.47it/s] 55%|█████▍    | 219892/400000 [00:27<00:21, 8457.48it/s] 55%|█████▌    | 220741/400000 [00:27<00:21, 8237.34it/s] 55%|█████▌    | 221568/400000 [00:27<00:21, 8165.71it/s] 56%|█████▌    | 222395/400000 [00:27<00:21, 8194.44it/s] 56%|█████▌    | 223254/400000 [00:27<00:21, 8307.03it/s] 56%|█████▌    | 224087/400000 [00:27<00:21, 8201.14it/s] 56%|█████▌    | 224943/400000 [00:27<00:21, 8304.73it/s] 56%|█████▋    | 225804/400000 [00:27<00:20, 8391.94it/s] 57%|█████▋    | 226645/400000 [00:27<00:20, 8357.29it/s] 57%|█████▋    | 227495/400000 [00:28<00:20, 8397.81it/s] 57%|█████▋    | 228336/400000 [00:28<00:20, 8252.45it/s] 57%|█████▋    | 229163/400000 [00:28<00:20, 8154.60it/s] 57%|█████▋    | 229980/400000 [00:28<00:20, 8151.97it/s] 58%|█████▊    | 230837/400000 [00:28<00:20, 8270.50it/s] 58%|█████▊    | 231694/400000 [00:28<00:20, 8357.73it/s] 58%|█████▊    | 232585/400000 [00:28<00:19, 8514.19it/s] 58%|█████▊    | 233483/400000 [00:28<00:19, 8646.50it/s] 59%|█████▊    | 234376/400000 [00:28<00:18, 8727.27it/s] 59%|█████▉    | 235250/400000 [00:28<00:19, 8526.91it/s] 59%|█████▉    | 236129/400000 [00:29<00:19, 8602.14it/s] 59%|█████▉    | 236995/400000 [00:29<00:18, 8616.95it/s] 59%|█████▉    | 237858/400000 [00:29<00:19, 8444.79it/s] 60%|█████▉    | 238704/400000 [00:29<00:19, 8282.39it/s] 60%|█████▉    | 239561/400000 [00:29<00:19, 8365.21it/s] 60%|██████    | 240445/400000 [00:29<00:18, 8500.62it/s] 60%|██████    | 241305/400000 [00:29<00:18, 8528.15it/s] 61%|██████    | 242245/400000 [00:29<00:17, 8767.57it/s] 61%|██████    | 243185/400000 [00:29<00:17, 8947.95it/s] 61%|██████    | 244083/400000 [00:29<00:17, 8941.84it/s] 61%|██████▏   | 245014/400000 [00:30<00:17, 9048.41it/s] 61%|██████▏   | 245921/400000 [00:30<00:17, 8981.16it/s] 62%|██████▏   | 246821/400000 [00:30<00:17, 8530.95it/s] 62%|██████▏   | 247680/400000 [00:30<00:18, 8250.49it/s] 62%|██████▏   | 248511/400000 [00:30<00:18, 8115.74it/s] 62%|██████▏   | 249328/400000 [00:30<00:18, 8090.72it/s] 63%|██████▎   | 250141/400000 [00:30<00:18, 8084.99it/s] 63%|██████▎   | 251029/400000 [00:30<00:17, 8305.54it/s] 63%|██████▎   | 251971/400000 [00:30<00:17, 8610.36it/s] 63%|██████▎   | 252838/400000 [00:31<00:17, 8578.61it/s] 63%|██████▎   | 253778/400000 [00:31<00:16, 8808.30it/s] 64%|██████▎   | 254663/400000 [00:31<00:16, 8806.12it/s] 64%|██████▍   | 255547/400000 [00:31<00:16, 8533.71it/s] 64%|██████▍   | 256405/400000 [00:31<00:17, 8350.91it/s] 64%|██████▍   | 257244/400000 [00:31<00:17, 8219.39it/s] 65%|██████▍   | 258069/400000 [00:31<00:17, 8153.43it/s] 65%|██████▍   | 258905/400000 [00:31<00:17, 8211.83it/s] 65%|██████▍   | 259808/400000 [00:31<00:16, 8440.99it/s] 65%|██████▌   | 260762/400000 [00:31<00:15, 8741.53it/s] 65%|██████▌   | 261668/400000 [00:32<00:15, 8833.30it/s] 66%|██████▌   | 262555/400000 [00:32<00:15, 8837.51it/s] 66%|██████▌   | 263442/400000 [00:32<00:15, 8702.20it/s] 66%|██████▌   | 264315/400000 [00:32<00:16, 8361.53it/s] 66%|██████▋   | 265156/400000 [00:32<00:16, 8169.76it/s] 66%|██████▋   | 265978/400000 [00:32<00:16, 8046.98it/s] 67%|██████▋   | 266795/400000 [00:32<00:16, 8081.77it/s] 67%|██████▋   | 267636/400000 [00:32<00:16, 8175.99it/s] 67%|██████▋   | 268472/400000 [00:32<00:15, 8228.35it/s] 67%|██████▋   | 269309/400000 [00:32<00:15, 8270.31it/s] 68%|██████▊   | 270144/400000 [00:33<00:15, 8292.39it/s] 68%|██████▊   | 270997/400000 [00:33<00:15, 8361.41it/s] 68%|██████▊   | 271871/400000 [00:33<00:15, 8468.97it/s] 68%|██████▊   | 272719/400000 [00:33<00:15, 8334.90it/s] 68%|██████▊   | 273554/400000 [00:33<00:15, 8281.36it/s] 69%|██████▊   | 274383/400000 [00:33<00:15, 8203.06it/s] 69%|██████▉   | 275287/400000 [00:33<00:14, 8437.25it/s] 69%|██████▉   | 276133/400000 [00:33<00:15, 8238.07it/s] 69%|██████▉   | 276960/400000 [00:33<00:14, 8234.20it/s] 69%|██████▉   | 277858/400000 [00:34<00:14, 8443.56it/s] 70%|██████▉   | 278798/400000 [00:34<00:13, 8708.67it/s] 70%|██████▉   | 279770/400000 [00:34<00:13, 8987.56it/s] 70%|███████   | 280737/400000 [00:34<00:12, 9179.60it/s] 70%|███████   | 281660/400000 [00:34<00:12, 9129.57it/s] 71%|███████   | 282577/400000 [00:34<00:13, 8797.81it/s] 71%|███████   | 283462/400000 [00:34<00:13, 8554.39it/s] 71%|███████   | 284336/400000 [00:34<00:13, 8607.67it/s] 71%|███████▏  | 285201/400000 [00:34<00:13, 8558.48it/s] 72%|███████▏  | 286060/400000 [00:34<00:13, 8430.73it/s] 72%|███████▏  | 286906/400000 [00:35<00:13, 8399.42it/s] 72%|███████▏  | 287792/400000 [00:35<00:13, 8530.08it/s] 72%|███████▏  | 288729/400000 [00:35<00:12, 8763.91it/s] 72%|███████▏  | 289673/400000 [00:35<00:12, 8956.19it/s] 73%|███████▎  | 290572/400000 [00:35<00:12, 8743.40it/s] 73%|███████▎  | 291450/400000 [00:35<00:12, 8386.19it/s] 73%|███████▎  | 292294/400000 [00:35<00:12, 8307.18it/s] 73%|███████▎  | 293184/400000 [00:35<00:12, 8475.19it/s] 74%|███████▎  | 294105/400000 [00:35<00:12, 8680.60it/s] 74%|███████▍  | 295016/400000 [00:35<00:11, 8803.45it/s] 74%|███████▍  | 295931/400000 [00:36<00:11, 8903.74it/s] 74%|███████▍  | 296824/400000 [00:36<00:11, 8818.20it/s] 74%|███████▍  | 297709/400000 [00:36<00:11, 8827.42it/s] 75%|███████▍  | 298628/400000 [00:36<00:11, 8931.17it/s] 75%|███████▍  | 299523/400000 [00:36<00:11, 8787.46it/s] 75%|███████▌  | 300404/400000 [00:36<00:11, 8461.94it/s] 75%|███████▌  | 301254/400000 [00:36<00:12, 8220.85it/s] 76%|███████▌  | 302081/400000 [00:36<00:12, 7867.85it/s] 76%|███████▌  | 302874/400000 [00:36<00:12, 7671.12it/s] 76%|███████▌  | 303647/400000 [00:37<00:12, 7649.49it/s] 76%|███████▌  | 304416/400000 [00:37<00:12, 7587.98it/s] 76%|███████▋  | 305285/400000 [00:37<00:12, 7886.87it/s] 77%|███████▋  | 306182/400000 [00:37<00:11, 8183.30it/s] 77%|███████▋  | 307099/400000 [00:37<00:10, 8453.88it/s] 77%|███████▋  | 307988/400000 [00:37<00:10, 8577.57it/s] 77%|███████▋  | 308851/400000 [00:37<00:11, 8266.98it/s] 77%|███████▋  | 309684/400000 [00:37<00:11, 8048.67it/s] 78%|███████▊  | 310495/400000 [00:37<00:11, 8034.72it/s] 78%|███████▊  | 311312/400000 [00:37<00:10, 8071.85it/s] 78%|███████▊  | 312152/400000 [00:38<00:10, 8165.33it/s] 78%|███████▊  | 312971/400000 [00:38<00:10, 8075.02it/s] 78%|███████▊  | 313880/400000 [00:38<00:10, 8352.90it/s] 79%|███████▊  | 314789/400000 [00:38<00:09, 8559.72it/s] 79%|███████▉  | 315723/400000 [00:38<00:09, 8779.16it/s] 79%|███████▉  | 316620/400000 [00:38<00:09, 8834.46it/s] 79%|███████▉  | 317507/400000 [00:38<00:09, 8769.58it/s] 80%|███████▉  | 318387/400000 [00:38<00:09, 8525.78it/s] 80%|███████▉  | 319243/400000 [00:38<00:09, 8390.07it/s] 80%|████████  | 320165/400000 [00:38<00:09, 8622.61it/s] 80%|████████  | 321064/400000 [00:39<00:09, 8727.85it/s] 80%|████████  | 321940/400000 [00:39<00:09, 8661.07it/s] 81%|████████  | 322848/400000 [00:39<00:08, 8782.37it/s] 81%|████████  | 323761/400000 [00:39<00:08, 8883.16it/s] 81%|████████  | 324718/400000 [00:39<00:08, 9076.81it/s] 81%|████████▏ | 325633/400000 [00:39<00:08, 9097.33it/s] 82%|████████▏ | 326545/400000 [00:39<00:08, 8737.45it/s] 82%|████████▏ | 327423/400000 [00:39<00:08, 8603.48it/s] 82%|████████▏ | 328287/400000 [00:39<00:08, 8573.94it/s] 82%|████████▏ | 329147/400000 [00:40<00:08, 8483.12it/s] 82%|████████▏ | 329998/400000 [00:40<00:08, 8338.55it/s] 83%|████████▎ | 330834/400000 [00:40<00:08, 8144.96it/s] 83%|████████▎ | 331651/400000 [00:40<00:08, 7852.62it/s] 83%|████████▎ | 332441/400000 [00:40<00:08, 7676.97it/s] 83%|████████▎ | 333213/400000 [00:40<00:08, 7579.57it/s] 84%|████████▎ | 334023/400000 [00:40<00:08, 7727.26it/s] 84%|████████▎ | 334824/400000 [00:40<00:08, 7808.93it/s] 84%|████████▍ | 335636/400000 [00:40<00:08, 7897.60it/s] 84%|████████▍ | 336502/400000 [00:40<00:07, 8109.90it/s] 84%|████████▍ | 337362/400000 [00:41<00:07, 8249.20it/s] 85%|████████▍ | 338227/400000 [00:41<00:07, 8363.35it/s] 85%|████████▍ | 339066/400000 [00:41<00:07, 8311.13it/s] 85%|████████▍ | 339899/400000 [00:41<00:07, 8148.88it/s] 85%|████████▌ | 340716/400000 [00:41<00:07, 8148.73it/s] 85%|████████▌ | 341533/400000 [00:41<00:07, 8121.97it/s] 86%|████████▌ | 342454/400000 [00:41<00:06, 8418.21it/s] 86%|████████▌ | 343300/400000 [00:41<00:06, 8315.61it/s] 86%|████████▌ | 344191/400000 [00:41<00:06, 8484.90it/s] 86%|████████▋ | 345093/400000 [00:41<00:06, 8638.45it/s] 86%|████████▋ | 345992/400000 [00:42<00:06, 8739.63it/s] 87%|████████▋ | 346921/400000 [00:42<00:05, 8896.58it/s] 87%|████████▋ | 347813/400000 [00:42<00:05, 8784.93it/s] 87%|████████▋ | 348694/400000 [00:42<00:06, 8509.85it/s] 87%|████████▋ | 349549/400000 [00:42<00:06, 8120.40it/s] 88%|████████▊ | 350367/400000 [00:42<00:06, 8087.26it/s] 88%|████████▊ | 351233/400000 [00:42<00:05, 8248.01it/s] 88%|████████▊ | 352062/400000 [00:42<00:05, 8222.38it/s] 88%|████████▊ | 352887/400000 [00:42<00:05, 8198.80it/s] 88%|████████▊ | 353763/400000 [00:43<00:05, 8359.28it/s] 89%|████████▊ | 354652/400000 [00:43<00:05, 8509.67it/s] 89%|████████▉ | 355546/400000 [00:43<00:05, 8633.44it/s] 89%|████████▉ | 356412/400000 [00:43<00:05, 8571.12it/s] 89%|████████▉ | 357285/400000 [00:43<00:04, 8617.20it/s] 90%|████████▉ | 358148/400000 [00:43<00:05, 8369.26it/s] 90%|████████▉ | 358988/400000 [00:43<00:04, 8230.97it/s] 90%|████████▉ | 359814/400000 [00:43<00:04, 8145.89it/s] 90%|█████████ | 360654/400000 [00:43<00:04, 8219.53it/s] 90%|█████████ | 361525/400000 [00:43<00:04, 8359.06it/s] 91%|█████████ | 362411/400000 [00:44<00:04, 8503.22it/s] 91%|█████████ | 363331/400000 [00:44<00:04, 8699.88it/s] 91%|█████████ | 364257/400000 [00:44<00:04, 8858.63it/s] 91%|█████████▏| 365146/400000 [00:44<00:03, 8822.42it/s] 92%|█████████▏| 366030/400000 [00:44<00:03, 8677.78it/s] 92%|█████████▏| 366900/400000 [00:44<00:03, 8460.89it/s] 92%|█████████▏| 367749/400000 [00:44<00:03, 8331.82it/s] 92%|█████████▏| 368600/400000 [00:44<00:03, 8383.31it/s] 92%|█████████▏| 369440/400000 [00:44<00:03, 8355.89it/s] 93%|█████████▎| 370277/400000 [00:44<00:03, 8322.57it/s] 93%|█████████▎| 371119/400000 [00:45<00:03, 8349.98it/s] 93%|█████████▎| 372014/400000 [00:45<00:03, 8520.44it/s] 93%|█████████▎| 372938/400000 [00:45<00:03, 8720.89it/s] 93%|█████████▎| 373813/400000 [00:45<00:03, 8723.00it/s] 94%|█████████▎| 374687/400000 [00:45<00:02, 8722.84it/s] 94%|█████████▍| 375561/400000 [00:45<00:02, 8422.92it/s] 94%|█████████▍| 376407/400000 [00:45<00:02, 8309.04it/s] 94%|█████████▍| 377241/400000 [00:45<00:02, 8221.11it/s] 95%|█████████▍| 378099/400000 [00:45<00:02, 8323.59it/s] 95%|█████████▍| 378969/400000 [00:45<00:02, 8432.57it/s] 95%|█████████▍| 379847/400000 [00:46<00:02, 8533.75it/s] 95%|█████████▌| 380766/400000 [00:46<00:02, 8719.21it/s] 95%|█████████▌| 381666/400000 [00:46<00:02, 8799.59it/s] 96%|█████████▌| 382563/400000 [00:46<00:01, 8847.38it/s] 96%|█████████▌| 383486/400000 [00:46<00:01, 8957.00it/s] 96%|█████████▌| 384383/400000 [00:46<00:01, 8720.19it/s] 96%|█████████▋| 385258/400000 [00:46<00:01, 8516.99it/s] 97%|█████████▋| 386113/400000 [00:46<00:01, 8318.23it/s] 97%|█████████▋| 387037/400000 [00:46<00:01, 8574.58it/s] 97%|█████████▋| 387899/400000 [00:47<00:01, 8536.82it/s] 97%|█████████▋| 388756/400000 [00:47<00:01, 8420.95it/s] 97%|█████████▋| 389618/400000 [00:47<00:01, 8477.98it/s] 98%|█████████▊| 390517/400000 [00:47<00:01, 8624.87it/s] 98%|█████████▊| 391382/400000 [00:47<00:01, 8548.20it/s] 98%|█████████▊| 392246/400000 [00:47<00:00, 8573.68it/s] 98%|█████████▊| 393105/400000 [00:47<00:00, 8391.59it/s] 98%|█████████▊| 393946/400000 [00:47<00:00, 8286.41it/s] 99%|█████████▊| 394777/400000 [00:47<00:00, 8156.44it/s] 99%|█████████▉| 395651/400000 [00:47<00:00, 8321.82it/s] 99%|█████████▉| 396493/400000 [00:48<00:00, 8350.62it/s] 99%|█████████▉| 397341/400000 [00:48<00:00, 8386.53it/s]100%|█████████▉| 398250/400000 [00:48<00:00, 8585.32it/s]100%|█████████▉| 399150/400000 [00:48<00:00, 8704.64it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8259.62it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f74aca89b70> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011090886354105326 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.011133942915045696 	 Accuracy: 53

  model saves at 53% accuracy 

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

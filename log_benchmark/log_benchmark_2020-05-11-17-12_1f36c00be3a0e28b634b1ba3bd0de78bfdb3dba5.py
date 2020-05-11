
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0e077c0f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 17:12:35.612749
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 17:12:35.616998
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 17:12:35.620519
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 17:12:35.623866
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0e13584438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352074.2188
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 178504.9219
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 78725.9766
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 34323.3867
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 16964.8105
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 9593.0781
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 6128.8809
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 4282.9697
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 3232.5796
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 2600.5452

  #### Inference Need return ypred, ytrue ######################### 
[[  2.6420853    3.2389045   -1.5286205    0.10484135   0.5628351
   -0.07184212  -0.05563563  -0.36518428   1.4056103    1.6672821
   -1.6092896    0.15578803  -2.9304168   -1.4166489    1.7711241
    1.5958791   -1.444857     0.6447345   -1.0907208    1.4911196
    0.5537987   -3.5849843   -0.44223738   3.2005289    2.0567462
   -2.6834369   -1.6201692    1.3616265    1.7235699    2.7606666
   -0.9909804   -0.43173045   2.2769089   -2.233349     2.9808302
    1.3500209   -1.2062006   -0.02537304   2.291448     0.150619
   -2.5081282    0.02666819   1.6132096   -0.6400051   -1.8437544
   -1.2467027    3.6955085   -2.9080124    3.2405236    2.0015485
   -1.150697     0.10984586   0.6263119    1.205714    -1.2601436
   -1.7546532    3.251649     3.302941     0.03870085  -1.8565313
    0.51297563  12.552703    15.135947    12.391588    17.007727
   14.855665    14.621284    13.190419    16.306738    17.121002
   15.205028    12.898135    16.055063    15.6991205   15.8317
   14.210219    14.348869    15.584974    13.810059    15.389631
   15.308458    12.498077    15.354676    13.938111    19.988497
   13.057723    16.156872    14.055013    13.35586     15.559193
   13.59886     14.107607    13.022148    15.634484    18.143734
   17.352884    16.648169    13.562279    15.409057    15.417495
   16.244972    17.722157    14.48764     16.192387    17.502369
   16.288391    16.221714    15.290936    14.465443    16.949507
   14.436603    13.674803    15.934347    16.495125    15.587247
   13.361811    14.175467    16.687574    16.06679     13.274758
   -2.43464      2.6433377    0.79694366  -0.02865943  -1.1852384
    0.06383276  -0.3311454   -0.24601719   0.19898069   0.26181483
   -0.7049034   -2.6966414   -0.18062076  -0.6163222    1.9816909
   -0.34434843  -2.6641612   -0.55314696  -4.027796     0.43475497
    3.482811     2.5882993    1.0688006    1.6539364   -0.80499256
    1.4106436    1.1918347    0.03112414   0.4092294    1.7349962
    0.76417685  -1.5866029   -1.1457943    0.9882009    1.7613711
    0.5156045   -2.8209262    1.3349335    1.4082265   -0.250002
   -2.9055295   -0.742109    -1.8077941    0.62780094   1.5707324
   -1.0806314   -0.34225276   3.4458604   -0.17341858  -0.67025054
    0.49468037   0.54432386  -0.02056605   0.56291634   1.0323617
   -0.38728467   1.0221863    0.11912361   2.4196885   -0.82472086
    2.3649201    0.13675559   1.6026189    1.5960683    1.9875894
    3.437334     1.493815     3.5649486    0.0980137    0.05420709
    1.4510548    0.29498965   3.5898428    1.5579457    0.33511174
    0.5377561    0.41822565   1.2117941    0.6944367    1.9690238
    0.82714844   0.12953615   0.08269554   0.7597864    1.0516415
    0.510807     0.41115326   1.5947728    0.07736969   0.44421917
    1.6973572    0.8812336    0.52436686   1.1280565    0.9408121
    2.1454225    1.6934413    1.2835621    0.05605656   1.5543272
    0.18348289   1.5188838    0.7744642    1.0379784    4.5494795
    1.0660498    0.5309612    0.59842783   1.8605192    0.02443969
    0.37100935   0.32835835   0.58642125   0.35848027   0.38490796
    2.8610187    2.118646     3.271421     2.2063022    0.5174849
    3.209116    16.038069    13.736328    13.595464    15.8779125
   14.315812    12.97093     15.678409    17.237278    14.162696
   16.725758    14.784759    14.736646    12.728709    13.486256
   14.17672     12.037679    14.31908     13.405048    14.517712
   12.528378    14.043195    13.585594    13.454086    16.869558
   14.856501    14.657564    14.8408575   13.804031    14.512433
   14.088242    12.931133    16.495306    11.541682    17.859732
   13.046447    14.643464    13.9985075   15.350765    13.764078
   15.384201    16.294369    16.0084      19.262531    16.310396
   14.982249    17.637836    13.644885    15.25537     13.004738
   15.786436    12.608108    17.411896    12.160578    13.481767
   14.7176075   17.955397    14.482906    17.398441    14.298221
    1.6037099    1.5270674    0.563222     2.1596332    0.9551212
    0.09514254   1.877058     1.0318835    4.5286965    1.3858027
    2.9754708    0.1022228    1.0092523    0.5972357    0.28242588
    2.8091946    0.5745912    1.5034422    0.23537588   0.75228643
    0.3023374    2.8070717    0.6451745    0.7810515    1.1552953
    2.2247586    1.2407525    2.9674282    1.2681997    0.5982044
    0.33829397   0.13058525   0.7985044    0.45132887   0.03166384
    0.5443186    0.5553644    1.8459085    1.4293668    0.4575668
    2.8104668    2.5094266    0.23949337   0.18478209   3.592276
    1.4533331    1.8328187    0.72541165   2.068769     2.1932626
    3.1869836    0.17996424   0.09440744   1.9362917    0.7101946
    0.26026052   0.36205685   0.75406045   0.7853627    3.4225993
  -16.070244     3.9460356  -18.028269  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 17:12:45.666357
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   87.1315
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 17:12:45.677741
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   7630.84
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 17:12:45.681933
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   86.7635
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 17:12:45.686002
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -682.396
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139697889882008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139697199635928
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139697199636432
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139697199211016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139697199211520
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139697199212024

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f0e0f407ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.484003
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.449618
grad_step = 000002, loss = 0.425365
grad_step = 000003, loss = 0.401553
grad_step = 000004, loss = 0.375602
grad_step = 000005, loss = 0.350504
grad_step = 000006, loss = 0.333463
grad_step = 000007, loss = 0.325540
grad_step = 000008, loss = 0.323287
grad_step = 000009, loss = 0.312256
grad_step = 000010, loss = 0.299388
grad_step = 000011, loss = 0.290641
grad_step = 000012, loss = 0.284772
grad_step = 000013, loss = 0.279266
grad_step = 000014, loss = 0.272708
grad_step = 000015, loss = 0.264894
grad_step = 000016, loss = 0.256421
grad_step = 000017, loss = 0.248217
grad_step = 000018, loss = 0.240957
grad_step = 000019, loss = 0.234585
grad_step = 000020, loss = 0.228675
grad_step = 000021, loss = 0.222871
grad_step = 000022, loss = 0.216708
grad_step = 000023, loss = 0.209999
grad_step = 000024, loss = 0.203295
grad_step = 000025, loss = 0.197248
grad_step = 000026, loss = 0.191973
grad_step = 000027, loss = 0.187108
grad_step = 000028, loss = 0.182190
grad_step = 000029, loss = 0.176958
grad_step = 000030, loss = 0.171522
grad_step = 000031, loss = 0.166277
grad_step = 000032, loss = 0.161539
grad_step = 000033, loss = 0.157231
grad_step = 000034, loss = 0.152948
grad_step = 000035, loss = 0.148457
grad_step = 000036, loss = 0.143992
grad_step = 000037, loss = 0.139860
grad_step = 000038, loss = 0.136031
grad_step = 000039, loss = 0.132319
grad_step = 000040, loss = 0.128648
grad_step = 000041, loss = 0.125014
grad_step = 000042, loss = 0.121429
grad_step = 000043, loss = 0.117952
grad_step = 000044, loss = 0.114677
grad_step = 000045, loss = 0.111612
grad_step = 000046, loss = 0.108636
grad_step = 000047, loss = 0.105635
grad_step = 000048, loss = 0.102687
grad_step = 000049, loss = 0.099915
grad_step = 000050, loss = 0.097295
grad_step = 000051, loss = 0.094730
grad_step = 000052, loss = 0.092195
grad_step = 000053, loss = 0.089736
grad_step = 000054, loss = 0.087376
grad_step = 000055, loss = 0.085113
grad_step = 000056, loss = 0.082939
grad_step = 000057, loss = 0.080820
grad_step = 000058, loss = 0.078731
grad_step = 000059, loss = 0.076706
grad_step = 000060, loss = 0.074784
grad_step = 000061, loss = 0.072930
grad_step = 000062, loss = 0.071099
grad_step = 000063, loss = 0.069304
grad_step = 000064, loss = 0.067580
grad_step = 000065, loss = 0.065930
grad_step = 000066, loss = 0.064330
grad_step = 000067, loss = 0.062762
grad_step = 000068, loss = 0.061225
grad_step = 000069, loss = 0.059742
grad_step = 000070, loss = 0.058317
grad_step = 000071, loss = 0.056930
grad_step = 000072, loss = 0.055568
grad_step = 000073, loss = 0.054249
grad_step = 000074, loss = 0.052981
grad_step = 000075, loss = 0.051745
grad_step = 000076, loss = 0.050534
grad_step = 000077, loss = 0.049354
grad_step = 000078, loss = 0.048215
grad_step = 000079, loss = 0.047111
grad_step = 000080, loss = 0.046034
grad_step = 000081, loss = 0.044980
grad_step = 000082, loss = 0.043958
grad_step = 000083, loss = 0.042966
grad_step = 000084, loss = 0.041997
grad_step = 000085, loss = 0.041050
grad_step = 000086, loss = 0.040131
grad_step = 000087, loss = 0.039237
grad_step = 000088, loss = 0.038363
grad_step = 000089, loss = 0.037510
grad_step = 000090, loss = 0.036678
grad_step = 000091, loss = 0.035869
grad_step = 000092, loss = 0.035079
grad_step = 000093, loss = 0.034306
grad_step = 000094, loss = 0.033552
grad_step = 000095, loss = 0.032817
grad_step = 000096, loss = 0.032098
grad_step = 000097, loss = 0.031396
grad_step = 000098, loss = 0.030710
grad_step = 000099, loss = 0.030040
grad_step = 000100, loss = 0.029384
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.028744
grad_step = 000102, loss = 0.028117
grad_step = 000103, loss = 0.027505
grad_step = 000104, loss = 0.026906
grad_step = 000105, loss = 0.026321
grad_step = 000106, loss = 0.025748
grad_step = 000107, loss = 0.025186
grad_step = 000108, loss = 0.024638
grad_step = 000109, loss = 0.024101
grad_step = 000110, loss = 0.023575
grad_step = 000111, loss = 0.023057
grad_step = 000112, loss = 0.022554
grad_step = 000113, loss = 0.022063
grad_step = 000114, loss = 0.021582
grad_step = 000115, loss = 0.021112
grad_step = 000116, loss = 0.020653
grad_step = 000117, loss = 0.020203
grad_step = 000118, loss = 0.019763
grad_step = 000119, loss = 0.019333
grad_step = 000120, loss = 0.018910
grad_step = 000121, loss = 0.018496
grad_step = 000122, loss = 0.018091
grad_step = 000123, loss = 0.017694
grad_step = 000124, loss = 0.017304
grad_step = 000125, loss = 0.016923
grad_step = 000126, loss = 0.016549
grad_step = 000127, loss = 0.016185
grad_step = 000128, loss = 0.015828
grad_step = 000129, loss = 0.015479
grad_step = 000130, loss = 0.015137
grad_step = 000131, loss = 0.014804
grad_step = 000132, loss = 0.014478
grad_step = 000133, loss = 0.014160
grad_step = 000134, loss = 0.013849
grad_step = 000135, loss = 0.013545
grad_step = 000136, loss = 0.013248
grad_step = 000137, loss = 0.012957
grad_step = 000138, loss = 0.012673
grad_step = 000139, loss = 0.012395
grad_step = 000140, loss = 0.012124
grad_step = 000141, loss = 0.011857
grad_step = 000142, loss = 0.011596
grad_step = 000143, loss = 0.011343
grad_step = 000144, loss = 0.011093
grad_step = 000145, loss = 0.010849
grad_step = 000146, loss = 0.010612
grad_step = 000147, loss = 0.010378
grad_step = 000148, loss = 0.010151
grad_step = 000149, loss = 0.009929
grad_step = 000150, loss = 0.009710
grad_step = 000151, loss = 0.009497
grad_step = 000152, loss = 0.009290
grad_step = 000153, loss = 0.009085
grad_step = 000154, loss = 0.008886
grad_step = 000155, loss = 0.008692
grad_step = 000156, loss = 0.008502
grad_step = 000157, loss = 0.008316
grad_step = 000158, loss = 0.008135
grad_step = 000159, loss = 0.007958
grad_step = 000160, loss = 0.007785
grad_step = 000161, loss = 0.007624
grad_step = 000162, loss = 0.007461
grad_step = 000163, loss = 0.007291
grad_step = 000164, loss = 0.007134
grad_step = 000165, loss = 0.006986
grad_step = 000166, loss = 0.006832
grad_step = 000167, loss = 0.006683
grad_step = 000168, loss = 0.006545
grad_step = 000169, loss = 0.006407
grad_step = 000170, loss = 0.006267
grad_step = 000171, loss = 0.006137
grad_step = 000172, loss = 0.006010
grad_step = 000173, loss = 0.005882
grad_step = 000174, loss = 0.005760
grad_step = 000175, loss = 0.005643
grad_step = 000176, loss = 0.005526
grad_step = 000177, loss = 0.005412
grad_step = 000178, loss = 0.005304
grad_step = 000179, loss = 0.005197
grad_step = 000180, loss = 0.005092
grad_step = 000181, loss = 0.004991
grad_step = 000182, loss = 0.004895
grad_step = 000183, loss = 0.004798
grad_step = 000184, loss = 0.004705
grad_step = 000185, loss = 0.004615
grad_step = 000186, loss = 0.004528
grad_step = 000187, loss = 0.004443
grad_step = 000188, loss = 0.004360
grad_step = 000189, loss = 0.004280
grad_step = 000190, loss = 0.004203
grad_step = 000191, loss = 0.004128
grad_step = 000192, loss = 0.004055
grad_step = 000193, loss = 0.003983
grad_step = 000194, loss = 0.003914
grad_step = 000195, loss = 0.003848
grad_step = 000196, loss = 0.003784
grad_step = 000197, loss = 0.003722
grad_step = 000198, loss = 0.003662
grad_step = 000199, loss = 0.003603
grad_step = 000200, loss = 0.003547
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.003492
grad_step = 000202, loss = 0.003439
grad_step = 000203, loss = 0.003388
grad_step = 000204, loss = 0.003338
grad_step = 000205, loss = 0.003291
grad_step = 000206, loss = 0.003245
grad_step = 000207, loss = 0.003201
grad_step = 000208, loss = 0.003160
grad_step = 000209, loss = 0.003121
grad_step = 000210, loss = 0.003087
grad_step = 000211, loss = 0.003054
grad_step = 000212, loss = 0.003022
grad_step = 000213, loss = 0.002982
grad_step = 000214, loss = 0.002937
grad_step = 000215, loss = 0.002897
grad_step = 000216, loss = 0.002867
grad_step = 000217, loss = 0.002843
grad_step = 000218, loss = 0.002816
grad_step = 000219, loss = 0.002783
grad_step = 000220, loss = 0.002750
grad_step = 000221, loss = 0.002723
grad_step = 000222, loss = 0.002701
grad_step = 000223, loss = 0.002680
grad_step = 000224, loss = 0.002656
grad_step = 000225, loss = 0.002629
grad_step = 000226, loss = 0.002605
grad_step = 000227, loss = 0.002586
grad_step = 000228, loss = 0.002569
grad_step = 000229, loss = 0.002551
grad_step = 000230, loss = 0.002532
grad_step = 000231, loss = 0.002512
grad_step = 000232, loss = 0.002494
grad_step = 000233, loss = 0.002478
grad_step = 000234, loss = 0.002464
grad_step = 000235, loss = 0.002451
grad_step = 000236, loss = 0.002437
grad_step = 000237, loss = 0.002423
grad_step = 000238, loss = 0.002409
grad_step = 000239, loss = 0.002396
grad_step = 000240, loss = 0.002383
grad_step = 000241, loss = 0.002372
grad_step = 000242, loss = 0.002361
grad_step = 000243, loss = 0.002351
grad_step = 000244, loss = 0.002342
grad_step = 000245, loss = 0.002334
grad_step = 000246, loss = 0.002327
grad_step = 000247, loss = 0.002321
grad_step = 000248, loss = 0.002317
grad_step = 000249, loss = 0.002316
grad_step = 000250, loss = 0.002315
grad_step = 000251, loss = 0.002316
grad_step = 000252, loss = 0.002311
grad_step = 000253, loss = 0.002300
grad_step = 000254, loss = 0.002280
grad_step = 000255, loss = 0.002262
grad_step = 000256, loss = 0.002252
grad_step = 000257, loss = 0.002251
grad_step = 000258, loss = 0.002255
grad_step = 000259, loss = 0.002255
grad_step = 000260, loss = 0.002252
grad_step = 000261, loss = 0.002241
grad_step = 000262, loss = 0.002229
grad_step = 000263, loss = 0.002219
grad_step = 000264, loss = 0.002215
grad_step = 000265, loss = 0.002214
grad_step = 000266, loss = 0.002215
grad_step = 000267, loss = 0.002215
grad_step = 000268, loss = 0.002212
grad_step = 000269, loss = 0.002206
grad_step = 000270, loss = 0.002200
grad_step = 000271, loss = 0.002193
grad_step = 000272, loss = 0.002188
grad_step = 000273, loss = 0.002184
grad_step = 000274, loss = 0.002182
grad_step = 000275, loss = 0.002180
grad_step = 000276, loss = 0.002180
grad_step = 000277, loss = 0.002182
grad_step = 000278, loss = 0.002185
grad_step = 000279, loss = 0.002190
grad_step = 000280, loss = 0.002197
grad_step = 000281, loss = 0.002200
grad_step = 000282, loss = 0.002202
grad_step = 000283, loss = 0.002193
grad_step = 000284, loss = 0.002181
grad_step = 000285, loss = 0.002166
grad_step = 000286, loss = 0.002155
grad_step = 000287, loss = 0.002150
grad_step = 000288, loss = 0.002152
grad_step = 000289, loss = 0.002157
grad_step = 000290, loss = 0.002162
grad_step = 000291, loss = 0.002165
grad_step = 000292, loss = 0.002163
grad_step = 000293, loss = 0.002157
grad_step = 000294, loss = 0.002149
grad_step = 000295, loss = 0.002141
grad_step = 000296, loss = 0.002136
grad_step = 000297, loss = 0.002133
grad_step = 000298, loss = 0.002131
grad_step = 000299, loss = 0.002129
grad_step = 000300, loss = 0.002129
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002130
grad_step = 000302, loss = 0.002133
grad_step = 000303, loss = 0.002139
grad_step = 000304, loss = 0.002152
grad_step = 000305, loss = 0.002170
grad_step = 000306, loss = 0.002198
grad_step = 000307, loss = 0.002214
grad_step = 000308, loss = 0.002219
grad_step = 000309, loss = 0.002186
grad_step = 000310, loss = 0.002144
grad_step = 000311, loss = 0.002115
grad_step = 000312, loss = 0.002118
grad_step = 000313, loss = 0.002141
grad_step = 000314, loss = 0.002156
grad_step = 000315, loss = 0.002152
grad_step = 000316, loss = 0.002127
grad_step = 000317, loss = 0.002108
grad_step = 000318, loss = 0.002107
grad_step = 000319, loss = 0.002118
grad_step = 000320, loss = 0.002130
grad_step = 000321, loss = 0.002128
grad_step = 000322, loss = 0.002118
grad_step = 000323, loss = 0.002104
grad_step = 000324, loss = 0.002098
grad_step = 000325, loss = 0.002101
grad_step = 000326, loss = 0.002108
grad_step = 000327, loss = 0.002112
grad_step = 000328, loss = 0.002111
grad_step = 000329, loss = 0.002104
grad_step = 000330, loss = 0.002097
grad_step = 000331, loss = 0.002092
grad_step = 000332, loss = 0.002091
grad_step = 000333, loss = 0.002093
grad_step = 000334, loss = 0.002096
grad_step = 000335, loss = 0.002098
grad_step = 000336, loss = 0.002100
grad_step = 000337, loss = 0.002098
grad_step = 000338, loss = 0.002096
grad_step = 000339, loss = 0.002090
grad_step = 000340, loss = 0.002086
grad_step = 000341, loss = 0.002084
grad_step = 000342, loss = 0.002084
grad_step = 000343, loss = 0.002086
grad_step = 000344, loss = 0.002091
grad_step = 000345, loss = 0.002097
grad_step = 000346, loss = 0.002106
grad_step = 000347, loss = 0.002115
grad_step = 000348, loss = 0.002122
grad_step = 000349, loss = 0.002127
grad_step = 000350, loss = 0.002123
grad_step = 000351, loss = 0.002112
grad_step = 000352, loss = 0.002096
grad_step = 000353, loss = 0.002082
grad_step = 000354, loss = 0.002075
grad_step = 000355, loss = 0.002075
grad_step = 000356, loss = 0.002082
grad_step = 000357, loss = 0.002090
grad_step = 000358, loss = 0.002097
grad_step = 000359, loss = 0.002095
grad_step = 000360, loss = 0.002090
grad_step = 000361, loss = 0.002080
grad_step = 000362, loss = 0.002073
grad_step = 000363, loss = 0.002068
grad_step = 000364, loss = 0.002068
grad_step = 000365, loss = 0.002072
grad_step = 000366, loss = 0.002077
grad_step = 000367, loss = 0.002082
grad_step = 000368, loss = 0.002086
grad_step = 000369, loss = 0.002088
grad_step = 000370, loss = 0.002090
grad_step = 000371, loss = 0.002087
grad_step = 000372, loss = 0.002082
grad_step = 000373, loss = 0.002074
grad_step = 000374, loss = 0.002067
grad_step = 000375, loss = 0.002062
grad_step = 000376, loss = 0.002060
grad_step = 000377, loss = 0.002059
grad_step = 000378, loss = 0.002060
grad_step = 000379, loss = 0.002063
grad_step = 000380, loss = 0.002069
grad_step = 000381, loss = 0.002074
grad_step = 000382, loss = 0.002080
grad_step = 000383, loss = 0.002081
grad_step = 000384, loss = 0.002083
grad_step = 000385, loss = 0.002078
grad_step = 000386, loss = 0.002071
grad_step = 000387, loss = 0.002062
grad_step = 000388, loss = 0.002055
grad_step = 000389, loss = 0.002051
grad_step = 000390, loss = 0.002049
grad_step = 000391, loss = 0.002049
grad_step = 000392, loss = 0.002049
grad_step = 000393, loss = 0.002048
grad_step = 000394, loss = 0.002046
grad_step = 000395, loss = 0.002044
grad_step = 000396, loss = 0.002043
grad_step = 000397, loss = 0.002043
grad_step = 000398, loss = 0.002044
grad_step = 000399, loss = 0.002045
grad_step = 000400, loss = 0.002047
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.002053
grad_step = 000402, loss = 0.002068
grad_step = 000403, loss = 0.002103
grad_step = 000404, loss = 0.002145
grad_step = 000405, loss = 0.002227
grad_step = 000406, loss = 0.002251
grad_step = 000407, loss = 0.002251
grad_step = 000408, loss = 0.002153
grad_step = 000409, loss = 0.002056
grad_step = 000410, loss = 0.002050
grad_step = 000411, loss = 0.002112
grad_step = 000412, loss = 0.002147
grad_step = 000413, loss = 0.002110
grad_step = 000414, loss = 0.002076
grad_step = 000415, loss = 0.002054
grad_step = 000416, loss = 0.002048
grad_step = 000417, loss = 0.002072
grad_step = 000418, loss = 0.002083
grad_step = 000419, loss = 0.002049
grad_step = 000420, loss = 0.002031
grad_step = 000421, loss = 0.002047
grad_step = 000422, loss = 0.002053
grad_step = 000423, loss = 0.002050
grad_step = 000424, loss = 0.002047
grad_step = 000425, loss = 0.002030
grad_step = 000426, loss = 0.002023
grad_step = 000427, loss = 0.002036
grad_step = 000428, loss = 0.002042
grad_step = 000429, loss = 0.002036
grad_step = 000430, loss = 0.002032
grad_step = 000431, loss = 0.002028
grad_step = 000432, loss = 0.002019
grad_step = 000433, loss = 0.002018
grad_step = 000434, loss = 0.002024
grad_step = 000435, loss = 0.002025
grad_step = 000436, loss = 0.002023
grad_step = 000437, loss = 0.002022
grad_step = 000438, loss = 0.002019
grad_step = 000439, loss = 0.002014
grad_step = 000440, loss = 0.002011
grad_step = 000441, loss = 0.002013
grad_step = 000442, loss = 0.002012
grad_step = 000443, loss = 0.002012
grad_step = 000444, loss = 0.002013
grad_step = 000445, loss = 0.002014
grad_step = 000446, loss = 0.002012
grad_step = 000447, loss = 0.002010
grad_step = 000448, loss = 0.002010
grad_step = 000449, loss = 0.002008
grad_step = 000450, loss = 0.002006
grad_step = 000451, loss = 0.002004
grad_step = 000452, loss = 0.002004
grad_step = 000453, loss = 0.002003
grad_step = 000454, loss = 0.002001
grad_step = 000455, loss = 0.002000
grad_step = 000456, loss = 0.002000
grad_step = 000457, loss = 0.002000
grad_step = 000458, loss = 0.002000
grad_step = 000459, loss = 0.002001
grad_step = 000460, loss = 0.002003
grad_step = 000461, loss = 0.002009
grad_step = 000462, loss = 0.002020
grad_step = 000463, loss = 0.002043
grad_step = 000464, loss = 0.002079
grad_step = 000465, loss = 0.002149
grad_step = 000466, loss = 0.002207
grad_step = 000467, loss = 0.002279
grad_step = 000468, loss = 0.002211
grad_step = 000469, loss = 0.002114
grad_step = 000470, loss = 0.002009
grad_step = 000471, loss = 0.001999
grad_step = 000472, loss = 0.002063
grad_step = 000473, loss = 0.002096
grad_step = 000474, loss = 0.002065
grad_step = 000475, loss = 0.001999
grad_step = 000476, loss = 0.001990
grad_step = 000477, loss = 0.002032
grad_step = 000478, loss = 0.002048
grad_step = 000479, loss = 0.002020
grad_step = 000480, loss = 0.001987
grad_step = 000481, loss = 0.001990
grad_step = 000482, loss = 0.002013
grad_step = 000483, loss = 0.002017
grad_step = 000484, loss = 0.001997
grad_step = 000485, loss = 0.001979
grad_step = 000486, loss = 0.001983
grad_step = 000487, loss = 0.001998
grad_step = 000488, loss = 0.002000
grad_step = 000489, loss = 0.001986
grad_step = 000490, loss = 0.001973
grad_step = 000491, loss = 0.001976
grad_step = 000492, loss = 0.001985
grad_step = 000493, loss = 0.001985
grad_step = 000494, loss = 0.001977
grad_step = 000495, loss = 0.001969
grad_step = 000496, loss = 0.001968
grad_step = 000497, loss = 0.001973
grad_step = 000498, loss = 0.001975
grad_step = 000499, loss = 0.001972
grad_step = 000500, loss = 0.001966
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001963
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

  date_run                              2020-05-11 17:13:08.645880
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.221723
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 17:13:08.653403
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.103274
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 17:13:08.661744
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.150032
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 17:13:08.668123
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.569289
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
0   2020-05-11 17:12:35.612749  ...    mean_absolute_error
1   2020-05-11 17:12:35.616998  ...     mean_squared_error
2   2020-05-11 17:12:35.620519  ...  median_absolute_error
3   2020-05-11 17:12:35.623866  ...               r2_score
4   2020-05-11 17:12:45.666357  ...    mean_absolute_error
5   2020-05-11 17:12:45.677741  ...     mean_squared_error
6   2020-05-11 17:12:45.681933  ...  median_absolute_error
7   2020-05-11 17:12:45.686002  ...               r2_score
8   2020-05-11 17:13:08.645880  ...    mean_absolute_error
9   2020-05-11 17:13:08.653403  ...     mean_squared_error
10  2020-05-11 17:13:08.661744  ...  median_absolute_error
11  2020-05-11 17:13:08.668123  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 317231.73it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 408760.41it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 565072.40it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 797949.19it/s] 77%|███████▋  | 7675904/9912422 [00:00<00:01, 1128211.52it/s]9920512it [00:00, 10014094.26it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 142390.14it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 305311.79it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 394491.78it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 545658.91it/s]1654784it [00:00, 2743023.61it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:01<?, ?it/s]8192it [00:01, 7958.31it/s]             >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff699a45fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff637162b00> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff6999d0ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff637162c50> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff63715f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff64c3c9e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff63715f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff64c3c9e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff699a45fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff640479630> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff699a0dcc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4ac82531d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d1fee24732e20d385f56665c70e0bd91c06c631e98bc44ac68038317abfe35be
  Stored in directory: /tmp/pip-ephem-wheel-cache-vzv15gm6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4a5fe3b160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 1:30
   57344/17464789 [..............................] - ETA: 1:18
  106496/17464789 [..............................] - ETA: 1:04
  180224/17464789 [..............................] - ETA: 46s 
  368640/17464789 [..............................] - ETA: 28s
  753664/17464789 [>.............................] - ETA: 16s
 1499136/17464789 [=>............................] - ETA: 8s 
 1687552/17464789 [=>............................] - ETA: 8s
 3031040/17464789 [====>.........................] - ETA: 4s
 4243456/17464789 [======>.......................] - ETA: 3s
 6078464/17464789 [=========>....................] - ETA: 2s
 7380992/17464789 [===========>..................] - ETA: 1s
 9125888/17464789 [==============>...............] - ETA: 1s
10444800/17464789 [================>.............] - ETA: 0s
12189696/17464789 [===================>..........] - ETA: 0s
13844480/17464789 [======================>.......] - ETA: 0s
15269888/17464789 [=========================>....] - ETA: 0s
16744448/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 2s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 17:14:42.560615: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 17:14:42.565250: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-11 17:14:42.565450: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55594602e1a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 17:14:42.565467: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 9s - loss: 7.3983 - accuracy: 0.5175 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5133 - accuracy: 0.5100
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5593 - accuracy: 0.5070
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7477 - accuracy: 0.4947
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7145 - accuracy: 0.4969
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7177 - accuracy: 0.4967
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7218 - accuracy: 0.4964
11000/25000 [============>.................] - ETA: 4s - loss: 7.6987 - accuracy: 0.4979
12000/25000 [=============>................] - ETA: 4s - loss: 7.7024 - accuracy: 0.4977
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6926 - accuracy: 0.4983
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 3s - loss: 7.6656 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6602 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6443 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6449 - accuracy: 0.5014
25000/25000 [==============================] - 9s 371us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 17:14:59.226689
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 17:14:59.226689  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 17:15:06.172375: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 17:15:06.178719: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-11 17:15:06.178866: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55619da70080 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 17:15:06.178884: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ffb5a0bdc18> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7334 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.5714 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ffb35e3e860> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8583 - accuracy: 0.4875
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7688 - accuracy: 0.4933 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6385 - accuracy: 0.5018
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6877 - accuracy: 0.4986
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6475 - accuracy: 0.5013
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6442 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 3s - loss: 7.6472 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6465 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6594 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6936 - accuracy: 0.4982
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6945 - accuracy: 0.4982
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6906 - accuracy: 0.4984
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ffaf0dca400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<13:09:32, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<9:13:34, 25.9kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.24M/862M [00:00<6:24:58, 37.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:00<4:26:36, 52.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:00<3:04:42, 75.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:00<2:07:50, 108kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.3M/862M [00:01<1:28:32, 154kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:01<1:01:19, 220kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:01<1:26:55, 155kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<1:01:05, 220kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<9:10:27, 24.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:03<6:24:05, 34.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<4:34:23, 48.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<3:12:58, 69.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:05<2:14:42, 98.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<1:44:13, 128kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<1:13:40, 181kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<53:23, 248kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<38:04, 347kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<28:35, 461kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:11<20:51, 631kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<16:32, 792kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<12:20, 1.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<10:36, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<08:31, 1.53MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<06:32, 1.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<5:59:08, 36.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:16<4:10:15, 51.7kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<3:22:38, 63.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:18<2:22:44, 90.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:18<1:39:34, 129kB/s] .vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<1:55:53, 111kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<1:21:42, 157kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<58:57, 217kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<41:47, 306kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<31:11, 408kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<22:23, 568kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<17:38, 717kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<12:54, 979kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<09:40, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<5:53:50, 35.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:27<4:06:32, 50.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<3:15:35, 64.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<2:17:24, 91.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<1:37:46, 128kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<1:09:02, 180kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<50:03, 248kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<35:30, 349kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<26:46, 460kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<19:23, 636kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<15:25, 795kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<11:24, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<08:34, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<5:39:13, 36.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:38<3:56:13, 51.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<16:33:46, 12.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<11:36:15, 17.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<8:06:43, 24.8kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<5:40:57, 35.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<3:59:40, 50.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<2:49:25, 70.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:44<1:58:23, 101kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<1:26:23, 138kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<1:01:29, 194kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<44:34, 266kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<32:11, 369kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<24:10, 489kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<17:49, 662kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<14:10, 828kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<11:01, 1.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<09:23, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<07:05, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:49, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:44, 2.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:42, 2.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:57, 2.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:09, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:14, 2.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:44, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:49, 2.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<03:09, 3.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<6:34:33, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:03<4:35:21, 41.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<3:15:50, 57.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<2:18:42, 81.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:05<1:36:42, 116kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<1:18:57, 142kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<56:10, 199kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<40:45, 273kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<29:27, 378kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<22:08, 500kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<16:24, 675kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<13:03, 843kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<10:04, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<08:38, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:58, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:15<04:56, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<5:56:23, 30.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<4:10:19, 43.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<2:55:47, 61.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<2:04:02, 87.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:19<1:26:28, 124kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<1:11:31, 150kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<51:06, 210kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:21<35:41, 299kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<38:30, 277kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<27:58, 381kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:23<19:34, 541kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<28:18, 374kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<21:53, 484kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<15:29, 682kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<13:05, 804kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<10:21, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<07:22, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<08:18, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:51, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<05:11, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<5:57:36, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<4:10:09, 41.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<2:55:44, 58.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:32<2:02:25, 83.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<1:45:47, 97.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<1:14:39, 137kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<53:30, 191kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<38:06, 267kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:38<28:03, 361kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<20:12, 501kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<15:37, 644kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<11:35, 868kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<09:35, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<07:08, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<06:35, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:20, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:22, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:30, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:42, 2.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:05, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:34, 2.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:56, 2.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:36, 2.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:52<02:35, 3.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<11:31, 831kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<09:01, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<07:39, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:47, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<5:36:20, 28.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<3:54:33, 40.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<2:46:44, 56.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<1:57:27, 79.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<1:23:03, 112kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<58:26, 159kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<42:11, 219kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<29:56, 309kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<22:17, 412kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<16:00, 573kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<12:36, 723kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<09:14, 986kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<07:52, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:14, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<05:41, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:27, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:35, 2.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:54, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:07, 2.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:35, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:13, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:32, 2.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<02:50, 3.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:23, 2.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<02:46, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:18, 2.59MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:41, 3.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<02:17, 3.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<4:42:39, 30.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<3:17:28, 42.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<2:18:50, 60.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<1:37:43, 85.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<1:08:45, 122kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<49:06, 169kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<34:53, 238kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<25:29, 323kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<18:17, 450kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<13:58, 585kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<10:07, 806kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<08:21, 969kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<06:09, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:37, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<04:22, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:17, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:17, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:37, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:48, 2.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:16, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:45, 2.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:07, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:41, 2.87MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:02, 2.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:34, 2.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:58, 2.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:42, 2.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<02:13, 3.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<4:11:07, 30.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<2:55:16, 42.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<2:03:13, 60.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:53<1:25:43, 86.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<1:06:31, 112kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<46:49, 159kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<33:43, 219kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<24:01, 306kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:57<16:44, 436kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<8:05:24, 15.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<5:39:49, 21.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<3:57:01, 30.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<2:45:58, 43.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<1:56:25, 61.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<1:21:51, 87.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<57:54, 122kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<40:57, 173kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<29:29, 238kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<21:02, 333kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<15:40, 444kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<11:29, 605kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<09:00, 765kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<06:52, 1.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:46, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<04:20, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<04:06, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:10, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<02:32, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<3:39:23, 30.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<2:32:57, 43.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<1:47:25, 61.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<1:15:29, 86.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<53:04, 123kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<37:51, 172kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<26:46, 242kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<19:35, 328kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<14:13, 451kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<10:48, 588kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<08:05, 785kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<06:32, 961kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<05:08, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:28, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:38, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:56, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:34, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:40, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:23, 2.52MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:15, 2.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:26, 2.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:00, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<01:39, 3.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<3:16:14, 29.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<2:16:34, 42.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<1:35:58, 59.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<1:07:18, 84.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<47:28, 120kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<33:43, 167kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<23:47, 236kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<17:25, 319kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<12:38, 439kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<09:33, 573kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<06:56, 788kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:41, 950kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<04:12, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:47, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:51, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:51, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:13, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:23, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:57, 2.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:10, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:48, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:02, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:44, 2.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:58, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:35, 3.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<01:21, 3.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<2:42:48, 30.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<1:53:05, 43.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<1:19:26, 61.4kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<55:38, 86.6kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<39:04, 123kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<27:48, 171kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<19:36, 242kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<14:20, 326kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<10:12, 457kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<07:48, 590kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<05:38, 814kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:38, 976kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:28, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:06, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:23, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:49, 2.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:59, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:34, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:46, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:25, 2.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:40, 2.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:23, 3.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:36, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:17, 3.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:04, 3.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<2:19:48, 29.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<1:36:51, 41.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<1:08:53, 58.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<48:37, 82.4kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<34:03, 116kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<24:02, 164kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<17:09, 226kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<12:09, 318kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<08:59, 423kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<06:28, 586kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<05:03, 739kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<03:47, 981kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:10, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<02:23, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:13, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:44, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:45, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:22, 2.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:30, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:12, 2.86MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:22, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:13, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:51<00:52, 3.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<14:56, 222kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<10:38, 311kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<07:48, 416kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<05:37, 576kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<04:02, 792kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:44:15, 30.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:56<1:11:36, 43.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<54:12, 57.9kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<38:06, 82.1kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<26:37, 115kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<18:45, 163kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<13:20, 225kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<09:26, 316kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<06:57, 421kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<04:59, 584kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<03:53, 734kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:50, 1.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:24, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:50, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:40, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:17, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:19, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:02, 2.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:07, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:54, 2.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:01, 2.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<11:34, 217kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<08:10, 300kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<05:44, 423kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<04:05, 586kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:25:17, 28.2kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<58:48, 40.2kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<41:18, 56.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<29:05, 80.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<20:08, 112kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<14:07, 159kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<10:00, 219kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<07:09, 306kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<05:11, 410kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<03:48, 557kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:53, 712kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:11, 937kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:46, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:25, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:14, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:58, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:50, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:49, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:44, 2.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:39, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:41, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:36, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:38, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:33, 2.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<00:26, 3.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<50:58, 30.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<34:17, 42.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<24:00, 60.5kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<16:18, 85.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<11:25, 121kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<07:50, 168kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<05:30, 238kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<03:53, 322kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<02:45, 450kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<02:02, 582kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:28, 797kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:09, 965kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:51, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:43, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:33, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:31, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:23, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:24, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:19, 2.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:21, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:17, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:18, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<00:11, 3.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<24:14, 29.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<15:27, 42.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<10:44, 60.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<06:53, 85.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<04:45, 121kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<03:05, 168kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<02:08, 237kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<01:23, 321kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:58, 449kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:39, 581kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:27, 800kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:19, 965kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:13, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:10, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:07, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.94MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 627/400000 [00:00<01:03, 6264.08it/s]  0%|          | 1310/400000 [00:00<01:02, 6423.47it/s]  0%|          | 1936/400000 [00:00<01:02, 6370.87it/s]  1%|          | 2549/400000 [00:00<01:03, 6294.91it/s]  1%|          | 3154/400000 [00:00<01:03, 6219.25it/s]  1%|          | 3811/400000 [00:00<01:02, 6318.49it/s]  1%|          | 4468/400000 [00:00<01:01, 6391.50it/s]  1%|▏         | 5231/400000 [00:00<00:58, 6717.19it/s]  1%|▏         | 5954/400000 [00:00<00:57, 6859.99it/s]  2%|▏         | 6620/400000 [00:01<00:57, 6798.11it/s]  2%|▏         | 7349/400000 [00:01<00:56, 6936.32it/s]  2%|▏         | 8034/400000 [00:01<00:56, 6888.09it/s]  2%|▏         | 8755/400000 [00:01<00:56, 6979.41it/s]  2%|▏         | 9485/400000 [00:01<00:55, 7069.91it/s]  3%|▎         | 10191/400000 [00:01<00:55, 7064.66it/s]  3%|▎         | 10930/400000 [00:01<00:54, 7159.11it/s]  3%|▎         | 11648/400000 [00:01<00:54, 7163.74it/s]  3%|▎         | 12382/400000 [00:01<00:53, 7213.22it/s]  3%|▎         | 13116/400000 [00:01<00:53, 7250.45it/s]  3%|▎         | 13873/400000 [00:02<00:52, 7342.81it/s]  4%|▎         | 14622/400000 [00:02<00:52, 7384.34it/s]  4%|▍         | 15361/400000 [00:02<00:52, 7337.13it/s]  4%|▍         | 16095/400000 [00:02<00:53, 7152.61it/s]  4%|▍         | 16829/400000 [00:02<00:53, 7205.83it/s]  4%|▍         | 17592/400000 [00:02<00:52, 7326.32it/s]  5%|▍         | 18342/400000 [00:02<00:51, 7376.24it/s]  5%|▍         | 19089/400000 [00:02<00:51, 7401.72it/s]  5%|▍         | 19830/400000 [00:02<00:52, 7303.83it/s]  5%|▌         | 20562/400000 [00:02<00:55, 6846.68it/s]  5%|▌         | 21254/400000 [00:03<00:58, 6525.25it/s]  5%|▌         | 21990/400000 [00:03<00:55, 6753.98it/s]  6%|▌         | 22764/400000 [00:03<00:53, 7020.84it/s]  6%|▌         | 23509/400000 [00:03<00:52, 7142.52it/s]  6%|▌         | 24243/400000 [00:03<00:52, 7199.36it/s]  6%|▌         | 24968/400000 [00:03<00:54, 6877.28it/s]  6%|▋         | 25662/400000 [00:03<00:56, 6598.99it/s]  7%|▋         | 26329/400000 [00:03<00:58, 6425.87it/s]  7%|▋         | 26978/400000 [00:03<01:00, 6207.64it/s]  7%|▋         | 27675/400000 [00:04<00:58, 6417.26it/s]  7%|▋         | 28420/400000 [00:04<00:55, 6693.25it/s]  7%|▋         | 29156/400000 [00:04<00:53, 6879.23it/s]  7%|▋         | 29851/400000 [00:04<00:54, 6740.45it/s]  8%|▊         | 30537/400000 [00:04<00:54, 6774.67it/s]  8%|▊         | 31218/400000 [00:04<00:56, 6522.94it/s]  8%|▊         | 31875/400000 [00:04<00:57, 6407.23it/s]  8%|▊         | 32543/400000 [00:04<00:56, 6486.13it/s]  8%|▊         | 33220/400000 [00:04<00:55, 6566.33it/s]  8%|▊         | 33956/400000 [00:04<00:53, 6785.75it/s]  9%|▊         | 34704/400000 [00:05<00:52, 6979.36it/s]  9%|▉         | 35462/400000 [00:05<00:50, 7148.83it/s]  9%|▉         | 36181/400000 [00:05<00:51, 7112.94it/s]  9%|▉         | 36895/400000 [00:05<00:51, 7068.21it/s]  9%|▉         | 37604/400000 [00:05<00:51, 6986.46it/s] 10%|▉         | 38305/400000 [00:05<00:53, 6724.84it/s] 10%|▉         | 39029/400000 [00:05<00:52, 6870.36it/s] 10%|▉         | 39756/400000 [00:05<00:51, 6984.98it/s] 10%|█         | 40531/400000 [00:05<00:49, 7197.67it/s] 10%|█         | 41269/400000 [00:05<00:49, 7249.89it/s] 11%|█         | 42053/400000 [00:06<00:48, 7415.02it/s] 11%|█         | 42817/400000 [00:06<00:47, 7480.31it/s] 11%|█         | 43568/400000 [00:06<00:49, 7271.23it/s] 11%|█         | 44298/400000 [00:06<00:51, 6934.53it/s] 11%|█         | 44997/400000 [00:06<00:53, 6638.41it/s] 11%|█▏        | 45668/400000 [00:06<00:53, 6578.36it/s] 12%|█▏        | 46407/400000 [00:06<00:51, 6800.75it/s] 12%|█▏        | 47191/400000 [00:06<00:49, 7080.99it/s] 12%|█▏        | 47960/400000 [00:06<00:48, 7251.96it/s] 12%|█▏        | 48694/400000 [00:07<00:48, 7274.32it/s] 12%|█▏        | 49426/400000 [00:07<00:48, 7275.59it/s] 13%|█▎        | 50157/400000 [00:07<00:48, 7270.59it/s] 13%|█▎        | 50886/400000 [00:07<00:48, 7223.57it/s] 13%|█▎        | 51615/400000 [00:07<00:48, 7242.58it/s] 13%|█▎        | 52341/400000 [00:07<00:48, 7203.61it/s] 13%|█▎        | 53063/400000 [00:07<00:48, 7175.64it/s] 13%|█▎        | 53782/400000 [00:07<00:48, 7139.08it/s] 14%|█▎        | 54497/400000 [00:07<00:50, 6888.93it/s] 14%|█▍        | 55189/400000 [00:07<00:51, 6675.28it/s] 14%|█▍        | 55860/400000 [00:08<00:51, 6623.65it/s] 14%|█▍        | 56578/400000 [00:08<00:50, 6779.73it/s] 14%|█▍        | 57308/400000 [00:08<00:49, 6925.57it/s] 15%|█▍        | 58004/400000 [00:08<00:49, 6873.14it/s] 15%|█▍        | 58722/400000 [00:08<00:49, 6962.41it/s] 15%|█▍        | 59420/400000 [00:08<00:49, 6830.64it/s] 15%|█▌        | 60105/400000 [00:08<00:51, 6618.49it/s] 15%|█▌        | 60770/400000 [00:08<00:51, 6529.85it/s] 15%|█▌        | 61445/400000 [00:08<00:51, 6592.09it/s] 16%|█▌        | 62158/400000 [00:08<00:50, 6744.36it/s] 16%|█▌        | 62884/400000 [00:09<00:48, 6888.58it/s] 16%|█▌        | 63622/400000 [00:09<00:47, 7027.74it/s] 16%|█▌        | 64368/400000 [00:09<00:46, 7150.14it/s] 16%|█▋        | 65150/400000 [00:09<00:45, 7337.80it/s] 16%|█▋        | 65892/400000 [00:09<00:45, 7360.06it/s] 17%|█▋        | 66630/400000 [00:09<00:48, 6907.43it/s] 17%|█▋        | 67328/400000 [00:09<00:49, 6726.79it/s] 17%|█▋        | 68007/400000 [00:09<00:50, 6607.64it/s] 17%|█▋        | 68744/400000 [00:09<00:48, 6816.94it/s] 17%|█▋        | 69473/400000 [00:10<00:47, 6950.21it/s] 18%|█▊        | 70242/400000 [00:10<00:46, 7155.36it/s] 18%|█▊        | 70962/400000 [00:10<00:47, 6886.98it/s] 18%|█▊        | 71656/400000 [00:10<00:48, 6791.40it/s] 18%|█▊        | 72400/400000 [00:10<00:47, 6967.26it/s] 18%|█▊        | 73101/400000 [00:10<00:48, 6743.83it/s] 18%|█▊        | 73816/400000 [00:10<00:47, 6860.71it/s] 19%|█▊        | 74572/400000 [00:10<00:46, 7054.02it/s] 19%|█▉        | 75313/400000 [00:10<00:45, 7154.76it/s] 19%|█▉        | 76032/400000 [00:10<00:45, 7055.98it/s] 19%|█▉        | 76766/400000 [00:11<00:45, 7138.73it/s] 19%|█▉        | 77509/400000 [00:11<00:44, 7222.14it/s] 20%|█▉        | 78243/400000 [00:11<00:44, 7254.68it/s] 20%|█▉        | 78995/400000 [00:11<00:43, 7330.62it/s] 20%|█▉        | 79733/400000 [00:11<00:43, 7345.09it/s] 20%|██        | 80469/400000 [00:11<00:44, 7254.79it/s] 20%|██        | 81212/400000 [00:11<00:43, 7303.66it/s] 20%|██        | 81959/400000 [00:11<00:43, 7352.54it/s] 21%|██        | 82695/400000 [00:11<00:43, 7299.50it/s] 21%|██        | 83426/400000 [00:11<00:43, 7258.98it/s] 21%|██        | 84153/400000 [00:12<00:43, 7203.11it/s] 21%|██        | 84880/400000 [00:12<00:43, 7221.23it/s] 21%|██▏       | 85604/400000 [00:12<00:43, 7225.29it/s] 22%|██▏       | 86351/400000 [00:12<00:42, 7296.48it/s] 22%|██▏       | 87081/400000 [00:12<00:44, 7081.67it/s] 22%|██▏       | 87791/400000 [00:12<00:45, 6923.95it/s] 22%|██▏       | 88486/400000 [00:12<00:46, 6710.38it/s] 22%|██▏       | 89182/400000 [00:12<00:45, 6781.42it/s] 22%|██▏       | 89863/400000 [00:12<00:47, 6578.57it/s] 23%|██▎       | 90613/400000 [00:13<00:45, 6828.50it/s] 23%|██▎       | 91359/400000 [00:13<00:44, 7006.04it/s] 23%|██▎       | 92064/400000 [00:13<00:43, 7000.14it/s] 23%|██▎       | 92805/400000 [00:13<00:43, 7117.78it/s] 23%|██▎       | 93520/400000 [00:13<00:44, 6911.27it/s] 24%|██▎       | 94260/400000 [00:13<00:43, 7049.13it/s] 24%|██▎       | 94968/400000 [00:13<00:43, 7046.33it/s] 24%|██▍       | 95709/400000 [00:13<00:42, 7149.89it/s] 24%|██▍       | 96455/400000 [00:13<00:41, 7238.67it/s] 24%|██▍       | 97183/400000 [00:13<00:41, 7250.37it/s] 24%|██▍       | 97910/400000 [00:14<00:43, 6923.78it/s] 25%|██▍       | 98607/400000 [00:14<00:45, 6644.50it/s] 25%|██▍       | 99345/400000 [00:14<00:43, 6841.24it/s] 25%|██▌       | 100035/400000 [00:14<00:43, 6834.10it/s] 25%|██▌       | 100785/400000 [00:14<00:42, 7018.94it/s] 25%|██▌       | 101491/400000 [00:14<00:42, 6977.47it/s] 26%|██▌       | 102225/400000 [00:14<00:42, 7079.49it/s] 26%|██▌       | 102936/400000 [00:14<00:42, 7038.43it/s] 26%|██▌       | 103642/400000 [00:14<00:42, 7001.62it/s] 26%|██▌       | 104374/400000 [00:14<00:41, 7093.22it/s] 26%|██▋       | 105089/400000 [00:15<00:41, 7109.29it/s] 26%|██▋       | 105835/400000 [00:15<00:40, 7205.39it/s] 27%|██▋       | 106587/400000 [00:15<00:40, 7293.90it/s] 27%|██▋       | 107318/400000 [00:15<00:40, 7152.13it/s] 27%|██▋       | 108075/400000 [00:15<00:40, 7271.09it/s] 27%|██▋       | 108804/400000 [00:15<00:40, 7217.86it/s] 27%|██▋       | 109527/400000 [00:15<00:40, 7157.24it/s] 28%|██▊       | 110244/400000 [00:15<00:41, 6996.03it/s] 28%|██▊       | 110979/400000 [00:15<00:40, 7098.31it/s] 28%|██▊       | 111735/400000 [00:15<00:39, 7230.01it/s] 28%|██▊       | 112472/400000 [00:16<00:39, 7270.93it/s] 28%|██▊       | 113216/400000 [00:16<00:39, 7320.38it/s] 28%|██▊       | 113949/400000 [00:16<00:39, 7286.33it/s] 29%|██▊       | 114679/400000 [00:16<00:39, 7247.47it/s] 29%|██▉       | 115405/400000 [00:16<00:40, 7067.85it/s] 29%|██▉       | 116114/400000 [00:16<00:40, 6959.14it/s] 29%|██▉       | 116812/400000 [00:16<00:40, 6960.73it/s] 29%|██▉       | 117537/400000 [00:16<00:40, 7043.09it/s] 30%|██▉       | 118312/400000 [00:16<00:38, 7239.39it/s] 30%|██▉       | 119085/400000 [00:16<00:38, 7379.13it/s] 30%|██▉       | 119825/400000 [00:17<00:38, 7329.05it/s] 30%|███       | 120560/400000 [00:17<00:38, 7313.25it/s] 30%|███       | 121293/400000 [00:17<00:38, 7193.24it/s] 31%|███       | 122014/400000 [00:17<00:39, 7000.72it/s] 31%|███       | 122776/400000 [00:17<00:38, 7174.77it/s] 31%|███       | 123533/400000 [00:17<00:37, 7286.92it/s] 31%|███       | 124293/400000 [00:17<00:37, 7377.09it/s] 31%|███▏      | 125033/400000 [00:17<00:37, 7341.31it/s] 31%|███▏      | 125769/400000 [00:17<00:37, 7248.57it/s] 32%|███▏      | 126496/400000 [00:18<00:38, 7163.67it/s] 32%|███▏      | 127214/400000 [00:18<00:40, 6748.07it/s] 32%|███▏      | 127947/400000 [00:18<00:39, 6912.12it/s] 32%|███▏      | 128676/400000 [00:18<00:38, 7020.20it/s] 32%|███▏      | 129382/400000 [00:18<00:38, 6948.19it/s] 33%|███▎      | 130126/400000 [00:18<00:38, 7083.85it/s] 33%|███▎      | 130867/400000 [00:18<00:37, 7177.75it/s] 33%|███▎      | 131633/400000 [00:18<00:36, 7315.38it/s] 33%|███▎      | 132367/400000 [00:18<00:38, 6969.04it/s] 33%|███▎      | 133069/400000 [00:18<00:38, 6861.60it/s] 33%|███▎      | 133760/400000 [00:19<00:41, 6472.44it/s] 34%|███▎      | 134415/400000 [00:19<00:41, 6452.74it/s] 34%|███▍      | 135185/400000 [00:19<00:39, 6781.17it/s] 34%|███▍      | 135929/400000 [00:19<00:37, 6965.88it/s] 34%|███▍      | 136708/400000 [00:19<00:36, 7192.20it/s] 34%|███▍      | 137434/400000 [00:19<00:37, 7066.84it/s] 35%|███▍      | 138146/400000 [00:19<00:38, 6791.77it/s] 35%|███▍      | 138838/400000 [00:19<00:38, 6828.16it/s] 35%|███▍      | 139530/400000 [00:19<00:38, 6852.87it/s] 35%|███▌      | 140219/400000 [00:20<00:38, 6832.95it/s] 35%|███▌      | 140905/400000 [00:20<00:39, 6628.58it/s] 35%|███▌      | 141571/400000 [00:20<00:42, 6054.43it/s] 36%|███▌      | 142260/400000 [00:20<00:41, 6282.89it/s] 36%|███▌      | 142901/400000 [00:20<00:40, 6318.55it/s] 36%|███▌      | 143541/400000 [00:20<00:40, 6267.86it/s] 36%|███▌      | 144173/400000 [00:20<00:40, 6265.48it/s] 36%|███▌      | 144804/400000 [00:20<00:40, 6267.12it/s] 36%|███▋      | 145434/400000 [00:20<00:41, 6090.81it/s] 37%|███▋      | 146046/400000 [00:20<00:41, 6089.15it/s] 37%|███▋      | 146745/400000 [00:21<00:39, 6331.74it/s] 37%|███▋      | 147383/400000 [00:21<00:40, 6278.96it/s] 37%|███▋      | 148052/400000 [00:21<00:39, 6394.75it/s] 37%|███▋      | 148738/400000 [00:21<00:38, 6527.48it/s] 37%|███▋      | 149398/400000 [00:21<00:38, 6546.92it/s] 38%|███▊      | 150081/400000 [00:21<00:37, 6627.92it/s] 38%|███▊      | 150763/400000 [00:21<00:37, 6682.56it/s] 38%|███▊      | 151466/400000 [00:21<00:36, 6782.61it/s] 38%|███▊      | 152147/400000 [00:21<00:36, 6789.52it/s] 38%|███▊      | 152847/400000 [00:21<00:36, 6848.95it/s] 38%|███▊      | 153555/400000 [00:22<00:35, 6914.98it/s] 39%|███▊      | 154248/400000 [00:22<00:36, 6661.82it/s] 39%|███▊      | 154950/400000 [00:22<00:36, 6765.02it/s] 39%|███▉      | 155629/400000 [00:22<00:36, 6766.31it/s] 39%|███▉      | 156349/400000 [00:22<00:35, 6889.71it/s] 39%|███▉      | 157056/400000 [00:22<00:35, 6939.50it/s] 39%|███▉      | 157756/400000 [00:22<00:34, 6954.89it/s] 40%|███▉      | 158459/400000 [00:22<00:34, 6976.85it/s] 40%|███▉      | 159162/400000 [00:22<00:34, 6990.25it/s] 40%|███▉      | 159906/400000 [00:22<00:33, 7111.46it/s] 40%|████      | 160647/400000 [00:23<00:33, 7198.33it/s] 40%|████      | 161368/400000 [00:23<00:33, 7194.26it/s] 41%|████      | 162089/400000 [00:23<00:33, 7160.02it/s] 41%|████      | 162808/400000 [00:23<00:33, 7168.02it/s] 41%|████      | 163526/400000 [00:23<00:33, 7159.05it/s] 41%|████      | 164260/400000 [00:23<00:32, 7210.09it/s] 41%|████      | 164997/400000 [00:23<00:32, 7251.09it/s] 41%|████▏     | 165723/400000 [00:23<00:33, 7037.02it/s] 42%|████▏     | 166429/400000 [00:23<00:34, 6806.93it/s] 42%|████▏     | 167113/400000 [00:24<00:34, 6658.68it/s] 42%|████▏     | 167782/400000 [00:24<00:35, 6576.51it/s] 42%|████▏     | 168464/400000 [00:24<00:34, 6645.37it/s] 42%|████▏     | 169222/400000 [00:24<00:33, 6899.37it/s] 42%|████▏     | 169991/400000 [00:24<00:32, 7118.69it/s] 43%|████▎     | 170710/400000 [00:24<00:32, 7138.63it/s] 43%|████▎     | 171427/400000 [00:24<00:32, 7122.75it/s] 43%|████▎     | 172177/400000 [00:24<00:31, 7230.57it/s] 43%|████▎     | 172902/400000 [00:24<00:32, 6935.49it/s] 43%|████▎     | 173600/400000 [00:24<00:33, 6754.46it/s] 44%|████▎     | 174280/400000 [00:25<00:33, 6691.11it/s] 44%|████▍     | 175050/400000 [00:25<00:32, 6963.75it/s] 44%|████▍     | 175791/400000 [00:25<00:31, 7090.92it/s] 44%|████▍     | 176567/400000 [00:25<00:30, 7276.64it/s] 44%|████▍     | 177351/400000 [00:25<00:29, 7435.94it/s] 45%|████▍     | 178122/400000 [00:25<00:29, 7515.73it/s] 45%|████▍     | 178877/400000 [00:25<00:30, 7187.03it/s] 45%|████▍     | 179601/400000 [00:25<00:32, 6829.83it/s] 45%|████▌     | 180292/400000 [00:25<00:33, 6631.76it/s] 45%|████▌     | 180963/400000 [00:26<00:32, 6655.04it/s] 45%|████▌     | 181671/400000 [00:26<00:32, 6775.80it/s] 46%|████▌     | 182386/400000 [00:26<00:31, 6882.82it/s] 46%|████▌     | 183078/400000 [00:26<00:33, 6541.76it/s] 46%|████▌     | 183792/400000 [00:26<00:32, 6709.36it/s] 46%|████▌     | 184529/400000 [00:26<00:31, 6889.63it/s] 46%|████▋     | 185223/400000 [00:26<00:31, 6841.43it/s] 46%|████▋     | 185977/400000 [00:26<00:30, 7016.53it/s] 47%|████▋     | 186683/400000 [00:26<00:30, 6931.47it/s] 47%|████▋     | 187464/400000 [00:26<00:29, 7172.14it/s] 47%|████▋     | 188227/400000 [00:27<00:29, 7302.22it/s] 47%|████▋     | 189008/400000 [00:27<00:28, 7446.86it/s] 47%|████▋     | 189756/400000 [00:27<00:28, 7270.04it/s] 48%|████▊     | 190487/400000 [00:27<00:29, 7121.10it/s] 48%|████▊     | 191265/400000 [00:27<00:28, 7305.28it/s] 48%|████▊     | 192031/400000 [00:27<00:28, 7405.74it/s] 48%|████▊     | 192787/400000 [00:27<00:27, 7449.92it/s] 48%|████▊     | 193534/400000 [00:27<00:27, 7389.03it/s] 49%|████▊     | 194275/400000 [00:27<00:28, 7293.73it/s] 49%|████▉     | 195006/400000 [00:27<00:28, 7256.03it/s] 49%|████▉     | 195733/400000 [00:28<00:28, 7238.79it/s] 49%|████▉     | 196477/400000 [00:28<00:27, 7295.38it/s] 49%|████▉     | 197244/400000 [00:28<00:27, 7402.13it/s] 49%|████▉     | 197985/400000 [00:28<00:27, 7334.07it/s] 50%|████▉     | 198720/400000 [00:28<00:27, 7310.56it/s] 50%|████▉     | 199466/400000 [00:28<00:27, 7353.61it/s] 50%|█████     | 200202/400000 [00:28<00:27, 7316.43it/s] 50%|█████     | 200934/400000 [00:28<00:27, 7276.13it/s] 50%|█████     | 201662/400000 [00:28<00:27, 7193.21it/s] 51%|█████     | 202385/400000 [00:28<00:27, 7202.76it/s] 51%|█████     | 203148/400000 [00:29<00:26, 7324.71it/s] 51%|█████     | 203882/400000 [00:29<00:26, 7279.58it/s] 51%|█████     | 204611/400000 [00:29<00:27, 7096.07it/s] 51%|█████▏    | 205323/400000 [00:29<00:27, 7018.20it/s] 52%|█████▏    | 206027/400000 [00:29<00:27, 6959.78it/s] 52%|█████▏    | 206766/400000 [00:29<00:27, 7081.56it/s] 52%|█████▏    | 207516/400000 [00:29<00:26, 7201.61it/s] 52%|█████▏    | 208238/400000 [00:29<00:27, 7073.68it/s] 52%|█████▏    | 208947/400000 [00:29<00:27, 7022.50it/s] 52%|█████▏    | 209708/400000 [00:30<00:26, 7188.25it/s] 53%|█████▎    | 210444/400000 [00:30<00:26, 7237.01it/s] 53%|█████▎    | 211169/400000 [00:30<00:26, 7195.35it/s] 53%|█████▎    | 211905/400000 [00:30<00:25, 7242.17it/s] 53%|█████▎    | 212630/400000 [00:30<00:26, 7072.60it/s] 53%|█████▎    | 213339/400000 [00:30<00:27, 6894.67it/s] 54%|█████▎    | 214051/400000 [00:30<00:26, 6959.20it/s] 54%|█████▎    | 214796/400000 [00:30<00:26, 7098.16it/s] 54%|█████▍    | 215516/400000 [00:30<00:25, 7126.52it/s] 54%|█████▍    | 216230/400000 [00:30<00:25, 7090.63it/s] 54%|█████▍    | 216973/400000 [00:31<00:25, 7187.96it/s] 54%|█████▍    | 217741/400000 [00:31<00:24, 7328.06it/s] 55%|█████▍    | 218495/400000 [00:31<00:24, 7390.34it/s] 55%|█████▍    | 219237/400000 [00:31<00:24, 7398.48it/s] 55%|█████▍    | 219978/400000 [00:31<00:24, 7384.28it/s] 55%|█████▌    | 220717/400000 [00:31<00:24, 7267.11it/s] 55%|█████▌    | 221494/400000 [00:31<00:24, 7409.63it/s] 56%|█████▌    | 222257/400000 [00:31<00:23, 7473.87it/s] 56%|█████▌    | 223006/400000 [00:31<00:24, 7143.96it/s] 56%|█████▌    | 223725/400000 [00:31<00:25, 6785.20it/s] 56%|█████▌    | 224410/400000 [00:32<00:26, 6638.10it/s] 56%|█████▋    | 225113/400000 [00:32<00:25, 6749.55it/s] 56%|█████▋    | 225873/400000 [00:32<00:24, 6981.41it/s] 57%|█████▋    | 226577/400000 [00:32<00:25, 6770.25it/s] 57%|█████▋    | 227278/400000 [00:32<00:25, 6838.37it/s] 57%|█████▋    | 228049/400000 [00:32<00:24, 7077.51it/s] 57%|█████▋    | 228788/400000 [00:32<00:23, 7167.10it/s] 57%|█████▋    | 229524/400000 [00:32<00:23, 7223.75it/s] 58%|█████▊    | 230249/400000 [00:32<00:23, 7207.81it/s] 58%|█████▊    | 231013/400000 [00:32<00:23, 7329.73it/s] 58%|█████▊    | 231786/400000 [00:33<00:22, 7444.36it/s] 58%|█████▊    | 232533/400000 [00:33<00:22, 7436.65it/s] 58%|█████▊    | 233278/400000 [00:33<00:22, 7392.18it/s] 59%|█████▊    | 234019/400000 [00:33<00:22, 7283.74it/s] 59%|█████▊    | 234749/400000 [00:33<00:23, 7178.85it/s] 59%|█████▉    | 235468/400000 [00:33<00:23, 6876.51it/s] 59%|█████▉    | 236160/400000 [00:33<00:24, 6744.31it/s] 59%|█████▉    | 236931/400000 [00:33<00:23, 7006.72it/s] 59%|█████▉    | 237666/400000 [00:33<00:22, 7105.40it/s] 60%|█████▉    | 238381/400000 [00:34<00:23, 7024.63it/s] 60%|█████▉    | 239087/400000 [00:34<00:23, 6792.91it/s] 60%|█████▉    | 239770/400000 [00:34<00:24, 6578.58it/s] 60%|██████    | 240432/400000 [00:34<00:24, 6469.33it/s] 60%|██████    | 241083/400000 [00:34<00:24, 6414.93it/s] 60%|██████    | 241727/400000 [00:34<00:24, 6352.19it/s] 61%|██████    | 242364/400000 [00:34<00:25, 6277.87it/s] 61%|██████    | 242994/400000 [00:34<00:25, 6204.03it/s] 61%|██████    | 243616/400000 [00:34<00:25, 6197.30it/s] 61%|██████    | 244237/400000 [00:34<00:25, 6095.80it/s] 61%|██████    | 244871/400000 [00:35<00:25, 6166.28it/s] 61%|██████▏   | 245494/400000 [00:35<00:24, 6182.97it/s] 62%|██████▏   | 246230/400000 [00:35<00:23, 6492.64it/s] 62%|██████▏   | 246994/400000 [00:35<00:22, 6798.79it/s] 62%|██████▏   | 247714/400000 [00:35<00:22, 6914.28it/s] 62%|██████▏   | 248411/400000 [00:35<00:22, 6774.55it/s] 62%|██████▏   | 249093/400000 [00:35<00:23, 6548.11it/s] 62%|██████▏   | 249801/400000 [00:35<00:22, 6698.27it/s] 63%|██████▎   | 250574/400000 [00:35<00:21, 6975.20it/s] 63%|██████▎   | 251318/400000 [00:35<00:20, 7107.61it/s] 63%|██████▎   | 252076/400000 [00:36<00:20, 7240.53it/s] 63%|██████▎   | 252831/400000 [00:36<00:20, 7330.05it/s] 63%|██████▎   | 253568/400000 [00:36<00:20, 7145.04it/s] 64%|██████▎   | 254343/400000 [00:36<00:19, 7314.81it/s] 64%|██████▍   | 255093/400000 [00:36<00:19, 7367.42it/s] 64%|██████▍   | 255833/400000 [00:36<00:20, 7074.31it/s] 64%|██████▍   | 256545/400000 [00:36<00:21, 6816.85it/s] 64%|██████▍   | 257232/400000 [00:36<00:21, 6643.31it/s] 64%|██████▍   | 257901/400000 [00:36<00:21, 6589.57it/s] 65%|██████▍   | 258615/400000 [00:37<00:20, 6744.06it/s] 65%|██████▍   | 259293/400000 [00:37<00:21, 6618.05it/s] 65%|██████▍   | 259958/400000 [00:37<00:21, 6522.13it/s] 65%|██████▌   | 260613/400000 [00:37<00:21, 6477.97it/s] 65%|██████▌   | 261367/400000 [00:37<00:20, 6763.28it/s] 66%|██████▌   | 262156/400000 [00:37<00:19, 7065.09it/s] 66%|██████▌   | 262920/400000 [00:37<00:18, 7226.91it/s] 66%|██████▌   | 263671/400000 [00:37<00:18, 7309.20it/s] 66%|██████▌   | 264407/400000 [00:37<00:19, 7008.66it/s] 66%|██████▋   | 265114/400000 [00:37<00:19, 6981.35it/s] 66%|██████▋   | 265864/400000 [00:38<00:18, 7127.74it/s] 67%|██████▋   | 266591/400000 [00:38<00:18, 7167.44it/s] 67%|██████▋   | 267311/400000 [00:38<00:18, 7138.04it/s] 67%|██████▋   | 268035/400000 [00:38<00:18, 7167.55it/s] 67%|██████▋   | 268754/400000 [00:38<00:18, 7045.69it/s] 67%|██████▋   | 269472/400000 [00:38<00:18, 7083.24it/s] 68%|██████▊   | 270182/400000 [00:38<00:18, 7080.69it/s] 68%|██████▊   | 270891/400000 [00:38<00:18, 6877.96it/s] 68%|██████▊   | 271581/400000 [00:38<00:19, 6670.74it/s] 68%|██████▊   | 272251/400000 [00:39<00:19, 6535.39it/s] 68%|██████▊   | 272949/400000 [00:39<00:19, 6660.71it/s] 68%|██████▊   | 273688/400000 [00:39<00:18, 6862.59it/s] 69%|██████▊   | 274450/400000 [00:39<00:17, 7071.71it/s] 69%|██████▉   | 275161/400000 [00:39<00:18, 6805.66it/s] 69%|██████▉   | 275847/400000 [00:39<00:18, 6536.58it/s] 69%|██████▉   | 276507/400000 [00:39<00:18, 6545.21it/s] 69%|██████▉   | 277286/400000 [00:39<00:17, 6871.89it/s] 70%|██████▉   | 278043/400000 [00:39<00:17, 7066.99it/s] 70%|██████▉   | 278822/400000 [00:39<00:16, 7268.55it/s] 70%|██████▉   | 279569/400000 [00:40<00:16, 7327.44it/s] 70%|███████   | 280306/400000 [00:40<00:16, 7319.22it/s] 70%|███████   | 281041/400000 [00:40<00:16, 7237.81it/s] 70%|███████   | 281806/400000 [00:40<00:16, 7354.14it/s] 71%|███████   | 282550/400000 [00:40<00:15, 7378.65it/s] 71%|███████   | 283290/400000 [00:40<00:15, 7369.37it/s] 71%|███████   | 284063/400000 [00:40<00:15, 7473.43it/s] 71%|███████   | 284812/400000 [00:40<00:15, 7380.96it/s] 71%|███████▏  | 285552/400000 [00:40<00:15, 7363.72it/s] 72%|███████▏  | 286329/400000 [00:40<00:15, 7478.57it/s] 72%|███████▏  | 287080/400000 [00:41<00:15, 7486.46it/s] 72%|███████▏  | 287830/400000 [00:41<00:15, 7046.32it/s] 72%|███████▏  | 288541/400000 [00:41<00:16, 6838.61it/s] 72%|███████▏  | 289231/400000 [00:41<00:16, 6606.63it/s] 72%|███████▏  | 289985/400000 [00:41<00:16, 6858.79it/s] 73%|███████▎  | 290740/400000 [00:41<00:15, 7051.47it/s] 73%|███████▎  | 291486/400000 [00:41<00:15, 7167.80it/s] 73%|███████▎  | 292208/400000 [00:41<00:15, 6916.79it/s] 73%|███████▎  | 292965/400000 [00:41<00:15, 7099.91it/s] 73%|███████▎  | 293680/400000 [00:42<00:14, 7098.61it/s] 74%|███████▎  | 294435/400000 [00:42<00:14, 7226.83it/s] 74%|███████▍  | 295161/400000 [00:42<00:14, 7207.44it/s] 74%|███████▍  | 295884/400000 [00:42<00:15, 6932.64it/s] 74%|███████▍  | 296581/400000 [00:42<00:15, 6635.60it/s] 74%|███████▍  | 297250/400000 [00:42<00:15, 6444.07it/s] 74%|███████▍  | 297900/400000 [00:42<00:16, 6369.43it/s] 75%|███████▍  | 298610/400000 [00:42<00:15, 6569.69it/s] 75%|███████▍  | 299272/400000 [00:42<00:15, 6372.11it/s] 75%|███████▍  | 299993/400000 [00:42<00:15, 6601.81it/s] 75%|███████▌  | 300732/400000 [00:43<00:14, 6817.96it/s] 75%|███████▌  | 301490/400000 [00:43<00:14, 7029.76it/s] 76%|███████▌  | 302244/400000 [00:43<00:13, 7175.23it/s] 76%|███████▌  | 303010/400000 [00:43<00:13, 7313.94it/s] 76%|███████▌  | 303768/400000 [00:43<00:13, 7391.02it/s] 76%|███████▌  | 304510/400000 [00:43<00:13, 7245.22it/s] 76%|███████▋  | 305238/400000 [00:43<00:13, 6933.02it/s] 76%|███████▋  | 305937/400000 [00:43<00:14, 6651.22it/s] 77%|███████▋  | 306608/400000 [00:43<00:14, 6532.86it/s] 77%|███████▋  | 307281/400000 [00:44<00:14, 6588.90it/s] 77%|███████▋  | 307973/400000 [00:44<00:13, 6684.22it/s] 77%|███████▋  | 308696/400000 [00:44<00:13, 6837.69it/s] 77%|███████▋  | 309435/400000 [00:44<00:12, 6993.48it/s] 78%|███████▊  | 310138/400000 [00:44<00:13, 6901.01it/s] 78%|███████▊  | 310831/400000 [00:44<00:13, 6764.84it/s] 78%|███████▊  | 311510/400000 [00:44<00:13, 6548.76it/s] 78%|███████▊  | 312185/400000 [00:44<00:13, 6607.31it/s] 78%|███████▊  | 312943/400000 [00:44<00:12, 6870.52it/s] 78%|███████▊  | 313659/400000 [00:44<00:12, 6952.29it/s] 79%|███████▊  | 314409/400000 [00:45<00:12, 7106.99it/s] 79%|███████▉  | 315123/400000 [00:45<00:12, 7010.85it/s] 79%|███████▉  | 315827/400000 [00:45<00:12, 6694.24it/s] 79%|███████▉  | 316502/400000 [00:45<00:12, 6698.63it/s] 79%|███████▉  | 317265/400000 [00:45<00:11, 6953.13it/s] 80%|███████▉  | 318014/400000 [00:45<00:11, 7104.41it/s] 80%|███████▉  | 318750/400000 [00:45<00:11, 7177.84it/s] 80%|███████▉  | 319508/400000 [00:45<00:11, 7293.34it/s] 80%|████████  | 320251/400000 [00:45<00:10, 7331.92it/s] 80%|████████  | 321011/400000 [00:45<00:10, 7408.82it/s] 80%|████████  | 321754/400000 [00:46<00:10, 7366.90it/s] 81%|████████  | 322492/400000 [00:46<00:10, 7338.51it/s] 81%|████████  | 323250/400000 [00:46<00:10, 7408.40it/s] 81%|████████  | 324025/400000 [00:46<00:10, 7505.54it/s] 81%|████████  | 324780/400000 [00:46<00:10, 7517.76it/s] 81%|████████▏ | 325561/400000 [00:46<00:09, 7601.51it/s] 82%|████████▏ | 326326/400000 [00:46<00:09, 7615.83it/s] 82%|████████▏ | 327089/400000 [00:46<00:09, 7423.12it/s] 82%|████████▏ | 327870/400000 [00:46<00:09, 7532.38it/s] 82%|████████▏ | 328625/400000 [00:47<00:10, 7050.75it/s] 82%|████████▏ | 329338/400000 [00:47<00:10, 6798.24it/s] 83%|████████▎ | 330097/400000 [00:47<00:09, 7015.50it/s] 83%|████████▎ | 330845/400000 [00:47<00:09, 7148.60it/s] 83%|████████▎ | 331566/400000 [00:47<00:10, 6838.62it/s] 83%|████████▎ | 332257/400000 [00:47<00:10, 6680.89it/s] 83%|████████▎ | 332931/400000 [00:47<00:10, 6532.52it/s] 83%|████████▎ | 333597/400000 [00:47<00:10, 6567.99it/s] 84%|████████▎ | 334257/400000 [00:47<00:10, 6510.88it/s] 84%|████████▎ | 334911/400000 [00:47<00:10, 6390.72it/s] 84%|████████▍ | 335553/400000 [00:48<00:10, 6358.89it/s] 84%|████████▍ | 336191/400000 [00:48<00:10, 6275.81it/s] 84%|████████▍ | 336820/400000 [00:48<00:10, 6253.29it/s] 84%|████████▍ | 337447/400000 [00:48<00:10, 6241.19it/s] 85%|████████▍ | 338072/400000 [00:48<00:09, 6241.38it/s] 85%|████████▍ | 338697/400000 [00:48<00:09, 6219.32it/s] 85%|████████▍ | 339407/400000 [00:48<00:09, 6459.10it/s] 85%|████████▌ | 340132/400000 [00:48<00:08, 6675.24it/s] 85%|████████▌ | 340877/400000 [00:48<00:08, 6887.46it/s] 85%|████████▌ | 341571/400000 [00:48<00:08, 6703.12it/s] 86%|████████▌ | 342246/400000 [00:49<00:08, 6629.51it/s] 86%|████████▌ | 342912/400000 [00:49<00:08, 6478.46it/s] 86%|████████▌ | 343563/400000 [00:49<00:08, 6446.62it/s] 86%|████████▌ | 344282/400000 [00:49<00:08, 6651.56it/s] 86%|████████▋ | 345029/400000 [00:49<00:07, 6877.40it/s] 86%|████████▋ | 345804/400000 [00:49<00:07, 7116.57it/s] 87%|████████▋ | 346548/400000 [00:49<00:07, 7208.30it/s] 87%|████████▋ | 347311/400000 [00:49<00:07, 7329.31it/s] 87%|████████▋ | 348048/400000 [00:49<00:07, 7242.96it/s] 87%|████████▋ | 348786/400000 [00:50<00:07, 7282.79it/s] 87%|████████▋ | 349536/400000 [00:50<00:06, 7344.47it/s] 88%|████████▊ | 350305/400000 [00:50<00:06, 7442.12it/s] 88%|████████▊ | 351077/400000 [00:50<00:06, 7521.21it/s] 88%|████████▊ | 351831/400000 [00:50<00:06, 7322.92it/s] 88%|████████▊ | 352594/400000 [00:50<00:06, 7410.96it/s] 88%|████████▊ | 353337/400000 [00:50<00:06, 7374.34it/s] 89%|████████▊ | 354076/400000 [00:50<00:06, 7143.16it/s] 89%|████████▊ | 354793/400000 [00:50<00:06, 6864.88it/s] 89%|████████▉ | 355484/400000 [00:50<00:06, 6626.94it/s] 89%|████████▉ | 356152/400000 [00:51<00:06, 6607.05it/s] 89%|████████▉ | 356928/400000 [00:51<00:06, 6915.11it/s] 89%|████████▉ | 357673/400000 [00:51<00:05, 7067.09it/s] 90%|████████▉ | 358442/400000 [00:51<00:05, 7242.39it/s] 90%|████████▉ | 359178/400000 [00:51<00:05, 7276.87it/s] 90%|████████▉ | 359923/400000 [00:51<00:05, 7325.98it/s] 90%|█████████ | 360690/400000 [00:51<00:05, 7425.25it/s] 90%|█████████ | 361435/400000 [00:51<00:05, 7339.72it/s] 91%|█████████ | 362201/400000 [00:51<00:05, 7431.86it/s] 91%|█████████ | 362959/400000 [00:51<00:04, 7475.04it/s] 91%|█████████ | 363708/400000 [00:52<00:05, 7005.29it/s] 91%|█████████ | 364416/400000 [00:52<00:05, 6694.71it/s] 91%|█████████▏| 365094/400000 [00:52<00:05, 6516.29it/s] 91%|█████████▏| 365753/400000 [00:52<00:05, 6392.17it/s] 92%|█████████▏| 366456/400000 [00:52<00:05, 6571.02it/s] 92%|█████████▏| 367222/400000 [00:52<00:04, 6861.91it/s] 92%|█████████▏| 367915/400000 [00:52<00:04, 6866.45it/s] 92%|█████████▏| 368642/400000 [00:52<00:04, 6980.75it/s] 92%|█████████▏| 369353/400000 [00:52<00:04, 7017.41it/s] 93%|█████████▎| 370070/400000 [00:53<00:04, 7061.07it/s] 93%|█████████▎| 370779/400000 [00:53<00:04, 6954.56it/s] 93%|█████████▎| 371506/400000 [00:53<00:04, 7044.08it/s] 93%|█████████▎| 372232/400000 [00:53<00:03, 7105.37it/s] 93%|█████████▎| 372944/400000 [00:53<00:03, 7059.38it/s] 93%|█████████▎| 373719/400000 [00:53<00:03, 7252.07it/s] 94%|█████████▎| 374473/400000 [00:53<00:03, 7334.10it/s] 94%|█████████▍| 375208/400000 [00:53<00:03, 7037.37it/s] 94%|█████████▍| 375965/400000 [00:53<00:03, 7186.86it/s] 94%|█████████▍| 376746/400000 [00:53<00:03, 7360.59it/s] 94%|█████████▍| 377486/400000 [00:54<00:03, 7275.45it/s] 95%|█████████▍| 378217/400000 [00:54<00:03, 7238.77it/s] 95%|█████████▍| 378953/400000 [00:54<00:02, 7272.85it/s] 95%|█████████▍| 379734/400000 [00:54<00:02, 7423.46it/s] 95%|█████████▌| 380479/400000 [00:54<00:02, 7350.84it/s] 95%|█████████▌| 381216/400000 [00:54<00:02, 7286.33it/s] 95%|█████████▌| 381946/400000 [00:54<00:02, 7116.45it/s] 96%|█████████▌| 382669/400000 [00:54<00:02, 7149.64it/s] 96%|█████████▌| 383425/400000 [00:54<00:02, 7265.94it/s] 96%|█████████▌| 384197/400000 [00:54<00:02, 7395.65it/s] 96%|█████████▌| 384973/400000 [00:55<00:02, 7499.22it/s] 96%|█████████▋| 385756/400000 [00:55<00:01, 7594.72it/s] 97%|█████████▋| 386517/400000 [00:55<00:01, 7554.23it/s] 97%|█████████▋| 387274/400000 [00:55<00:01, 7383.50it/s] 97%|█████████▋| 388014/400000 [00:55<00:01, 6998.86it/s] 97%|█████████▋| 388720/400000 [00:55<00:01, 6761.78it/s] 97%|█████████▋| 389402/400000 [00:55<00:01, 6666.74it/s] 98%|█████████▊| 390137/400000 [00:55<00:01, 6857.12it/s] 98%|█████████▊| 390835/400000 [00:55<00:01, 6891.59it/s] 98%|█████████▊| 391601/400000 [00:56<00:01, 7103.43it/s] 98%|█████████▊| 392329/400000 [00:56<00:01, 7154.96it/s] 98%|█████████▊| 393085/400000 [00:56<00:00, 7269.26it/s] 98%|█████████▊| 393833/400000 [00:56<00:00, 7328.93it/s] 99%|█████████▊| 394594/400000 [00:56<00:00, 7410.71it/s] 99%|█████████▉| 395368/400000 [00:56<00:00, 7505.40it/s] 99%|█████████▉| 396120/400000 [00:56<00:00, 7487.38it/s] 99%|█████████▉| 396896/400000 [00:56<00:00, 7566.74it/s] 99%|█████████▉| 397654/400000 [00:56<00:00, 7427.71it/s]100%|█████████▉| 398398/400000 [00:56<00:00, 7427.17it/s]100%|█████████▉| 399173/400000 [00:57<00:00, 7520.14it/s]100%|█████████▉| 399940/400000 [00:57<00:00, 7563.69it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 7002.34it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ffb34746f60> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010985793814979739 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.011327183166873894 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15993 out of table with 15660 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15993 out of table with 15660 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

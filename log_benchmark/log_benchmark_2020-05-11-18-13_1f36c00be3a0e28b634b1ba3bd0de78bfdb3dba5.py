
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f31b8a9afd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 18:14:02.403404
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 18:14:02.407765
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 18:14:02.410934
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 18:14:02.414705
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f31c4ab2438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354100.2188
Epoch 2/10

1/1 [==============================] - 0s 116ms/step - loss: 288459.1250
Epoch 3/10

1/1 [==============================] - 0s 115ms/step - loss: 198623.2031
Epoch 4/10

1/1 [==============================] - 0s 118ms/step - loss: 128374.1953
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 75592.0156
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 42103.2969
Epoch 7/10

1/1 [==============================] - 0s 109ms/step - loss: 23966.9805
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 14853.2783
Epoch 9/10

1/1 [==============================] - 0s 115ms/step - loss: 9958.2910
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 7162.5269

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.0557435e-01 -1.9348993e+00 -7.0409268e-01  1.2713814e-01
  -6.6761959e-01 -9.9314463e-01 -5.9176427e-01 -2.6218289e-01
   3.4039503e-01  8.0376111e-02  2.3312053e-01  1.5570527e+00
  -6.6145551e-01  8.3012533e-01 -2.3842597e-01  6.7966211e-01
   2.8728628e-01 -2.1037668e-02  5.6586504e-01 -3.1591511e-01
   4.5888245e-02  7.6359785e-01  5.3876996e-01  2.2782147e-02
   1.9424734e-01 -1.8236160e+00  2.3493157e-01  9.6752912e-01
   1.5956862e-01 -9.3751734e-01  1.2561066e+00 -1.7338841e+00
   1.3773531e-01  4.5376062e-01  8.5133427e-01  5.1427758e-01
  -1.4209480e+00 -8.8530570e-01  4.5874760e-01  1.1315525e+00
  -5.1216477e-01  7.5824916e-02  4.4303215e-01 -3.6797479e-01
   1.1667329e-01  3.9222887e-01 -3.8899660e-02  8.6998522e-01
   1.7948127e+00 -2.7678281e-01 -1.4690766e+00 -5.0638181e-01
   1.0441928e+00 -4.7287321e-01  2.2653122e+00 -4.4866365e-01
  -1.1354798e+00 -1.2008774e+00  4.0739012e-01  8.8834262e-01
  -2.7508071e-01  7.8068728e+00  8.9053679e+00  8.8728676e+00
   8.0221624e+00  7.2588825e+00  7.7310920e+00  8.4163780e+00
   7.9898329e+00  7.3897867e+00  6.8221207e+00  8.5363007e+00
   6.9180164e+00  6.2218032e+00  7.1338363e+00  6.3476305e+00
   7.0933633e+00  8.6966839e+00  6.3462653e+00  7.6995306e+00
   7.4596663e+00  8.2968740e+00  8.5470200e+00  7.1672974e+00
   7.5438786e+00  8.3474703e+00  9.1530361e+00  9.1795521e+00
   7.9493399e+00  7.9215617e+00  8.2248507e+00  7.7798729e+00
   6.2255230e+00  8.0781746e+00  7.5439429e+00  8.0107517e+00
   8.5902557e+00  8.0080252e+00  7.7251306e+00  7.1620264e+00
   6.6755953e+00  6.8527484e+00  8.4175177e+00  7.6900659e+00
   6.3139076e+00  6.5044370e+00  7.6704364e+00  7.2738848e+00
   7.7566910e+00  8.3833771e+00  7.0749030e+00  9.1361704e+00
   8.5282173e+00  7.2725506e+00  6.3578892e+00  8.1503754e+00
   6.6060390e+00  8.2516527e+00  5.7110729e+00  7.4470911e+00
  -2.5872153e-01 -5.3810668e-01  1.7144424e-01 -1.4128311e+00
  -6.0862517e-01 -1.1119686e-01  9.9379945e-01  7.0512366e-01
   1.2954218e+00 -1.3063303e+00 -1.3209638e-01  1.7041577e+00
   5.8027786e-01  4.0049645e-01 -5.7244152e-02 -1.9101197e-01
   6.7126423e-01  1.7782834e-01  2.8859743e-01  4.0753245e-02
  -6.4000547e-01 -1.5368140e+00  2.7571774e-01  9.0763718e-01
   1.4839039e+00  3.0237460e-01 -1.4341199e+00  1.4499466e+00
  -4.9966672e-01  1.3542533e-02  5.3495169e-01  1.0451539e+00
   8.2828915e-01 -6.5764093e-01 -8.0152094e-01 -7.9251599e-01
  -6.5741009e-01  6.2242353e-01 -9.2138183e-01  4.9488965e-01
  -2.3399204e-02 -1.2886147e+00  5.3226578e-01 -1.2835873e+00
  -8.4326309e-01 -4.8500830e-01  2.0837912e-01 -1.3060715e+00
  -4.7579652e-01  8.1169057e-01  5.2954602e-01 -9.9706841e-01
   1.8715692e-01 -3.1849566e-01 -8.4964156e-02 -5.1367760e-01
  -6.4063710e-01 -1.2855166e+00  7.8255260e-01  5.5956811e-01
   5.8965015e-01  5.1858371e-01  2.5759563e+00  8.3169305e-01
   1.0949960e+00  1.5995402e+00  9.5194644e-01  7.3442245e-01
   1.2433938e+00  1.0529267e+00  1.0467834e+00  1.1009023e+00
   2.3188734e-01  1.3962716e-01  9.8141170e-01  3.0883694e-01
   1.2987221e+00  8.9239067e-01  6.8870568e-01  8.9935660e-01
   9.4161862e-01  1.9124866e-01  7.6907527e-01  5.1896888e-01
   1.2633064e+00  7.2349286e-01  3.6591101e-01  2.1133218e+00
   2.5197899e-01  8.3409899e-01  1.8759290e+00  1.3612367e+00
   9.1623294e-01  3.1117034e-01  4.9397111e-01  4.7752178e-01
   9.2349154e-01  1.6407560e+00  6.4583510e-01  1.1178869e+00
   5.3938961e-01  4.9984634e-01  2.9947889e-01  7.8120887e-01
   2.5498481e+00  1.4388027e+00  2.3610950e+00  1.8314362e+00
   1.7686014e+00  4.7302186e-01  1.8292844e-01  9.1079336e-01
   5.0292826e-01  3.3169901e-01  2.5898724e+00  6.3472599e-01
   3.0236870e-01  4.4169211e-01  1.5051843e+00  2.0906764e-01
   9.4356239e-02  8.2773161e+00  7.4539223e+00  8.6156216e+00
   7.9643006e+00  7.8140893e+00  7.2428136e+00  7.0036626e+00
   7.9097538e+00  7.9645967e+00  8.8370314e+00  7.9399514e+00
   7.1120987e+00  7.6354904e+00  9.3484812e+00  8.1198606e+00
   8.7871790e+00  6.2263470e+00  9.3342295e+00  9.3208761e+00
   7.0252428e+00  8.0157118e+00  7.4695067e+00  8.3894157e+00
   8.6399517e+00  7.4956150e+00  9.0352230e+00  7.7219062e+00
   9.4180145e+00  9.4033194e+00  9.0444975e+00  9.4655609e+00
   8.6380148e+00  9.8605061e+00  7.7712421e+00  7.1702523e+00
   9.1816845e+00  7.4947438e+00  9.1712675e+00  8.4573822e+00
   9.1400528e+00  8.4745073e+00  9.3663683e+00  8.9439068e+00
   8.6169395e+00  8.0070848e+00  8.8577671e+00  6.8818955e+00
   8.5261421e+00  6.8572946e+00  8.6858273e+00  8.6220484e+00
   9.2041903e+00  7.7711921e+00  8.9250698e+00  8.2460957e+00
   8.1139145e+00  8.3397474e+00  9.1602659e+00  8.5173368e+00
   2.3158985e-01  1.8968461e+00  1.6682479e+00  1.3464547e+00
   9.5567369e-01  1.6394157e+00  1.4643019e-01  7.4762666e-01
   1.7077360e+00  2.5384369e+00  1.9487909e+00  1.6179533e+00
   1.7930334e+00  8.3549714e-01  1.0221071e+00  1.0492259e+00
   1.1161325e+00  2.2573066e+00  2.1567903e+00  3.3162498e-01
   2.1571097e+00  7.4920392e-01  6.9703996e-01  3.0573261e-01
   5.1412505e-01  1.0823519e+00  6.9838554e-01  8.1241512e-01
   4.2567825e-01  1.3042799e+00  2.7544417e+00  1.3336020e+00
   1.5930524e+00  5.0063109e-01  7.0520186e-01  1.1469063e+00
   1.7755129e+00  3.3332796e+00  2.5905466e+00  6.6038215e-01
   1.4107654e+00  1.2614383e+00  9.1785520e-01  2.0222490e+00
   5.5990928e-01  6.0996842e-01  2.2193301e-01  1.7304727e+00
   2.0003607e+00  1.4032166e+00  9.3062240e-01  6.8612611e-01
   1.2324901e+00  3.6152840e-01  5.0966465e-01  7.8386444e-01
   1.5066626e+00  2.6357126e-01  2.2929223e+00  6.3567901e-01
  -3.1990962e+00  3.4396832e+00 -1.3608469e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 18:14:13.761218
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    93.977
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 18:14:13.765078
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8855.68
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 18:14:13.768363
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.1602
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 18:14:13.771817
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -792.089
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139851188698584
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139849961713168
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139849961275464
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139849961275968
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139849961276472
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139849961276976

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f31c0935ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.502503
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.475585
grad_step = 000002, loss = 0.454134
grad_step = 000003, loss = 0.430708
grad_step = 000004, loss = 0.405888
grad_step = 000005, loss = 0.382034
grad_step = 000006, loss = 0.364651
grad_step = 000007, loss = 0.359008
grad_step = 000008, loss = 0.348555
grad_step = 000009, loss = 0.331774
grad_step = 000010, loss = 0.318166
grad_step = 000011, loss = 0.310038
grad_step = 000012, loss = 0.303744
grad_step = 000013, loss = 0.296035
grad_step = 000014, loss = 0.286341
grad_step = 000015, loss = 0.275491
grad_step = 000016, loss = 0.264829
grad_step = 000017, loss = 0.255368
grad_step = 000018, loss = 0.246915
grad_step = 000019, loss = 0.238364
grad_step = 000020, loss = 0.229331
grad_step = 000021, loss = 0.220411
grad_step = 000022, loss = 0.212158
grad_step = 000023, loss = 0.204478
grad_step = 000024, loss = 0.196853
grad_step = 000025, loss = 0.189002
grad_step = 000026, loss = 0.181171
grad_step = 000027, loss = 0.173710
grad_step = 000028, loss = 0.166666
grad_step = 000029, loss = 0.159783
grad_step = 000030, loss = 0.152865
grad_step = 000031, loss = 0.146059
grad_step = 000032, loss = 0.139637
grad_step = 000033, loss = 0.133540
grad_step = 000034, loss = 0.127495
grad_step = 000035, loss = 0.121441
grad_step = 000036, loss = 0.115594
grad_step = 000037, loss = 0.110119
grad_step = 000038, loss = 0.104855
grad_step = 000039, loss = 0.099607
grad_step = 000040, loss = 0.094471
grad_step = 000041, loss = 0.089620
grad_step = 000042, loss = 0.084936
grad_step = 000043, loss = 0.080297
grad_step = 000044, loss = 0.075826
grad_step = 000045, loss = 0.071609
grad_step = 000046, loss = 0.067557
grad_step = 000047, loss = 0.063624
grad_step = 000048, loss = 0.059818
grad_step = 000049, loss = 0.056205
grad_step = 000050, loss = 0.052766
grad_step = 000051, loss = 0.049418
grad_step = 000052, loss = 0.046235
grad_step = 000053, loss = 0.043265
grad_step = 000054, loss = 0.040416
grad_step = 000055, loss = 0.037689
grad_step = 000056, loss = 0.035133
grad_step = 000057, loss = 0.032696
grad_step = 000058, loss = 0.030369
grad_step = 000059, loss = 0.028189
grad_step = 000060, loss = 0.026174
grad_step = 000061, loss = 0.024281
grad_step = 000062, loss = 0.022537
grad_step = 000063, loss = 0.020871
grad_step = 000064, loss = 0.019214
grad_step = 000065, loss = 0.017689
grad_step = 000066, loss = 0.016388
grad_step = 000067, loss = 0.015158
grad_step = 000068, loss = 0.013899
grad_step = 000069, loss = 0.012799
grad_step = 000070, loss = 0.011850
grad_step = 000071, loss = 0.010905
grad_step = 000072, loss = 0.010007
grad_step = 000073, loss = 0.009235
grad_step = 000074, loss = 0.008554
grad_step = 000075, loss = 0.007892
grad_step = 000076, loss = 0.007257
grad_step = 000077, loss = 0.006718
grad_step = 000078, loss = 0.006256
grad_step = 000079, loss = 0.005806
grad_step = 000080, loss = 0.005376
grad_step = 000081, loss = 0.005001
grad_step = 000082, loss = 0.004689
grad_step = 000083, loss = 0.004411
grad_step = 000084, loss = 0.004135
grad_step = 000085, loss = 0.003882
grad_step = 000086, loss = 0.003664
grad_step = 000087, loss = 0.003487
grad_step = 000088, loss = 0.003334
grad_step = 000089, loss = 0.003187
grad_step = 000090, loss = 0.003050
grad_step = 000091, loss = 0.002924
grad_step = 000092, loss = 0.002823
grad_step = 000093, loss = 0.002740
grad_step = 000094, loss = 0.002672
grad_step = 000095, loss = 0.002612
grad_step = 000096, loss = 0.002555
grad_step = 000097, loss = 0.002504
grad_step = 000098, loss = 0.002457
grad_step = 000099, loss = 0.002415
grad_step = 000100, loss = 0.002380
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002352
grad_step = 000102, loss = 0.002328
grad_step = 000103, loss = 0.002308
grad_step = 000104, loss = 0.002293
grad_step = 000105, loss = 0.002281
grad_step = 000106, loss = 0.002274
grad_step = 000107, loss = 0.002274
grad_step = 000108, loss = 0.002290
grad_step = 000109, loss = 0.002340
grad_step = 000110, loss = 0.002435
grad_step = 000111, loss = 0.002543
grad_step = 000112, loss = 0.002528
grad_step = 000113, loss = 0.002349
grad_step = 000114, loss = 0.002233
grad_step = 000115, loss = 0.002289
grad_step = 000116, loss = 0.002371
grad_step = 000117, loss = 0.002333
grad_step = 000118, loss = 0.002222
grad_step = 000119, loss = 0.002215
grad_step = 000120, loss = 0.002288
grad_step = 000121, loss = 0.002283
grad_step = 000122, loss = 0.002194
grad_step = 000123, loss = 0.002183
grad_step = 000124, loss = 0.002250
grad_step = 000125, loss = 0.002235
grad_step = 000126, loss = 0.002167
grad_step = 000127, loss = 0.002170
grad_step = 000128, loss = 0.002209
grad_step = 000129, loss = 0.002188
grad_step = 000130, loss = 0.002153
grad_step = 000131, loss = 0.002153
grad_step = 000132, loss = 0.002163
grad_step = 000133, loss = 0.002157
grad_step = 000134, loss = 0.002143
grad_step = 000135, loss = 0.002134
grad_step = 000136, loss = 0.002129
grad_step = 000137, loss = 0.002126
grad_step = 000138, loss = 0.002124
grad_step = 000139, loss = 0.002125
grad_step = 000140, loss = 0.002118
grad_step = 000141, loss = 0.002108
grad_step = 000142, loss = 0.002093
grad_step = 000143, loss = 0.002087
grad_step = 000144, loss = 0.002092
grad_step = 000145, loss = 0.002106
grad_step = 000146, loss = 0.002132
grad_step = 000147, loss = 0.002144
grad_step = 000148, loss = 0.002150
grad_step = 000149, loss = 0.002122
grad_step = 000150, loss = 0.002103
grad_step = 000151, loss = 0.002091
grad_step = 000152, loss = 0.002074
grad_step = 000153, loss = 0.002059
grad_step = 000154, loss = 0.002063
grad_step = 000155, loss = 0.002092
grad_step = 000156, loss = 0.002134
grad_step = 000157, loss = 0.002194
grad_step = 000158, loss = 0.002245
grad_step = 000159, loss = 0.002280
grad_step = 000160, loss = 0.002144
grad_step = 000161, loss = 0.002029
grad_step = 000162, loss = 0.002072
grad_step = 000163, loss = 0.002150
grad_step = 000164, loss = 0.002124
grad_step = 000165, loss = 0.002038
grad_step = 000166, loss = 0.002060
grad_step = 000167, loss = 0.002102
grad_step = 000168, loss = 0.002060
grad_step = 000169, loss = 0.002038
grad_step = 000170, loss = 0.002066
grad_step = 000171, loss = 0.002059
grad_step = 000172, loss = 0.002024
grad_step = 000173, loss = 0.002013
grad_step = 000174, loss = 0.002041
grad_step = 000175, loss = 0.002050
grad_step = 000176, loss = 0.002015
grad_step = 000177, loss = 0.001989
grad_step = 000178, loss = 0.001999
grad_step = 000179, loss = 0.002020
grad_step = 000180, loss = 0.002021
grad_step = 000181, loss = 0.001995
grad_step = 000182, loss = 0.001974
grad_step = 000183, loss = 0.001975
grad_step = 000184, loss = 0.001989
grad_step = 000185, loss = 0.001996
grad_step = 000186, loss = 0.001989
grad_step = 000187, loss = 0.001973
grad_step = 000188, loss = 0.001963
grad_step = 000189, loss = 0.001973
grad_step = 000190, loss = 0.002063
grad_step = 000191, loss = 0.002411
grad_step = 000192, loss = 0.003317
grad_step = 000193, loss = 0.003036
grad_step = 000194, loss = 0.002420
grad_step = 000195, loss = 0.002654
grad_step = 000196, loss = 0.002211
grad_step = 000197, loss = 0.002493
grad_step = 000198, loss = 0.002500
grad_step = 000199, loss = 0.002017
grad_step = 000200, loss = 0.002526
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002137
grad_step = 000202, loss = 0.002220
grad_step = 000203, loss = 0.002154
grad_step = 000204, loss = 0.002137
grad_step = 000205, loss = 0.002174
grad_step = 000206, loss = 0.002000
grad_step = 000207, loss = 0.002207
grad_step = 000208, loss = 0.001977
grad_step = 000209, loss = 0.002119
grad_step = 000210, loss = 0.002037
grad_step = 000211, loss = 0.002049
grad_step = 000212, loss = 0.002033
grad_step = 000213, loss = 0.002011
grad_step = 000214, loss = 0.002058
grad_step = 000215, loss = 0.001953
grad_step = 000216, loss = 0.002047
grad_step = 000217, loss = 0.001966
grad_step = 000218, loss = 0.001997
grad_step = 000219, loss = 0.001979
grad_step = 000220, loss = 0.001973
grad_step = 000221, loss = 0.001981
grad_step = 000222, loss = 0.001941
grad_step = 000223, loss = 0.001985
grad_step = 000224, loss = 0.001939
grad_step = 000225, loss = 0.001952
grad_step = 000226, loss = 0.001949
grad_step = 000227, loss = 0.001936
grad_step = 000228, loss = 0.001943
grad_step = 000229, loss = 0.001924
grad_step = 000230, loss = 0.001936
grad_step = 000231, loss = 0.001925
grad_step = 000232, loss = 0.001913
grad_step = 000233, loss = 0.001926
grad_step = 000234, loss = 0.001910
grad_step = 000235, loss = 0.001910
grad_step = 000236, loss = 0.001911
grad_step = 000237, loss = 0.001899
grad_step = 000238, loss = 0.001906
grad_step = 000239, loss = 0.001900
grad_step = 000240, loss = 0.001891
grad_step = 000241, loss = 0.001897
grad_step = 000242, loss = 0.001893
grad_step = 000243, loss = 0.001886
grad_step = 000244, loss = 0.001887
grad_step = 000245, loss = 0.001884
grad_step = 000246, loss = 0.001880
grad_step = 000247, loss = 0.001881
grad_step = 000248, loss = 0.001878
grad_step = 000249, loss = 0.001872
grad_step = 000250, loss = 0.001871
grad_step = 000251, loss = 0.001872
grad_step = 000252, loss = 0.001871
grad_step = 000253, loss = 0.001871
grad_step = 000254, loss = 0.001875
grad_step = 000255, loss = 0.001877
grad_step = 000256, loss = 0.001877
grad_step = 000257, loss = 0.001877
grad_step = 000258, loss = 0.001875
grad_step = 000259, loss = 0.001871
grad_step = 000260, loss = 0.001864
grad_step = 000261, loss = 0.001857
grad_step = 000262, loss = 0.001851
grad_step = 000263, loss = 0.001846
grad_step = 000264, loss = 0.001842
grad_step = 000265, loss = 0.001839
grad_step = 000266, loss = 0.001838
grad_step = 000267, loss = 0.001837
grad_step = 000268, loss = 0.001837
grad_step = 000269, loss = 0.001838
grad_step = 000270, loss = 0.001843
grad_step = 000271, loss = 0.001851
grad_step = 000272, loss = 0.001867
grad_step = 000273, loss = 0.001898
grad_step = 000274, loss = 0.001946
grad_step = 000275, loss = 0.002020
grad_step = 000276, loss = 0.002066
grad_step = 000277, loss = 0.002057
grad_step = 000278, loss = 0.001957
grad_step = 000279, loss = 0.001851
grad_step = 000280, loss = 0.001828
grad_step = 000281, loss = 0.001870
grad_step = 000282, loss = 0.001908
grad_step = 000283, loss = 0.001895
grad_step = 000284, loss = 0.001843
grad_step = 000285, loss = 0.001811
grad_step = 000286, loss = 0.001830
grad_step = 000287, loss = 0.001864
grad_step = 000288, loss = 0.001859
grad_step = 000289, loss = 0.001817
grad_step = 000290, loss = 0.001793
grad_step = 000291, loss = 0.001809
grad_step = 000292, loss = 0.001830
grad_step = 000293, loss = 0.001827
grad_step = 000294, loss = 0.001805
grad_step = 000295, loss = 0.001793
grad_step = 000296, loss = 0.001801
grad_step = 000297, loss = 0.001810
grad_step = 000298, loss = 0.001804
grad_step = 000299, loss = 0.001787
grad_step = 000300, loss = 0.001778
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001781
grad_step = 000302, loss = 0.001788
grad_step = 000303, loss = 0.001788
grad_step = 000304, loss = 0.001780
grad_step = 000305, loss = 0.001773
grad_step = 000306, loss = 0.001774
grad_step = 000307, loss = 0.001780
grad_step = 000308, loss = 0.001787
grad_step = 000309, loss = 0.001788
grad_step = 000310, loss = 0.001795
grad_step = 000311, loss = 0.001797
grad_step = 000312, loss = 0.001813
grad_step = 000313, loss = 0.001825
grad_step = 000314, loss = 0.001833
grad_step = 000315, loss = 0.001826
grad_step = 000316, loss = 0.001804
grad_step = 000317, loss = 0.001779
grad_step = 000318, loss = 0.001761
grad_step = 000319, loss = 0.001756
grad_step = 000320, loss = 0.001762
grad_step = 000321, loss = 0.001768
grad_step = 000322, loss = 0.001771
grad_step = 000323, loss = 0.001770
grad_step = 000324, loss = 0.001775
grad_step = 000325, loss = 0.001788
grad_step = 000326, loss = 0.001835
grad_step = 000327, loss = 0.001896
grad_step = 000328, loss = 0.001999
grad_step = 000329, loss = 0.002008
grad_step = 000330, loss = 0.001963
grad_step = 000331, loss = 0.001887
grad_step = 000332, loss = 0.001825
grad_step = 000333, loss = 0.001779
grad_step = 000334, loss = 0.001766
grad_step = 000335, loss = 0.001822
grad_step = 000336, loss = 0.001877
grad_step = 000337, loss = 0.001846
grad_step = 000338, loss = 0.001783
grad_step = 000339, loss = 0.001743
grad_step = 000340, loss = 0.001761
grad_step = 000341, loss = 0.001801
grad_step = 000342, loss = 0.001806
grad_step = 000343, loss = 0.001772
grad_step = 000344, loss = 0.001733
grad_step = 000345, loss = 0.001725
grad_step = 000346, loss = 0.001748
grad_step = 000347, loss = 0.001768
grad_step = 000348, loss = 0.001764
grad_step = 000349, loss = 0.001742
grad_step = 000350, loss = 0.001729
grad_step = 000351, loss = 0.001734
grad_step = 000352, loss = 0.001746
grad_step = 000353, loss = 0.001749
grad_step = 000354, loss = 0.001733
grad_step = 000355, loss = 0.001718
grad_step = 000356, loss = 0.001712
grad_step = 000357, loss = 0.001718
grad_step = 000358, loss = 0.001723
grad_step = 000359, loss = 0.001720
grad_step = 000360, loss = 0.001712
grad_step = 000361, loss = 0.001703
grad_step = 000362, loss = 0.001701
grad_step = 000363, loss = 0.001704
grad_step = 000364, loss = 0.001707
grad_step = 000365, loss = 0.001709
grad_step = 000366, loss = 0.001707
grad_step = 000367, loss = 0.001705
grad_step = 000368, loss = 0.001706
grad_step = 000369, loss = 0.001717
grad_step = 000370, loss = 0.001738
grad_step = 000371, loss = 0.001785
grad_step = 000372, loss = 0.001844
grad_step = 000373, loss = 0.001929
grad_step = 000374, loss = 0.001948
grad_step = 000375, loss = 0.001906
grad_step = 000376, loss = 0.001808
grad_step = 000377, loss = 0.001754
grad_step = 000378, loss = 0.001751
grad_step = 000379, loss = 0.001730
grad_step = 000380, loss = 0.001729
grad_step = 000381, loss = 0.001759
grad_step = 000382, loss = 0.001790
grad_step = 000383, loss = 0.001831
grad_step = 000384, loss = 0.001795
grad_step = 000385, loss = 0.001770
grad_step = 000386, loss = 0.001715
grad_step = 000387, loss = 0.001701
grad_step = 000388, loss = 0.001724
grad_step = 000389, loss = 0.001741
grad_step = 000390, loss = 0.001723
grad_step = 000391, loss = 0.001678
grad_step = 000392, loss = 0.001672
grad_step = 000393, loss = 0.001695
grad_step = 000394, loss = 0.001693
grad_step = 000395, loss = 0.001679
grad_step = 000396, loss = 0.001676
grad_step = 000397, loss = 0.001689
grad_step = 000398, loss = 0.001703
grad_step = 000399, loss = 0.001690
grad_step = 000400, loss = 0.001676
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001666
grad_step = 000402, loss = 0.001665
grad_step = 000403, loss = 0.001666
grad_step = 000404, loss = 0.001661
grad_step = 000405, loss = 0.001652
grad_step = 000406, loss = 0.001646
grad_step = 000407, loss = 0.001646
grad_step = 000408, loss = 0.001649
grad_step = 000409, loss = 0.001649
grad_step = 000410, loss = 0.001647
grad_step = 000411, loss = 0.001643
grad_step = 000412, loss = 0.001641
grad_step = 000413, loss = 0.001643
grad_step = 000414, loss = 0.001648
grad_step = 000415, loss = 0.001656
grad_step = 000416, loss = 0.001665
grad_step = 000417, loss = 0.001681
grad_step = 000418, loss = 0.001705
grad_step = 000419, loss = 0.001747
grad_step = 000420, loss = 0.001814
grad_step = 000421, loss = 0.001896
grad_step = 000422, loss = 0.001992
grad_step = 000423, loss = 0.002017
grad_step = 000424, loss = 0.001961
grad_step = 000425, loss = 0.001811
grad_step = 000426, loss = 0.001683
grad_step = 000427, loss = 0.001647
grad_step = 000428, loss = 0.001709
grad_step = 000429, loss = 0.001768
grad_step = 000430, loss = 0.001757
grad_step = 000431, loss = 0.001686
grad_step = 000432, loss = 0.001635
grad_step = 000433, loss = 0.001650
grad_step = 000434, loss = 0.001682
grad_step = 000435, loss = 0.001685
grad_step = 000436, loss = 0.001647
grad_step = 000437, loss = 0.001624
grad_step = 000438, loss = 0.001638
grad_step = 000439, loss = 0.001652
grad_step = 000440, loss = 0.001650
grad_step = 000441, loss = 0.001620
grad_step = 000442, loss = 0.001603
grad_step = 000443, loss = 0.001613
grad_step = 000444, loss = 0.001627
grad_step = 000445, loss = 0.001631
grad_step = 000446, loss = 0.001610
grad_step = 000447, loss = 0.001595
grad_step = 000448, loss = 0.001597
grad_step = 000449, loss = 0.001607
grad_step = 000450, loss = 0.001608
grad_step = 000451, loss = 0.001597
grad_step = 000452, loss = 0.001588
grad_step = 000453, loss = 0.001588
grad_step = 000454, loss = 0.001593
grad_step = 000455, loss = 0.001596
grad_step = 000456, loss = 0.001592
grad_step = 000457, loss = 0.001587
grad_step = 000458, loss = 0.001583
grad_step = 000459, loss = 0.001583
grad_step = 000460, loss = 0.001585
grad_step = 000461, loss = 0.001586
grad_step = 000462, loss = 0.001584
grad_step = 000463, loss = 0.001579
grad_step = 000464, loss = 0.001574
grad_step = 000465, loss = 0.001570
grad_step = 000466, loss = 0.001567
grad_step = 000467, loss = 0.001567
grad_step = 000468, loss = 0.001568
grad_step = 000469, loss = 0.001569
grad_step = 000470, loss = 0.001569
grad_step = 000471, loss = 0.001571
grad_step = 000472, loss = 0.001573
grad_step = 000473, loss = 0.001580
grad_step = 000474, loss = 0.001587
grad_step = 000475, loss = 0.001608
grad_step = 000476, loss = 0.001620
grad_step = 000477, loss = 0.001652
grad_step = 000478, loss = 0.001648
grad_step = 000479, loss = 0.001656
grad_step = 000480, loss = 0.001646
grad_step = 000481, loss = 0.001662
grad_step = 000482, loss = 0.001697
grad_step = 000483, loss = 0.001742
grad_step = 000484, loss = 0.001762
grad_step = 000485, loss = 0.001737
grad_step = 000486, loss = 0.001679
grad_step = 000487, loss = 0.001596
grad_step = 000488, loss = 0.001543
grad_step = 000489, loss = 0.001541
grad_step = 000490, loss = 0.001574
grad_step = 000491, loss = 0.001602
grad_step = 000492, loss = 0.001600
grad_step = 000493, loss = 0.001579
grad_step = 000494, loss = 0.001568
grad_step = 000495, loss = 0.001578
grad_step = 000496, loss = 0.001621
grad_step = 000497, loss = 0.001634
grad_step = 000498, loss = 0.001667
grad_step = 000499, loss = 0.001617
grad_step = 000500, loss = 0.001582
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001582
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

  date_run                              2020-05-11 18:14:35.780448
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.190444
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 18:14:35.790986
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0760152
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 18:14:35.799232
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.129869
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 18:14:35.804895
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.155078
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
0   2020-05-11 18:14:02.403404  ...    mean_absolute_error
1   2020-05-11 18:14:02.407765  ...     mean_squared_error
2   2020-05-11 18:14:02.410934  ...  median_absolute_error
3   2020-05-11 18:14:02.414705  ...               r2_score
4   2020-05-11 18:14:13.761218  ...    mean_absolute_error
5   2020-05-11 18:14:13.765078  ...     mean_squared_error
6   2020-05-11 18:14:13.768363  ...  median_absolute_error
7   2020-05-11 18:14:13.771817  ...               r2_score
8   2020-05-11 18:14:35.780448  ...    mean_absolute_error
9   2020-05-11 18:14:35.790986  ...     mean_squared_error
10  2020-05-11 18:14:35.799232  ...  median_absolute_error
11  2020-05-11 18:14:35.804895  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 15%|█▌        | 1507328/9912422 [00:00<00:00, 14785985.92it/s]9920512it [00:00, 32582728.55it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1007646.51it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456463.25it/s]1654784it [00:00, 11603168.25it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208357.01it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcbd97fbfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb76f17f28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcbd9786ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb769ef048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb76f140b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb8c180e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb76f17e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb8c18ff28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb76f140b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcb769ef048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fcbd97fbfd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fcba4b0d208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=bdcb9d792e86c6f1eb8be2ea51df782227e7ae6ca08ab995bd1a3bec9969693a
  Stored in directory: /tmp/pip-ephem-wheel-cache-vcokffqa/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fcb3d7fda90> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1794048/17464789 [==>...........................] - ETA: 0s
 5750784/17464789 [========>.....................] - ETA: 0s
11763712/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 18:16:03.514008: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 18:16:03.518160: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 18:16:03.518878: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a0815759c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 18:16:03.518896: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4443 - accuracy: 0.5145 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5133 - accuracy: 0.5100
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.4903 - accuracy: 0.5115
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5225 - accuracy: 0.5094
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5785 - accuracy: 0.5058
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5576 - accuracy: 0.5071
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5578 - accuracy: 0.5071
11000/25000 [============>.................] - ETA: 4s - loss: 7.5760 - accuracy: 0.5059
12000/25000 [=============>................] - ETA: 3s - loss: 7.5976 - accuracy: 0.5045
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6064 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 2s - loss: 7.5930 - accuracy: 0.5048
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5909 - accuracy: 0.5049
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6161 - accuracy: 0.5033
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6198 - accuracy: 0.5031
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6239 - accuracy: 0.5028
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6383 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6352 - accuracy: 0.5020
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 8s 313us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 18:16:18.375213
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 18:16:18.375213  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 18:16:24.749220: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 18:16:24.754545: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 18:16:24.754808: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5561caf6a7a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 18:16:24.754889: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f734d764dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.1212 - crf_viterbi_accuracy: 0.0267 - val_loss: 2.0565 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7343a06898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8404 - accuracy: 0.4887
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7709 - accuracy: 0.4932
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7075 - accuracy: 0.4973
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7061 - accuracy: 0.4974
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7145 - accuracy: 0.4969
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 3s - loss: 7.6290 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 3s - loss: 7.6040 - accuracy: 0.5041
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6076 - accuracy: 0.5038
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6381 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6448 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 8s 310us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f73281142b0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<101:26:03, 2.36kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<71:14:02, 3.36kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<49:54:33, 4.80kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<34:55:23, 6.85kB/s].vector_cache/glove.6B.zip:   0%|          | 2.88M/862M [00:03<24:23:37, 9.79kB/s].vector_cache/glove.6B.zip:   1%|          | 6.42M/862M [00:04<17:00:26, 14.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:04<11:50:42, 20.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:04<8:15:07, 28.5kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.8M/862M [00:04<5:44:44, 40.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:04<4:00:18, 58.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:04<2:47:20, 83.0kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:04<1:56:40, 119kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:04<1:21:20, 169kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:04<56:43, 241kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 45.8M/862M [00:05<39:35, 344kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:05<27:40, 489kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:05<19:28, 693kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:07<16:08, 833kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:07<13:07, 1.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:07<09:37, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:09<09:16, 1.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:09<08:18, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:09<06:10, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:09<04:28, 2.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:11<45:04, 295kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:11<34:46, 383kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<25:00, 531kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:11<17:37, 752kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:13<19:07, 692kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:13<14:51, 890kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:13<10:42, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:15<10:19, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:15<08:39, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:15<06:19, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:17<07:32, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:17<06:35, 1.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:17<04:53, 2.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:19<06:31, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:19<05:53, 2.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:19<04:27, 2.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:20<06:10, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:21<07:51, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:21<06:12, 2.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:21<04:37, 2.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:22<06:22, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:23<07:46, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:23<06:11, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:23<04:31, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:24<07:05, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:25<08:07, 1.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:25<06:24, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:25<04:37, 2.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:26<09:35, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:27<09:49, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:27<07:39, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:27<05:33, 2.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<10:49, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<10:49, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<08:22, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<06:02, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<10:49, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<10:29, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<07:59, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<07:02, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<07:58, 1.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<06:14, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<04:39, 2.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:11, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<07:06, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<05:40, 2.20MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:35<04:08, 3.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<12:23, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<11:28, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<19:54, 623kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<14:56, 826kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<11:57, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<08:41, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<08:33, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<11:58, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<09:54, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<07:15, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<17:48, 685kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<15:20, 795kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<11:20, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<08:07, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<05:56, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<40:05, 303kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<30:31, 397kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<21:53, 553kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<15:29, 780kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<14:55, 808kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<12:55, 933kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<09:35, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:47<06:51, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<11:10, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<10:53, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<08:16, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<08:36, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<08:55, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<07:00, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<05:02, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<09:39, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<09:09, 1.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<06:56, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:53<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<10:43, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<10:15, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<07:53, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<05:40, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<10:40, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<10:28, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<08:04, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<05:49, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<10:19, 1.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<10:03, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<07:45, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<10:32, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<10:04, 1.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<07:42, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<05:33, 2.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<10:56, 1.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<09:54, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<07:25, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<05:25, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<07:06, 1.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<07:17, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:04<05:41, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<04:06, 2.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<21:04, 539kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<17:05, 665kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<12:30, 908kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<08:53, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<19:07, 591kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<15:42, 719kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<11:33, 976kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<08:13, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<17:23, 646kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<14:31, 773kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<10:42, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<07:45, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<07:59, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<07:51, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<06:03, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<04:23, 2.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<15:44, 705kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<13:15, 836kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<09:45, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<06:58, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<09:55, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<09:10, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<06:58, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<25:05, 437kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<20:06, 545kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<14:36, 749kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<10:20, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<11:46, 925kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<10:27, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<07:47, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<05:34, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<11:08, 970kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<10:25, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<07:56, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<05:41, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<10:13, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<09:15, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<07:00, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<05:01, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<15:22, 695kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<13:29, 792kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<10:02, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<07:12, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<07:56, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<08:01, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<06:13, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<04:30, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<09:44, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<09:23, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<07:13, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:10, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<09:39, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<09:11, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<06:57, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<04:59, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<08:26, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<08:27, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<06:26, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<06:07, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<06:48, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<05:21, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<03:55, 2.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<05:49, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<06:04, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<04:41, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<03:25, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<07:29, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<07:14, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<05:33, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:00, 2.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<18:08, 558kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<14:40, 690kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<10:44, 941kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<07:38, 1.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:44<20:29, 491kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<16:12, 620kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<11:48, 850kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<08:22, 1.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<24:19, 411kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<18:53, 529kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<13:38, 731kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<09:36, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<17:47, 558kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<14:51, 667kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<10:58, 902kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<07:47, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<12:04, 816kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<10:37, 927kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<07:53, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<05:46, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<06:13, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<03:26, 2.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<07:12, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<06:47, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:54<05:09, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<03:42, 2.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<10:30, 918kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<09:42, 993kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<07:17, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<05:11, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<09:44, 983kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<08:56, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:58<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<04:51, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<11:43, 811kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<10:18, 922kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<07:44, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<05:30, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<11:40, 808kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<10:28, 901kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<07:48, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<08:36, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<08:20, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<06:20, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<04:33, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<07:07, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<07:07, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<05:31, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<03:59, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<09:51, 937kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<09:02, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<06:53, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<04:54, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<09:25, 971kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<08:44, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<06:39, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<04:46, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<10:33, 861kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<09:29, 957kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<07:06, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<05:05, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<07:00, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<06:55, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<05:15, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<03:54, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<04:50, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<05:30, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<04:18, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<03:09, 2.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<05:04, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<05:27, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<04:12, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<03:04, 2.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<05:57, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<04:35, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<03:19, 2.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<05:58, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<05:57, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<04:33, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<03:16, 2.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<09:42, 894kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<08:28, 1.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<06:21, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<04:31, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<24:50, 346kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<19:17, 446kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<13:58, 615kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<09:51, 867kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<15:01, 568kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<12:24, 688kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<09:08, 932kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<06:28, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<14:10, 597kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<11:31, 735kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<08:23, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<05:58, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<08:07, 1.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<07:37, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<04:08, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<06:24, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<06:03, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<04:37, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:20, 2.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<06:23, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<06:23, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<04:51, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:32, 2.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:57, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:57, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<02:44, 2.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<11:19, 717kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<09:23, 864kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<06:57, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<06:11, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<05:47, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<04:24, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:24, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:56, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<03:55, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<02:50, 2.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<08:38, 915kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<07:44, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<05:49, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:10, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<15:21, 511kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<12:26, 630kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:47<09:07, 858kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<06:27, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<13:22, 581kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<11:07, 699kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<08:11, 947kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<05:48, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<13:02, 591kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<10:30, 733kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:51<07:39, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<05:27, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<07:00, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<06:42, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<08:34, 883kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<07:18, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<05:23, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:50, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<10:14, 732kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<10:53, 688kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<08:29, 882kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<06:08, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<05:46, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<05:21, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<02:54, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:01<06:15, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<06:05, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:40, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:21, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<08:20, 874kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<07:29, 974kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<05:38, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<04:00, 1.80MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<09:58, 724kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<08:16, 872kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<06:06, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<05:27, 1.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<05:33, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<03:02, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<06:47, 1.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<06:01, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:31, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<04:20, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<06:28, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<05:20, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:54, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<04:07, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<04:25, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:25, 2.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<02:27, 2.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<06:04, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<05:46, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<04:25, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<03:09, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<09:00, 756kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<07:29, 907kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<05:32, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:57, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<06:02, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<05:25, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<04:30, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:16, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:22, 2.80MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<25:25, 262kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<18:57, 352kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<13:29, 493kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<09:28, 698kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<10:37, 621kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<08:58, 736kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<06:38, 991kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:23<04:42, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<08:28, 771kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<07:19, 892kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<05:25, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<03:50, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:26<08:43, 740kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<07:29, 862kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<05:32, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<03:56, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<06:28, 987kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<05:53, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<03:12, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<04:03, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<04:15, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<03:18, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<02:24, 2.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<03:39, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<03:38, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<02:49, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<03:01, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<03:33, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<02:04, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<06:27, 948kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<05:33, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<04:08, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:56, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<09:25, 642kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<08:02, 752kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<05:56, 1.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<04:13, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<05:19, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<04:59, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:45, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<02:42, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<04:02, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<03:49, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:52, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<02:05, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<04:36, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<04:39, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<03:46, 1.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<02:47, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<02:00, 2.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<1:23:36, 69.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<59:25, 97.1kB/s]  .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<41:40, 138kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<29:06, 197kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<23:02, 248kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<17:19, 329kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<12:22, 460kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<08:43, 649kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<07:44, 727kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<06:20, 889kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:38, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:52<04:12, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:52<04:07, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<03:08, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:16, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<03:42, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<03:27, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:35, 2.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<02:46, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<04:12, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<03:31, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<02:35, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:58, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<03:09, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<02:26, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<01:46, 2.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<03:51, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<01:42, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<1:43:30, 50.4kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<1:13:10, 71.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<51:12, 102kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<35:41, 145kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<28:03, 184kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<20:21, 253kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<14:21, 357kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<10:01, 507kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<15:55, 319kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<12:03, 421kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<08:38, 586kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<06:03, 829kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<18:12, 275kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<13:41, 366kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<09:48, 510kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<06:57, 716kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<04:56, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<08:58, 552kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<07:12, 686kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<05:16, 935kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:43, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<06:20, 769kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<05:20, 911kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:57, 1.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<03:32, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<03:25, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<02:36, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:35, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:42, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<01:30, 3.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<14:02, 333kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:18<10:43, 436kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<07:42, 604kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<05:24, 853kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<06:34, 700kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<05:30, 835kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<04:03, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<02:53, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<04:00, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<03:41, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<02:46, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<01:58, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<34:09, 131kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<24:43, 180kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<17:28, 255kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<12:48, 343kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<09:47, 449kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:26<07:02, 622kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<05:35, 775kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<05:41, 761kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<04:22, 988kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<03:08, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<03:11, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<03:02, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:30<02:19, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<02:17, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<02:25, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:53, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<01:21, 3.05MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<06:41, 617kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<05:26, 757kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<04:11, 982kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<03:01, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<02:08, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<3:52:24, 17.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<2:43:04, 24.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<1:53:46, 35.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<1:19:28, 50.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<56:06, 70.9kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<39:12, 101kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<27:53, 140kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<20:52, 187kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<14:56, 261kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<10:28, 370kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<08:16, 465kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<06:29, 592kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<04:40, 820kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<03:19, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<03:35, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<03:05, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:21, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<01:40, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<05:40, 653kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<04:38, 799kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<03:22, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<02:25, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:47, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:35, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:56, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:24, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<02:23, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<02:17, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:44, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<01:16, 2.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<02:18, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<02:14, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<01:42, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:14, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:37, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:25, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:49, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:18, 2.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:30, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:53, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:53, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:27, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:05, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:37, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:41, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:18, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<00:56, 3.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:11, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:39, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:03<01:40, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:42, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:26, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<01:10, 2.61MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<00:58, 3.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<00:50, 3.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<00:44, 4.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<03:19, 909kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<03:57, 763kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<03:11, 946kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<02:22, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:24, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<01:06, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<00:56, 3.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<07:40, 385kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<06:07, 482kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<04:28, 657kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<03:15, 898kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<02:23, 1.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:48, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<01:23, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<06:18, 458kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<05:48, 496kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<04:26, 649kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<03:16, 874kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<02:24, 1.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:49, 1.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:23, 2.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:05, 2.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<02:23, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<03:02, 923kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<02:28, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:51, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:24, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<01:07, 2.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<00:54, 3.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<02:06, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<02:49, 974kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<02:18, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:44, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:19, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:02, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<00:51, 3.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<04:41, 571kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<04:34, 584kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<03:31, 757kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<02:35, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:54, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:24, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:11, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<02:13, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<02:49, 922kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<02:18, 1.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:43, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<01:19, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:02, 2.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<00:50, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<04:05, 621kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<04:06, 619kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<03:10, 797kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<02:19, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:43, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:20, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:01, 2.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<00:50, 2.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<10:03, 246kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<08:15, 299kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<06:01, 409kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<04:19, 567kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<03:06, 782kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<02:16, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:40, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<04:04, 590kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<04:02, 594kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<03:05, 774kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<02:16, 1.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<01:40, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<01:16, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<00:58, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<09:46, 239kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<07:53, 295kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<05:47, 402kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<04:07, 560kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<02:58, 774kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<02:10, 1.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:35, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<02:53, 784kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<03:01, 749kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<02:19, 968kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:43, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:18, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:59, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:47, 2.78MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:34, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<02:06, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:41, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:15, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<00:56, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:45, 2.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:35, 3.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<06:16, 339kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<05:11, 409kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<03:47, 558kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<02:45, 766kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<01:58, 1.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:27, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:04, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<04:26, 462kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<03:50, 536kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<02:51, 716kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<02:03, 985kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<01:29, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:06, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<02:00, 994kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<02:04, 961kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:35<01:36, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<01:10, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<00:52, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:26, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<02:20, 822kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<02:00, 956kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:34, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<01:08, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:38, 2.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<02:26, 760kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<02:12, 835kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:38, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:39<01:13, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:52, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:40, 2.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<03:00, 593kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<03:06, 573kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<02:23, 741kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<01:43, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:13, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<01:41, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<01:34, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<01:10, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:52, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:43<00:38, 2.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<01:13, 1.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<01:32, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<01:14, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:45<00:54, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:39, 2.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<03:45, 418kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<03:13, 487kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<02:23, 656kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<01:43, 902kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<01:12, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:30, 1.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:37, 927kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:14, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:38, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<01:15, 1.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<01:20, 1.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:03, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:51<00:45, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:53, 1.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:52<01:02, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:49, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:35, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:46, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:55, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:44, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:31, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<01:10, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:57, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:31, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:38, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:42, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:35, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:26, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:19, 3.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:55, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:53, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:40, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:28, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:58, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:53, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:40, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:27, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<01:01, 934kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:56, 1.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:42, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:30, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:32, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:32, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:24, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:07<00:17, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:32, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:31, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:08<00:24, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:16, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<01:05, 679kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:56, 784kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:41, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:28, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:30, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:29, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:22, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:15, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:23, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:19, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:12, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:33, 963kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:30, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:22, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:14, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:25, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:21, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:15, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:14, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:09, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:09, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:10, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:07, 2.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 1.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:05, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:05, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:04, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:03, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:02, 2.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:00, 3.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:31, 101kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:21, 140kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:10, 198kB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 818/400000 [00:00<00:48, 8173.23it/s]  0%|          | 1679/400000 [00:00<00:48, 8295.93it/s]  1%|          | 2501/400000 [00:00<00:48, 8272.60it/s]  1%|          | 3288/400000 [00:00<00:48, 8146.57it/s]  1%|          | 4104/400000 [00:00<00:48, 8148.05it/s]  1%|          | 4926/400000 [00:00<00:48, 8167.87it/s]  1%|▏         | 5764/400000 [00:00<00:47, 8228.72it/s]  2%|▏         | 6571/400000 [00:00<00:48, 8178.14it/s]  2%|▏         | 7407/400000 [00:00<00:47, 8231.61it/s]  2%|▏         | 8262/400000 [00:01<00:47, 8323.29it/s]  2%|▏         | 9107/400000 [00:01<00:46, 8360.90it/s]  2%|▏         | 9949/400000 [00:01<00:46, 8376.84it/s]  3%|▎         | 10776/400000 [00:01<00:46, 8324.49it/s]  3%|▎         | 11613/400000 [00:01<00:46, 8337.98it/s]  3%|▎         | 12460/400000 [00:01<00:46, 8376.56it/s]  3%|▎         | 13309/400000 [00:01<00:45, 8408.34it/s]  4%|▎         | 14148/400000 [00:01<00:46, 8326.43it/s]  4%|▎         | 14980/400000 [00:01<00:46, 8233.52it/s]  4%|▍         | 15813/400000 [00:01<00:46, 8260.20it/s]  4%|▍         | 16639/400000 [00:02<00:46, 8242.29it/s]  4%|▍         | 17479/400000 [00:02<00:46, 8286.50it/s]  5%|▍         | 18335/400000 [00:02<00:45, 8366.28it/s]  5%|▍         | 19197/400000 [00:02<00:45, 8439.99it/s]  5%|▌         | 20042/400000 [00:02<00:45, 8433.54it/s]  5%|▌         | 20891/400000 [00:02<00:44, 8447.96it/s]  5%|▌         | 21757/400000 [00:02<00:44, 8509.89it/s]  6%|▌         | 22636/400000 [00:02<00:43, 8590.90it/s]  6%|▌         | 23512/400000 [00:02<00:43, 8639.31it/s]  6%|▌         | 24390/400000 [00:02<00:43, 8680.41it/s]  6%|▋         | 25259/400000 [00:03<00:43, 8671.77it/s]  7%|▋         | 26136/400000 [00:03<00:42, 8700.57it/s]  7%|▋         | 27016/400000 [00:03<00:42, 8727.54it/s]  7%|▋         | 27889/400000 [00:03<00:42, 8710.69it/s]  7%|▋         | 28761/400000 [00:03<00:42, 8710.73it/s]  7%|▋         | 29633/400000 [00:03<00:43, 8594.08it/s]  8%|▊         | 30493/400000 [00:03<00:45, 8164.09it/s]  8%|▊         | 31315/400000 [00:03<00:45, 8126.38it/s]  8%|▊         | 32132/400000 [00:03<00:45, 8137.14it/s]  8%|▊         | 32949/400000 [00:03<00:45, 8057.35it/s]  8%|▊         | 33808/400000 [00:04<00:44, 8208.80it/s]  9%|▊         | 34685/400000 [00:04<00:43, 8369.41it/s]  9%|▉         | 35568/400000 [00:04<00:42, 8502.04it/s]  9%|▉         | 36451/400000 [00:04<00:42, 8596.51it/s]  9%|▉         | 37333/400000 [00:04<00:41, 8657.24it/s] 10%|▉         | 38204/400000 [00:04<00:41, 8670.15it/s] 10%|▉         | 39085/400000 [00:04<00:41, 8710.94it/s] 10%|▉         | 39970/400000 [00:04<00:41, 8750.94it/s] 10%|█         | 40851/400000 [00:04<00:40, 8766.09it/s] 10%|█         | 41734/400000 [00:04<00:40, 8784.33it/s] 11%|█         | 42613/400000 [00:05<00:40, 8785.53it/s] 11%|█         | 43492/400000 [00:05<00:40, 8766.66it/s] 11%|█         | 44369/400000 [00:05<00:40, 8765.90it/s] 11%|█▏        | 45246/400000 [00:05<00:41, 8580.88it/s] 12%|█▏        | 46106/400000 [00:05<00:41, 8573.97it/s] 12%|█▏        | 46976/400000 [00:05<00:41, 8608.94it/s] 12%|█▏        | 47859/400000 [00:05<00:40, 8672.08it/s] 12%|█▏        | 48734/400000 [00:05<00:40, 8694.34it/s] 12%|█▏        | 49618/400000 [00:05<00:40, 8735.97it/s] 13%|█▎        | 50492/400000 [00:05<00:42, 8319.98it/s] 13%|█▎        | 51357/400000 [00:06<00:41, 8414.22it/s] 13%|█▎        | 52239/400000 [00:06<00:40, 8529.38it/s] 13%|█▎        | 53116/400000 [00:06<00:40, 8599.26it/s] 13%|█▎        | 53987/400000 [00:06<00:40, 8631.26it/s] 14%|█▎        | 54852/400000 [00:06<00:40, 8570.54it/s] 14%|█▍        | 55711/400000 [00:06<00:41, 8270.59it/s] 14%|█▍        | 56542/400000 [00:06<00:41, 8227.41it/s] 14%|█▍        | 57399/400000 [00:06<00:41, 8325.48it/s] 15%|█▍        | 58267/400000 [00:06<00:40, 8428.22it/s] 15%|█▍        | 59148/400000 [00:06<00:39, 8538.54it/s] 15%|█▌        | 60004/400000 [00:07<00:40, 8479.54it/s] 15%|█▌        | 60854/400000 [00:07<00:40, 8467.93it/s] 15%|█▌        | 61702/400000 [00:07<00:40, 8381.95it/s] 16%|█▌        | 62541/400000 [00:07<00:40, 8258.05it/s] 16%|█▌        | 63401/400000 [00:07<00:40, 8355.64it/s] 16%|█▌        | 64242/400000 [00:07<00:40, 8370.62it/s] 16%|█▋        | 65080/400000 [00:07<00:40, 8192.28it/s] 16%|█▋        | 65921/400000 [00:07<00:40, 8254.54it/s] 17%|█▋        | 66758/400000 [00:07<00:40, 8288.18it/s] 17%|█▋        | 67588/400000 [00:08<00:40, 8196.44it/s] 17%|█▋        | 68415/400000 [00:08<00:40, 8216.69it/s] 17%|█▋        | 69281/400000 [00:08<00:39, 8342.71it/s] 18%|█▊        | 70162/400000 [00:08<00:38, 8476.66it/s] 18%|█▊        | 71032/400000 [00:08<00:38, 8541.92it/s] 18%|█▊        | 71913/400000 [00:08<00:38, 8620.55it/s] 18%|█▊        | 72776/400000 [00:08<00:38, 8609.27it/s] 18%|█▊        | 73651/400000 [00:08<00:37, 8650.23it/s] 19%|█▊        | 74518/400000 [00:08<00:37, 8655.43it/s] 19%|█▉        | 75384/400000 [00:08<00:38, 8465.86it/s] 19%|█▉        | 76234/400000 [00:09<00:38, 8473.30it/s] 19%|█▉        | 77083/400000 [00:09<00:38, 8431.00it/s] 19%|█▉        | 77951/400000 [00:09<00:37, 8503.81it/s] 20%|█▉        | 78824/400000 [00:09<00:37, 8570.30it/s] 20%|█▉        | 79702/400000 [00:09<00:37, 8632.14it/s] 20%|██        | 80579/400000 [00:09<00:36, 8671.75it/s] 20%|██        | 81447/400000 [00:09<00:36, 8671.19it/s] 21%|██        | 82315/400000 [00:09<00:36, 8643.54it/s] 21%|██        | 83187/400000 [00:09<00:36, 8666.32it/s] 21%|██        | 84054/400000 [00:09<00:36, 8642.30it/s] 21%|██        | 84921/400000 [00:10<00:36, 8649.73it/s] 21%|██▏       | 85787/400000 [00:10<00:36, 8608.93it/s] 22%|██▏       | 86648/400000 [00:10<00:36, 8561.95it/s] 22%|██▏       | 87505/400000 [00:10<00:36, 8551.32it/s] 22%|██▏       | 88365/400000 [00:10<00:36, 8564.97it/s] 22%|██▏       | 89222/400000 [00:10<00:36, 8532.99it/s] 23%|██▎       | 90081/400000 [00:10<00:36, 8547.24it/s] 23%|██▎       | 90940/400000 [00:10<00:36, 8557.36it/s] 23%|██▎       | 91820/400000 [00:10<00:35, 8628.14it/s] 23%|██▎       | 92683/400000 [00:10<00:38, 8078.26it/s] 23%|██▎       | 93520/400000 [00:11<00:37, 8162.83it/s] 24%|██▎       | 94370/400000 [00:11<00:36, 8260.80it/s] 24%|██▍       | 95202/400000 [00:11<00:36, 8272.28it/s] 24%|██▍       | 96039/400000 [00:11<00:36, 8300.32it/s] 24%|██▍       | 96872/400000 [00:11<00:36, 8290.08it/s] 24%|██▍       | 97703/400000 [00:11<00:37, 8115.91it/s] 25%|██▍       | 98541/400000 [00:11<00:36, 8191.19it/s] 25%|██▍       | 99371/400000 [00:11<00:36, 8222.52it/s] 25%|██▌       | 100211/400000 [00:11<00:36, 8273.64it/s] 25%|██▌       | 101040/400000 [00:11<00:36, 8148.16it/s] 25%|██▌       | 101912/400000 [00:12<00:35, 8309.44it/s] 26%|██▌       | 102783/400000 [00:12<00:35, 8424.06it/s] 26%|██▌       | 103627/400000 [00:12<00:35, 8412.84it/s] 26%|██▌       | 104470/400000 [00:12<00:35, 8379.43it/s] 26%|██▋       | 105318/400000 [00:12<00:35, 8408.15it/s] 27%|██▋       | 106163/400000 [00:12<00:34, 8418.20it/s] 27%|██▋       | 107010/400000 [00:12<00:34, 8431.30it/s] 27%|██▋       | 107854/400000 [00:12<00:34, 8352.26it/s] 27%|██▋       | 108705/400000 [00:12<00:34, 8397.81it/s] 27%|██▋       | 109570/400000 [00:12<00:34, 8471.87it/s] 28%|██▊       | 110449/400000 [00:13<00:33, 8562.66it/s] 28%|██▊       | 111307/400000 [00:13<00:33, 8567.47it/s] 28%|██▊       | 112183/400000 [00:13<00:33, 8622.74it/s] 28%|██▊       | 113049/400000 [00:13<00:33, 8632.96it/s] 28%|██▊       | 113921/400000 [00:13<00:33, 8656.60it/s] 29%|██▊       | 114798/400000 [00:13<00:32, 8688.32it/s] 29%|██▉       | 115667/400000 [00:13<00:32, 8639.95it/s] 29%|██▉       | 116532/400000 [00:13<00:32, 8634.10it/s] 29%|██▉       | 117396/400000 [00:13<00:33, 8536.69it/s] 30%|██▉       | 118251/400000 [00:13<00:33, 8432.42it/s] 30%|██▉       | 119105/400000 [00:14<00:33, 8461.61it/s] 30%|██▉       | 119977/400000 [00:14<00:32, 8537.17it/s] 30%|███       | 120844/400000 [00:14<00:32, 8576.34it/s] 30%|███       | 121711/400000 [00:14<00:32, 8604.15it/s] 31%|███       | 122572/400000 [00:14<00:32, 8571.13it/s] 31%|███       | 123430/400000 [00:14<00:32, 8395.83it/s] 31%|███       | 124304/400000 [00:14<00:32, 8495.59it/s] 31%|███▏      | 125181/400000 [00:14<00:32, 8574.85it/s] 32%|███▏      | 126056/400000 [00:14<00:31, 8626.46it/s] 32%|███▏      | 126940/400000 [00:14<00:31, 8687.03it/s] 32%|███▏      | 127810/400000 [00:15<00:31, 8674.54it/s] 32%|███▏      | 128678/400000 [00:15<00:31, 8549.98it/s] 32%|███▏      | 129558/400000 [00:15<00:31, 8623.02it/s] 33%|███▎      | 130438/400000 [00:15<00:31, 8667.60it/s] 33%|███▎      | 131312/400000 [00:15<00:30, 8686.86it/s] 33%|███▎      | 132196/400000 [00:15<00:30, 8730.48it/s] 33%|███▎      | 133070/400000 [00:15<00:30, 8732.03it/s] 33%|███▎      | 133949/400000 [00:15<00:30, 8748.48it/s] 34%|███▎      | 134825/400000 [00:15<00:30, 8721.12it/s] 34%|███▍      | 135706/400000 [00:15<00:30, 8746.34it/s] 34%|███▍      | 136581/400000 [00:16<00:30, 8652.13it/s] 34%|███▍      | 137447/400000 [00:16<00:30, 8471.40it/s] 35%|███▍      | 138296/400000 [00:16<00:31, 8434.37it/s] 35%|███▍      | 139145/400000 [00:16<00:30, 8450.63it/s] 35%|███▍      | 139995/400000 [00:16<00:30, 8464.01it/s] 35%|███▌      | 140856/400000 [00:16<00:30, 8505.43it/s] 35%|███▌      | 141729/400000 [00:16<00:30, 8568.73it/s] 36%|███▌      | 142611/400000 [00:16<00:29, 8641.46it/s] 36%|███▌      | 143489/400000 [00:16<00:29, 8682.18it/s] 36%|███▌      | 144371/400000 [00:17<00:29, 8721.23it/s] 36%|███▋      | 145247/400000 [00:17<00:29, 8730.37it/s] 37%|███▋      | 146125/400000 [00:17<00:29, 8743.43it/s] 37%|███▋      | 147003/400000 [00:17<00:28, 8754.30it/s] 37%|███▋      | 147879/400000 [00:17<00:28, 8750.36it/s] 37%|███▋      | 148755/400000 [00:17<00:28, 8703.82it/s] 37%|███▋      | 149630/400000 [00:17<00:28, 8714.78it/s] 38%|███▊      | 150505/400000 [00:17<00:28, 8723.51it/s] 38%|███▊      | 151386/400000 [00:17<00:28, 8748.31it/s] 38%|███▊      | 152269/400000 [00:17<00:28, 8770.71it/s] 38%|███▊      | 153153/400000 [00:18<00:28, 8790.53it/s] 39%|███▊      | 154038/400000 [00:18<00:27, 8806.76it/s] 39%|███▊      | 154920/400000 [00:18<00:27, 8807.63it/s] 39%|███▉      | 155801/400000 [00:18<00:27, 8721.90it/s] 39%|███▉      | 156674/400000 [00:18<00:27, 8695.80it/s] 39%|███▉      | 157544/400000 [00:18<00:28, 8647.93it/s] 40%|███▉      | 158409/400000 [00:18<00:28, 8605.64it/s] 40%|███▉      | 159270/400000 [00:18<00:27, 8602.82it/s] 40%|████      | 160131/400000 [00:18<00:27, 8572.54it/s] 40%|████      | 160989/400000 [00:18<00:28, 8528.37it/s] 40%|████      | 161842/400000 [00:19<00:28, 8497.42it/s] 41%|████      | 162692/400000 [00:19<00:28, 8416.72it/s] 41%|████      | 163538/400000 [00:19<00:28, 8428.98it/s] 41%|████      | 164413/400000 [00:19<00:27, 8522.50it/s] 41%|████▏     | 165297/400000 [00:19<00:27, 8613.06it/s] 42%|████▏     | 166183/400000 [00:19<00:26, 8683.48it/s] 42%|████▏     | 167067/400000 [00:19<00:26, 8727.78it/s] 42%|████▏     | 167944/400000 [00:19<00:26, 8737.48it/s] 42%|████▏     | 168821/400000 [00:19<00:26, 8746.11it/s] 42%|████▏     | 169702/400000 [00:19<00:26, 8763.75it/s] 43%|████▎     | 170584/400000 [00:20<00:26, 8779.08it/s] 43%|████▎     | 171463/400000 [00:20<00:26, 8648.69it/s] 43%|████▎     | 172329/400000 [00:20<00:27, 8371.39it/s] 43%|████▎     | 173209/400000 [00:20<00:26, 8494.68it/s] 44%|████▎     | 174061/400000 [00:20<00:26, 8465.51it/s] 44%|████▎     | 174941/400000 [00:20<00:26, 8562.21it/s] 44%|████▍     | 175799/400000 [00:20<00:26, 8478.13it/s] 44%|████▍     | 176648/400000 [00:20<00:26, 8437.31it/s] 44%|████▍     | 177509/400000 [00:20<00:26, 8485.80it/s] 45%|████▍     | 178359/400000 [00:20<00:26, 8320.97it/s] 45%|████▍     | 179193/400000 [00:21<00:26, 8284.57it/s] 45%|████▌     | 180059/400000 [00:21<00:26, 8391.98it/s] 45%|████▌     | 180900/400000 [00:21<00:26, 8376.35it/s] 45%|████▌     | 181744/400000 [00:21<00:25, 8395.31it/s] 46%|████▌     | 182623/400000 [00:21<00:25, 8509.84it/s] 46%|████▌     | 183498/400000 [00:21<00:25, 8580.44it/s] 46%|████▌     | 184357/400000 [00:21<00:26, 8288.54it/s] 46%|████▋     | 185189/400000 [00:21<00:25, 8294.65it/s] 47%|████▋     | 186021/400000 [00:21<00:26, 8150.31it/s] 47%|████▋     | 186894/400000 [00:21<00:25, 8314.00it/s] 47%|████▋     | 187774/400000 [00:22<00:25, 8453.88it/s] 47%|████▋     | 188654/400000 [00:22<00:24, 8552.55it/s] 47%|████▋     | 189531/400000 [00:22<00:24, 8616.23it/s] 48%|████▊     | 190394/400000 [00:22<00:24, 8490.94it/s] 48%|████▊     | 191265/400000 [00:22<00:24, 8554.30it/s] 48%|████▊     | 192133/400000 [00:22<00:24, 8589.82it/s] 48%|████▊     | 192993/400000 [00:22<00:24, 8558.33it/s] 48%|████▊     | 193850/400000 [00:22<00:24, 8506.59it/s] 49%|████▊     | 194702/400000 [00:22<00:24, 8472.49it/s] 49%|████▉     | 195571/400000 [00:22<00:23, 8534.73it/s] 49%|████▉     | 196448/400000 [00:23<00:23, 8601.63it/s] 49%|████▉     | 197329/400000 [00:23<00:23, 8661.51it/s] 50%|████▉     | 198196/400000 [00:23<00:23, 8660.17it/s] 50%|████▉     | 199071/400000 [00:23<00:23, 8685.55it/s] 50%|████▉     | 199940/400000 [00:23<00:23, 8686.72it/s] 50%|█████     | 200812/400000 [00:23<00:22, 8696.54it/s] 50%|█████     | 201697/400000 [00:23<00:22, 8739.22it/s] 51%|█████     | 202577/400000 [00:23<00:22, 8756.71it/s] 51%|█████     | 203456/400000 [00:23<00:22, 8763.17it/s] 51%|█████     | 204335/400000 [00:23<00:22, 8769.51it/s] 51%|█████▏    | 205216/400000 [00:24<00:22, 8779.93it/s] 52%|█████▏    | 206098/400000 [00:24<00:22, 8789.26it/s] 52%|█████▏    | 206980/400000 [00:24<00:21, 8797.98it/s] 52%|█████▏    | 207860/400000 [00:24<00:21, 8792.64it/s] 52%|█████▏    | 208744/400000 [00:24<00:21, 8805.86it/s] 52%|█████▏    | 209625/400000 [00:24<00:21, 8802.57it/s] 53%|█████▎    | 210511/400000 [00:24<00:21, 8818.50it/s] 53%|█████▎    | 211393/400000 [00:24<00:21, 8802.91it/s] 53%|█████▎    | 212274/400000 [00:24<00:21, 8761.47it/s] 53%|█████▎    | 213151/400000 [00:24<00:21, 8688.00it/s] 54%|█████▎    | 214020/400000 [00:25<00:21, 8633.72it/s] 54%|█████▎    | 214884/400000 [00:25<00:21, 8594.62it/s] 54%|█████▍    | 215744/400000 [00:25<00:21, 8481.36it/s] 54%|█████▍    | 216593/400000 [00:25<00:22, 8300.66it/s] 54%|█████▍    | 217451/400000 [00:25<00:21, 8380.20it/s] 55%|█████▍    | 218304/400000 [00:25<00:21, 8422.80it/s] 55%|█████▍    | 219148/400000 [00:25<00:22, 8046.27it/s] 55%|█████▍    | 219957/400000 [00:25<00:22, 8057.07it/s] 55%|█████▌    | 220819/400000 [00:25<00:21, 8217.59it/s] 55%|█████▌    | 221690/400000 [00:26<00:21, 8357.08it/s] 56%|█████▌    | 222559/400000 [00:26<00:20, 8453.45it/s] 56%|█████▌    | 223430/400000 [00:26<00:20, 8527.22it/s] 56%|█████▌    | 224304/400000 [00:26<00:20, 8587.58it/s] 56%|█████▋    | 225164/400000 [00:26<00:20, 8580.16it/s] 57%|█████▋    | 226023/400000 [00:26<00:20, 8463.70it/s] 57%|█████▋    | 226873/400000 [00:26<00:20, 8473.05it/s] 57%|█████▋    | 227721/400000 [00:26<00:20, 8403.97it/s] 57%|█████▋    | 228562/400000 [00:26<00:20, 8404.28it/s] 57%|█████▋    | 229443/400000 [00:26<00:20, 8520.53it/s] 58%|█████▊    | 230323/400000 [00:27<00:19, 8600.00it/s] 58%|█████▊    | 231184/400000 [00:27<00:19, 8498.87it/s] 58%|█████▊    | 232035/400000 [00:27<00:19, 8437.69it/s] 58%|█████▊    | 232909/400000 [00:27<00:19, 8524.99it/s] 58%|█████▊    | 233763/400000 [00:27<00:19, 8359.43it/s] 59%|█████▊    | 234604/400000 [00:27<00:19, 8373.91it/s] 59%|█████▉    | 235461/400000 [00:27<00:19, 8431.10it/s] 59%|█████▉    | 236305/400000 [00:27<00:19, 8364.86it/s] 59%|█████▉    | 237143/400000 [00:27<00:19, 8177.52it/s] 59%|█████▉    | 237995/400000 [00:27<00:19, 8275.60it/s] 60%|█████▉    | 238855/400000 [00:28<00:19, 8370.17it/s] 60%|█████▉    | 239694/400000 [00:28<00:19, 8376.05it/s] 60%|██████    | 240533/400000 [00:28<00:19, 8307.35it/s] 60%|██████    | 241365/400000 [00:28<00:19, 8292.82it/s] 61%|██████    | 242195/400000 [00:28<00:19, 8157.20it/s] 61%|██████    | 243028/400000 [00:28<00:19, 8207.85it/s] 61%|██████    | 243876/400000 [00:28<00:18, 8285.88it/s] 61%|██████    | 244746/400000 [00:28<00:18, 8403.26it/s] 61%|██████▏   | 245617/400000 [00:28<00:18, 8492.66it/s] 62%|██████▏   | 246487/400000 [00:28<00:17, 8552.11it/s] 62%|██████▏   | 247343/400000 [00:29<00:17, 8511.80it/s] 62%|██████▏   | 248195/400000 [00:29<00:18, 8386.96it/s] 62%|██████▏   | 249057/400000 [00:29<00:17, 8453.43it/s] 62%|██████▏   | 249904/400000 [00:29<00:17, 8438.14it/s] 63%|██████▎   | 250774/400000 [00:29<00:17, 8514.44it/s] 63%|██████▎   | 251645/400000 [00:29<00:17, 8569.14it/s] 63%|██████▎   | 252523/400000 [00:29<00:17, 8630.81it/s] 63%|██████▎   | 253390/400000 [00:29<00:16, 8642.15it/s] 64%|██████▎   | 254270/400000 [00:29<00:16, 8687.13it/s] 64%|██████▍   | 255145/400000 [00:29<00:16, 8704.81it/s] 64%|██████▍   | 256016/400000 [00:30<00:16, 8671.22it/s] 64%|██████▍   | 256884/400000 [00:30<00:16, 8426.54it/s] 64%|██████▍   | 257745/400000 [00:30<00:16, 8478.68it/s] 65%|██████▍   | 258598/400000 [00:30<00:16, 8492.12it/s] 65%|██████▍   | 259449/400000 [00:30<00:16, 8398.26it/s] 65%|██████▌   | 260290/400000 [00:30<00:16, 8319.71it/s] 65%|██████▌   | 261137/400000 [00:30<00:16, 8362.43it/s] 65%|██████▌   | 261974/400000 [00:30<00:16, 8328.46it/s] 66%|██████▌   | 262835/400000 [00:30<00:16, 8410.91it/s] 66%|██████▌   | 263686/400000 [00:30<00:16, 8437.92it/s] 66%|██████▌   | 264531/400000 [00:31<00:16, 8438.42it/s] 66%|██████▋   | 265376/400000 [00:31<00:17, 7872.76it/s] 67%|██████▋   | 266172/400000 [00:31<00:17, 7861.18it/s] 67%|██████▋   | 266988/400000 [00:31<00:16, 7946.04it/s] 67%|██████▋   | 267861/400000 [00:31<00:16, 8165.94it/s] 67%|██████▋   | 268742/400000 [00:31<00:15, 8347.83it/s] 67%|██████▋   | 269615/400000 [00:31<00:15, 8457.44it/s] 68%|██████▊   | 270464/400000 [00:31<00:15, 8195.73it/s] 68%|██████▊   | 271292/400000 [00:31<00:15, 8218.62it/s] 68%|██████▊   | 272143/400000 [00:32<00:15, 8302.09it/s] 68%|██████▊   | 272986/400000 [00:32<00:15, 8338.09it/s] 68%|██████▊   | 273823/400000 [00:32<00:15, 8340.17it/s] 69%|██████▊   | 274659/400000 [00:32<00:15, 8327.16it/s] 69%|██████▉   | 275516/400000 [00:32<00:14, 8397.71it/s] 69%|██████▉   | 276357/400000 [00:32<00:14, 8336.32it/s] 69%|██████▉   | 277230/400000 [00:32<00:14, 8450.12it/s] 70%|██████▉   | 278098/400000 [00:32<00:14, 8516.27it/s] 70%|██████▉   | 278982/400000 [00:32<00:14, 8608.15it/s] 70%|██████▉   | 279858/400000 [00:32<00:13, 8652.23it/s] 70%|███████   | 280724/400000 [00:33<00:13, 8630.50it/s] 70%|███████   | 281588/400000 [00:33<00:13, 8510.99it/s] 71%|███████   | 282467/400000 [00:33<00:13, 8592.64it/s] 71%|███████   | 283349/400000 [00:33<00:13, 8657.40it/s] 71%|███████   | 284224/400000 [00:33<00:13, 8684.32it/s] 71%|███████▏  | 285093/400000 [00:33<00:13, 8627.58it/s] 71%|███████▏  | 285957/400000 [00:33<00:13, 8623.87it/s] 72%|███████▏  | 286820/400000 [00:33<00:13, 8618.74it/s] 72%|███████▏  | 287685/400000 [00:33<00:13, 8625.32it/s] 72%|███████▏  | 288561/400000 [00:33<00:12, 8664.59it/s] 72%|███████▏  | 289437/400000 [00:34<00:12, 8690.26it/s] 73%|███████▎  | 290311/400000 [00:34<00:12, 8704.09it/s] 73%|███████▎  | 291194/400000 [00:34<00:12, 8740.54it/s] 73%|███████▎  | 292074/400000 [00:34<00:12, 8755.64it/s] 73%|███████▎  | 292950/400000 [00:34<00:12, 8748.70it/s] 73%|███████▎  | 293827/400000 [00:34<00:12, 8753.41it/s] 74%|███████▎  | 294711/400000 [00:34<00:11, 8777.60it/s] 74%|███████▍  | 295596/400000 [00:34<00:11, 8797.17it/s] 74%|███████▍  | 296476/400000 [00:34<00:11, 8794.66it/s] 74%|███████▍  | 297356/400000 [00:34<00:11, 8784.27it/s] 75%|███████▍  | 298235/400000 [00:35<00:11, 8771.66it/s] 75%|███████▍  | 299113/400000 [00:35<00:11, 8761.36it/s] 75%|███████▍  | 299996/400000 [00:35<00:11, 8781.40it/s] 75%|███████▌  | 300875/400000 [00:35<00:11, 8756.74it/s] 75%|███████▌  | 301751/400000 [00:35<00:11, 8754.49it/s] 76%|███████▌  | 302627/400000 [00:35<00:11, 8567.18it/s] 76%|███████▌  | 303510/400000 [00:35<00:11, 8641.71it/s] 76%|███████▌  | 304393/400000 [00:35<00:10, 8697.08it/s] 76%|███████▋  | 305274/400000 [00:35<00:10, 8719.54it/s] 77%|███████▋  | 306147/400000 [00:35<00:10, 8597.18it/s] 77%|███████▋  | 307009/400000 [00:36<00:10, 8602.49it/s] 77%|███████▋  | 307886/400000 [00:36<00:10, 8651.49it/s] 77%|███████▋  | 308760/400000 [00:36<00:10, 8676.25it/s] 77%|███████▋  | 309636/400000 [00:36<00:10, 8700.66it/s] 78%|███████▊  | 310516/400000 [00:36<00:10, 8729.91it/s] 78%|███████▊  | 311394/400000 [00:36<00:10, 8743.79it/s] 78%|███████▊  | 312273/400000 [00:36<00:10, 8757.04it/s] 78%|███████▊  | 313156/400000 [00:36<00:09, 8778.49it/s] 79%|███████▊  | 314034/400000 [00:36<00:09, 8767.23it/s] 79%|███████▊  | 314914/400000 [00:36<00:09, 8776.13it/s] 79%|███████▉  | 315792/400000 [00:37<00:09, 8698.43it/s] 79%|███████▉  | 316663/400000 [00:37<00:09, 8648.33it/s] 79%|███████▉  | 317529/400000 [00:37<00:09, 8643.35it/s] 80%|███████▉  | 318398/400000 [00:37<00:09, 8654.66it/s] 80%|███████▉  | 319280/400000 [00:37<00:09, 8701.90it/s] 80%|████████  | 320151/400000 [00:37<00:09, 8702.93it/s] 80%|████████  | 321023/400000 [00:37<00:09, 8706.95it/s] 80%|████████  | 321894/400000 [00:37<00:09, 8669.27it/s] 81%|████████  | 322778/400000 [00:37<00:08, 8717.30it/s] 81%|████████  | 323651/400000 [00:37<00:08, 8720.97it/s] 81%|████████  | 324526/400000 [00:38<00:08, 8728.67it/s] 81%|████████▏ | 325399/400000 [00:38<00:08, 8648.59it/s] 82%|████████▏ | 326265/400000 [00:38<00:08, 8568.44it/s] 82%|████████▏ | 327123/400000 [00:38<00:08, 8458.15it/s] 82%|████████▏ | 327986/400000 [00:38<00:08, 8506.65it/s] 82%|████████▏ | 328838/400000 [00:38<00:08, 8484.95it/s] 82%|████████▏ | 329722/400000 [00:38<00:08, 8586.53it/s] 83%|████████▎ | 330605/400000 [00:38<00:08, 8657.23it/s] 83%|████████▎ | 331472/400000 [00:38<00:07, 8626.72it/s] 83%|████████▎ | 332336/400000 [00:38<00:07, 8615.58it/s] 83%|████████▎ | 333198/400000 [00:39<00:07, 8613.69it/s] 84%|████████▎ | 334060/400000 [00:39<00:07, 8610.91it/s] 84%|████████▎ | 334939/400000 [00:39<00:07, 8661.70it/s] 84%|████████▍ | 335809/400000 [00:39<00:07, 8672.58it/s] 84%|████████▍ | 336685/400000 [00:39<00:07, 8698.59it/s] 84%|████████▍ | 337559/400000 [00:39<00:07, 8709.22it/s] 85%|████████▍ | 338435/400000 [00:39<00:07, 8724.26it/s] 85%|████████▍ | 339312/400000 [00:39<00:06, 8736.40it/s] 85%|████████▌ | 340186/400000 [00:39<00:06, 8724.07it/s] 85%|████████▌ | 341068/400000 [00:39<00:06, 8750.73it/s] 85%|████████▌ | 341944/400000 [00:40<00:06, 8716.86it/s] 86%|████████▌ | 342816/400000 [00:40<00:06, 8670.98it/s] 86%|████████▌ | 343684/400000 [00:40<00:06, 8658.52it/s] 86%|████████▌ | 344550/400000 [00:40<00:06, 8469.37it/s] 86%|████████▋ | 345398/400000 [00:40<00:06, 8419.76it/s] 87%|████████▋ | 346241/400000 [00:40<00:06, 8400.45it/s] 87%|████████▋ | 347089/400000 [00:40<00:06, 8422.33it/s] 87%|████████▋ | 347932/400000 [00:40<00:06, 8260.59it/s] 87%|████████▋ | 348805/400000 [00:40<00:06, 8394.38it/s] 87%|████████▋ | 349667/400000 [00:41<00:05, 8459.88it/s] 88%|████████▊ | 350549/400000 [00:41<00:05, 8562.58it/s] 88%|████████▊ | 351422/400000 [00:41<00:05, 8611.97it/s] 88%|████████▊ | 352299/400000 [00:41<00:05, 8658.49it/s] 88%|████████▊ | 353181/400000 [00:41<00:05, 8705.73it/s] 89%|████████▊ | 354053/400000 [00:41<00:05, 8536.32it/s] 89%|████████▊ | 354908/400000 [00:41<00:05, 8406.19it/s] 89%|████████▉ | 355750/400000 [00:41<00:05, 8344.75it/s] 89%|████████▉ | 356623/400000 [00:41<00:05, 8455.75it/s] 89%|████████▉ | 357496/400000 [00:41<00:04, 8535.89it/s] 90%|████████▉ | 358372/400000 [00:42<00:04, 8600.71it/s] 90%|████████▉ | 359233/400000 [00:42<00:04, 8530.35it/s] 90%|█████████ | 360099/400000 [00:42<00:04, 8566.89it/s] 90%|█████████ | 360964/400000 [00:42<00:04, 8590.44it/s] 90%|█████████ | 361829/400000 [00:42<00:04, 8605.96it/s] 91%|█████████ | 362695/400000 [00:42<00:04, 8620.66it/s] 91%|█████████ | 363558/400000 [00:42<00:04, 8595.35it/s] 91%|█████████ | 364418/400000 [00:42<00:04, 8583.48it/s] 91%|█████████▏| 365287/400000 [00:42<00:04, 8614.75it/s] 92%|█████████▏| 366149/400000 [00:42<00:03, 8608.59it/s] 92%|█████████▏| 367033/400000 [00:43<00:03, 8675.32it/s] 92%|█████████▏| 367916/400000 [00:43<00:03, 8719.05it/s] 92%|█████████▏| 368789/400000 [00:43<00:03, 8715.61it/s] 92%|█████████▏| 369668/400000 [00:43<00:03, 8736.39it/s] 93%|█████████▎| 370550/400000 [00:43<00:03, 8760.44it/s] 93%|█████████▎| 371434/400000 [00:43<00:03, 8781.83it/s] 93%|█████████▎| 372313/400000 [00:43<00:03, 8766.83it/s] 93%|█████████▎| 373190/400000 [00:43<00:03, 8596.48it/s] 94%|█████████▎| 374051/400000 [00:43<00:03, 8571.09it/s] 94%|█████████▎| 374909/400000 [00:43<00:02, 8566.92it/s] 94%|█████████▍| 375771/400000 [00:44<00:02, 8581.18it/s] 94%|█████████▍| 376630/400000 [00:44<00:02, 8574.24it/s] 94%|█████████▍| 377493/400000 [00:44<00:02, 8588.14it/s] 95%|█████████▍| 378357/400000 [00:44<00:02, 8602.00it/s] 95%|█████████▍| 379220/400000 [00:44<00:02, 8609.46it/s] 95%|█████████▌| 380082/400000 [00:44<00:02, 8549.04it/s] 95%|█████████▌| 380940/400000 [00:44<00:02, 8555.45it/s] 95%|█████████▌| 381796/400000 [00:44<00:02, 8498.83it/s] 96%|█████████▌| 382659/400000 [00:44<00:02, 8536.38it/s] 96%|█████████▌| 383513/400000 [00:44<00:01, 8531.89it/s] 96%|█████████▌| 384369/400000 [00:45<00:01, 8539.33it/s] 96%|█████████▋| 385224/400000 [00:45<00:01, 8483.12it/s] 97%|█████████▋| 386073/400000 [00:45<00:01, 8432.88it/s] 97%|█████████▋| 386938/400000 [00:45<00:01, 8494.41it/s] 97%|█████████▋| 387788/400000 [00:45<00:01, 8425.53it/s] 97%|█████████▋| 388656/400000 [00:45<00:01, 8497.69it/s] 97%|█████████▋| 389511/400000 [00:45<00:01, 8511.48it/s] 98%|█████████▊| 390363/400000 [00:45<00:01, 8443.93it/s] 98%|█████████▊| 391224/400000 [00:45<00:01, 8491.34it/s] 98%|█████████▊| 392081/400000 [00:45<00:00, 8512.69it/s] 98%|█████████▊| 392951/400000 [00:46<00:00, 8566.97it/s] 98%|█████████▊| 393808/400000 [00:46<00:00, 8482.59it/s] 99%|█████████▊| 394657/400000 [00:46<00:00, 8449.42it/s] 99%|█████████▉| 395503/400000 [00:46<00:00, 8417.67it/s] 99%|█████████▉| 396352/400000 [00:46<00:00, 8438.42it/s] 99%|█████████▉| 397235/400000 [00:46<00:00, 8551.93it/s]100%|█████████▉| 398112/400000 [00:46<00:00, 8613.46it/s]100%|█████████▉| 398987/400000 [00:46<00:00, 8653.95it/s]100%|█████████▉| 399866/400000 [00:46<00:00, 8692.49it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8533.31it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7308760cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010949232718122533 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011152485142583433 	 Accuracy: 56

  model saves at 56% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16101 out of table with 15812 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16101 out of table with 15812 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

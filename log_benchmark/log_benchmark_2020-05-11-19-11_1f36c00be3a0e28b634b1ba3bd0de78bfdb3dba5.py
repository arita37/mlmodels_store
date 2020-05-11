
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efeee217fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 19:12:03.566419
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 19:12:03.571185
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 19:12:03.574409
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 19:12:03.577662
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efefa22f470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355025.2812
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 241417.8594
Epoch 3/10

1/1 [==============================] - 0s 109ms/step - loss: 148699.7656
Epoch 4/10

1/1 [==============================] - 0s 114ms/step - loss: 82579.9375
Epoch 5/10

1/1 [==============================] - 0s 117ms/step - loss: 46465.5977
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 28716.4902
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 19231.2363
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 13691.4502
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 10331.5869
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 8110.7974

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.1901476e-03  6.8201966e+00  5.8380227e+00  7.5328441e+00
   5.1032000e+00  6.3953409e+00  6.6858344e+00  6.1302123e+00
   8.3735313e+00  6.7545242e+00  8.0019264e+00  7.1473684e+00
   8.9230738e+00  7.1640353e+00  7.1202011e+00  6.4531093e+00
   6.9312716e+00  7.3826919e+00  8.9014530e+00  7.2892456e+00
   8.3404512e+00  6.6863046e+00  7.6780367e+00  7.7183275e+00
   7.9241524e+00  5.9490862e+00  7.8867817e+00  5.3384061e+00
   7.2026486e+00  6.6738982e+00  6.4973798e+00  6.3968186e+00
   7.9341259e+00  5.0183911e+00  5.9155030e+00  7.3699274e+00
   5.8413930e+00  6.6892958e+00  7.3233213e+00  5.7089086e+00
   7.1860476e+00  6.7225957e+00  7.6773186e+00  6.7601457e+00
   6.2012992e+00  7.3314114e+00  7.0961056e+00  7.6980219e+00
   7.5571966e+00  6.5318136e+00  5.5214214e+00  7.6736536e+00
   6.0696340e+00  7.6388431e+00  6.9286113e+00  6.8145657e+00
   6.0413141e+00  7.6100368e+00  7.7111535e+00  8.0304365e+00
  -6.7586350e-01 -4.7690880e-01 -1.0659314e+00  1.2567264e-01
  -6.2755758e-01 -5.8821565e-01 -5.7468712e-01 -1.2436420e-01
   1.3274369e+00  9.2970943e-01  6.8867022e-01 -9.6094596e-01
   5.9409998e-02  2.2222974e+00 -3.8342506e-01  7.7728653e-01
   9.1394699e-01  5.4113698e-01 -1.2049079e-02 -6.9383073e-01
  -3.4830478e-01  2.0121226e+00 -1.4003477e+00  1.7405570e-02
  -2.1515908e+00 -1.3877809e+00 -8.2483637e-01  1.1574932e+00
   1.6306162e-02  9.5000499e-01 -1.4571210e+00  1.7136701e+00
   1.2682917e+00 -3.7542278e-01 -2.3423377e-01  3.6856395e-01
   1.5797789e-01  2.4040237e-01  1.0516660e+00  1.8256097e+00
   1.3631648e+00 -5.7954663e-01  6.3708431e-01  3.3038700e-01
  -9.4395775e-01 -1.9417727e-01  1.5066850e-01 -1.1570547e+00
   3.3595744e-01  3.0736524e-01  2.2175713e+00 -1.2070000e+00
   9.7671211e-02 -8.6221516e-01  9.1804069e-01 -6.2101728e-01
   1.1107352e+00  1.4376812e+00  4.1687208e-01  1.9978422e-01
  -1.2431188e+00 -3.0616641e-02  1.8896103e-01  1.4670047e+00
  -1.1709577e-01  1.9230454e+00 -7.9803371e-01 -1.2568257e+00
   3.3887434e-01 -1.3311245e+00  8.7345749e-02 -1.0461805e+00
   1.0402499e+00  2.9446000e-01  8.0149800e-01 -1.6049945e-01
   1.4152918e+00 -1.4310122e+00  9.5161408e-02 -4.2520511e-01
  -1.5944049e+00  3.4000772e-01 -6.9891340e-01  1.4519485e+00
   9.6004128e-02 -1.4853204e+00  5.6057250e-01 -1.9058669e-01
   9.3109608e-02  2.3605881e+00 -1.1521951e+00 -2.7662328e-01
  -2.3420691e-02  9.7060007e-01  1.0839505e+00  2.7244765e-01
  -3.1758159e-01 -1.9707191e-01  3.4268180e-01 -1.3744020e-01
  -2.7029112e-01  7.8971946e-01 -2.3049751e-01  6.3329673e-01
  -1.9441406e+00  1.5589336e-01  1.0330163e+00  7.4516529e-01
   6.2014735e-01 -1.1649331e+00 -7.8712583e-01 -4.2381144e-01
  -3.9804554e-01 -1.3537092e+00  1.4820507e+00 -9.6208608e-01
  -8.5857117e-01 -5.6938177e-01  1.1814955e+00  1.3004380e+00
   6.8831682e-02  7.8880658e+00  8.5418482e+00  6.8596773e+00
   7.4420209e+00  8.4218760e+00  8.0988274e+00  8.8260431e+00
   8.2273436e+00  7.3884630e+00  8.3522100e+00  8.6567402e+00
   6.7315402e+00  7.2212958e+00  8.0186310e+00  6.2431698e+00
   8.5038490e+00  8.4582853e+00  8.0564499e+00  7.8971357e+00
   7.2814388e+00  6.1683769e+00  8.6458120e+00  7.9990759e+00
   6.6165752e+00  7.4555001e+00  8.4151678e+00  6.6494107e+00
   6.4588742e+00  7.1507959e+00  7.2858396e+00  7.9877801e+00
   6.7420321e+00  6.8122697e+00  7.7443628e+00  7.0847950e+00
   7.9837255e+00  7.1598220e+00  7.6501827e+00  8.8045492e+00
   8.2724886e+00  7.5156918e+00  8.2650881e+00  6.4534106e+00
   7.7288465e+00  7.2073236e+00  6.0835209e+00  7.3138714e+00
   5.7353721e+00  7.9572453e+00  8.7831373e+00  6.9047494e+00
   6.9624338e+00  8.3718033e+00  7.0694766e+00  7.6196756e+00
   7.7370348e+00  7.4098911e+00  4.9756508e+00  6.4944453e+00
   1.9247525e+00  5.2817875e-01  1.0203068e+00  1.3074212e+00
   1.1539389e+00  2.4572768e+00  1.6165482e+00  1.2188640e+00
   5.2174342e-01  1.2574658e+00  9.6622419e-01  7.9197836e-01
   3.0745602e-01  1.7120385e+00  3.7204993e-01  1.0800457e-01
   9.5614076e-02  2.4127941e+00  1.6877768e+00  1.7499571e+00
   2.9879169e+00  3.1997508e-01  2.2653303e+00  8.3901316e-01
   9.5342332e-01  2.6298180e+00  9.0596747e-01  1.9124680e+00
   1.0568087e+00  2.2564018e+00  7.9208118e-01  1.6532125e+00
   3.8585550e-01  1.4216410e+00  2.7286482e-01  6.2073374e-01
   2.1237345e+00  1.6351658e+00  1.2378150e+00  4.2549455e-01
   5.4852998e-01  1.1747179e+00  1.5124056e+00  2.3330340e+00
   6.2208450e-01  4.2883682e-01  6.5450227e-01  1.2640792e+00
   1.1296512e+00  6.8246394e-01  1.7602384e+00  2.0676756e+00
   5.5179685e-01  4.8728311e-01  3.3102590e-01  4.5845151e-01
   1.5858592e+00  1.3860099e+00  7.4118161e-01  1.4442812e+00
   1.1871681e+00  1.4064775e+00  1.7831070e+00  1.1566461e+00
   1.3970032e+00  1.4826813e+00  3.7166971e-01  2.5956135e+00
   1.3778434e+00  1.8748581e+00  1.0934352e+00  5.0635314e-01
   5.0862449e-01  6.4417690e-01  1.2719978e+00  3.5039508e-01
   9.3122286e-01  2.9396832e-01  1.2103826e+00  1.8062158e+00
   9.3634015e-01  2.1711290e-01  2.4559743e+00  9.6334463e-01
   1.3190098e+00  1.8185871e+00  4.1420478e-01  1.2134609e+00
   6.2882698e-01  8.4266269e-01  1.6019496e+00  1.4638318e+00
   1.2550080e+00  1.5472691e+00  8.1744385e-01  5.2588201e-01
   3.1366730e-01  3.2784557e+00  1.3712409e+00  2.5384436e+00
   2.3604590e-01  9.3663925e-01  1.9307687e+00  2.9662466e-01
   2.5908656e+00  5.1233464e-01  2.0301670e-01  1.0152930e-01
   6.8001735e-01  1.2424886e+00  2.4318635e-01  2.8254395e+00
   2.0565357e+00  9.8287064e-01  1.3164830e+00  1.8684061e+00
   1.6327431e+00  1.6282601e+00  5.2639014e-01  4.7008002e-01
   8.5712185e+00 -6.0372987e+00 -4.2760386e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 19:12:14.495465
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6849
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 19:12:14.499468
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8985.67
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 19:12:14.502925
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0812
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 19:12:14.506647
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -803.73
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139633025215848
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139632083721744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139632083284040
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139632083284544
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139632083285048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139632083285552

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efee7c623c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.672147
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.628840
grad_step = 000002, loss = 0.594681
grad_step = 000003, loss = 0.556119
grad_step = 000004, loss = 0.511513
grad_step = 000005, loss = 0.463517
grad_step = 000006, loss = 0.419932
grad_step = 000007, loss = 0.390072
grad_step = 000008, loss = 0.380110
grad_step = 000009, loss = 0.356949
grad_step = 000010, loss = 0.324928
grad_step = 000011, loss = 0.299406
grad_step = 000012, loss = 0.284324
grad_step = 000013, loss = 0.273963
grad_step = 000014, loss = 0.262750
grad_step = 000015, loss = 0.249719
grad_step = 000016, loss = 0.235736
grad_step = 000017, loss = 0.221758
grad_step = 000018, loss = 0.207935
grad_step = 000019, loss = 0.196085
grad_step = 000020, loss = 0.185316
grad_step = 000021, loss = 0.173476
grad_step = 000022, loss = 0.161919
grad_step = 000023, loss = 0.151883
grad_step = 000024, loss = 0.143120
grad_step = 000025, loss = 0.134449
grad_step = 000026, loss = 0.125234
grad_step = 000027, loss = 0.115823
grad_step = 000028, loss = 0.106896
grad_step = 000029, loss = 0.098759
grad_step = 000030, loss = 0.091441
grad_step = 000031, loss = 0.084653
grad_step = 000032, loss = 0.077927
grad_step = 000033, loss = 0.071369
grad_step = 000034, loss = 0.065335
grad_step = 000035, loss = 0.059841
grad_step = 000036, loss = 0.054626
grad_step = 000037, loss = 0.049532
grad_step = 000038, loss = 0.044662
grad_step = 000039, loss = 0.040275
grad_step = 000040, loss = 0.036481
grad_step = 000041, loss = 0.033005
grad_step = 000042, loss = 0.029566
grad_step = 000043, loss = 0.026379
grad_step = 000044, loss = 0.023548
grad_step = 000045, loss = 0.021004
grad_step = 000046, loss = 0.018703
grad_step = 000047, loss = 0.016574
grad_step = 000048, loss = 0.014661
grad_step = 000049, loss = 0.013046
grad_step = 000050, loss = 0.011606
grad_step = 000051, loss = 0.010272
grad_step = 000052, loss = 0.009130
grad_step = 000053, loss = 0.008158
grad_step = 000054, loss = 0.007323
grad_step = 000055, loss = 0.006593
grad_step = 000056, loss = 0.005935
grad_step = 000057, loss = 0.005400
grad_step = 000058, loss = 0.004957
grad_step = 000059, loss = 0.004556
grad_step = 000060, loss = 0.004232
grad_step = 000061, loss = 0.003980
grad_step = 000062, loss = 0.003762
grad_step = 000063, loss = 0.003556
grad_step = 000064, loss = 0.003377
grad_step = 000065, loss = 0.003249
grad_step = 000066, loss = 0.003137
grad_step = 000067, loss = 0.003015
grad_step = 000068, loss = 0.002907
grad_step = 000069, loss = 0.002834
grad_step = 000070, loss = 0.002777
grad_step = 000071, loss = 0.002706
grad_step = 000072, loss = 0.002636
grad_step = 000073, loss = 0.002582
grad_step = 000074, loss = 0.002533
grad_step = 000075, loss = 0.002480
grad_step = 000076, loss = 0.002433
grad_step = 000077, loss = 0.002402
grad_step = 000078, loss = 0.002373
grad_step = 000079, loss = 0.002338
grad_step = 000080, loss = 0.002309
grad_step = 000081, loss = 0.002285
grad_step = 000082, loss = 0.002263
grad_step = 000083, loss = 0.002241
grad_step = 000084, loss = 0.002225
grad_step = 000085, loss = 0.002214
grad_step = 000086, loss = 0.002201
grad_step = 000087, loss = 0.002189
grad_step = 000088, loss = 0.002179
grad_step = 000089, loss = 0.002169
grad_step = 000090, loss = 0.002158
grad_step = 000091, loss = 0.002150
grad_step = 000092, loss = 0.002143
grad_step = 000093, loss = 0.002136
grad_step = 000094, loss = 0.002128
grad_step = 000095, loss = 0.002121
grad_step = 000096, loss = 0.002114
grad_step = 000097, loss = 0.002106
grad_step = 000098, loss = 0.002098
grad_step = 000099, loss = 0.002093
grad_step = 000100, loss = 0.002086
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002080
grad_step = 000102, loss = 0.002073
grad_step = 000103, loss = 0.002068
grad_step = 000104, loss = 0.002062
grad_step = 000105, loss = 0.002056
grad_step = 000106, loss = 0.002051
grad_step = 000107, loss = 0.002047
grad_step = 000108, loss = 0.002042
grad_step = 000109, loss = 0.002037
grad_step = 000110, loss = 0.002033
grad_step = 000111, loss = 0.002029
grad_step = 000112, loss = 0.002025
grad_step = 000113, loss = 0.002021
grad_step = 000114, loss = 0.002018
grad_step = 000115, loss = 0.002014
grad_step = 000116, loss = 0.002011
grad_step = 000117, loss = 0.002008
grad_step = 000118, loss = 0.002005
grad_step = 000119, loss = 0.002002
grad_step = 000120, loss = 0.001999
grad_step = 000121, loss = 0.001996
grad_step = 000122, loss = 0.001993
grad_step = 000123, loss = 0.001991
grad_step = 000124, loss = 0.001988
grad_step = 000125, loss = 0.001985
grad_step = 000126, loss = 0.001983
grad_step = 000127, loss = 0.001980
grad_step = 000128, loss = 0.001978
grad_step = 000129, loss = 0.001975
grad_step = 000130, loss = 0.001973
grad_step = 000131, loss = 0.001970
grad_step = 000132, loss = 0.001968
grad_step = 000133, loss = 0.001965
grad_step = 000134, loss = 0.001963
grad_step = 000135, loss = 0.001960
grad_step = 000136, loss = 0.001958
grad_step = 000137, loss = 0.001955
grad_step = 000138, loss = 0.001953
grad_step = 000139, loss = 0.001950
grad_step = 000140, loss = 0.001948
grad_step = 000141, loss = 0.001945
grad_step = 000142, loss = 0.001943
grad_step = 000143, loss = 0.001940
grad_step = 000144, loss = 0.001938
grad_step = 000145, loss = 0.001935
grad_step = 000146, loss = 0.001932
grad_step = 000147, loss = 0.001930
grad_step = 000148, loss = 0.001927
grad_step = 000149, loss = 0.001924
grad_step = 000150, loss = 0.001922
grad_step = 000151, loss = 0.001919
grad_step = 000152, loss = 0.001917
grad_step = 000153, loss = 0.001914
grad_step = 000154, loss = 0.001911
grad_step = 000155, loss = 0.001908
grad_step = 000156, loss = 0.001906
grad_step = 000157, loss = 0.001903
grad_step = 000158, loss = 0.001900
grad_step = 000159, loss = 0.001897
grad_step = 000160, loss = 0.001895
grad_step = 000161, loss = 0.001892
grad_step = 000162, loss = 0.001889
grad_step = 000163, loss = 0.001886
grad_step = 000164, loss = 0.001883
grad_step = 000165, loss = 0.001881
grad_step = 000166, loss = 0.001878
grad_step = 000167, loss = 0.001875
grad_step = 000168, loss = 0.001872
grad_step = 000169, loss = 0.001869
grad_step = 000170, loss = 0.001866
grad_step = 000171, loss = 0.001863
grad_step = 000172, loss = 0.001860
grad_step = 000173, loss = 0.001857
grad_step = 000174, loss = 0.001854
grad_step = 000175, loss = 0.001851
grad_step = 000176, loss = 0.001848
grad_step = 000177, loss = 0.001845
grad_step = 000178, loss = 0.001842
grad_step = 000179, loss = 0.001839
grad_step = 000180, loss = 0.001836
grad_step = 000181, loss = 0.001833
grad_step = 000182, loss = 0.001830
grad_step = 000183, loss = 0.001827
grad_step = 000184, loss = 0.001824
grad_step = 000185, loss = 0.001821
grad_step = 000186, loss = 0.001818
grad_step = 000187, loss = 0.001815
grad_step = 000188, loss = 0.001811
grad_step = 000189, loss = 0.001808
grad_step = 000190, loss = 0.001805
grad_step = 000191, loss = 0.001802
grad_step = 000192, loss = 0.001799
grad_step = 000193, loss = 0.001796
grad_step = 000194, loss = 0.001792
grad_step = 000195, loss = 0.001789
grad_step = 000196, loss = 0.001786
grad_step = 000197, loss = 0.001783
grad_step = 000198, loss = 0.001780
grad_step = 000199, loss = 0.001776
grad_step = 000200, loss = 0.001773
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001770
grad_step = 000202, loss = 0.001767
grad_step = 000203, loss = 0.001764
grad_step = 000204, loss = 0.001760
grad_step = 000205, loss = 0.001757
grad_step = 000206, loss = 0.001754
grad_step = 000207, loss = 0.001751
grad_step = 000208, loss = 0.001748
grad_step = 000209, loss = 0.001744
grad_step = 000210, loss = 0.001741
grad_step = 000211, loss = 0.001738
grad_step = 000212, loss = 0.001735
grad_step = 000213, loss = 0.001732
grad_step = 000214, loss = 0.001729
grad_step = 000215, loss = 0.001726
grad_step = 000216, loss = 0.001722
grad_step = 000217, loss = 0.001719
grad_step = 000218, loss = 0.001716
grad_step = 000219, loss = 0.001713
grad_step = 000220, loss = 0.001710
grad_step = 000221, loss = 0.001707
grad_step = 000222, loss = 0.001704
grad_step = 000223, loss = 0.001701
grad_step = 000224, loss = 0.001699
grad_step = 000225, loss = 0.001696
grad_step = 000226, loss = 0.001693
grad_step = 000227, loss = 0.001690
grad_step = 000228, loss = 0.001687
grad_step = 000229, loss = 0.001684
grad_step = 000230, loss = 0.001682
grad_step = 000231, loss = 0.001679
grad_step = 000232, loss = 0.001676
grad_step = 000233, loss = 0.001674
grad_step = 000234, loss = 0.001671
grad_step = 000235, loss = 0.001669
grad_step = 000236, loss = 0.001666
grad_step = 000237, loss = 0.001664
grad_step = 000238, loss = 0.001664
grad_step = 000239, loss = 0.001667
grad_step = 000240, loss = 0.001678
grad_step = 000241, loss = 0.001703
grad_step = 000242, loss = 0.001736
grad_step = 000243, loss = 0.001745
grad_step = 000244, loss = 0.001706
grad_step = 000245, loss = 0.001656
grad_step = 000246, loss = 0.001653
grad_step = 000247, loss = 0.001687
grad_step = 000248, loss = 0.001698
grad_step = 000249, loss = 0.001667
grad_step = 000250, loss = 0.001639
grad_step = 000251, loss = 0.001649
grad_step = 000252, loss = 0.001668
grad_step = 000253, loss = 0.001663
grad_step = 000254, loss = 0.001642
grad_step = 000255, loss = 0.001634
grad_step = 000256, loss = 0.001642
grad_step = 000257, loss = 0.001646
grad_step = 000258, loss = 0.001639
grad_step = 000259, loss = 0.001631
grad_step = 000260, loss = 0.001632
grad_step = 000261, loss = 0.001633
grad_step = 000262, loss = 0.001628
grad_step = 000263, loss = 0.001622
grad_step = 000264, loss = 0.001622
grad_step = 000265, loss = 0.001625
grad_step = 000266, loss = 0.001624
grad_step = 000267, loss = 0.001617
grad_step = 000268, loss = 0.001611
grad_step = 000269, loss = 0.001610
grad_step = 000270, loss = 0.001613
grad_step = 000271, loss = 0.001615
grad_step = 000272, loss = 0.001611
grad_step = 000273, loss = 0.001607
grad_step = 000274, loss = 0.001604
grad_step = 000275, loss = 0.001603
grad_step = 000276, loss = 0.001603
grad_step = 000277, loss = 0.001601
grad_step = 000278, loss = 0.001598
grad_step = 000279, loss = 0.001595
grad_step = 000280, loss = 0.001594
grad_step = 000281, loss = 0.001595
grad_step = 000282, loss = 0.001595
grad_step = 000283, loss = 0.001595
grad_step = 000284, loss = 0.001596
grad_step = 000285, loss = 0.001598
grad_step = 000286, loss = 0.001604
grad_step = 000287, loss = 0.001614
grad_step = 000288, loss = 0.001633
grad_step = 000289, loss = 0.001657
grad_step = 000290, loss = 0.001686
grad_step = 000291, loss = 0.001695
grad_step = 000292, loss = 0.001685
grad_step = 000293, loss = 0.001643
grad_step = 000294, loss = 0.001606
grad_step = 000295, loss = 0.001590
grad_step = 000296, loss = 0.001593
grad_step = 000297, loss = 0.001604
grad_step = 000298, loss = 0.001607
grad_step = 000299, loss = 0.001605
grad_step = 000300, loss = 0.001600
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001594
grad_step = 000302, loss = 0.001586
grad_step = 000303, loss = 0.001576
grad_step = 000304, loss = 0.001570
grad_step = 000305, loss = 0.001572
grad_step = 000306, loss = 0.001579
grad_step = 000307, loss = 0.001584
grad_step = 000308, loss = 0.001581
grad_step = 000309, loss = 0.001571
grad_step = 000310, loss = 0.001559
grad_step = 000311, loss = 0.001553
grad_step = 000312, loss = 0.001555
grad_step = 000313, loss = 0.001559
grad_step = 000314, loss = 0.001562
grad_step = 000315, loss = 0.001561
grad_step = 000316, loss = 0.001557
grad_step = 000317, loss = 0.001553
grad_step = 000318, loss = 0.001551
grad_step = 000319, loss = 0.001551
grad_step = 000320, loss = 0.001551
grad_step = 000321, loss = 0.001551
grad_step = 000322, loss = 0.001548
grad_step = 000323, loss = 0.001545
grad_step = 000324, loss = 0.001542
grad_step = 000325, loss = 0.001539
grad_step = 000326, loss = 0.001538
grad_step = 000327, loss = 0.001538
grad_step = 000328, loss = 0.001539
grad_step = 000329, loss = 0.001540
grad_step = 000330, loss = 0.001543
grad_step = 000331, loss = 0.001547
grad_step = 000332, loss = 0.001555
grad_step = 000333, loss = 0.001570
grad_step = 000334, loss = 0.001596
grad_step = 000335, loss = 0.001646
grad_step = 000336, loss = 0.001711
grad_step = 000337, loss = 0.001802
grad_step = 000338, loss = 0.001837
grad_step = 000339, loss = 0.001810
grad_step = 000340, loss = 0.001662
grad_step = 000341, loss = 0.001546
grad_step = 000342, loss = 0.001543
grad_step = 000343, loss = 0.001609
grad_step = 000344, loss = 0.001641
grad_step = 000345, loss = 0.001593
grad_step = 000346, loss = 0.001550
grad_step = 000347, loss = 0.001561
grad_step = 000348, loss = 0.001582
grad_step = 000349, loss = 0.001563
grad_step = 000350, loss = 0.001521
grad_step = 000351, loss = 0.001521
grad_step = 000352, loss = 0.001556
grad_step = 000353, loss = 0.001558
grad_step = 000354, loss = 0.001524
grad_step = 000355, loss = 0.001500
grad_step = 000356, loss = 0.001514
grad_step = 000357, loss = 0.001533
grad_step = 000358, loss = 0.001522
grad_step = 000359, loss = 0.001502
grad_step = 000360, loss = 0.001499
grad_step = 000361, loss = 0.001510
grad_step = 000362, loss = 0.001511
grad_step = 000363, loss = 0.001497
grad_step = 000364, loss = 0.001488
grad_step = 000365, loss = 0.001492
grad_step = 000366, loss = 0.001500
grad_step = 000367, loss = 0.001499
grad_step = 000368, loss = 0.001489
grad_step = 000369, loss = 0.001481
grad_step = 000370, loss = 0.001482
grad_step = 000371, loss = 0.001485
grad_step = 000372, loss = 0.001483
grad_step = 000373, loss = 0.001478
grad_step = 000374, loss = 0.001472
grad_step = 000375, loss = 0.001471
grad_step = 000376, loss = 0.001473
grad_step = 000377, loss = 0.001475
grad_step = 000378, loss = 0.001474
grad_step = 000379, loss = 0.001472
grad_step = 000380, loss = 0.001470
grad_step = 000381, loss = 0.001473
grad_step = 000382, loss = 0.001478
grad_step = 000383, loss = 0.001487
grad_step = 000384, loss = 0.001499
grad_step = 000385, loss = 0.001514
grad_step = 000386, loss = 0.001538
grad_step = 000387, loss = 0.001567
grad_step = 000388, loss = 0.001609
grad_step = 000389, loss = 0.001648
grad_step = 000390, loss = 0.001683
grad_step = 000391, loss = 0.001681
grad_step = 000392, loss = 0.001643
grad_step = 000393, loss = 0.001565
grad_step = 000394, loss = 0.001488
grad_step = 000395, loss = 0.001444
grad_step = 000396, loss = 0.001447
grad_step = 000397, loss = 0.001481
grad_step = 000398, loss = 0.001516
grad_step = 000399, loss = 0.001532
grad_step = 000400, loss = 0.001516
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001487
grad_step = 000402, loss = 0.001457
grad_step = 000403, loss = 0.001442
grad_step = 000404, loss = 0.001443
grad_step = 000405, loss = 0.001453
grad_step = 000406, loss = 0.001460
grad_step = 000407, loss = 0.001459
grad_step = 000408, loss = 0.001451
grad_step = 000409, loss = 0.001441
grad_step = 000410, loss = 0.001436
grad_step = 000411, loss = 0.001438
grad_step = 000412, loss = 0.001446
grad_step = 000413, loss = 0.001453
grad_step = 000414, loss = 0.001459
grad_step = 000415, loss = 0.001457
grad_step = 000416, loss = 0.001451
grad_step = 000417, loss = 0.001441
grad_step = 000418, loss = 0.001436
grad_step = 000419, loss = 0.001437
grad_step = 000420, loss = 0.001446
grad_step = 000421, loss = 0.001457
grad_step = 000422, loss = 0.001470
grad_step = 000423, loss = 0.001480
grad_step = 000424, loss = 0.001493
grad_step = 000425, loss = 0.001505
grad_step = 000426, loss = 0.001528
grad_step = 000427, loss = 0.001550
grad_step = 000428, loss = 0.001578
grad_step = 000429, loss = 0.001581
grad_step = 000430, loss = 0.001560
grad_step = 000431, loss = 0.001496
grad_step = 000432, loss = 0.001429
grad_step = 000433, loss = 0.001388
grad_step = 000434, loss = 0.001393
grad_step = 000435, loss = 0.001426
grad_step = 000436, loss = 0.001454
grad_step = 000437, loss = 0.001460
grad_step = 000438, loss = 0.001442
grad_step = 000439, loss = 0.001419
grad_step = 000440, loss = 0.001403
grad_step = 000441, loss = 0.001400
grad_step = 000442, loss = 0.001402
grad_step = 000443, loss = 0.001402
grad_step = 000444, loss = 0.001399
grad_step = 000445, loss = 0.001384
grad_step = 000446, loss = 0.001373
grad_step = 000447, loss = 0.001366
grad_step = 000448, loss = 0.001367
grad_step = 000449, loss = 0.001372
grad_step = 000450, loss = 0.001376
grad_step = 000451, loss = 0.001377
grad_step = 000452, loss = 0.001374
grad_step = 000453, loss = 0.001367
grad_step = 000454, loss = 0.001361
grad_step = 000455, loss = 0.001356
grad_step = 000456, loss = 0.001353
grad_step = 000457, loss = 0.001353
grad_step = 000458, loss = 0.001355
grad_step = 000459, loss = 0.001357
grad_step = 000460, loss = 0.001359
grad_step = 000461, loss = 0.001361
grad_step = 000462, loss = 0.001365
grad_step = 000463, loss = 0.001375
grad_step = 000464, loss = 0.001401
grad_step = 000465, loss = 0.001452
grad_step = 000466, loss = 0.001549
grad_step = 000467, loss = 0.001697
grad_step = 000468, loss = 0.001919
grad_step = 000469, loss = 0.002042
grad_step = 000470, loss = 0.002034
grad_step = 000471, loss = 0.001737
grad_step = 000472, loss = 0.001469
grad_step = 000473, loss = 0.001454
grad_step = 000474, loss = 0.001576
grad_step = 000475, loss = 0.001619
grad_step = 000476, loss = 0.001494
grad_step = 000477, loss = 0.001425
grad_step = 000478, loss = 0.001486
grad_step = 000479, loss = 0.001507
grad_step = 000480, loss = 0.001431
grad_step = 000481, loss = 0.001363
grad_step = 000482, loss = 0.001410
grad_step = 000483, loss = 0.001469
grad_step = 000484, loss = 0.001411
grad_step = 000485, loss = 0.001329
grad_step = 000486, loss = 0.001344
grad_step = 000487, loss = 0.001405
grad_step = 000488, loss = 0.001407
grad_step = 000489, loss = 0.001336
grad_step = 000490, loss = 0.001313
grad_step = 000491, loss = 0.001353
grad_step = 000492, loss = 0.001370
grad_step = 000493, loss = 0.001338
grad_step = 000494, loss = 0.001309
grad_step = 000495, loss = 0.001322
grad_step = 000496, loss = 0.001340
grad_step = 000497, loss = 0.001325
grad_step = 000498, loss = 0.001303
grad_step = 000499, loss = 0.001305
grad_step = 000500, loss = 0.001319
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001316
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

  date_run                              2020-05-11 19:12:34.985329
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.247424
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 19:12:34.991142
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140985
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 19:12:34.999055
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.154733
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 19:12:35.004601
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.14232
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
0   2020-05-11 19:12:03.566419  ...    mean_absolute_error
1   2020-05-11 19:12:03.571185  ...     mean_squared_error
2   2020-05-11 19:12:03.574409  ...  median_absolute_error
3   2020-05-11 19:12:03.577662  ...               r2_score
4   2020-05-11 19:12:14.495465  ...    mean_absolute_error
5   2020-05-11 19:12:14.499468  ...     mean_squared_error
6   2020-05-11 19:12:14.502925  ...  median_absolute_error
7   2020-05-11 19:12:14.506647  ...               r2_score
8   2020-05-11 19:12:34.985329  ...    mean_absolute_error
9   2020-05-11 19:12:34.991142  ...     mean_squared_error
10  2020-05-11 19:12:34.999055  ...  median_absolute_error
11  2020-05-11 19:12:35.004601  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3047424/9912422 [00:00<00:00, 30127830.97it/s]9920512it [00:00, 33884736.48it/s]                             
0it [00:00, ?it/s]32768it [00:00, 622972.53it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 899900.05it/s]1654784it [00:00, 12324021.63it/s]                         
0it [00:00, ?it/s]8192it [00:00, 128509.59it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2feb1db70> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2b14d9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2feb1dc88> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2feae1eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2b14e9eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb29c26e048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2b14d9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2a598ddd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2feb29ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2feae1eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb2b14e9d68> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fca200e91d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=74ea37f9d8c26d872eb12eb32377db5f1cc3e29c7320fc80dc2a6acb561ff4fd
  Stored in directory: /tmp/pip-ephem-wheel-cache-8jynyvr9/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc9b7cd1208> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2646016/17464789 [===>..........................] - ETA: 0s
 8658944/17464789 [=============>................] - ETA: 0s
13148160/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 19:14:02.813423: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 19:14:02.819126: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 19:14:02.820312: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cbb0a9ecb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 19:14:02.820338: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4980 - accuracy: 0.5110 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6104 - accuracy: 0.5037
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6145 - accuracy: 0.5034
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7372 - accuracy: 0.4954
11000/25000 [============>.................] - ETA: 3s - loss: 7.7600 - accuracy: 0.4939
12000/25000 [=============>................] - ETA: 3s - loss: 7.7407 - accuracy: 0.4952
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
15000/25000 [=================>............] - ETA: 2s - loss: 7.7116 - accuracy: 0.4971
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7081 - accuracy: 0.4973
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6939 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6989 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6863 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6860 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 19:14:17.329237
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 19:14:17.329237  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 19:14:23.743922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 19:14:23.749279: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 19:14:23.750530: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55deca42d150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 19:14:23.750605: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f406aabfdd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3694 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.3657 - val_crf_viterbi_accuracy: 0.2800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4060d608d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5440 - accuracy: 0.5080
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5746 - accuracy: 0.5060
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6398 - accuracy: 0.5017
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6121 - accuracy: 0.5036
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6938 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6963 - accuracy: 0.4981
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6705 - accuracy: 0.4997
25000/25000 [==============================] - 7s 298us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f401bd0b400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<51:30:10, 4.65kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<36:17:07, 6.60kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<25:27:07, 9.41kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:49:00, 13.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<12:26:07, 19.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.63M/862M [00:02<8:38:44, 27.4kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:02<6:00:56, 39.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:02<4:11:49, 55.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:02<2:55:12, 79.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:02<2:01:54, 114kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:03<1:24:50, 162kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:03<59:19, 232kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:03<41:22, 330kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:03<28:56, 470kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:03<20:13, 668kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:04<16:09, 835kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<11:21, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<1:19:30, 169kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<57:54, 232kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:06<41:03, 327kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<30:57, 431kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:08<23:12, 575kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:08<16:35, 803kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<14:26, 920kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:10<12:50, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:10<09:33, 1.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<06:52, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<10:25, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:12<08:42, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:12<06:23, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<07:32, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:14<07:57, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:14<06:14, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<06:28, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<05:52, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:16<04:23, 2.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<06:07, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<05:37, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:18<04:12, 3.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<06:00, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<06:52, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:20<05:27, 2.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<05:54, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<05:28, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<04:06, 3.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<05:25, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:07, 3.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<05:53, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<06:46, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<05:22, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:48, 2.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:23, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:48, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:05, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:39, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:11, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<03:47, 3.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<09:13, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:55, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:01, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<08:15, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<08:18, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:24, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:38, 2.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<09:11, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:43, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:42, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:50, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:49, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:19, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<03:11, 3.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<48:14, 253kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<36:14, 336kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<25:53, 470kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<18:11, 667kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<24:27, 496kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<18:10, 667kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<13:00, 930kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<11:55, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<10:49, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<08:05, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<10:04, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<08:17, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:05, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:02, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:22, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:42, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<04:06, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<20:16, 584kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<15:25, 768kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<11:04, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<10:29, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<09:58, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<07:38, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<18:01, 650kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<15:06, 775kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<11:07, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:55, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<10:45, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<10:00, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<07:37, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:28, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<27:22, 423kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<21:37, 535kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<15:43, 735kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<11:05, 1.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<21:44, 529kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<17:46, 647kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<13:03, 879kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:14, 1.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<23:12, 493kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<18:31, 617kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<13:25, 850kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:31, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<12:59, 875kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<11:19, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<08:23, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:02, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<08:42, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<08:13, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:07<06:17, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<06:14, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<06:52, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:26, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<03:56, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<14:03, 794kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<11:53, 939kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<08:49, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:00, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<08:05, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<06:16, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:30, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<15:51, 695kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<13:07, 840kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<09:36, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:49, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<18:15, 600kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<15:14, 719kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<11:10, 979kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<07:57, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<09:30, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<08:58, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:53, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<29:19, 369kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<22:56, 471kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<16:37, 649kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:44, 916kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<24:39, 436kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<19:08, 561kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<13:51, 774kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<11:28, 930kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<10:19, 1.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<07:40, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:57, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:52, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:59, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:19, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:55, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:50, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:57, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:19, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<06:54, 1.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<07:03, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:29, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<03:58, 2.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<27:31, 378kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<21:01, 494kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<15:08, 686kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<12:20, 838kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<10:57, 942kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<08:14, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<05:52, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<15:50, 648kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<14:31, 706kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<10:56, 938kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<07:49, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<07:53, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<07:48, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<06:02, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<13:08, 770kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<10:58, 922kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:06, 1.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<07:22, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<07:18, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:39, 1.77MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:04, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<17:47, 561kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<14:35, 684kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<10:38, 937kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:38, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<08:05, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<07:48, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:59, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:17, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<23:14, 424kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<18:22, 536kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<13:17, 739kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:25, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<10:52, 899kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<09:43, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:14, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:10, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<09:20, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<08:08, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:06, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<05:55, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<06:18, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<04:51, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:32, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<05:44, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<06:03, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<04:39, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<03:22, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<07:10, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<06:37, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:02, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:08, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:48, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<04:30, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:16, 2.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<06:02, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<05:50, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:28, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<04:43, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<05:23, 1.73MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:13, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:02, 3.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<13:39, 676kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<11:04, 833kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<08:07, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<07:15, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<07:03, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<05:27, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:57, 2.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<05:56, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<06:05, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<04:40, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:24, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<05:41, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<06:00, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<04:41, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:22, 2.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<14:51, 603kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<11:54, 751kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<08:41, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<07:36, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:58, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<05:27, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:55, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<12:58, 680kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<10:38, 829kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<07:53, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:40, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<06:50, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<06:13, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:42, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<05:20, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<04:13, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:02, 2.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<09:00, 955kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<08:09, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<06:05, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:23, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:03, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<05:42, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<04:20, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:30, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<05:02, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:59, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:53, 2.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<15:12, 553kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<11:42, 717kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<08:55, 939kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:22, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:50, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:37, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<05:04, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:38, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<08:20, 991kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:08, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<05:18, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:10, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:08, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:57, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<04:07, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<04:35, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<03:37, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<02:36, 3.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<15:13, 529kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<11:51, 679kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:35, 936kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<07:25, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<06:47, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:06, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:39, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<08:10, 968kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<07:22, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:32, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:58, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<06:12, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<05:58, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:34, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:15, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<19:42, 395kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<15:24, 505kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<11:06, 699kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:53, 981kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<08:04, 956kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<07:14, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:27, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<05:02, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<05:01, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<03:51, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:48, 2.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:47, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:57, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:50, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:46, 2.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<1:49:11, 68.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<1:17:57, 96.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<54:49, 137kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<38:22, 195kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<29:09, 255kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<21:55, 339kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<15:40, 474kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<11:03, 669kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<10:12, 722kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<08:38, 852kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<06:22, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:31, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<08:54, 820kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<07:43, 945kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:46, 1.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:11, 1.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<03:57, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:50, 2.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<06:37, 1.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<06:05, 1.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<04:34, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:18, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<05:15, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<05:07, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:57, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:49, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<22:27, 313kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<16:40, 421kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<11:52, 590kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<09:35, 725kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<08:02, 865kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<05:53, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:11, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<06:30, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<05:51, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<04:21, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:08, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<05:35, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<05:09, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<03:53, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:50, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:55, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<03:00, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:11, 3.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<05:15, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:19<04:54, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<03:44, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<03:47, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<02:54, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:06, 3.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<05:59, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:23<05:22, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<04:03, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:54, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:55, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<03:02, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:21, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<02:38, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:54, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:16, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<02:34, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:11, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:31, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<02:47, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:07, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:28, 2.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:44, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:04, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:23, 2.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<01:44, 3.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<08:23, 723kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<06:58, 870kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:06, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:38, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<05:10, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:43, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:34, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<03:31, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:57, 3.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<39:25, 149kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<28:38, 204kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<20:13, 289kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<14:08, 410kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<13:29, 429kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<10:30, 551kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:34, 761kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<06:13, 919kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<05:24, 1.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:59, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:51, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<04:57, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<04:30, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:24, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:18, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:19, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:32, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<01:51, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<03:49, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<03:43, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:49, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:01, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<07:36, 717kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<06:21, 857kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<04:40, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<04:09, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:55, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:58, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:57, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<03:04, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:21, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:41, 3.10MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<06:32, 802kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<05:33, 942kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:07, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:43, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:33, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:40, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:57, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<03:16, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<02:58, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:14, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<02:32, 1.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<03:40, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:57, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<02:42, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<02:46, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:07, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:32, 3.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<04:11, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:50, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:53, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:49, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<02:49, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:10, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:33, 3.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<05:06, 933kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<04:24, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<03:17, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:04, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<03:55, 1.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<03:10, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:17, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:47, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:06, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:31, 3.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:21, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<03:50, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:52, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:43, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:05, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:12, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:07, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<01:36, 2.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:10, 3.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<14:27, 300kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<10:39, 407kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<07:32, 573kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<05:16, 810kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<1:28:56, 48.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<1:03:44, 67.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<44:51, 95.1kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<31:22, 135kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<22:35, 186kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<16:30, 255kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<11:40, 359kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<08:10, 509kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<07:41, 538kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<06:04, 680kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<04:25, 933kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:05, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<23:47, 171kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<17:21, 234kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<12:16, 330kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<09:10, 436kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<07:05, 563kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:07, 778kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<04:13, 932kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:37, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:39, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:54, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:06, 941kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:14, 912kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<03:14, 1.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:23, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:18, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:44, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:16, 2.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:10, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:39, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:12, 3.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:08, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:07, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:36, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:09, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<03:17, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:54, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:09, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:31, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<06:42, 525kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<05:16, 666kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<03:48, 919kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:40, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<07:33, 456kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<05:53, 584kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<04:15, 806kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<03:31, 960kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<03:02, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<02:07, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:02, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:34, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:38, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:42, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:19, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:27, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:35, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:14, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:29, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:10, 2.63MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<00:54, 3.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:29, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:47, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:26, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:02, 2.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<02:35, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<02:32, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:56, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<03:01, 961kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<02:49, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:08, 1.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:31, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<03:05, 917kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:49, 1.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:06, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:30, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:02, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:02, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:33, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:07, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:47, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:52, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:26, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:03, 2.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:27, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:36, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:15, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:54, 2.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<02:46, 919kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<02:32, 1.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:53, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:20, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:55, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:55, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:03, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:34, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:40, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:18, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:55, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<02:26, 961kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<02:15, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<01:42, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:14, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:33, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:49, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:27, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:03, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:24, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:39, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:18, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:56, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:19, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:33, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:14, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:32, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:13, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:52, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:16, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:27, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:10, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:50, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:13, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:24, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:48, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:01, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:13, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:57, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:42, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:53, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:06, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:53, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:38, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:02, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:13, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:41, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<01:02, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<01:11, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:56, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:40, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:02, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:09, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:59, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:47<01:06, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:52, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:37, 2.35MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:56, 1.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<01:04, 1.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:50, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:36, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:54, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<01:02, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:33, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:57, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:44, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:45, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:51, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<00:40, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:28, 2.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:46, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:37, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:26, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:41, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:47, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:37, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:26, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:40, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:45, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:35, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:25, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:36, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:28, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:20, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:25, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:32, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:26, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:18, 2.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:31, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:35, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:27, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:19, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:29, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:32, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:25, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:17, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:26, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:30, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:23, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:16, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:24, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:28, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:22, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:15, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:21, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:24, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:18, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:12, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:18, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:20, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:15, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:10, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:16, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:13, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:13, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:14, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:11, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:09, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:08, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:05, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:03, 3.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:23, 498kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:19, 601kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:13, 812kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:07, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:07, 1.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:06, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:04, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:01, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.41MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 761/400000 [00:00<00:52, 7604.55it/s]  0%|          | 1562/400000 [00:00<00:51, 7719.64it/s]  1%|          | 2378/400000 [00:00<00:50, 7844.69it/s]  1%|          | 3181/400000 [00:00<00:50, 7897.90it/s]  1%|          | 4011/400000 [00:00<00:49, 8012.02it/s]  1%|          | 4827/400000 [00:00<00:49, 8053.57it/s]  1%|▏         | 5619/400000 [00:00<00:49, 8011.74it/s]  2%|▏         | 6432/400000 [00:00<00:48, 8044.55it/s]  2%|▏         | 7226/400000 [00:00<00:49, 8010.95it/s]  2%|▏         | 8020/400000 [00:01<00:49, 7987.81it/s]  2%|▏         | 8811/400000 [00:01<00:49, 7964.03it/s]  2%|▏         | 9593/400000 [00:01<00:51, 7650.34it/s]  3%|▎         | 10386/400000 [00:01<00:50, 7729.87it/s]  3%|▎         | 11154/400000 [00:01<00:50, 7666.33it/s]  3%|▎         | 11968/400000 [00:01<00:49, 7801.03it/s]  3%|▎         | 12781/400000 [00:01<00:49, 7894.25it/s]  3%|▎         | 13607/400000 [00:01<00:48, 7999.97it/s]  4%|▎         | 14428/400000 [00:01<00:47, 8061.49it/s]  4%|▍         | 15234/400000 [00:01<00:48, 7966.69it/s]  4%|▍         | 16031/400000 [00:02<00:48, 7884.99it/s]  4%|▍         | 16846/400000 [00:02<00:48, 7960.57it/s]  4%|▍         | 17669/400000 [00:02<00:47, 8039.00it/s]  5%|▍         | 18499/400000 [00:02<00:47, 8115.55it/s]  5%|▍         | 19326/400000 [00:02<00:46, 8160.78it/s]  5%|▌         | 20153/400000 [00:02<00:46, 8191.13it/s]  5%|▌         | 20978/400000 [00:02<00:46, 8208.36it/s]  5%|▌         | 21802/400000 [00:02<00:46, 8215.05it/s]  6%|▌         | 22624/400000 [00:02<00:46, 8135.05it/s]  6%|▌         | 23438/400000 [00:02<00:46, 8030.89it/s]  6%|▌         | 24262/400000 [00:03<00:46, 8090.04it/s]  6%|▋         | 25072/400000 [00:03<00:46, 8063.37it/s]  6%|▋         | 25879/400000 [00:03<00:46, 8034.32it/s]  7%|▋         | 26704/400000 [00:03<00:46, 8095.76it/s]  7%|▋         | 27514/400000 [00:03<00:48, 7752.30it/s]  7%|▋         | 28293/400000 [00:03<00:48, 7628.53it/s]  7%|▋         | 29123/400000 [00:03<00:47, 7818.11it/s]  7%|▋         | 29933/400000 [00:03<00:46, 7898.53it/s]  8%|▊         | 30726/400000 [00:03<00:46, 7898.49it/s]  8%|▊         | 31538/400000 [00:03<00:46, 7960.67it/s]  8%|▊         | 32336/400000 [00:04<00:46, 7964.45it/s]  8%|▊         | 33146/400000 [00:04<00:45, 8003.51it/s]  8%|▊         | 33948/400000 [00:04<00:46, 7902.22it/s]  9%|▊         | 34775/400000 [00:04<00:45, 8007.66it/s]  9%|▉         | 35589/400000 [00:04<00:45, 8044.38it/s]  9%|▉         | 36408/400000 [00:04<00:44, 8085.55it/s]  9%|▉         | 37218/400000 [00:04<00:46, 7765.25it/s]  9%|▉         | 37998/400000 [00:04<00:46, 7735.92it/s] 10%|▉         | 38774/400000 [00:04<00:47, 7669.01it/s] 10%|▉         | 39564/400000 [00:04<00:46, 7736.08it/s] 10%|█         | 40398/400000 [00:05<00:45, 7907.43it/s] 10%|█         | 41191/400000 [00:05<00:45, 7858.99it/s] 11%|█         | 42022/400000 [00:05<00:44, 7987.51it/s] 11%|█         | 42837/400000 [00:05<00:44, 8033.82it/s] 11%|█         | 43673/400000 [00:05<00:43, 8126.99it/s] 11%|█         | 44509/400000 [00:05<00:43, 8193.47it/s] 11%|█▏        | 45345/400000 [00:05<00:43, 8240.02it/s] 12%|█▏        | 46176/400000 [00:05<00:42, 8258.22it/s] 12%|█▏        | 47012/400000 [00:05<00:42, 8286.73it/s] 12%|█▏        | 47842/400000 [00:05<00:42, 8276.34it/s] 12%|█▏        | 48675/400000 [00:06<00:42, 8289.65it/s] 12%|█▏        | 49505/400000 [00:06<00:42, 8204.07it/s] 13%|█▎        | 50326/400000 [00:06<00:43, 8054.53it/s] 13%|█▎        | 51139/400000 [00:06<00:43, 8076.86it/s] 13%|█▎        | 51955/400000 [00:06<00:42, 8100.54it/s] 13%|█▎        | 52792/400000 [00:06<00:42, 8179.31it/s] 13%|█▎        | 53611/400000 [00:06<00:42, 8135.93it/s] 14%|█▎        | 54431/400000 [00:06<00:42, 8153.41it/s] 14%|█▍        | 55255/400000 [00:06<00:42, 8178.22it/s] 14%|█▍        | 56074/400000 [00:07<00:42, 8027.28it/s] 14%|█▍        | 56878/400000 [00:07<00:42, 7988.94it/s] 14%|█▍        | 57688/400000 [00:07<00:42, 8021.36it/s] 15%|█▍        | 58522/400000 [00:07<00:42, 8113.83it/s] 15%|█▍        | 59356/400000 [00:07<00:41, 8178.44it/s] 15%|█▌        | 60182/400000 [00:07<00:41, 8202.11it/s] 15%|█▌        | 61009/400000 [00:07<00:41, 8221.92it/s] 15%|█▌        | 61843/400000 [00:07<00:40, 8255.79it/s] 16%|█▌        | 62673/400000 [00:07<00:40, 8267.23it/s] 16%|█▌        | 63500/400000 [00:07<00:40, 8224.16it/s] 16%|█▌        | 64323/400000 [00:08<00:41, 8184.45it/s] 16%|█▋        | 65158/400000 [00:08<00:40, 8231.05it/s] 16%|█▋        | 65982/400000 [00:08<00:40, 8227.35it/s] 17%|█▋        | 66817/400000 [00:08<00:40, 8261.45it/s] 17%|█▋        | 67644/400000 [00:08<00:40, 8260.57it/s] 17%|█▋        | 68475/400000 [00:08<00:40, 8274.54it/s] 17%|█▋        | 69313/400000 [00:08<00:39, 8304.11it/s] 18%|█▊        | 70150/400000 [00:08<00:39, 8322.00it/s] 18%|█▊        | 70988/400000 [00:08<00:39, 8338.77it/s] 18%|█▊        | 71822/400000 [00:08<00:39, 8338.77it/s] 18%|█▊        | 72656/400000 [00:09<00:39, 8333.50it/s] 18%|█▊        | 73490/400000 [00:09<00:39, 8331.00it/s] 19%|█▊        | 74324/400000 [00:09<00:39, 8223.73it/s] 19%|█▉        | 75160/400000 [00:09<00:39, 8262.43it/s] 19%|█▉        | 75991/400000 [00:09<00:39, 8274.79it/s] 19%|█▉        | 76819/400000 [00:09<00:39, 8249.82it/s] 19%|█▉        | 77645/400000 [00:09<00:39, 8195.51it/s] 20%|█▉        | 78465/400000 [00:09<00:39, 8158.01it/s] 20%|█▉        | 79281/400000 [00:09<00:39, 8089.04it/s] 20%|██        | 80091/400000 [00:09<00:39, 8066.58it/s] 20%|██        | 80898/400000 [00:10<00:39, 8012.99it/s] 20%|██        | 81700/400000 [00:10<00:40, 7943.77it/s] 21%|██        | 82514/400000 [00:10<00:39, 8001.29it/s] 21%|██        | 83319/400000 [00:10<00:39, 8013.78it/s] 21%|██        | 84121/400000 [00:10<00:39, 8010.62it/s] 21%|██        | 84923/400000 [00:10<00:39, 7992.45it/s] 21%|██▏       | 85746/400000 [00:10<00:38, 8061.05it/s] 22%|██▏       | 86553/400000 [00:10<00:39, 8036.55it/s] 22%|██▏       | 87357/400000 [00:10<00:39, 8004.78it/s] 22%|██▏       | 88168/400000 [00:10<00:38, 8035.65it/s] 22%|██▏       | 88984/400000 [00:11<00:38, 8070.90it/s] 22%|██▏       | 89792/400000 [00:11<00:38, 7978.97it/s] 23%|██▎       | 90591/400000 [00:11<00:39, 7916.80it/s] 23%|██▎       | 91384/400000 [00:11<00:41, 7514.71it/s] 23%|██▎       | 92200/400000 [00:11<00:39, 7696.36it/s] 23%|██▎       | 93012/400000 [00:11<00:39, 7813.42it/s] 23%|██▎       | 93813/400000 [00:11<00:38, 7869.25it/s] 24%|██▎       | 94627/400000 [00:11<00:38, 7947.28it/s] 24%|██▍       | 95446/400000 [00:11<00:37, 8017.86it/s] 24%|██▍       | 96275/400000 [00:11<00:37, 8096.87it/s] 24%|██▍       | 97104/400000 [00:12<00:37, 8152.49it/s] 24%|██▍       | 97938/400000 [00:12<00:36, 8205.40it/s] 25%|██▍       | 98760/400000 [00:12<00:37, 8103.18it/s] 25%|██▍       | 99572/400000 [00:12<00:38, 7842.85it/s] 25%|██▌       | 100400/400000 [00:12<00:37, 7967.09it/s] 25%|██▌       | 101212/400000 [00:12<00:37, 8010.02it/s] 26%|██▌       | 102029/400000 [00:12<00:36, 8056.20it/s] 26%|██▌       | 102856/400000 [00:12<00:36, 8116.82it/s] 26%|██▌       | 103676/400000 [00:12<00:36, 8138.82it/s] 26%|██▌       | 104491/400000 [00:12<00:36, 8079.02it/s] 26%|██▋       | 105300/400000 [00:13<00:36, 8046.69it/s] 27%|██▋       | 106114/400000 [00:13<00:36, 8072.40it/s] 27%|██▋       | 106922/400000 [00:13<00:36, 8030.41it/s] 27%|██▋       | 107735/400000 [00:13<00:36, 8058.30it/s] 27%|██▋       | 108556/400000 [00:13<00:35, 8101.23it/s] 27%|██▋       | 109367/400000 [00:13<00:36, 7995.76it/s] 28%|██▊       | 110202/400000 [00:13<00:35, 8097.37it/s] 28%|██▊       | 111013/400000 [00:13<00:36, 8020.34it/s] 28%|██▊       | 111816/400000 [00:13<00:35, 8016.00it/s] 28%|██▊       | 112619/400000 [00:13<00:36, 7921.60it/s] 28%|██▊       | 113441/400000 [00:14<00:35, 8006.24it/s] 29%|██▊       | 114243/400000 [00:14<00:35, 7987.08it/s] 29%|██▉       | 115043/400000 [00:14<00:35, 7948.12it/s] 29%|██▉       | 115855/400000 [00:14<00:35, 7996.39it/s] 29%|██▉       | 116655/400000 [00:14<00:35, 7978.81it/s] 29%|██▉       | 117484/400000 [00:14<00:35, 8068.34it/s] 30%|██▉       | 118309/400000 [00:14<00:34, 8120.77it/s] 30%|██▉       | 119122/400000 [00:14<00:35, 7930.47it/s] 30%|██▉       | 119917/400000 [00:14<00:35, 7872.42it/s] 30%|███       | 120726/400000 [00:15<00:35, 7936.31it/s] 30%|███       | 121543/400000 [00:15<00:34, 8002.94it/s] 31%|███       | 122372/400000 [00:15<00:34, 8084.68it/s] 31%|███       | 123191/400000 [00:15<00:34, 8113.30it/s] 31%|███       | 124003/400000 [00:15<00:34, 7909.16it/s] 31%|███       | 124826/400000 [00:15<00:34, 8001.06it/s] 31%|███▏      | 125640/400000 [00:15<00:34, 8039.51it/s] 32%|███▏      | 126460/400000 [00:15<00:33, 8085.62it/s] 32%|███▏      | 127299/400000 [00:15<00:33, 8173.50it/s] 32%|███▏      | 128127/400000 [00:15<00:33, 8203.62it/s] 32%|███▏      | 128963/400000 [00:16<00:32, 8247.93it/s] 32%|███▏      | 129798/400000 [00:16<00:32, 8275.55it/s] 33%|███▎      | 130633/400000 [00:16<00:32, 8296.24it/s] 33%|███▎      | 131467/400000 [00:16<00:32, 8306.88it/s] 33%|███▎      | 132298/400000 [00:16<00:33, 8073.61it/s] 33%|███▎      | 133135/400000 [00:16<00:32, 8158.24it/s] 33%|███▎      | 133969/400000 [00:16<00:32, 8209.92it/s] 34%|███▎      | 134791/400000 [00:16<00:32, 8197.28it/s] 34%|███▍      | 135629/400000 [00:16<00:32, 8250.30it/s] 34%|███▍      | 136461/400000 [00:16<00:31, 8270.16it/s] 34%|███▍      | 137289/400000 [00:17<00:33, 7877.83it/s] 35%|███▍      | 138094/400000 [00:17<00:33, 7927.72it/s] 35%|███▍      | 138890/400000 [00:17<00:32, 7923.90it/s] 35%|███▍      | 139685/400000 [00:17<00:32, 7921.19it/s] 35%|███▌      | 140483/400000 [00:17<00:32, 7937.77it/s] 35%|███▌      | 141291/400000 [00:17<00:32, 7977.92it/s] 36%|███▌      | 142091/400000 [00:17<00:32, 7981.81it/s] 36%|███▌      | 142902/400000 [00:17<00:32, 8018.96it/s] 36%|███▌      | 143720/400000 [00:17<00:31, 8066.60it/s] 36%|███▌      | 144528/400000 [00:17<00:31, 8067.01it/s] 36%|███▋      | 145335/400000 [00:18<00:31, 8041.86it/s] 37%|███▋      | 146140/400000 [00:18<00:32, 7736.30it/s] 37%|███▋      | 146960/400000 [00:18<00:32, 7869.72it/s] 37%|███▋      | 147785/400000 [00:18<00:31, 7977.61it/s] 37%|███▋      | 148589/400000 [00:18<00:31, 7993.68it/s] 37%|███▋      | 149421/400000 [00:18<00:30, 8088.76it/s] 38%|███▊      | 150252/400000 [00:18<00:30, 8142.24it/s] 38%|███▊      | 151069/400000 [00:18<00:30, 8148.53it/s] 38%|███▊      | 151908/400000 [00:18<00:30, 8218.21it/s] 38%|███▊      | 152737/400000 [00:18<00:30, 8238.84it/s] 38%|███▊      | 153562/400000 [00:19<00:30, 8209.69it/s] 39%|███▊      | 154384/400000 [00:19<00:30, 8172.82it/s] 39%|███▉      | 155202/400000 [00:19<00:30, 8151.32it/s] 39%|███▉      | 156018/400000 [00:19<00:30, 8130.57it/s] 39%|███▉      | 156832/400000 [00:19<00:30, 7883.07it/s] 39%|███▉      | 157652/400000 [00:19<00:30, 7973.47it/s] 40%|███▉      | 158451/400000 [00:19<00:30, 7951.74it/s] 40%|███▉      | 159248/400000 [00:19<00:30, 7880.31it/s] 40%|████      | 160065/400000 [00:19<00:30, 7962.28it/s] 40%|████      | 160876/400000 [00:19<00:29, 8003.27it/s] 40%|████      | 161685/400000 [00:20<00:29, 8028.08it/s] 41%|████      | 162489/400000 [00:20<00:30, 7890.12it/s] 41%|████      | 163296/400000 [00:20<00:29, 7940.46it/s] 41%|████      | 164091/400000 [00:20<00:31, 7551.40it/s] 41%|████      | 164851/400000 [00:20<00:31, 7511.28it/s] 41%|████▏     | 165606/400000 [00:20<00:31, 7465.20it/s] 42%|████▏     | 166355/400000 [00:20<00:31, 7322.39it/s] 42%|████▏     | 167162/400000 [00:20<00:30, 7529.64it/s] 42%|████▏     | 167931/400000 [00:20<00:30, 7575.00it/s] 42%|████▏     | 168691/400000 [00:21<00:31, 7374.74it/s] 42%|████▏     | 169440/400000 [00:21<00:31, 7407.41it/s] 43%|████▎     | 170212/400000 [00:21<00:30, 7496.89it/s] 43%|████▎     | 171033/400000 [00:21<00:29, 7695.70it/s] 43%|████▎     | 171845/400000 [00:21<00:29, 7817.76it/s] 43%|████▎     | 172666/400000 [00:21<00:28, 7929.08it/s] 43%|████▎     | 173484/400000 [00:21<00:28, 8002.38it/s] 44%|████▎     | 174308/400000 [00:21<00:27, 8071.81it/s] 44%|████▍     | 175117/400000 [00:21<00:28, 7902.89it/s] 44%|████▍     | 175939/400000 [00:21<00:28, 7993.89it/s] 44%|████▍     | 176759/400000 [00:22<00:27, 8052.02it/s] 44%|████▍     | 177571/400000 [00:22<00:27, 8070.73it/s] 45%|████▍     | 178379/400000 [00:22<00:27, 7969.74it/s] 45%|████▍     | 179203/400000 [00:22<00:27, 8047.64it/s] 45%|████▌     | 180009/400000 [00:22<00:28, 7809.05it/s] 45%|████▌     | 180827/400000 [00:22<00:27, 7915.00it/s] 45%|████▌     | 181646/400000 [00:22<00:27, 7995.20it/s] 46%|████▌     | 182469/400000 [00:22<00:26, 8063.77it/s] 46%|████▌     | 183279/400000 [00:22<00:26, 8073.42it/s] 46%|████▌     | 184091/400000 [00:22<00:26, 8087.15it/s] 46%|████▌     | 184911/400000 [00:23<00:26, 8120.31it/s] 46%|████▋     | 185724/400000 [00:23<00:26, 8057.14it/s] 47%|████▋     | 186531/400000 [00:23<00:26, 8030.25it/s] 47%|████▋     | 187337/400000 [00:23<00:26, 8036.31it/s] 47%|████▋     | 188141/400000 [00:23<00:26, 7995.52it/s] 47%|████▋     | 188949/400000 [00:23<00:26, 8020.57it/s] 47%|████▋     | 189752/400000 [00:23<00:26, 7839.14it/s] 48%|████▊     | 190573/400000 [00:23<00:26, 7944.46it/s] 48%|████▊     | 191382/400000 [00:23<00:26, 7985.41it/s] 48%|████▊     | 192197/400000 [00:23<00:25, 8033.41it/s] 48%|████▊     | 193002/400000 [00:24<00:25, 8035.73it/s] 48%|████▊     | 193812/400000 [00:24<00:25, 8052.54it/s] 49%|████▊     | 194634/400000 [00:24<00:25, 8100.11it/s] 49%|████▉     | 195450/400000 [00:24<00:25, 8115.63it/s] 49%|████▉     | 196262/400000 [00:24<00:25, 8081.89it/s] 49%|████▉     | 197071/400000 [00:24<00:25, 7996.87it/s] 49%|████▉     | 197899/400000 [00:24<00:25, 8077.29it/s] 50%|████▉     | 198708/400000 [00:24<00:25, 7996.10it/s] 50%|████▉     | 199513/400000 [00:24<00:25, 8009.51it/s] 50%|█████     | 200315/400000 [00:24<00:25, 7974.13it/s] 50%|█████     | 201141/400000 [00:25<00:24, 8055.11it/s] 50%|█████     | 201956/400000 [00:25<00:24, 8081.69it/s] 51%|█████     | 202765/400000 [00:25<00:24, 8072.54it/s] 51%|█████     | 203573/400000 [00:25<00:24, 8035.09it/s] 51%|█████     | 204377/400000 [00:25<00:24, 7942.19it/s] 51%|█████▏    | 205172/400000 [00:25<00:25, 7657.96it/s] 51%|█████▏    | 205992/400000 [00:25<00:24, 7812.01it/s] 52%|█████▏    | 206798/400000 [00:25<00:24, 7884.66it/s] 52%|█████▏    | 207602/400000 [00:25<00:24, 7930.44it/s] 52%|█████▏    | 208404/400000 [00:25<00:24, 7956.54it/s] 52%|█████▏    | 209222/400000 [00:26<00:23, 8019.57it/s] 53%|█████▎    | 210034/400000 [00:26<00:23, 8046.98it/s] 53%|█████▎    | 210840/400000 [00:26<00:23, 8028.88it/s] 53%|█████▎    | 211651/400000 [00:26<00:23, 8050.55it/s] 53%|█████▎    | 212469/400000 [00:26<00:23, 8088.10it/s] 53%|█████▎    | 213279/400000 [00:26<00:23, 8046.50it/s] 54%|█████▎    | 214103/400000 [00:26<00:22, 8102.96it/s] 54%|█████▎    | 214916/400000 [00:26<00:22, 8110.14it/s] 54%|█████▍    | 215728/400000 [00:26<00:23, 7977.59it/s] 54%|█████▍    | 216527/400000 [00:26<00:22, 7979.87it/s] 54%|█████▍    | 217326/400000 [00:27<00:23, 7875.34it/s] 55%|█████▍    | 218115/400000 [00:27<00:23, 7700.96it/s] 55%|█████▍    | 218930/400000 [00:27<00:23, 7828.99it/s] 55%|█████▍    | 219722/400000 [00:27<00:22, 7854.97it/s] 55%|█████▌    | 220509/400000 [00:27<00:23, 7791.39it/s] 55%|█████▌    | 221298/400000 [00:27<00:22, 7818.03it/s] 56%|█████▌    | 222110/400000 [00:27<00:22, 7904.94it/s] 56%|█████▌    | 222938/400000 [00:27<00:22, 8013.41it/s] 56%|█████▌    | 223741/400000 [00:27<00:22, 7946.69it/s] 56%|█████▌    | 224537/400000 [00:28<00:22, 7935.74it/s] 56%|█████▋    | 225356/400000 [00:28<00:21, 8009.89it/s] 57%|█████▋    | 226158/400000 [00:28<00:21, 7985.39it/s] 57%|█████▋    | 226978/400000 [00:28<00:21, 8047.82it/s] 57%|█████▋    | 227784/400000 [00:28<00:21, 8003.38it/s] 57%|█████▋    | 228594/400000 [00:28<00:21, 8030.01it/s] 57%|█████▋    | 229412/400000 [00:28<00:21, 8072.29it/s] 58%|█████▊    | 230222/400000 [00:28<00:21, 8078.47it/s] 58%|█████▊    | 231031/400000 [00:28<00:20, 8070.85it/s] 58%|█████▊    | 231839/400000 [00:28<00:20, 8034.76it/s] 58%|█████▊    | 232644/400000 [00:29<00:20, 8036.73it/s] 58%|█████▊    | 233468/400000 [00:29<00:20, 8094.68it/s] 59%|█████▊    | 234278/400000 [00:29<00:20, 8093.12it/s] 59%|█████▉    | 235088/400000 [00:29<00:21, 7841.83it/s] 59%|█████▉    | 235898/400000 [00:29<00:20, 7915.33it/s] 59%|█████▉    | 236691/400000 [00:29<00:21, 7691.38it/s] 59%|█████▉    | 237482/400000 [00:29<00:20, 7755.38it/s] 60%|█████▉    | 238311/400000 [00:29<00:20, 7906.84it/s] 60%|█████▉    | 239125/400000 [00:29<00:20, 7975.12it/s] 60%|█████▉    | 239936/400000 [00:29<00:19, 8014.78it/s] 60%|██████    | 240752/400000 [00:30<00:19, 8056.93it/s] 60%|██████    | 241574/400000 [00:30<00:19, 8104.58it/s] 61%|██████    | 242396/400000 [00:30<00:19, 8138.71it/s] 61%|██████    | 243211/400000 [00:30<00:19, 8057.28it/s] 61%|██████    | 244018/400000 [00:30<00:19, 8000.98it/s] 61%|██████    | 244819/400000 [00:30<00:19, 7931.18it/s] 61%|██████▏   | 245613/400000 [00:30<00:19, 7845.77it/s] 62%|██████▏   | 246420/400000 [00:30<00:19, 7911.28it/s] 62%|██████▏   | 247243/400000 [00:30<00:19, 8002.02it/s] 62%|██████▏   | 248044/400000 [00:30<00:19, 7866.53it/s] 62%|██████▏   | 248832/400000 [00:31<00:19, 7785.81it/s] 62%|██████▏   | 249620/400000 [00:31<00:19, 7811.88it/s] 63%|██████▎   | 250443/400000 [00:31<00:18, 7930.15it/s] 63%|██████▎   | 251247/400000 [00:31<00:18, 7961.38it/s] 63%|██████▎   | 252044/400000 [00:31<00:18, 7932.42it/s] 63%|██████▎   | 252838/400000 [00:31<00:18, 7827.13it/s] 63%|██████▎   | 253657/400000 [00:31<00:18, 7930.83it/s] 64%|██████▎   | 254478/400000 [00:31<00:18, 8011.97it/s] 64%|██████▍   | 255311/400000 [00:31<00:17, 8102.11it/s] 64%|██████▍   | 256128/400000 [00:31<00:17, 8120.46it/s] 64%|██████▍   | 256954/400000 [00:32<00:17, 8160.67it/s] 64%|██████▍   | 257771/400000 [00:32<00:17, 7906.97it/s] 65%|██████▍   | 258564/400000 [00:32<00:17, 7881.13it/s] 65%|██████▍   | 259358/400000 [00:32<00:17, 7897.51it/s] 65%|██████▌   | 260177/400000 [00:32<00:17, 7981.16it/s] 65%|██████▌   | 260992/400000 [00:32<00:17, 8030.97it/s] 65%|██████▌   | 261818/400000 [00:32<00:17, 8097.35it/s] 66%|██████▌   | 262629/400000 [00:32<00:16, 8083.93it/s] 66%|██████▌   | 263458/400000 [00:32<00:16, 8138.58it/s] 66%|██████▌   | 264286/400000 [00:32<00:16, 8178.55it/s] 66%|██████▋   | 265105/400000 [00:33<00:16, 8140.63it/s] 66%|██████▋   | 265935/400000 [00:33<00:16, 8185.99it/s] 67%|██████▋   | 266754/400000 [00:33<00:16, 8179.09it/s] 67%|██████▋   | 267573/400000 [00:33<00:16, 8171.93it/s] 67%|██████▋   | 268391/400000 [00:33<00:16, 8075.31it/s] 67%|██████▋   | 269199/400000 [00:33<00:16, 8039.12it/s] 68%|██████▊   | 270009/400000 [00:33<00:16, 8055.38it/s] 68%|██████▊   | 270833/400000 [00:33<00:15, 8108.44it/s] 68%|██████▊   | 271658/400000 [00:33<00:15, 8148.68it/s] 68%|██████▊   | 272474/400000 [00:34<00:16, 7822.48it/s] 68%|██████▊   | 273284/400000 [00:34<00:16, 7901.58it/s] 69%|██████▊   | 274091/400000 [00:34<00:15, 7949.71it/s] 69%|██████▊   | 274888/400000 [00:34<00:15, 7910.80it/s] 69%|██████▉   | 275702/400000 [00:34<00:15, 7977.11it/s] 69%|██████▉   | 276502/400000 [00:34<00:15, 7980.68it/s] 69%|██████▉   | 277301/400000 [00:34<00:15, 7816.73it/s] 70%|██████▉   | 278084/400000 [00:34<00:15, 7795.69it/s] 70%|██████▉   | 278879/400000 [00:34<00:15, 7838.88it/s] 70%|██████▉   | 279671/400000 [00:34<00:15, 7861.80it/s] 70%|███████   | 280486/400000 [00:35<00:15, 7945.17it/s] 70%|███████   | 281304/400000 [00:35<00:14, 8012.05it/s] 71%|███████   | 282128/400000 [00:35<00:14, 8077.67it/s] 71%|███████   | 282937/400000 [00:35<00:14, 7908.22it/s] 71%|███████   | 283729/400000 [00:35<00:14, 7829.61it/s] 71%|███████   | 284528/400000 [00:35<00:14, 7876.56it/s] 71%|███████▏  | 285317/400000 [00:35<00:15, 7608.48it/s] 72%|███████▏  | 286101/400000 [00:35<00:14, 7670.98it/s] 72%|███████▏  | 286908/400000 [00:35<00:14, 7783.95it/s] 72%|███████▏  | 287718/400000 [00:35<00:14, 7874.16it/s] 72%|███████▏  | 288520/400000 [00:36<00:14, 7917.09it/s] 72%|███████▏  | 289313/400000 [00:36<00:14, 7904.52it/s] 73%|███████▎  | 290116/400000 [00:36<00:13, 7941.60it/s] 73%|███████▎  | 290927/400000 [00:36<00:13, 7990.48it/s] 73%|███████▎  | 291732/400000 [00:36<00:13, 8007.01it/s] 73%|███████▎  | 292542/400000 [00:36<00:13, 8033.68it/s] 73%|███████▎  | 293346/400000 [00:36<00:13, 7886.22it/s] 74%|███████▎  | 294138/400000 [00:36<00:13, 7894.81it/s] 74%|███████▎  | 294929/400000 [00:36<00:13, 7877.84it/s] 74%|███████▍  | 295750/400000 [00:36<00:13, 7972.35it/s] 74%|███████▍  | 296552/400000 [00:37<00:12, 7985.43it/s] 74%|███████▍  | 297367/400000 [00:37<00:12, 8032.00it/s] 75%|███████▍  | 298171/400000 [00:37<00:12, 7857.34it/s] 75%|███████▍  | 298975/400000 [00:37<00:12, 7907.51it/s] 75%|███████▍  | 299774/400000 [00:37<00:12, 7931.48it/s] 75%|███████▌  | 300581/400000 [00:37<00:12, 7971.36it/s] 75%|███████▌  | 301379/400000 [00:37<00:12, 7706.01it/s] 76%|███████▌  | 302152/400000 [00:37<00:12, 7703.26it/s] 76%|███████▌  | 302924/400000 [00:37<00:12, 7638.99it/s] 76%|███████▌  | 303748/400000 [00:37<00:12, 7807.76it/s] 76%|███████▌  | 304571/400000 [00:38<00:12, 7927.49it/s] 76%|███████▋  | 305377/400000 [00:38<00:11, 7965.81it/s] 77%|███████▋  | 306175/400000 [00:38<00:11, 7964.95it/s] 77%|███████▋  | 306973/400000 [00:38<00:11, 7932.59it/s] 77%|███████▋  | 307784/400000 [00:38<00:11, 7984.92it/s] 77%|███████▋  | 308584/400000 [00:38<00:11, 7985.21it/s] 77%|███████▋  | 309383/400000 [00:38<00:11, 7788.96it/s] 78%|███████▊  | 310204/400000 [00:38<00:11, 7904.47it/s] 78%|███████▊  | 311004/400000 [00:38<00:11, 7932.70it/s] 78%|███████▊  | 311815/400000 [00:38<00:11, 7982.06it/s] 78%|███████▊  | 312617/400000 [00:39<00:10, 7992.35it/s] 78%|███████▊  | 313417/400000 [00:39<00:10, 7978.34it/s] 79%|███████▊  | 314218/400000 [00:39<00:10, 7987.34it/s] 79%|███████▉  | 315041/400000 [00:39<00:10, 8056.85it/s] 79%|███████▉  | 315865/400000 [00:39<00:10, 8108.27it/s] 79%|███████▉  | 316686/400000 [00:39<00:10, 8136.45it/s] 79%|███████▉  | 317500/400000 [00:39<00:10, 8102.62it/s] 80%|███████▉  | 318311/400000 [00:39<00:10, 8044.33it/s] 80%|███████▉  | 319118/400000 [00:39<00:10, 8049.45it/s] 80%|███████▉  | 319924/400000 [00:39<00:10, 7993.59it/s] 80%|████████  | 320734/400000 [00:40<00:09, 8023.22it/s] 80%|████████  | 321539/400000 [00:40<00:09, 8030.76it/s] 81%|████████  | 322363/400000 [00:40<00:09, 8089.77it/s] 81%|████████  | 323173/400000 [00:40<00:09, 8055.37it/s] 81%|████████  | 323984/400000 [00:40<00:09, 8070.28it/s] 81%|████████  | 324816/400000 [00:40<00:09, 8142.46it/s] 81%|████████▏ | 325631/400000 [00:40<00:09, 7945.37it/s] 82%|████████▏ | 326438/400000 [00:40<00:09, 7980.88it/s] 82%|████████▏ | 327241/400000 [00:40<00:09, 7995.07it/s] 82%|████████▏ | 328042/400000 [00:41<00:09, 7940.90it/s] 82%|████████▏ | 328848/400000 [00:41<00:08, 7975.21it/s] 82%|████████▏ | 329662/400000 [00:41<00:08, 8021.77it/s] 83%|████████▎ | 330489/400000 [00:41<00:08, 8093.39it/s] 83%|████████▎ | 331300/400000 [00:41<00:08, 8097.71it/s] 83%|████████▎ | 332111/400000 [00:41<00:08, 8049.01it/s] 83%|████████▎ | 332917/400000 [00:41<00:08, 8007.41it/s] 83%|████████▎ | 333720/400000 [00:41<00:08, 8013.47it/s] 84%|████████▎ | 334531/400000 [00:41<00:08, 8039.49it/s] 84%|████████▍ | 335336/400000 [00:41<00:08, 8012.81it/s] 84%|████████▍ | 336151/400000 [00:42<00:07, 8052.24it/s] 84%|████████▍ | 336960/400000 [00:42<00:07, 8062.80it/s] 84%|████████▍ | 337767/400000 [00:42<00:08, 7756.47it/s] 85%|████████▍ | 338547/400000 [00:42<00:07, 7768.31it/s] 85%|████████▍ | 339326/400000 [00:42<00:07, 7767.43it/s] 85%|████████▌ | 340125/400000 [00:42<00:07, 7814.16it/s] 85%|████████▌ | 340908/400000 [00:42<00:07, 7789.35it/s] 85%|████████▌ | 341688/400000 [00:42<00:07, 7688.25it/s] 86%|████████▌ | 342503/400000 [00:42<00:07, 7819.00it/s] 86%|████████▌ | 343328/400000 [00:42<00:07, 7943.12it/s] 86%|████████▌ | 344126/400000 [00:43<00:07, 7953.55it/s] 86%|████████▌ | 344939/400000 [00:43<00:06, 8003.95it/s] 86%|████████▋ | 345756/400000 [00:43<00:06, 8050.72it/s] 87%|████████▋ | 346573/400000 [00:43<00:06, 8085.73it/s] 87%|████████▋ | 347388/400000 [00:43<00:06, 8102.33it/s] 87%|████████▋ | 348199/400000 [00:43<00:06, 8042.95it/s] 87%|████████▋ | 349014/400000 [00:43<00:06, 8072.88it/s] 87%|████████▋ | 349822/400000 [00:43<00:06, 8017.61it/s] 88%|████████▊ | 350625/400000 [00:43<00:06, 8021.32it/s] 88%|████████▊ | 351439/400000 [00:43<00:06, 8055.92it/s] 88%|████████▊ | 352245/400000 [00:44<00:05, 7979.05it/s] 88%|████████▊ | 353057/400000 [00:44<00:05, 8019.16it/s] 88%|████████▊ | 353860/400000 [00:44<00:05, 7882.82it/s] 89%|████████▊ | 354649/400000 [00:44<00:05, 7743.96it/s] 89%|████████▉ | 355455/400000 [00:44<00:05, 7833.99it/s] 89%|████████▉ | 356261/400000 [00:44<00:05, 7900.18it/s] 89%|████████▉ | 357058/400000 [00:44<00:05, 7918.65it/s] 89%|████████▉ | 357882/400000 [00:44<00:05, 8010.21it/s] 90%|████████▉ | 358684/400000 [00:44<00:05, 8005.98it/s] 90%|████████▉ | 359486/400000 [00:44<00:05, 7927.71it/s] 90%|█████████ | 360280/400000 [00:45<00:05, 7635.05it/s] 90%|█████████ | 361078/400000 [00:45<00:05, 7734.94it/s] 90%|█████████ | 361873/400000 [00:45<00:04, 7796.81it/s] 91%|█████████ | 362663/400000 [00:45<00:04, 7824.33it/s] 91%|█████████ | 363447/400000 [00:45<00:04, 7765.59it/s] 91%|█████████ | 364232/400000 [00:45<00:04, 7790.47it/s] 91%|█████████▏| 365012/400000 [00:45<00:04, 7762.36it/s] 91%|█████████▏| 365789/400000 [00:45<00:04, 7636.56it/s] 92%|█████████▏| 366554/400000 [00:45<00:04, 7548.81it/s] 92%|█████████▏| 367375/400000 [00:45<00:04, 7735.21it/s] 92%|█████████▏| 368178/400000 [00:46<00:04, 7820.61it/s] 92%|█████████▏| 369009/400000 [00:46<00:03, 7961.25it/s] 92%|█████████▏| 369828/400000 [00:46<00:03, 8024.46it/s] 93%|█████████▎| 370644/400000 [00:46<00:03, 8062.48it/s] 93%|█████████▎| 371456/400000 [00:46<00:03, 8077.63it/s] 93%|█████████▎| 372265/400000 [00:46<00:03, 7927.68it/s] 93%|█████████▎| 373092/400000 [00:46<00:03, 8027.13it/s] 93%|█████████▎| 373896/400000 [00:46<00:03, 7994.87it/s] 94%|█████████▎| 374711/400000 [00:46<00:03, 8040.19it/s] 94%|█████████▍| 375526/400000 [00:46<00:03, 8072.32it/s] 94%|█████████▍| 376354/400000 [00:47<00:02, 8131.24it/s] 94%|█████████▍| 377168/400000 [00:47<00:02, 8102.00it/s] 94%|█████████▍| 377979/400000 [00:47<00:02, 8089.04it/s] 95%|█████████▍| 378789/400000 [00:47<00:02, 8078.49it/s] 95%|█████████▍| 379598/400000 [00:47<00:02, 8019.60it/s] 95%|█████████▌| 380401/400000 [00:47<00:02, 7836.75it/s] 95%|█████████▌| 381213/400000 [00:47<00:02, 7917.07it/s] 96%|█████████▌| 382006/400000 [00:47<00:02, 7863.08it/s] 96%|█████████▌| 382801/400000 [00:47<00:02, 7888.30it/s] 96%|█████████▌| 383616/400000 [00:48<00:02, 7962.78it/s] 96%|█████████▌| 384436/400000 [00:48<00:01, 8031.50it/s] 96%|█████████▋| 385266/400000 [00:48<00:01, 8109.78it/s] 97%|█████████▋| 386090/400000 [00:48<00:01, 8146.69it/s] 97%|█████████▋| 386910/400000 [00:48<00:01, 8162.26it/s] 97%|█████████▋| 387727/400000 [00:48<00:01, 8162.74it/s] 97%|█████████▋| 388544/400000 [00:48<00:01, 8130.09it/s] 97%|█████████▋| 389358/400000 [00:48<00:01, 8125.18it/s] 98%|█████████▊| 390175/400000 [00:48<00:01, 8137.01it/s] 98%|█████████▊| 391010/400000 [00:48<00:01, 8198.34it/s] 98%|█████████▊| 391831/400000 [00:49<00:01, 7983.03it/s] 98%|█████████▊| 392631/400000 [00:49<00:00, 7813.82it/s] 98%|█████████▊| 393454/400000 [00:49<00:00, 7933.24it/s] 99%|█████████▊| 394275/400000 [00:49<00:00, 8011.62it/s] 99%|█████████▉| 395105/400000 [00:49<00:00, 8094.14it/s] 99%|█████████▉| 395916/400000 [00:49<00:00, 8088.58it/s] 99%|█████████▉| 396726/400000 [00:49<00:00, 8063.77it/s] 99%|█████████▉| 397557/400000 [00:49<00:00, 8134.95it/s]100%|█████████▉| 398396/400000 [00:49<00:00, 8208.94it/s]100%|█████████▉| 399221/400000 [00:49<00:00, 8219.27it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7995.72it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4025aa0cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011334866754316294 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.01098021496099772 	 Accuracy: 66

  model saves at 66% accuracy 

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

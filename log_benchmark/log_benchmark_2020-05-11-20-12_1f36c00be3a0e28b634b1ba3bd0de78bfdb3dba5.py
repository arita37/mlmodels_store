
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5a358d9f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 20:12:40.567668
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 20:12:40.571912
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 20:12:40.575506
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 20:12:40.579017
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5a4169d3c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351374.6562
Epoch 2/10

1/1 [==============================] - 0s 112ms/step - loss: 238705.1094
Epoch 3/10

1/1 [==============================] - 0s 122ms/step - loss: 139634.0000
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 77000.3281
Epoch 5/10

1/1 [==============================] - 0s 123ms/step - loss: 44783.7656
Epoch 6/10

1/1 [==============================] - 0s 132ms/step - loss: 28373.3281
Epoch 7/10

1/1 [==============================] - 0s 109ms/step - loss: 19349.9453
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 14008.2900
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 10653.9834
Epoch 10/10

1/1 [==============================] - 0s 117ms/step - loss: 8448.0195

  #### Inference Need return ypred, ytrue ######################### 
[[-0.40045395  0.84904814 -0.11000368  1.6968876   1.4561554  -0.69428
  -0.78736615  1.3662785   1.2822577  -0.14452234 -0.16780935  0.0612343
  -0.10527712 -1.9427654   0.13697638  1.603116    0.98078793 -0.34698635
   0.46785814  1.1493037  -0.80210227  1.0786531  -1.6020443  -0.69472986
   1.2248063   1.1993539   0.5784871  -1.744771   -0.7462034   0.44405246
  -0.74224746 -0.31470895  0.3825446   0.70601344 -0.34544015  1.6276722
  -1.6126684   0.03750101  0.5015593  -0.9290436  -0.92366326 -0.70507675
  -0.37150314 -0.46731764 -0.08842954  1.5684985   0.61199445  0.9616992
   0.13724153  0.43077913 -0.17701393 -1.8814828  -0.18796879 -0.7103878
  -0.30766582  1.2915118   0.01992053 -0.5191862   0.28357652  0.32891375
   0.39291644 -0.648913    1.1462505   0.5121459  -0.47761172 -0.03624272
  -0.8052263   1.3168646   1.0372212   0.6067568   2.3567176   0.28088945
   0.70604855 -0.20381042  1.4783549   1.3294415   1.1475973  -0.4074784
   0.07466626 -0.32479    -0.27739844  0.56195253 -1.0678391  -0.9081044
   0.92621267  0.3508493   0.9636005  -0.21376151 -0.98578954  0.6235033
   0.06114404 -0.01453194 -0.12296517  0.8549972  -0.70901227  0.3740847
   1.5913712   0.02323437  0.8606459  -0.6159682   0.42798275  0.37359208
   1.8308673   0.06995662  0.9638658  -0.07277252 -1.0970966   1.1046906
   0.9575442  -0.5526216  -0.35539547 -0.13222694 -0.01821306 -0.48318362
  -0.3245894  -0.41516575 -2.0614965   0.41230106 -1.4190212   1.7163395
  -0.11003623  6.7951727   8.104716    7.883784    5.7558002   7.354093
   7.5595675   7.0051103   8.24153     6.3323627   7.745344    7.533933
   5.694289    7.4571486   8.425466    6.8822083   6.1871457   6.9679832
   7.085464    6.269819    6.1788745   6.907262    5.81514     7.915078
   6.200051    7.2042      5.4121146   5.9024916   7.2999516   5.3780737
   7.310646    6.8365865   7.3521304   6.3753495   6.9853344   7.9478245
   7.467074    6.8635736   7.136996    7.8752937   5.819846    7.040619
   6.3339677   5.3456526   6.990943    6.2961235   7.1500134   7.828128
   6.194664    4.768322    6.47008     7.1242557   6.6540427   6.9459023
   6.9859695   6.033644    5.821139    7.415054    7.232064    7.896524
   0.42534596  0.4059245   0.12325883  0.18707395  1.8855296   0.2908703
   0.7589425   0.7721697   2.2115555   1.5514002   1.6508155   2.4813814
   0.62637913  0.20212764  1.586507    0.2645896   0.6123878   0.76424193
   1.1950752   3.319211    0.9359551   1.9984062   1.6664436   0.3504057
   1.1624053   1.6815926   0.394171    1.8471701   2.0714564   0.54930556
   0.6583768   1.3219819   1.770766    1.6164112   2.0393882   1.4973009
   1.3730139   1.3338555   2.5685081   1.2968038   2.3429947   1.7194874
   0.5648183   2.6597104   1.6609204   2.3896565   1.0329669   0.23807466
   0.14801526  0.9103107   0.38423896  2.6873555   1.2551632   1.9671721
   0.43396378  0.9086172   0.3560828   0.8000532   0.869891    0.7172803
   0.93711126  0.7243672   0.7920836   0.37443155  0.6950439   0.14549184
   1.6288579   0.86568093  0.50275326  0.59818125  1.0199897   0.19281572
   1.5290899   0.56472903  1.0091474   2.3466969   1.309751    0.3005079
   1.4630841   0.26229668  0.44177294  1.5146952   0.9183969   1.414001
   1.3822285   0.9048823   1.3527344   0.8571231   0.40552652  2.0022812
   1.3953001   0.8474587   1.3826069   0.50880986  0.518843    1.4584606
   1.3370402   0.49680448  0.34296107  1.7514393   0.75003433  0.55420053
   0.7107116   0.65512204  1.3964132   2.8462725   2.3730683   0.8924057
   0.6221781   0.8036786   0.96196234  2.2306461   1.7061915   1.7404597
   2.293414    2.4589176   1.4615843   2.305253    0.59802616  1.1103636
   0.29636395  6.576735    5.2206807   8.063211    8.280313    6.92647
   7.0475373   7.6776614   8.283029    6.972506    8.6795225   7.4121113
   8.633838    6.590185    6.7544      7.543676    7.1793923   8.176432
   7.050739    7.051059    8.160357    5.976437    8.107099    8.287453
   8.011904    7.8676453   5.6207857   8.460551    7.6661596   6.286874
   8.162859    7.3832164   7.733113    6.1902704   6.897247    8.641914
   5.343126    7.1006107   7.87791     7.1275043   5.8234687   8.024365
   7.139397    6.9960737   6.6531324   8.544418    8.649639    8.256785
   7.536168    6.633919    6.664524    7.056944    8.768497    8.023058
   7.2277656   5.8394103   7.731363    5.7303653   7.3245473   7.332905
  -2.4369154  -8.280565    7.0276012 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 20:12:50.834146
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4649
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 20:12:50.839341
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9131.69
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 20:12:50.842970
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.3141
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 20:12:50.846573
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -816.807
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140025063203224
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140024121511328
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140024121511832
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140022549815760
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140022549816264
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140022549816768

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5a3d520e80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.563091
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.518321
grad_step = 000002, loss = 0.483470
grad_step = 000003, loss = 0.446114
grad_step = 000004, loss = 0.404743
grad_step = 000005, loss = 0.363107
grad_step = 000006, loss = 0.330939
grad_step = 000007, loss = 0.308472
grad_step = 000008, loss = 0.294906
grad_step = 000009, loss = 0.270846
grad_step = 000010, loss = 0.247003
grad_step = 000011, loss = 0.229602
grad_step = 000012, loss = 0.218014
grad_step = 000013, loss = 0.208795
grad_step = 000014, loss = 0.198628
grad_step = 000015, loss = 0.185721
grad_step = 000016, loss = 0.171070
grad_step = 000017, loss = 0.157733
grad_step = 000018, loss = 0.147777
grad_step = 000019, loss = 0.139218
grad_step = 000020, loss = 0.129792
grad_step = 000021, loss = 0.119961
grad_step = 000022, loss = 0.110681
grad_step = 000023, loss = 0.102323
grad_step = 000024, loss = 0.094461
grad_step = 000025, loss = 0.086444
grad_step = 000026, loss = 0.078481
grad_step = 000027, loss = 0.071434
grad_step = 000028, loss = 0.065568
grad_step = 000029, loss = 0.060402
grad_step = 000030, loss = 0.055393
grad_step = 000031, loss = 0.050037
grad_step = 000032, loss = 0.044833
grad_step = 000033, loss = 0.040683
grad_step = 000034, loss = 0.037244
grad_step = 000035, loss = 0.033742
grad_step = 000036, loss = 0.030116
grad_step = 000037, loss = 0.026693
grad_step = 000038, loss = 0.023902
grad_step = 000039, loss = 0.021757
grad_step = 000040, loss = 0.019648
grad_step = 000041, loss = 0.017402
grad_step = 000042, loss = 0.015369
grad_step = 000043, loss = 0.013735
grad_step = 000044, loss = 0.012460
grad_step = 000045, loss = 0.011246
grad_step = 000046, loss = 0.009993
grad_step = 000047, loss = 0.008872
grad_step = 000048, loss = 0.007942
grad_step = 000049, loss = 0.007174
grad_step = 000050, loss = 0.006444
grad_step = 000051, loss = 0.005772
grad_step = 000052, loss = 0.005278
grad_step = 000053, loss = 0.004884
grad_step = 000054, loss = 0.004517
grad_step = 000055, loss = 0.004157
grad_step = 000056, loss = 0.003864
grad_step = 000057, loss = 0.003685
grad_step = 000058, loss = 0.003527
grad_step = 000059, loss = 0.003350
grad_step = 000060, loss = 0.003163
grad_step = 000061, loss = 0.003037
grad_step = 000062, loss = 0.002975
grad_step = 000063, loss = 0.002924
grad_step = 000064, loss = 0.002864
grad_step = 000065, loss = 0.002800
grad_step = 000066, loss = 0.002764
grad_step = 000067, loss = 0.002734
grad_step = 000068, loss = 0.002700
grad_step = 000069, loss = 0.002659
grad_step = 000070, loss = 0.002629
grad_step = 000071, loss = 0.002604
grad_step = 000072, loss = 0.002571
grad_step = 000073, loss = 0.002540
grad_step = 000074, loss = 0.002517
grad_step = 000075, loss = 0.002501
grad_step = 000076, loss = 0.002474
grad_step = 000077, loss = 0.002443
grad_step = 000078, loss = 0.002417
grad_step = 000079, loss = 0.002398
grad_step = 000080, loss = 0.002374
grad_step = 000081, loss = 0.002347
grad_step = 000082, loss = 0.002325
grad_step = 000083, loss = 0.002309
grad_step = 000084, loss = 0.002292
grad_step = 000085, loss = 0.002273
grad_step = 000086, loss = 0.002257
grad_step = 000087, loss = 0.002245
grad_step = 000088, loss = 0.002233
grad_step = 000089, loss = 0.002221
grad_step = 000090, loss = 0.002210
grad_step = 000091, loss = 0.002202
grad_step = 000092, loss = 0.002193
grad_step = 000093, loss = 0.002186
grad_step = 000094, loss = 0.002179
grad_step = 000095, loss = 0.002174
grad_step = 000096, loss = 0.002168
grad_step = 000097, loss = 0.002162
grad_step = 000098, loss = 0.002156
grad_step = 000099, loss = 0.002152
grad_step = 000100, loss = 0.002148
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002142
grad_step = 000102, loss = 0.002137
grad_step = 000103, loss = 0.002132
grad_step = 000104, loss = 0.002128
grad_step = 000105, loss = 0.002122
grad_step = 000106, loss = 0.002117
grad_step = 000107, loss = 0.002113
grad_step = 000108, loss = 0.002108
grad_step = 000109, loss = 0.002103
grad_step = 000110, loss = 0.002098
grad_step = 000111, loss = 0.002092
grad_step = 000112, loss = 0.002087
grad_step = 000113, loss = 0.002082
grad_step = 000114, loss = 0.002077
grad_step = 000115, loss = 0.002072
grad_step = 000116, loss = 0.002066
grad_step = 000117, loss = 0.002061
grad_step = 000118, loss = 0.002056
grad_step = 000119, loss = 0.002051
grad_step = 000120, loss = 0.002045
grad_step = 000121, loss = 0.002040
grad_step = 000122, loss = 0.002034
grad_step = 000123, loss = 0.002029
grad_step = 000124, loss = 0.002024
grad_step = 000125, loss = 0.002018
grad_step = 000126, loss = 0.002013
grad_step = 000127, loss = 0.002007
grad_step = 000128, loss = 0.002002
grad_step = 000129, loss = 0.001998
grad_step = 000130, loss = 0.001997
grad_step = 000131, loss = 0.001997
grad_step = 000132, loss = 0.001988
grad_step = 000133, loss = 0.001978
grad_step = 000134, loss = 0.001977
grad_step = 000135, loss = 0.001977
grad_step = 000136, loss = 0.001970
grad_step = 000137, loss = 0.001962
grad_step = 000138, loss = 0.001959
grad_step = 000139, loss = 0.001959
grad_step = 000140, loss = 0.001956
grad_step = 000141, loss = 0.001949
grad_step = 000142, loss = 0.001943
grad_step = 000143, loss = 0.001941
grad_step = 000144, loss = 0.001940
grad_step = 000145, loss = 0.001939
grad_step = 000146, loss = 0.001937
grad_step = 000147, loss = 0.001932
grad_step = 000148, loss = 0.001927
grad_step = 000149, loss = 0.001922
grad_step = 000150, loss = 0.001918
grad_step = 000151, loss = 0.001916
grad_step = 000152, loss = 0.001914
grad_step = 000153, loss = 0.001914
grad_step = 000154, loss = 0.001916
grad_step = 000155, loss = 0.001923
grad_step = 000156, loss = 0.001936
grad_step = 000157, loss = 0.001948
grad_step = 000158, loss = 0.001948
grad_step = 000159, loss = 0.001921
grad_step = 000160, loss = 0.001896
grad_step = 000161, loss = 0.001894
grad_step = 000162, loss = 0.001909
grad_step = 000163, loss = 0.001915
grad_step = 000164, loss = 0.001901
grad_step = 000165, loss = 0.001884
grad_step = 000166, loss = 0.001879
grad_step = 000167, loss = 0.001886
grad_step = 000168, loss = 0.001894
grad_step = 000169, loss = 0.001889
grad_step = 000170, loss = 0.001877
grad_step = 000171, loss = 0.001866
grad_step = 000172, loss = 0.001863
grad_step = 000173, loss = 0.001867
grad_step = 000174, loss = 0.001870
grad_step = 000175, loss = 0.001869
grad_step = 000176, loss = 0.001863
grad_step = 000177, loss = 0.001854
grad_step = 000178, loss = 0.001847
grad_step = 000179, loss = 0.001844
grad_step = 000180, loss = 0.001843
grad_step = 000181, loss = 0.001843
grad_step = 000182, loss = 0.001844
grad_step = 000183, loss = 0.001845
grad_step = 000184, loss = 0.001845
grad_step = 000185, loss = 0.001845
grad_step = 000186, loss = 0.001846
grad_step = 000187, loss = 0.001845
grad_step = 000188, loss = 0.001844
grad_step = 000189, loss = 0.001841
grad_step = 000190, loss = 0.001837
grad_step = 000191, loss = 0.001832
grad_step = 000192, loss = 0.001826
grad_step = 000193, loss = 0.001819
grad_step = 000194, loss = 0.001812
grad_step = 000195, loss = 0.001807
grad_step = 000196, loss = 0.001801
grad_step = 000197, loss = 0.001797
grad_step = 000198, loss = 0.001794
grad_step = 000199, loss = 0.001792
grad_step = 000200, loss = 0.001790
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001791
grad_step = 000202, loss = 0.001796
grad_step = 000203, loss = 0.001809
grad_step = 000204, loss = 0.001840
grad_step = 000205, loss = 0.001899
grad_step = 000206, loss = 0.001986
grad_step = 000207, loss = 0.002046
grad_step = 000208, loss = 0.001990
grad_step = 000209, loss = 0.001841
grad_step = 000210, loss = 0.001766
grad_step = 000211, loss = 0.001835
grad_step = 000212, loss = 0.001908
grad_step = 000213, loss = 0.001864
grad_step = 000214, loss = 0.001777
grad_step = 000215, loss = 0.001759
grad_step = 000216, loss = 0.001816
grad_step = 000217, loss = 0.001851
grad_step = 000218, loss = 0.001807
grad_step = 000219, loss = 0.001752
grad_step = 000220, loss = 0.001754
grad_step = 000221, loss = 0.001791
grad_step = 000222, loss = 0.001800
grad_step = 000223, loss = 0.001766
grad_step = 000224, loss = 0.001737
grad_step = 000225, loss = 0.001746
grad_step = 000226, loss = 0.001768
grad_step = 000227, loss = 0.001766
grad_step = 000228, loss = 0.001741
grad_step = 000229, loss = 0.001727
grad_step = 000230, loss = 0.001736
grad_step = 000231, loss = 0.001747
grad_step = 000232, loss = 0.001743
grad_step = 000233, loss = 0.001727
grad_step = 000234, loss = 0.001718
grad_step = 000235, loss = 0.001721
grad_step = 000236, loss = 0.001728
grad_step = 000237, loss = 0.001729
grad_step = 000238, loss = 0.001722
grad_step = 000239, loss = 0.001712
grad_step = 000240, loss = 0.001707
grad_step = 000241, loss = 0.001708
grad_step = 000242, loss = 0.001711
grad_step = 000243, loss = 0.001712
grad_step = 000244, loss = 0.001709
grad_step = 000245, loss = 0.001704
grad_step = 000246, loss = 0.001698
grad_step = 000247, loss = 0.001694
grad_step = 000248, loss = 0.001692
grad_step = 000249, loss = 0.001692
grad_step = 000250, loss = 0.001693
grad_step = 000251, loss = 0.001693
grad_step = 000252, loss = 0.001692
grad_step = 000253, loss = 0.001691
grad_step = 000254, loss = 0.001689
grad_step = 000255, loss = 0.001687
grad_step = 000256, loss = 0.001685
grad_step = 000257, loss = 0.001683
grad_step = 000258, loss = 0.001681
grad_step = 000259, loss = 0.001680
grad_step = 000260, loss = 0.001679
grad_step = 000261, loss = 0.001680
grad_step = 000262, loss = 0.001682
grad_step = 000263, loss = 0.001688
grad_step = 000264, loss = 0.001698
grad_step = 000265, loss = 0.001718
grad_step = 000266, loss = 0.001745
grad_step = 000267, loss = 0.001789
grad_step = 000268, loss = 0.001830
grad_step = 000269, loss = 0.001860
grad_step = 000270, loss = 0.001836
grad_step = 000271, loss = 0.001765
grad_step = 000272, loss = 0.001685
grad_step = 000273, loss = 0.001647
grad_step = 000274, loss = 0.001665
grad_step = 000275, loss = 0.001712
grad_step = 000276, loss = 0.001757
grad_step = 000277, loss = 0.001775
grad_step = 000278, loss = 0.001770
grad_step = 000279, loss = 0.001707
grad_step = 000280, loss = 0.001647
grad_step = 000281, loss = 0.001639
grad_step = 000282, loss = 0.001670
grad_step = 000283, loss = 0.001692
grad_step = 000284, loss = 0.001686
grad_step = 000285, loss = 0.001669
grad_step = 000286, loss = 0.001648
grad_step = 000287, loss = 0.001626
grad_step = 000288, loss = 0.001625
grad_step = 000289, loss = 0.001645
grad_step = 000290, loss = 0.001659
grad_step = 000291, loss = 0.001659
grad_step = 000292, loss = 0.001651
grad_step = 000293, loss = 0.001644
grad_step = 000294, loss = 0.001632
grad_step = 000295, loss = 0.001614
grad_step = 000296, loss = 0.001604
grad_step = 000297, loss = 0.001607
grad_step = 000298, loss = 0.001612
grad_step = 000299, loss = 0.001613
grad_step = 000300, loss = 0.001613
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001618
grad_step = 000302, loss = 0.001627
grad_step = 000303, loss = 0.001631
grad_step = 000304, loss = 0.001634
grad_step = 000305, loss = 0.001641
grad_step = 000306, loss = 0.001652
grad_step = 000307, loss = 0.001666
grad_step = 000308, loss = 0.001672
grad_step = 000309, loss = 0.001677
grad_step = 000310, loss = 0.001678
grad_step = 000311, loss = 0.001672
grad_step = 000312, loss = 0.001654
grad_step = 000313, loss = 0.001629
grad_step = 000314, loss = 0.001605
grad_step = 000315, loss = 0.001586
grad_step = 000316, loss = 0.001572
grad_step = 000317, loss = 0.001565
grad_step = 000318, loss = 0.001566
grad_step = 000319, loss = 0.001572
grad_step = 000320, loss = 0.001579
grad_step = 000321, loss = 0.001588
grad_step = 000322, loss = 0.001603
grad_step = 000323, loss = 0.001623
grad_step = 000324, loss = 0.001641
grad_step = 000325, loss = 0.001653
grad_step = 000326, loss = 0.001653
grad_step = 000327, loss = 0.001639
grad_step = 000328, loss = 0.001616
grad_step = 000329, loss = 0.001586
grad_step = 000330, loss = 0.001561
grad_step = 000331, loss = 0.001547
grad_step = 000332, loss = 0.001542
grad_step = 000333, loss = 0.001545
grad_step = 000334, loss = 0.001554
grad_step = 000335, loss = 0.001570
grad_step = 000336, loss = 0.001590
grad_step = 000337, loss = 0.001616
grad_step = 000338, loss = 0.001646
grad_step = 000339, loss = 0.001682
grad_step = 000340, loss = 0.001700
grad_step = 000341, loss = 0.001699
grad_step = 000342, loss = 0.001659
grad_step = 000343, loss = 0.001601
grad_step = 000344, loss = 0.001547
grad_step = 000345, loss = 0.001524
grad_step = 000346, loss = 0.001537
grad_step = 000347, loss = 0.001567
grad_step = 000348, loss = 0.001589
grad_step = 000349, loss = 0.001593
grad_step = 000350, loss = 0.001586
grad_step = 000351, loss = 0.001567
grad_step = 000352, loss = 0.001545
grad_step = 000353, loss = 0.001521
grad_step = 000354, loss = 0.001510
grad_step = 000355, loss = 0.001512
grad_step = 000356, loss = 0.001518
grad_step = 000357, loss = 0.001520
grad_step = 000358, loss = 0.001523
grad_step = 000359, loss = 0.001528
grad_step = 000360, loss = 0.001530
grad_step = 000361, loss = 0.001525
grad_step = 000362, loss = 0.001516
grad_step = 000363, loss = 0.001509
grad_step = 000364, loss = 0.001507
grad_step = 000365, loss = 0.001507
grad_step = 000366, loss = 0.001502
grad_step = 000367, loss = 0.001497
grad_step = 000368, loss = 0.001493
grad_step = 000369, loss = 0.001493
grad_step = 000370, loss = 0.001496
grad_step = 000371, loss = 0.001496
grad_step = 000372, loss = 0.001498
grad_step = 000373, loss = 0.001502
grad_step = 000374, loss = 0.001514
grad_step = 000375, loss = 0.001539
grad_step = 000376, loss = 0.001578
grad_step = 000377, loss = 0.001647
grad_step = 000378, loss = 0.001736
grad_step = 000379, loss = 0.001846
grad_step = 000380, loss = 0.001929
grad_step = 000381, loss = 0.001861
grad_step = 000382, loss = 0.001686
grad_step = 000383, loss = 0.001495
grad_step = 000384, loss = 0.001475
grad_step = 000385, loss = 0.001581
grad_step = 000386, loss = 0.001631
grad_step = 000387, loss = 0.001589
grad_step = 000388, loss = 0.001549
grad_step = 000389, loss = 0.001501
grad_step = 000390, loss = 0.001470
grad_step = 000391, loss = 0.001484
grad_step = 000392, loss = 0.001519
grad_step = 000393, loss = 0.001499
grad_step = 000394, loss = 0.001437
grad_step = 000395, loss = 0.001441
grad_step = 000396, loss = 0.001478
grad_step = 000397, loss = 0.001464
grad_step = 000398, loss = 0.001437
grad_step = 000399, loss = 0.001436
grad_step = 000400, loss = 0.001436
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001421
grad_step = 000402, loss = 0.001413
grad_step = 000403, loss = 0.001427
grad_step = 000404, loss = 0.001440
grad_step = 000405, loss = 0.001429
grad_step = 000406, loss = 0.001412
grad_step = 000407, loss = 0.001405
grad_step = 000408, loss = 0.001404
grad_step = 000409, loss = 0.001398
grad_step = 000410, loss = 0.001390
grad_step = 000411, loss = 0.001390
grad_step = 000412, loss = 0.001397
grad_step = 000413, loss = 0.001403
grad_step = 000414, loss = 0.001406
grad_step = 000415, loss = 0.001408
grad_step = 000416, loss = 0.001418
grad_step = 000417, loss = 0.001441
grad_step = 000418, loss = 0.001468
grad_step = 000419, loss = 0.001500
grad_step = 000420, loss = 0.001527
grad_step = 000421, loss = 0.001565
grad_step = 000422, loss = 0.001608
grad_step = 000423, loss = 0.001574
grad_step = 000424, loss = 0.001504
grad_step = 000425, loss = 0.001419
grad_step = 000426, loss = 0.001379
grad_step = 000427, loss = 0.001386
grad_step = 000428, loss = 0.001419
grad_step = 000429, loss = 0.001460
grad_step = 000430, loss = 0.001473
grad_step = 000431, loss = 0.001439
grad_step = 000432, loss = 0.001396
grad_step = 000433, loss = 0.001372
grad_step = 000434, loss = 0.001364
grad_step = 000435, loss = 0.001359
grad_step = 000436, loss = 0.001357
grad_step = 000437, loss = 0.001367
grad_step = 000438, loss = 0.001383
grad_step = 000439, loss = 0.001381
grad_step = 000440, loss = 0.001369
grad_step = 000441, loss = 0.001355
grad_step = 000442, loss = 0.001348
grad_step = 000443, loss = 0.001345
grad_step = 000444, loss = 0.001340
grad_step = 000445, loss = 0.001333
grad_step = 000446, loss = 0.001331
grad_step = 000447, loss = 0.001334
grad_step = 000448, loss = 0.001340
grad_step = 000449, loss = 0.001344
grad_step = 000450, loss = 0.001348
grad_step = 000451, loss = 0.001352
grad_step = 000452, loss = 0.001361
grad_step = 000453, loss = 0.001375
grad_step = 000454, loss = 0.001383
grad_step = 000455, loss = 0.001397
grad_step = 000456, loss = 0.001404
grad_step = 000457, loss = 0.001414
grad_step = 000458, loss = 0.001427
grad_step = 000459, loss = 0.001421
grad_step = 000460, loss = 0.001411
grad_step = 000461, loss = 0.001383
grad_step = 000462, loss = 0.001359
grad_step = 000463, loss = 0.001346
grad_step = 000464, loss = 0.001328
grad_step = 000465, loss = 0.001313
grad_step = 000466, loss = 0.001301
grad_step = 000467, loss = 0.001300
grad_step = 000468, loss = 0.001308
grad_step = 000469, loss = 0.001318
grad_step = 000470, loss = 0.001330
grad_step = 000471, loss = 0.001334
grad_step = 000472, loss = 0.001343
grad_step = 000473, loss = 0.001354
grad_step = 000474, loss = 0.001373
grad_step = 000475, loss = 0.001396
grad_step = 000476, loss = 0.001407
grad_step = 000477, loss = 0.001409
grad_step = 000478, loss = 0.001396
grad_step = 000479, loss = 0.001363
grad_step = 000480, loss = 0.001331
grad_step = 000481, loss = 0.001296
grad_step = 000482, loss = 0.001278
grad_step = 000483, loss = 0.001282
grad_step = 000484, loss = 0.001300
grad_step = 000485, loss = 0.001329
grad_step = 000486, loss = 0.001350
grad_step = 000487, loss = 0.001377
grad_step = 000488, loss = 0.001382
grad_step = 000489, loss = 0.001382
grad_step = 000490, loss = 0.001373
grad_step = 000491, loss = 0.001347
grad_step = 000492, loss = 0.001316
grad_step = 000493, loss = 0.001285
grad_step = 000494, loss = 0.001265
grad_step = 000495, loss = 0.001260
grad_step = 000496, loss = 0.001266
grad_step = 000497, loss = 0.001275
grad_step = 000498, loss = 0.001284
grad_step = 000499, loss = 0.001291
grad_step = 000500, loss = 0.001304
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001310
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

  date_run                              2020-05-11 20:13:11.511300
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.231858
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 20:13:11.517489
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138872
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 20:13:11.526038
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.134964
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 20:13:11.532686
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.1102
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
0   2020-05-11 20:12:40.567668  ...    mean_absolute_error
1   2020-05-11 20:12:40.571912  ...     mean_squared_error
2   2020-05-11 20:12:40.575506  ...  median_absolute_error
3   2020-05-11 20:12:40.579017  ...               r2_score
4   2020-05-11 20:12:50.834146  ...    mean_absolute_error
5   2020-05-11 20:12:50.839341  ...     mean_squared_error
6   2020-05-11 20:12:50.842970  ...  median_absolute_error
7   2020-05-11 20:12:50.846573  ...               r2_score
8   2020-05-11 20:13:11.511300  ...    mean_absolute_error
9   2020-05-11 20:13:11.517489  ...     mean_squared_error
10  2020-05-11 20:13:11.526038  ...  median_absolute_error
11  2020-05-11 20:13:11.532686  ...               r2_score

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
0it [00:00, ?it/s]  1%|          | 81920/9912422 [00:00<00:12, 818976.23it/s] 95%|█████████▍| 9412608/9912422 [00:00<00:00, 1165483.60it/s]9920512it [00:00, 48506885.76it/s]                            
0it [00:00, ?it/s]32768it [00:00, 609704.39it/s]
0it [00:00, ?it/s]  7%|▋         | 114688/1648877 [00:00<00:01, 1146468.35it/s]1654784it [00:00, 13094140.75it/s]                           
0it [00:00, ?it/s]8192it [00:00, 229946.38it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54ef418fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f548cb34ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54ef3a3ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f548c60c0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54ef418fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54a1d9de80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54ef418fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f549624f748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54ef3e0ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54a1d9de80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f548cb34fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe3662a5208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=917e1e6e795cbdecb561da3ac0df6eeb55f350ae1fa0ae98eb78ad3c7915872c
  Stored in directory: /tmp/pip-ephem-wheel-cache-o4da8v1a/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe2fef80ef0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1851392/17464789 [==>...........................] - ETA: 0s
 6684672/17464789 [==========>...................] - ETA: 0s
11173888/17464789 [==================>...........] - ETA: 0s
16171008/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 20:14:39.115167: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 20:14:39.119267: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-11 20:14:39.119400: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5599bd317ab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 20:14:39.119414: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4136 - accuracy: 0.5165 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6762 - accuracy: 0.4994
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 3s - loss: 7.6429 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 3s - loss: 7.6526 - accuracy: 0.5009
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6607 - accuracy: 0.5004
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
15000/25000 [=================>............] - ETA: 2s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
25000/25000 [==============================] - 8s 300us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 20:14:53.719158
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 20:14:53.719158  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 20:15:00.179522: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 20:15:00.184662: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-11 20:15:00.184802: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5608c6407e70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 20:15:00.184815: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f358dd74dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3203 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.2076 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f35840168d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8890 - accuracy: 0.4855 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.9222 - accuracy: 0.4833
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.9196 - accuracy: 0.4835
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8108 - accuracy: 0.4906
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8072 - accuracy: 0.4908
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7805 - accuracy: 0.4926
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6888 - accuracy: 0.4986
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6728 - accuracy: 0.4996
11000/25000 [============>.................] - ETA: 3s - loss: 7.6861 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 3s - loss: 7.7177 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7315 - accuracy: 0.4958
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7400 - accuracy: 0.4952
15000/25000 [=================>............] - ETA: 2s - loss: 7.7341 - accuracy: 0.4956
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7385 - accuracy: 0.4953
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7135 - accuracy: 0.4969
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7135 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7134 - accuracy: 0.4969
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7096 - accuracy: 0.4972
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7082 - accuracy: 0.4973
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 8s 302us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f35800f05f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:27:36, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:39:22, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:42:47, 20.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:12:25, 29.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:43:48, 41.6kB/s].vector_cache/glove.6B.zip:   1%|          | 8.84M/862M [00:01<3:59:17, 59.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:46:48, 84.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:01<1:56:24, 121kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:21:06, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:01<56:35, 246kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:01<39:28, 351kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:02<27:36, 500kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<19:18, 711kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.8M/862M [00:02<13:34, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:02<09:31, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:03<06:39, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:33, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:20, 2.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:02, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<05:43, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<04:20, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<06:51, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<05:28, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<05:57, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<05:31, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<04:12, 3.14MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<06:01, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<06:55, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<05:31, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:58, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<05:32, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<04:10, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<05:55, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<06:48, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<05:25, 2.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<05:53, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<05:28, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<04:06, 3.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:26, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:21<04:07, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<05:55, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:26, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<04:08, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:53, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<06:44, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:25<05:17, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:50, 3.30MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:38, 1.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<10:10, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<07:26, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:08, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:16, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:26, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:37, 2.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:33:45, 134kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<1:06:53, 187kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<47:01, 266kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<35:44, 349kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<27:35, 452kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<19:49, 628kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<14:01, 885kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<15:04, 822kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<11:51, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<08:32, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:51, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:14, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:28, 2.24MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<03:59, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:09:03, 177kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<50:50, 241kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<36:07, 339kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<25:21, 481kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<24:38, 495kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<18:29, 659kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<13:13, 919kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:04, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:58, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:17, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:43, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:40, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:58, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:16, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:50, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:58:21, 101kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:24:01, 142kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<59:00, 201kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<43:57, 269kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<33:11, 357kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<23:49, 496kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<16:45, 703kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:35:41, 123kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:08:09, 173kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<47:54, 245kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<36:10, 324kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<26:40, 439kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<18:56, 617kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<15:41, 742kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<13:42, 848kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<10:11, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<07:15, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<10:45, 1.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:50, 1.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<06:27, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:58, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:13, 1.85MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:38, 2.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:41, 2.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<06:32, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:14, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:48, 2.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:50, 821kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:59, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:00, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:59, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:14, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:34, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:02, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:36, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:38, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:19, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:54, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:24, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:57, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:22, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:46, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:18, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<05:20, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:18, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:57, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:39, 3.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:06, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:31, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:10, 2.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:55, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<03:42, 2.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:54, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:58, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:42, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:31, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:22, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:59, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<03:44, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:54, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:35, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:23, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:57, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:23, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:04, 2.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:05, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:27, 3.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:02, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:07, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:32, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:22, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:11, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:32, 2.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:21, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:38, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:14, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:08, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:46, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:37, 2.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:42, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:27, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:24, 2.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:32, 2.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:19, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:18, 3.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:28, 2.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:16, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:14, 3.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:24, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:25, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:16, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:08, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:26, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:56, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:41, 2.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:41, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:28, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:19, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:11, 3.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:18, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:51, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:38, 2.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:37, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:30, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:21, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:10, 3.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:23, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:00, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:52, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:41, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:27, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:13, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:52, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:36, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:53, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:26, 1.73MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [01:59<04:52, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:38, 2.58MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:31, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:21, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:17, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:05, 3.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<10:42, 867kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<08:32, 1.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<06:13, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:21, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:36, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:08, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:42, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<10:56, 836kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:41, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<06:18, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:19, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:23, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:58, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:47, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:29, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:17, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:05, 2.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<09:54, 902kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:56, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<05:48, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:56, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:04, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:39, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:23, 2.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:55, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:26, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:01, 2.87MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:51, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:47, 3.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:06, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:30, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:20, 2.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:13, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:51, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:48, 3.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:37, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:11, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:08, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:58, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:39, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:37, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:40, 3.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:45, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:16, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:13, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:00, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:45, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:42, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:46, 2.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:59, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:43, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:50, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:42, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:30, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:32, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:35, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:56, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:20, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:14, 2.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:02, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:44, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:50, 2.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:20, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:24, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:29, 3.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:40, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:09, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:07, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:50, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:30, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:31, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:36, 2.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:04, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:35, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:43, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:02, 3.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:59, 953kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:24, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:39, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:56, 1.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:05, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:57, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<02:50, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<17:16, 433kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<12:57, 576kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<09:15, 804kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:01, 922kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<06:19, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:36, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<03:19, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:05, 907kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:13, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:23, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:51, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<06:01, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:55, 1.47MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:39, 1.98MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:38, 2.72MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:29, 959kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:04, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:26, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:46, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:45, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:30, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:28, 2.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:13, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:45, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:56, 2.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:13, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:11, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:02, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:17, 3.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<01:41, 4.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<12:17, 557kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<10:09, 674kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:27, 917kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<05:18, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:52, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:54, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:35, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:56, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:20, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:22, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:26, 2.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:13, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:22, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:12, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:42, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:09, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:17, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:22, 2.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<08:02, 808kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:37, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:35, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:38, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:33, 2.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:53, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:09, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:04, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:29, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:51, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:58, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<02:10, 2.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:44, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:16, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:27, 2.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:05, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:16, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:37, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<01:56, 3.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:14, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:15, 2.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:51, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:39, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:00, 2.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:43, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:14, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:33, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:52, 3.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:18, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:59, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:14, 2.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:48, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:17, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:35, 2.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:55, 3.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:56, 1.95MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:37, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<01:58, 2.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:27, 3.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:24, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<04:24, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:13, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:29, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:41, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:51, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:05, 2.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:02, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:03, 2.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:36, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:03, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:23, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:44, 3.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<04:04, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:25, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:31, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:19, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:35, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:53, 2.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:57, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:40, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:59, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:29, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:55, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:17, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:39, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:35, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:03, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:15, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:42, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:04, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<02:23, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:43, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:30, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:41, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:41, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:58, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:10, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:27, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:46, 2.73MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:42, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:07, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:18, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:40, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:17, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:42, 2.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:15, 3.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<06:31, 720kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<05:34, 842kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:08, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:55, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<15:58, 290kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<11:40, 396kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:15, 557kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<06:44, 676kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:41, 800kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:10, 1.09MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:57, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:19, 1.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:31, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:33, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:45, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:13, 1.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:35, 2.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:32, 1.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:56, 1.47MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:10, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:27, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:43, 1.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:06, 2.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:26<01:33, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:12, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:01, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:31, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:48, 2.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:21, 3.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:50, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:46, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:21, 2.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:47, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:42, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:17, 3.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<00:57, 4.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:37, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:58, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:10, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:35, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:52, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:28, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:50, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:03, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:19, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:58, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:27, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:41, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:15, 2.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:38, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:33, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:11, 2.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:34, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:54, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:29, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:05, 3.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:02, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:48, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:21, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:41, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:58, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:32, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:06, 3.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:00, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:48, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:56, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:44, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:17, 2.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:28, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:06, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:19, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<00:59, 3.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:38, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<00:57, 3.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:32, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:03, 2.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<00:59, 2.90MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:17, 2.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<00:55, 3.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:13, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:10, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:53, 3.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:11, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:24, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:06, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:49, 3.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:19, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:54, 2.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:10, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:25, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:08, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:48, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:00, 825kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:22, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:42, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:42, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:44, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:19, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:57, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:30, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:19, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:58, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:09, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:20, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:03, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:45, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:41, 817kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:05, 1.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:31, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:04, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:24, 886kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:55, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:23, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:23, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:26, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:47, 2.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:23, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:11, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:52, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:00, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:08, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:53, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:38, 2.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:20, 553kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:32, 727kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:48, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:37, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:31, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:09, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:48, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:24, 1.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:10, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:56, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:00, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:47, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:33, 2.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:06, 308kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:44, 420kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:36, 590kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:07, 711kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:48, 834kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:19, 1.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:55, 1.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:12, 1.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:00, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:31, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:38, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:44, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:34, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:24, 3.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:37, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:27, 2.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:33, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:23, 2.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:29, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:28, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:19, 3.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:37, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:33, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:24, 2.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:32, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:17, 2.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:02, 849kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:49, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:34, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:33, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:34, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:18, 2.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:29, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:18, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:58, 624kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:43, 820kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:30, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:26, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:26, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:12, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:36, 768kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:28, 971kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:19, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:17, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:17, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 3.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:01, 3.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.71MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 869/400000 [00:00<00:45, 8677.90it/s]  0%|          | 1695/400000 [00:00<00:46, 8546.45it/s]  1%|          | 2579/400000 [00:00<00:46, 8631.49it/s]  1%|          | 3451/400000 [00:00<00:45, 8656.95it/s]  1%|          | 4315/400000 [00:00<00:45, 8650.67it/s]  1%|▏         | 5183/400000 [00:00<00:45, 8658.98it/s]  2%|▏         | 6011/400000 [00:00<00:46, 8539.00it/s]  2%|▏         | 6869/400000 [00:00<00:45, 8548.77it/s]  2%|▏         | 7715/400000 [00:00<00:46, 8520.64it/s]  2%|▏         | 8535/400000 [00:01<00:46, 8421.16it/s]  2%|▏         | 9408/400000 [00:01<00:45, 8510.96it/s]  3%|▎         | 10244/400000 [00:01<00:46, 8464.94it/s]  3%|▎         | 11099/400000 [00:01<00:45, 8490.27it/s]  3%|▎         | 11977/400000 [00:01<00:45, 8574.26it/s]  3%|▎         | 12844/400000 [00:01<00:45, 8600.61it/s]  3%|▎         | 13724/400000 [00:01<00:44, 8657.24it/s]  4%|▎         | 14590/400000 [00:01<00:44, 8657.87it/s]  4%|▍         | 15468/400000 [00:01<00:44, 8692.73it/s]  4%|▍         | 16343/400000 [00:01<00:44, 8709.58it/s]  4%|▍         | 17214/400000 [00:02<00:43, 8708.63it/s]  5%|▍         | 18085/400000 [00:02<00:43, 8697.63it/s]  5%|▍         | 18955/400000 [00:02<00:43, 8688.33it/s]  5%|▍         | 19831/400000 [00:02<00:43, 8708.77it/s]  5%|▌         | 20702/400000 [00:02<00:43, 8677.51it/s]  5%|▌         | 21570/400000 [00:02<00:44, 8518.55it/s]  6%|▌         | 22448/400000 [00:02<00:43, 8593.92it/s]  6%|▌         | 23316/400000 [00:02<00:43, 8617.63it/s]  6%|▌         | 24181/400000 [00:02<00:43, 8625.41it/s]  6%|▋         | 25044/400000 [00:02<00:43, 8588.75it/s]  6%|▋         | 25916/400000 [00:03<00:43, 8625.28it/s]  7%|▋         | 26780/400000 [00:03<00:43, 8627.35it/s]  7%|▋         | 27659/400000 [00:03<00:42, 8673.22it/s]  7%|▋         | 28528/400000 [00:03<00:42, 8677.46it/s]  7%|▋         | 29396/400000 [00:03<00:43, 8555.14it/s]  8%|▊         | 30252/400000 [00:03<00:43, 8528.67it/s]  8%|▊         | 31106/400000 [00:03<00:44, 8307.71it/s]  8%|▊         | 31962/400000 [00:03<00:43, 8374.68it/s]  8%|▊         | 32801/400000 [00:03<00:44, 8306.56it/s]  8%|▊         | 33644/400000 [00:03<00:43, 8341.43it/s]  9%|▊         | 34479/400000 [00:04<00:43, 8335.99it/s]  9%|▉         | 35335/400000 [00:04<00:43, 8399.49it/s]  9%|▉         | 36176/400000 [00:04<00:43, 8369.86it/s]  9%|▉         | 37041/400000 [00:04<00:42, 8450.59it/s]  9%|▉         | 37897/400000 [00:04<00:42, 8481.34it/s] 10%|▉         | 38762/400000 [00:04<00:42, 8531.09it/s] 10%|▉         | 39616/400000 [00:04<00:42, 8531.29it/s] 10%|█         | 40476/400000 [00:04<00:42, 8549.69it/s] 10%|█         | 41332/400000 [00:04<00:42, 8509.63it/s] 11%|█         | 42184/400000 [00:04<00:42, 8403.14it/s] 11%|█         | 43025/400000 [00:05<00:42, 8312.57it/s] 11%|█         | 43889/400000 [00:05<00:42, 8405.86it/s] 11%|█         | 44740/400000 [00:05<00:42, 8435.72it/s] 11%|█▏        | 45585/400000 [00:05<00:42, 8374.54it/s] 12%|█▏        | 46423/400000 [00:05<00:42, 8370.44it/s] 12%|█▏        | 47269/400000 [00:05<00:42, 8396.47it/s] 12%|█▏        | 48115/400000 [00:05<00:41, 8415.12it/s] 12%|█▏        | 48981/400000 [00:05<00:41, 8485.07it/s] 12%|█▏        | 49855/400000 [00:05<00:40, 8558.40it/s] 13%|█▎        | 50712/400000 [00:05<00:41, 8416.65it/s] 13%|█▎        | 51562/400000 [00:06<00:41, 8440.42it/s] 13%|█▎        | 52407/400000 [00:06<00:41, 8386.37it/s] 13%|█▎        | 53270/400000 [00:06<00:40, 8457.94it/s] 14%|█▎        | 54141/400000 [00:06<00:40, 8530.10it/s] 14%|█▍        | 55016/400000 [00:06<00:40, 8594.12it/s] 14%|█▍        | 55886/400000 [00:06<00:39, 8624.61it/s] 14%|█▍        | 56749/400000 [00:06<00:40, 8578.03it/s] 14%|█▍        | 57629/400000 [00:06<00:39, 8642.07it/s] 15%|█▍        | 58510/400000 [00:06<00:39, 8690.35it/s] 15%|█▍        | 59380/400000 [00:06<00:39, 8678.56it/s] 15%|█▌        | 60249/400000 [00:07<00:39, 8663.15it/s] 15%|█▌        | 61116/400000 [00:07<00:40, 8454.47it/s] 15%|█▌        | 61963/400000 [00:07<00:40, 8379.17it/s] 16%|█▌        | 62847/400000 [00:07<00:39, 8510.73it/s] 16%|█▌        | 63726/400000 [00:07<00:39, 8591.91it/s] 16%|█▌        | 64587/400000 [00:07<00:39, 8510.70it/s] 16%|█▋        | 65456/400000 [00:07<00:39, 8560.91it/s] 17%|█▋        | 66328/400000 [00:07<00:38, 8607.53it/s] 17%|█▋        | 67203/400000 [00:07<00:38, 8649.42it/s] 17%|█▋        | 68070/400000 [00:07<00:38, 8654.47it/s] 17%|█▋        | 68953/400000 [00:08<00:38, 8703.94it/s] 17%|█▋        | 69824/400000 [00:08<00:38, 8658.50it/s] 18%|█▊        | 70695/400000 [00:08<00:37, 8670.75it/s] 18%|█▊        | 71563/400000 [00:08<00:39, 8395.02it/s] 18%|█▊        | 72405/400000 [00:08<00:39, 8355.91it/s] 18%|█▊        | 73265/400000 [00:08<00:38, 8426.24it/s] 19%|█▊        | 74132/400000 [00:08<00:38, 8496.07it/s] 19%|█▊        | 74999/400000 [00:08<00:38, 8546.70it/s] 19%|█▉        | 75855/400000 [00:08<00:38, 8374.38it/s] 19%|█▉        | 76717/400000 [00:08<00:38, 8443.80it/s] 19%|█▉        | 77563/400000 [00:09<00:38, 8413.78it/s] 20%|█▉        | 78406/400000 [00:09<00:38, 8321.06it/s] 20%|█▉        | 79280/400000 [00:09<00:37, 8440.89it/s] 20%|██        | 80158/400000 [00:09<00:37, 8538.65it/s] 20%|██        | 81045/400000 [00:09<00:36, 8633.24it/s] 20%|██        | 81910/400000 [00:09<00:36, 8626.60it/s] 21%|██        | 82787/400000 [00:09<00:36, 8666.78it/s] 21%|██        | 83664/400000 [00:09<00:36, 8697.01it/s] 21%|██        | 84546/400000 [00:09<00:36, 8731.48it/s] 21%|██▏       | 85432/400000 [00:09<00:35, 8767.98it/s] 22%|██▏       | 86310/400000 [00:10<00:35, 8716.83it/s] 22%|██▏       | 87189/400000 [00:10<00:35, 8735.72it/s] 22%|██▏       | 88073/400000 [00:10<00:35, 8764.27it/s] 22%|██▏       | 88950/400000 [00:10<00:35, 8753.53it/s] 22%|██▏       | 89838/400000 [00:10<00:35, 8790.18it/s] 23%|██▎       | 90723/400000 [00:10<00:35, 8805.98it/s] 23%|██▎       | 91604/400000 [00:10<00:35, 8780.60it/s] 23%|██▎       | 92483/400000 [00:10<00:35, 8716.01it/s] 23%|██▎       | 93355/400000 [00:10<00:35, 8684.81it/s] 24%|██▎       | 94236/400000 [00:11<00:35, 8720.27it/s] 24%|██▍       | 95112/400000 [00:11<00:34, 8729.77it/s] 24%|██▍       | 95986/400000 [00:11<00:34, 8691.03it/s] 24%|██▍       | 96856/400000 [00:11<00:34, 8687.95it/s] 24%|██▍       | 97725/400000 [00:11<00:34, 8674.51it/s] 25%|██▍       | 98593/400000 [00:11<00:34, 8617.84it/s] 25%|██▍       | 99459/400000 [00:11<00:34, 8628.34it/s] 25%|██▌       | 100325/400000 [00:11<00:34, 8636.67it/s] 25%|██▌       | 101202/400000 [00:11<00:34, 8675.27it/s] 26%|██▌       | 102077/400000 [00:11<00:34, 8695.98it/s] 26%|██▌       | 102948/400000 [00:12<00:34, 8698.18it/s] 26%|██▌       | 103818/400000 [00:12<00:34, 8651.28it/s] 26%|██▌       | 104684/400000 [00:12<00:34, 8556.04it/s] 26%|██▋       | 105549/400000 [00:12<00:34, 8583.60it/s] 27%|██▋       | 106411/400000 [00:12<00:34, 8594.48it/s] 27%|██▋       | 107288/400000 [00:12<00:33, 8645.83it/s] 27%|██▋       | 108153/400000 [00:12<00:33, 8608.05it/s] 27%|██▋       | 109014/400000 [00:12<00:33, 8587.83it/s] 27%|██▋       | 109873/400000 [00:12<00:34, 8451.37it/s] 28%|██▊       | 110757/400000 [00:12<00:33, 8564.20it/s] 28%|██▊       | 111619/400000 [00:13<00:33, 8578.09it/s] 28%|██▊       | 112488/400000 [00:13<00:33, 8609.21it/s] 28%|██▊       | 113350/400000 [00:13<00:33, 8600.95it/s] 29%|██▊       | 114215/400000 [00:13<00:33, 8614.58it/s] 29%|██▉       | 115077/400000 [00:13<00:33, 8611.25it/s] 29%|██▉       | 115939/400000 [00:13<00:33, 8607.54it/s] 29%|██▉       | 116800/400000 [00:13<00:32, 8603.06it/s] 29%|██▉       | 117661/400000 [00:13<00:33, 8528.43it/s] 30%|██▉       | 118515/400000 [00:13<00:33, 8501.83it/s] 30%|██▉       | 119395/400000 [00:13<00:32, 8588.44it/s] 30%|███       | 120282/400000 [00:14<00:32, 8669.60it/s] 30%|███       | 121150/400000 [00:14<00:32, 8656.76it/s] 31%|███       | 122016/400000 [00:14<00:32, 8635.06it/s] 31%|███       | 122903/400000 [00:14<00:31, 8702.79it/s] 31%|███       | 123774/400000 [00:14<00:31, 8688.00it/s] 31%|███       | 124651/400000 [00:14<00:31, 8710.47it/s] 31%|███▏      | 125523/400000 [00:14<00:32, 8525.48it/s] 32%|███▏      | 126377/400000 [00:14<00:32, 8486.84it/s] 32%|███▏      | 127227/400000 [00:14<00:32, 8460.19it/s] 32%|███▏      | 128098/400000 [00:14<00:31, 8532.40it/s] 32%|███▏      | 128952/400000 [00:15<00:31, 8510.80it/s] 32%|███▏      | 129804/400000 [00:15<00:31, 8509.79it/s] 33%|███▎      | 130656/400000 [00:15<00:32, 8337.92it/s] 33%|███▎      | 131518/400000 [00:15<00:31, 8420.48it/s] 33%|███▎      | 132397/400000 [00:15<00:31, 8526.09it/s] 33%|███▎      | 133272/400000 [00:15<00:31, 8589.96it/s] 34%|███▎      | 134156/400000 [00:15<00:30, 8660.95it/s] 34%|███▍      | 135023/400000 [00:15<00:30, 8638.44it/s] 34%|███▍      | 135897/400000 [00:15<00:30, 8667.79it/s] 34%|███▍      | 136765/400000 [00:15<00:30, 8658.37it/s] 34%|███▍      | 137633/400000 [00:16<00:30, 8664.23it/s] 35%|███▍      | 138508/400000 [00:16<00:30, 8688.23it/s] 35%|███▍      | 139377/400000 [00:16<00:30, 8521.59it/s] 35%|███▌      | 140250/400000 [00:16<00:30, 8581.00it/s] 35%|███▌      | 141109/400000 [00:16<00:30, 8464.88it/s] 35%|███▌      | 141957/400000 [00:16<00:30, 8396.68it/s] 36%|███▌      | 142823/400000 [00:16<00:30, 8473.47it/s] 36%|███▌      | 143672/400000 [00:16<00:30, 8412.34it/s] 36%|███▌      | 144514/400000 [00:16<00:30, 8365.79it/s] 36%|███▋      | 145352/400000 [00:16<00:30, 8331.28it/s] 37%|███▋      | 146204/400000 [00:17<00:30, 8386.13it/s] 37%|███▋      | 147043/400000 [00:17<00:30, 8304.76it/s] 37%|███▋      | 147874/400000 [00:17<00:30, 8153.24it/s] 37%|███▋      | 148749/400000 [00:17<00:30, 8321.16it/s] 37%|███▋      | 149617/400000 [00:17<00:29, 8425.49it/s] 38%|███▊      | 150461/400000 [00:17<00:29, 8412.83it/s] 38%|███▊      | 151341/400000 [00:17<00:29, 8524.05it/s] 38%|███▊      | 152208/400000 [00:17<00:28, 8565.72it/s] 38%|███▊      | 153073/400000 [00:17<00:28, 8590.63it/s] 38%|███▊      | 153948/400000 [00:17<00:28, 8635.09it/s] 39%|███▊      | 154825/400000 [00:18<00:28, 8672.87it/s] 39%|███▉      | 155701/400000 [00:18<00:28, 8696.79it/s] 39%|███▉      | 156571/400000 [00:18<00:28, 8654.66it/s] 39%|███▉      | 157438/400000 [00:18<00:28, 8657.77it/s] 40%|███▉      | 158304/400000 [00:18<00:28, 8562.18it/s] 40%|███▉      | 159173/400000 [00:18<00:28, 8599.22it/s] 40%|████      | 160038/400000 [00:18<00:27, 8612.09it/s] 40%|████      | 160900/400000 [00:18<00:27, 8601.85it/s] 40%|████      | 161761/400000 [00:18<00:27, 8524.40it/s] 41%|████      | 162614/400000 [00:18<00:28, 8439.33it/s] 41%|████      | 163461/400000 [00:19<00:28, 8447.78it/s] 41%|████      | 164307/400000 [00:19<00:27, 8434.01it/s] 41%|████▏     | 165151/400000 [00:19<00:28, 8364.43it/s] 42%|████▏     | 166022/400000 [00:19<00:27, 8462.53it/s] 42%|████▏     | 166892/400000 [00:19<00:27, 8532.06it/s] 42%|████▏     | 167761/400000 [00:19<00:27, 8577.76it/s] 42%|████▏     | 168620/400000 [00:19<00:27, 8529.93it/s] 42%|████▏     | 169476/400000 [00:19<00:27, 8537.79it/s] 43%|████▎     | 170331/400000 [00:19<00:26, 8531.53it/s] 43%|████▎     | 171185/400000 [00:20<00:27, 8254.95it/s] 43%|████▎     | 172060/400000 [00:20<00:27, 8395.89it/s] 43%|████▎     | 172942/400000 [00:20<00:26, 8517.87it/s] 43%|████▎     | 173796/400000 [00:20<00:26, 8519.82it/s] 44%|████▎     | 174650/400000 [00:20<00:26, 8483.49it/s] 44%|████▍     | 175529/400000 [00:20<00:26, 8571.64it/s] 44%|████▍     | 176399/400000 [00:20<00:25, 8608.46it/s] 44%|████▍     | 177261/400000 [00:20<00:25, 8586.85it/s] 45%|████▍     | 178129/400000 [00:20<00:25, 8611.96it/s] 45%|████▍     | 178991/400000 [00:20<00:25, 8570.51it/s] 45%|████▍     | 179879/400000 [00:21<00:25, 8658.07it/s] 45%|████▌     | 180752/400000 [00:21<00:25, 8678.88it/s] 45%|████▌     | 181621/400000 [00:21<00:25, 8679.60it/s] 46%|████▌     | 182490/400000 [00:21<00:25, 8654.49it/s] 46%|████▌     | 183366/400000 [00:21<00:24, 8684.94it/s] 46%|████▌     | 184235/400000 [00:21<00:24, 8678.31it/s] 46%|████▋     | 185108/400000 [00:21<00:24, 8692.50it/s] 46%|████▋     | 185978/400000 [00:21<00:24, 8646.37it/s] 47%|████▋     | 186849/400000 [00:21<00:24, 8665.05it/s] 47%|████▋     | 187717/400000 [00:21<00:24, 8667.89it/s] 47%|████▋     | 188594/400000 [00:22<00:24, 8696.69it/s] 47%|████▋     | 189464/400000 [00:22<00:24, 8653.37it/s] 48%|████▊     | 190330/400000 [00:22<00:24, 8644.23it/s] 48%|████▊     | 191214/400000 [00:22<00:24, 8697.63it/s] 48%|████▊     | 192093/400000 [00:22<00:23, 8724.45it/s] 48%|████▊     | 192976/400000 [00:22<00:23, 8753.62it/s] 48%|████▊     | 193852/400000 [00:22<00:24, 8540.85it/s] 49%|████▊     | 194709/400000 [00:22<00:24, 8547.10it/s] 49%|████▉     | 195576/400000 [00:22<00:23, 8581.16it/s] 49%|████▉     | 196448/400000 [00:22<00:23, 8622.20it/s] 49%|████▉     | 197329/400000 [00:23<00:23, 8675.55it/s] 50%|████▉     | 198197/400000 [00:23<00:23, 8596.13it/s] 50%|████▉     | 199062/400000 [00:23<00:23, 8610.52it/s] 50%|████▉     | 199933/400000 [00:23<00:23, 8640.08it/s] 50%|█████     | 200798/400000 [00:23<00:23, 8594.16it/s] 50%|█████     | 201658/400000 [00:23<00:23, 8527.56it/s] 51%|█████     | 202519/400000 [00:23<00:23, 8552.08it/s] 51%|█████     | 203382/400000 [00:23<00:22, 8572.56it/s] 51%|█████     | 204257/400000 [00:23<00:22, 8624.53it/s] 51%|█████▏    | 205120/400000 [00:23<00:22, 8552.07it/s] 51%|█████▏    | 205976/400000 [00:24<00:22, 8466.51it/s] 52%|█████▏    | 206824/400000 [00:24<00:22, 8445.46it/s] 52%|█████▏    | 207680/400000 [00:24<00:22, 8477.01it/s] 52%|█████▏    | 208551/400000 [00:24<00:22, 8544.23it/s] 52%|█████▏    | 209406/400000 [00:24<00:22, 8541.54it/s] 53%|█████▎    | 210261/400000 [00:24<00:22, 8463.33it/s] 53%|█████▎    | 211114/400000 [00:24<00:22, 8482.73it/s] 53%|█████▎    | 211963/400000 [00:24<00:22, 8461.19it/s] 53%|█████▎    | 212810/400000 [00:24<00:22, 8460.38it/s] 53%|█████▎    | 213657/400000 [00:24<00:22, 8392.60it/s] 54%|█████▎    | 214497/400000 [00:25<00:22, 8391.81it/s] 54%|█████▍    | 215366/400000 [00:25<00:21, 8477.29it/s] 54%|█████▍    | 216230/400000 [00:25<00:21, 8525.06it/s] 54%|█████▍    | 217091/400000 [00:25<00:21, 8547.86it/s] 54%|█████▍    | 217967/400000 [00:25<00:21, 8608.87it/s] 55%|█████▍    | 218829/400000 [00:25<00:21, 8596.14it/s] 55%|█████▍    | 219689/400000 [00:25<00:21, 8528.61it/s] 55%|█████▌    | 220553/400000 [00:25<00:20, 8557.45it/s] 55%|█████▌    | 221415/400000 [00:25<00:20, 8575.23it/s] 56%|█████▌    | 222285/400000 [00:25<00:20, 8612.03it/s] 56%|█████▌    | 223152/400000 [00:26<00:20, 8627.12it/s] 56%|█████▌    | 224015/400000 [00:26<00:20, 8516.33it/s] 56%|█████▌    | 224871/400000 [00:26<00:20, 8528.56it/s] 56%|█████▋    | 225741/400000 [00:26<00:20, 8577.81it/s] 57%|█████▋    | 226600/400000 [00:26<00:20, 8353.46it/s] 57%|█████▋    | 227461/400000 [00:26<00:20, 8428.77it/s] 57%|█████▋    | 228337/400000 [00:26<00:20, 8524.68it/s] 57%|█████▋    | 229207/400000 [00:26<00:19, 8576.22it/s] 58%|█████▊    | 230070/400000 [00:26<00:19, 8590.79it/s] 58%|█████▊    | 230930/400000 [00:26<00:19, 8556.90it/s] 58%|█████▊    | 231792/400000 [00:27<00:19, 8574.95it/s] 58%|█████▊    | 232669/400000 [00:27<00:19, 8630.03it/s] 58%|█████▊    | 233533/400000 [00:27<00:19, 8594.60it/s] 59%|█████▊    | 234416/400000 [00:27<00:19, 8661.20it/s] 59%|█████▉    | 235283/400000 [00:27<00:19, 8658.00it/s] 59%|█████▉    | 236150/400000 [00:27<00:19, 8510.17it/s] 59%|█████▉    | 237018/400000 [00:27<00:19, 8558.49it/s] 59%|█████▉    | 237883/400000 [00:27<00:18, 8583.84it/s] 60%|█████▉    | 238742/400000 [00:27<00:18, 8568.39it/s] 60%|█████▉    | 239608/400000 [00:27<00:18, 8595.00it/s] 60%|██████    | 240482/400000 [00:28<00:18, 8636.70it/s] 60%|██████    | 241346/400000 [00:28<00:18, 8402.30it/s] 61%|██████    | 242188/400000 [00:28<00:18, 8392.10it/s] 61%|██████    | 243063/400000 [00:28<00:18, 8493.80it/s] 61%|██████    | 243921/400000 [00:28<00:18, 8519.14it/s] 61%|██████    | 244795/400000 [00:28<00:18, 8583.11it/s] 61%|██████▏   | 245670/400000 [00:28<00:17, 8629.65it/s] 62%|██████▏   | 246534/400000 [00:28<00:18, 8503.92it/s] 62%|██████▏   | 247396/400000 [00:28<00:17, 8537.80it/s] 62%|██████▏   | 248251/400000 [00:29<00:17, 8439.89it/s] 62%|██████▏   | 249123/400000 [00:29<00:17, 8519.67it/s] 62%|██████▏   | 249983/400000 [00:29<00:17, 8542.77it/s] 63%|██████▎   | 250838/400000 [00:29<00:17, 8540.41it/s] 63%|██████▎   | 251704/400000 [00:29<00:17, 8573.39it/s] 63%|██████▎   | 252562/400000 [00:29<00:17, 8552.04it/s] 63%|██████▎   | 253434/400000 [00:29<00:17, 8600.53it/s] 64%|██████▎   | 254295/400000 [00:29<00:17, 8566.21it/s] 64%|██████▍   | 255157/400000 [00:29<00:16, 8581.34it/s] 64%|██████▍   | 256038/400000 [00:29<00:16, 8647.71it/s] 64%|██████▍   | 256903/400000 [00:30<00:16, 8454.56it/s] 64%|██████▍   | 257750/400000 [00:30<00:16, 8421.90it/s] 65%|██████▍   | 258605/400000 [00:30<00:16, 8459.46it/s] 65%|██████▍   | 259452/400000 [00:30<00:16, 8417.40it/s] 65%|██████▌   | 260323/400000 [00:30<00:16, 8502.22it/s] 65%|██████▌   | 261174/400000 [00:30<00:16, 8447.10it/s] 66%|██████▌   | 262039/400000 [00:30<00:16, 8506.59it/s] 66%|██████▌   | 262899/400000 [00:30<00:16, 8532.89it/s] 66%|██████▌   | 263759/400000 [00:30<00:15, 8552.90it/s] 66%|██████▌   | 264622/400000 [00:30<00:15, 8573.29it/s] 66%|██████▋   | 265480/400000 [00:31<00:15, 8566.65it/s] 67%|██████▋   | 266355/400000 [00:31<00:15, 8619.33it/s] 67%|██████▋   | 267235/400000 [00:31<00:15, 8672.74it/s] 67%|██████▋   | 268104/400000 [00:31<00:15, 8675.93it/s] 67%|██████▋   | 268980/400000 [00:31<00:15, 8700.42it/s] 67%|██████▋   | 269859/400000 [00:31<00:14, 8724.65it/s] 68%|██████▊   | 270732/400000 [00:31<00:14, 8631.82it/s] 68%|██████▊   | 271596/400000 [00:31<00:15, 8455.16it/s] 68%|██████▊   | 272467/400000 [00:31<00:14, 8521.12it/s] 68%|██████▊   | 273320/400000 [00:31<00:14, 8522.19it/s] 69%|██████▊   | 274173/400000 [00:32<00:14, 8456.06it/s] 69%|██████▉   | 275047/400000 [00:32<00:14, 8538.68it/s] 69%|██████▉   | 275921/400000 [00:32<00:14, 8597.03it/s] 69%|██████▉   | 276802/400000 [00:32<00:14, 8658.79it/s] 69%|██████▉   | 277669/400000 [00:32<00:14, 8537.47it/s] 70%|██████▉   | 278524/400000 [00:32<00:14, 8463.91it/s] 70%|██████▉   | 279398/400000 [00:32<00:14, 8544.64it/s] 70%|███████   | 280260/400000 [00:32<00:13, 8564.39it/s] 70%|███████   | 281117/400000 [00:32<00:14, 8421.61it/s] 70%|███████   | 281961/400000 [00:32<00:14, 8404.23it/s] 71%|███████   | 282808/400000 [00:33<00:13, 8422.87it/s] 71%|███████   | 283674/400000 [00:33<00:13, 8488.68it/s] 71%|███████   | 284552/400000 [00:33<00:13, 8573.01it/s] 71%|███████▏  | 285430/400000 [00:33<00:13, 8632.79it/s] 72%|███████▏  | 286306/400000 [00:33<00:13, 8670.31it/s] 72%|███████▏  | 287188/400000 [00:33<00:12, 8713.60it/s] 72%|███████▏  | 288064/400000 [00:33<00:12, 8725.87it/s] 72%|███████▏  | 288937/400000 [00:33<00:12, 8674.87it/s] 72%|███████▏  | 289805/400000 [00:33<00:12, 8639.04it/s] 73%|███████▎  | 290688/400000 [00:33<00:12, 8693.08it/s] 73%|███████▎  | 291565/400000 [00:34<00:12, 8713.39it/s] 73%|███████▎  | 292437/400000 [00:34<00:12, 8626.48it/s] 73%|███████▎  | 293327/400000 [00:34<00:12, 8705.49it/s] 74%|███████▎  | 294198/400000 [00:34<00:12, 8608.52it/s] 74%|███████▍  | 295060/400000 [00:34<00:12, 8579.91it/s] 74%|███████▍  | 295922/400000 [00:34<00:12, 8589.97it/s] 74%|███████▍  | 296782/400000 [00:34<00:12, 8485.99it/s] 74%|███████▍  | 297645/400000 [00:34<00:12, 8528.25it/s] 75%|███████▍  | 298528/400000 [00:34<00:11, 8616.11it/s] 75%|███████▍  | 299391/400000 [00:34<00:12, 8382.61it/s] 75%|███████▌  | 300233/400000 [00:35<00:11, 8391.25it/s] 75%|███████▌  | 301074/400000 [00:35<00:11, 8391.20it/s] 75%|███████▌  | 301953/400000 [00:35<00:11, 8505.33it/s] 76%|███████▌  | 302825/400000 [00:35<00:11, 8565.49it/s] 76%|███████▌  | 303703/400000 [00:35<00:11, 8627.43it/s] 76%|███████▌  | 304567/400000 [00:35<00:11, 8419.56it/s] 76%|███████▋  | 305447/400000 [00:35<00:11, 8528.73it/s] 77%|███████▋  | 306319/400000 [00:35<00:10, 8584.42it/s] 77%|███████▋  | 307190/400000 [00:35<00:10, 8619.22it/s] 77%|███████▋  | 308064/400000 [00:35<00:10, 8652.30it/s] 77%|███████▋  | 308930/400000 [00:36<00:11, 8253.86it/s] 77%|███████▋  | 309789/400000 [00:36<00:10, 8350.49it/s] 78%|███████▊  | 310639/400000 [00:36<00:10, 8392.46it/s] 78%|███████▊  | 311511/400000 [00:36<00:10, 8487.61it/s] 78%|███████▊  | 312382/400000 [00:36<00:10, 8552.31it/s] 78%|███████▊  | 313253/400000 [00:36<00:10, 8598.10it/s] 79%|███████▊  | 314116/400000 [00:36<00:09, 8604.96it/s] 79%|███████▊  | 314984/400000 [00:36<00:09, 8626.47it/s] 79%|███████▉  | 315848/400000 [00:36<00:09, 8593.73it/s] 79%|███████▉  | 316708/400000 [00:37<00:09, 8442.90it/s] 79%|███████▉  | 317554/400000 [00:37<00:09, 8445.94it/s] 80%|███████▉  | 318400/400000 [00:37<00:09, 8447.13it/s] 80%|███████▉  | 319257/400000 [00:37<00:09, 8481.56it/s] 80%|████████  | 320119/400000 [00:37<00:09, 8521.58it/s] 80%|████████  | 321000/400000 [00:37<00:09, 8601.28it/s] 80%|████████  | 321861/400000 [00:37<00:09, 8552.47it/s] 81%|████████  | 322717/400000 [00:37<00:09, 8435.21it/s] 81%|████████  | 323593/400000 [00:37<00:08, 8528.16it/s] 81%|████████  | 324473/400000 [00:37<00:08, 8606.70it/s] 81%|████████▏ | 325349/400000 [00:38<00:08, 8651.29it/s] 82%|████████▏ | 326215/400000 [00:38<00:08, 8635.48it/s] 82%|████████▏ | 327079/400000 [00:38<00:08, 8596.49it/s] 82%|████████▏ | 327940/400000 [00:38<00:08, 8597.76it/s] 82%|████████▏ | 328810/400000 [00:38<00:08, 8626.28it/s] 82%|████████▏ | 329673/400000 [00:38<00:08, 8624.94it/s] 83%|████████▎ | 330545/400000 [00:38<00:08, 8652.53it/s] 83%|████████▎ | 331411/400000 [00:38<00:07, 8609.35it/s] 83%|████████▎ | 332273/400000 [00:38<00:07, 8566.33it/s] 83%|████████▎ | 333146/400000 [00:38<00:07, 8613.20it/s] 84%|████████▎ | 334029/400000 [00:39<00:07, 8675.06it/s] 84%|████████▎ | 334897/400000 [00:39<00:07, 8667.49it/s] 84%|████████▍ | 335764/400000 [00:39<00:07, 8602.21it/s] 84%|████████▍ | 336647/400000 [00:39<00:07, 8667.65it/s] 84%|████████▍ | 337515/400000 [00:39<00:07, 8619.79it/s] 85%|████████▍ | 338386/400000 [00:39<00:07, 8645.86it/s] 85%|████████▍ | 339251/400000 [00:39<00:07, 8644.51it/s] 85%|████████▌ | 340116/400000 [00:39<00:07, 8382.48it/s] 85%|████████▌ | 340957/400000 [00:39<00:07, 8362.41it/s] 85%|████████▌ | 341806/400000 [00:39<00:06, 8399.45it/s] 86%|████████▌ | 342667/400000 [00:40<00:06, 8461.34it/s] 86%|████████▌ | 343514/400000 [00:40<00:06, 8438.33it/s] 86%|████████▌ | 344395/400000 [00:40<00:06, 8545.17it/s] 86%|████████▋ | 345277/400000 [00:40<00:06, 8624.36it/s] 87%|████████▋ | 346165/400000 [00:40<00:06, 8697.24it/s] 87%|████████▋ | 347048/400000 [00:40<00:06, 8736.59it/s] 87%|████████▋ | 347923/400000 [00:40<00:05, 8697.29it/s] 87%|████████▋ | 348800/400000 [00:40<00:05, 8717.93it/s] 87%|████████▋ | 349676/400000 [00:40<00:05, 8728.49it/s] 88%|████████▊ | 350559/400000 [00:40<00:05, 8758.12it/s] 88%|████████▊ | 351440/400000 [00:41<00:05, 8772.96it/s] 88%|████████▊ | 352318/400000 [00:41<00:05, 8757.46it/s] 88%|████████▊ | 353194/400000 [00:41<00:05, 8728.69it/s] 89%|████████▊ | 354067/400000 [00:41<00:05, 8714.84it/s] 89%|████████▊ | 354939/400000 [00:41<00:05, 8699.48it/s] 89%|████████▉ | 355817/400000 [00:41<00:05, 8721.50it/s] 89%|████████▉ | 356698/400000 [00:41<00:04, 8746.67it/s] 89%|████████▉ | 357573/400000 [00:41<00:04, 8722.06it/s] 90%|████████▉ | 358450/400000 [00:41<00:04, 8734.74it/s] 90%|████████▉ | 359335/400000 [00:41<00:04, 8768.22it/s] 90%|█████████ | 360212/400000 [00:42<00:04, 8678.98it/s] 90%|█████████ | 361086/400000 [00:42<00:04, 8694.35it/s] 90%|█████████ | 361956/400000 [00:42<00:04, 8651.73it/s] 91%|█████████ | 362828/400000 [00:42<00:04, 8669.33it/s] 91%|█████████ | 363696/400000 [00:42<00:04, 8558.05it/s] 91%|█████████ | 364553/400000 [00:42<00:04, 8550.73it/s] 91%|█████████▏| 365425/400000 [00:42<00:04, 8598.83it/s] 92%|█████████▏| 366287/400000 [00:42<00:03, 8602.42it/s] 92%|█████████▏| 367157/400000 [00:42<00:03, 8629.56it/s] 92%|█████████▏| 368021/400000 [00:42<00:03, 8594.33it/s] 92%|█████████▏| 368881/400000 [00:43<00:03, 8588.97it/s] 92%|█████████▏| 369742/400000 [00:43<00:03, 8595.22it/s] 93%|█████████▎| 370602/400000 [00:43<00:03, 8523.05it/s] 93%|█████████▎| 371484/400000 [00:43<00:03, 8607.64it/s] 93%|█████████▎| 372364/400000 [00:43<00:03, 8662.41it/s] 93%|█████████▎| 373245/400000 [00:43<00:03, 8705.29it/s] 94%|█████████▎| 374116/400000 [00:43<00:03, 8620.33it/s] 94%|█████████▎| 374979/400000 [00:43<00:02, 8568.84it/s] 94%|█████████▍| 375851/400000 [00:43<00:02, 8612.51it/s] 94%|█████████▍| 376713/400000 [00:43<00:02, 8613.14it/s] 94%|█████████▍| 377575/400000 [00:44<00:02, 8545.45it/s] 95%|█████████▍| 378445/400000 [00:44<00:02, 8589.41it/s] 95%|█████████▍| 379314/400000 [00:44<00:02, 8619.20it/s] 95%|█████████▌| 380185/400000 [00:44<00:02, 8644.17it/s] 95%|█████████▌| 381050/400000 [00:44<00:02, 8456.21it/s] 95%|█████████▌| 381897/400000 [00:44<00:02, 8252.72it/s] 96%|█████████▌| 382747/400000 [00:44<00:02, 8325.15it/s] 96%|█████████▌| 383598/400000 [00:44<00:01, 8378.51it/s] 96%|█████████▌| 384472/400000 [00:44<00:01, 8483.02it/s] 96%|█████████▋| 385353/400000 [00:44<00:01, 8577.76it/s] 97%|█████████▋| 386219/400000 [00:45<00:01, 8600.92it/s] 97%|█████████▋| 387080/400000 [00:45<00:01, 8565.04it/s] 97%|█████████▋| 387938/400000 [00:45<00:01, 8539.56it/s] 97%|█████████▋| 388811/400000 [00:45<00:01, 8594.53it/s] 97%|█████████▋| 389683/400000 [00:45<00:01, 8630.30it/s] 98%|█████████▊| 390547/400000 [00:45<00:01, 8609.81it/s] 98%|█████████▊| 391416/400000 [00:45<00:00, 8632.39it/s] 98%|█████████▊| 392280/400000 [00:45<00:00, 8530.11it/s] 98%|█████████▊| 393150/400000 [00:45<00:00, 8579.88it/s] 99%|█████████▊| 394012/400000 [00:45<00:00, 8590.55it/s] 99%|█████████▊| 394872/400000 [00:46<00:00, 8367.99it/s] 99%|█████████▉| 395711/400000 [00:46<00:00, 8362.97it/s] 99%|█████████▉| 396549/400000 [00:46<00:00, 8365.69it/s] 99%|█████████▉| 397410/400000 [00:46<00:00, 8434.38it/s]100%|█████████▉| 398273/400000 [00:46<00:00, 8491.41it/s]100%|█████████▉| 399145/400000 [00:46<00:00, 8556.16it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8564.69it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3548d60cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011136732558494645 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011381327308539961 	 Accuracy: 48

  model saves at 48% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15955 out of table with 15921 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15955 out of table with 15921 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

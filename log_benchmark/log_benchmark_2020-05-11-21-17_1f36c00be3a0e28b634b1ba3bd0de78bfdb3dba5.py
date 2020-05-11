
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1f92d2afd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 21:17:43.544667
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 21:17:43.551362
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 21:17:43.554532
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 21:17:43.557743
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1f9ed42470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356198.5625
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 260958.4844
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 156362.4375
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 80138.1094
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 42293.1172
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 24352.1172
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 15383.4873
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 10527.1934
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 7657.1406
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 5880.9082

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.72276729e-01  9.34619427e+00  8.51850986e+00  7.98395824e+00
   9.34173870e+00  9.47655582e+00  9.27039242e+00  7.04928017e+00
   8.50218201e+00  7.88739777e+00  1.08268385e+01  8.74016476e+00
   7.90234041e+00  7.74121094e+00  7.39603662e+00  8.10204887e+00
   7.21765947e+00  8.86292744e+00  9.35527420e+00  1.07381516e+01
   9.41381550e+00  9.28820801e+00  8.60833454e+00  8.99531555e+00
   9.43992138e+00  5.77545023e+00  6.19554090e+00  9.37635803e+00
   7.42564821e+00  8.08641052e+00  8.48700142e+00  9.40437222e+00
   8.59176636e+00  8.13485718e+00  7.28114223e+00  9.67047501e+00
   1.05839577e+01  9.60339832e+00  1.01650009e+01  8.81828117e+00
   9.02422619e+00  9.20160580e+00  7.30605221e+00  9.49846172e+00
   8.44212818e+00  1.08500595e+01  9.39986420e+00  1.01788282e+01
   9.61968994e+00  1.02088470e+01  8.18156052e+00  7.30452442e+00
   7.69454908e+00  8.00226498e+00  1.02549095e+01  8.71895409e+00
   1.08814993e+01  8.84042168e+00  1.18462124e+01  8.61615467e+00
   5.41059896e-02 -5.20410597e-01 -1.23878941e-01 -1.58216321e+00
   2.06814408e-01 -4.91877735e-01 -4.93955910e-01 -1.47148705e+00
  -7.85563231e-01 -2.10079789e-01  6.87747777e-01 -7.00659096e-01
   2.56585741e+00  4.91118014e-01  1.98473722e-01 -3.87110710e-01
  -2.14307857e+00  3.55040550e-01  1.60832608e+00  6.07259631e-01
   4.34317321e-01 -1.49765122e+00  1.01651587e-01 -5.72167039e-01
   2.56501853e-01 -2.22326294e-01  9.94814754e-01  1.35682189e+00
  -6.46990836e-01 -4.78173584e-01  4.62263405e-01 -9.41084027e-01
  -8.40797544e-01  1.60130143e+00 -1.13409251e-01  6.45564437e-01
  -7.39430130e-01 -6.26234889e-01 -1.05269045e-01 -3.04765224e-01
   4.49628085e-01 -4.50049520e-01 -9.44330215e-01  1.93078542e+00
  -2.09650040e+00 -1.04272580e+00  7.32491076e-01 -1.14431357e+00
   2.80615509e-01  3.18920851e+00  2.88816035e-01  1.87992382e+00
  -2.08055139e+00  2.25072622e-01 -1.63605952e+00  6.54401124e-01
  -4.22827512e-01  1.01161325e+00 -5.47659919e-02  7.42100179e-02
   5.30713677e-01 -5.69394529e-01  4.08334583e-01 -2.00332093e+00
  -7.10405648e-01  1.30482626e+00 -4.21382487e-01  8.95398855e-01
  -7.62681425e-01 -3.04741442e-01  1.51977205e+00  3.51415396e-01
   6.57762825e-01  7.15833008e-01 -3.93982738e-01  6.77797794e-02
   3.04362327e-02  1.48647785e+00 -8.22189152e-01  1.61319822e-01
   1.39411807e+00 -1.43137410e-01 -2.34451199e+00  2.11605835e+00
   2.11794138e-01 -1.64627814e+00  2.46514082e-01 -9.36968803e-01
   1.17312912e-02 -1.02264190e+00  4.13453102e-01  1.24752223e+00
   3.32044393e-01 -1.23209465e+00  1.60171783e+00 -1.90589690e+00
  -9.39957440e-01 -7.63213813e-01 -1.41333711e+00  1.86097217e+00
   1.16675186e+00  3.01546991e-01  1.24179935e+00  1.73851192e-01
  -8.61633718e-02  1.30471182e+00 -1.16757178e+00 -9.70808446e-01
  -5.12575209e-01  1.09523642e+00 -6.59648627e-02  1.90381259e-01
  -1.35868466e+00  5.26712298e-01 -9.09953773e-01  9.80442941e-01
   1.30232692e-01 -6.67584360e-01 -2.81129420e-01 -1.60628045e+00
   2.39551842e-01  8.18592262e+00  8.93473911e+00  7.91377068e+00
   9.37872124e+00  9.27548695e+00  1.02296381e+01  8.23039150e+00
   8.32922745e+00  8.48976994e+00  9.83854389e+00  7.98573637e+00
   8.50784969e+00  8.22624588e+00  9.60351944e+00  9.87676811e+00
   7.00650024e+00  1.00901461e+01  7.48398209e+00  8.10812569e+00
   8.60183334e+00  1.01546354e+01  8.06579399e+00  1.04343042e+01
   1.06275530e+01  9.04122257e+00  8.71085167e+00  7.95363474e+00
   8.71863079e+00  8.73010540e+00  9.84515381e+00  8.37716675e+00
   1.11513386e+01  7.98825407e+00  8.62957096e+00  8.25110149e+00
   9.84452438e+00  8.98996162e+00  8.10070419e+00  8.92848015e+00
   8.48556423e+00  9.15325832e+00  1.01444149e+01  9.68190289e+00
   9.75973606e+00  8.71951199e+00  8.96082687e+00  8.90421104e+00
   7.11157036e+00  9.58099747e+00  8.89753151e+00  8.47587585e+00
   9.83024025e+00  7.31691265e+00  9.67992020e+00  8.75753689e+00
   8.28808975e+00  8.90724277e+00  8.30973530e+00  1.00662851e+01
   2.08751917e+00  1.20528197e+00  4.08477247e-01  6.86879396e-01
   3.19025815e-01  4.85954344e-01  2.36391211e+00  2.26340246e+00
   2.57233334e+00  1.69122362e+00  7.91808426e-01  1.87925458e-01
   1.55493724e+00  1.36722755e+00  2.26366138e+00  4.62132215e-01
   9.15907979e-01  1.68196189e+00  1.01110828e+00  4.58272099e-01
   8.17654312e-01  6.48445487e-02  7.55656362e-01  2.54825115e-01
   5.67579031e-01  6.43022776e-01  1.77026451e-01  1.03728020e+00
   4.17683780e-01  3.92120028e+00  2.06139684e-01  2.95385003e-01
   1.32331347e+00  1.65423346e+00  3.40177417e-01  1.48989916e+00
   3.16450500e+00  1.27757072e+00  1.51782095e+00  5.68885922e-01
   1.48453355e+00  4.71476316e-02  4.63782191e-01  8.69826853e-01
   2.59514928e+00  7.22185016e-01  2.88623333e+00  1.59654260e-01
   7.75238872e-01  2.54797578e-01  7.39946246e-01  3.97266269e-01
   8.74023914e-01  3.65651190e-01  1.36144805e+00  1.46508360e+00
   8.07817936e-01  3.88695657e-01  9.61578012e-01  1.76078415e+00
   1.83095241e+00  2.12590313e+00  1.35893714e+00  1.52265739e+00
   1.33592772e+00  1.52012134e+00  2.88635612e-01  1.46985757e+00
   2.48420238e-01  5.01449049e-01  2.36040401e+00  1.14807487e+00
   1.95341682e+00  2.23968089e-01  1.13046682e+00  3.97436619e-01
   1.03002882e+00  8.18952799e-01  1.58952439e+00  1.75195312e+00
   1.16263509e+00  9.92102146e-01  7.84768105e-01  2.23978221e-01
   1.39527869e+00  1.82863081e+00  3.14777136e+00  9.59407926e-01
   1.41957831e+00  2.17855835e+00  1.73913169e+00  1.91706121e-01
   5.59424460e-01  2.40133119e+00  1.00474739e+00  5.44488251e-01
   6.81760907e-01  3.25231373e-01  4.82726336e-01  2.71373749e-01
   5.15612125e-01  2.77559090e+00  1.36488950e+00  1.66276693e+00
   2.73674488e-01  1.73629022e+00  6.64679289e-01  7.23979473e-01
   5.94920456e-01  1.55212688e+00  9.40268517e-01  1.65848696e+00
   6.34507179e-01  2.14720011e-01  2.73898888e+00  3.08984423e+00
   1.83559990e+00  1.02936792e+00  2.30863380e+00  2.79351783e+00
   1.76258159e+00 -1.09850998e+01 -1.12410755e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 21:17:51.938407
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.6368
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 21:17:51.942255
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8787.07
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 21:17:51.945580
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.6715
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 21:17:51.948637
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -785.944
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139773244440360
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139772285926928
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139772285489224
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139772285489728
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139772285490232
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139772285490736

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1f7f399908> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.487779
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.447942
grad_step = 000002, loss = 0.413010
grad_step = 000003, loss = 0.375552
grad_step = 000004, loss = 0.334595
grad_step = 000005, loss = 0.296477
grad_step = 000006, loss = 0.261325
grad_step = 000007, loss = 0.247792
grad_step = 000008, loss = 0.241463
grad_step = 000009, loss = 0.217307
grad_step = 000010, loss = 0.191474
grad_step = 000011, loss = 0.173827
grad_step = 000012, loss = 0.163307
grad_step = 000013, loss = 0.155766
grad_step = 000014, loss = 0.149750
grad_step = 000015, loss = 0.143403
grad_step = 000016, loss = 0.133417
grad_step = 000017, loss = 0.123092
grad_step = 000018, loss = 0.115298
grad_step = 000019, loss = 0.108816
grad_step = 000020, loss = 0.102052
grad_step = 000021, loss = 0.095425
grad_step = 000022, loss = 0.090056
grad_step = 000023, loss = 0.085322
grad_step = 000024, loss = 0.079897
grad_step = 000025, loss = 0.073885
grad_step = 000026, loss = 0.068281
grad_step = 000027, loss = 0.063640
grad_step = 000028, loss = 0.059737
grad_step = 000029, loss = 0.055818
grad_step = 000030, loss = 0.051456
grad_step = 000031, loss = 0.047211
grad_step = 000032, loss = 0.043653
grad_step = 000033, loss = 0.040753
grad_step = 000034, loss = 0.038166
grad_step = 000035, loss = 0.035495
grad_step = 000036, loss = 0.032775
grad_step = 000037, loss = 0.030145
grad_step = 000038, loss = 0.027627
grad_step = 000039, loss = 0.025327
grad_step = 000040, loss = 0.023289
grad_step = 000041, loss = 0.021579
grad_step = 000042, loss = 0.020052
grad_step = 000043, loss = 0.018438
grad_step = 000044, loss = 0.016799
grad_step = 000045, loss = 0.015329
grad_step = 000046, loss = 0.014149
grad_step = 000047, loss = 0.013117
grad_step = 000048, loss = 0.012079
grad_step = 000049, loss = 0.011020
grad_step = 000050, loss = 0.010028
grad_step = 000051, loss = 0.009221
grad_step = 000052, loss = 0.008546
grad_step = 000053, loss = 0.007927
grad_step = 000054, loss = 0.007310
grad_step = 000055, loss = 0.006728
grad_step = 000056, loss = 0.006198
grad_step = 000057, loss = 0.005745
grad_step = 000058, loss = 0.005364
grad_step = 000059, loss = 0.005014
grad_step = 000060, loss = 0.004664
grad_step = 000061, loss = 0.004333
grad_step = 000062, loss = 0.004069
grad_step = 000063, loss = 0.003868
grad_step = 000064, loss = 0.003689
grad_step = 000065, loss = 0.003498
grad_step = 000066, loss = 0.003317
grad_step = 000067, loss = 0.003168
grad_step = 000068, loss = 0.003049
grad_step = 000069, loss = 0.002936
grad_step = 000070, loss = 0.002833
grad_step = 000071, loss = 0.002747
grad_step = 000072, loss = 0.002677
grad_step = 000073, loss = 0.002613
grad_step = 000074, loss = 0.002556
grad_step = 000075, loss = 0.002511
grad_step = 000076, loss = 0.002473
grad_step = 000077, loss = 0.002434
grad_step = 000078, loss = 0.002398
grad_step = 000079, loss = 0.002371
grad_step = 000080, loss = 0.002350
grad_step = 000081, loss = 0.002331
grad_step = 000082, loss = 0.002311
grad_step = 000083, loss = 0.002294
grad_step = 000084, loss = 0.002279
grad_step = 000085, loss = 0.002269
grad_step = 000086, loss = 0.002261
grad_step = 000087, loss = 0.002251
grad_step = 000088, loss = 0.002241
grad_step = 000089, loss = 0.002233
grad_step = 000090, loss = 0.002229
grad_step = 000091, loss = 0.002224
grad_step = 000092, loss = 0.002217
grad_step = 000093, loss = 0.002211
grad_step = 000094, loss = 0.002208
grad_step = 000095, loss = 0.002205
grad_step = 000096, loss = 0.002200
grad_step = 000097, loss = 0.002197
grad_step = 000098, loss = 0.002194
grad_step = 000099, loss = 0.002191
grad_step = 000100, loss = 0.002188
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002186
grad_step = 000102, loss = 0.002183
grad_step = 000103, loss = 0.002179
grad_step = 000104, loss = 0.002177
grad_step = 000105, loss = 0.002174
grad_step = 000106, loss = 0.002171
grad_step = 000107, loss = 0.002168
grad_step = 000108, loss = 0.002166
grad_step = 000109, loss = 0.002163
grad_step = 000110, loss = 0.002160
grad_step = 000111, loss = 0.002157
grad_step = 000112, loss = 0.002154
grad_step = 000113, loss = 0.002151
grad_step = 000114, loss = 0.002148
grad_step = 000115, loss = 0.002146
grad_step = 000116, loss = 0.002143
grad_step = 000117, loss = 0.002140
grad_step = 000118, loss = 0.002137
grad_step = 000119, loss = 0.002134
grad_step = 000120, loss = 0.002131
grad_step = 000121, loss = 0.002129
grad_step = 000122, loss = 0.002126
grad_step = 000123, loss = 0.002123
grad_step = 000124, loss = 0.002120
grad_step = 000125, loss = 0.002118
grad_step = 000126, loss = 0.002115
grad_step = 000127, loss = 0.002112
grad_step = 000128, loss = 0.002110
grad_step = 000129, loss = 0.002107
grad_step = 000130, loss = 0.002104
grad_step = 000131, loss = 0.002102
grad_step = 000132, loss = 0.002099
grad_step = 000133, loss = 0.002096
grad_step = 000134, loss = 0.002094
grad_step = 000135, loss = 0.002091
grad_step = 000136, loss = 0.002088
grad_step = 000137, loss = 0.002086
grad_step = 000138, loss = 0.002083
grad_step = 000139, loss = 0.002080
grad_step = 000140, loss = 0.002078
grad_step = 000141, loss = 0.002075
grad_step = 000142, loss = 0.002072
grad_step = 000143, loss = 0.002069
grad_step = 000144, loss = 0.002067
grad_step = 000145, loss = 0.002064
grad_step = 000146, loss = 0.002061
grad_step = 000147, loss = 0.002059
grad_step = 000148, loss = 0.002057
grad_step = 000149, loss = 0.002058
grad_step = 000150, loss = 0.002064
grad_step = 000151, loss = 0.002075
grad_step = 000152, loss = 0.002080
grad_step = 000153, loss = 0.002072
grad_step = 000154, loss = 0.002048
grad_step = 000155, loss = 0.002036
grad_step = 000156, loss = 0.002042
grad_step = 000157, loss = 0.002052
grad_step = 000158, loss = 0.002049
grad_step = 000159, loss = 0.002032
grad_step = 000160, loss = 0.002021
grad_step = 000161, loss = 0.002024
grad_step = 000162, loss = 0.002030
grad_step = 000163, loss = 0.002026
grad_step = 000164, loss = 0.002015
grad_step = 000165, loss = 0.002006
grad_step = 000166, loss = 0.002006
grad_step = 000167, loss = 0.002009
grad_step = 000168, loss = 0.002009
grad_step = 000169, loss = 0.002002
grad_step = 000170, loss = 0.001993
grad_step = 000171, loss = 0.001988
grad_step = 000172, loss = 0.001987
grad_step = 000173, loss = 0.001987
grad_step = 000174, loss = 0.001987
grad_step = 000175, loss = 0.001984
grad_step = 000176, loss = 0.001979
grad_step = 000177, loss = 0.001973
grad_step = 000178, loss = 0.001967
grad_step = 000179, loss = 0.001963
grad_step = 000180, loss = 0.001959
grad_step = 000181, loss = 0.001956
grad_step = 000182, loss = 0.001954
grad_step = 000183, loss = 0.001953
grad_step = 000184, loss = 0.001954
grad_step = 000185, loss = 0.001959
grad_step = 000186, loss = 0.001973
grad_step = 000187, loss = 0.001997
grad_step = 000188, loss = 0.002031
grad_step = 000189, loss = 0.002025
grad_step = 000190, loss = 0.001979
grad_step = 000191, loss = 0.001928
grad_step = 000192, loss = 0.001936
grad_step = 000193, loss = 0.001973
grad_step = 000194, loss = 0.001965
grad_step = 000195, loss = 0.001926
grad_step = 000196, loss = 0.001914
grad_step = 000197, loss = 0.001936
grad_step = 000198, loss = 0.001944
grad_step = 000199, loss = 0.001917
grad_step = 000200, loss = 0.001902
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001914
grad_step = 000202, loss = 0.001922
grad_step = 000203, loss = 0.001908
grad_step = 000204, loss = 0.001892
grad_step = 000205, loss = 0.001895
grad_step = 000206, loss = 0.001904
grad_step = 000207, loss = 0.001900
grad_step = 000208, loss = 0.001887
grad_step = 000209, loss = 0.001881
grad_step = 000210, loss = 0.001885
grad_step = 000211, loss = 0.001888
grad_step = 000212, loss = 0.001885
grad_step = 000213, loss = 0.001876
grad_step = 000214, loss = 0.001870
grad_step = 000215, loss = 0.001869
grad_step = 000216, loss = 0.001871
grad_step = 000217, loss = 0.001873
grad_step = 000218, loss = 0.001871
grad_step = 000219, loss = 0.001866
grad_step = 000220, loss = 0.001861
grad_step = 000221, loss = 0.001857
grad_step = 000222, loss = 0.001854
grad_step = 000223, loss = 0.001852
grad_step = 000224, loss = 0.001851
grad_step = 000225, loss = 0.001851
grad_step = 000226, loss = 0.001852
grad_step = 000227, loss = 0.001856
grad_step = 000228, loss = 0.001863
grad_step = 000229, loss = 0.001879
grad_step = 000230, loss = 0.001907
grad_step = 000231, loss = 0.001948
grad_step = 000232, loss = 0.001972
grad_step = 000233, loss = 0.001942
grad_step = 000234, loss = 0.001868
grad_step = 000235, loss = 0.001829
grad_step = 000236, loss = 0.001857
grad_step = 000237, loss = 0.001892
grad_step = 000238, loss = 0.001860
grad_step = 000239, loss = 0.001826
grad_step = 000240, loss = 0.001841
grad_step = 000241, loss = 0.001858
grad_step = 000242, loss = 0.001841
grad_step = 000243, loss = 0.001817
grad_step = 000244, loss = 0.001824
grad_step = 000245, loss = 0.001843
grad_step = 000246, loss = 0.001841
grad_step = 000247, loss = 0.001820
grad_step = 000248, loss = 0.001807
grad_step = 000249, loss = 0.001812
grad_step = 000250, loss = 0.001823
grad_step = 000251, loss = 0.001824
grad_step = 000252, loss = 0.001812
grad_step = 000253, loss = 0.001800
grad_step = 000254, loss = 0.001796
grad_step = 000255, loss = 0.001801
grad_step = 000256, loss = 0.001806
grad_step = 000257, loss = 0.001805
grad_step = 000258, loss = 0.001799
grad_step = 000259, loss = 0.001791
grad_step = 000260, loss = 0.001785
grad_step = 000261, loss = 0.001783
grad_step = 000262, loss = 0.001785
grad_step = 000263, loss = 0.001787
grad_step = 000264, loss = 0.001789
grad_step = 000265, loss = 0.001789
grad_step = 000266, loss = 0.001788
grad_step = 000267, loss = 0.001786
grad_step = 000268, loss = 0.001783
grad_step = 000269, loss = 0.001780
grad_step = 000270, loss = 0.001778
grad_step = 000271, loss = 0.001776
grad_step = 000272, loss = 0.001775
grad_step = 000273, loss = 0.001776
grad_step = 000274, loss = 0.001778
grad_step = 000275, loss = 0.001783
grad_step = 000276, loss = 0.001793
grad_step = 000277, loss = 0.001810
grad_step = 000278, loss = 0.001831
grad_step = 000279, loss = 0.001855
grad_step = 000280, loss = 0.001857
grad_step = 000281, loss = 0.001831
grad_step = 000282, loss = 0.001782
grad_step = 000283, loss = 0.001749
grad_step = 000284, loss = 0.001753
grad_step = 000285, loss = 0.001779
grad_step = 000286, loss = 0.001792
grad_step = 000287, loss = 0.001778
grad_step = 000288, loss = 0.001750
grad_step = 000289, loss = 0.001737
grad_step = 000290, loss = 0.001746
grad_step = 000291, loss = 0.001758
grad_step = 000292, loss = 0.001758
grad_step = 000293, loss = 0.001743
grad_step = 000294, loss = 0.001731
grad_step = 000295, loss = 0.001732
grad_step = 000296, loss = 0.001740
grad_step = 000297, loss = 0.001745
grad_step = 000298, loss = 0.001742
grad_step = 000299, loss = 0.001733
grad_step = 000300, loss = 0.001724
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001720
grad_step = 000302, loss = 0.001720
grad_step = 000303, loss = 0.001723
grad_step = 000304, loss = 0.001727
grad_step = 000305, loss = 0.001729
grad_step = 000306, loss = 0.001729
grad_step = 000307, loss = 0.001727
grad_step = 000308, loss = 0.001724
grad_step = 000309, loss = 0.001720
grad_step = 000310, loss = 0.001716
grad_step = 000311, loss = 0.001712
grad_step = 000312, loss = 0.001709
grad_step = 000313, loss = 0.001706
grad_step = 000314, loss = 0.001704
grad_step = 000315, loss = 0.001702
grad_step = 000316, loss = 0.001701
grad_step = 000317, loss = 0.001700
grad_step = 000318, loss = 0.001700
grad_step = 000319, loss = 0.001701
grad_step = 000320, loss = 0.001705
grad_step = 000321, loss = 0.001717
grad_step = 000322, loss = 0.001741
grad_step = 000323, loss = 0.001795
grad_step = 000324, loss = 0.001888
grad_step = 000325, loss = 0.001991
grad_step = 000326, loss = 0.002007
grad_step = 000327, loss = 0.001838
grad_step = 000328, loss = 0.001693
grad_step = 000329, loss = 0.001755
grad_step = 000330, loss = 0.001820
grad_step = 000331, loss = 0.001731
grad_step = 000332, loss = 0.001709
grad_step = 000333, loss = 0.001783
grad_step = 000334, loss = 0.001750
grad_step = 000335, loss = 0.001693
grad_step = 000336, loss = 0.001761
grad_step = 000337, loss = 0.001755
grad_step = 000338, loss = 0.001691
grad_step = 000339, loss = 0.001730
grad_step = 000340, loss = 0.001751
grad_step = 000341, loss = 0.001688
grad_step = 000342, loss = 0.001707
grad_step = 000343, loss = 0.001735
grad_step = 000344, loss = 0.001686
grad_step = 000345, loss = 0.001690
grad_step = 000346, loss = 0.001715
grad_step = 000347, loss = 0.001685
grad_step = 000348, loss = 0.001677
grad_step = 000349, loss = 0.001699
grad_step = 000350, loss = 0.001681
grad_step = 000351, loss = 0.001670
grad_step = 000352, loss = 0.001684
grad_step = 000353, loss = 0.001679
grad_step = 000354, loss = 0.001664
grad_step = 000355, loss = 0.001671
grad_step = 000356, loss = 0.001675
grad_step = 000357, loss = 0.001663
grad_step = 000358, loss = 0.001660
grad_step = 000359, loss = 0.001666
grad_step = 000360, loss = 0.001663
grad_step = 000361, loss = 0.001654
grad_step = 000362, loss = 0.001655
grad_step = 000363, loss = 0.001658
grad_step = 000364, loss = 0.001653
grad_step = 000365, loss = 0.001644
grad_step = 000366, loss = 0.001642
grad_step = 000367, loss = 0.001643
grad_step = 000368, loss = 0.001641
grad_step = 000369, loss = 0.001638
grad_step = 000370, loss = 0.001635
grad_step = 000371, loss = 0.001632
grad_step = 000372, loss = 0.001630
grad_step = 000373, loss = 0.001629
grad_step = 000374, loss = 0.001628
grad_step = 000375, loss = 0.001628
grad_step = 000376, loss = 0.001628
grad_step = 000377, loss = 0.001628
grad_step = 000378, loss = 0.001628
grad_step = 000379, loss = 0.001629
grad_step = 000380, loss = 0.001631
grad_step = 000381, loss = 0.001636
grad_step = 000382, loss = 0.001647
grad_step = 000383, loss = 0.001666
grad_step = 000384, loss = 0.001702
grad_step = 000385, loss = 0.001748
grad_step = 000386, loss = 0.001810
grad_step = 000387, loss = 0.001813
grad_step = 000388, loss = 0.001750
grad_step = 000389, loss = 0.001649
grad_step = 000390, loss = 0.001612
grad_step = 000391, loss = 0.001661
grad_step = 000392, loss = 0.001702
grad_step = 000393, loss = 0.001670
grad_step = 000394, loss = 0.001612
grad_step = 000395, loss = 0.001612
grad_step = 000396, loss = 0.001651
grad_step = 000397, loss = 0.001663
grad_step = 000398, loss = 0.001630
grad_step = 000399, loss = 0.001599
grad_step = 000400, loss = 0.001601
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001624
grad_step = 000402, loss = 0.001633
grad_step = 000403, loss = 0.001617
grad_step = 000404, loss = 0.001594
grad_step = 000405, loss = 0.001590
grad_step = 000406, loss = 0.001602
grad_step = 000407, loss = 0.001610
grad_step = 000408, loss = 0.001602
grad_step = 000409, loss = 0.001588
grad_step = 000410, loss = 0.001582
grad_step = 000411, loss = 0.001587
grad_step = 000412, loss = 0.001593
grad_step = 000413, loss = 0.001591
grad_step = 000414, loss = 0.001582
grad_step = 000415, loss = 0.001575
grad_step = 000416, loss = 0.001575
grad_step = 000417, loss = 0.001578
grad_step = 000418, loss = 0.001580
grad_step = 000419, loss = 0.001578
grad_step = 000420, loss = 0.001572
grad_step = 000421, loss = 0.001567
grad_step = 000422, loss = 0.001565
grad_step = 000423, loss = 0.001566
grad_step = 000424, loss = 0.001567
grad_step = 000425, loss = 0.001566
grad_step = 000426, loss = 0.001564
grad_step = 000427, loss = 0.001561
grad_step = 000428, loss = 0.001558
grad_step = 000429, loss = 0.001555
grad_step = 000430, loss = 0.001553
grad_step = 000431, loss = 0.001552
grad_step = 000432, loss = 0.001551
grad_step = 000433, loss = 0.001551
grad_step = 000434, loss = 0.001551
grad_step = 000435, loss = 0.001551
grad_step = 000436, loss = 0.001552
grad_step = 000437, loss = 0.001555
grad_step = 000438, loss = 0.001560
grad_step = 000439, loss = 0.001569
grad_step = 000440, loss = 0.001583
grad_step = 000441, loss = 0.001603
grad_step = 000442, loss = 0.001634
grad_step = 000443, loss = 0.001656
grad_step = 000444, loss = 0.001670
grad_step = 000445, loss = 0.001641
grad_step = 000446, loss = 0.001588
grad_step = 000447, loss = 0.001540
grad_step = 000448, loss = 0.001533
grad_step = 000449, loss = 0.001560
grad_step = 000450, loss = 0.001584
grad_step = 000451, loss = 0.001582
grad_step = 000452, loss = 0.001554
grad_step = 000453, loss = 0.001526
grad_step = 000454, loss = 0.001519
grad_step = 000455, loss = 0.001530
grad_step = 000456, loss = 0.001548
grad_step = 000457, loss = 0.001561
grad_step = 000458, loss = 0.001560
grad_step = 000459, loss = 0.001544
grad_step = 000460, loss = 0.001524
grad_step = 000461, loss = 0.001510
grad_step = 000462, loss = 0.001509
grad_step = 000463, loss = 0.001516
grad_step = 000464, loss = 0.001525
grad_step = 000465, loss = 0.001526
grad_step = 000466, loss = 0.001521
grad_step = 000467, loss = 0.001511
grad_step = 000468, loss = 0.001502
grad_step = 000469, loss = 0.001497
grad_step = 000470, loss = 0.001497
grad_step = 000471, loss = 0.001500
grad_step = 000472, loss = 0.001503
grad_step = 000473, loss = 0.001507
grad_step = 000474, loss = 0.001508
grad_step = 000475, loss = 0.001507
grad_step = 000476, loss = 0.001504
grad_step = 000477, loss = 0.001499
grad_step = 000478, loss = 0.001494
grad_step = 000479, loss = 0.001490
grad_step = 000480, loss = 0.001485
grad_step = 000481, loss = 0.001482
grad_step = 000482, loss = 0.001479
grad_step = 000483, loss = 0.001477
grad_step = 000484, loss = 0.001475
grad_step = 000485, loss = 0.001473
grad_step = 000486, loss = 0.001471
grad_step = 000487, loss = 0.001469
grad_step = 000488, loss = 0.001468
grad_step = 000489, loss = 0.001466
grad_step = 000490, loss = 0.001465
grad_step = 000491, loss = 0.001464
grad_step = 000492, loss = 0.001465
grad_step = 000493, loss = 0.001472
grad_step = 000494, loss = 0.001494
grad_step = 000495, loss = 0.001556
grad_step = 000496, loss = 0.001723
grad_step = 000497, loss = 0.001961
grad_step = 000498, loss = 0.002309
grad_step = 000499, loss = 0.002007
grad_step = 000500, loss = 0.001535
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001573
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

  date_run                              2020-05-11 21:18:09.881573
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.259848
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 21:18:09.887397
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.170741
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 21:18:09.894993
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.15595
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 21:18:09.899651
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.59447
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
0   2020-05-11 21:17:43.544667  ...    mean_absolute_error
1   2020-05-11 21:17:43.551362  ...     mean_squared_error
2   2020-05-11 21:17:43.554532  ...  median_absolute_error
3   2020-05-11 21:17:43.557743  ...               r2_score
4   2020-05-11 21:17:51.938407  ...    mean_absolute_error
5   2020-05-11 21:17:51.942255  ...     mean_squared_error
6   2020-05-11 21:17:51.945580  ...  median_absolute_error
7   2020-05-11 21:17:51.948637  ...               r2_score
8   2020-05-11 21:18:09.881573  ...    mean_absolute_error
9   2020-05-11 21:18:09.887397  ...     mean_squared_error
10  2020-05-11 21:18:09.894993  ...  median_absolute_error
11  2020-05-11 21:18:09.899651  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▌      | 3506176/9912422 [00:00<00:00, 35040990.90it/s]9920512it [00:00, 33291602.99it/s]                             
0it [00:00, ?it/s]32768it [00:00, 609082.93it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 487103.87it/s]1654784it [00:00, 11883462.77it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194674.94it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d54277be0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4cf19c7f28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d54277cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4cf14a2048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4cf19c80b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d06c32e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d06c43d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d54283f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d54277cf8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d06c43eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4d54277be0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd986d9c1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7b7d95e006f691643bc6440e6f88d9061267d09e4b4dbf7dc47f53ee2c39e38d
  Stored in directory: /tmp/pip-ephem-wheel-cache-mt7y65uk/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd97cf0a080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1114112/17464789 [>.............................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
17293312/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 21:19:35.214372: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 21:19:35.219546: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-11 21:19:35.219688: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562406020240 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 21:19:35.219702: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5363 - accuracy: 0.5085 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6168 - accuracy: 0.5033
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5470 - accuracy: 0.5078
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7416 - accuracy: 0.4951
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7218 - accuracy: 0.4964
11000/25000 [============>.................] - ETA: 3s - loss: 7.7043 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 3s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6879 - accuracy: 0.4986
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6633 - accuracy: 0.5002
15000/25000 [=================>............] - ETA: 2s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6387 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6283 - accuracy: 0.5025
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6400 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6367 - accuracy: 0.5020
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6389 - accuracy: 0.5018
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6429 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 263us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 21:19:48.138230
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 21:19:48.138230  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 21:19:53.897564: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 21:19:53.903435: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-11 21:19:53.903597: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562d20b2ddf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 21:19:53.903611: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd905ae4cf8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8935 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.8050 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd8e1825898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2066 - accuracy: 0.5300
 2000/25000 [=>............................] - ETA: 7s - loss: 7.2066 - accuracy: 0.5300 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4008 - accuracy: 0.5173
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4366 - accuracy: 0.5150
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4305 - accuracy: 0.5154
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.4698 - accuracy: 0.5128
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.4914 - accuracy: 0.5114
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5133 - accuracy: 0.5100
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5593 - accuracy: 0.5070
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5716 - accuracy: 0.5062
11000/25000 [============>.................] - ETA: 3s - loss: 7.6137 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 3s - loss: 7.6245 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6029 - accuracy: 0.5042
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5965 - accuracy: 0.5046
15000/25000 [=================>............] - ETA: 2s - loss: 7.6176 - accuracy: 0.5032
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6177 - accuracy: 0.5032
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6342 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6836 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6912 - accuracy: 0.4984
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7009 - accuracy: 0.4978
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6959 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7066 - accuracy: 0.4974
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6839 - accuracy: 0.4989
25000/25000 [==============================] - 7s 266us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd8e01005f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:42:51, 9.69kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:31:55, 13.7kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<12:19:42, 19.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:38:13, 27.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.62M/862M [00:01<6:01:49, 39.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<4:11:41, 56.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:55:06, 80.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<2:01:54, 115kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:25:07, 164kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:01<59:17, 234kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:02<41:27, 334kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<28:55, 475kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:02<20:17, 675kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<14:10, 959kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.3M/862M [00:02<10:01, 1.35MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<07:41, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<07:16, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:57, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<05:19, 2.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:15, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<05:45, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:19, 3.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:07, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:59, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<04:32, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<05:48, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<07:39, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<06:04, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<04:32, 2.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:17, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:05, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<04:40, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<05:51, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<07:27, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<06:04, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<04:25, 2.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<09:33, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<08:21, 1.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:17<06:11, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<06:29, 1.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:19<04:57, 2.61MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:00, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:32, 1.71MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:21<04:23, 2.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:19, 1.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<06:46, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<05:08, 2.49MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<07:32, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<05:58, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:23, 2.90MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:53, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:24, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:49, 2.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:21, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:58, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:20, 2.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:01, 1.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:55, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:33, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:46, 2.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:48, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:28, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:19, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:51, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:09, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:44, 2.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:38, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:19, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:10, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:44, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:51, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:17, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:35, 2.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:27, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:11, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<06:06, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:40, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:13, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:40, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:40, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:31, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:11, 2.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:18, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:47, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:57, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:44, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:14, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:41, 2.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:37, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:58, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:38, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:07, 2.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:56, 1.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:45, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:44, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:20, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:25, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:50, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:15, 2.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:51, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:17, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:43, 2.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:36, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:53, 1.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:26, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:03, 2.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:32, 2.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:06, 2.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:08, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:41, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:57, 2.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:26, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:16, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:59, 2.84MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:01, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:25, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:06, 2.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:43, 3.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:41, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:07, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:35, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:25, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:38, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:15, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:50, 2.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:10, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:42, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:20, 2.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:12, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:28, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<05:14, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:48, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:06, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:04, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:14, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:01, 2.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:57, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:14, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:57, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:37, 2.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:10, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:41, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:19, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:07, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:43, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<03:38, 2.91MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<02:40, 3.97MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<59:39, 178kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<44:37, 237kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<31:46, 333kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<22:21, 472kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<18:48, 559kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<14:30, 725kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<10:25, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<07:24, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<23:04, 453kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<17:27, 598kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<12:29, 835kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<08:51, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<16:53, 615kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<13:08, 789kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<09:28, 1.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<06:45, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<14:04, 732kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<11:10, 922kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<08:05, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:40, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<08:20, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<06:30, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:40, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:19, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:07, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:16, 1.93MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:40, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:33, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:10, 1.95MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:47, 2.66MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:33, 1.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:05, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:49, 2.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:43, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:35, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:28, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:23, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:20, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:18, 2.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:25, 4.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<25:12, 390kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<20:10, 487kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<14:38, 670kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<10:24, 940kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<09:51, 989kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<08:09, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:57, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:05, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:30, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:09, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:48, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<04:35, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:29, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:19, 2.21MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:29, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:27, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:14, 2.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:01, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:07, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:32, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:02, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [01:59<05:56, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:43, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:26, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<06:05, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:30, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:59, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:52, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<04:37, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:23, 2.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:51, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:35, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:27, 2.65MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:03, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:56, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<10:31, 861kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:24, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:05, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:11, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:38, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:13, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:47, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:13, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<06:01, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:25, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:17, 2.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:52, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:54, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:24, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:37, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:30, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:15, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:32, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:41, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:13, 2.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:39, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:37, 1.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:26, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:14, 2.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:13, 1.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:27, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:30, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:20, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<03:05, 2.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<06:17, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:29, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:05, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:30, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:17, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:14, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<03:05, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:25, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:33, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:07, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:29, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:15, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:07, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:00, 2.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:43, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:21, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:17, 2.47MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:52, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:48, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:59, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:55, 2.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:54, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:29, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:23, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:46, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:53, 2.75MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:34, 2.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:38, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:41, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:40, 2.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:54, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:18, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:40, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:46, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:45, 2.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:51, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:05, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:48, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:09, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:53, 1.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:56, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:24, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:30, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:36, 2.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:38, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:23, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:34, 2.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:22, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:16, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:23, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:33, 2.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:26, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:14, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:27, 2.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:15, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:08, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:19, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:25, 2.98MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:03, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:37, 1.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:45, 2.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:20, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:15, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:27, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<01:48, 3.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<22:24, 315kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<17:35, 401kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<12:42, 555kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<08:59, 780kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<08:09, 857kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<06:29, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:43, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:46, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:07, 1.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:01, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:54, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:33, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:45, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:30, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:31, 2.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<42:22, 160kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<31:34, 215kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<22:34, 300kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<15:58, 423kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<12:22, 542kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<09:31, 704kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:51, 974kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:05, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:08, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:48, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:56, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:35, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:41, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:10, 2.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:53, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:03, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:21, 2.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:54, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:46, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:05, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<01:32, 4.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<21:11, 300kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<16:15, 391kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<11:39, 544kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<08:13, 769kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<08:05, 777kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<06:27, 975kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:40, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:29, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:44, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:42, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:41, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:54, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:14, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:08, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:15, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<15:17, 398kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<12:16, 496kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<08:54, 682kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<06:18, 959kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<06:14, 963kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<05:02, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:40, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:39, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:47, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:40, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:39, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:41, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:12, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:23, 2.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:47, 3.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:17, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:41, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:39, 1.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:38, 2.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:45, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:04, 1.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:01, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:13, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:59, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:14, 2.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:39, 3.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:12, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:40, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:44, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:59, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<03:36, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:05, 2.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:08, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:36, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:54, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:04, 2.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:29, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:34, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:52, 2.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:00, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:28, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:34, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:18, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:36, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:54, 2.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:53, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:40, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:01, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:24, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:21, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:44, 2.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:31, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:23, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<01:48, 2.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:14, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:10, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<01:39, 2.94MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:06, 2.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:42, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:09, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:34, 3.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:26, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:50, 2.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:12, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:13, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:37, 2.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:37, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:25, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:50, 2.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:10, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:41, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:33, 2.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:33, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:21, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:46, 2.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:06, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:40, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:06, 2.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:31, 2.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:31, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:44, 2.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:02, 2.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:32, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:03, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:29, 2.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:05, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:42, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:59, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:12, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:36, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:05, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:30, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:05, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:39, 1.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:56, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:11, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:33, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:00, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:27, 2.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:17, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:00, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:31, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<01:06, 3.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<13:39, 283kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<10:32, 366kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<07:34, 509kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<05:20, 717kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:42, 806kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:46, 1.00MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:45, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:38, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:51, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:35, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:00, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:29, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:40, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:16, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:34, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:00, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:35, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:09, 2.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:57, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:48, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:21, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:36, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:59, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:36, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:10, 2.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:30, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:10, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:36, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:08, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<1:05:41, 49.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<46:46, 69.3kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<32:49, 98.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<22:49, 140kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<16:47, 189kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<12:07, 261kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<08:32, 369kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<06:30, 477kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:56, 628kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:31, 872kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:02, 999kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:30, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:50, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:51, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:03, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:35, 1.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:09, 2.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:37, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:12, 2.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:51, 3.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<24:46, 114kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<18:03, 157kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<12:44, 221kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<08:51, 314kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<07:05, 389kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<05:14, 525kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:47, 723kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<02:39, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:16, 822kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:53, 931kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:08, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:30, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:30, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:05, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:32, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:24, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:03, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:30, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:53, 2.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:07, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:05, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:49, 2.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:01, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:21, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:05, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:43, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:26, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:05, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:47, 2.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:37, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:43, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:20, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:57, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:43, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:28, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:04, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:09, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:04, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:47, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:10, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:39, 2.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:11, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:02, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:46, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:38, 2.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:21, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:10, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:55, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:38, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:57, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:32, 2.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:04, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:57, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:42, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:52, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:29, 2.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:37, 885kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:18, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:56, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:54, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:59, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:32, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:00, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:38, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:28, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:33, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:41, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:23, 2.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:23, 2.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:35, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:19, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:32, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:22, 2.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:25, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:18, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:12, 3.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:22, 345kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:44, 467kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<01:12, 656kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:49, 921kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:56, 801kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:50, 885kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:37, 1.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:25, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:28, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:22, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:16, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:10, 2.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:21, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:12, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:08, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 815kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 1.41MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 837/400000 [00:00<00:47, 8361.83it/s]  0%|          | 1679/400000 [00:00<00:47, 8378.62it/s]  1%|          | 2474/400000 [00:00<00:48, 8244.93it/s]  1%|          | 3296/400000 [00:00<00:48, 8236.72it/s]  1%|          | 4155/400000 [00:00<00:47, 8337.40it/s]  1%|▏         | 5009/400000 [00:00<00:47, 8395.68it/s]  1%|▏         | 5870/400000 [00:00<00:46, 8457.25it/s]  2%|▏         | 6739/400000 [00:00<00:46, 8525.01it/s]  2%|▏         | 7602/400000 [00:00<00:45, 8553.76it/s]  2%|▏         | 8465/400000 [00:01<00:45, 8573.67it/s]  2%|▏         | 9320/400000 [00:01<00:45, 8565.15it/s]  3%|▎         | 10178/400000 [00:01<00:45, 8566.72it/s]  3%|▎         | 11044/400000 [00:01<00:45, 8593.24it/s]  3%|▎         | 11909/400000 [00:01<00:45, 8609.59it/s]  3%|▎         | 12775/400000 [00:01<00:44, 8623.13it/s]  3%|▎         | 13634/400000 [00:01<00:44, 8596.07it/s]  4%|▎         | 14499/400000 [00:01<00:44, 8610.54it/s]  4%|▍         | 15359/400000 [00:01<00:47, 8121.30it/s]  4%|▍         | 16220/400000 [00:01<00:46, 8260.81it/s]  4%|▍         | 17084/400000 [00:02<00:45, 8370.98it/s]  4%|▍         | 17925/400000 [00:02<00:45, 8326.85it/s]  5%|▍         | 18797/400000 [00:02<00:45, 8438.94it/s]  5%|▍         | 19658/400000 [00:02<00:44, 8489.39it/s]  5%|▌         | 20509/400000 [00:02<00:45, 8397.20it/s]  5%|▌         | 21376/400000 [00:02<00:44, 8476.38it/s]  6%|▌         | 22225/400000 [00:02<00:44, 8397.08it/s]  6%|▌         | 23066/400000 [00:02<00:44, 8378.92it/s]  6%|▌         | 23905/400000 [00:02<00:45, 8326.67it/s]  6%|▌         | 24772/400000 [00:02<00:44, 8424.74it/s]  6%|▋         | 25637/400000 [00:03<00:44, 8488.52it/s]  7%|▋         | 26495/400000 [00:03<00:43, 8513.14it/s]  7%|▋         | 27362/400000 [00:03<00:43, 8558.75it/s]  7%|▋         | 28231/400000 [00:03<00:43, 8596.03it/s]  7%|▋         | 29091/400000 [00:03<00:44, 8346.04it/s]  7%|▋         | 29958/400000 [00:03<00:43, 8439.34it/s]  8%|▊         | 30808/400000 [00:03<00:43, 8455.26it/s]  8%|▊         | 31655/400000 [00:03<00:43, 8453.25it/s]  8%|▊         | 32526/400000 [00:03<00:43, 8526.31it/s]  8%|▊         | 33394/400000 [00:03<00:42, 8569.58it/s]  9%|▊         | 34252/400000 [00:04<00:42, 8570.90it/s]  9%|▉         | 35110/400000 [00:04<00:42, 8489.27it/s]  9%|▉         | 35960/400000 [00:04<00:43, 8396.90it/s]  9%|▉         | 36824/400000 [00:04<00:42, 8467.86it/s]  9%|▉         | 37672/400000 [00:04<00:43, 8409.52it/s] 10%|▉         | 38514/400000 [00:04<00:43, 8299.03it/s] 10%|▉         | 39372/400000 [00:04<00:43, 8381.21it/s] 10%|█         | 40245/400000 [00:04<00:42, 8481.56it/s] 10%|█         | 41112/400000 [00:04<00:42, 8536.35it/s] 10%|█         | 41983/400000 [00:04<00:41, 8587.61it/s] 11%|█         | 42843/400000 [00:05<00:41, 8588.55it/s] 11%|█         | 43703/400000 [00:05<00:41, 8577.38it/s] 11%|█         | 44570/400000 [00:05<00:41, 8602.89it/s] 11%|█▏        | 45437/400000 [00:05<00:41, 8620.73it/s] 12%|█▏        | 46307/400000 [00:05<00:40, 8642.23it/s] 12%|█▏        | 47177/400000 [00:05<00:40, 8657.17it/s] 12%|█▏        | 48043/400000 [00:05<00:40, 8635.91it/s] 12%|█▏        | 48910/400000 [00:05<00:40, 8644.84it/s] 12%|█▏        | 49775/400000 [00:05<00:41, 8490.11it/s] 13%|█▎        | 50644/400000 [00:05<00:40, 8548.60it/s] 13%|█▎        | 51514/400000 [00:06<00:40, 8590.42it/s] 13%|█▎        | 52375/400000 [00:06<00:40, 8594.89it/s] 13%|█▎        | 53240/400000 [00:06<00:40, 8611.22it/s] 14%|█▎        | 54106/400000 [00:06<00:40, 8625.26it/s] 14%|█▎        | 54979/400000 [00:06<00:39, 8654.35it/s] 14%|█▍        | 55851/400000 [00:06<00:39, 8673.85it/s] 14%|█▍        | 56719/400000 [00:06<00:40, 8567.56it/s] 14%|█▍        | 57577/400000 [00:06<00:39, 8560.62it/s] 15%|█▍        | 58453/400000 [00:06<00:39, 8617.12it/s] 15%|█▍        | 59325/400000 [00:06<00:39, 8644.92it/s] 15%|█▌        | 60190/400000 [00:07<00:39, 8645.02it/s] 15%|█▌        | 61055/400000 [00:07<00:39, 8573.68it/s] 15%|█▌        | 61914/400000 [00:07<00:39, 8577.76it/s] 16%|█▌        | 62784/400000 [00:07<00:39, 8613.37it/s] 16%|█▌        | 63653/400000 [00:07<00:38, 8634.34it/s] 16%|█▌        | 64520/400000 [00:07<00:38, 8642.83it/s] 16%|█▋        | 65385/400000 [00:07<00:38, 8584.77it/s] 17%|█▋        | 66263/400000 [00:07<00:38, 8640.88it/s] 17%|█▋        | 67136/400000 [00:07<00:38, 8667.07it/s] 17%|█▋        | 68003/400000 [00:07<00:38, 8643.37it/s] 17%|█▋        | 68872/400000 [00:08<00:38, 8655.75it/s] 17%|█▋        | 69738/400000 [00:08<00:38, 8613.53it/s] 18%|█▊        | 70605/400000 [00:08<00:38, 8628.26it/s] 18%|█▊        | 71476/400000 [00:08<00:37, 8649.79it/s] 18%|█▊        | 72342/400000 [00:08<00:38, 8605.37it/s] 18%|█▊        | 73203/400000 [00:08<00:37, 8601.83it/s] 19%|█▊        | 74064/400000 [00:08<00:37, 8581.16it/s] 19%|█▊        | 74929/400000 [00:08<00:37, 8599.31it/s] 19%|█▉        | 75793/400000 [00:08<00:37, 8609.43it/s] 19%|█▉        | 76654/400000 [00:08<00:37, 8566.21it/s] 19%|█▉        | 77535/400000 [00:09<00:37, 8636.12it/s] 20%|█▉        | 78399/400000 [00:09<00:37, 8620.87it/s] 20%|█▉        | 79270/400000 [00:09<00:37, 8646.14it/s] 20%|██        | 80137/400000 [00:09<00:36, 8650.57it/s] 20%|██        | 81003/400000 [00:09<00:36, 8649.93it/s] 20%|██        | 81874/400000 [00:09<00:36, 8666.74it/s] 21%|██        | 82741/400000 [00:09<00:36, 8631.34it/s] 21%|██        | 83605/400000 [00:09<00:36, 8622.26it/s] 21%|██        | 84468/400000 [00:09<00:36, 8584.46it/s] 21%|██▏       | 85337/400000 [00:09<00:36, 8615.05it/s] 22%|██▏       | 86199/400000 [00:10<00:36, 8614.37it/s] 22%|██▏       | 87061/400000 [00:10<00:36, 8539.48it/s] 22%|██▏       | 87917/400000 [00:10<00:36, 8544.76it/s] 22%|██▏       | 88779/400000 [00:10<00:36, 8563.15it/s] 22%|██▏       | 89641/400000 [00:10<00:36, 8577.45it/s] 23%|██▎       | 90509/400000 [00:10<00:35, 8605.43it/s] 23%|██▎       | 91370/400000 [00:10<00:35, 8604.13it/s] 23%|██▎       | 92240/400000 [00:10<00:35, 8630.18it/s] 23%|██▎       | 93110/400000 [00:10<00:35, 8649.30it/s] 23%|██▎       | 93977/400000 [00:10<00:35, 8652.81it/s] 24%|██▎       | 94849/400000 [00:11<00:35, 8671.47it/s] 24%|██▍       | 95717/400000 [00:11<00:35, 8656.94it/s] 24%|██▍       | 96587/400000 [00:11<00:35, 8667.36it/s] 24%|██▍       | 97454/400000 [00:11<00:34, 8662.73it/s] 25%|██▍       | 98336/400000 [00:11<00:34, 8706.68it/s] 25%|██▍       | 99207/400000 [00:11<00:34, 8703.85it/s] 25%|██▌       | 100078/400000 [00:11<00:34, 8667.69it/s] 25%|██▌       | 100949/400000 [00:11<00:34, 8678.02it/s] 25%|██▌       | 101817/400000 [00:11<00:34, 8644.55it/s] 26%|██▌       | 102688/400000 [00:11<00:34, 8663.32it/s] 26%|██▌       | 103559/400000 [00:12<00:34, 8675.44it/s] 26%|██▌       | 104427/400000 [00:12<00:34, 8640.23it/s] 26%|██▋       | 105295/400000 [00:12<00:34, 8649.24it/s] 27%|██▋       | 106165/400000 [00:12<00:33, 8664.39it/s] 27%|██▋       | 107032/400000 [00:12<00:34, 8549.11it/s] 27%|██▋       | 107904/400000 [00:12<00:33, 8598.59it/s] 27%|██▋       | 108767/400000 [00:12<00:33, 8606.93it/s] 27%|██▋       | 109644/400000 [00:12<00:33, 8654.88it/s] 28%|██▊       | 110516/400000 [00:12<00:33, 8673.30it/s] 28%|██▊       | 111384/400000 [00:13<00:33, 8671.27it/s] 28%|██▊       | 112257/400000 [00:13<00:33, 8686.76it/s] 28%|██▊       | 113126/400000 [00:13<00:33, 8653.40it/s] 28%|██▊       | 113992/400000 [00:13<00:33, 8646.03it/s] 29%|██▊       | 114858/400000 [00:13<00:32, 8648.89it/s] 29%|██▉       | 115728/400000 [00:13<00:32, 8663.35it/s] 29%|██▉       | 116596/400000 [00:13<00:32, 8666.53it/s] 29%|██▉       | 117463/400000 [00:13<00:32, 8658.55it/s] 30%|██▉       | 118342/400000 [00:13<00:32, 8696.00it/s] 30%|██▉       | 119221/400000 [00:13<00:32, 8721.96it/s] 30%|███       | 120094/400000 [00:14<00:32, 8719.13it/s] 30%|███       | 120978/400000 [00:14<00:31, 8752.50it/s] 30%|███       | 121854/400000 [00:14<00:31, 8720.15it/s] 31%|███       | 122727/400000 [00:14<00:31, 8710.73it/s] 31%|███       | 123599/400000 [00:14<00:31, 8711.96it/s] 31%|███       | 124471/400000 [00:14<00:31, 8697.36it/s] 31%|███▏      | 125347/400000 [00:14<00:31, 8715.02it/s] 32%|███▏      | 126219/400000 [00:14<00:31, 8688.11it/s] 32%|███▏      | 127088/400000 [00:14<00:31, 8601.13it/s] 32%|███▏      | 127949/400000 [00:14<00:31, 8539.17it/s] 32%|███▏      | 128812/400000 [00:15<00:31, 8565.22it/s] 32%|███▏      | 129680/400000 [00:15<00:31, 8598.31it/s] 33%|███▎      | 130541/400000 [00:15<00:31, 8583.27it/s] 33%|███▎      | 131409/400000 [00:15<00:31, 8609.35it/s] 33%|███▎      | 132281/400000 [00:15<00:30, 8642.01it/s] 33%|███▎      | 133152/400000 [00:15<00:30, 8661.30it/s] 34%|███▎      | 134031/400000 [00:15<00:30, 8697.20it/s] 34%|███▎      | 134901/400000 [00:15<00:30, 8672.82it/s] 34%|███▍      | 135772/400000 [00:15<00:30, 8682.48it/s] 34%|███▍      | 136641/400000 [00:15<00:30, 8681.94it/s] 34%|███▍      | 137510/400000 [00:16<00:30, 8638.04it/s] 35%|███▍      | 138380/400000 [00:16<00:30, 8653.77it/s] 35%|███▍      | 139246/400000 [00:16<00:30, 8633.40it/s] 35%|███▌      | 140110/400000 [00:16<00:30, 8613.83it/s] 35%|███▌      | 140974/400000 [00:16<00:30, 8620.94it/s] 35%|███▌      | 141843/400000 [00:16<00:29, 8640.00it/s] 36%|███▌      | 142713/400000 [00:16<00:29, 8655.89it/s] 36%|███▌      | 143579/400000 [00:16<00:29, 8653.60it/s] 36%|███▌      | 144445/400000 [00:16<00:29, 8654.43it/s] 36%|███▋      | 145311/400000 [00:16<00:29, 8650.90it/s] 37%|███▋      | 146180/400000 [00:17<00:29, 8660.03it/s] 37%|███▋      | 147052/400000 [00:17<00:29, 8674.96it/s] 37%|███▋      | 147920/400000 [00:17<00:29, 8557.16it/s] 37%|███▋      | 148777/400000 [00:17<00:29, 8512.17it/s] 37%|███▋      | 149641/400000 [00:17<00:29, 8549.42it/s] 38%|███▊      | 150502/400000 [00:17<00:29, 8565.31it/s] 38%|███▊      | 151359/400000 [00:17<00:29, 8461.94it/s] 38%|███▊      | 152232/400000 [00:17<00:29, 8537.76it/s] 38%|███▊      | 153087/400000 [00:17<00:29, 8462.07it/s] 38%|███▊      | 153954/400000 [00:17<00:28, 8521.38it/s] 39%|███▊      | 154807/400000 [00:18<00:28, 8503.01it/s] 39%|███▉      | 155667/400000 [00:18<00:28, 8527.33it/s] 39%|███▉      | 156520/400000 [00:18<00:28, 8506.22it/s] 39%|███▉      | 157387/400000 [00:18<00:28, 8553.28it/s] 40%|███▉      | 158255/400000 [00:18<00:28, 8589.96it/s] 40%|███▉      | 159130/400000 [00:18<00:27, 8635.85it/s] 40%|███▉      | 159994/400000 [00:18<00:27, 8630.74it/s] 40%|████      | 160858/400000 [00:18<00:27, 8610.03it/s] 40%|████      | 161723/400000 [00:18<00:27, 8620.62it/s] 41%|████      | 162586/400000 [00:18<00:27, 8623.10it/s] 41%|████      | 163459/400000 [00:19<00:27, 8652.76it/s] 41%|████      | 164325/400000 [00:19<00:27, 8637.08it/s] 41%|████▏     | 165189/400000 [00:19<00:27, 8609.88it/s] 42%|████▏     | 166051/400000 [00:19<00:27, 8579.80it/s] 42%|████▏     | 166912/400000 [00:19<00:27, 8588.14it/s] 42%|████▏     | 167780/400000 [00:19<00:26, 8613.95it/s] 42%|████▏     | 168646/400000 [00:19<00:26, 8626.40it/s] 42%|████▏     | 169511/400000 [00:19<00:26, 8631.05it/s] 43%|████▎     | 170375/400000 [00:19<00:26, 8603.67it/s] 43%|████▎     | 171252/400000 [00:19<00:26, 8650.43it/s] 43%|████▎     | 172127/400000 [00:20<00:26, 8678.88it/s] 43%|████▎     | 173015/400000 [00:20<00:25, 8737.70it/s] 43%|████▎     | 173889/400000 [00:20<00:26, 8687.12it/s] 44%|████▎     | 174758/400000 [00:20<00:26, 8644.73it/s] 44%|████▍     | 175626/400000 [00:20<00:25, 8652.86it/s] 44%|████▍     | 176499/400000 [00:20<00:25, 8675.56it/s] 44%|████▍     | 177374/400000 [00:20<00:25, 8695.68it/s] 45%|████▍     | 178247/400000 [00:20<00:25, 8705.13it/s] 45%|████▍     | 179118/400000 [00:20<00:25, 8653.83it/s] 45%|████▍     | 179984/400000 [00:20<00:25, 8583.73it/s] 45%|████▌     | 180851/400000 [00:21<00:25, 8608.30it/s] 45%|████▌     | 181712/400000 [00:21<00:25, 8599.12it/s] 46%|████▌     | 182573/400000 [00:21<00:25, 8589.45it/s] 46%|████▌     | 183433/400000 [00:21<00:25, 8566.50it/s] 46%|████▌     | 184290/400000 [00:21<00:25, 8473.82it/s] 46%|████▋     | 185156/400000 [00:21<00:25, 8526.43it/s] 47%|████▋     | 186021/400000 [00:21<00:24, 8562.96it/s] 47%|████▋     | 186890/400000 [00:21<00:24, 8599.97it/s] 47%|████▋     | 187751/400000 [00:21<00:24, 8593.80it/s] 47%|████▋     | 188616/400000 [00:21<00:24, 8609.90it/s] 47%|████▋     | 189480/400000 [00:22<00:24, 8617.58it/s] 48%|████▊     | 190350/400000 [00:22<00:24, 8640.42it/s] 48%|████▊     | 191233/400000 [00:22<00:24, 8694.88it/s] 48%|████▊     | 192103/400000 [00:22<00:24, 8639.04it/s] 48%|████▊     | 192979/400000 [00:22<00:23, 8674.41it/s] 48%|████▊     | 193850/400000 [00:22<00:23, 8684.32it/s] 49%|████▊     | 194722/400000 [00:22<00:23, 8693.83it/s] 49%|████▉     | 195597/400000 [00:22<00:23, 8708.56it/s] 49%|████▉     | 196468/400000 [00:22<00:23, 8680.79it/s] 49%|████▉     | 197337/400000 [00:22<00:23, 8680.36it/s] 50%|████▉     | 198206/400000 [00:23<00:23, 8632.87it/s] 50%|████▉     | 199073/400000 [00:23<00:23, 8643.05it/s] 50%|████▉     | 199948/400000 [00:23<00:23, 8672.91it/s] 50%|█████     | 200816/400000 [00:23<00:23, 8644.10it/s] 50%|█████     | 201687/400000 [00:23<00:22, 8662.51it/s] 51%|█████     | 202560/400000 [00:23<00:22, 8679.58it/s] 51%|█████     | 203433/400000 [00:23<00:22, 8693.17it/s] 51%|█████     | 204303/400000 [00:23<00:22, 8685.89it/s] 51%|█████▏    | 205172/400000 [00:23<00:22, 8644.21it/s] 52%|█████▏    | 206037/400000 [00:23<00:22, 8645.88it/s] 52%|█████▏    | 206907/400000 [00:24<00:22, 8660.36it/s] 52%|█████▏    | 207782/400000 [00:24<00:22, 8685.24it/s] 52%|█████▏    | 208651/400000 [00:24<00:22, 8610.63it/s] 52%|█████▏    | 209513/400000 [00:24<00:22, 8602.13it/s] 53%|█████▎    | 210379/400000 [00:24<00:22, 8616.96it/s] 53%|█████▎    | 211250/400000 [00:24<00:21, 8641.93it/s] 53%|█████▎    | 212116/400000 [00:24<00:21, 8645.62it/s] 53%|█████▎    | 212986/400000 [00:24<00:21, 8660.10it/s] 53%|█████▎    | 213859/400000 [00:24<00:21, 8679.97it/s] 54%|█████▎    | 214728/400000 [00:24<00:21, 8658.57it/s] 54%|█████▍    | 215597/400000 [00:25<00:21, 8665.95it/s] 54%|█████▍    | 216464/400000 [00:25<00:21, 8641.78it/s] 54%|█████▍    | 217334/400000 [00:25<00:21, 8657.35it/s] 55%|█████▍    | 218200/400000 [00:25<00:21, 8611.29it/s] 55%|█████▍    | 219064/400000 [00:25<00:20, 8618.87it/s] 55%|█████▍    | 219935/400000 [00:25<00:20, 8645.38it/s] 55%|█████▌    | 220809/400000 [00:25<00:20, 8672.63it/s] 55%|█████▌    | 221686/400000 [00:25<00:20, 8700.29it/s] 56%|█████▌    | 222560/400000 [00:25<00:20, 8710.94it/s] 56%|█████▌    | 223432/400000 [00:25<00:20, 8687.30it/s] 56%|█████▌    | 224302/400000 [00:26<00:20, 8689.13it/s] 56%|█████▋    | 225171/400000 [00:26<00:20, 8686.98it/s] 57%|█████▋    | 226040/400000 [00:26<00:20, 8684.09it/s] 57%|█████▋    | 226909/400000 [00:26<00:20, 8425.40it/s] 57%|█████▋    | 227780/400000 [00:26<00:20, 8508.46it/s] 57%|█████▋    | 228653/400000 [00:26<00:19, 8572.64it/s] 57%|█████▋    | 229520/400000 [00:26<00:19, 8600.27it/s] 58%|█████▊    | 230390/400000 [00:26<00:19, 8628.48it/s] 58%|█████▊    | 231264/400000 [00:26<00:19, 8659.97it/s] 58%|█████▊    | 232131/400000 [00:26<00:19, 8626.62it/s] 58%|█████▊    | 232998/400000 [00:27<00:19, 8639.41it/s] 58%|█████▊    | 233867/400000 [00:27<00:19, 8651.94it/s] 59%|█████▊    | 234733/400000 [00:27<00:19, 8403.77it/s] 59%|█████▉    | 235598/400000 [00:27<00:19, 8473.99it/s] 59%|█████▉    | 236447/400000 [00:27<00:19, 8476.87it/s] 59%|█████▉    | 237315/400000 [00:27<00:19, 8536.56it/s] 60%|█████▉    | 238194/400000 [00:27<00:18, 8610.83it/s] 60%|█████▉    | 239067/400000 [00:27<00:18, 8643.75it/s] 60%|█████▉    | 239932/400000 [00:27<00:18, 8591.74it/s] 60%|██████    | 240792/400000 [00:28<00:18, 8585.40it/s] 60%|██████    | 241656/400000 [00:28<00:18, 8599.33it/s] 61%|██████    | 242527/400000 [00:28<00:18, 8629.99it/s] 61%|██████    | 243391/400000 [00:28<00:18, 8447.61it/s] 61%|██████    | 244261/400000 [00:28<00:18, 8520.55it/s] 61%|██████▏   | 245121/400000 [00:28<00:18, 8541.97it/s] 61%|██████▏   | 245986/400000 [00:28<00:17, 8571.80it/s] 62%|██████▏   | 246858/400000 [00:28<00:17, 8612.90it/s] 62%|██████▏   | 247725/400000 [00:28<00:17, 8629.75it/s] 62%|██████▏   | 248593/400000 [00:28<00:17, 8642.97it/s] 62%|██████▏   | 249458/400000 [00:29<00:17, 8633.72it/s] 63%|██████▎   | 250322/400000 [00:29<00:17, 8626.36it/s] 63%|██████▎   | 251194/400000 [00:29<00:17, 8651.25it/s] 63%|██████▎   | 252063/400000 [00:29<00:17, 8661.27it/s] 63%|██████▎   | 252933/400000 [00:29<00:16, 8670.50it/s] 63%|██████▎   | 253801/400000 [00:29<00:16, 8656.96it/s] 64%|██████▎   | 254667/400000 [00:29<00:16, 8656.87it/s] 64%|██████▍   | 255533/400000 [00:29<00:16, 8638.67it/s] 64%|██████▍   | 256407/400000 [00:29<00:16, 8668.62it/s] 64%|██████▍   | 257283/400000 [00:29<00:16, 8692.83it/s] 65%|██████▍   | 258153/400000 [00:30<00:16, 8665.98it/s] 65%|██████▍   | 259020/400000 [00:30<00:16, 8659.17it/s] 65%|██████▍   | 259886/400000 [00:30<00:16, 8582.13it/s] 65%|██████▌   | 260755/400000 [00:30<00:16, 8613.72it/s] 65%|██████▌   | 261620/400000 [00:30<00:16, 8622.61it/s] 66%|██████▌   | 262483/400000 [00:30<00:16, 8593.10it/s] 66%|██████▌   | 263360/400000 [00:30<00:15, 8642.41it/s] 66%|██████▌   | 264234/400000 [00:30<00:15, 8670.47it/s] 66%|██████▋   | 265102/400000 [00:30<00:15, 8658.26it/s] 66%|██████▋   | 265968/400000 [00:30<00:15, 8647.81it/s] 67%|██████▋   | 266833/400000 [00:31<00:15, 8607.20it/s] 67%|██████▋   | 267697/400000 [00:31<00:15, 8614.54it/s] 67%|██████▋   | 268573/400000 [00:31<00:15, 8656.39it/s] 67%|██████▋   | 269445/400000 [00:31<00:15, 8674.34it/s] 68%|██████▊   | 270314/400000 [00:31<00:14, 8676.84it/s] 68%|██████▊   | 271182/400000 [00:31<00:14, 8640.58it/s] 68%|██████▊   | 272047/400000 [00:31<00:14, 8633.58it/s] 68%|██████▊   | 272915/400000 [00:31<00:14, 8647.37it/s] 68%|██████▊   | 273787/400000 [00:31<00:14, 8667.20it/s] 69%|██████▊   | 274665/400000 [00:31<00:14, 8698.98it/s] 69%|██████▉   | 275535/400000 [00:32<00:14, 8666.88it/s] 69%|██████▉   | 276402/400000 [00:32<00:14, 8662.21it/s] 69%|██████▉   | 277269/400000 [00:32<00:14, 8663.94it/s] 70%|██████▉   | 278136/400000 [00:32<00:14, 8548.80it/s] 70%|██████▉   | 279003/400000 [00:32<00:14, 8584.70it/s] 70%|██████▉   | 279868/400000 [00:32<00:13, 8602.45it/s] 70%|███████   | 280736/400000 [00:32<00:13, 8623.75it/s] 70%|███████   | 281599/400000 [00:32<00:13, 8611.13it/s] 71%|███████   | 282461/400000 [00:32<00:13, 8571.31it/s] 71%|███████   | 283332/400000 [00:32<00:13, 8611.87it/s] 71%|███████   | 284199/400000 [00:33<00:13, 8627.79it/s] 71%|███████▏  | 285064/400000 [00:33<00:13, 8633.62it/s] 71%|███████▏  | 285943/400000 [00:33<00:13, 8679.56it/s] 72%|███████▏  | 286815/400000 [00:33<00:13, 8690.70it/s] 72%|███████▏  | 287686/400000 [00:33<00:12, 8695.56it/s] 72%|███████▏  | 288556/400000 [00:33<00:12, 8607.59it/s] 72%|███████▏  | 289425/400000 [00:33<00:12, 8631.86it/s] 73%|███████▎  | 290295/400000 [00:33<00:12, 8650.17it/s] 73%|███████▎  | 291165/400000 [00:33<00:12, 8662.55it/s] 73%|███████▎  | 292032/400000 [00:33<00:12, 8664.33it/s] 73%|███████▎  | 292899/400000 [00:34<00:12, 8611.48it/s] 73%|███████▎  | 293767/400000 [00:34<00:12, 8629.14it/s] 74%|███████▎  | 294636/400000 [00:34<00:12, 8644.42it/s] 74%|███████▍  | 295501/400000 [00:34<00:12, 8643.92it/s] 74%|███████▍  | 296369/400000 [00:34<00:11, 8654.10it/s] 74%|███████▍  | 297235/400000 [00:34<00:11, 8617.36it/s] 75%|███████▍  | 298097/400000 [00:34<00:11, 8612.20it/s] 75%|███████▍  | 298963/400000 [00:34<00:11, 8625.75it/s] 75%|███████▍  | 299829/400000 [00:34<00:11, 8635.03it/s] 75%|███████▌  | 300695/400000 [00:34<00:11, 8641.31it/s] 75%|███████▌  | 301560/400000 [00:35<00:11, 8570.19it/s] 76%|███████▌  | 302429/400000 [00:35<00:11, 8602.94it/s] 76%|███████▌  | 303290/400000 [00:35<00:11, 8595.62it/s] 76%|███████▌  | 304157/400000 [00:35<00:11, 8616.83it/s] 76%|███████▋  | 305023/400000 [00:35<00:11, 8629.66it/s] 76%|███████▋  | 305887/400000 [00:35<00:10, 8596.80it/s] 77%|███████▋  | 306750/400000 [00:35<00:10, 8606.65it/s] 77%|███████▋  | 307617/400000 [00:35<00:10, 8623.25it/s] 77%|███████▋  | 308483/400000 [00:35<00:10, 8631.75it/s] 77%|███████▋  | 309347/400000 [00:35<00:10, 8618.04it/s] 78%|███████▊  | 310209/400000 [00:36<00:10, 8605.57it/s] 78%|███████▊  | 311074/400000 [00:36<00:10, 8617.98it/s] 78%|███████▊  | 311941/400000 [00:36<00:10, 8631.17it/s] 78%|███████▊  | 312807/400000 [00:36<00:10, 8638.13it/s] 78%|███████▊  | 313673/400000 [00:36<00:09, 8642.06it/s] 79%|███████▊  | 314538/400000 [00:36<00:09, 8619.00it/s] 79%|███████▉  | 315411/400000 [00:36<00:09, 8650.75it/s] 79%|███████▉  | 316277/400000 [00:36<00:09, 8648.36it/s] 79%|███████▉  | 317145/400000 [00:36<00:09, 8655.00it/s] 80%|███████▉  | 318014/400000 [00:36<00:09, 8664.70it/s] 80%|███████▉  | 318881/400000 [00:37<00:09, 8631.56it/s] 80%|███████▉  | 319750/400000 [00:37<00:09, 8646.08it/s] 80%|████████  | 320622/400000 [00:37<00:09, 8667.80it/s] 80%|████████  | 321489/400000 [00:37<00:09, 8653.88it/s] 81%|████████  | 322355/400000 [00:37<00:09, 8624.20it/s] 81%|████████  | 323218/400000 [00:37<00:08, 8603.65it/s] 81%|████████  | 324079/400000 [00:37<00:08, 8602.87it/s] 81%|████████  | 324945/400000 [00:37<00:08, 8617.08it/s] 81%|████████▏ | 325815/400000 [00:37<00:08, 8639.81it/s] 82%|████████▏ | 326680/400000 [00:37<00:08, 8630.54it/s] 82%|████████▏ | 327544/400000 [00:38<00:08, 8558.64it/s] 82%|████████▏ | 328409/400000 [00:38<00:08, 8584.75it/s] 82%|████████▏ | 329268/400000 [00:38<00:08, 8554.97it/s] 83%|████████▎ | 330131/400000 [00:38<00:08, 8577.06it/s] 83%|████████▎ | 331001/400000 [00:38<00:08, 8611.07it/s] 83%|████████▎ | 331863/400000 [00:38<00:07, 8597.86it/s] 83%|████████▎ | 332730/400000 [00:38<00:07, 8617.26it/s] 83%|████████▎ | 333600/400000 [00:38<00:07, 8639.00it/s] 84%|████████▎ | 334465/400000 [00:38<00:07, 8641.64it/s] 84%|████████▍ | 335330/400000 [00:38<00:07, 8621.07it/s] 84%|████████▍ | 336193/400000 [00:39<00:07, 8580.41it/s] 84%|████████▍ | 337057/400000 [00:39<00:07, 8597.70it/s] 84%|████████▍ | 337925/400000 [00:39<00:07, 8621.24it/s] 85%|████████▍ | 338788/400000 [00:39<00:07, 8621.67it/s] 85%|████████▍ | 339651/400000 [00:39<00:07, 8619.75it/s] 85%|████████▌ | 340514/400000 [00:39<00:06, 8586.70it/s] 85%|████████▌ | 341388/400000 [00:39<00:06, 8629.25it/s] 86%|████████▌ | 342258/400000 [00:39<00:06, 8647.56it/s] 86%|████████▌ | 343125/400000 [00:39<00:06, 8653.00it/s] 86%|████████▌ | 343991/400000 [00:39<00:06, 8622.21it/s] 86%|████████▌ | 344854/400000 [00:40<00:06, 8574.71it/s] 86%|████████▋ | 345719/400000 [00:40<00:06, 8594.27it/s] 87%|████████▋ | 346579/400000 [00:40<00:06, 8584.86it/s] 87%|████████▋ | 347446/400000 [00:40<00:06, 8608.16it/s] 87%|████████▋ | 348307/400000 [00:40<00:06, 8604.13it/s] 87%|████████▋ | 349168/400000 [00:40<00:05, 8552.38it/s] 88%|████████▊ | 350024/400000 [00:40<00:05, 8543.79it/s] 88%|████████▊ | 350879/400000 [00:40<00:05, 8532.43it/s] 88%|████████▊ | 351743/400000 [00:40<00:05, 8563.76it/s] 88%|████████▊ | 352609/400000 [00:40<00:05, 8591.75it/s] 88%|████████▊ | 353469/400000 [00:41<00:05, 8552.76it/s] 89%|████████▊ | 354338/400000 [00:41<00:05, 8592.89it/s] 89%|████████▉ | 355203/400000 [00:41<00:05, 8608.04it/s] 89%|████████▉ | 356070/400000 [00:41<00:05, 8626.36it/s] 89%|████████▉ | 356933/400000 [00:41<00:04, 8627.13it/s] 89%|████████▉ | 357796/400000 [00:41<00:04, 8614.27it/s] 90%|████████▉ | 358665/400000 [00:41<00:04, 8634.86it/s] 90%|████████▉ | 359536/400000 [00:41<00:04, 8657.18it/s] 90%|█████████ | 360402/400000 [00:41<00:04, 8607.57it/s] 90%|█████████ | 361267/400000 [00:41<00:04, 8619.01it/s] 91%|█████████ | 362129/400000 [00:42<00:04, 8582.63it/s] 91%|█████████ | 363004/400000 [00:42<00:04, 8629.53it/s] 91%|█████████ | 363875/400000 [00:42<00:04, 8652.19it/s] 91%|█████████ | 364749/400000 [00:42<00:04, 8677.60it/s] 91%|█████████▏| 365618/400000 [00:42<00:03, 8678.37it/s] 92%|█████████▏| 366486/400000 [00:42<00:03, 8641.80it/s] 92%|█████████▏| 367351/400000 [00:42<00:03, 8637.76it/s] 92%|█████████▏| 368220/400000 [00:42<00:03, 8652.85it/s] 92%|█████████▏| 369088/400000 [00:42<00:03, 8658.32it/s] 92%|█████████▏| 369958/400000 [00:42<00:03, 8668.01it/s] 93%|█████████▎| 370825/400000 [00:43<00:03, 8653.68it/s] 93%|█████████▎| 371707/400000 [00:43<00:03, 8701.60it/s] 93%|█████████▎| 372578/400000 [00:43<00:03, 8704.03it/s] 93%|█████████▎| 373449/400000 [00:43<00:03, 8697.03it/s] 94%|█████████▎| 374319/400000 [00:43<00:02, 8677.33it/s] 94%|█████████▍| 375187/400000 [00:43<00:02, 8650.70it/s] 94%|█████████▍| 376053/400000 [00:43<00:02, 8640.30it/s] 94%|█████████▍| 376918/400000 [00:43<00:02, 8608.14it/s] 94%|█████████▍| 377781/400000 [00:43<00:02, 8613.82it/s] 95%|█████████▍| 378651/400000 [00:43<00:02, 8638.92it/s] 95%|█████████▍| 379515/400000 [00:44<00:02, 8633.54it/s] 95%|█████████▌| 380382/400000 [00:44<00:02, 8643.82it/s] 95%|█████████▌| 381247/400000 [00:44<00:02, 8626.49it/s] 96%|█████████▌| 382111/400000 [00:44<00:02, 8629.02it/s] 96%|█████████▌| 382980/400000 [00:44<00:01, 8644.80it/s] 96%|█████████▌| 383848/400000 [00:44<00:01, 8654.52it/s] 96%|█████████▌| 384716/400000 [00:44<00:01, 8659.54it/s] 96%|█████████▋| 385583/400000 [00:44<00:01, 8660.85it/s] 97%|█████████▋| 386450/400000 [00:44<00:01, 8648.87it/s] 97%|█████████▋| 387317/400000 [00:44<00:01, 8654.89it/s] 97%|█████████▋| 388183/400000 [00:45<00:01, 8655.33it/s] 97%|█████████▋| 389049/400000 [00:45<00:01, 8606.79it/s] 97%|█████████▋| 389917/400000 [00:45<00:01, 8626.22it/s] 98%|█████████▊| 390786/400000 [00:45<00:01, 8643.56it/s] 98%|█████████▊| 391651/400000 [00:45<00:00, 8630.95it/s] 98%|█████████▊| 392521/400000 [00:45<00:00, 8649.56it/s] 98%|█████████▊| 393386/400000 [00:45<00:00, 8615.21it/s] 99%|█████████▊| 394252/400000 [00:45<00:00, 8627.04it/s] 99%|█████████▉| 395128/400000 [00:45<00:00, 8663.59it/s] 99%|█████████▉| 395997/400000 [00:45<00:00, 8668.73it/s] 99%|█████████▉| 396864/400000 [00:46<00:00, 8662.84it/s] 99%|█████████▉| 397731/400000 [00:46<00:00, 8638.74it/s]100%|█████████▉| 398595/400000 [00:46<00:00, 8614.08it/s]100%|█████████▉| 399457/400000 [00:46<00:00, 8607.72it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8610.13it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd8a5f89b70> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010875929152334539 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.011072279816885855 	 Accuracy: 54

  model saves at 54% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15881 out of table with 15818 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15881 out of table with 15818 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

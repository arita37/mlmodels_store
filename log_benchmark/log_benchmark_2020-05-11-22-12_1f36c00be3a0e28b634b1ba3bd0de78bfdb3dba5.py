
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7af2640f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 22:13:16.592708
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 22:13:16.597308
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 22:13:16.601025
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 22:13:16.605034
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7afe404400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354687.9375
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 240849.5938
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 135375.3906
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 69424.2422
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 37025.7930
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 21737.5098
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 13946.7871
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 9729.0850
Epoch 9/10

1/1 [==============================] - 0s 110ms/step - loss: 7139.0264
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 5551.0630

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.36795235e-01 -4.49908912e-01 -8.70175660e-01 -9.44160819e-02
   9.80696797e-01  2.26835936e-01  1.53623685e-01 -2.30144918e-01
   2.65368521e-01 -8.75817120e-01 -4.32401597e-01  7.04560995e-01
  -6.56316161e-01 -9.93274093e-01 -1.17763352e+00  1.10566449e+00
   1.40682089e+00  7.01842308e-01  5.54782808e-01  1.53408086e+00
  -2.79021788e+00  7.98001587e-01  3.48679185e-01  1.35467482e+00
   1.02376354e+00 -9.93092835e-01  1.17609429e+00 -1.56873894e+00
  -3.94625545e-01 -6.83511674e-01 -1.71681619e+00 -3.17448735e-01
   4.63146597e-01  4.59815621e-01  7.83180654e-01 -5.45748830e-01
   2.27492154e-01  3.58444452e-03  5.84996939e-01  5.57160616e-01
   1.69560361e+00 -3.74655724e-02  1.84626460e-01 -3.69400978e-01
  -2.07849145e-01  4.61718857e-01  1.08800328e+00  9.33697104e-01
   3.17510366e-01 -1.60558432e-01 -2.87994951e-01 -2.76498747e+00
  -7.00772524e-01 -2.92722970e-01  2.82000542e-01 -1.60528743e+00
  -1.59133005e+00  1.15800333e+00  1.35714620e-01 -9.57327127e-01
  -2.09381253e-01 -8.62124190e-02 -2.19412351e+00 -2.10682201e+00
   1.01306486e+00  1.51507044e+00  5.30780077e-01 -1.51830256e-01
  -3.80994618e-01  1.62464011e+00  1.33735633e+00  1.45301208e-01
  -6.82344317e-01 -8.49581718e-01  1.54682052e+00 -1.76544443e-01
   7.31562972e-01  1.93151653e+00 -8.43433142e-01  1.01587147e-01
   6.84963048e-01  1.27829599e+00 -5.03044188e-01  2.47523874e-01
  -7.89620519e-01  6.73392713e-01 -2.68652797e-01 -2.33764529e-01
   1.38755709e-01 -3.75832230e-01  8.72290015e-01 -1.16290796e+00
  -1.84306026e-01 -2.22331810e+00 -5.01919746e-01 -4.71762061e-01
  -1.65011406e+00 -5.58721423e-01  1.07752156e+00 -1.58973646e+00
  -2.16967762e-01  5.14887571e-02 -1.51652312e+00 -3.54210198e-01
  -3.88417333e-01 -2.03615874e-01 -7.46546566e-01  8.50501895e-01
   1.31310248e+00 -5.39703488e-01 -1.63285244e+00  5.14523506e-01
  -1.45207238e+00  6.53477311e-01 -5.11774123e-02  8.23844314e-01
  -8.42475891e-03  8.52935433e-01  9.53460693e-01  1.88644409e-01
   4.30489033e-01  1.10864210e+01  9.37775135e+00  1.09106073e+01
   9.03781796e+00  1.02253017e+01  1.09901590e+01  8.31738853e+00
   9.94075680e+00  7.14870405e+00  8.71496105e+00  7.89189625e+00
   9.95870686e+00  8.61598015e+00  9.48411560e+00  8.13797855e+00
   9.84377384e+00  9.08025551e+00  9.14617634e+00  8.61334896e+00
   8.32713985e+00  9.88613892e+00  8.20788956e+00  9.41643047e+00
   7.23555470e+00  7.96558142e+00  1.13757858e+01  9.43247986e+00
   8.62481499e+00  7.73015785e+00  1.06249819e+01  8.82500648e+00
   8.49655056e+00  8.97166061e+00  8.46663666e+00  7.12444782e+00
   9.22582054e+00  8.22999001e+00  8.26704788e+00  8.51542091e+00
   1.02405796e+01  8.34707165e+00  1.05820370e+01  9.51436806e+00
   9.96843433e+00  1.00796995e+01  1.08617468e+01  1.05608187e+01
   9.47925758e+00  8.79172993e+00  9.79829121e+00  8.78651714e+00
   9.50551701e+00  7.74256897e+00  8.38636684e+00  8.69032001e+00
   7.01441336e+00  1.00664539e+01  9.35399246e+00  9.50531960e+00
   2.43216085e+00  1.30490303e+00  1.68491054e+00  1.47926879e+00
   2.57950735e+00  2.84572124e-01  1.30649686e+00  1.81040812e+00
   1.47169423e+00  6.06909275e-01  2.48454118e+00  1.99933863e+00
   1.23204207e+00  2.20260024e+00  8.66195738e-01  4.18213129e-01
   1.38471317e+00  3.20219576e-01  1.49847066e+00  1.79886103e-01
   3.44038582e+00  1.35233247e+00  2.43141174e-01  1.83651924e+00
   2.32437682e+00  1.63118029e+00  3.52912092e+00  7.55624592e-01
   3.28154206e-01  1.20594680e+00  5.24700344e-01  1.23650408e+00
   2.85074949e-01  4.35642600e-01  1.08794045e+00  1.56144786e+00
   1.72485173e-01  1.82315469e-01  6.18349671e-01  4.11463499e-01
   1.16825426e+00  2.37065792e-01  7.41962910e-01  2.23087883e+00
   7.42959559e-01  3.38461256e+00  4.94263291e-01  1.30440807e+00
   1.38210928e+00  1.13760448e+00  8.52034807e-01  2.76115656e+00
   1.12845147e+00  1.39372361e+00  5.30591965e-01  1.26888120e+00
   1.78085709e+00  2.08958030e-01  1.67242050e-01  7.18252778e-01
   7.30624437e-01  2.67693043e-01  2.84959698e+00  4.45134103e-01
   5.12245476e-01  3.08359504e-01  1.68112302e+00  8.39383721e-01
   6.97511792e-01  1.41782880e-01  2.07224655e+00  2.25014091e-01
   2.07397747e+00  1.64158010e+00  1.71429873e-01  2.20566750e-01
   2.10185552e+00  3.83457422e-01  2.33717775e+00  2.44651198e-01
   1.86057067e+00  4.73639965e-01  1.81358922e+00  1.28817081e+00
   8.19818914e-01  1.39963746e+00  9.53759968e-01  3.62239480e-01
   9.12335336e-01  2.45450115e+00  9.25306261e-01  2.06212997e+00
   2.49578297e-01  2.47848809e-01  7.21236348e-01  1.43237591e-01
   1.51914406e+00  4.72427964e-01  8.65216851e-01  3.26676726e-01
   2.94376433e-01  8.60904932e-01  1.23853719e+00  6.83307827e-01
   3.18345070e+00  1.54869270e+00  7.46611238e-01  1.62749612e+00
   2.47804880e+00  5.76497972e-01  2.19837403e+00  2.07492590e+00
   2.20121002e+00  1.74481535e+00  2.30127740e+00  1.23976648e+00
   1.16215658e+00  2.74838746e-01  6.64650083e-01  3.22851276e+00
   9.42709446e-02  9.22873116e+00  9.59290981e+00  1.11963158e+01
   1.17163115e+01  1.01070833e+01  9.32453346e+00  9.71390438e+00
   1.08876991e+01  9.33778667e+00  8.59525871e+00  7.87933683e+00
   1.09206057e+01  1.01764488e+01  7.55925512e+00  8.09270954e+00
   1.01856422e+01  6.89902020e+00  9.18296242e+00  1.07906933e+01
   9.94798470e+00  8.76200104e+00  9.45022106e+00  7.72302437e+00
   1.16428490e+01  7.11186218e+00  7.97753143e+00  7.65337610e+00
   7.01734161e+00  9.66712761e+00  9.41363621e+00  9.09703159e+00
   9.71177959e+00  7.18780899e+00  9.59499836e+00  1.06537151e+01
   7.65261364e+00  1.01955414e+01  8.29017448e+00  8.16639519e+00
   7.59453440e+00  1.03622723e+01  1.11678410e+01  9.16817570e+00
   9.41189194e+00  8.97901821e+00  1.03970299e+01  1.13115911e+01
   8.70118713e+00  7.95709991e+00  1.01762400e+01  7.98804617e+00
   9.61881733e+00  9.56810856e+00  1.05661182e+01  8.17219543e+00
   9.72905064e+00  7.94638395e+00  1.08678064e+01  8.29649544e+00
  -1.55297565e+01 -5.17767715e+00  1.97991443e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 22:13:26.381532
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5355
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 22:13:26.386401
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8586.25
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 22:13:26.390241
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7314
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 22:13:26.394410
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -767.959
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140165687384944
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140164997041568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140164997042072
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140164996620752
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140164996621256
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140164996621760

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7afa287eb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.514754
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.485949
grad_step = 000002, loss = 0.467950
grad_step = 000003, loss = 0.449609
grad_step = 000004, loss = 0.429901
grad_step = 000005, loss = 0.409008
grad_step = 000006, loss = 0.387111
grad_step = 000007, loss = 0.370371
grad_step = 000008, loss = 0.363005
grad_step = 000009, loss = 0.350722
grad_step = 000010, loss = 0.336242
grad_step = 000011, loss = 0.325670
grad_step = 000012, loss = 0.316546
grad_step = 000013, loss = 0.307375
grad_step = 000014, loss = 0.298394
grad_step = 000015, loss = 0.289825
grad_step = 000016, loss = 0.281357
grad_step = 000017, loss = 0.272536
grad_step = 000018, loss = 0.262937
grad_step = 000019, loss = 0.252578
grad_step = 000020, loss = 0.242096
grad_step = 000021, loss = 0.232541
grad_step = 000022, loss = 0.224156
grad_step = 000023, loss = 0.215172
grad_step = 000024, loss = 0.205852
grad_step = 000025, loss = 0.197254
grad_step = 000026, loss = 0.189006
grad_step = 000027, loss = 0.180973
grad_step = 000028, loss = 0.173011
grad_step = 000029, loss = 0.165122
grad_step = 000030, loss = 0.157378
grad_step = 000031, loss = 0.149579
grad_step = 000032, loss = 0.141819
grad_step = 000033, loss = 0.134679
grad_step = 000034, loss = 0.127848
grad_step = 000035, loss = 0.120848
grad_step = 000036, loss = 0.113916
grad_step = 000037, loss = 0.107337
grad_step = 000038, loss = 0.101078
grad_step = 000039, loss = 0.094870
grad_step = 000040, loss = 0.088818
grad_step = 000041, loss = 0.082921
grad_step = 000042, loss = 0.077449
grad_step = 000043, loss = 0.072374
grad_step = 000044, loss = 0.067382
grad_step = 000045, loss = 0.062493
grad_step = 000046, loss = 0.058050
grad_step = 000047, loss = 0.053803
grad_step = 000048, loss = 0.049673
grad_step = 000049, loss = 0.045770
grad_step = 000050, loss = 0.042113
grad_step = 000051, loss = 0.038735
grad_step = 000052, loss = 0.035659
grad_step = 000053, loss = 0.032667
grad_step = 000054, loss = 0.029910
grad_step = 000055, loss = 0.027427
grad_step = 000056, loss = 0.025067
grad_step = 000057, loss = 0.022899
grad_step = 000058, loss = 0.020947
grad_step = 000059, loss = 0.019181
grad_step = 000060, loss = 0.017533
grad_step = 000061, loss = 0.016027
grad_step = 000062, loss = 0.014690
grad_step = 000063, loss = 0.013444
grad_step = 000064, loss = 0.012329
grad_step = 000065, loss = 0.011334
grad_step = 000066, loss = 0.010445
grad_step = 000067, loss = 0.009648
grad_step = 000068, loss = 0.008916
grad_step = 000069, loss = 0.008267
grad_step = 000070, loss = 0.007670
grad_step = 000071, loss = 0.007129
grad_step = 000072, loss = 0.006646
grad_step = 000073, loss = 0.006222
grad_step = 000074, loss = 0.005818
grad_step = 000075, loss = 0.005445
grad_step = 000076, loss = 0.005121
grad_step = 000077, loss = 0.004819
grad_step = 000078, loss = 0.004543
grad_step = 000079, loss = 0.004299
grad_step = 000080, loss = 0.004081
grad_step = 000081, loss = 0.003875
grad_step = 000082, loss = 0.003692
grad_step = 000083, loss = 0.003525
grad_step = 000084, loss = 0.003376
grad_step = 000085, loss = 0.003238
grad_step = 000086, loss = 0.003117
grad_step = 000087, loss = 0.003007
grad_step = 000088, loss = 0.002907
grad_step = 000089, loss = 0.002821
grad_step = 000090, loss = 0.002746
grad_step = 000091, loss = 0.002677
grad_step = 000092, loss = 0.002621
grad_step = 000093, loss = 0.002575
grad_step = 000094, loss = 0.002541
grad_step = 000095, loss = 0.002515
grad_step = 000096, loss = 0.002498
grad_step = 000097, loss = 0.002467
grad_step = 000098, loss = 0.002424
grad_step = 000099, loss = 0.002371
grad_step = 000100, loss = 0.002333
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002319
grad_step = 000102, loss = 0.002321
grad_step = 000103, loss = 0.002323
grad_step = 000104, loss = 0.002308
grad_step = 000105, loss = 0.002285
grad_step = 000106, loss = 0.002263
grad_step = 000107, loss = 0.002254
grad_step = 000108, loss = 0.002257
grad_step = 000109, loss = 0.002260
grad_step = 000110, loss = 0.002257
grad_step = 000111, loss = 0.002246
grad_step = 000112, loss = 0.002233
grad_step = 000113, loss = 0.002223
grad_step = 000114, loss = 0.002219
grad_step = 000115, loss = 0.002220
grad_step = 000116, loss = 0.002221
grad_step = 000117, loss = 0.002220
grad_step = 000118, loss = 0.002215
grad_step = 000119, loss = 0.002209
grad_step = 000120, loss = 0.002200
grad_step = 000121, loss = 0.002192
grad_step = 000122, loss = 0.002186
grad_step = 000123, loss = 0.002180
grad_step = 000124, loss = 0.002177
grad_step = 000125, loss = 0.002174
grad_step = 000126, loss = 0.002172
grad_step = 000127, loss = 0.002171
grad_step = 000128, loss = 0.002172
grad_step = 000129, loss = 0.002176
grad_step = 000130, loss = 0.002189
grad_step = 000131, loss = 0.002211
grad_step = 000132, loss = 0.002252
grad_step = 000133, loss = 0.002282
grad_step = 000134, loss = 0.002299
grad_step = 000135, loss = 0.002236
grad_step = 000136, loss = 0.002157
grad_step = 000137, loss = 0.002121
grad_step = 000138, loss = 0.002147
grad_step = 000139, loss = 0.002195
grad_step = 000140, loss = 0.002200
grad_step = 000141, loss = 0.002163
grad_step = 000142, loss = 0.002112
grad_step = 000143, loss = 0.002099
grad_step = 000144, loss = 0.002123
grad_step = 000145, loss = 0.002145
grad_step = 000146, loss = 0.002144
grad_step = 000147, loss = 0.002111
grad_step = 000148, loss = 0.002082
grad_step = 000149, loss = 0.002077
grad_step = 000150, loss = 0.002092
grad_step = 000151, loss = 0.002111
grad_step = 000152, loss = 0.002112
grad_step = 000153, loss = 0.002103
grad_step = 000154, loss = 0.002077
grad_step = 000155, loss = 0.002057
grad_step = 000156, loss = 0.002049
grad_step = 000157, loss = 0.002053
grad_step = 000158, loss = 0.002063
grad_step = 000159, loss = 0.002072
grad_step = 000160, loss = 0.002083
grad_step = 000161, loss = 0.002082
grad_step = 000162, loss = 0.002082
grad_step = 000163, loss = 0.002063
grad_step = 000164, loss = 0.002045
grad_step = 000165, loss = 0.002026
grad_step = 000166, loss = 0.002014
grad_step = 000167, loss = 0.002012
grad_step = 000168, loss = 0.002012
grad_step = 000169, loss = 0.002016
grad_step = 000170, loss = 0.002032
grad_step = 000171, loss = 0.002075
grad_step = 000172, loss = 0.002095
grad_step = 000173, loss = 0.002140
grad_step = 000174, loss = 0.002130
grad_step = 000175, loss = 0.002079
grad_step = 000176, loss = 0.002003
grad_step = 000177, loss = 0.001972
grad_step = 000178, loss = 0.001988
grad_step = 000179, loss = 0.002036
grad_step = 000180, loss = 0.002120
grad_step = 000181, loss = 0.002086
grad_step = 000182, loss = 0.002046
grad_step = 000183, loss = 0.001975
grad_step = 000184, loss = 0.001940
grad_step = 000185, loss = 0.001967
grad_step = 000186, loss = 0.002001
grad_step = 000187, loss = 0.002022
grad_step = 000188, loss = 0.001999
grad_step = 000189, loss = 0.001975
grad_step = 000190, loss = 0.001920
grad_step = 000191, loss = 0.001910
grad_step = 000192, loss = 0.001917
grad_step = 000193, loss = 0.001935
grad_step = 000194, loss = 0.001952
grad_step = 000195, loss = 0.001934
grad_step = 000196, loss = 0.001923
grad_step = 000197, loss = 0.001893
grad_step = 000198, loss = 0.001866
grad_step = 000199, loss = 0.001862
grad_step = 000200, loss = 0.001869
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001878
grad_step = 000202, loss = 0.001893
grad_step = 000203, loss = 0.001912
grad_step = 000204, loss = 0.001916
grad_step = 000205, loss = 0.001920
grad_step = 000206, loss = 0.001890
grad_step = 000207, loss = 0.001858
grad_step = 000208, loss = 0.001839
grad_step = 000209, loss = 0.001831
grad_step = 000210, loss = 0.001839
grad_step = 000211, loss = 0.001855
grad_step = 000212, loss = 0.001870
grad_step = 000213, loss = 0.001878
grad_step = 000214, loss = 0.001896
grad_step = 000215, loss = 0.001873
grad_step = 000216, loss = 0.001854
grad_step = 000217, loss = 0.001832
grad_step = 000218, loss = 0.001814
grad_step = 000219, loss = 0.001816
grad_step = 000220, loss = 0.001825
grad_step = 000221, loss = 0.001835
grad_step = 000222, loss = 0.001848
grad_step = 000223, loss = 0.001868
grad_step = 000224, loss = 0.001867
grad_step = 000225, loss = 0.001873
grad_step = 000226, loss = 0.001863
grad_step = 000227, loss = 0.001845
grad_step = 000228, loss = 0.001822
grad_step = 000229, loss = 0.001803
grad_step = 000230, loss = 0.001792
grad_step = 000231, loss = 0.001793
grad_step = 000232, loss = 0.001799
grad_step = 000233, loss = 0.001809
grad_step = 000234, loss = 0.001821
grad_step = 000235, loss = 0.001830
grad_step = 000236, loss = 0.001849
grad_step = 000237, loss = 0.001857
grad_step = 000238, loss = 0.001869
grad_step = 000239, loss = 0.001864
grad_step = 000240, loss = 0.001855
grad_step = 000241, loss = 0.001825
grad_step = 000242, loss = 0.001796
grad_step = 000243, loss = 0.001775
grad_step = 000244, loss = 0.001769
grad_step = 000245, loss = 0.001777
grad_step = 000246, loss = 0.001791
grad_step = 000247, loss = 0.001808
grad_step = 000248, loss = 0.001819
grad_step = 000249, loss = 0.001835
grad_step = 000250, loss = 0.001837
grad_step = 000251, loss = 0.001841
grad_step = 000252, loss = 0.001825
grad_step = 000253, loss = 0.001807
grad_step = 000254, loss = 0.001781
grad_step = 000255, loss = 0.001761
grad_step = 000256, loss = 0.001750
grad_step = 000257, loss = 0.001749
grad_step = 000258, loss = 0.001754
grad_step = 000259, loss = 0.001764
grad_step = 000260, loss = 0.001777
grad_step = 000261, loss = 0.001786
grad_step = 000262, loss = 0.001799
grad_step = 000263, loss = 0.001801
grad_step = 000264, loss = 0.001801
grad_step = 000265, loss = 0.001788
grad_step = 000266, loss = 0.001774
grad_step = 000267, loss = 0.001755
grad_step = 000268, loss = 0.001740
grad_step = 000269, loss = 0.001730
grad_step = 000270, loss = 0.001726
grad_step = 000271, loss = 0.001728
grad_step = 000272, loss = 0.001734
grad_step = 000273, loss = 0.001743
grad_step = 000274, loss = 0.001753
grad_step = 000275, loss = 0.001773
grad_step = 000276, loss = 0.001796
grad_step = 000277, loss = 0.001835
grad_step = 000278, loss = 0.001864
grad_step = 000279, loss = 0.001902
grad_step = 000280, loss = 0.001885
grad_step = 000281, loss = 0.001848
grad_step = 000282, loss = 0.001772
grad_step = 000283, loss = 0.001716
grad_step = 000284, loss = 0.001710
grad_step = 000285, loss = 0.001742
grad_step = 000286, loss = 0.001780
grad_step = 000287, loss = 0.001787
grad_step = 000288, loss = 0.001770
grad_step = 000289, loss = 0.001729
grad_step = 000290, loss = 0.001699
grad_step = 000291, loss = 0.001693
grad_step = 000292, loss = 0.001707
grad_step = 000293, loss = 0.001729
grad_step = 000294, loss = 0.001743
grad_step = 000295, loss = 0.001751
grad_step = 000296, loss = 0.001738
grad_step = 000297, loss = 0.001720
grad_step = 000298, loss = 0.001699
grad_step = 000299, loss = 0.001683
grad_step = 000300, loss = 0.001676
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001678
grad_step = 000302, loss = 0.001685
grad_step = 000303, loss = 0.001694
grad_step = 000304, loss = 0.001702
grad_step = 000305, loss = 0.001705
grad_step = 000306, loss = 0.001708
grad_step = 000307, loss = 0.001704
grad_step = 000308, loss = 0.001699
grad_step = 000309, loss = 0.001690
grad_step = 000310, loss = 0.001687
grad_step = 000311, loss = 0.001683
grad_step = 000312, loss = 0.001679
grad_step = 000313, loss = 0.001674
grad_step = 000314, loss = 0.001672
grad_step = 000315, loss = 0.001667
grad_step = 000316, loss = 0.001664
grad_step = 000317, loss = 0.001661
grad_step = 000318, loss = 0.001658
grad_step = 000319, loss = 0.001657
grad_step = 000320, loss = 0.001658
grad_step = 000321, loss = 0.001661
grad_step = 000322, loss = 0.001668
grad_step = 000323, loss = 0.001680
grad_step = 000324, loss = 0.001702
grad_step = 000325, loss = 0.001730
grad_step = 000326, loss = 0.001784
grad_step = 000327, loss = 0.001829
grad_step = 000328, loss = 0.001885
grad_step = 000329, loss = 0.001871
grad_step = 000330, loss = 0.001813
grad_step = 000331, loss = 0.001705
grad_step = 000332, loss = 0.001630
grad_step = 000333, loss = 0.001624
grad_step = 000334, loss = 0.001671
grad_step = 000335, loss = 0.001724
grad_step = 000336, loss = 0.001734
grad_step = 000337, loss = 0.001710
grad_step = 000338, loss = 0.001651
grad_step = 000339, loss = 0.001611
grad_step = 000340, loss = 0.001606
grad_step = 000341, loss = 0.001630
grad_step = 000342, loss = 0.001661
grad_step = 000343, loss = 0.001673
grad_step = 000344, loss = 0.001668
grad_step = 000345, loss = 0.001638
grad_step = 000346, loss = 0.001609
grad_step = 000347, loss = 0.001590
grad_step = 000348, loss = 0.001587
grad_step = 000349, loss = 0.001597
grad_step = 000350, loss = 0.001611
grad_step = 000351, loss = 0.001622
grad_step = 000352, loss = 0.001621
grad_step = 000353, loss = 0.001617
grad_step = 000354, loss = 0.001602
grad_step = 000355, loss = 0.001587
grad_step = 000356, loss = 0.001574
grad_step = 000357, loss = 0.001567
grad_step = 000358, loss = 0.001565
grad_step = 000359, loss = 0.001567
grad_step = 000360, loss = 0.001572
grad_step = 000361, loss = 0.001578
grad_step = 000362, loss = 0.001586
grad_step = 000363, loss = 0.001592
grad_step = 000364, loss = 0.001602
grad_step = 000365, loss = 0.001609
grad_step = 000366, loss = 0.001618
grad_step = 000367, loss = 0.001618
grad_step = 000368, loss = 0.001619
grad_step = 000369, loss = 0.001608
grad_step = 000370, loss = 0.001596
grad_step = 000371, loss = 0.001576
grad_step = 000372, loss = 0.001558
grad_step = 000373, loss = 0.001542
grad_step = 000374, loss = 0.001532
grad_step = 000375, loss = 0.001527
grad_step = 000376, loss = 0.001526
grad_step = 000377, loss = 0.001529
grad_step = 000378, loss = 0.001533
grad_step = 000379, loss = 0.001540
grad_step = 000380, loss = 0.001550
grad_step = 000381, loss = 0.001567
grad_step = 000382, loss = 0.001588
grad_step = 000383, loss = 0.001627
grad_step = 000384, loss = 0.001667
grad_step = 000385, loss = 0.001730
grad_step = 000386, loss = 0.001754
grad_step = 000387, loss = 0.001771
grad_step = 000388, loss = 0.001701
grad_step = 000389, loss = 0.001607
grad_step = 000390, loss = 0.001520
grad_step = 000391, loss = 0.001493
grad_step = 000392, loss = 0.001526
grad_step = 000393, loss = 0.001577
grad_step = 000394, loss = 0.001615
grad_step = 000395, loss = 0.001599
grad_step = 000396, loss = 0.001562
grad_step = 000397, loss = 0.001507
grad_step = 000398, loss = 0.001476
grad_step = 000399, loss = 0.001477
grad_step = 000400, loss = 0.001499
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001524
grad_step = 000402, loss = 0.001532
grad_step = 000403, loss = 0.001529
grad_step = 000404, loss = 0.001504
grad_step = 000405, loss = 0.001479
grad_step = 000406, loss = 0.001458
grad_step = 000407, loss = 0.001448
grad_step = 000408, loss = 0.001449
grad_step = 000409, loss = 0.001458
grad_step = 000410, loss = 0.001467
grad_step = 000411, loss = 0.001473
grad_step = 000412, loss = 0.001480
grad_step = 000413, loss = 0.001477
grad_step = 000414, loss = 0.001472
grad_step = 000415, loss = 0.001461
grad_step = 000416, loss = 0.001450
grad_step = 000417, loss = 0.001437
grad_step = 000418, loss = 0.001427
grad_step = 000419, loss = 0.001417
grad_step = 000420, loss = 0.001410
grad_step = 000421, loss = 0.001404
grad_step = 000422, loss = 0.001400
grad_step = 000423, loss = 0.001397
grad_step = 000424, loss = 0.001394
grad_step = 000425, loss = 0.001393
grad_step = 000426, loss = 0.001393
grad_step = 000427, loss = 0.001397
grad_step = 000428, loss = 0.001405
grad_step = 000429, loss = 0.001429
grad_step = 000430, loss = 0.001473
grad_step = 000431, loss = 0.001572
grad_step = 000432, loss = 0.001700
grad_step = 000433, loss = 0.001931
grad_step = 000434, loss = 0.002015
grad_step = 000435, loss = 0.001986
grad_step = 000436, loss = 0.001683
grad_step = 000437, loss = 0.001403
grad_step = 000438, loss = 0.001386
grad_step = 000439, loss = 0.001573
grad_step = 000440, loss = 0.001717
grad_step = 000441, loss = 0.001604
grad_step = 000442, loss = 0.001419
grad_step = 000443, loss = 0.001342
grad_step = 000444, loss = 0.001444
grad_step = 000445, loss = 0.001550
grad_step = 000446, loss = 0.001485
grad_step = 000447, loss = 0.001361
grad_step = 000448, loss = 0.001326
grad_step = 000449, loss = 0.001401
grad_step = 000450, loss = 0.001453
grad_step = 000451, loss = 0.001419
grad_step = 000452, loss = 0.001366
grad_step = 000453, loss = 0.001310
grad_step = 000454, loss = 0.001321
grad_step = 000455, loss = 0.001372
grad_step = 000456, loss = 0.001372
grad_step = 000457, loss = 0.001348
grad_step = 000458, loss = 0.001308
grad_step = 000459, loss = 0.001288
grad_step = 000460, loss = 0.001304
grad_step = 000461, loss = 0.001324
grad_step = 000462, loss = 0.001325
grad_step = 000463, loss = 0.001306
grad_step = 000464, loss = 0.001283
grad_step = 000465, loss = 0.001271
grad_step = 000466, loss = 0.001278
grad_step = 000467, loss = 0.001286
grad_step = 000468, loss = 0.001287
grad_step = 000469, loss = 0.001281
grad_step = 000470, loss = 0.001265
grad_step = 000471, loss = 0.001256
grad_step = 000472, loss = 0.001256
grad_step = 000473, loss = 0.001258
grad_step = 000474, loss = 0.001262
grad_step = 000475, loss = 0.001260
grad_step = 000476, loss = 0.001253
grad_step = 000477, loss = 0.001245
grad_step = 000478, loss = 0.001239
grad_step = 000479, loss = 0.001235
grad_step = 000480, loss = 0.001234
grad_step = 000481, loss = 0.001235
grad_step = 000482, loss = 0.001234
grad_step = 000483, loss = 0.001233
grad_step = 000484, loss = 0.001230
grad_step = 000485, loss = 0.001225
grad_step = 000486, loss = 0.001221
grad_step = 000487, loss = 0.001217
grad_step = 000488, loss = 0.001213
grad_step = 000489, loss = 0.001210
grad_step = 000490, loss = 0.001208
grad_step = 000491, loss = 0.001206
grad_step = 000492, loss = 0.001204
grad_step = 000493, loss = 0.001202
grad_step = 000494, loss = 0.001201
grad_step = 000495, loss = 0.001200
grad_step = 000496, loss = 0.001200
grad_step = 000497, loss = 0.001200
grad_step = 000498, loss = 0.001202
grad_step = 000499, loss = 0.001206
grad_step = 000500, loss = 0.001215
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001227
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

  date_run                              2020-05-11 22:13:50.211549
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.277142
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 22:13:50.218355
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.202134
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 22:13:50.225837
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.148033
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 22:13:50.232391
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -2.0715
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
0   2020-05-11 22:13:16.592708  ...    mean_absolute_error
1   2020-05-11 22:13:16.597308  ...     mean_squared_error
2   2020-05-11 22:13:16.601025  ...  median_absolute_error
3   2020-05-11 22:13:16.605034  ...               r2_score
4   2020-05-11 22:13:26.381532  ...    mean_absolute_error
5   2020-05-11 22:13:26.386401  ...     mean_squared_error
6   2020-05-11 22:13:26.390241  ...  median_absolute_error
7   2020-05-11 22:13:26.394410  ...               r2_score
8   2020-05-11 22:13:50.211549  ...    mean_absolute_error
9   2020-05-11 22:13:50.218355  ...     mean_squared_error
10  2020-05-11 22:13:50.225837  ...  median_absolute_error
11  2020-05-11 22:13:50.232391  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 313729.79it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 405527.63it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 561273.88it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 792935.86it/s] 78%|███████▊  | 7692288/9912422 [00:00<00:01, 1121823.49it/s]9920512it [00:00, 10352838.81it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 151039.29it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314395.30it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 408127.22it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 564949.32it/s]1654784it [00:00, 2797457.04it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51787.07it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e9c1a7fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e398c4c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e9c132ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e3939c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4eb3bd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4eb2ce48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4e108240> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e4eb2ce48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e9c132ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e42bdc6a0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e9c132ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3c51a801d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b9ebfe958e441201be0051c99198291c07f8191cd4d960a166fa6ee9e4d43e04
  Stored in directory: /tmp/pip-ephem-wheel-cache-pigq02xy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3be96681d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 38s
  106496/17464789 [..............................] - ETA: 31s
  212992/17464789 [..............................] - ETA: 20s
  458752/17464789 [..............................] - ETA: 11s
  909312/17464789 [>.............................] - ETA: 6s 
 1835008/17464789 [==>...........................] - ETA: 3s
 3645440/17464789 [=====>........................] - ETA: 1s
 6758400/17464789 [==========>...................] - ETA: 0s
 9805824/17464789 [===============>..............] - ETA: 0s
12886016/17464789 [=====================>........] - ETA: 0s
14999552/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 22:15:23.511039: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 22:15:23.515540: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 22:15:23.515736: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5569a6fe5170 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 22:15:23.515752: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5286 - accuracy: 0.5090
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.4775 - accuracy: 0.5123 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5261 - accuracy: 0.5092
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6031 - accuracy: 0.5041
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5708 - accuracy: 0.5063
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5814 - accuracy: 0.5056
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
11000/25000 [============>.................] - ETA: 4s - loss: 7.5635 - accuracy: 0.5067
12000/25000 [=============>................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6656 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6387 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6311 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6229 - accuracy: 0.5028
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6345 - accuracy: 0.5021
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6532 - accuracy: 0.5009
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 22:15:41.428944
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 22:15:41.428944  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 22:15:48.598944: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 22:15:48.604642: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-11 22:15:48.604830: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5605815d3c10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 22:15:48.604847: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7efc45c90c88> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2526 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.2109 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7efc20bd4f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 8.0193 - accuracy: 0.4770
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7586 - accuracy: 0.4940
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7433 - accuracy: 0.4950 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7241 - accuracy: 0.4963
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7188 - accuracy: 0.4966
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.7688 - accuracy: 0.4933
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7395 - accuracy: 0.4952
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7263 - accuracy: 0.4961
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7295 - accuracy: 0.4959
11000/25000 [============>.................] - ETA: 4s - loss: 7.7084 - accuracy: 0.4973
12000/25000 [=============>................] - ETA: 4s - loss: 7.7190 - accuracy: 0.4966
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7574 - accuracy: 0.4941
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7849 - accuracy: 0.4923
15000/25000 [=================>............] - ETA: 3s - loss: 7.7842 - accuracy: 0.4923
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7596 - accuracy: 0.4939
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7685 - accuracy: 0.4934
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7569 - accuracy: 0.4941
19000/25000 [=====================>........] - ETA: 2s - loss: 7.7449 - accuracy: 0.4949
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7226 - accuracy: 0.4963
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7170 - accuracy: 0.4967
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6873 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 10s 406us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7efbeda72518> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:56:10, 20.1kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<8:22:14, 28.6kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.82M/862M [00:00<5:49:02, 40.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:00<4:01:51, 58.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.0M/862M [00:00<2:47:28, 83.3kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:00<1:55:40, 119kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:01<1:19:55, 170kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:01<55:57, 241kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<39:20, 342kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<11:48:03, 19.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:03<8:15:37, 27.1kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<5:47:17, 38.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<9:07:31, 24.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:22:57, 34.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<9:41:11, 23.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<6:46:21, 32.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:27:57, 21.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:19:01, 30.3kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:47:10, 20.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:32:27, 29.3kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:44:38, 20.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:30:38, 29.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:51:59, 20.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:35:47, 29.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:48:37, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:33:25, 29.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:53:29, 20.1kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:36:50, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:47:44, 20.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:32:52, 28.9kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:29:58, 20.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:20:22, 29.7kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:40:59, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:28:04, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:43:52, 20.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:30:06, 28.9kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:36:49, 20.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:25:14, 29.1kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:21:44, 20.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:14:36, 29.7kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:35:26, 20.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:24:15, 29.0kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:17:00, 20.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:11:18, 29.8kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:29:57, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:20:25, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:12:25, 20.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:08:05, 29.9kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:25:55, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:17:30, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:30:06, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:20:33, 28.9kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:01:36, 21.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:00:31, 30.1kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:17:57, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:11:58, 29.3kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:17:55, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:11:54, 29.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:24:17, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:16:21, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:22:54, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:15:24, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:16:27, 20.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<7:10:52, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:17:42, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<7:11:46, 28.9kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:10:27, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:06:45, 29.1kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<9:53:47, 20.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<6:55:05, 29.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<9:48:27, 21.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<6:51:22, 30.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<9:45:25, 21.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<6:49:16, 30.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<9:39:04, 21.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<6:44:44, 30.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<9:57:57, 20.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<6:57:55, 29.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<10:02:26, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<7:01:04, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<9:57:00, 20.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<6:57:14, 29.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:03:48, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<7:02:00, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:02:38, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<7:01:12, 28.8kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:53:45, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:54:57, 29.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<10:00:00, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:59:20, 28.7kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:57:02, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:57:14, 28.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:58:13, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:58:05, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:52:58, 20.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:54:24, 28.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:51:47, 20.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:53:34, 28.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:49:43, 20.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:52:07, 28.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:49:17, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:51:55, 28.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:20:53, 21.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:31:58, 30.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:35:27, 20.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:42:13, 29.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:20:13, 21.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:31:34, 30.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:13:58, 21.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<6:27:08, 30.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:30:19, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:38:33, 29.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:32:45, 20.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:40:16, 29.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:28:30, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:37:16, 29.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:34:27, 20.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:41:25, 28.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:33:04, 20.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:40:29, 28.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:25:15, 20.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:35:04, 29.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:09:49, 20.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:24:12, 29.8kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:22:40, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:33:10, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:24:59, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:34:49, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:16:55, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:29:09, 29.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:21:57, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:32:40, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:19:05, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:30:41, 28.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:12:03, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:25:44, 29.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<9:17:12, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:29:20, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:16:21, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:28:46, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<9:05:34, 20.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:21:11, 29.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:12:09, 20.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:25:47, 28.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:11:09, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:25:07, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<9:03:37, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:19:49, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:08:28, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:23:13, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<9:05:41, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:21:16, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<8:58:36, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:16:22, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<8:44:29, 20.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:06:26, 29.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:56:00, 20.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:14:28, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:57:46, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:15:44, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:51:35, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:11:23, 29.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:55:28, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<6:14:06, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:53:36, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:12:49, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:44:27, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<6:06:26, 29.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:36:53, 20.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:01:06, 29.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:45:03, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:06:48, 28.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:44:42, 20.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<6:06:35, 28.9kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:38:20, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<6:02:09, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:29:57, 20.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<5:56:14, 29.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:35:32, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<6:00:09, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:35:43, 20.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<6:00:17, 29.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:31:00, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<5:57:01, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:19:42, 20.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:49:08, 29.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:12:28, 21.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:44:00, 30.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:26:18, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:53:44, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:11:56, 20.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:43:38, 29.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:20:10, 20.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:49:24, 29.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:19:10, 20.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:48:41, 29.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:19:33, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:48:56, 29.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:28:26, 20.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:55:08, 28.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:23:27, 20.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:51:38, 28.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:25:43, 20.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:53:14, 28.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:17:55, 20.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:47:46, 28.8kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:19:25, 20.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:48:54, 28.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<7:54:29, 21.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:31:25, 30.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:04:50, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:38:44, 29.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<7:41:20, 21.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:22:14, 30.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<8:01:09, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:36:09, 29.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<7:40:35, 21.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:21:42, 30.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<7:58:56, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:34:30, 29.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<7:57:58, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:33:49, 29.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<8:03:35, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:37:44, 28.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:59:08, 20.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:34:40, 28.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:46:22, 20.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:25:42, 29.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:54:33, 20.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:31:28, 29.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:39:20, 20.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:20:51, 29.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:32:29, 21.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:16:03, 30.2kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:29:21, 21.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:13:48, 30.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:44:05, 20.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:24:05, 29.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:46:07, 20.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:25:31, 29.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:39:34, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:20:59, 29.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:30:17, 20.9kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:14:30, 29.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:22:10, 21.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:08:46, 30.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:34:57, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:17:45, 29.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:21:56, 21.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:08:36, 30.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:32:27, 20.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:15:56, 29.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:33:54, 20.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:16:57, 29.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:29:09, 20.4kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<5:13:37, 29.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:33:18, 20.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<5:16:30, 28.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:31:11, 20.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<5:15:02, 28.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:25:47, 20.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<5:11:15, 29.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:29:03, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<5:13:35, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:12:51, 20.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:02:15, 29.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:12:00, 20.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<5:01:29, 29.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<3:34:08, 41.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<6:02:07, 24.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<4:12:40, 35.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<3:04:18, 48.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<5:40:16, 26.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<3:57:26, 37.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<2:52:59, 51.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<5:31:51, 26.7kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<3:51:34, 38.2kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<2:47:52, 52.6kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<5:27:31, 27.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<3:48:31, 38.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<2:51:28, 51.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<5:29:26, 26.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<3:50:08, 38.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<6:10:52, 23.6kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<4:19:01, 33.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<6:28:17, 22.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<4:31:09, 32.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<6:36:39, 21.9kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<4:37:14, 31.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<3:14:35, 44.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<5:30:49, 26.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<3:51:08, 37.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<5:53:13, 24.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<4:06:41, 34.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<6:18:46, 22.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:24:14, 32.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<3:12:50, 44.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<5:38:01, 25.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<3:55:51, 36.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<2:49:11, 50.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<5:20:49, 26.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<3:44:04, 37.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<5:58:24, 23.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<4:10:17, 33.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:15:42, 22.5kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<4:22:10, 32.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<3:06:12, 45.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<5:30:21, 25.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<3:50:43, 36.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:00:26, 23.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<4:11:29, 33.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<2:59:10, 46.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<5:23:47, 25.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<3:46:10, 36.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<5:42:47, 24.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<3:59:11, 34.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<2:50:06, 48.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<5:14:53, 26.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<3:39:40, 37.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<2:39:39, 51.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<5:06:32, 26.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<3:34:08, 38.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<5:31:27, 24.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<3:51:29, 35.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<5:43:54, 23.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<3:59:56, 33.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<2:51:38, 47.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<4:57:42, 27.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<3:27:54, 39.0kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<5:39:04, 23.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<3:56:31, 34.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<2:50:26, 47.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<4:56:39, 27.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<3:27:09, 38.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<5:36:45, 23.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<3:54:54, 34.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<2:48:59, 47.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<4:54:09, 27.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<3:25:24, 38.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<5:33:56, 23.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<3:52:57, 34.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<2:46:10, 47.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<5:03:48, 26.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<3:32:10, 37.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<5:23:42, 24.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<3:45:47, 34.8kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<2:43:22, 48.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<4:46:53, 27.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<3:20:22, 39.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<5:15:14, 24.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<3:39:50, 35.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<2:39:24, 48.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<4:56:03, 26.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<3:26:42, 37.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<5:30:05, 23.5kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<3:50:10, 33.5kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<2:54:56, 44.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<4:53:00, 26.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<3:24:34, 37.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<5:26:34, 23.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<3:47:57, 33.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<5:41:45, 22.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<3:58:31, 31.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<5:48:46, 21.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<4:03:24, 31.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<5:51:19, 21.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<4:05:10, 30.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<5:52:14, 21.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<4:05:48, 30.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<5:50:17, 21.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<4:04:08, 30.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<2:52:13, 43.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<2:01:02, 61.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<1:24:48, 87.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<3:39:10, 33.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:56<2:32:41, 48.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<1:48:46, 67.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<1:16:37, 95.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<53:50, 135kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<3:14:53, 37.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<2:16:08, 53.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<4:28:52, 27.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<3:07:40, 38.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:04:15, 23.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<3:31:58, 33.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<2:29:51, 47.7kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<1:45:22, 67.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<1:13:51, 96.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<3:26:07, 34.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<2:23:56, 49.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<4:30:01, 26.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<3:08:26, 37.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:00:30, 23.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<3:29:39, 33.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<5:14:19, 22.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<3:39:19, 31.8kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:09:00, 22.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<3:35:34, 32.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:16:25, 21.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<3:40:44, 31.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:19:35, 21.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<3:42:54, 30.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:24:40, 21.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:46:08, 30.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<2:39:32, 42.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<1:52:35, 60.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<1:18:48, 85.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<3:19:06, 34.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<2:18:59, 48.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<4:18:22, 26.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<2:59:55, 37.2kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<2:07:22, 52.3kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<1:29:39, 74.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<1:02:50, 106kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<3:06:35, 35.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<2:10:15, 50.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<4:07:51, 26.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<2:52:54, 37.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<4:40:42, 23.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<3:15:35, 33.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<2:25:42, 44.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<4:06:30, 26.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<2:51:56, 37.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<4:38:22, 23.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<3:14:07, 33.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<4:54:30, 21.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<3:25:21, 31.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<4:56:14, 21.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<3:26:35, 30.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<4:50:47, 22.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:25<3:22:17, 31.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<2:23:11, 44.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<1:40:33, 62.7kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<1:10:26, 89.2kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<3:06:24, 33.7kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<2:10:07, 48.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<3:49:21, 27.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<2:39:57, 38.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<4:20:17, 23.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<3:01:40, 34.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<2:07:36, 48.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<3:53:24, 26.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<2:42:22, 37.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<1:55:11, 53.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<1:21:01, 75.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<56:47, 107kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<2:52:05, 35.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<1:59:46, 50.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<1:25:13, 70.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<1:00:05, 99.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<42:11, 141kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<2:36:38, 38.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<1:49:03, 54.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<1:17:37, 76.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<54:44, 108kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<38:27, 152kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<2:35:01, 37.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<1:47:58, 54.0kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<1:16:39, 75.6kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<54:14, 107kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<38:05, 151kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<2:26:42, 39.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<1:42:03, 56.0kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<1:12:55, 78.0kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<51:31, 110kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<36:10, 156kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<2:29:20, 37.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<1:43:58, 54.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<1:13:52, 75.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<52:20, 107kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<36:43, 151kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<2:23:54, 38.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:49<1:40:05, 55.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:49<1:09:56, 78.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<6:09:28, 14.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<4:18:45, 21.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:51<2:59:27, 30.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<2:10:16, 41.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:31:26, 59.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<1:04:00, 83.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<2:37:06, 34.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<1:49:22, 48.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<1:17:27, 68.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<54:52, 96.6kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<38:27, 137kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<2:21:56, 37.1kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:57<1:38:40, 53.0kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:57<1:08:58, 75.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<5:44:09, 15.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<4:00:59, 21.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [03:59<2:47:05, 30.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<2:01:06, 42.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<1:25:04, 60.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<59:29, 85.6kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<2:34:31, 33.0kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<1:47:32, 47.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<1:16:03, 66.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<53:47, 93.3kB/s]  .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<37:40, 132kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<2:14:36, 37.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<1:33:34, 52.9kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<1:06:34, 73.9kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<46:56, 105kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:07<32:36, 149kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<25:22, 191kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<18:07, 267kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:09<12:51, 374kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<1:58:51, 40.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<1:22:30, 57.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<59:10, 80.2kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<41:44, 113kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:12<28:55, 162kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<25:51, 181kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<18:27, 253kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<13:05, 355kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<1:53:20, 40.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<1:18:41, 58.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<56:16, 81.2kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<39:42, 115kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:17<27:34, 164kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<21:34, 209kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<15:26, 291kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<10:58, 406kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:48:24, 41.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<1:15:15, 58.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<53:43, 81.8kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<37:55, 116kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:22<26:16, 165kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<22:14, 194kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<15:53, 272kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<11:18, 379kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:37:45, 43.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:25<1:07:45, 62.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<48:45, 86.5kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<34:27, 122kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<24:12, 173kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<1:39:37, 42.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:28<1:08:59, 60.0kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<49:50, 82.5kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<35:10, 117kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<24:40, 165kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<1:45:27, 38.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<1:13:07, 55.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<52:13, 76.8kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<36:46, 109kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<25:46, 154kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<1:46:09, 37.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<1:13:34, 53.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<52:32, 74.3kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<37:01, 105kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<25:57, 149kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<1:38:19, 39.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<1:08:08, 56.2kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<48:37, 78.1kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<34:21, 110kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<24:03, 156kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<1:37:28, 38.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:40<1:07:23, 55.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<48:38, 76.0kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<34:17, 108kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<24:00, 152kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<1:38:41, 37.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<1:08:19, 52.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<48:47, 73.6kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<34:32, 104kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<24:08, 147kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<1:35:15, 37.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:46<1:05:48, 53.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<47:25, 73.5kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<33:25, 104kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<23:22, 148kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:33:04, 37.1kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:49<1:04:19, 52.9kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<46:05, 73.4kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<32:31, 104kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<22:44, 147kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<1:29:39, 37.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:52<1:01:47, 53.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<44:58, 72.8kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<31:41, 103kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<22:09, 146kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:26:07, 37.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:55<59:19, 53.7kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<43:11, 73.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<30:26, 104kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<21:16, 147kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<1:24:18, 37.2kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:58<57:59, 53.1kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<42:37, 71.9kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<30:00, 102kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<20:58, 144kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<1:20:46, 37.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:01<55:30, 53.6kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<40:57, 72.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<28:50, 102kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<20:09, 145kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<1:17:59, 37.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:04<53:35, 53.6kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<39:15, 72.8kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<27:42, 103kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<19:20, 146kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<1:13:47, 38.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:07<50:38, 54.6kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<37:27, 73.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<26:23, 104kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<18:25, 147kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<1:12:32, 37.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:10<49:40, 53.5kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<37:30, 70.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<26:24, 100kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<18:26, 142kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:05:08, 40.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:13<44:35, 57.2kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<33:23, 76.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<23:32, 108kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<16:25, 153kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<1:05:14, 38.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:16<44:36, 54.9kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<33:33, 72.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<23:35, 103kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<16:27, 146kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:05:03, 36.9kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:19<44:34, 52.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:19<31:09, 74.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<2:39:54, 14.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<1:51:59, 20.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:21<1:16:35, 29.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<55:22, 40.9kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<38:49, 58.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<26:56, 82.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<1:08:27, 32.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:24<46:32, 46.4kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<37:38, 57.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<26:36, 80.9kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<18:27, 115kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<59:05, 35.9kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:27<40:22, 51.3kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<29:08, 70.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<20:29, 99.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<14:16, 141kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<50:55, 39.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<34:41, 56.1kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<24:21, 79.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<16:55, 113kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<50:09, 38.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<34:06, 54.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<24:01, 76.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<16:38, 109kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<49:41, 36.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<34:11, 51.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:36<23:29, 73.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<1:58:40, 14.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<1:23:08, 20.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:38<56:30, 29.8kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<40:14, 41.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<28:12, 58.9kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<19:28, 83.8kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<49:56, 32.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<33:43, 46.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<23:45, 65.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<16:22, 93.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<42:15, 36.1kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:44<28:15, 51.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<1:47:53, 13.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<1:15:15, 19.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:46<50:32, 27.5kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<40:56, 33.9kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<28:43, 48.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<19:43, 68.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<41:42, 32.4kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<27:54, 46.0kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<19:33, 65.4kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<13:26, 92.9kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<35:27, 35.2kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<24:38, 50.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:52<16:28, 71.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<1:19:49, 14.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<55:12, 21.1kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<37:02, 29.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<25:51, 42.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<17:41, 60.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<36:03, 29.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<23:46, 42.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<16:42, 59.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<11:21, 85.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<28:14, 34.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<18:30, 48.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<12:55, 69.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<08:47, 98.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<25:37, 33.7kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<16:37, 47.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<11:41, 67.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<07:53, 96.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<22:29, 33.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<14:25, 47.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<10:05, 67.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<06:46, 96.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<18:13, 35.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<11:30, 50.8kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<08:01, 72.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<05:21, 103kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<16:03, 34.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<09:54, 48.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<06:52, 68.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<04:32, 97.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<12:28, 35.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<07:26, 50.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<05:08, 71.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<03:19, 102kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<09:55, 34.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<05:34, 48.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<03:49, 68.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<02:23, 97.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<06:56, 33.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<03:27, 47.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<02:21, 67.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<01:24, 96.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:57, 136kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<03:21, 38.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:06, 54.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:41, 77.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:13, 110kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:42, 34.5kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 716/400000 [00:00<00:55, 7156.66it/s]  0%|          | 1440/400000 [00:00<00:55, 7178.93it/s]  1%|          | 2157/400000 [00:00<00:55, 7173.29it/s]  1%|          | 2870/400000 [00:00<00:55, 7157.90it/s]  1%|          | 3592/400000 [00:00<00:55, 7175.27it/s]  1%|          | 4311/400000 [00:00<00:55, 7175.95it/s]  1%|▏         | 5028/400000 [00:00<00:55, 7174.09it/s]  1%|▏         | 5754/400000 [00:00<00:54, 7198.63it/s]  2%|▏         | 6478/400000 [00:00<00:54, 7209.39it/s]  2%|▏         | 7217/400000 [00:01<00:54, 7260.10it/s]  2%|▏         | 7923/400000 [00:01<00:56, 6907.46it/s]  2%|▏         | 8638/400000 [00:01<00:56, 6976.32it/s]  2%|▏         | 9364/400000 [00:01<00:55, 7057.31it/s]  3%|▎         | 10095/400000 [00:01<00:54, 7129.61it/s]  3%|▎         | 10817/400000 [00:01<00:54, 7155.83it/s]  3%|▎         | 11532/400000 [00:01<00:54, 7153.21it/s]  3%|▎         | 12269/400000 [00:01<00:53, 7216.34it/s]  3%|▎         | 13002/400000 [00:01<00:53, 7249.72it/s]  3%|▎         | 13727/400000 [00:01<00:54, 7147.07it/s]  4%|▎         | 14458/400000 [00:02<00:53, 7193.82it/s]  4%|▍         | 15178/400000 [00:02<00:53, 7195.09it/s]  4%|▍         | 15898/400000 [00:02<00:53, 7180.02it/s]  4%|▍         | 16617/400000 [00:02<00:53, 7107.09it/s]  4%|▍         | 17328/400000 [00:02<00:54, 7021.91it/s]  5%|▍         | 18031/400000 [00:02<00:54, 7017.80it/s]  5%|▍         | 18734/400000 [00:02<00:54, 7011.79it/s]  5%|▍         | 19436/400000 [00:02<00:54, 6997.60it/s]  5%|▌         | 20136/400000 [00:02<00:55, 6875.37it/s]  5%|▌         | 20836/400000 [00:02<00:54, 6911.00it/s]  5%|▌         | 21556/400000 [00:03<00:54, 6994.07it/s]  6%|▌         | 22265/400000 [00:03<00:53, 7021.72it/s]  6%|▌         | 22984/400000 [00:03<00:53, 7069.18it/s]  6%|▌         | 23699/400000 [00:03<00:53, 7092.85it/s]  6%|▌         | 24418/400000 [00:03<00:52, 7119.06it/s]  6%|▋         | 25159/400000 [00:03<00:52, 7202.04it/s]  6%|▋         | 25890/400000 [00:03<00:51, 7233.18it/s]  7%|▋         | 26637/400000 [00:03<00:51, 7301.11it/s]  7%|▋         | 27368/400000 [00:03<00:51, 7295.76it/s]  7%|▋         | 28098/400000 [00:03<00:51, 7247.33it/s]  7%|▋         | 28823/400000 [00:04<00:51, 7235.61it/s]  7%|▋         | 29547/400000 [00:04<00:51, 7220.01it/s]  8%|▊         | 30270/400000 [00:04<00:51, 7148.58it/s]  8%|▊         | 30993/400000 [00:04<00:51, 7171.10it/s]  8%|▊         | 31711/400000 [00:04<00:51, 7150.95it/s]  8%|▊         | 32435/400000 [00:04<00:51, 7175.37it/s]  8%|▊         | 33155/400000 [00:04<00:51, 7180.41it/s]  8%|▊         | 33874/400000 [00:04<00:51, 7070.90it/s]  9%|▊         | 34609/400000 [00:04<00:51, 7151.16it/s]  9%|▉         | 35335/400000 [00:04<00:50, 7183.29it/s]  9%|▉         | 36088/400000 [00:05<00:49, 7282.42it/s]  9%|▉         | 36817/400000 [00:05<00:49, 7266.39it/s]  9%|▉         | 37545/400000 [00:05<00:50, 7175.56it/s] 10%|▉         | 38264/400000 [00:05<00:51, 7017.07it/s] 10%|▉         | 38967/400000 [00:05<00:51, 7017.71it/s] 10%|▉         | 39696/400000 [00:05<00:50, 7094.22it/s] 10%|█         | 40424/400000 [00:05<00:50, 7147.39it/s] 10%|█         | 41149/400000 [00:05<00:50, 7175.13it/s] 10%|█         | 41867/400000 [00:05<00:49, 7163.32it/s] 11%|█         | 42584/400000 [00:05<00:50, 7141.98it/s] 11%|█         | 43306/400000 [00:06<00:49, 7163.57it/s] 11%|█         | 44023/400000 [00:06<00:49, 7148.60it/s] 11%|█         | 44738/400000 [00:06<00:50, 7101.43it/s] 11%|█▏        | 45466/400000 [00:06<00:49, 7148.36it/s] 12%|█▏        | 46182/400000 [00:06<00:49, 7141.12it/s] 12%|█▏        | 46910/400000 [00:06<00:49, 7180.67it/s] 12%|█▏        | 47659/400000 [00:06<00:48, 7267.09it/s] 12%|█▏        | 48392/400000 [00:06<00:48, 7283.76it/s] 12%|█▏        | 49132/400000 [00:06<00:47, 7318.15it/s] 12%|█▏        | 49865/400000 [00:06<00:48, 7242.37it/s] 13%|█▎        | 50590/400000 [00:07<00:48, 7223.62it/s] 13%|█▎        | 51318/400000 [00:07<00:48, 7240.29it/s] 13%|█▎        | 52043/400000 [00:07<00:48, 7184.19it/s] 13%|█▎        | 52762/400000 [00:07<00:48, 7142.87it/s] 13%|█▎        | 53477/400000 [00:07<00:48, 7094.32it/s] 14%|█▎        | 54187/400000 [00:07<00:48, 7076.08it/s] 14%|█▎        | 54896/400000 [00:07<00:48, 7079.58it/s] 14%|█▍        | 55612/400000 [00:07<00:48, 7100.99it/s] 14%|█▍        | 56323/400000 [00:07<00:48, 7047.74it/s] 14%|█▍        | 57028/400000 [00:07<00:48, 7019.61it/s] 14%|█▍        | 57745/400000 [00:08<00:48, 7062.52it/s] 15%|█▍        | 58452/400000 [00:08<00:48, 7046.18it/s] 15%|█▍        | 59162/400000 [00:08<00:48, 7060.04it/s] 15%|█▍        | 59879/400000 [00:08<00:47, 7089.49it/s] 15%|█▌        | 60589/400000 [00:08<00:48, 7068.27it/s] 15%|█▌        | 61310/400000 [00:08<00:47, 7107.68it/s] 16%|█▌        | 62033/400000 [00:08<00:47, 7143.71it/s] 16%|█▌        | 62748/400000 [00:08<00:47, 7094.58it/s] 16%|█▌        | 63470/400000 [00:08<00:47, 7129.31it/s] 16%|█▌        | 64184/400000 [00:08<00:47, 7122.58it/s] 16%|█▌        | 64908/400000 [00:09<00:46, 7156.57it/s] 16%|█▋        | 65624/400000 [00:09<00:46, 7154.81it/s] 17%|█▋        | 66340/400000 [00:09<00:46, 7143.04it/s] 17%|█▋        | 67061/400000 [00:09<00:46, 7162.71it/s] 17%|█▋        | 67785/400000 [00:09<00:46, 7185.16it/s] 17%|█▋        | 68504/400000 [00:09<00:46, 7056.50it/s] 17%|█▋        | 69229/400000 [00:09<00:46, 7112.74it/s] 17%|█▋        | 69945/400000 [00:09<00:46, 7124.85it/s] 18%|█▊        | 70667/400000 [00:09<00:46, 7151.38it/s] 18%|█▊        | 71386/400000 [00:09<00:45, 7161.10it/s] 18%|█▊        | 72124/400000 [00:10<00:45, 7224.48it/s] 18%|█▊        | 72852/400000 [00:10<00:45, 7240.74it/s] 18%|█▊        | 73577/400000 [00:10<00:45, 7199.41it/s] 19%|█▊        | 74298/400000 [00:10<00:45, 7179.36it/s] 19%|█▉        | 75017/400000 [00:10<00:46, 7033.62it/s] 19%|█▉        | 75722/400000 [00:10<00:46, 6990.56it/s] 19%|█▉        | 76437/400000 [00:10<00:45, 7036.39it/s] 19%|█▉        | 77142/400000 [00:10<00:45, 7021.33it/s] 19%|█▉        | 77845/400000 [00:10<00:46, 6981.80it/s] 20%|█▉        | 78544/400000 [00:11<00:46, 6948.50it/s] 20%|█▉        | 79260/400000 [00:11<00:45, 7010.28it/s] 20%|█▉        | 79982/400000 [00:11<00:45, 7070.15it/s] 20%|██        | 80690/400000 [00:11<00:45, 6947.16it/s] 20%|██        | 81415/400000 [00:11<00:45, 7033.31it/s] 21%|██        | 82129/400000 [00:11<00:44, 7064.12it/s] 21%|██        | 82848/400000 [00:11<00:44, 7098.91it/s] 21%|██        | 83582/400000 [00:11<00:44, 7168.57it/s] 21%|██        | 84318/400000 [00:11<00:43, 7223.17it/s] 21%|██▏       | 85041/400000 [00:11<00:43, 7224.76it/s] 21%|██▏       | 85764/400000 [00:12<00:43, 7176.04it/s] 22%|██▏       | 86482/400000 [00:12<00:43, 7166.81it/s] 22%|██▏       | 87209/400000 [00:12<00:43, 7196.17it/s] 22%|██▏       | 87929/400000 [00:12<00:43, 7162.93it/s] 22%|██▏       | 88646/400000 [00:12<00:43, 7148.58it/s] 22%|██▏       | 89361/400000 [00:12<00:43, 7120.74it/s] 23%|██▎       | 90090/400000 [00:12<00:43, 7170.39it/s] 23%|██▎       | 90808/400000 [00:12<00:43, 7127.36it/s] 23%|██▎       | 91521/400000 [00:12<00:43, 7091.21it/s] 23%|██▎       | 92231/400000 [00:12<00:43, 7093.39it/s] 23%|██▎       | 92941/400000 [00:13<00:43, 7072.40it/s] 23%|██▎       | 93649/400000 [00:13<00:43, 7066.99it/s] 24%|██▎       | 94356/400000 [00:13<00:43, 7057.84it/s] 24%|██▍       | 95082/400000 [00:13<00:42, 7115.30it/s] 24%|██▍       | 95818/400000 [00:13<00:42, 7185.21it/s] 24%|██▍       | 96537/400000 [00:13<00:42, 7126.15it/s] 24%|██▍       | 97250/400000 [00:13<00:42, 7069.05it/s] 24%|██▍       | 97958/400000 [00:13<00:42, 7068.28it/s] 25%|██▍       | 98666/400000 [00:13<00:42, 7014.26it/s] 25%|██▍       | 99376/400000 [00:13<00:42, 7038.73it/s] 25%|██▌       | 100082/400000 [00:14<00:42, 7043.92it/s] 25%|██▌       | 100787/400000 [00:14<00:42, 7034.07it/s] 25%|██▌       | 101491/400000 [00:14<00:42, 7032.73it/s] 26%|██▌       | 102195/400000 [00:14<00:42, 7030.71it/s] 26%|██▌       | 102917/400000 [00:14<00:41, 7085.23it/s] 26%|██▌       | 103626/400000 [00:14<00:41, 7057.41it/s] 26%|██▌       | 104332/400000 [00:14<00:42, 7033.76it/s] 26%|██▋       | 105047/400000 [00:14<00:41, 7067.33it/s] 26%|██▋       | 105768/400000 [00:14<00:41, 7107.01it/s] 27%|██▋       | 106480/400000 [00:14<00:41, 7109.04it/s] 27%|██▋       | 107203/400000 [00:15<00:40, 7144.36it/s] 27%|██▋       | 107918/400000 [00:15<00:41, 7089.61it/s] 27%|██▋       | 108629/400000 [00:15<00:41, 7095.00it/s] 27%|██▋       | 109363/400000 [00:15<00:40, 7164.45it/s] 28%|██▊       | 110080/400000 [00:15<00:40, 7155.53it/s] 28%|██▊       | 110797/400000 [00:15<00:40, 7158.75it/s] 28%|██▊       | 111513/400000 [00:15<00:40, 7065.59it/s] 28%|██▊       | 112234/400000 [00:15<00:40, 7106.14it/s] 28%|██▊       | 112951/400000 [00:15<00:40, 7122.42it/s] 28%|██▊       | 113664/400000 [00:15<00:40, 7095.98it/s] 29%|██▊       | 114374/400000 [00:16<00:40, 7091.95it/s] 29%|██▉       | 115108/400000 [00:16<00:39, 7164.01it/s] 29%|██▉       | 115829/400000 [00:16<00:39, 7175.68it/s] 29%|██▉       | 116547/400000 [00:16<00:39, 7168.35it/s] 29%|██▉       | 117269/400000 [00:16<00:39, 7181.22it/s] 29%|██▉       | 117988/400000 [00:16<00:39, 7143.82it/s] 30%|██▉       | 118703/400000 [00:16<00:40, 6945.98it/s] 30%|██▉       | 119399/400000 [00:16<00:40, 6944.86it/s] 30%|███       | 120095/400000 [00:16<00:40, 6919.79it/s] 30%|███       | 120788/400000 [00:16<00:40, 6915.43it/s] 30%|███       | 121480/400000 [00:17<00:40, 6893.95it/s] 31%|███       | 122185/400000 [00:17<00:40, 6937.56it/s] 31%|███       | 122890/400000 [00:17<00:39, 6969.59it/s] 31%|███       | 123591/400000 [00:17<00:39, 6981.23it/s] 31%|███       | 124297/400000 [00:17<00:39, 7002.42it/s] 31%|███       | 124998/400000 [00:17<00:39, 6971.04it/s] 31%|███▏      | 125696/400000 [00:17<00:39, 6935.60it/s] 32%|███▏      | 126392/400000 [00:17<00:39, 6940.74it/s] 32%|███▏      | 127087/400000 [00:17<00:39, 6935.18it/s] 32%|███▏      | 127791/400000 [00:17<00:39, 6965.73it/s] 32%|███▏      | 128488/400000 [00:18<00:39, 6942.16it/s] 32%|███▏      | 129183/400000 [00:18<00:39, 6909.78it/s] 32%|███▏      | 129882/400000 [00:18<00:38, 6929.68it/s] 33%|███▎      | 130590/400000 [00:18<00:38, 6971.67it/s] 33%|███▎      | 131296/400000 [00:18<00:38, 6996.19it/s] 33%|███▎      | 131996/400000 [00:18<00:38, 6979.12it/s] 33%|███▎      | 132702/400000 [00:18<00:38, 7002.52it/s] 33%|███▎      | 133406/400000 [00:18<00:38, 7010.67it/s] 34%|███▎      | 134109/400000 [00:18<00:37, 7013.99it/s] 34%|███▎      | 134815/400000 [00:18<00:37, 7025.46it/s] 34%|███▍      | 135532/400000 [00:19<00:37, 7066.73it/s] 34%|███▍      | 136242/400000 [00:19<00:37, 7074.41it/s] 34%|███▍      | 136953/400000 [00:19<00:37, 7084.82it/s] 34%|███▍      | 137662/400000 [00:19<00:37, 7068.31it/s] 35%|███▍      | 138369/400000 [00:19<00:37, 7040.13it/s] 35%|███▍      | 139081/400000 [00:19<00:36, 7062.64it/s] 35%|███▍      | 139788/400000 [00:19<00:36, 7054.27it/s] 35%|███▌      | 140508/400000 [00:19<00:36, 7094.74it/s] 35%|███▌      | 141218/400000 [00:19<00:36, 7094.05it/s] 35%|███▌      | 141928/400000 [00:19<00:36, 7073.70it/s] 36%|███▌      | 142636/400000 [00:20<00:37, 6915.40it/s] 36%|███▌      | 143371/400000 [00:20<00:36, 7037.80it/s] 36%|███▌      | 144089/400000 [00:20<00:36, 7077.36it/s] 36%|███▌      | 144798/400000 [00:20<00:36, 7059.49it/s] 36%|███▋      | 145507/400000 [00:20<00:36, 7068.42it/s] 37%|███▋      | 146215/400000 [00:20<00:35, 7070.28it/s] 37%|███▋      | 146938/400000 [00:20<00:35, 7117.20it/s] 37%|███▋      | 147673/400000 [00:20<00:35, 7183.32it/s] 37%|███▋      | 148392/400000 [00:20<00:35, 7176.75it/s] 37%|███▋      | 149113/400000 [00:20<00:34, 7184.20it/s] 37%|███▋      | 149845/400000 [00:21<00:34, 7223.06it/s] 38%|███▊      | 150568/400000 [00:21<00:34, 7192.91it/s] 38%|███▊      | 151293/400000 [00:21<00:34, 7207.81it/s] 38%|███▊      | 152014/400000 [00:21<00:34, 7194.36it/s] 38%|███▊      | 152734/400000 [00:21<00:34, 7195.19it/s] 38%|███▊      | 153464/400000 [00:21<00:34, 7224.11it/s] 39%|███▊      | 154187/400000 [00:21<00:34, 7194.06it/s] 39%|███▊      | 154907/400000 [00:21<00:34, 7175.24it/s] 39%|███▉      | 155632/400000 [00:21<00:33, 7195.14it/s] 39%|███▉      | 156369/400000 [00:22<00:33, 7245.64it/s] 39%|███▉      | 157097/400000 [00:22<00:33, 7255.46it/s] 39%|███▉      | 157823/400000 [00:22<00:33, 7244.56it/s] 40%|███▉      | 158548/400000 [00:22<00:33, 7197.83it/s] 40%|███▉      | 159268/400000 [00:22<00:33, 7190.45it/s] 40%|████      | 160008/400000 [00:22<00:33, 7249.73it/s] 40%|████      | 160734/400000 [00:22<00:33, 7091.66it/s] 40%|████      | 161445/400000 [00:22<00:33, 7041.69it/s] 41%|████      | 162164/400000 [00:22<00:33, 7083.31it/s] 41%|████      | 162873/400000 [00:22<00:33, 7068.71it/s] 41%|████      | 163601/400000 [00:23<00:33, 7128.40it/s] 41%|████      | 164315/400000 [00:23<00:33, 7082.05it/s] 41%|████▏     | 165024/400000 [00:23<00:34, 6894.49it/s] 41%|████▏     | 165746/400000 [00:23<00:33, 6986.40it/s] 42%|████▏     | 166447/400000 [00:23<00:33, 6990.76it/s] 42%|████▏     | 167177/400000 [00:23<00:32, 7079.77it/s] 42%|████▏     | 167900/400000 [00:23<00:32, 7124.01it/s] 42%|████▏     | 168615/400000 [00:23<00:32, 7129.55it/s] 42%|████▏     | 169340/400000 [00:23<00:32, 7164.38it/s] 43%|████▎     | 170059/400000 [00:23<00:32, 7171.56it/s] 43%|████▎     | 170777/400000 [00:24<00:32, 7161.69it/s] 43%|████▎     | 171495/400000 [00:24<00:31, 7166.91it/s] 43%|████▎     | 172212/400000 [00:24<00:31, 7132.33it/s] 43%|████▎     | 172926/400000 [00:24<00:31, 7097.55it/s] 43%|████▎     | 173649/400000 [00:24<00:31, 7134.69it/s] 44%|████▎     | 174363/400000 [00:24<00:31, 7131.50it/s] 44%|████▍     | 175077/400000 [00:24<00:31, 7101.20it/s] 44%|████▍     | 175788/400000 [00:24<00:31, 7088.61it/s] 44%|████▍     | 176497/400000 [00:24<00:31, 7050.12it/s] 44%|████▍     | 177203/400000 [00:24<00:31, 7051.23it/s] 44%|████▍     | 177909/400000 [00:25<00:31, 7025.24it/s] 45%|████▍     | 178615/400000 [00:25<00:31, 7033.86it/s] 45%|████▍     | 179330/400000 [00:25<00:31, 7066.68it/s] 45%|████▌     | 180053/400000 [00:25<00:30, 7114.15it/s] 45%|████▌     | 180765/400000 [00:25<00:30, 7115.46it/s] 45%|████▌     | 181485/400000 [00:25<00:30, 7140.35it/s] 46%|████▌     | 182209/400000 [00:25<00:30, 7168.17it/s] 46%|████▌     | 182930/400000 [00:25<00:30, 7178.35it/s] 46%|████▌     | 183648/400000 [00:25<00:30, 7112.01it/s] 46%|████▌     | 184360/400000 [00:25<00:30, 7078.91it/s] 46%|████▋     | 185070/400000 [00:26<00:30, 7083.92it/s] 46%|████▋     | 185779/400000 [00:26<00:30, 7078.21it/s] 47%|████▋     | 186493/400000 [00:26<00:30, 7094.73it/s] 47%|████▋     | 187203/400000 [00:26<00:30, 7022.89it/s] 47%|████▋     | 187906/400000 [00:26<00:30, 7021.23it/s] 47%|████▋     | 188620/400000 [00:26<00:29, 7056.18it/s] 47%|████▋     | 189338/400000 [00:26<00:29, 7091.00it/s] 48%|████▊     | 190048/400000 [00:26<00:29, 7061.64it/s] 48%|████▊     | 190755/400000 [00:26<00:29, 7018.34it/s] 48%|████▊     | 191462/400000 [00:26<00:29, 7030.94it/s] 48%|████▊     | 192179/400000 [00:27<00:29, 7069.97it/s] 48%|████▊     | 192904/400000 [00:27<00:29, 7120.62it/s] 48%|████▊     | 193634/400000 [00:27<00:28, 7172.13it/s] 49%|████▊     | 194352/400000 [00:27<00:28, 7147.23it/s] 49%|████▉     | 195069/400000 [00:27<00:28, 7151.96it/s] 49%|████▉     | 195795/400000 [00:27<00:28, 7182.81it/s] 49%|████▉     | 196514/400000 [00:27<00:28, 7142.83it/s] 49%|████▉     | 197229/400000 [00:27<00:28, 7104.44it/s] 49%|████▉     | 197940/400000 [00:27<00:28, 7068.67it/s] 50%|████▉     | 198648/400000 [00:27<00:28, 7030.17it/s] 50%|████▉     | 199352/400000 [00:28<00:28, 6995.68it/s] 50%|█████     | 200065/400000 [00:28<00:28, 7033.22it/s] 50%|█████     | 200775/400000 [00:28<00:28, 7051.11it/s] 50%|█████     | 201481/400000 [00:28<00:28, 7039.82it/s] 51%|█████     | 202200/400000 [00:28<00:27, 7082.13it/s] 51%|█████     | 202910/400000 [00:28<00:27, 7084.99it/s] 51%|█████     | 203624/400000 [00:28<00:27, 7099.03it/s] 51%|█████     | 204353/400000 [00:28<00:27, 7151.73it/s] 51%|█████▏    | 205069/400000 [00:28<00:27, 7137.37it/s] 51%|█████▏    | 205800/400000 [00:28<00:27, 7185.63it/s] 52%|█████▏    | 206534/400000 [00:29<00:26, 7230.86it/s] 52%|█████▏    | 207258/400000 [00:29<00:26, 7220.14it/s] 52%|█████▏    | 207981/400000 [00:29<00:26, 7163.62it/s] 52%|█████▏    | 208698/400000 [00:29<00:26, 7109.89it/s] 52%|█████▏    | 209418/400000 [00:29<00:26, 7135.75it/s] 53%|█████▎    | 210137/400000 [00:29<00:26, 7151.01it/s] 53%|█████▎    | 210857/400000 [00:29<00:26, 7164.60it/s] 53%|█████▎    | 211574/400000 [00:29<00:26, 7125.48it/s] 53%|█████▎    | 212287/400000 [00:29<00:26, 7116.32it/s] 53%|█████▎    | 212999/400000 [00:29<00:26, 7033.67it/s] 53%|█████▎    | 213703/400000 [00:30<00:26, 7029.30it/s] 54%|█████▎    | 214407/400000 [00:30<00:26, 6988.54it/s] 54%|█████▍    | 215123/400000 [00:30<00:26, 7037.69it/s] 54%|█████▍    | 215827/400000 [00:30<00:26, 7035.61it/s] 54%|█████▍    | 216538/400000 [00:30<00:26, 7055.67it/s] 54%|█████▍    | 217263/400000 [00:30<00:25, 7112.51it/s] 54%|█████▍    | 217980/400000 [00:30<00:25, 7128.47it/s] 55%|█████▍    | 218701/400000 [00:30<00:25, 7152.20it/s] 55%|█████▍    | 219417/400000 [00:30<00:25, 7133.32it/s] 55%|█████▌    | 220131/400000 [00:30<00:25, 7118.87it/s] 55%|█████▌    | 220843/400000 [00:31<00:25, 6990.30it/s] 55%|█████▌    | 221545/400000 [00:31<00:25, 6997.42it/s] 56%|█████▌    | 222255/400000 [00:31<00:25, 7027.24it/s] 56%|█████▌    | 222959/400000 [00:31<00:25, 6991.41it/s] 56%|█████▌    | 223659/400000 [00:31<00:25, 6951.20it/s] 56%|█████▌    | 224373/400000 [00:31<00:25, 7005.10it/s] 56%|█████▋    | 225074/400000 [00:31<00:25, 6973.76it/s] 56%|█████▋    | 225779/400000 [00:31<00:24, 6996.42it/s] 57%|█████▋    | 226483/400000 [00:31<00:24, 7007.73it/s] 57%|█████▋    | 227184/400000 [00:31<00:24, 6942.30it/s] 57%|█████▋    | 227879/400000 [00:32<00:25, 6879.49it/s] 57%|█████▋    | 228599/400000 [00:32<00:24, 6971.12it/s] 57%|█████▋    | 229306/400000 [00:32<00:24, 7000.05it/s] 58%|█████▊    | 230009/400000 [00:32<00:24, 7007.63it/s] 58%|█████▊    | 230711/400000 [00:32<00:24, 6952.55it/s] 58%|█████▊    | 231407/400000 [00:32<00:24, 6867.43it/s] 58%|█████▊    | 232102/400000 [00:32<00:24, 6891.55it/s] 58%|█████▊    | 232814/400000 [00:32<00:24, 6956.15it/s] 58%|█████▊    | 233510/400000 [00:32<00:23, 6946.86it/s] 59%|█████▊    | 234224/400000 [00:33<00:23, 7003.20it/s] 59%|█████▊    | 234934/400000 [00:33<00:23, 7030.92it/s] 59%|█████▉    | 235638/400000 [00:33<00:23, 7014.21it/s] 59%|█████▉    | 236340/400000 [00:33<00:23, 6984.60it/s] 59%|█████▉    | 237052/400000 [00:33<00:23, 7021.95it/s] 59%|█████▉    | 237759/400000 [00:33<00:23, 7036.24it/s] 60%|█████▉    | 238474/400000 [00:33<00:22, 7069.87it/s] 60%|█████▉    | 239187/400000 [00:33<00:22, 7085.01it/s] 60%|█████▉    | 239913/400000 [00:33<00:22, 7134.95it/s] 60%|██████    | 240628/400000 [00:33<00:22, 7138.44it/s] 60%|██████    | 241361/400000 [00:34<00:22, 7192.98it/s] 61%|██████    | 242109/400000 [00:34<00:21, 7274.33it/s] 61%|██████    | 242867/400000 [00:34<00:21, 7360.19it/s] 61%|██████    | 243620/400000 [00:34<00:21, 7407.92it/s] 61%|██████    | 244362/400000 [00:34<00:21, 7299.85it/s] 61%|██████▏   | 245093/400000 [00:34<00:21, 7287.15it/s] 61%|██████▏   | 245823/400000 [00:34<00:21, 7285.48it/s] 62%|██████▏   | 246552/400000 [00:34<00:21, 7275.74it/s] 62%|██████▏   | 247280/400000 [00:34<00:21, 7246.75it/s] 62%|██████▏   | 248005/400000 [00:34<00:21, 7209.63it/s] 62%|██████▏   | 248727/400000 [00:35<00:21, 7173.16it/s] 62%|██████▏   | 249445/400000 [00:35<00:21, 7160.74it/s] 63%|██████▎   | 250162/400000 [00:35<00:21, 7133.81it/s] 63%|██████▎   | 250876/400000 [00:35<00:20, 7116.95it/s] 63%|██████▎   | 251592/400000 [00:35<00:20, 7129.24it/s] 63%|██████▎   | 252316/400000 [00:35<00:20, 7159.79it/s] 63%|██████▎   | 253033/400000 [00:35<00:20, 7128.21it/s] 63%|██████▎   | 253746/400000 [00:35<00:20, 7082.39it/s] 64%|██████▎   | 254455/400000 [00:35<00:20, 7055.21it/s] 64%|██████▍   | 255167/400000 [00:35<00:20, 7071.72it/s] 64%|██████▍   | 255878/400000 [00:36<00:20, 7082.73it/s] 64%|██████▍   | 256593/400000 [00:36<00:20, 7100.81it/s] 64%|██████▍   | 257306/400000 [00:36<00:20, 7107.73it/s] 65%|██████▍   | 258017/400000 [00:36<00:19, 7107.18it/s] 65%|██████▍   | 258728/400000 [00:36<00:19, 7066.91it/s] 65%|██████▍   | 259450/400000 [00:36<00:19, 7110.23it/s] 65%|██████▌   | 260173/400000 [00:36<00:19, 7144.29it/s] 65%|██████▌   | 260898/400000 [00:36<00:19, 7174.56it/s] 65%|██████▌   | 261626/400000 [00:36<00:19, 7204.92it/s] 66%|██████▌   | 262350/400000 [00:36<00:19, 7213.76it/s] 66%|██████▌   | 263073/400000 [00:37<00:18, 7216.96it/s] 66%|██████▌   | 263795/400000 [00:37<00:18, 7203.74it/s] 66%|██████▌   | 264516/400000 [00:37<00:18, 7147.57it/s] 66%|██████▋   | 265231/400000 [00:37<00:18, 7131.64it/s] 66%|██████▋   | 265945/400000 [00:37<00:18, 7099.29it/s] 67%|██████▋   | 266656/400000 [00:37<00:18, 7071.65it/s] 67%|██████▋   | 267364/400000 [00:37<00:19, 6924.69it/s] 67%|██████▋   | 268076/400000 [00:37<00:18, 6982.02it/s] 67%|██████▋   | 268782/400000 [00:37<00:18, 7003.95it/s] 67%|██████▋   | 269483/400000 [00:37<00:18, 6985.82it/s] 68%|██████▊   | 270190/400000 [00:38<00:18, 7010.19it/s] 68%|██████▊   | 270892/400000 [00:38<00:18, 7003.33it/s] 68%|██████▊   | 271611/400000 [00:38<00:18, 7055.72it/s] 68%|██████▊   | 272329/400000 [00:38<00:18, 7090.58it/s] 68%|██████▊   | 273039/400000 [00:38<00:17, 7090.98it/s] 68%|██████▊   | 273764/400000 [00:38<00:17, 7137.60it/s] 69%|██████▊   | 274492/400000 [00:38<00:17, 7176.87it/s] 69%|██████▉   | 275210/400000 [00:38<00:17, 7076.96it/s] 69%|██████▉   | 275920/400000 [00:38<00:17, 7082.08it/s] 69%|██████▉   | 276629/400000 [00:38<00:17, 7048.42it/s] 69%|██████▉   | 277340/400000 [00:39<00:17, 7058.91it/s] 70%|██████▉   | 278050/400000 [00:39<00:17, 7069.70it/s] 70%|██████▉   | 278778/400000 [00:39<00:17, 7129.35it/s] 70%|██████▉   | 279492/400000 [00:39<00:17, 6943.15it/s] 70%|███████   | 280202/400000 [00:39<00:17, 6987.77it/s] 70%|███████   | 280917/400000 [00:39<00:16, 7033.20it/s] 70%|███████   | 281623/400000 [00:39<00:16, 7040.79it/s] 71%|███████   | 282345/400000 [00:39<00:16, 7090.83it/s] 71%|███████   | 283055/400000 [00:39<00:16, 6965.02it/s] 71%|███████   | 283753/400000 [00:39<00:16, 6906.50it/s] 71%|███████   | 284446/400000 [00:40<00:16, 6912.01it/s] 71%|███████▏  | 285141/400000 [00:40<00:16, 6922.01it/s] 71%|███████▏  | 285850/400000 [00:40<00:16, 6969.40it/s] 72%|███████▏  | 286548/400000 [00:40<00:16, 6964.01it/s] 72%|███████▏  | 287253/400000 [00:40<00:16, 6988.17it/s] 72%|███████▏  | 287968/400000 [00:40<00:15, 7034.21it/s] 72%|███████▏  | 288672/400000 [00:40<00:15, 6992.05it/s] 72%|███████▏  | 289392/400000 [00:40<00:15, 7051.40it/s] 73%|███████▎  | 290108/400000 [00:40<00:15, 7083.51it/s] 73%|███████▎  | 290822/400000 [00:40<00:15, 7099.34it/s] 73%|███████▎  | 291550/400000 [00:41<00:15, 7150.42it/s] 73%|███████▎  | 292266/400000 [00:41<00:15, 7141.44it/s] 73%|███████▎  | 292993/400000 [00:41<00:14, 7177.74it/s] 73%|███████▎  | 293743/400000 [00:41<00:14, 7267.97it/s] 74%|███████▎  | 294472/400000 [00:41<00:14, 7271.39it/s] 74%|███████▍  | 295200/400000 [00:41<00:14, 7215.98it/s] 74%|███████▍  | 295922/400000 [00:41<00:14, 7151.33it/s] 74%|███████▍  | 296638/400000 [00:41<00:14, 7109.26it/s] 74%|███████▍  | 297350/400000 [00:41<00:14, 7110.25it/s] 75%|███████▍  | 298062/400000 [00:41<00:14, 7084.34it/s] 75%|███████▍  | 298776/400000 [00:42<00:14, 7099.77it/s] 75%|███████▍  | 299493/400000 [00:42<00:14, 7120.35it/s] 75%|███████▌  | 300216/400000 [00:42<00:13, 7150.80it/s] 75%|███████▌  | 300932/400000 [00:42<00:13, 7128.30it/s] 75%|███████▌  | 301651/400000 [00:42<00:13, 7145.53it/s] 76%|███████▌  | 302366/400000 [00:42<00:13, 7121.34it/s] 76%|███████▌  | 303079/400000 [00:42<00:13, 6949.54it/s] 76%|███████▌  | 303785/400000 [00:42<00:13, 6980.92it/s] 76%|███████▌  | 304513/400000 [00:42<00:13, 7067.31it/s] 76%|███████▋  | 305233/400000 [00:43<00:13, 7105.47it/s] 76%|███████▋  | 305945/400000 [00:43<00:13, 7096.78it/s] 77%|███████▋  | 306676/400000 [00:43<00:13, 7158.45it/s] 77%|███████▋  | 307397/400000 [00:43<00:12, 7171.52it/s] 77%|███████▋  | 308115/400000 [00:43<00:13, 7048.56it/s] 77%|███████▋  | 308821/400000 [00:43<00:12, 7051.41it/s] 77%|███████▋  | 309527/400000 [00:43<00:13, 6939.96it/s] 78%|███████▊  | 310228/400000 [00:43<00:12, 6958.55it/s] 78%|███████▊  | 310925/400000 [00:43<00:12, 6957.23it/s] 78%|███████▊  | 311633/400000 [00:43<00:12, 6993.03it/s] 78%|███████▊  | 312336/400000 [00:44<00:12, 7003.80it/s] 78%|███████▊  | 313037/400000 [00:44<00:12, 6969.49it/s] 78%|███████▊  | 313741/400000 [00:44<00:12, 6990.47it/s] 79%|███████▊  | 314460/400000 [00:44<00:12, 7046.94it/s] 79%|███████▉  | 315194/400000 [00:44<00:11, 7129.82it/s] 79%|███████▉  | 315908/400000 [00:44<00:11, 7131.33it/s] 79%|███████▉  | 316622/400000 [00:44<00:11, 7114.76it/s] 79%|███████▉  | 317337/400000 [00:44<00:11, 7124.27it/s] 80%|███████▉  | 318050/400000 [00:44<00:11, 7008.22it/s] 80%|███████▉  | 318752/400000 [00:44<00:11, 6894.37it/s] 80%|███████▉  | 319486/400000 [00:45<00:11, 7020.33it/s] 80%|████████  | 320212/400000 [00:45<00:11, 7088.24it/s] 80%|████████  | 320939/400000 [00:45<00:11, 7140.51it/s] 80%|████████  | 321660/400000 [00:45<00:10, 7159.26it/s] 81%|████████  | 322377/400000 [00:45<00:10, 7122.24it/s] 81%|████████  | 323094/400000 [00:45<00:10, 7135.80it/s] 81%|████████  | 323808/400000 [00:45<00:10, 7057.03it/s] 81%|████████  | 324518/400000 [00:45<00:10, 7066.92it/s] 81%|████████▏ | 325250/400000 [00:45<00:10, 7139.91it/s] 81%|████████▏ | 325965/400000 [00:45<00:10, 7076.23it/s] 82%|████████▏ | 326709/400000 [00:46<00:10, 7181.46it/s] 82%|████████▏ | 327449/400000 [00:46<00:10, 7244.97it/s] 82%|████████▏ | 328200/400000 [00:46<00:09, 7322.24it/s] 82%|████████▏ | 328948/400000 [00:46<00:09, 7367.99it/s] 82%|████████▏ | 329705/400000 [00:46<00:09, 7427.41it/s] 83%|████████▎ | 330449/400000 [00:46<00:09, 7350.93it/s] 83%|████████▎ | 331185/400000 [00:46<00:09, 7233.12it/s] 83%|████████▎ | 331915/400000 [00:46<00:09, 7250.86it/s] 83%|████████▎ | 332641/400000 [00:46<00:09, 7224.23it/s] 83%|████████▎ | 333364/400000 [00:46<00:09, 7138.35it/s] 84%|████████▎ | 334079/400000 [00:47<00:09, 7127.36it/s] 84%|████████▎ | 334793/400000 [00:47<00:09, 7121.96it/s] 84%|████████▍ | 335512/400000 [00:47<00:09, 7140.55it/s] 84%|████████▍ | 336227/400000 [00:47<00:08, 7131.57it/s] 84%|████████▍ | 336941/400000 [00:47<00:08, 7114.10it/s] 84%|████████▍ | 337657/400000 [00:47<00:08, 7126.37it/s] 85%|████████▍ | 338370/400000 [00:47<00:08, 7096.18it/s] 85%|████████▍ | 339093/400000 [00:47<00:08, 7135.26it/s] 85%|████████▍ | 339818/400000 [00:47<00:08, 7168.59it/s] 85%|████████▌ | 340535/400000 [00:47<00:08, 7116.38it/s] 85%|████████▌ | 341247/400000 [00:48<00:08, 7025.86it/s] 85%|████████▌ | 341950/400000 [00:48<00:08, 7018.64it/s] 86%|████████▌ | 342653/400000 [00:48<00:08, 7021.04it/s] 86%|████████▌ | 343360/400000 [00:48<00:08, 7032.93it/s] 86%|████████▌ | 344064/400000 [00:48<00:07, 7013.30it/s] 86%|████████▌ | 344767/400000 [00:48<00:07, 7018.17it/s] 86%|████████▋ | 345470/400000 [00:48<00:07, 7021.39it/s] 87%|████████▋ | 346173/400000 [00:48<00:07, 7010.26it/s] 87%|████████▋ | 346882/400000 [00:48<00:07, 7032.92it/s] 87%|████████▋ | 347586/400000 [00:48<00:07, 7011.10it/s] 87%|████████▋ | 348302/400000 [00:49<00:07, 7053.00it/s] 87%|████████▋ | 349008/400000 [00:49<00:07, 7035.91it/s] 87%|████████▋ | 349728/400000 [00:49<00:07, 7083.33it/s] 88%|████████▊ | 350446/400000 [00:49<00:06, 7110.24it/s] 88%|████████▊ | 351183/400000 [00:49<00:06, 7186.13it/s] 88%|████████▊ | 351908/400000 [00:49<00:06, 7204.52it/s] 88%|████████▊ | 352630/400000 [00:49<00:06, 7207.02it/s] 88%|████████▊ | 353369/400000 [00:49<00:06, 7259.37it/s] 89%|████████▊ | 354096/400000 [00:49<00:06, 7253.45it/s] 89%|████████▊ | 354822/400000 [00:49<00:06, 7084.19it/s] 89%|████████▉ | 355532/400000 [00:50<00:06, 6942.12it/s] 89%|████████▉ | 356235/400000 [00:50<00:06, 6967.49it/s] 89%|████████▉ | 356963/400000 [00:50<00:06, 7056.00it/s] 89%|████████▉ | 357678/400000 [00:50<00:05, 7083.64it/s] 90%|████████▉ | 358391/400000 [00:50<00:05, 7097.40it/s] 90%|████████▉ | 359102/400000 [00:50<00:05, 7095.92it/s] 90%|████████▉ | 359812/400000 [00:50<00:05, 7034.02it/s] 90%|█████████ | 360516/400000 [00:50<00:05, 7026.12it/s] 90%|█████████ | 361219/400000 [00:50<00:05, 7024.62it/s] 90%|█████████ | 361930/400000 [00:50<00:05, 7049.24it/s] 91%|█████████ | 362653/400000 [00:51<00:05, 7101.52it/s] 91%|█████████ | 363364/400000 [00:51<00:05, 7075.63it/s] 91%|█████████ | 364076/400000 [00:51<00:05, 7087.29it/s] 91%|█████████ | 364785/400000 [00:51<00:04, 7067.51it/s] 91%|█████████▏| 365494/400000 [00:51<00:04, 7072.28it/s] 92%|█████████▏| 366202/400000 [00:51<00:04, 7010.91it/s] 92%|█████████▏| 366904/400000 [00:51<00:04, 6969.40it/s] 92%|█████████▏| 367602/400000 [00:51<00:04, 6952.74it/s] 92%|█████████▏| 368316/400000 [00:51<00:04, 7006.95it/s] 92%|█████████▏| 369017/400000 [00:51<00:04, 7002.34it/s] 92%|█████████▏| 369718/400000 [00:52<00:04, 7000.48it/s] 93%|█████████▎| 370419/400000 [00:52<00:04, 6938.10it/s] 93%|█████████▎| 371119/400000 [00:52<00:04, 6956.38it/s] 93%|█████████▎| 371815/400000 [00:52<00:04, 6942.84it/s] 93%|█████████▎| 372544/400000 [00:52<00:03, 7043.09it/s] 93%|█████████▎| 373249/400000 [00:52<00:03, 7011.41it/s] 93%|█████████▎| 373954/400000 [00:52<00:03, 7020.65it/s] 94%|█████████▎| 374669/400000 [00:52<00:03, 7057.62it/s] 94%|█████████▍| 375380/400000 [00:52<00:03, 7069.50it/s] 94%|█████████▍| 376097/400000 [00:53<00:03, 7096.93it/s] 94%|█████████▍| 376808/400000 [00:53<00:03, 7100.16it/s] 94%|█████████▍| 377519/400000 [00:53<00:03, 7021.16it/s] 95%|█████████▍| 378238/400000 [00:53<00:03, 7067.98it/s] 95%|█████████▍| 378950/400000 [00:53<00:02, 7082.20it/s] 95%|█████████▍| 379671/400000 [00:53<00:02, 7118.30it/s] 95%|█████████▌| 380391/400000 [00:53<00:02, 7140.64it/s] 95%|█████████▌| 381106/400000 [00:53<00:02, 7072.32it/s] 95%|█████████▌| 381824/400000 [00:53<00:02, 7102.37it/s] 96%|█████████▌| 382547/400000 [00:53<00:02, 7140.11it/s] 96%|█████████▌| 383283/400000 [00:54<00:02, 7200.70it/s] 96%|█████████▌| 384024/400000 [00:54<00:02, 7260.15it/s] 96%|█████████▌| 384751/400000 [00:54<00:02, 7207.92it/s] 96%|█████████▋| 385477/400000 [00:54<00:02, 7221.69it/s] 97%|█████████▋| 386210/400000 [00:54<00:01, 7252.81it/s] 97%|█████████▋| 386943/400000 [00:54<00:01, 7272.48it/s] 97%|█████████▋| 387677/400000 [00:54<00:01, 7291.91it/s] 97%|█████████▋| 388407/400000 [00:54<00:01, 7242.51it/s] 97%|█████████▋| 389146/400000 [00:54<00:01, 7282.88it/s] 97%|█████████▋| 389875/400000 [00:54<00:01, 7273.56it/s] 98%|█████████▊| 390603/400000 [00:55<00:01, 7225.99it/s] 98%|█████████▊| 391326/400000 [00:55<00:01, 7193.41it/s] 98%|█████████▊| 392046/400000 [00:55<00:01, 7139.62it/s] 98%|█████████▊| 392761/400000 [00:55<00:01, 7124.34it/s] 98%|█████████▊| 393474/400000 [00:55<00:00, 7121.84it/s] 99%|█████████▊| 394187/400000 [00:55<00:00, 7122.31it/s] 99%|█████████▊| 394900/400000 [00:55<00:00, 7100.71it/s] 99%|█████████▉| 395611/400000 [00:55<00:00, 7073.26it/s] 99%|█████████▉| 396335/400000 [00:55<00:00, 7120.51it/s] 99%|█████████▉| 397053/400000 [00:55<00:00, 7136.71it/s] 99%|█████████▉| 397767/400000 [00:56<00:00, 7119.30it/s]100%|█████████▉| 398486/400000 [00:56<00:00, 7138.95it/s]100%|█████████▉| 399200/400000 [00:56<00:00, 7083.33it/s]100%|█████████▉| 399913/400000 [00:56<00:00, 7096.76it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7099.32it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7efbe6118c50> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011162675671993577 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011237217032391092 	 Accuracy: 49

  model saves at 49% accuracy 

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

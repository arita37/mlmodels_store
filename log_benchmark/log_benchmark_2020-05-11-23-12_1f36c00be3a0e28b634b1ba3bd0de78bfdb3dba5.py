
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f85368eafd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 23:12:49.596743
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 23:12:49.601384
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 23:12:49.605260
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 23:12:49.609202
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f8542902438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353132.1250
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 246750.6719
Epoch 3/10

1/1 [==============================] - 0s 111ms/step - loss: 148208.8438
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 77065.6719
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 40296.3633
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 22991.6094
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 14414.6855
Epoch 8/10

1/1 [==============================] - 0s 108ms/step - loss: 9774.0439
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 7117.3276
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 5476.7222

  #### Inference Need return ypred, ytrue ######################### 
[[-1.01494938e-02  8.36988068e+00  9.47359371e+00  9.13028336e+00
   1.01728373e+01  7.78102732e+00  1.11082001e+01  9.61347866e+00
   7.98357582e+00  1.00739307e+01  6.68569469e+00  9.29098129e+00
   1.05150156e+01  8.17062473e+00  8.73911476e+00  9.53966999e+00
   1.02366476e+01  1.09016056e+01  8.46389294e+00  9.67201805e+00
   8.32163334e+00  9.42670727e+00  9.71952534e+00  8.50262451e+00
   9.20372677e+00  9.42261028e+00  9.89635372e+00  6.67539024e+00
   9.86218929e+00  8.59705162e+00  9.28163528e+00  7.59055853e+00
   8.37679958e+00  8.44313908e+00  8.77481651e+00  6.62303209e+00
   9.07445621e+00  8.97202778e+00  9.35482788e+00  9.11585331e+00
   8.82392502e+00  1.10969858e+01  9.96018505e+00  9.09830093e+00
   1.03168936e+01  9.53477955e+00  8.76353455e+00  1.02327394e+01
   9.81178284e+00  1.00779657e+01  1.03543434e+01  9.05405712e+00
   7.98594904e+00  7.90837574e+00  8.47571659e+00  8.99563694e+00
   8.90431595e+00  8.54434299e+00  9.36909676e+00  7.95809174e+00
   1.57106161e-01 -3.08503568e-01  2.74238014e+00  1.67848170e+00
   1.31942475e+00  1.61014497e-01  3.11269492e-01  1.22629571e+00
  -2.95595616e-01  1.36540091e+00 -1.23219323e+00 -9.89516258e-01
  -2.73121953e+00  4.52206463e-01  1.00503993e+00 -8.47397596e-02
   2.38730520e-01 -1.24794757e+00  3.67955267e-01 -3.70434403e-01
  -5.31555474e-01  3.36158007e-01  5.76046646e-01  4.39938635e-01
   1.33990598e+00 -4.04639542e-01 -2.33860016e+00 -8.94848049e-01
   2.74940193e-01 -1.29624116e+00 -1.18038833e+00  6.10413432e-01
   8.26201200e-01  3.42692971e-01  1.91285408e+00 -1.68530154e+00
   1.03819609e+00 -2.98018545e-01  3.74974340e-01 -5.95505476e-01
  -2.42232382e-01 -1.43836510e+00 -6.94477558e-03  7.42976516e-02
   5.45055866e-02 -9.51989472e-01 -1.29185021e+00 -7.98139095e-01
  -2.41496325e-01 -7.16194630e-01 -2.39732170e+00  6.73942387e-01
   2.13417739e-01 -6.46031141e-01  5.09104192e-01  5.52383006e-01
   9.71528709e-01  6.21000230e-01 -6.69431269e-01 -1.38254273e+00
   2.74516165e-01  1.03661537e+00 -1.99403262e+00  4.48989451e-01
  -8.52689326e-01 -7.71569848e-01 -1.46035659e+00 -3.34966660e-01
   6.35377705e-01 -1.18260217e+00  1.09371090e+00  1.39717233e+00
   3.61959517e-01 -4.56450582e-02  6.54808998e-01 -1.78700894e-01
  -3.94822061e-01  1.15757120e+00 -3.26988280e-01  1.60968959e+00
  -8.20003152e-01 -1.00770295e-02  1.73888421e+00 -7.98507929e-01
   8.78025115e-01  2.44905889e-01  4.20340240e-01 -1.00136721e+00
  -1.28389150e-01 -4.81513798e-01 -6.39734805e-01  1.29737616e+00
   5.58503449e-01  1.28726053e+00 -1.74535775e+00  1.99518061e+00
   4.73532051e-01  1.17777359e+00  6.08558834e-01 -3.75516295e-01
  -6.34094119e-01  3.51381510e-01  4.76500034e-01 -2.31157756e+00
   1.02280831e+00 -1.50676966e+00  2.01858997e+00 -4.40042078e-01
   5.02459049e-01  1.08999360e+00 -7.80186832e-01 -1.70338809e-01
   5.17013013e-01 -1.08032203e+00  4.13827300e-01  2.01334906e+00
   1.20418096e+00  1.68826139e+00  5.31829417e-01 -1.03707159e+00
   1.17599905e-01  1.04304953e+01  8.27881241e+00  8.76914024e+00
   9.73826408e+00  9.30807400e+00  1.05862017e+01  1.09526825e+01
   1.06155405e+01  1.08369379e+01  9.96627617e+00  1.09371758e+01
   8.76032925e+00  1.04194403e+01  9.68293381e+00  1.03666334e+01
   9.63176155e+00  8.63774490e+00  8.07498360e+00  1.02374239e+01
   8.41536427e+00  9.24215508e+00  7.57499409e+00  9.78532982e+00
   7.61504030e+00  1.04364605e+01  9.58977032e+00  1.12014885e+01
   9.52862072e+00  9.81919384e+00  9.30706501e+00  7.05499411e+00
   1.10848112e+01  8.84113884e+00  8.47203159e+00  1.09866085e+01
   1.03717718e+01  1.09717932e+01  1.02256966e+01  1.00976286e+01
   7.12668324e+00  7.82146025e+00  1.01295128e+01  1.00025539e+01
   9.11835384e+00  9.08310127e+00  9.14870548e+00  1.02131739e+01
   8.66454029e+00  7.84503269e+00  8.27342701e+00  9.60848713e+00
   6.74855709e+00  9.55142975e+00  1.06616611e+01  9.59975243e+00
   6.91872931e+00  9.35658455e+00  8.49797916e+00  8.13405037e+00
   1.38034391e+00  4.44713712e-01  8.08404446e-01  7.20281839e-01
   3.98912549e-01  1.09322107e+00  2.30553246e+00  8.61648679e-01
   1.97593045e+00  5.83739281e-02  2.71326923e+00  3.54680920e+00
   3.89691591e-01  4.33881521e-01  2.08615899e+00  1.92054784e+00
   1.75271368e+00  1.37437177e+00  2.00617981e+00  1.06378114e+00
   8.40469599e-01  1.29781246e+00  1.61967635e+00  1.14909196e+00
   1.22644186e-01  2.02902365e+00  2.72227764e-01  1.49685097e+00
   1.12669611e+00  1.64690638e+00  2.26743460e-01  1.97885156e-01
   5.30426264e-01  2.68268967e+00  6.08398378e-01  1.84640384e+00
   3.89862537e-01  9.35212374e-02  7.48963356e-01  2.38802624e+00
   2.87982130e+00  3.64125204e+00  8.30829918e-01  2.03016400e-01
   5.01190245e-01  9.16993618e-01  1.79617548e+00  1.32785010e+00
   5.45660257e-01  7.80600369e-01  3.25171769e-01  6.22429848e-01
   1.35731578e+00  9.54298377e-01  1.36605501e+00  9.41095114e-01
   2.80545449e+00  2.17124414e+00  6.35658622e-01  8.53630304e-02
   1.80292070e+00  2.87191200e+00  7.33827055e-01  2.72921801e-01
   8.28124046e-01  9.09325123e-01  1.64474392e+00  1.25050759e+00
   2.29628181e+00  1.62889361e+00  7.23629594e-01  1.16703856e+00
   1.96032608e+00  7.31687188e-01  9.88127828e-01  1.64759636e-01
   1.26480484e+00  1.10399437e+00  1.55733192e+00  9.96537209e-02
   2.57355356e+00  2.88098145e+00  3.86852801e-01  1.89872050e+00
   1.92930877e+00  1.26281261e+00  1.93170214e+00  2.32254326e-01
   2.60368204e+00  2.98687637e-01  5.44502497e-01  4.20633376e-01
   2.61162567e+00  6.10674977e-01  2.82953167e+00  4.42964673e-01
   6.13894105e-01  1.43026900e+00  1.75960708e+00  2.23873758e+00
   2.61642599e+00  2.59944797e-01  1.64371729e+00  2.51524615e+00
   2.05714583e-01  1.19657505e+00  6.12892389e-01  9.34905648e-01
   8.69663775e-01  1.53828812e+00  8.99976075e-01  6.21609986e-01
   5.55689454e-01  9.25900817e-01  1.62543464e+00  1.58297241e+00
   7.63124585e-01  7.89086938e-01  1.23014188e+00  1.17325675e+00
   9.97047997e+00 -1.80863404e+00 -5.60295820e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 23:12:59.804925
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.0689
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 23:12:59.809584
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8681.62
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 23:12:59.813332
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.0197
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 23:12:59.817102
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -776.501
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140209783144232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140208824602128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140208824164424
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208824164928
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208824165432
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208824165936

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f85303353c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.515082
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.490564
grad_step = 000002, loss = 0.469553
grad_step = 000003, loss = 0.447188
grad_step = 000004, loss = 0.423829
grad_step = 000005, loss = 0.400830
grad_step = 000006, loss = 0.382701
grad_step = 000007, loss = 0.378144
grad_step = 000008, loss = 0.370177
grad_step = 000009, loss = 0.352941
grad_step = 000010, loss = 0.336287
grad_step = 000011, loss = 0.323623
grad_step = 000012, loss = 0.313293
grad_step = 000013, loss = 0.303487
grad_step = 000014, loss = 0.292667
grad_step = 000015, loss = 0.281273
grad_step = 000016, loss = 0.270055
grad_step = 000017, loss = 0.259665
grad_step = 000018, loss = 0.249973
grad_step = 000019, loss = 0.240165
grad_step = 000020, loss = 0.230186
grad_step = 000021, loss = 0.219673
grad_step = 000022, loss = 0.209275
grad_step = 000023, loss = 0.199721
grad_step = 000024, loss = 0.190882
grad_step = 000025, loss = 0.182201
grad_step = 000026, loss = 0.173455
grad_step = 000027, loss = 0.164662
grad_step = 000028, loss = 0.156212
grad_step = 000029, loss = 0.148633
grad_step = 000030, loss = 0.141103
grad_step = 000031, loss = 0.133104
grad_step = 000032, loss = 0.125487
grad_step = 000033, loss = 0.118350
grad_step = 000034, loss = 0.111459
grad_step = 000035, loss = 0.104813
grad_step = 000036, loss = 0.098479
grad_step = 000037, loss = 0.092296
grad_step = 000038, loss = 0.086520
grad_step = 000039, loss = 0.081302
grad_step = 000040, loss = 0.076195
grad_step = 000041, loss = 0.071032
grad_step = 000042, loss = 0.066318
grad_step = 000043, loss = 0.062001
grad_step = 000044, loss = 0.057729
grad_step = 000045, loss = 0.053678
grad_step = 000046, loss = 0.049939
grad_step = 000047, loss = 0.046406
grad_step = 000048, loss = 0.043215
grad_step = 000049, loss = 0.040091
grad_step = 000050, loss = 0.037093
grad_step = 000051, loss = 0.034400
grad_step = 000052, loss = 0.031839
grad_step = 000053, loss = 0.029387
grad_step = 000054, loss = 0.027109
grad_step = 000055, loss = 0.024956
grad_step = 000056, loss = 0.023026
grad_step = 000057, loss = 0.021149
grad_step = 000058, loss = 0.019398
grad_step = 000059, loss = 0.017816
grad_step = 000060, loss = 0.016322
grad_step = 000061, loss = 0.014943
grad_step = 000062, loss = 0.013643
grad_step = 000063, loss = 0.012461
grad_step = 000064, loss = 0.011383
grad_step = 000065, loss = 0.010373
grad_step = 000066, loss = 0.009469
grad_step = 000067, loss = 0.008634
grad_step = 000068, loss = 0.007878
grad_step = 000069, loss = 0.007180
grad_step = 000070, loss = 0.006550
grad_step = 000071, loss = 0.005992
grad_step = 000072, loss = 0.005494
grad_step = 000073, loss = 0.005048
grad_step = 000074, loss = 0.004643
grad_step = 000075, loss = 0.004297
grad_step = 000076, loss = 0.003981
grad_step = 000077, loss = 0.003707
grad_step = 000078, loss = 0.003468
grad_step = 000079, loss = 0.003267
grad_step = 000080, loss = 0.003088
grad_step = 000081, loss = 0.002936
grad_step = 000082, loss = 0.002810
grad_step = 000083, loss = 0.002701
grad_step = 000084, loss = 0.002611
grad_step = 000085, loss = 0.002538
grad_step = 000086, loss = 0.002481
grad_step = 000087, loss = 0.002433
grad_step = 000088, loss = 0.002396
grad_step = 000089, loss = 0.002366
grad_step = 000090, loss = 0.002345
grad_step = 000091, loss = 0.002327
grad_step = 000092, loss = 0.002317
grad_step = 000093, loss = 0.002307
grad_step = 000094, loss = 0.002301
grad_step = 000095, loss = 0.002296
grad_step = 000096, loss = 0.002293
grad_step = 000097, loss = 0.002290
grad_step = 000098, loss = 0.002288
grad_step = 000099, loss = 0.002285
grad_step = 000100, loss = 0.002283
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002282
grad_step = 000102, loss = 0.002285
grad_step = 000103, loss = 0.002296
grad_step = 000104, loss = 0.002315
grad_step = 000105, loss = 0.002344
grad_step = 000106, loss = 0.002337
grad_step = 000107, loss = 0.002301
grad_step = 000108, loss = 0.002256
grad_step = 000109, loss = 0.002247
grad_step = 000110, loss = 0.002264
grad_step = 000111, loss = 0.002270
grad_step = 000112, loss = 0.002258
grad_step = 000113, loss = 0.002229
grad_step = 000114, loss = 0.002211
grad_step = 000115, loss = 0.002217
grad_step = 000116, loss = 0.002227
grad_step = 000117, loss = 0.002218
grad_step = 000118, loss = 0.002193
grad_step = 000119, loss = 0.002183
grad_step = 000120, loss = 0.002187
grad_step = 000121, loss = 0.002189
grad_step = 000122, loss = 0.002184
grad_step = 000123, loss = 0.002174
grad_step = 000124, loss = 0.002163
grad_step = 000125, loss = 0.002156
grad_step = 000126, loss = 0.002157
grad_step = 000127, loss = 0.002161
grad_step = 000128, loss = 0.002159
grad_step = 000129, loss = 0.002153
grad_step = 000130, loss = 0.002146
grad_step = 000131, loss = 0.002142
grad_step = 000132, loss = 0.002136
grad_step = 000133, loss = 0.002131
grad_step = 000134, loss = 0.002128
grad_step = 000135, loss = 0.002128
grad_step = 000136, loss = 0.002129
grad_step = 000137, loss = 0.002129
grad_step = 000138, loss = 0.002130
grad_step = 000139, loss = 0.002133
grad_step = 000140, loss = 0.002138
grad_step = 000141, loss = 0.002144
grad_step = 000142, loss = 0.002151
grad_step = 000143, loss = 0.002153
grad_step = 000144, loss = 0.002153
grad_step = 000145, loss = 0.002144
grad_step = 000146, loss = 0.002130
grad_step = 000147, loss = 0.002114
grad_step = 000148, loss = 0.002102
grad_step = 000149, loss = 0.002097
grad_step = 000150, loss = 0.002097
grad_step = 000151, loss = 0.002101
grad_step = 000152, loss = 0.002106
grad_step = 000153, loss = 0.002114
grad_step = 000154, loss = 0.002122
grad_step = 000155, loss = 0.002133
grad_step = 000156, loss = 0.002139
grad_step = 000157, loss = 0.002145
grad_step = 000158, loss = 0.002136
grad_step = 000159, loss = 0.002123
grad_step = 000160, loss = 0.002100
grad_step = 000161, loss = 0.002082
grad_step = 000162, loss = 0.002073
grad_step = 000163, loss = 0.002072
grad_step = 000164, loss = 0.002079
grad_step = 000165, loss = 0.002088
grad_step = 000166, loss = 0.002097
grad_step = 000167, loss = 0.002105
grad_step = 000168, loss = 0.002115
grad_step = 000169, loss = 0.002119
grad_step = 000170, loss = 0.002119
grad_step = 000171, loss = 0.002101
grad_step = 000172, loss = 0.002079
grad_step = 000173, loss = 0.002059
grad_step = 000174, loss = 0.002050
grad_step = 000175, loss = 0.002050
grad_step = 000176, loss = 0.002055
grad_step = 000177, loss = 0.002064
grad_step = 000178, loss = 0.002071
grad_step = 000179, loss = 0.002077
grad_step = 000180, loss = 0.002075
grad_step = 000181, loss = 0.002070
grad_step = 000182, loss = 0.002060
grad_step = 000183, loss = 0.002050
grad_step = 000184, loss = 0.002040
grad_step = 000185, loss = 0.002031
grad_step = 000186, loss = 0.002024
grad_step = 000187, loss = 0.002019
grad_step = 000188, loss = 0.002017
grad_step = 000189, loss = 0.002016
grad_step = 000190, loss = 0.002017
grad_step = 000191, loss = 0.002018
grad_step = 000192, loss = 0.002024
grad_step = 000193, loss = 0.002036
grad_step = 000194, loss = 0.002062
grad_step = 000195, loss = 0.002104
grad_step = 000196, loss = 0.002177
grad_step = 000197, loss = 0.002226
grad_step = 000198, loss = 0.002247
grad_step = 000199, loss = 0.002141
grad_step = 000200, loss = 0.002028
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001992
grad_step = 000202, loss = 0.002041
grad_step = 000203, loss = 0.002097
grad_step = 000204, loss = 0.002074
grad_step = 000205, loss = 0.002010
grad_step = 000206, loss = 0.001968
grad_step = 000207, loss = 0.001994
grad_step = 000208, loss = 0.002037
grad_step = 000209, loss = 0.002021
grad_step = 000210, loss = 0.001976
grad_step = 000211, loss = 0.001953
grad_step = 000212, loss = 0.001964
grad_step = 000213, loss = 0.001979
grad_step = 000214, loss = 0.001976
grad_step = 000215, loss = 0.001962
grad_step = 000216, loss = 0.001942
grad_step = 000217, loss = 0.001929
grad_step = 000218, loss = 0.001933
grad_step = 000219, loss = 0.001943
grad_step = 000220, loss = 0.001941
grad_step = 000221, loss = 0.001927
grad_step = 000222, loss = 0.001913
grad_step = 000223, loss = 0.001906
grad_step = 000224, loss = 0.001907
grad_step = 000225, loss = 0.001912
grad_step = 000226, loss = 0.001913
grad_step = 000227, loss = 0.001907
grad_step = 000228, loss = 0.001894
grad_step = 000229, loss = 0.001882
grad_step = 000230, loss = 0.001877
grad_step = 000231, loss = 0.001881
grad_step = 000232, loss = 0.001890
grad_step = 000233, loss = 0.001901
grad_step = 000234, loss = 0.001888
grad_step = 000235, loss = 0.001876
grad_step = 000236, loss = 0.001875
grad_step = 000237, loss = 0.001878
grad_step = 000238, loss = 0.001886
grad_step = 000239, loss = 0.001886
grad_step = 000240, loss = 0.001894
grad_step = 000241, loss = 0.001910
grad_step = 000242, loss = 0.001942
grad_step = 000243, loss = 0.001953
grad_step = 000244, loss = 0.001948
grad_step = 000245, loss = 0.001903
grad_step = 000246, loss = 0.001867
grad_step = 000247, loss = 0.001854
grad_step = 000248, loss = 0.001857
grad_step = 000249, loss = 0.001858
grad_step = 000250, loss = 0.001857
grad_step = 000251, loss = 0.001859
grad_step = 000252, loss = 0.001855
grad_step = 000253, loss = 0.001846
grad_step = 000254, loss = 0.001828
grad_step = 000255, loss = 0.001816
grad_step = 000256, loss = 0.001815
grad_step = 000257, loss = 0.001823
grad_step = 000258, loss = 0.001828
grad_step = 000259, loss = 0.001825
grad_step = 000260, loss = 0.001829
grad_step = 000261, loss = 0.001841
grad_step = 000262, loss = 0.001868
grad_step = 000263, loss = 0.001894
grad_step = 000264, loss = 0.001917
grad_step = 000265, loss = 0.001910
grad_step = 000266, loss = 0.001890
grad_step = 000267, loss = 0.001857
grad_step = 000268, loss = 0.001828
grad_step = 000269, loss = 0.001803
grad_step = 000270, loss = 0.001800
grad_step = 000271, loss = 0.001819
grad_step = 000272, loss = 0.001837
grad_step = 000273, loss = 0.001838
grad_step = 000274, loss = 0.001812
grad_step = 000275, loss = 0.001788
grad_step = 000276, loss = 0.001780
grad_step = 000277, loss = 0.001787
grad_step = 000278, loss = 0.001797
grad_step = 000279, loss = 0.001798
grad_step = 000280, loss = 0.001798
grad_step = 000281, loss = 0.001802
grad_step = 000282, loss = 0.001809
grad_step = 000283, loss = 0.001806
grad_step = 000284, loss = 0.001794
grad_step = 000285, loss = 0.001781
grad_step = 000286, loss = 0.001773
grad_step = 000287, loss = 0.001769
grad_step = 000288, loss = 0.001765
grad_step = 000289, loss = 0.001760
grad_step = 000290, loss = 0.001758
grad_step = 000291, loss = 0.001759
grad_step = 000292, loss = 0.001763
grad_step = 000293, loss = 0.001766
grad_step = 000294, loss = 0.001766
grad_step = 000295, loss = 0.001766
grad_step = 000296, loss = 0.001765
grad_step = 000297, loss = 0.001768
grad_step = 000298, loss = 0.001772
grad_step = 000299, loss = 0.001779
grad_step = 000300, loss = 0.001789
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001801
grad_step = 000302, loss = 0.001814
grad_step = 000303, loss = 0.001830
grad_step = 000304, loss = 0.001840
grad_step = 000305, loss = 0.001847
grad_step = 000306, loss = 0.001836
grad_step = 000307, loss = 0.001818
grad_step = 000308, loss = 0.001785
grad_step = 000309, loss = 0.001754
grad_step = 000310, loss = 0.001731
grad_step = 000311, loss = 0.001727
grad_step = 000312, loss = 0.001738
grad_step = 000313, loss = 0.001755
grad_step = 000314, loss = 0.001767
grad_step = 000315, loss = 0.001764
grad_step = 000316, loss = 0.001751
grad_step = 000317, loss = 0.001733
grad_step = 000318, loss = 0.001720
grad_step = 000319, loss = 0.001716
grad_step = 000320, loss = 0.001718
grad_step = 000321, loss = 0.001726
grad_step = 000322, loss = 0.001731
grad_step = 000323, loss = 0.001734
grad_step = 000324, loss = 0.001730
grad_step = 000325, loss = 0.001724
grad_step = 000326, loss = 0.001720
grad_step = 000327, loss = 0.001719
grad_step = 000328, loss = 0.001719
grad_step = 000329, loss = 0.001716
grad_step = 000330, loss = 0.001709
grad_step = 000331, loss = 0.001701
grad_step = 000332, loss = 0.001695
grad_step = 000333, loss = 0.001691
grad_step = 000334, loss = 0.001689
grad_step = 000335, loss = 0.001689
grad_step = 000336, loss = 0.001691
grad_step = 000337, loss = 0.001695
grad_step = 000338, loss = 0.001702
grad_step = 000339, loss = 0.001717
grad_step = 000340, loss = 0.001727
grad_step = 000341, loss = 0.001743
grad_step = 000342, loss = 0.001756
grad_step = 000343, loss = 0.001791
grad_step = 000344, loss = 0.001848
grad_step = 000345, loss = 0.001932
grad_step = 000346, loss = 0.001975
grad_step = 000347, loss = 0.001958
grad_step = 000348, loss = 0.001834
grad_step = 000349, loss = 0.001706
grad_step = 000350, loss = 0.001666
grad_step = 000351, loss = 0.001719
grad_step = 000352, loss = 0.001784
grad_step = 000353, loss = 0.001791
grad_step = 000354, loss = 0.001760
grad_step = 000355, loss = 0.001755
grad_step = 000356, loss = 0.001716
grad_step = 000357, loss = 0.001700
grad_step = 000358, loss = 0.001708
grad_step = 000359, loss = 0.001721
grad_step = 000360, loss = 0.001696
grad_step = 000361, loss = 0.001678
grad_step = 000362, loss = 0.001679
grad_step = 000363, loss = 0.001683
grad_step = 000364, loss = 0.001687
grad_step = 000365, loss = 0.001673
grad_step = 000366, loss = 0.001645
grad_step = 000367, loss = 0.001639
grad_step = 000368, loss = 0.001662
grad_step = 000369, loss = 0.001679
grad_step = 000370, loss = 0.001683
grad_step = 000371, loss = 0.001682
grad_step = 000372, loss = 0.001686
grad_step = 000373, loss = 0.001647
grad_step = 000374, loss = 0.001636
grad_step = 000375, loss = 0.001655
grad_step = 000376, loss = 0.001661
grad_step = 000377, loss = 0.001642
grad_step = 000378, loss = 0.001640
grad_step = 000379, loss = 0.001646
grad_step = 000380, loss = 0.001628
grad_step = 000381, loss = 0.001617
grad_step = 000382, loss = 0.001619
grad_step = 000383, loss = 0.001619
grad_step = 000384, loss = 0.001615
grad_step = 000385, loss = 0.001614
grad_step = 000386, loss = 0.001614
grad_step = 000387, loss = 0.001610
grad_step = 000388, loss = 0.001607
grad_step = 000389, loss = 0.001609
grad_step = 000390, loss = 0.001607
grad_step = 000391, loss = 0.001601
grad_step = 000392, loss = 0.001596
grad_step = 000393, loss = 0.001595
grad_step = 000394, loss = 0.001597
grad_step = 000395, loss = 0.001602
grad_step = 000396, loss = 0.001615
grad_step = 000397, loss = 0.001649
grad_step = 000398, loss = 0.001690
grad_step = 000399, loss = 0.001754
grad_step = 000400, loss = 0.001790
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001847
grad_step = 000402, loss = 0.001887
grad_step = 000403, loss = 0.001905
grad_step = 000404, loss = 0.001798
grad_step = 000405, loss = 0.001643
grad_step = 000406, loss = 0.001579
grad_step = 000407, loss = 0.001651
grad_step = 000408, loss = 0.001733
grad_step = 000409, loss = 0.001708
grad_step = 000410, loss = 0.001616
grad_step = 000411, loss = 0.001570
grad_step = 000412, loss = 0.001609
grad_step = 000413, loss = 0.001654
grad_step = 000414, loss = 0.001637
grad_step = 000415, loss = 0.001600
grad_step = 000416, loss = 0.001581
grad_step = 000417, loss = 0.001579
grad_step = 000418, loss = 0.001581
grad_step = 000419, loss = 0.001589
grad_step = 000420, loss = 0.001587
grad_step = 000421, loss = 0.001570
grad_step = 000422, loss = 0.001562
grad_step = 000423, loss = 0.001557
grad_step = 000424, loss = 0.001555
grad_step = 000425, loss = 0.001559
grad_step = 000426, loss = 0.001562
grad_step = 000427, loss = 0.001554
grad_step = 000428, loss = 0.001537
grad_step = 000429, loss = 0.001534
grad_step = 000430, loss = 0.001541
grad_step = 000431, loss = 0.001542
grad_step = 000432, loss = 0.001538
grad_step = 000433, loss = 0.001531
grad_step = 000434, loss = 0.001525
grad_step = 000435, loss = 0.001524
grad_step = 000436, loss = 0.001526
grad_step = 000437, loss = 0.001525
grad_step = 000438, loss = 0.001517
grad_step = 000439, loss = 0.001512
grad_step = 000440, loss = 0.001512
grad_step = 000441, loss = 0.001513
grad_step = 000442, loss = 0.001511
grad_step = 000443, loss = 0.001508
grad_step = 000444, loss = 0.001506
grad_step = 000445, loss = 0.001503
grad_step = 000446, loss = 0.001502
grad_step = 000447, loss = 0.001504
grad_step = 000448, loss = 0.001510
grad_step = 000449, loss = 0.001519
grad_step = 000450, loss = 0.001538
grad_step = 000451, loss = 0.001580
grad_step = 000452, loss = 0.001648
grad_step = 000453, loss = 0.001746
grad_step = 000454, loss = 0.001831
grad_step = 000455, loss = 0.001858
grad_step = 000456, loss = 0.001729
grad_step = 000457, loss = 0.001575
grad_step = 000458, loss = 0.001517
grad_step = 000459, loss = 0.001568
grad_step = 000460, loss = 0.001613
grad_step = 000461, loss = 0.001574
grad_step = 000462, loss = 0.001529
grad_step = 000463, loss = 0.001537
grad_step = 000464, loss = 0.001557
grad_step = 000465, loss = 0.001544
grad_step = 000466, loss = 0.001494
grad_step = 000467, loss = 0.001490
grad_step = 000468, loss = 0.001519
grad_step = 000469, loss = 0.001513
grad_step = 000470, loss = 0.001485
grad_step = 000471, loss = 0.001473
grad_step = 000472, loss = 0.001486
grad_step = 000473, loss = 0.001491
grad_step = 000474, loss = 0.001474
grad_step = 000475, loss = 0.001459
grad_step = 000476, loss = 0.001461
grad_step = 000477, loss = 0.001471
grad_step = 000478, loss = 0.001469
grad_step = 000479, loss = 0.001456
grad_step = 000480, loss = 0.001447
grad_step = 000481, loss = 0.001447
grad_step = 000482, loss = 0.001451
grad_step = 000483, loss = 0.001448
grad_step = 000484, loss = 0.001440
grad_step = 000485, loss = 0.001436
grad_step = 000486, loss = 0.001437
grad_step = 000487, loss = 0.001436
grad_step = 000488, loss = 0.001431
grad_step = 000489, loss = 0.001428
grad_step = 000490, loss = 0.001427
grad_step = 000491, loss = 0.001427
grad_step = 000492, loss = 0.001426
grad_step = 000493, loss = 0.001421
grad_step = 000494, loss = 0.001416
grad_step = 000495, loss = 0.001414
grad_step = 000496, loss = 0.001414
grad_step = 000497, loss = 0.001414
grad_step = 000498, loss = 0.001412
grad_step = 000499, loss = 0.001411
grad_step = 000500, loss = 0.001411
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001413
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

  date_run                              2020-05-11 23:13:25.378858
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.263574
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 23:13:25.385386
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.177726
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 23:13:25.395123
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140806
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 23:13:25.401598
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.70061
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
0   2020-05-11 23:12:49.596743  ...    mean_absolute_error
1   2020-05-11 23:12:49.601384  ...     mean_squared_error
2   2020-05-11 23:12:49.605260  ...  median_absolute_error
3   2020-05-11 23:12:49.609202  ...               r2_score
4   2020-05-11 23:12:59.804925  ...    mean_absolute_error
5   2020-05-11 23:12:59.809584  ...     mean_squared_error
6   2020-05-11 23:12:59.813332  ...  median_absolute_error
7   2020-05-11 23:12:59.817102  ...               r2_score
8   2020-05-11 23:13:25.378858  ...    mean_absolute_error
9   2020-05-11 23:13:25.385386  ...     mean_squared_error
10  2020-05-11 23:13:25.395123  ...  median_absolute_error
11  2020-05-11 23:13:25.401598  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:19, 124662.54it/s] 86%|████████▋ | 8552448/9912422 [00:00<00:07, 177977.91it/s]9920512it [00:00, 39980939.63it/s]                           
0it [00:00, ?it/s]32768it [00:00, 513830.82it/s]
0it [00:00, ?it/s]  2%|▏         | 32768/1648877 [00:00<00:04, 327604.55it/s]1654784it [00:00, 11077329.81it/s]                         
0it [00:00, ?it/s]8192it [00:00, 201945.05it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5fd2eb1fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f705cdef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5fd2e3cef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f700a5048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f705ca0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f798e66a0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f84e11240> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f85835e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5fd2e3cef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f798e66a0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5f705cdfd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd9002e01d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=447622f6641ac44061540fc1b13406e3eb036bfccbbba6139548b1c0791880a8
  Stored in directory: /tmp/pip-ephem-wheel-cache-bd9vl45s/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd8f644f048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3260416/17464789 [====>.........................] - ETA: 0s
11026432/17464789 [=================>............] - ETA: 0s
15294464/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 23:14:55.341953: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 23:14:55.347072: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-11 23:14:55.347247: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562cae2a22d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 23:14:55.347264: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8276 - accuracy: 0.4895
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7241 - accuracy: 0.4963
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7740 - accuracy: 0.4930
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8200 - accuracy: 0.4900
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7082 - accuracy: 0.4973
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6475 - accuracy: 0.5013
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6803 - accuracy: 0.4991
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7019 - accuracy: 0.4977
11000/25000 [============>.................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
12000/25000 [=============>................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6546 - accuracy: 0.5008
15000/25000 [=================>............] - ETA: 3s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6733 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 23:15:13.266966
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 23:15:13.266966  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 23:15:20.300602: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 23:15:20.306659: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-11 23:15:20.306842: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559a91a6e1e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 23:15:20.306859: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6037834dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7178 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.6492 - val_crf_viterbi_accuracy: 0.0000e+00

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f602dad58d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 16s - loss: 7.2986 - accuracy: 0.5240
 2000/25000 [=>............................] - ETA: 11s - loss: 7.3906 - accuracy: 0.5180
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.3753 - accuracy: 0.5190 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.4405 - accuracy: 0.5148
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.4734 - accuracy: 0.5126
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.4877 - accuracy: 0.5117
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5834 - accuracy: 0.5054
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5985 - accuracy: 0.5044
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.6346 - accuracy: 0.5021
12000/25000 [=============>................] - ETA: 4s - loss: 7.6257 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
15000/25000 [=================>............] - ETA: 3s - loss: 7.6390 - accuracy: 0.5018
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6456 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6533 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f5fe8a4a400> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<46:45:12, 5.12kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<32:57:28, 7.27kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:07:16, 10.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<16:11:10, 14.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:17:51, 21.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.77M/862M [00:02<7:51:45, 30.2kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:02<5:28:41, 43.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:02<3:48:56, 61.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:02<2:39:32, 87.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:02<1:50:58, 125kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:02<1:17:33, 179kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:02<53:59, 255kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<37:48, 363kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:03<26:22, 517kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:03<18:30, 733kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<13:22, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<11:13, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<11:56, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<09:21, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:06<06:47, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<09:22, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<08:28, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<06:22, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<06:54, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<07:38, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<06:01, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<04:22, 3.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<1:37:30, 136kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<1:09:34, 190kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<48:56, 270kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<37:15, 353kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<27:24, 480kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<19:26, 675kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<16:39, 786kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<12:58, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<09:23, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<09:38, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<09:23, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<07:08, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:17<05:08, 2.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<14:04, 920kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<10:57, 1.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:19<07:59, 1.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<08:36, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<08:37, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<06:41, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<06:44, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:23<04:30, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<06:06, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<05:33, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<04:12, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:56, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:44, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:15, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:48, 3.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<15:30, 813kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<12:07, 1.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<08:47, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<09:05, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:37, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:38, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:53, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:05, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:33, 2.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<06:07, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<06:48, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:24, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<05:46, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:17, 2.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:00, 3.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:29, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:08, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<03:55, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<05:27, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:04, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<03:47, 3.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:31, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:06, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<03:52, 3.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:34, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:20, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<04:57, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<03:38, 3.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:38, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:03, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:26, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<03:55, 3.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<50:47, 235kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<37:57, 314kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<27:05, 440kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<19:02, 623kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<22:23, 529kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<16:53, 701kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<12:06, 977kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<11:13, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<10:15, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<07:41, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<16:39, 703kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<12:51, 910kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<09:17, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<09:12, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:50, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:46, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<06:35, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:49, 1.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:21, 2.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:43, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:57, 2.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:49, 3.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<02:47, 4.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<55:17, 207kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<39:50, 287kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<28:04, 406kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<22:18, 510kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<17:54, 635kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:05, 867kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<10:57, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:27, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:07, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:15, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:39, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<05:45, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:10, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<03:54, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<05:19, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<05:58, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<04:44, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:05, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<04:40, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:33, 3.10MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<04:27, 2.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<03:19, 3.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<02:30, 4.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<1:23:06, 131kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<1:00:22, 180kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<42:41, 255kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<29:53, 362kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<29:47, 363kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<21:57, 493kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<15:36, 691kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<13:24, 802kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<10:17, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:27, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<07:44, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<07:34, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:10, 2.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<10:19:04, 17.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<7:14:11, 24.4kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<5:03:25, 34.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<3:34:09, 49.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<2:31:59, 69.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<1:46:44, 98.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<1:16:03, 138kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<54:16, 193kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<38:09, 274kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<29:04, 358kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<22:27, 464kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<16:14, 640kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<11:26, 904kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:21:58, 126kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<58:23, 177kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<41:02, 251kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<31:02, 331kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<23:43, 433kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<17:06, 600kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<13:34, 752kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<10:32, 967kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:35, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:39, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<07:24, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<05:38, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:03, 2.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<1:15:38, 133kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<53:57, 187kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<37:53, 265kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<28:45, 348kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<22:09, 451kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<16:00, 624kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<12:45, 779kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:55, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:10, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<07:19, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:06, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:23, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:54, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<07:22, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:10, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:26, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:47, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:35, 2.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:47, 2.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:18, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:12, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:28, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:06, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:04, 3.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<04:23, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<05:01, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<03:59, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:18, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:00, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:01, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:49, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<02:54, 3.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:52, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:46, 3.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<07:28, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:10, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<05:19, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:29, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:35, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:00, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:15, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<03:53, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<02:57, 3.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:09, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:44, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:41, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:45, 3.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:03, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:03, 2.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<04:12, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<04:49, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<03:49, 2.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<02:46, 3.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<1:04:32, 136kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<46:03, 190kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<32:20, 270kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<24:35, 354kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<18:59, 458kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<13:42, 633kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<10:56, 788kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<08:32, 1.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:11, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:19, 1.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<05:17, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<03:54, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:42, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:09, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:05, 2.74MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:10, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:46, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:50, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:58, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:33, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<02:33, 3.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<20:26, 406kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<15:09, 546kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<10:44, 768kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<09:26, 871kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<08:16, 992kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<06:12, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:24, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<7:54:16, 17.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<5:32:34, 24.5kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<3:52:16, 34.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<2:43:45, 49.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:56:14, 69.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<1:21:38, 98.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<58:04, 138kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<41:26, 193kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<29:05, 274kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<22:08, 359kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<17:08, 463kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<12:21, 642kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<08:42, 906kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<10:42, 736kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<08:19, 946kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:00, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:00, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:39, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:43, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<02:52, 2.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:06, 3.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<06:07, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:55, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:38, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:38, 2.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<11:25, 665kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<08:46, 866kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<06:18, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<06:10, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<05:50, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:24, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:14, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<04:17, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:46, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<02:49, 2.63MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<03:42, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:20, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<02:29, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:29, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:55, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:04, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:16, 3.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<03:56, 1.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:30, 2.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:36, 2.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:30, 2.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<03:11, 2.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:24, 2.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:21, 2.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<02:57, 2.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:09, 3.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<04:50, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<04:06, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:01, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:44, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<04:02, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:07, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<08:59, 768kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<07:00, 985kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:03, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<05:08, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:10, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:05, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:15, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<42:00, 161kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<30:46, 220kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<21:50, 310kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<16:15, 413kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<12:04, 555kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:34, 779kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<07:31, 883kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:55, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:17, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:32, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<04:30, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:26, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:27, 2.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<08:42, 746kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:45, 962kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<04:52, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:54, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:44, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<03:38, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:36, 2.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<26:22, 241kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<19:05, 333kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<13:28, 470kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<10:51, 580kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<08:52, 709kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<06:28, 970kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<07:10, 869kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:39, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:04, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:36, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:39, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:18, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:11, 2.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:57, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:17, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:51, 3.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<13:44, 434kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<10:12, 583kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<07:14, 817kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<06:24, 919kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<05:38, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:11, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:59, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<05:42, 1.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<04:35, 1.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<03:40, 1.57MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:52, 1.99MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:55, 1.95MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:37, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<01:58, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<02:41, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:22, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:42, 3.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<06:00, 923kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<04:24, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:48, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:14, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:23, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:56, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:36, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<01:57, 2.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:37, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:22, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<01:46, 2.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:16, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<01:42, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:13, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<01:40, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:23, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:42, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:06, 2.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:33, 3.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<02:48, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:28, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<01:51, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<02:27, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:46, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:09, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:33, 3.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:39, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:03, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:42, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:23, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:46, 2.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:21, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:08, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<01:36, 2.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:14, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:31, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:00, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:26, 3.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<4:29:03, 17.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<3:08:33, 24.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<2:11:21, 35.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<1:32:18, 49.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<1:05:30, 69.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<45:56, 99.4kB/s]  .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<31:50, 142kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<4:42:39, 16.0kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<3:18:02, 22.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<2:17:56, 32.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<1:36:49, 45.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<1:08:07, 65.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<47:31, 92.9kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<34:01, 129kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<24:13, 180kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<16:58, 256kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<12:47, 337kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<09:49, 438kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<07:03, 608kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<04:56, 859kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<4:44:16, 14.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<3:17:32, 21.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<2:18:01, 30.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<1:36:53, 43.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<1:07:29, 61.3kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<47:48, 85.8kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<34:16, 120kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<24:05, 170kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<16:46, 242kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<13:43, 294kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<10:00, 403kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<07:03, 567kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<05:49, 680kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<04:54, 807kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:37, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:32, 1.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<39:36, 98.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<28:00, 139kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<19:46, 196kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<13:42, 279kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<32:20, 118kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<23:00, 166kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<16:04, 236kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<12:02, 312kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<09:11, 409kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<06:35, 568kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:35, 804kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<14:35, 253kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<10:35, 348kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<07:27, 491kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<06:00, 603kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<04:34, 791kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:16, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<03:06, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:32, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:51, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:06, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<01:49, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:21, 2.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<01:45, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:52<01:34, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:10, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:47, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:23, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:00, 3.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:30, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:05, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:32, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:49, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:57, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:30, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:05, 2.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:57, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:29, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:07, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:29, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:40, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:19, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:23, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:16, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<00:57, 3.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:21, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:33, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:12, 2.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<00:52, 3.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:16, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:53, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:22, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:41, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<00:56, 2.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:44, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:29, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:05, 2.46MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:47, 3.36MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<11:07, 239kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<08:02, 331kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<05:37, 468kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<04:30, 576kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:40, 704kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:41, 960kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:52, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<17:48, 142kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<12:41, 199kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<08:51, 282kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<06:41, 367kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<04:55, 498kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:28, 698kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:57, 807kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:18, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:41, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:39, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:16, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:53, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<04:44, 476kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:32, 635kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:30, 887kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:14, 976kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:01, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:29, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:04, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:31, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:17, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:56, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:07, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:12, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:40, 2.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<03:45, 526kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:49, 697kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:00, 970kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:49, 1.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:40, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:14, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:53, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:58, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:43, 2.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:54, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:00, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:34, 3.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:49, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:45, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:34, 2.96MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:24, 4.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<06:20, 258kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<04:45, 343kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<03:23, 478kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<02:18, 677kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<1:31:51, 17.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<1:04:12, 24.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<44:23, 34.7kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<30:18, 49.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<25:52, 57.9kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<18:11, 82.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<12:35, 117kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<08:51, 161kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<06:28, 220kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<04:34, 309kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<03:05, 440kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<1:20:18, 16.9kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<56:05, 24.2kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<38:32, 34.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<26:21, 49.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<20:46, 62.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<14:37, 88.0kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<10:05, 125kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<07:05, 172kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<05:03, 241kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<03:31, 341kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<02:24, 483kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:34, 449kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:55, 598kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:20, 834kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:08, 948kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:01, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:45, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:32, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:39, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:33, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:24, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:26, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:19, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:16, 2.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:19, 2.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:14, 3.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:10, 4.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<03:06, 238kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:13, 329kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:31, 465kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:10, 575kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:57, 704kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:40, 963kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:27, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:33, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:26, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:19, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:16, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:12, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:15, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:11, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:07, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 3.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 3.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:01, 3.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 4.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:13, 238kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:08, 329kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 465kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 694/400000 [00:00<00:57, 6935.41it/s]  0%|          | 1374/400000 [00:00<00:57, 6894.15it/s]  1%|          | 2064/400000 [00:00<00:57, 6894.31it/s]  1%|          | 2766/400000 [00:00<00:57, 6930.52it/s]  1%|          | 3455/400000 [00:00<00:57, 6917.51it/s]  1%|          | 4133/400000 [00:00<00:57, 6873.57it/s]  1%|          | 4776/400000 [00:00<00:58, 6733.91it/s]  1%|▏         | 5452/400000 [00:00<00:58, 6740.50it/s]  2%|▏         | 6136/400000 [00:00<00:58, 6769.60it/s]  2%|▏         | 6816/400000 [00:01<00:58, 6778.44it/s]  2%|▏         | 7518/400000 [00:01<00:57, 6846.36it/s]  2%|▏         | 8208/400000 [00:01<00:57, 6861.31it/s]  2%|▏         | 8917/400000 [00:01<00:56, 6927.87it/s]  2%|▏         | 9630/400000 [00:01<00:55, 6985.55it/s]  3%|▎         | 10345/400000 [00:01<00:55, 7033.76it/s]  3%|▎         | 11050/400000 [00:01<00:55, 7038.19it/s]  3%|▎         | 11769/400000 [00:01<00:54, 7081.26it/s]  3%|▎         | 12476/400000 [00:01<00:54, 7053.75it/s]  3%|▎         | 13181/400000 [00:01<00:55, 7019.06it/s]  3%|▎         | 13883/400000 [00:02<00:55, 7010.95it/s]  4%|▎         | 14584/400000 [00:02<00:57, 6698.50it/s]  4%|▍         | 15257/400000 [00:02<00:58, 6610.93it/s]  4%|▍         | 15932/400000 [00:02<00:57, 6650.40it/s]  4%|▍         | 16599/400000 [00:02<00:57, 6655.23it/s]  4%|▍         | 17282/400000 [00:02<00:57, 6704.96it/s]  4%|▍         | 17958/400000 [00:02<00:56, 6719.33it/s]  5%|▍         | 18670/400000 [00:02<00:55, 6834.29it/s]  5%|▍         | 19380/400000 [00:02<00:55, 6909.88it/s]  5%|▌         | 20072/400000 [00:02<00:54, 6911.94it/s]  5%|▌         | 20764/400000 [00:03<00:56, 6745.38it/s]  5%|▌         | 21480/400000 [00:03<00:55, 6862.69it/s]  6%|▌         | 22168/400000 [00:03<00:55, 6852.67it/s]  6%|▌         | 22861/400000 [00:03<00:54, 6874.29it/s]  6%|▌         | 23581/400000 [00:03<00:54, 6968.37it/s]  6%|▌         | 24290/400000 [00:03<00:53, 7003.08it/s]  6%|▌         | 24991/400000 [00:03<00:53, 6981.08it/s]  6%|▋         | 25690/400000 [00:03<00:53, 6938.00it/s]  7%|▋         | 26385/400000 [00:03<00:54, 6871.02it/s]  7%|▋         | 27093/400000 [00:03<00:53, 6932.35it/s]  7%|▋         | 27804/400000 [00:04<00:53, 6982.38it/s]  7%|▋         | 28507/400000 [00:04<00:53, 6994.54it/s]  7%|▋         | 29209/400000 [00:04<00:52, 7001.15it/s]  7%|▋         | 29924/400000 [00:04<00:52, 7044.35it/s]  8%|▊         | 30642/400000 [00:04<00:52, 7083.39it/s]  8%|▊         | 31363/400000 [00:04<00:51, 7117.54it/s]  8%|▊         | 32075/400000 [00:04<00:53, 6900.65it/s]  8%|▊         | 32767/400000 [00:04<00:54, 6762.32it/s]  8%|▊         | 33474/400000 [00:04<00:53, 6849.92it/s]  9%|▊         | 34161/400000 [00:04<00:53, 6835.86it/s]  9%|▊         | 34862/400000 [00:05<00:53, 6884.43it/s]  9%|▉         | 35565/400000 [00:05<00:52, 6927.32it/s]  9%|▉         | 36259/400000 [00:05<00:53, 6816.01it/s]  9%|▉         | 36968/400000 [00:05<00:52, 6895.35it/s]  9%|▉         | 37669/400000 [00:05<00:52, 6929.04it/s] 10%|▉         | 38363/400000 [00:05<00:52, 6917.75it/s] 10%|▉         | 39072/400000 [00:05<00:51, 6965.26it/s] 10%|▉         | 39769/400000 [00:05<00:51, 6927.69it/s] 10%|█         | 40463/400000 [00:05<00:52, 6888.59it/s] 10%|█         | 41153/400000 [00:05<00:52, 6873.59it/s] 10%|█         | 41841/400000 [00:06<00:52, 6855.59it/s] 11%|█         | 42530/400000 [00:06<00:52, 6865.12it/s] 11%|█         | 43231/400000 [00:06<00:51, 6907.27it/s] 11%|█         | 43953/400000 [00:06<00:50, 6997.00it/s] 11%|█         | 44654/400000 [00:06<00:52, 6734.89it/s] 11%|█▏        | 45330/400000 [00:06<00:54, 6548.61it/s] 12%|█▏        | 46030/400000 [00:06<00:53, 6676.33it/s] 12%|█▏        | 46730/400000 [00:06<00:52, 6769.14it/s] 12%|█▏        | 47410/400000 [00:06<00:52, 6757.41it/s] 12%|█▏        | 48088/400000 [00:07<00:52, 6669.02it/s] 12%|█▏        | 48757/400000 [00:07<00:54, 6470.33it/s] 12%|█▏        | 49447/400000 [00:07<00:53, 6592.30it/s] 13%|█▎        | 50109/400000 [00:07<00:53, 6580.98it/s] 13%|█▎        | 50819/400000 [00:07<00:51, 6727.27it/s] 13%|█▎        | 51539/400000 [00:07<00:50, 6862.16it/s] 13%|█▎        | 52232/400000 [00:07<00:50, 6882.19it/s] 13%|█▎        | 52925/400000 [00:07<00:50, 6895.95it/s] 13%|█▎        | 53616/400000 [00:07<00:50, 6795.07it/s] 14%|█▎        | 54347/400000 [00:07<00:49, 6941.42it/s] 14%|█▍        | 55067/400000 [00:08<00:49, 7013.92it/s] 14%|█▍        | 55770/400000 [00:08<00:49, 7010.82it/s] 14%|█▍        | 56472/400000 [00:08<00:49, 6955.24it/s] 14%|█▍        | 57169/400000 [00:08<00:50, 6846.38it/s] 14%|█▍        | 57867/400000 [00:08<00:49, 6884.88it/s] 15%|█▍        | 58557/400000 [00:08<00:49, 6833.61it/s] 15%|█▍        | 59256/400000 [00:08<00:49, 6877.81it/s] 15%|█▍        | 59961/400000 [00:08<00:49, 6926.72it/s] 15%|█▌        | 60663/400000 [00:08<00:48, 6952.93it/s] 15%|█▌        | 61372/400000 [00:08<00:48, 6991.73it/s] 16%|█▌        | 62083/400000 [00:09<00:48, 7024.62it/s] 16%|█▌        | 62792/400000 [00:09<00:47, 7042.53it/s] 16%|█▌        | 63497/400000 [00:09<00:48, 6986.72it/s] 16%|█▌        | 64199/400000 [00:09<00:47, 6996.57it/s] 16%|█▌        | 64922/400000 [00:09<00:47, 7062.70it/s] 16%|█▋        | 65629/400000 [00:09<00:47, 7053.94it/s] 17%|█▋        | 66335/400000 [00:09<00:47, 7001.58it/s] 17%|█▋        | 67036/400000 [00:09<00:48, 6890.01it/s] 17%|█▋        | 67729/400000 [00:09<00:48, 6901.83it/s] 17%|█▋        | 68434/400000 [00:09<00:47, 6943.83it/s] 17%|█▋        | 69129/400000 [00:10<00:47, 6917.31it/s] 17%|█▋        | 69846/400000 [00:10<00:47, 6991.05it/s] 18%|█▊        | 70549/400000 [00:10<00:47, 7000.47it/s] 18%|█▊        | 71250/400000 [00:10<00:47, 6979.26it/s] 18%|█▊        | 71949/400000 [00:10<00:47, 6906.45it/s] 18%|█▊        | 72644/400000 [00:10<00:47, 6919.01it/s] 18%|█▊        | 73351/400000 [00:10<00:46, 6961.02it/s] 19%|█▊        | 74057/400000 [00:10<00:46, 6989.61it/s] 19%|█▊        | 74758/400000 [00:10<00:46, 6995.70it/s] 19%|█▉        | 75478/400000 [00:10<00:46, 7054.60it/s] 19%|█▉        | 76189/400000 [00:11<00:45, 7068.02it/s] 19%|█▉        | 76909/400000 [00:11<00:45, 7104.51it/s] 19%|█▉        | 77626/400000 [00:11<00:45, 7122.16it/s] 20%|█▉        | 78339/400000 [00:11<00:47, 6740.91it/s] 20%|█▉        | 79025/400000 [00:11<00:47, 6776.00it/s] 20%|█▉        | 79716/400000 [00:11<00:46, 6815.57it/s] 20%|██        | 80410/400000 [00:11<00:46, 6851.91it/s] 20%|██        | 81097/400000 [00:11<00:46, 6823.48it/s] 20%|██        | 81819/400000 [00:11<00:45, 6935.17it/s] 21%|██        | 82524/400000 [00:11<00:45, 6967.16it/s] 21%|██        | 83222/400000 [00:12<00:45, 6954.40it/s] 21%|██        | 83919/400000 [00:12<00:46, 6865.70it/s] 21%|██        | 84631/400000 [00:12<00:45, 6939.12it/s] 21%|██▏       | 85327/400000 [00:12<00:45, 6941.23it/s] 22%|██▏       | 86022/400000 [00:12<00:45, 6909.55it/s] 22%|██▏       | 86734/400000 [00:12<00:44, 6969.19it/s] 22%|██▏       | 87439/400000 [00:12<00:44, 6991.96it/s] 22%|██▏       | 88151/400000 [00:12<00:44, 7027.11it/s] 22%|██▏       | 88863/400000 [00:12<00:44, 7053.23it/s] 22%|██▏       | 89578/400000 [00:12<00:43, 7080.31it/s] 23%|██▎       | 90291/400000 [00:13<00:43, 7091.30it/s] 23%|██▎       | 91015/400000 [00:13<00:43, 7132.65it/s] 23%|██▎       | 91729/400000 [00:13<00:43, 7097.80it/s] 23%|██▎       | 92457/400000 [00:13<00:43, 7149.03it/s] 23%|██▎       | 93173/400000 [00:13<00:43, 7097.09it/s] 23%|██▎       | 93883/400000 [00:13<00:43, 7019.17it/s] 24%|██▎       | 94598/400000 [00:13<00:43, 7055.73it/s] 24%|██▍       | 95304/400000 [00:13<00:43, 7039.52it/s] 24%|██▍       | 96012/400000 [00:13<00:43, 7049.10it/s] 24%|██▍       | 96722/400000 [00:13<00:42, 7063.57it/s] 24%|██▍       | 97430/400000 [00:14<00:42, 7067.89it/s] 25%|██▍       | 98164/400000 [00:14<00:42, 7145.23it/s] 25%|██▍       | 98879/400000 [00:14<00:42, 7140.09it/s] 25%|██▍       | 99594/400000 [00:14<00:42, 7122.94it/s] 25%|██▌       | 100307/400000 [00:14<00:42, 7062.49it/s] 25%|██▌       | 101014/400000 [00:14<00:42, 7057.51it/s] 25%|██▌       | 101740/400000 [00:14<00:41, 7116.56it/s] 26%|██▌       | 102452/400000 [00:14<00:42, 6929.61it/s] 26%|██▌       | 103147/400000 [00:14<00:43, 6751.47it/s] 26%|██▌       | 103852/400000 [00:15<00:43, 6837.37it/s] 26%|██▌       | 104577/400000 [00:15<00:42, 6955.64it/s] 26%|██▋       | 105275/400000 [00:15<00:42, 6907.52it/s] 26%|██▋       | 105997/400000 [00:15<00:42, 6997.27it/s] 27%|██▋       | 106718/400000 [00:15<00:41, 7059.31it/s] 27%|██▋       | 107425/400000 [00:15<00:41, 7028.23it/s] 27%|██▋       | 108129/400000 [00:15<00:42, 6823.82it/s] 27%|██▋       | 108829/400000 [00:15<00:42, 6874.98it/s] 27%|██▋       | 109547/400000 [00:15<00:41, 6961.67it/s] 28%|██▊       | 110270/400000 [00:15<00:41, 7039.46it/s] 28%|██▊       | 110987/400000 [00:16<00:40, 7077.61it/s] 28%|██▊       | 111696/400000 [00:16<00:40, 7067.01it/s] 28%|██▊       | 112410/400000 [00:16<00:40, 7086.55it/s] 28%|██▊       | 113128/400000 [00:16<00:40, 7112.75it/s] 28%|██▊       | 113840/400000 [00:16<00:40, 7037.20it/s] 29%|██▊       | 114551/400000 [00:16<00:40, 7058.57it/s] 29%|██▉       | 115259/400000 [00:16<00:40, 7064.82it/s] 29%|██▉       | 115983/400000 [00:16<00:39, 7116.22it/s] 29%|██▉       | 116695/400000 [00:16<00:39, 7113.95it/s] 29%|██▉       | 117407/400000 [00:16<00:39, 7074.75it/s] 30%|██▉       | 118115/400000 [00:17<00:39, 7069.63it/s] 30%|██▉       | 118823/400000 [00:17<00:40, 6877.67it/s] 30%|██▉       | 119512/400000 [00:17<00:41, 6836.47it/s] 30%|███       | 120225/400000 [00:17<00:40, 6920.63it/s] 30%|███       | 120932/400000 [00:17<00:40, 6963.22it/s] 30%|███       | 121630/400000 [00:17<00:40, 6940.95it/s] 31%|███       | 122338/400000 [00:17<00:39, 6981.91it/s] 31%|███       | 123061/400000 [00:17<00:39, 7053.07it/s] 31%|███       | 123767/400000 [00:17<00:39, 7028.96it/s] 31%|███       | 124471/400000 [00:17<00:39, 6934.59it/s] 31%|███▏      | 125181/400000 [00:18<00:39, 6982.70it/s] 31%|███▏      | 125907/400000 [00:18<00:38, 7063.58it/s] 32%|███▏      | 126621/400000 [00:18<00:38, 7085.87it/s] 32%|███▏      | 127343/400000 [00:18<00:38, 7124.90it/s] 32%|███▏      | 128061/400000 [00:18<00:38, 7139.89it/s] 32%|███▏      | 128776/400000 [00:18<00:38, 7083.20it/s] 32%|███▏      | 129499/400000 [00:18<00:37, 7125.41it/s] 33%|███▎      | 130215/400000 [00:18<00:37, 7132.37it/s] 33%|███▎      | 130929/400000 [00:18<00:37, 7083.85it/s] 33%|███▎      | 131638/400000 [00:18<00:37, 7064.40it/s] 33%|███▎      | 132353/400000 [00:19<00:37, 7088.54it/s] 33%|███▎      | 133062/400000 [00:19<00:37, 7036.87it/s] 33%|███▎      | 133776/400000 [00:19<00:37, 7066.79it/s] 34%|███▎      | 134486/400000 [00:19<00:37, 7076.33it/s] 34%|███▍      | 135194/400000 [00:19<00:37, 7074.74it/s] 34%|███▍      | 135902/400000 [00:19<00:37, 7049.36it/s] 34%|███▍      | 136624/400000 [00:19<00:37, 7097.24it/s] 34%|███▍      | 137346/400000 [00:19<00:36, 7131.61it/s] 35%|███▍      | 138060/400000 [00:19<00:36, 7118.46it/s] 35%|███▍      | 138779/400000 [00:19<00:36, 7137.50it/s] 35%|███▍      | 139494/400000 [00:20<00:36, 7137.95it/s] 35%|███▌      | 140208/400000 [00:20<00:36, 7113.29it/s] 35%|███▌      | 140920/400000 [00:20<00:36, 7103.03it/s] 35%|███▌      | 141655/400000 [00:20<00:36, 7174.45it/s] 36%|███▌      | 142373/400000 [00:20<00:35, 7175.38it/s] 36%|███▌      | 143091/400000 [00:20<00:36, 7081.10it/s] 36%|███▌      | 143805/400000 [00:20<00:36, 7095.84it/s] 36%|███▌      | 144531/400000 [00:20<00:35, 7142.47it/s] 36%|███▋      | 145246/400000 [00:20<00:35, 7123.48it/s] 36%|███▋      | 145960/400000 [00:20<00:35, 7128.04it/s] 37%|███▋      | 146673/400000 [00:21<00:35, 7119.26it/s] 37%|███▋      | 147386/400000 [00:21<00:35, 7029.86it/s] 37%|███▋      | 148102/400000 [00:21<00:35, 7068.29it/s] 37%|███▋      | 148829/400000 [00:21<00:35, 7125.41it/s] 37%|███▋      | 149554/400000 [00:21<00:34, 7159.27it/s] 38%|███▊      | 150271/400000 [00:21<00:34, 7140.32it/s] 38%|███▊      | 150986/400000 [00:21<00:34, 7143.13it/s] 38%|███▊      | 151708/400000 [00:21<00:34, 7163.04it/s] 38%|███▊      | 152435/400000 [00:21<00:34, 7194.50it/s] 38%|███▊      | 153155/400000 [00:21<00:34, 7178.09it/s] 38%|███▊      | 153873/400000 [00:22<00:34, 7174.95it/s] 39%|███▊      | 154591/400000 [00:22<00:34, 7126.05it/s] 39%|███▉      | 155304/400000 [00:22<00:34, 7103.88it/s] 39%|███▉      | 156024/400000 [00:22<00:34, 7131.79it/s] 39%|███▉      | 156743/400000 [00:22<00:34, 7147.72it/s] 39%|███▉      | 157458/400000 [00:22<00:34, 7048.03it/s] 40%|███▉      | 158164/400000 [00:22<00:34, 6996.62it/s] 40%|███▉      | 158865/400000 [00:22<00:35, 6862.00it/s] 40%|███▉      | 159574/400000 [00:22<00:34, 6927.22it/s] 40%|████      | 160287/400000 [00:22<00:34, 6985.89it/s] 40%|████      | 160998/400000 [00:23<00:34, 7020.17it/s] 40%|████      | 161717/400000 [00:23<00:33, 7069.60it/s] 41%|████      | 162428/400000 [00:23<00:33, 7080.04it/s] 41%|████      | 163151/400000 [00:23<00:33, 7121.62it/s] 41%|████      | 163877/400000 [00:23<00:32, 7158.35it/s] 41%|████      | 164598/400000 [00:23<00:32, 7172.50it/s] 41%|████▏     | 165318/400000 [00:23<00:32, 7180.47it/s] 42%|████▏     | 166037/400000 [00:23<00:32, 7177.08it/s] 42%|████▏     | 166755/400000 [00:23<00:32, 7164.22it/s] 42%|████▏     | 167472/400000 [00:23<00:32, 7161.41it/s] 42%|████▏     | 168189/400000 [00:24<00:32, 7101.61it/s] 42%|████▏     | 168901/400000 [00:24<00:32, 7106.26it/s] 42%|████▏     | 169629/400000 [00:24<00:32, 7153.59it/s] 43%|████▎     | 170365/400000 [00:24<00:31, 7211.86it/s] 43%|████▎     | 171099/400000 [00:24<00:31, 7247.89it/s] 43%|████▎     | 171837/400000 [00:24<00:31, 7285.13it/s] 43%|████▎     | 172566/400000 [00:24<00:31, 7177.05it/s] 43%|████▎     | 173300/400000 [00:24<00:31, 7223.98it/s] 44%|████▎     | 174023/400000 [00:24<00:31, 7153.62it/s] 44%|████▎     | 174739/400000 [00:25<00:31, 7154.27it/s] 44%|████▍     | 175457/400000 [00:25<00:31, 7161.50it/s] 44%|████▍     | 176174/400000 [00:25<00:31, 7158.39it/s] 44%|████▍     | 176890/400000 [00:25<00:31, 7132.59it/s] 44%|████▍     | 177628/400000 [00:25<00:30, 7203.02it/s] 45%|████▍     | 178349/400000 [00:25<00:31, 7106.62it/s] 45%|████▍     | 179061/400000 [00:25<00:31, 6988.74it/s] 45%|████▍     | 179781/400000 [00:25<00:31, 7050.48it/s] 45%|████▌     | 180507/400000 [00:25<00:30, 7110.11it/s] 45%|████▌     | 181224/400000 [00:25<00:30, 7125.96it/s] 45%|████▌     | 181939/400000 [00:26<00:30, 7132.79it/s] 46%|████▌     | 182653/400000 [00:26<00:30, 7123.57it/s] 46%|████▌     | 183373/400000 [00:26<00:30, 7143.64it/s] 46%|████▌     | 184088/400000 [00:26<00:30, 7142.48it/s] 46%|████▌     | 184806/400000 [00:26<00:30, 7152.01it/s] 46%|████▋     | 185522/400000 [00:26<00:30, 7140.98it/s] 47%|████▋     | 186237/400000 [00:26<00:30, 7015.92it/s] 47%|████▋     | 186940/400000 [00:26<00:30, 6981.91it/s] 47%|████▋     | 187648/400000 [00:26<00:30, 7009.11it/s] 47%|████▋     | 188375/400000 [00:26<00:29, 7084.87it/s] 47%|████▋     | 189092/400000 [00:27<00:29, 7107.78it/s] 47%|████▋     | 189804/400000 [00:27<00:29, 7036.25it/s] 48%|████▊     | 190509/400000 [00:27<00:29, 7036.60it/s] 48%|████▊     | 191213/400000 [00:27<00:29, 7031.12it/s] 48%|████▊     | 191926/400000 [00:27<00:29, 7058.27it/s] 48%|████▊     | 192636/400000 [00:27<00:29, 7069.85it/s] 48%|████▊     | 193344/400000 [00:27<00:29, 6957.67it/s] 49%|████▊     | 194061/400000 [00:27<00:29, 7019.26it/s] 49%|████▊     | 194767/400000 [00:27<00:29, 7029.30it/s] 49%|████▉     | 195471/400000 [00:27<00:29, 6883.41it/s] 49%|████▉     | 196166/400000 [00:28<00:29, 6901.43it/s] 49%|████▉     | 196857/400000 [00:28<00:30, 6752.82it/s] 49%|████▉     | 197545/400000 [00:28<00:29, 6789.00it/s] 50%|████▉     | 198239/400000 [00:28<00:29, 6831.06it/s] 50%|████▉     | 198944/400000 [00:28<00:29, 6895.08it/s] 50%|████▉     | 199661/400000 [00:28<00:28, 6973.49it/s] 50%|█████     | 200362/400000 [00:28<00:28, 6983.01it/s] 50%|█████     | 201066/400000 [00:28<00:28, 6996.90it/s] 50%|█████     | 201768/400000 [00:28<00:28, 7000.79it/s] 51%|█████     | 202492/400000 [00:28<00:27, 7070.49it/s] 51%|█████     | 203200/400000 [00:29<00:27, 7066.29it/s] 51%|█████     | 203907/400000 [00:29<00:27, 7056.38it/s] 51%|█████     | 204613/400000 [00:29<00:27, 7049.99it/s] 51%|█████▏    | 205319/400000 [00:29<00:27, 7035.88it/s] 52%|█████▏    | 206031/400000 [00:29<00:27, 7059.17it/s] 52%|█████▏    | 206737/400000 [00:29<00:27, 6987.51it/s] 52%|█████▏    | 207436/400000 [00:29<00:27, 6946.13it/s] 52%|█████▏    | 208165/400000 [00:29<00:27, 7044.99it/s] 52%|█████▏    | 208878/400000 [00:29<00:27, 7067.51it/s] 52%|█████▏    | 209600/400000 [00:29<00:26, 7111.02it/s] 53%|█████▎    | 210312/400000 [00:30<00:27, 7014.69it/s] 53%|█████▎    | 211020/400000 [00:30<00:26, 7032.23it/s] 53%|█████▎    | 211726/400000 [00:30<00:26, 7039.67it/s] 53%|█████▎    | 212434/400000 [00:30<00:26, 7051.29it/s] 53%|█████▎    | 213140/400000 [00:30<00:27, 6690.44it/s] 53%|█████▎    | 213814/400000 [00:30<00:27, 6681.08it/s] 54%|█████▎    | 214515/400000 [00:30<00:27, 6775.58it/s] 54%|█████▍    | 215223/400000 [00:30<00:26, 6863.04it/s] 54%|█████▍    | 215938/400000 [00:30<00:26, 6944.57it/s] 54%|█████▍    | 216646/400000 [00:30<00:26, 6984.08it/s] 54%|█████▍    | 217349/400000 [00:31<00:26, 6997.10it/s] 55%|█████▍    | 218059/400000 [00:31<00:25, 7024.77it/s] 55%|█████▍    | 218763/400000 [00:31<00:25, 7024.91it/s] 55%|█████▍    | 219470/400000 [00:31<00:25, 7036.78it/s] 55%|█████▌    | 220178/400000 [00:31<00:25, 7046.48it/s] 55%|█████▌    | 220889/400000 [00:31<00:25, 7064.16it/s] 55%|█████▌    | 221596/400000 [00:31<00:25, 7021.42it/s] 56%|█████▌    | 222299/400000 [00:31<00:25, 7023.56it/s] 56%|█████▌    | 223002/400000 [00:31<00:25, 6981.44it/s] 56%|█████▌    | 223701/400000 [00:31<00:25, 6953.58it/s] 56%|█████▌    | 224397/400000 [00:32<00:25, 6936.72it/s] 56%|█████▋    | 225091/400000 [00:32<00:25, 6849.95it/s] 56%|█████▋    | 225796/400000 [00:32<00:25, 6906.77it/s] 57%|█████▋    | 226519/400000 [00:32<00:24, 7000.16it/s] 57%|█████▋    | 227234/400000 [00:32<00:24, 7043.64it/s] 57%|█████▋    | 227949/400000 [00:32<00:24, 7074.33it/s] 57%|█████▋    | 228657/400000 [00:32<00:24, 7066.05it/s] 57%|█████▋    | 229384/400000 [00:32<00:23, 7124.66it/s] 58%|█████▊    | 230097/400000 [00:32<00:23, 7088.83it/s] 58%|█████▊    | 230809/400000 [00:33<00:23, 7094.66it/s] 58%|█████▊    | 231529/400000 [00:33<00:23, 7124.23it/s] 58%|█████▊    | 232242/400000 [00:33<00:23, 7121.89it/s] 58%|█████▊    | 232962/400000 [00:33<00:23, 7144.05it/s] 58%|█████▊    | 233685/400000 [00:33<00:23, 7167.87it/s] 59%|█████▊    | 234409/400000 [00:33<00:23, 7188.59it/s] 59%|█████▉    | 235128/400000 [00:33<00:23, 7153.57it/s] 59%|█████▉    | 235844/400000 [00:33<00:23, 7101.60it/s] 59%|█████▉    | 236558/400000 [00:33<00:22, 7110.98it/s] 59%|█████▉    | 237270/400000 [00:33<00:22, 7079.93it/s] 59%|█████▉    | 237985/400000 [00:34<00:22, 7099.86it/s] 60%|█████▉    | 238718/400000 [00:34<00:22, 7164.02it/s] 60%|█████▉    | 239435/400000 [00:34<00:22, 7152.01it/s] 60%|██████    | 240151/400000 [00:34<00:22, 7149.45it/s] 60%|██████    | 240874/400000 [00:34<00:22, 7170.73it/s] 60%|██████    | 241592/400000 [00:34<00:22, 7159.53it/s] 61%|██████    | 242309/400000 [00:34<00:22, 7102.84it/s] 61%|██████    | 243025/400000 [00:34<00:22, 7118.13it/s] 61%|██████    | 243737/400000 [00:34<00:22, 6968.23it/s] 61%|██████    | 244443/400000 [00:34<00:22, 6995.28it/s] 61%|██████▏   | 245154/400000 [00:35<00:22, 7025.34it/s] 61%|██████▏   | 245867/400000 [00:35<00:21, 7055.61it/s] 62%|██████▏   | 246585/400000 [00:35<00:21, 7090.77it/s] 62%|██████▏   | 247295/400000 [00:35<00:21, 7019.93it/s] 62%|██████▏   | 248008/400000 [00:35<00:21, 7051.21it/s] 62%|██████▏   | 248714/400000 [00:35<00:21, 6982.37it/s] 62%|██████▏   | 249413/400000 [00:35<00:21, 6846.51it/s] 63%|██████▎   | 250122/400000 [00:35<00:21, 6915.25it/s] 63%|██████▎   | 250827/400000 [00:35<00:21, 6952.89it/s] 63%|██████▎   | 251535/400000 [00:35<00:21, 6988.37it/s] 63%|██████▎   | 252235/400000 [00:36<00:21, 6971.07it/s] 63%|██████▎   | 252948/400000 [00:36<00:20, 7016.53it/s] 63%|██████▎   | 253650/400000 [00:36<00:20, 6972.01it/s] 64%|██████▎   | 254348/400000 [00:36<00:21, 6905.45it/s] 64%|██████▍   | 255057/400000 [00:36<00:20, 6958.09it/s] 64%|██████▍   | 255786/400000 [00:36<00:20, 7051.93it/s] 64%|██████▍   | 256502/400000 [00:36<00:20, 7083.77it/s] 64%|██████▍   | 257211/400000 [00:36<00:20, 7069.40it/s] 64%|██████▍   | 257925/400000 [00:36<00:20, 7088.62it/s] 65%|██████▍   | 258635/400000 [00:36<00:20, 7046.63it/s] 65%|██████▍   | 259340/400000 [00:37<00:20, 7024.67it/s] 65%|██████▌   | 260056/400000 [00:37<00:19, 7062.13it/s] 65%|██████▌   | 260769/400000 [00:37<00:19, 7080.50it/s] 65%|██████▌   | 261478/400000 [00:37<00:19, 7077.62it/s] 66%|██████▌   | 262195/400000 [00:37<00:19, 7104.03it/s] 66%|██████▌   | 262906/400000 [00:37<00:19, 7084.25it/s] 66%|██████▌   | 263615/400000 [00:37<00:19, 7083.03it/s] 66%|██████▌   | 264339/400000 [00:37<00:19, 7127.46it/s] 66%|██████▋   | 265052/400000 [00:37<00:18, 7116.65it/s] 66%|██████▋   | 265764/400000 [00:37<00:18, 7080.55it/s] 67%|██████▋   | 266473/400000 [00:38<00:19, 6968.38it/s] 67%|██████▋   | 267181/400000 [00:38<00:18, 7000.18it/s] 67%|██████▋   | 267892/400000 [00:38<00:18, 7030.24it/s] 67%|██████▋   | 268596/400000 [00:38<00:18, 7005.07it/s] 67%|██████▋   | 269297/400000 [00:38<00:18, 6997.31it/s] 68%|██████▊   | 270007/400000 [00:38<00:18, 7027.58it/s] 68%|██████▊   | 270722/400000 [00:38<00:18, 7060.73it/s] 68%|██████▊   | 271429/400000 [00:38<00:18, 7049.63it/s] 68%|██████▊   | 272135/400000 [00:38<00:18, 7007.52it/s] 68%|██████▊   | 272848/400000 [00:38<00:18, 7042.09it/s] 68%|██████▊   | 273554/400000 [00:39<00:17, 7045.17it/s] 69%|██████▊   | 274268/400000 [00:39<00:17, 7072.27it/s] 69%|██████▊   | 274976/400000 [00:39<00:17, 7059.48it/s] 69%|██████▉   | 275683/400000 [00:39<00:17, 7006.06it/s] 69%|██████▉   | 276390/400000 [00:39<00:17, 7024.21it/s] 69%|██████▉   | 277093/400000 [00:39<00:17, 6920.66it/s] 69%|██████▉   | 277816/400000 [00:39<00:17, 7008.53it/s] 70%|██████▉   | 278531/400000 [00:39<00:17, 7048.77it/s] 70%|██████▉   | 279237/400000 [00:39<00:17, 7042.67it/s] 70%|██████▉   | 279942/400000 [00:39<00:17, 7025.97it/s] 70%|███████   | 280651/400000 [00:40<00:16, 7044.31it/s] 70%|███████   | 281366/400000 [00:40<00:16, 7074.71it/s] 71%|███████   | 282084/400000 [00:40<00:16, 7104.60it/s] 71%|███████   | 282795/400000 [00:40<00:16, 6918.60it/s] 71%|███████   | 283502/400000 [00:40<00:16, 6961.45it/s] 71%|███████   | 284200/400000 [00:40<00:16, 6935.17it/s] 71%|███████   | 284914/400000 [00:40<00:16, 6994.61it/s] 71%|███████▏  | 285615/400000 [00:40<00:16, 6794.57it/s] 72%|███████▏  | 286310/400000 [00:40<00:16, 6839.93it/s] 72%|███████▏  | 287032/400000 [00:40<00:16, 6948.30it/s] 72%|███████▏  | 287756/400000 [00:41<00:15, 7031.56it/s] 72%|███████▏  | 288477/400000 [00:41<00:15, 7081.69it/s] 72%|███████▏  | 289198/400000 [00:41<00:15, 7117.26it/s] 72%|███████▏  | 289911/400000 [00:41<00:15, 7113.09it/s] 73%|███████▎  | 290623/400000 [00:41<00:15, 7103.93it/s] 73%|███████▎  | 291350/400000 [00:41<00:15, 7152.67it/s] 73%|███████▎  | 292067/400000 [00:41<00:15, 7155.77it/s] 73%|███████▎  | 292783/400000 [00:41<00:14, 7152.29it/s] 73%|███████▎  | 293499/400000 [00:41<00:14, 7150.02it/s] 74%|███████▎  | 294215/400000 [00:41<00:14, 7146.58it/s] 74%|███████▎  | 294930/400000 [00:42<00:14, 7127.78it/s] 74%|███████▍  | 295643/400000 [00:42<00:14, 7091.09it/s] 74%|███████▍  | 296353/400000 [00:42<00:14, 7090.18it/s] 74%|███████▍  | 297064/400000 [00:42<00:14, 7094.50it/s] 74%|███████▍  | 297774/400000 [00:42<00:14, 7071.39it/s] 75%|███████▍  | 298498/400000 [00:42<00:14, 7118.67it/s] 75%|███████▍  | 299210/400000 [00:42<00:14, 6999.62it/s] 75%|███████▍  | 299933/400000 [00:42<00:14, 7064.97it/s] 75%|███████▌  | 300641/400000 [00:42<00:14, 7050.27it/s] 75%|███████▌  | 301347/400000 [00:43<00:14, 7042.09it/s] 76%|███████▌  | 302063/400000 [00:43<00:13, 7072.87it/s] 76%|███████▌  | 302789/400000 [00:43<00:13, 7127.24it/s] 76%|███████▌  | 303502/400000 [00:43<00:13, 7042.02it/s] 76%|███████▌  | 304207/400000 [00:43<00:13, 6995.38it/s] 76%|███████▌  | 304926/400000 [00:43<00:13, 7051.04it/s] 76%|███████▋  | 305632/400000 [00:43<00:13, 7000.98it/s] 77%|███████▋  | 306357/400000 [00:43<00:13, 7073.74it/s] 77%|███████▋  | 307077/400000 [00:43<00:13, 7108.61it/s] 77%|███████▋  | 307789/400000 [00:43<00:12, 7105.92it/s] 77%|███████▋  | 308503/400000 [00:44<00:12, 7115.24it/s] 77%|███████▋  | 309215/400000 [00:44<00:12, 7081.65it/s] 77%|███████▋  | 309942/400000 [00:44<00:12, 7136.50it/s] 78%|███████▊  | 310663/400000 [00:44<00:12, 7156.55it/s] 78%|███████▊  | 311379/400000 [00:44<00:12, 7122.05it/s] 78%|███████▊  | 312092/400000 [00:44<00:12, 7117.17it/s] 78%|███████▊  | 312808/400000 [00:44<00:12, 7128.46it/s] 78%|███████▊  | 313533/400000 [00:44<00:12, 7161.72it/s] 79%|███████▊  | 314256/400000 [00:44<00:11, 7180.92it/s] 79%|███████▊  | 314975/400000 [00:44<00:11, 7155.38it/s] 79%|███████▉  | 315693/400000 [00:45<00:11, 7160.98it/s] 79%|███████▉  | 316410/400000 [00:45<00:11, 7145.61it/s] 79%|███████▉  | 317125/400000 [00:45<00:11, 7143.78it/s] 79%|███████▉  | 317840/400000 [00:45<00:11, 7127.61it/s] 80%|███████▉  | 318560/400000 [00:45<00:11, 7148.02it/s] 80%|███████▉  | 319275/400000 [00:45<00:11, 7147.63it/s] 80%|███████▉  | 319990/400000 [00:45<00:11, 7129.80it/s] 80%|████████  | 320711/400000 [00:45<00:11, 7152.90it/s] 80%|████████  | 321434/400000 [00:45<00:10, 7174.50it/s] 81%|████████  | 322152/400000 [00:45<00:11, 6957.90it/s] 81%|████████  | 322858/400000 [00:46<00:11, 6987.69it/s] 81%|████████  | 323574/400000 [00:46<00:10, 7036.56it/s] 81%|████████  | 324295/400000 [00:46<00:10, 7083.64it/s] 81%|████████▏ | 325005/400000 [00:46<00:10, 7083.71it/s] 81%|████████▏ | 325714/400000 [00:46<00:10, 7053.58it/s] 82%|████████▏ | 326429/400000 [00:46<00:10, 7080.24it/s] 82%|████████▏ | 327138/400000 [00:46<00:10, 7056.92it/s] 82%|████████▏ | 327844/400000 [00:46<00:10, 7035.36it/s] 82%|████████▏ | 328557/400000 [00:46<00:10, 7062.12it/s] 82%|████████▏ | 329264/400000 [00:46<00:10, 7061.04it/s] 82%|████████▏ | 329971/400000 [00:47<00:09, 7061.14it/s] 83%|████████▎ | 330678/400000 [00:47<00:09, 7025.91it/s] 83%|████████▎ | 331381/400000 [00:47<00:09, 6917.84it/s] 83%|████████▎ | 332093/400000 [00:47<00:09, 6975.56it/s] 83%|████████▎ | 332791/400000 [00:47<00:09, 6963.37it/s] 83%|████████▎ | 333504/400000 [00:47<00:09, 7009.43it/s] 84%|████████▎ | 334219/400000 [00:47<00:09, 7049.91it/s] 84%|████████▎ | 334943/400000 [00:47<00:09, 7105.32it/s] 84%|████████▍ | 335666/400000 [00:47<00:09, 7140.94it/s] 84%|████████▍ | 336383/400000 [00:47<00:08, 7146.94it/s] 84%|████████▍ | 337098/400000 [00:48<00:08, 7065.34it/s] 84%|████████▍ | 337805/400000 [00:48<00:09, 6740.10it/s] 85%|████████▍ | 338506/400000 [00:48<00:09, 6816.68it/s] 85%|████████▍ | 339225/400000 [00:48<00:08, 6922.23it/s] 85%|████████▍ | 339939/400000 [00:48<00:08, 6985.06it/s] 85%|████████▌ | 340658/400000 [00:48<00:08, 7045.10it/s] 85%|████████▌ | 341364/400000 [00:48<00:08, 6986.56it/s] 86%|████████▌ | 342069/400000 [00:48<00:08, 7005.37it/s] 86%|████████▌ | 342777/400000 [00:48<00:08, 7026.14it/s] 86%|████████▌ | 343481/400000 [00:48<00:08, 7027.00it/s] 86%|████████▌ | 344198/400000 [00:49<00:07, 7068.29it/s] 86%|████████▌ | 344923/400000 [00:49<00:07, 7120.98it/s] 86%|████████▋ | 345639/400000 [00:49<00:07, 7132.61it/s] 87%|████████▋ | 346361/400000 [00:49<00:07, 7158.17it/s] 87%|████████▋ | 347084/400000 [00:49<00:07, 7177.23it/s] 87%|████████▋ | 347802/400000 [00:49<00:07, 7171.87it/s] 87%|████████▋ | 348527/400000 [00:49<00:07, 7194.15it/s] 87%|████████▋ | 349247/400000 [00:49<00:07, 7166.67it/s] 87%|████████▋ | 349967/400000 [00:49<00:06, 7175.01it/s] 88%|████████▊ | 350687/400000 [00:49<00:06, 7182.12it/s] 88%|████████▊ | 351406/400000 [00:50<00:06, 7166.41it/s] 88%|████████▊ | 352131/400000 [00:50<00:06, 7188.69it/s] 88%|████████▊ | 352860/400000 [00:50<00:06, 7216.39it/s] 88%|████████▊ | 353582/400000 [00:50<00:06, 7202.53it/s] 89%|████████▊ | 354303/400000 [00:50<00:06, 7073.02it/s] 89%|████████▉ | 355011/400000 [00:50<00:06, 7030.21it/s] 89%|████████▉ | 355718/400000 [00:50<00:06, 7040.61it/s] 89%|████████▉ | 356434/400000 [00:50<00:06, 7073.31it/s] 89%|████████▉ | 357149/400000 [00:50<00:06, 7093.77it/s] 89%|████████▉ | 357871/400000 [00:50<00:05, 7128.43it/s] 90%|████████▉ | 358585/400000 [00:51<00:05, 7120.71it/s] 90%|████████▉ | 359298/400000 [00:51<00:05, 7104.05it/s] 90%|█████████ | 360009/400000 [00:51<00:05, 7055.56it/s] 90%|█████████ | 360715/400000 [00:51<00:05, 7048.64it/s] 90%|█████████ | 361420/400000 [00:51<00:05, 7031.02it/s] 91%|█████████ | 362124/400000 [00:51<00:05, 6976.42it/s] 91%|█████████ | 362842/400000 [00:51<00:05, 7034.60it/s] 91%|█████████ | 363561/400000 [00:51<00:05, 7080.06it/s] 91%|█████████ | 364279/400000 [00:51<00:05, 7107.57it/s] 91%|█████████ | 364990/400000 [00:51<00:04, 7105.12it/s] 91%|█████████▏| 365703/400000 [00:52<00:04, 7112.36it/s] 92%|█████████▏| 366415/400000 [00:52<00:04, 7100.77it/s] 92%|█████████▏| 367129/400000 [00:52<00:04, 7112.10it/s] 92%|█████████▏| 367850/400000 [00:52<00:04, 7138.99it/s] 92%|█████████▏| 368567/400000 [00:52<00:04, 7146.58it/s] 92%|█████████▏| 369282/400000 [00:52<00:04, 7034.05it/s] 93%|█████████▎| 370001/400000 [00:52<00:04, 7078.92it/s] 93%|█████████▎| 370717/400000 [00:52<00:04, 7100.25it/s] 93%|█████████▎| 371428/400000 [00:52<00:04, 7045.83it/s] 93%|█████████▎| 372133/400000 [00:53<00:04, 6858.14it/s] 93%|█████████▎| 372836/400000 [00:53<00:03, 6907.86it/s] 93%|█████████▎| 373537/400000 [00:53<00:03, 6936.09it/s] 94%|█████████▎| 374237/400000 [00:53<00:03, 6954.62it/s] 94%|█████████▎| 374933/400000 [00:53<00:03, 6915.02it/s] 94%|█████████▍| 375645/400000 [00:53<00:03, 6973.54it/s] 94%|█████████▍| 376343/400000 [00:53<00:03, 6925.44it/s] 94%|█████████▍| 377040/400000 [00:53<00:03, 6937.65it/s] 94%|█████████▍| 377735/400000 [00:53<00:03, 6918.89it/s] 95%|█████████▍| 378429/400000 [00:53<00:03, 6922.40it/s] 95%|█████████▍| 379122/400000 [00:54<00:03, 6913.74it/s] 95%|█████████▍| 379815/400000 [00:54<00:02, 6917.71it/s] 95%|█████████▌| 380507/400000 [00:54<00:02, 6878.64it/s] 95%|█████████▌| 381221/400000 [00:54<00:02, 6954.09it/s] 95%|█████████▌| 381932/400000 [00:54<00:02, 6998.42it/s] 96%|█████████▌| 382633/400000 [00:54<00:02, 6976.71it/s] 96%|█████████▌| 383340/400000 [00:54<00:02, 7004.38it/s] 96%|█████████▌| 384041/400000 [00:54<00:02, 6992.75it/s] 96%|█████████▌| 384741/400000 [00:54<00:02, 6988.52it/s] 96%|█████████▋| 385454/400000 [00:54<00:02, 7028.95it/s] 97%|█████████▋| 386159/400000 [00:55<00:01, 7033.88it/s] 97%|█████████▋| 386863/400000 [00:55<00:01, 6951.23it/s] 97%|█████████▋| 387577/400000 [00:55<00:01, 7005.59it/s] 97%|█████████▋| 388278/400000 [00:55<00:01, 6846.64it/s] 97%|█████████▋| 388967/400000 [00:55<00:01, 6859.02it/s] 97%|█████████▋| 389659/400000 [00:55<00:01, 6875.35it/s] 98%|█████████▊| 390365/400000 [00:55<00:01, 6927.67it/s] 98%|█████████▊| 391068/400000 [00:55<00:01, 6955.14it/s] 98%|█████████▊| 391774/400000 [00:55<00:01, 6985.52it/s] 98%|█████████▊| 392490/400000 [00:55<00:01, 7036.89it/s] 98%|█████████▊| 393205/400000 [00:56<00:00, 7067.87it/s] 98%|█████████▊| 393913/400000 [00:56<00:00, 7067.91it/s] 99%|█████████▊| 394620/400000 [00:56<00:00, 7068.50it/s] 99%|█████████▉| 395333/400000 [00:56<00:00, 7083.84it/s] 99%|█████████▉| 396042/400000 [00:56<00:00, 6956.33it/s] 99%|█████████▉| 396762/400000 [00:56<00:00, 7024.91it/s] 99%|█████████▉| 397466/400000 [00:56<00:00, 7025.83it/s]100%|█████████▉| 398184/400000 [00:56<00:00, 7068.34it/s]100%|█████████▉| 398892/400000 [00:56<00:00, 7071.83it/s]100%|█████████▉| 399600/400000 [00:56<00:00, 7048.64it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 7017.52it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5ff2218cf8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011729482673268461 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.01166509226413076 	 Accuracy: 47

  model saves at 47% accuracy 

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

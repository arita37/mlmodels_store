
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb106d41fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 00:18:53.665691
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 00:18:53.670002
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 00:18:53.673626
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 00:18:53.677484
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb112d59400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354264.9688
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 282161.2812
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 205919.6094
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 141088.0312
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 93811.0703
Epoch 6/10

1/1 [==============================] - 0s 107ms/step - loss: 61912.6406
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 41400.5391
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 28447.5000
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 20180.3281
Epoch 10/10

1/1 [==============================] - 0s 100ms/step - loss: 14971.5410

  #### Inference Need return ypred, ytrue ######################### 
[[-1.95227936e-01 -6.64105594e-01  9.30916071e-01 -8.70901346e-01
  -8.28405261e-01  4.04276289e-02  1.45019472e-01 -1.29932702e+00
  -1.99880853e-01  2.42359057e-01 -1.11699313e-01  5.53030968e-02
   1.14149141e+00  4.62026179e-01  2.60165811e-01  1.02468646e+00
  -7.50068843e-01  1.03250325e-01 -2.25094378e-01  1.03937423e+00
  -2.19926238e-03 -2.09660888e-01 -1.09246969e+00 -1.53977656e+00
  -6.62195802e-01  1.01388812e+00 -1.32458019e+00  6.66702509e-01
   2.27289557e-01  4.51288164e-01 -1.92441076e-01  5.66642940e-01
   9.21322703e-02 -1.12070012e+00  3.68121147e-01  3.33529800e-01
   6.79845929e-01  1.22691691e-01 -1.24311037e-01 -4.66788769e-01
  -1.29052091e+00 -4.73588020e-01  4.67834711e-01 -3.41666341e-01
   1.34556681e-01  1.32510543e+00 -2.61466444e-01 -7.73294687e-01
  -3.03814709e-01 -9.85600412e-01  9.75343823e-01  5.79297841e-02
  -2.33576775e-01 -6.25331700e-01 -1.01552403e+00 -1.77148044e-01
  -1.29434884e-01 -1.01096332e-01 -1.32634431e-01 -4.26367849e-01
  -6.08908981e-02  4.07839680e+00  3.35378003e+00  5.81190968e+00
   4.38529348e+00  4.26679945e+00  5.45568514e+00  3.62298870e+00
   3.41917562e+00  4.88962889e+00  4.35432291e+00  5.13848495e+00
   4.49852514e+00  5.92068577e+00  4.19221163e+00  5.93620682e+00
   3.84895182e+00  4.36176395e+00  4.67251825e+00  4.74854374e+00
   4.91212082e+00  5.09937096e+00  5.77776194e+00  3.36386299e+00
   4.30928516e+00  5.81400108e+00  3.92721105e+00  5.92888308e+00
   4.30934906e+00  3.81931114e+00  4.26720381e+00  5.27121258e+00
   5.13211489e+00  5.24039364e+00  4.86063242e+00  4.58782053e+00
   3.39597249e+00  5.80645466e+00  6.25533247e+00  5.73049164e+00
   5.14926386e+00  5.47063065e+00  4.62851238e+00  4.96594238e+00
   5.32332087e+00  4.93436861e+00  5.61777258e+00  5.02472496e+00
   4.41156149e+00  5.04816341e+00  4.32460928e+00  4.66848278e+00
   5.06928682e+00  3.71027303e+00  4.88807964e+00  4.63326120e+00
   4.79808760e+00  4.90691614e+00  4.20456600e+00  5.03144979e+00
  -8.99761796e-01 -4.88103718e-01  2.00238004e-01 -2.05654204e-02
  -1.14507127e+00  2.78595030e-01 -3.76701117e-01 -1.12507975e+00
  -1.34497046e+00 -5.06542265e-01  1.12919283e+00 -2.90990323e-01
  -4.58329201e-01 -3.81240010e-01 -9.05637145e-02 -1.16529912e-01
  -4.52474564e-01 -2.54742354e-01 -3.93214405e-01 -1.92186296e-01
  -5.77436209e-01 -7.68251717e-01 -1.85538054e-01 -3.57002079e-01
  -8.04411173e-01 -4.72401917e-01  5.00414371e-01  3.76737177e-01
   4.61429060e-02  1.00563312e+00 -8.67676139e-02  9.27354813e-01
  -3.03522170e-01  3.30142379e-01 -6.41255230e-02  3.74277890e-01
  -3.37233663e-01  5.50259590e-01 -6.11678541e-01  4.59460258e-01
   4.98568296e-01 -1.15336967e+00 -3.19544137e-01  5.90047717e-01
   9.02073860e-01 -6.04471385e-01 -2.51505047e-01 -1.32300603e+00
   3.74370217e-01 -9.33708489e-01 -1.46419513e+00  3.54102671e-01
  -1.90424502e-01  8.17995310e-01 -1.46957469e+00 -8.59271049e-01
   7.67345279e-02 -3.29893321e-01  2.12671101e-01  4.74887431e-01
   1.15499115e+00  3.35537910e-01  8.56168628e-01  9.54093158e-01
   1.94104159e+00  1.81562734e+00  8.96116734e-01  1.97240758e+00
   3.45533371e-01  3.53735805e-01  1.53172970e+00  2.34482169e-01
   1.65166259e+00  6.43912375e-01  7.18441129e-01  1.01163137e+00
   1.44661820e+00  1.14626074e+00  1.94978034e+00  7.33199954e-01
   3.25229764e-01  1.02775300e+00  1.33636987e+00  7.56109595e-01
   2.18995285e+00  6.06913626e-01  9.89526749e-01  1.04727697e+00
   1.13749015e+00  1.38209677e+00  1.81432688e+00  8.00731957e-01
   3.03095937e-01  3.86298001e-01  1.13225234e+00  1.30908704e+00
   1.56702995e+00  1.51738012e+00  4.81535554e-01  1.14043963e+00
   1.32179356e+00  8.96924317e-01  2.56348419e+00  1.65652990e+00
   6.17422283e-01  7.79926598e-01  1.99478626e+00  8.21779609e-01
   1.62126565e+00  1.05814517e+00  2.62352109e-01  1.06266987e+00
   4.92042542e-01  1.46335959e+00  7.67709494e-01  1.14049399e+00
   1.33579075e+00  6.70571148e-01  1.17716205e+00  7.58490384e-01
   2.59602666e-02  5.86088800e+00  4.36023474e+00  5.10916042e+00
   6.41968966e+00  4.75373745e+00  6.22397327e+00  5.44798946e+00
   4.67464209e+00  4.97114134e+00  5.46204519e+00  5.87639189e+00
   5.46552849e+00  4.94535589e+00  6.16938782e+00  6.42043447e+00
   5.73058510e+00  5.72379589e+00  5.98089886e+00  5.81774282e+00
   5.91798973e+00  5.66284609e+00  4.88786936e+00  5.19708395e+00
   5.25712872e+00  5.53096914e+00  6.10376358e+00  6.17667770e+00
   4.51720095e+00  5.90203524e+00  5.47372866e+00  6.38441992e+00
   6.09927559e+00  6.36236763e+00  4.12386560e+00  4.73329639e+00
   6.08225298e+00  6.05020857e+00  4.50189924e+00  5.49099207e+00
   5.89985752e+00  5.25683165e+00  5.19770336e+00  4.63977575e+00
   6.16202688e+00  6.17684174e+00  5.52353191e+00  5.87671041e+00
   6.66956615e+00  6.03403568e+00  4.93598509e+00  5.48083782e+00
   5.18729639e+00  4.85668898e+00  5.75877142e+00  5.44707870e+00
   5.87449598e+00  6.09904385e+00  5.59261656e+00  6.02806330e+00
   1.30260754e+00  6.99323416e-01  2.15907216e-01  2.06780958e+00
   1.26658309e+00  7.00106323e-01  1.07130623e+00  6.28676713e-01
   6.05472863e-01  1.95093012e+00  4.84526098e-01  2.89803267e-01
   9.90547597e-01  1.51868176e+00  8.74291539e-01  1.01549077e+00
   1.22965872e+00  3.88460994e-01  1.25254846e+00  3.27838182e-01
   1.30124235e+00  2.00641298e+00  1.92943203e+00  1.21982050e+00
   2.34391141e+00  1.51339364e+00  3.77250493e-01  7.00836837e-01
   4.30291653e-01  1.25245237e+00  1.34014428e+00  6.91843331e-01
   3.24598432e-01  1.59760034e+00  1.49934649e+00  7.68650174e-01
   1.58109069e+00  3.39913249e-01  9.47822809e-01  3.04652214e-01
   5.05551636e-01  1.12725520e+00  2.00036240e+00  1.59428716e+00
   1.82888269e+00  6.01418436e-01  1.41860664e+00  1.52552748e+00
   5.36256135e-01  1.51713800e+00  1.47804284e+00  2.09139204e+00
   1.17558658e+00  1.36366320e+00  1.04525232e+00  1.04500067e+00
   6.37476325e-01  9.03708398e-01  3.32857013e-01  6.45005107e-01
  -4.88085699e+00  5.19317102e+00 -6.10874891e-02]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 00:19:03.141975
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.7511
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 00:19:03.146809
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9571.38
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 00:19:03.150522
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.2209
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 00:19:03.154479
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -856.185
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140397943755272
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140397002395888
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140397002396392
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140397002396896
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140397002397400
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140397002397904

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb0f296dfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.634843
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.602511
grad_step = 000002, loss = 0.575661
grad_step = 000003, loss = 0.547504
grad_step = 000004, loss = 0.516169
grad_step = 000005, loss = 0.485851
grad_step = 000006, loss = 0.466109
grad_step = 000007, loss = 0.455555
grad_step = 000008, loss = 0.446207
grad_step = 000009, loss = 0.431180
grad_step = 000010, loss = 0.412882
grad_step = 000011, loss = 0.399204
grad_step = 000012, loss = 0.390604
grad_step = 000013, loss = 0.381327
grad_step = 000014, loss = 0.367485
grad_step = 000015, loss = 0.350517
grad_step = 000016, loss = 0.332912
grad_step = 000017, loss = 0.317164
grad_step = 000018, loss = 0.305576
grad_step = 000019, loss = 0.297790
grad_step = 000020, loss = 0.289210
grad_step = 000021, loss = 0.277996
grad_step = 000022, loss = 0.266841
grad_step = 000023, loss = 0.256886
grad_step = 000024, loss = 0.247840
grad_step = 000025, loss = 0.238565
grad_step = 000026, loss = 0.228643
grad_step = 000027, loss = 0.219129
grad_step = 000028, loss = 0.208707
grad_step = 000029, loss = 0.198643
grad_step = 000030, loss = 0.189107
grad_step = 000031, loss = 0.180284
grad_step = 000032, loss = 0.171794
grad_step = 000033, loss = 0.163511
grad_step = 000034, loss = 0.155569
grad_step = 000035, loss = 0.148018
grad_step = 000036, loss = 0.140788
grad_step = 000037, loss = 0.133455
grad_step = 000038, loss = 0.126093
grad_step = 000039, loss = 0.119127
grad_step = 000040, loss = 0.112745
grad_step = 000041, loss = 0.106901
grad_step = 000042, loss = 0.101054
grad_step = 000043, loss = 0.095250
grad_step = 000044, loss = 0.089651
grad_step = 000045, loss = 0.084463
grad_step = 000046, loss = 0.079745
grad_step = 000047, loss = 0.075066
grad_step = 000048, loss = 0.070421
grad_step = 000049, loss = 0.065940
grad_step = 000050, loss = 0.061772
grad_step = 000051, loss = 0.057976
grad_step = 000052, loss = 0.054370
grad_step = 000053, loss = 0.050811
grad_step = 000054, loss = 0.047377
grad_step = 000055, loss = 0.044190
grad_step = 000056, loss = 0.041185
grad_step = 000057, loss = 0.038357
grad_step = 000058, loss = 0.035589
grad_step = 000059, loss = 0.032980
grad_step = 000060, loss = 0.030581
grad_step = 000061, loss = 0.028340
grad_step = 000062, loss = 0.026207
grad_step = 000063, loss = 0.024151
grad_step = 000064, loss = 0.022247
grad_step = 000065, loss = 0.020498
grad_step = 000066, loss = 0.018853
grad_step = 000067, loss = 0.017286
grad_step = 000068, loss = 0.015832
grad_step = 000069, loss = 0.014516
grad_step = 000070, loss = 0.013308
grad_step = 000071, loss = 0.012170
grad_step = 000072, loss = 0.011115
grad_step = 000073, loss = 0.010156
grad_step = 000074, loss = 0.009293
grad_step = 000075, loss = 0.008499
grad_step = 000076, loss = 0.007770
grad_step = 000077, loss = 0.007123
grad_step = 000078, loss = 0.006534
grad_step = 000079, loss = 0.005998
grad_step = 000080, loss = 0.005524
grad_step = 000081, loss = 0.005109
grad_step = 000082, loss = 0.004735
grad_step = 000083, loss = 0.004396
grad_step = 000084, loss = 0.004104
grad_step = 000085, loss = 0.003849
grad_step = 000086, loss = 0.003620
grad_step = 000087, loss = 0.003417
grad_step = 000088, loss = 0.003245
grad_step = 000089, loss = 0.003098
grad_step = 000090, loss = 0.002969
grad_step = 000091, loss = 0.002861
grad_step = 000092, loss = 0.002767
grad_step = 000093, loss = 0.002684
grad_step = 000094, loss = 0.002614
grad_step = 000095, loss = 0.002558
grad_step = 000096, loss = 0.002510
grad_step = 000097, loss = 0.002467
grad_step = 000098, loss = 0.002432
grad_step = 000099, loss = 0.002403
grad_step = 000100, loss = 0.002378
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002357
grad_step = 000102, loss = 0.002339
grad_step = 000103, loss = 0.002322
grad_step = 000104, loss = 0.002307
grad_step = 000105, loss = 0.002295
grad_step = 000106, loss = 0.002284
grad_step = 000107, loss = 0.002273
grad_step = 000108, loss = 0.002265
grad_step = 000109, loss = 0.002259
grad_step = 000110, loss = 0.002258
grad_step = 000111, loss = 0.002268
grad_step = 000112, loss = 0.002286
grad_step = 000113, loss = 0.002307
grad_step = 000114, loss = 0.002274
grad_step = 000115, loss = 0.002217
grad_step = 000116, loss = 0.002197
grad_step = 000117, loss = 0.002223
grad_step = 000118, loss = 0.002232
grad_step = 000119, loss = 0.002190
grad_step = 000120, loss = 0.002166
grad_step = 000121, loss = 0.002183
grad_step = 000122, loss = 0.002187
grad_step = 000123, loss = 0.002161
grad_step = 000124, loss = 0.002144
grad_step = 000125, loss = 0.002152
grad_step = 000126, loss = 0.002155
grad_step = 000127, loss = 0.002139
grad_step = 000128, loss = 0.002126
grad_step = 000129, loss = 0.002129
grad_step = 000130, loss = 0.002133
grad_step = 000131, loss = 0.002122
grad_step = 000132, loss = 0.002109
grad_step = 000133, loss = 0.002114
grad_step = 000134, loss = 0.002117
grad_step = 000135, loss = 0.002108
grad_step = 000136, loss = 0.002100
grad_step = 000137, loss = 0.002099
grad_step = 000138, loss = 0.002100
grad_step = 000139, loss = 0.002099
grad_step = 000140, loss = 0.002091
grad_step = 000141, loss = 0.002085
grad_step = 000142, loss = 0.002086
grad_step = 000143, loss = 0.002086
grad_step = 000144, loss = 0.002084
grad_step = 000145, loss = 0.002081
grad_step = 000146, loss = 0.002078
grad_step = 000147, loss = 0.002074
grad_step = 000148, loss = 0.002069
grad_step = 000149, loss = 0.002068
grad_step = 000150, loss = 0.002067
grad_step = 000151, loss = 0.002065
grad_step = 000152, loss = 0.002064
grad_step = 000153, loss = 0.002065
grad_step = 000154, loss = 0.002067
grad_step = 000155, loss = 0.002070
grad_step = 000156, loss = 0.002075
grad_step = 000157, loss = 0.002084
grad_step = 000158, loss = 0.002086
grad_step = 000159, loss = 0.002084
grad_step = 000160, loss = 0.002073
grad_step = 000161, loss = 0.002056
grad_step = 000162, loss = 0.002041
grad_step = 000163, loss = 0.002037
grad_step = 000164, loss = 0.002041
grad_step = 000165, loss = 0.002047
grad_step = 000166, loss = 0.002053
grad_step = 000167, loss = 0.002051
grad_step = 000168, loss = 0.002043
grad_step = 000169, loss = 0.002033
grad_step = 000170, loss = 0.002024
grad_step = 000171, loss = 0.002018
grad_step = 000172, loss = 0.002018
grad_step = 000173, loss = 0.002020
grad_step = 000174, loss = 0.002022
grad_step = 000175, loss = 0.002025
grad_step = 000176, loss = 0.002026
grad_step = 000177, loss = 0.002025
grad_step = 000178, loss = 0.002022
grad_step = 000179, loss = 0.002017
grad_step = 000180, loss = 0.002010
grad_step = 000181, loss = 0.002004
grad_step = 000182, loss = 0.001998
grad_step = 000183, loss = 0.001993
grad_step = 000184, loss = 0.001989
grad_step = 000185, loss = 0.001986
grad_step = 000186, loss = 0.001983
grad_step = 000187, loss = 0.001981
grad_step = 000188, loss = 0.001979
grad_step = 000189, loss = 0.001979
grad_step = 000190, loss = 0.001982
grad_step = 000191, loss = 0.001993
grad_step = 000192, loss = 0.002024
grad_step = 000193, loss = 0.002083
grad_step = 000194, loss = 0.002187
grad_step = 000195, loss = 0.002213
grad_step = 000196, loss = 0.002125
grad_step = 000197, loss = 0.001978
grad_step = 000198, loss = 0.001980
grad_step = 000199, loss = 0.002081
grad_step = 000200, loss = 0.002074
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001969
grad_step = 000202, loss = 0.001956
grad_step = 000203, loss = 0.002027
grad_step = 000204, loss = 0.002015
grad_step = 000205, loss = 0.001943
grad_step = 000206, loss = 0.001950
grad_step = 000207, loss = 0.001993
grad_step = 000208, loss = 0.001968
grad_step = 000209, loss = 0.001924
grad_step = 000210, loss = 0.001940
grad_step = 000211, loss = 0.001963
grad_step = 000212, loss = 0.001936
grad_step = 000213, loss = 0.001911
grad_step = 000214, loss = 0.001927
grad_step = 000215, loss = 0.001936
grad_step = 000216, loss = 0.001912
grad_step = 000217, loss = 0.001898
grad_step = 000218, loss = 0.001909
grad_step = 000219, loss = 0.001912
grad_step = 000220, loss = 0.001895
grad_step = 000221, loss = 0.001883
grad_step = 000222, loss = 0.001887
grad_step = 000223, loss = 0.001890
grad_step = 000224, loss = 0.001882
grad_step = 000225, loss = 0.001870
grad_step = 000226, loss = 0.001864
grad_step = 000227, loss = 0.001866
grad_step = 000228, loss = 0.001866
grad_step = 000229, loss = 0.001858
grad_step = 000230, loss = 0.001847
grad_step = 000231, loss = 0.001842
grad_step = 000232, loss = 0.001841
grad_step = 000233, loss = 0.001841
grad_step = 000234, loss = 0.001837
grad_step = 000235, loss = 0.001831
grad_step = 000236, loss = 0.001823
grad_step = 000237, loss = 0.001815
grad_step = 000238, loss = 0.001809
grad_step = 000239, loss = 0.001804
grad_step = 000240, loss = 0.001799
grad_step = 000241, loss = 0.001795
grad_step = 000242, loss = 0.001797
grad_step = 000243, loss = 0.001812
grad_step = 000244, loss = 0.001871
grad_step = 000245, loss = 0.002029
grad_step = 000246, loss = 0.002067
grad_step = 000247, loss = 0.002027
grad_step = 000248, loss = 0.001816
grad_step = 000249, loss = 0.001849
grad_step = 000250, loss = 0.001946
grad_step = 000251, loss = 0.001836
grad_step = 000252, loss = 0.001852
grad_step = 000253, loss = 0.001876
grad_step = 000254, loss = 0.001764
grad_step = 000255, loss = 0.001826
grad_step = 000256, loss = 0.001875
grad_step = 000257, loss = 0.001750
grad_step = 000258, loss = 0.001774
grad_step = 000259, loss = 0.001847
grad_step = 000260, loss = 0.001756
grad_step = 000261, loss = 0.001741
grad_step = 000262, loss = 0.001801
grad_step = 000263, loss = 0.001754
grad_step = 000264, loss = 0.001728
grad_step = 000265, loss = 0.001762
grad_step = 000266, loss = 0.001741
grad_step = 000267, loss = 0.001718
grad_step = 000268, loss = 0.001732
grad_step = 000269, loss = 0.001729
grad_step = 000270, loss = 0.001710
grad_step = 000271, loss = 0.001708
grad_step = 000272, loss = 0.001716
grad_step = 000273, loss = 0.001713
grad_step = 000274, loss = 0.001696
grad_step = 000275, loss = 0.001695
grad_step = 000276, loss = 0.001702
grad_step = 000277, loss = 0.001697
grad_step = 000278, loss = 0.001690
grad_step = 000279, loss = 0.001684
grad_step = 000280, loss = 0.001684
grad_step = 000281, loss = 0.001688
grad_step = 000282, loss = 0.001685
grad_step = 000283, loss = 0.001680
grad_step = 000284, loss = 0.001674
grad_step = 000285, loss = 0.001670
grad_step = 000286, loss = 0.001672
grad_step = 000287, loss = 0.001674
grad_step = 000288, loss = 0.001674
grad_step = 000289, loss = 0.001668
grad_step = 000290, loss = 0.001662
grad_step = 000291, loss = 0.001658
grad_step = 000292, loss = 0.001658
grad_step = 000293, loss = 0.001659
grad_step = 000294, loss = 0.001660
grad_step = 000295, loss = 0.001659
grad_step = 000296, loss = 0.001657
grad_step = 000297, loss = 0.001656
grad_step = 000298, loss = 0.001652
grad_step = 000299, loss = 0.001649
grad_step = 000300, loss = 0.001645
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001642
grad_step = 000302, loss = 0.001640
grad_step = 000303, loss = 0.001639
grad_step = 000304, loss = 0.001639
grad_step = 000305, loss = 0.001638
grad_step = 000306, loss = 0.001637
grad_step = 000307, loss = 0.001637
grad_step = 000308, loss = 0.001639
grad_step = 000309, loss = 0.001640
grad_step = 000310, loss = 0.001646
grad_step = 000311, loss = 0.001652
grad_step = 000312, loss = 0.001673
grad_step = 000313, loss = 0.001681
grad_step = 000314, loss = 0.001709
grad_step = 000315, loss = 0.001673
grad_step = 000316, loss = 0.001644
grad_step = 000317, loss = 0.001624
grad_step = 000318, loss = 0.001630
grad_step = 000319, loss = 0.001644
grad_step = 000320, loss = 0.001633
grad_step = 000321, loss = 0.001621
grad_step = 000322, loss = 0.001618
grad_step = 000323, loss = 0.001624
grad_step = 000324, loss = 0.001626
grad_step = 000325, loss = 0.001620
grad_step = 000326, loss = 0.001613
grad_step = 000327, loss = 0.001609
grad_step = 000328, loss = 0.001615
grad_step = 000329, loss = 0.001624
grad_step = 000330, loss = 0.001639
grad_step = 000331, loss = 0.001650
grad_step = 000332, loss = 0.001678
grad_step = 000333, loss = 0.001690
grad_step = 000334, loss = 0.001728
grad_step = 000335, loss = 0.001658
grad_step = 000336, loss = 0.001608
grad_step = 000337, loss = 0.001584
grad_step = 000338, loss = 0.001606
grad_step = 000339, loss = 0.001636
grad_step = 000340, loss = 0.001609
grad_step = 000341, loss = 0.001588
grad_step = 000342, loss = 0.001595
grad_step = 000343, loss = 0.001603
grad_step = 000344, loss = 0.001594
grad_step = 000345, loss = 0.001577
grad_step = 000346, loss = 0.001575
grad_step = 000347, loss = 0.001585
grad_step = 000348, loss = 0.001587
grad_step = 000349, loss = 0.001578
grad_step = 000350, loss = 0.001567
grad_step = 000351, loss = 0.001564
grad_step = 000352, loss = 0.001570
grad_step = 000353, loss = 0.001573
grad_step = 000354, loss = 0.001567
grad_step = 000355, loss = 0.001557
grad_step = 000356, loss = 0.001550
grad_step = 000357, loss = 0.001550
grad_step = 000358, loss = 0.001554
grad_step = 000359, loss = 0.001555
grad_step = 000360, loss = 0.001553
grad_step = 000361, loss = 0.001546
grad_step = 000362, loss = 0.001540
grad_step = 000363, loss = 0.001537
grad_step = 000364, loss = 0.001537
grad_step = 000365, loss = 0.001539
grad_step = 000366, loss = 0.001539
grad_step = 000367, loss = 0.001536
grad_step = 000368, loss = 0.001533
grad_step = 000369, loss = 0.001529
grad_step = 000370, loss = 0.001528
grad_step = 000371, loss = 0.001529
grad_step = 000372, loss = 0.001533
grad_step = 000373, loss = 0.001539
grad_step = 000374, loss = 0.001554
grad_step = 000375, loss = 0.001576
grad_step = 000376, loss = 0.001636
grad_step = 000377, loss = 0.001687
grad_step = 000378, loss = 0.001810
grad_step = 000379, loss = 0.001694
grad_step = 000380, loss = 0.001601
grad_step = 000381, loss = 0.001532
grad_step = 000382, loss = 0.001560
grad_step = 000383, loss = 0.001604
grad_step = 000384, loss = 0.001561
grad_step = 000385, loss = 0.001547
grad_step = 000386, loss = 0.001574
grad_step = 000387, loss = 0.001554
grad_step = 000388, loss = 0.001513
grad_step = 000389, loss = 0.001504
grad_step = 000390, loss = 0.001536
grad_step = 000391, loss = 0.001554
grad_step = 000392, loss = 0.001519
grad_step = 000393, loss = 0.001487
grad_step = 000394, loss = 0.001491
grad_step = 000395, loss = 0.001511
grad_step = 000396, loss = 0.001518
grad_step = 000397, loss = 0.001496
grad_step = 000398, loss = 0.001476
grad_step = 000399, loss = 0.001478
grad_step = 000400, loss = 0.001489
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001493
grad_step = 000402, loss = 0.001480
grad_step = 000403, loss = 0.001467
grad_step = 000404, loss = 0.001465
grad_step = 000405, loss = 0.001470
grad_step = 000406, loss = 0.001473
grad_step = 000407, loss = 0.001467
grad_step = 000408, loss = 0.001458
grad_step = 000409, loss = 0.001454
grad_step = 000410, loss = 0.001455
grad_step = 000411, loss = 0.001457
grad_step = 000412, loss = 0.001456
grad_step = 000413, loss = 0.001451
grad_step = 000414, loss = 0.001446
grad_step = 000415, loss = 0.001443
grad_step = 000416, loss = 0.001442
grad_step = 000417, loss = 0.001443
grad_step = 000418, loss = 0.001442
grad_step = 000419, loss = 0.001439
grad_step = 000420, loss = 0.001435
grad_step = 000421, loss = 0.001432
grad_step = 000422, loss = 0.001431
grad_step = 000423, loss = 0.001433
grad_step = 000424, loss = 0.001437
grad_step = 000425, loss = 0.001449
grad_step = 000426, loss = 0.001465
grad_step = 000427, loss = 0.001514
grad_step = 000428, loss = 0.001547
grad_step = 000429, loss = 0.001647
grad_step = 000430, loss = 0.001619
grad_step = 000431, loss = 0.001632
grad_step = 000432, loss = 0.001547
grad_step = 000433, loss = 0.001484
grad_step = 000434, loss = 0.001449
grad_step = 000435, loss = 0.001447
grad_step = 000436, loss = 0.001478
grad_step = 000437, loss = 0.001487
grad_step = 000438, loss = 0.001474
grad_step = 000439, loss = 0.001438
grad_step = 000440, loss = 0.001417
grad_step = 000441, loss = 0.001438
grad_step = 000442, loss = 0.001455
grad_step = 000443, loss = 0.001450
grad_step = 000444, loss = 0.001413
grad_step = 000445, loss = 0.001388
grad_step = 000446, loss = 0.001392
grad_step = 000447, loss = 0.001411
grad_step = 000448, loss = 0.001419
grad_step = 000449, loss = 0.001405
grad_step = 000450, loss = 0.001395
grad_step = 000451, loss = 0.001400
grad_step = 000452, loss = 0.001421
grad_step = 000453, loss = 0.001429
grad_step = 000454, loss = 0.001433
grad_step = 000455, loss = 0.001414
grad_step = 000456, loss = 0.001408
grad_step = 000457, loss = 0.001394
grad_step = 000458, loss = 0.001385
grad_step = 000459, loss = 0.001373
grad_step = 000460, loss = 0.001365
grad_step = 000461, loss = 0.001365
grad_step = 000462, loss = 0.001367
grad_step = 000463, loss = 0.001369
grad_step = 000464, loss = 0.001365
grad_step = 000465, loss = 0.001356
grad_step = 000466, loss = 0.001347
grad_step = 000467, loss = 0.001343
grad_step = 000468, loss = 0.001344
grad_step = 000469, loss = 0.001348
grad_step = 000470, loss = 0.001352
grad_step = 000471, loss = 0.001350
grad_step = 000472, loss = 0.001347
grad_step = 000473, loss = 0.001340
grad_step = 000474, loss = 0.001337
grad_step = 000475, loss = 0.001334
grad_step = 000476, loss = 0.001337
grad_step = 000477, loss = 0.001339
grad_step = 000478, loss = 0.001347
grad_step = 000479, loss = 0.001348
grad_step = 000480, loss = 0.001351
grad_step = 000481, loss = 0.001343
grad_step = 000482, loss = 0.001341
grad_step = 000483, loss = 0.001325
grad_step = 000484, loss = 0.001317
grad_step = 000485, loss = 0.001311
grad_step = 000486, loss = 0.001312
grad_step = 000487, loss = 0.001320
grad_step = 000488, loss = 0.001329
grad_step = 000489, loss = 0.001328
grad_step = 000490, loss = 0.001315
grad_step = 000491, loss = 0.001302
grad_step = 000492, loss = 0.001289
grad_step = 000493, loss = 0.001279
grad_step = 000494, loss = 0.001279
grad_step = 000495, loss = 0.001286
grad_step = 000496, loss = 0.001289
grad_step = 000497, loss = 0.001293
grad_step = 000498, loss = 0.001302
grad_step = 000499, loss = 0.001307
grad_step = 000500, loss = 0.001306
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001296
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

  date_run                              2020-05-12 00:19:27.267688
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.251719
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 00:19:27.273655
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.156431
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 00:19:27.281328
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139841
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 00:19:27.286885
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.37702
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
0   2020-05-12 00:18:53.665691  ...    mean_absolute_error
1   2020-05-12 00:18:53.670002  ...     mean_squared_error
2   2020-05-12 00:18:53.673626  ...  median_absolute_error
3   2020-05-12 00:18:53.677484  ...               r2_score
4   2020-05-12 00:19:03.141975  ...    mean_absolute_error
5   2020-05-12 00:19:03.146809  ...     mean_squared_error
6   2020-05-12 00:19:03.150522  ...  median_absolute_error
7   2020-05-12 00:19:03.154479  ...               r2_score
8   2020-05-12 00:19:27.267688  ...    mean_absolute_error
9   2020-05-12 00:19:27.273655  ...     mean_squared_error
10  2020-05-12 00:19:27.281328  ...  median_absolute_error
11  2020-05-12 00:19:27.286885  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 133667.39it/s] 79%|███████▉  | 7864320/9912422 [00:00<00:10, 190814.12it/s]9920512it [00:00, 40936080.89it/s]                           
0it [00:00, ?it/s]32768it [00:00, 599717.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160342.61it/s]1654784it [00:00, 10822066.08it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202947.00it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd9bc76a20> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd393c2ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd9bc32eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd38e9a048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd393bf0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd4e629e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd393c2fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd4e63af28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd393bf0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd393bf080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbd9bc76a20> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8351de91d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2d481f1aea8bcd12a1adf30f18c6e19887e12d7c708292b7be0f710f0156883f
  Stored in directory: /tmp/pip-ephem-wheel-cache-j2k9wbkm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f82e99d10f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3391488/17464789 [====>.........................] - ETA: 0s
11558912/17464789 [==================>...........] - ETA: 0s
16523264/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 00:20:55.783719: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 00:20:55.787970: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 00:20:55.788097: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c6352bc080 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 00:20:55.788111: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.3566 - accuracy: 0.4550
 2000/25000 [=>............................] - ETA: 10s - loss: 8.2416 - accuracy: 0.4625
 3000/25000 [==>...........................] - ETA: 8s - loss: 8.0142 - accuracy: 0.4773 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.9426 - accuracy: 0.4820
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8936 - accuracy: 0.4852
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.8046 - accuracy: 0.4910
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7797 - accuracy: 0.4926
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7723 - accuracy: 0.4931
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7908 - accuracy: 0.4919
11000/25000 [============>.................] - ETA: 4s - loss: 7.7726 - accuracy: 0.4931
12000/25000 [=============>................] - ETA: 4s - loss: 7.7586 - accuracy: 0.4940
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7232 - accuracy: 0.4963
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7291 - accuracy: 0.4959
15000/25000 [=================>............] - ETA: 3s - loss: 7.7167 - accuracy: 0.4967
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7260 - accuracy: 0.4961
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7153 - accuracy: 0.4968
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6854 - accuracy: 0.4988
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6896 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7097 - accuracy: 0.4972
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 10s 391us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 00:21:13.250742
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 00:21:13.250742  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 00:21:19.894224: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 00:21:19.899528: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 00:21:19.900239: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564d38f78fa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 00:21:19.900255: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f95a73eb1d0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7875 - crf_viterbi_accuracy: 0.2533 - val_loss: 1.7482 - val_crf_viterbi_accuracy: 0.2133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f95ada59128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4596 - accuracy: 0.5135
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
11000/25000 [============>.................] - ETA: 4s - loss: 7.6192 - accuracy: 0.5031
12000/25000 [=============>................] - ETA: 4s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6289 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 3s - loss: 7.6676 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6847 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6876 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6861 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f953cfc4a90> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<46:16:17, 5.18kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<32:37:27, 7.34kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<22:53:22, 10.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<16:01:26, 14.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:11:04, 21.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.43M/862M [00:02<7:46:39, 30.5kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:02<5:25:24, 43.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:02<3:46:36, 62.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:02<2:37:58, 88.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:02<1:49:55, 127kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:02<1:16:36, 181kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.2M/862M [00:02<53:28, 257kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<37:19, 367kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.7M/862M [00:03<26:06, 522kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:03<18:16, 742kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:03<13:15, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<09:45, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:05<10:13:19, 21.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<7:09:09, 31.3kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:07<5:01:48, 44.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:07<3:34:28, 62.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<2:30:49, 88.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:07<1:45:24, 126kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:09<1:22:37, 161kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<59:15, 225kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<41:43, 318kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:11<32:01, 413kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:11<25:06, 527kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<18:14, 725kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<12:52, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:13<59:47, 220kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:13<43:12, 305kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<30:28, 431kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:15<24:20, 538kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:15<19:41, 665kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<14:19, 913kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<10:09, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:17<16:10, 805kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:17<12:40, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<09:11, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:19<09:27, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:19<09:22, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<07:07, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<05:08, 2.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<10:43, 1.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:21<08:50, 1.46MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<06:30, 1.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<07:31, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:23<07:52, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<06:09, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<06:22, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:25<05:47, 2.20MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<04:22, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:00, 2.11MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:47, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:23, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:48, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:21, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<04:04, 3.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:38, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:16, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:18, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<04:01, 3.09MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:43, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:08, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<04:13, 2.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:05, 4.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<15:47, 782kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:36, 907kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:04, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<07:10, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:38<14:49, 829kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:40, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:25, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<08:44, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:12, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:29, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<03:57, 3.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<48:31, 250kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<36:28, 333kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<26:08, 464kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<20:10, 599kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<15:20, 786kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<11:02, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<08:29, 1.41MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<8:03:31, 24.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<5:38:15, 35.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<3:58:07, 50.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<2:49:05, 70.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:58:50, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<1:24:45, 140kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<1:00:30, 196kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<42:34, 278kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<32:28, 363kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<25:08, 470kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<18:11, 648kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<12:49, 915kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<1:32:49, 126kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<1:05:56, 178kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<46:18, 253kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<32:28, 359kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<1:09:41, 167kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<49:55, 234kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<35:06, 331kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<27:16, 425kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<20:15, 572kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<14:26, 801kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<12:48, 900kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<10:08, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:22, 1.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:51, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<06:43, 1.70MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:59, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:10, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<05:28, 2.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<04:06, 2.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:33, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:02, 2.25MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<03:48, 2.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<04:41, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<03:40, 3.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<02:41, 4.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<44:10, 253kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<32:02, 349kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<22:39, 492kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<18:25, 603kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<15:09, 733kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:05, 1.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:53, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<10:26, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<08:26, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:10, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<06:52, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:57, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:26, 2.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:39, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:04, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:49, 2.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:19<05:14, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:52, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:39, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<04:59, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<04:35, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<03:28, 3.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<04:56, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<05:44, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:34, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:45, 2.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<6:22:01, 27.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<4:27:16, 39.8kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<3:08:12, 56.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<2:13:52, 79.0kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<1:34:02, 112kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<1:05:42, 160kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<53:05, 198kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<38:13, 275kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<26:55, 389kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<21:13, 492kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<17:00, 614kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<12:20, 845kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<08:44, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<12:04, 860kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<09:29, 1.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:53, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<07:14, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<06:05, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:28, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<05:34, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<03:40, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<02:42, 3.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<45:12, 225kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<32:39, 311kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<23:03, 440kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<18:28, 547kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<13:57, 723kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<09:57, 1.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<09:20, 1.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<08:39, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<06:28, 1.55MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:42, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:25, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:21, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<03:59, 2.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:08, 1.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:36, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:47<04:25, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:40, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:18, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<03:15, 3.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:32, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:10, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:26, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<03:58, 2.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<02:58, 3.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<02:13, 4.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<19:39, 489kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<15:44, 611kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<11:29, 835kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<08:06, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<9:17:53, 17.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<6:31:13, 24.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<4:33:19, 34.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<3:12:50, 49.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<2:16:53, 69.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<1:36:11, 98.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<1:07:05, 140kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<1:54:05, 82.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<1:20:46, 116kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<56:35, 166kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<41:39, 224kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<30:06, 310kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<21:15, 438kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<15:26, 601kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<6:21:08, 24.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<4:27:00, 34.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<3:06:06, 49.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<2:17:21, 67.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<1:38:04, 94.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<1:08:58, 133kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<48:10, 190kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<39:41, 230kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<28:43, 318kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<20:15, 450kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<16:14, 559kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<12:17, 739kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<08:46, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<08:16, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<06:42, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:54, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<05:32, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<05:41, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:25, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:31, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:05, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:02, 2.91MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:12, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<03:42, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<02:41, 3.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:51, 987kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<07:04, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:09, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:38, 1.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:34, 2.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:32, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:55, 1.75MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<03:52, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:05, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:44, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<02:47, 3.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<03:57, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:37, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<02:44, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:27, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:29, 2.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:31, 3.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<10:23, 801kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:08, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:52, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:01, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:54, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:29, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:14, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:16, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:13, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:51, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<04:35, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<04:53, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:49, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<03:58, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<03:36, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<02:43, 2.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:33, 3.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<4:49:23, 27.6kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<3:22:20, 39.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<2:22:15, 55.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<1:41:36, 78.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<1:11:29, 111kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<49:57, 158kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<37:26, 210kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<27:06, 290kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<19:07, 409kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<14:56, 521kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<12:06, 643kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<08:48, 882kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:16, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<07:01, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:41, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:08, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<04:42, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<04:52, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<03:44, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:42, 2.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<06:38, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:25, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:59, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:31, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:56, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:56, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:48, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:25, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<02:32, 2.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:55, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:06, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:15, 3.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:41, 950kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<06:08, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:28, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:47, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:58, 1.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:57, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:08, 3.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<28:35, 251kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<21:28, 334kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<15:18, 467kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<10:47, 660kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<10:10, 697kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:51, 903kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:39, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<05:34, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:37, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:22, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<04:00, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:30, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:37, 2.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:27, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:07, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:21, 2.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:14, 3.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:09, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:11, 3.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:07, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<02:50, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:09, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:42, 2.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:03, 3.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:59, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<02:45, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:05, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<01:53, 3.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<4:18:51, 25.1kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<3:00:50, 35.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<2:06:51, 50.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<1:30:07, 71.2kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<1:03:18, 101kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<44:02, 144kB/s]  .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<46:04, 138kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<32:53, 193kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<23:03, 274kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<17:31, 358kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<13:32, 464kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<09:47, 641kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:51, 906kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<6:03:00, 17.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<4:14:24, 24.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<2:57:31, 34.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<2:04:56, 49.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<1:28:42, 69.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<1:02:16, 98.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<43:19, 140kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<44:50, 136kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<31:59, 190kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<22:27, 269kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<17:01, 353kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<12:31, 479kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<08:52, 673kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<07:34, 784kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<05:54, 1.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:14, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:20, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<04:14, 1.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:13, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:19, 2.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:51, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:18, 1.75MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:25, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:02, 1.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:17, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<02:35, 2.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:43, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:28, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<01:52, 3.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:58, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:22, 2.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<01:42, 3.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<46:16, 120kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<32:55, 168kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<23:05, 238kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<17:19, 315kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<13:17, 411kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<09:34, 569kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:43, 805kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<07:53, 683kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<06:04, 886kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:22, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:16, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:07, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:09, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:17, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:32, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:01, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:13, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<01:56, 2.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<3:14:52, 26.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<2:16:02, 38.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<1:35:18, 53.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<1:07:52, 75.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<47:42, 107kB/s]   .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<33:09, 153kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<26:56, 188kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<19:25, 260kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<13:39, 368kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<10:33, 473kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<07:48, 639kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<05:32, 894kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:55, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<14:39, 336kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<11:15, 437kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<08:06, 605kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<06:24, 758kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<04:58, 975kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:33, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:36, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:29, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:38, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:54, 2.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<03:39, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:02, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:13, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:37, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:47, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:10, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<02:40, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<02:56, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:04, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<33:23, 135kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<23:47, 189kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<16:39, 269kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<12:37, 352kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<09:17, 478kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<06:34, 671kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:35, 781kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<04:50, 903kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:35, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:32, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<4:10:27, 17.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<2:55:30, 24.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<2:02:13, 35.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<1:25:49, 49.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<1:00:54, 69.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<42:42, 98.8kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<29:39, 141kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<23:36, 176kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<16:56, 246kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<11:52, 348kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<09:12, 445kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:32<07:17, 562kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<05:16, 775kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:42, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:32, 889kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:31, 1.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:37, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:52, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<20:56, 189kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<15:02, 263kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<10:33, 372kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<08:14, 472kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<06:35, 590kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:48, 807kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:56, 972kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:16, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:50, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<2:27:33, 25.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<1:42:50, 36.4kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<1:11:45, 51.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<51:03, 72.3kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<35:51, 103kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<24:50, 146kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<19:58, 182kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<14:20, 252kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<10:04, 357kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<07:45, 458kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<06:09, 577kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<04:28, 791kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<03:39, 955kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:54, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:06, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<02:15, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:52<01:55, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:25, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:46, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:34, 2.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:10, 2.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<01:35, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:26, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:11, 2.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:20, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:00, 3.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:26, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:38, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:16, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<00:55, 3.33MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:16, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:23, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:40, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:46, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<00:58, 3.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:45, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:13, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:37, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:47, 1.60MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:51, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:25, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:01, 2.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:57, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:39, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:13, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:29, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:35, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:13, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<00:52, 3.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:07, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:45, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:16, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:29, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:34, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:13, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:52, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<2:26:15, 17.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<1:42:23, 24.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<1:10:59, 35.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<49:33, 49.6kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<34:50, 70.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<24:11, 100kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<17:15, 139kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<12:33, 190kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<08:51, 268kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<06:12, 376kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<1:30:58, 25.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<1:03:14, 36.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<43:20, 52.2kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<1:42:43, 22.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<1:12:20, 31.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<50:29, 44.5kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<34:37, 63.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<26:49, 81.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<18:57, 115kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<13:09, 164kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<09:33, 222kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<07:06, 298kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<05:03, 417kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<03:47, 543kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<02:51, 717kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:01, 998kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:51, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:42, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:39, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:21, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:10, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:53, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:38, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:00, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:53, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:39, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:50, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:56, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:43, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:31, 3.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<1:35:21, 17.3kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<1:06:37, 24.6kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<46:07, 35.1kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<31:29, 50.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<28:22, 55.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<20:08, 78.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<14:03, 111kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<09:45, 155kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<06:57, 216kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<04:49, 306kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<03:18, 435kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<23:44, 60.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<16:52, 85.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<11:46, 121kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<08:05, 172kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<06:03, 226kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<04:21, 312kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<03:01, 442kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:22, 550kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:45, 735kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:14, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:08, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:54, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:39, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:43, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:37, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:27, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:33, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:37, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:28, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:21, 3.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:19, 3.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:17, 3.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<38:13, 25.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<26:08, 36.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<17:36, 51.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<13:05, 68.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<08:35, 96.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<05:57, 137kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<03:56, 196kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<03:29, 218kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<02:36, 292kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:49, 408kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<01:12, 578kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<05:46, 120kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<04:04, 168kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<02:45, 239kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:58, 317kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:30, 414kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:03, 577kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:42, 814kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:43, 766kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:33, 985kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:21, 1.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:21, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:15, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:17, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:14, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:12, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:08, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:05, 2.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 584kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 767kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 659/400000 [00:00<01:00, 6585.83it/s]  0%|          | 1428/400000 [00:00<00:57, 6881.26it/s]  1%|          | 2192/400000 [00:00<00:56, 7091.29it/s]  1%|          | 2950/400000 [00:00<00:54, 7229.22it/s]  1%|          | 3712/400000 [00:00<00:53, 7340.26it/s]  1%|          | 4486/400000 [00:00<00:53, 7453.26it/s]  1%|▏         | 5215/400000 [00:00<00:53, 7402.27it/s]  1%|▏         | 5960/400000 [00:00<00:53, 7415.47it/s]  2%|▏         | 6665/400000 [00:00<00:54, 7236.27it/s]  2%|▏         | 7401/400000 [00:01<00:53, 7272.09it/s]  2%|▏         | 8150/400000 [00:01<00:53, 7334.72it/s]  2%|▏         | 8910/400000 [00:01<00:52, 7412.24it/s]  2%|▏         | 9671/400000 [00:01<00:52, 7468.69it/s]  3%|▎         | 10418/400000 [00:01<00:52, 7465.49it/s]  3%|▎         | 11161/400000 [00:01<00:52, 7442.23it/s]  3%|▎         | 11903/400000 [00:01<00:52, 7422.85it/s]  3%|▎         | 12659/400000 [00:01<00:51, 7461.34it/s]  3%|▎         | 13404/400000 [00:01<00:51, 7456.95it/s]  4%|▎         | 14176/400000 [00:01<00:51, 7533.18it/s]  4%|▎         | 14929/400000 [00:02<00:51, 7488.47it/s]  4%|▍         | 15688/400000 [00:02<00:51, 7517.31it/s]  4%|▍         | 16452/400000 [00:02<00:50, 7552.02it/s]  4%|▍         | 17208/400000 [00:02<00:51, 7462.91it/s]  4%|▍         | 17955/400000 [00:02<00:51, 7421.37it/s]  5%|▍         | 18698/400000 [00:02<00:51, 7418.94it/s]  5%|▍         | 19453/400000 [00:02<00:51, 7456.48it/s]  5%|▌         | 20210/400000 [00:02<00:50, 7489.74it/s]  5%|▌         | 20960/400000 [00:02<00:50, 7486.53it/s]  5%|▌         | 21718/400000 [00:02<00:50, 7510.89it/s]  6%|▌         | 22470/400000 [00:03<00:50, 7447.39it/s]  6%|▌         | 23252/400000 [00:03<00:49, 7553.87it/s]  6%|▌         | 24008/400000 [00:03<00:49, 7544.01it/s]  6%|▌         | 24780/400000 [00:03<00:49, 7595.29it/s]  6%|▋         | 25540/400000 [00:03<00:49, 7581.05it/s]  7%|▋         | 26299/400000 [00:03<00:49, 7560.63it/s]  7%|▋         | 27085/400000 [00:03<00:48, 7647.36it/s]  7%|▋         | 27851/400000 [00:03<00:48, 7648.72it/s]  7%|▋         | 28631/400000 [00:03<00:48, 7693.20it/s]  7%|▋         | 29401/400000 [00:03<00:48, 7681.99it/s]  8%|▊         | 30170/400000 [00:04<00:48, 7607.55it/s]  8%|▊         | 30932/400000 [00:04<00:48, 7536.55it/s]  8%|▊         | 31687/400000 [00:04<00:48, 7536.66it/s]  8%|▊         | 32450/400000 [00:04<00:48, 7563.72it/s]  8%|▊         | 33215/400000 [00:04<00:48, 7587.41it/s]  8%|▊         | 33974/400000 [00:04<00:48, 7486.51it/s]  9%|▊         | 34737/400000 [00:04<00:48, 7528.39it/s]  9%|▉         | 35491/400000 [00:04<00:50, 7285.80it/s]  9%|▉         | 36237/400000 [00:04<00:49, 7335.50it/s]  9%|▉         | 36972/400000 [00:04<00:49, 7310.65it/s]  9%|▉         | 37705/400000 [00:05<00:49, 7269.93it/s] 10%|▉         | 38463/400000 [00:05<00:49, 7358.00it/s] 10%|▉         | 39220/400000 [00:05<00:48, 7418.44it/s] 10%|▉         | 39976/400000 [00:05<00:48, 7459.95it/s] 10%|█         | 40733/400000 [00:05<00:47, 7492.47it/s] 10%|█         | 41483/400000 [00:05<00:48, 7443.77it/s] 11%|█         | 42228/400000 [00:05<00:48, 7433.54it/s] 11%|█         | 42972/400000 [00:05<00:48, 7340.89it/s] 11%|█         | 43722/400000 [00:05<00:48, 7384.02it/s] 11%|█         | 44492/400000 [00:05<00:47, 7473.54it/s] 11%|█▏        | 45241/400000 [00:06<00:47, 7477.53it/s] 11%|█▏        | 45990/400000 [00:06<00:48, 7371.89it/s] 12%|█▏        | 46771/400000 [00:06<00:47, 7497.44it/s] 12%|█▏        | 47555/400000 [00:06<00:46, 7596.91it/s] 12%|█▏        | 48320/400000 [00:06<00:46, 7611.87it/s] 12%|█▏        | 49087/400000 [00:06<00:45, 7629.20it/s] 12%|█▏        | 49871/400000 [00:06<00:45, 7690.77it/s] 13%|█▎        | 50641/400000 [00:06<00:45, 7656.26it/s] 13%|█▎        | 51421/400000 [00:06<00:45, 7697.52it/s] 13%|█▎        | 52203/400000 [00:06<00:44, 7733.27it/s] 13%|█▎        | 52977/400000 [00:07<00:45, 7690.15it/s] 13%|█▎        | 53762/400000 [00:07<00:44, 7737.41it/s] 14%|█▎        | 54537/400000 [00:07<00:44, 7738.98it/s] 14%|█▍        | 55312/400000 [00:07<00:44, 7700.40it/s] 14%|█▍        | 56083/400000 [00:07<00:45, 7635.20it/s] 14%|█▍        | 56847/400000 [00:07<00:45, 7621.26it/s] 14%|█▍        | 57610/400000 [00:07<00:45, 7596.47it/s] 15%|█▍        | 58388/400000 [00:07<00:44, 7650.09it/s] 15%|█▍        | 59168/400000 [00:07<00:44, 7694.10it/s] 15%|█▍        | 59949/400000 [00:07<00:44, 7725.03it/s] 15%|█▌        | 60722/400000 [00:08<00:44, 7703.86it/s] 15%|█▌        | 61501/400000 [00:08<00:43, 7727.76it/s] 16%|█▌        | 62274/400000 [00:08<00:43, 7689.91it/s] 16%|█▌        | 63057/400000 [00:08<00:43, 7728.72it/s] 16%|█▌        | 63831/400000 [00:08<00:43, 7683.89it/s] 16%|█▌        | 64600/400000 [00:08<00:45, 7377.65it/s] 16%|█▋        | 65341/400000 [00:08<00:45, 7314.10it/s] 17%|█▋        | 66075/400000 [00:08<00:46, 7215.46it/s] 17%|█▋        | 66799/400000 [00:08<00:46, 7210.70it/s] 17%|█▋        | 67522/400000 [00:09<00:47, 7073.37it/s] 17%|█▋        | 68231/400000 [00:09<00:47, 7031.28it/s] 17%|█▋        | 69016/400000 [00:09<00:45, 7257.92it/s] 17%|█▋        | 69782/400000 [00:09<00:44, 7370.62it/s] 18%|█▊        | 70522/400000 [00:09<00:45, 7306.44it/s] 18%|█▊        | 71281/400000 [00:09<00:44, 7386.39it/s] 18%|█▊        | 72037/400000 [00:09<00:44, 7434.77it/s] 18%|█▊        | 72803/400000 [00:09<00:43, 7500.76it/s] 18%|█▊        | 73559/400000 [00:09<00:43, 7517.14it/s] 19%|█▊        | 74339/400000 [00:09<00:42, 7598.98it/s] 19%|█▉        | 75100/400000 [00:10<00:43, 7486.25it/s] 19%|█▉        | 75850/400000 [00:10<00:43, 7421.66it/s] 19%|█▉        | 76610/400000 [00:10<00:43, 7473.56it/s] 19%|█▉        | 77369/400000 [00:10<00:42, 7507.59it/s] 20%|█▉        | 78133/400000 [00:10<00:42, 7545.25it/s] 20%|█▉        | 78915/400000 [00:10<00:42, 7623.67it/s] 20%|█▉        | 79678/400000 [00:10<00:42, 7613.30it/s] 20%|██        | 80440/400000 [00:10<00:42, 7571.61it/s] 20%|██        | 81202/400000 [00:10<00:42, 7584.51it/s] 20%|██        | 81968/400000 [00:10<00:41, 7604.57it/s] 21%|██        | 82729/400000 [00:11<00:41, 7597.14it/s] 21%|██        | 83489/400000 [00:11<00:41, 7573.04it/s] 21%|██        | 84247/400000 [00:11<00:42, 7467.91it/s] 21%|██        | 84998/400000 [00:11<00:42, 7477.57it/s] 21%|██▏       | 85778/400000 [00:11<00:41, 7569.50it/s] 22%|██▏       | 86557/400000 [00:11<00:41, 7631.68it/s] 22%|██▏       | 87329/400000 [00:11<00:40, 7656.03it/s] 22%|██▏       | 88095/400000 [00:11<00:40, 7619.45it/s] 22%|██▏       | 88867/400000 [00:11<00:40, 7647.41it/s] 22%|██▏       | 89632/400000 [00:11<00:40, 7607.47it/s] 23%|██▎       | 90396/400000 [00:12<00:40, 7615.80it/s] 23%|██▎       | 91158/400000 [00:12<00:40, 7587.84it/s] 23%|██▎       | 91917/400000 [00:12<00:40, 7569.54it/s] 23%|██▎       | 92678/400000 [00:12<00:40, 7579.81it/s] 23%|██▎       | 93437/400000 [00:12<00:41, 7448.33it/s] 24%|██▎       | 94186/400000 [00:12<00:41, 7458.57it/s] 24%|██▎       | 94933/400000 [00:12<00:41, 7400.17it/s] 24%|██▍       | 95674/400000 [00:12<00:41, 7379.08it/s] 24%|██▍       | 96425/400000 [00:12<00:40, 7417.30it/s] 24%|██▍       | 97187/400000 [00:12<00:40, 7475.34it/s] 24%|██▍       | 97950/400000 [00:13<00:40, 7518.32it/s] 25%|██▍       | 98703/400000 [00:13<00:40, 7483.53it/s] 25%|██▍       | 99453/400000 [00:13<00:40, 7486.40it/s] 25%|██▌       | 100202/400000 [00:13<00:40, 7405.87it/s] 25%|██▌       | 100943/400000 [00:13<00:40, 7397.65it/s] 25%|██▌       | 101715/400000 [00:13<00:39, 7490.11it/s] 26%|██▌       | 102465/400000 [00:13<00:39, 7485.73it/s] 26%|██▌       | 103214/400000 [00:13<00:39, 7474.54it/s] 26%|██▌       | 103976/400000 [00:13<00:39, 7517.57it/s] 26%|██▌       | 104749/400000 [00:13<00:38, 7579.05it/s] 26%|██▋       | 105508/400000 [00:14<00:39, 7520.53it/s] 27%|██▋       | 106261/400000 [00:14<00:39, 7503.19it/s] 27%|██▋       | 107012/400000 [00:14<00:39, 7488.75it/s] 27%|██▋       | 107776/400000 [00:14<00:38, 7531.67it/s] 27%|██▋       | 108564/400000 [00:14<00:38, 7631.00it/s] 27%|██▋       | 109328/400000 [00:14<00:38, 7628.08it/s] 28%|██▊       | 110098/400000 [00:14<00:37, 7647.86it/s] 28%|██▊       | 110864/400000 [00:14<00:38, 7591.93it/s] 28%|██▊       | 111625/400000 [00:14<00:37, 7595.55it/s] 28%|██▊       | 112408/400000 [00:14<00:37, 7663.90it/s] 28%|██▊       | 113175/400000 [00:15<00:37, 7631.93it/s] 28%|██▊       | 113939/400000 [00:15<00:37, 7616.45it/s] 29%|██▊       | 114701/400000 [00:15<00:37, 7587.58it/s] 29%|██▉       | 115460/400000 [00:15<00:37, 7541.89it/s] 29%|██▉       | 116215/400000 [00:15<00:37, 7528.09it/s] 29%|██▉       | 116969/400000 [00:15<00:37, 7529.11it/s] 29%|██▉       | 117722/400000 [00:15<00:38, 7356.19it/s] 30%|██▉       | 118463/400000 [00:15<00:38, 7370.24it/s] 30%|██▉       | 119232/400000 [00:15<00:37, 7463.20it/s] 30%|██▉       | 119980/400000 [00:15<00:37, 7387.76it/s] 30%|███       | 120756/400000 [00:16<00:37, 7493.19it/s] 30%|███       | 121507/400000 [00:16<00:37, 7496.47it/s] 31%|███       | 122258/400000 [00:16<00:37, 7472.21it/s] 31%|███       | 123026/400000 [00:16<00:36, 7531.80it/s] 31%|███       | 123780/400000 [00:16<00:36, 7492.38it/s] 31%|███       | 124530/400000 [00:16<00:36, 7446.39it/s] 31%|███▏      | 125306/400000 [00:16<00:36, 7535.37it/s] 32%|███▏      | 126061/400000 [00:16<00:36, 7408.10it/s] 32%|███▏      | 126803/400000 [00:16<00:36, 7394.36it/s] 32%|███▏      | 127544/400000 [00:16<00:36, 7372.93it/s] 32%|███▏      | 128315/400000 [00:17<00:36, 7469.02it/s] 32%|███▏      | 129078/400000 [00:17<00:36, 7513.68it/s] 32%|███▏      | 129830/400000 [00:17<00:36, 7394.67it/s] 33%|███▎      | 130571/400000 [00:17<00:36, 7291.54it/s] 33%|███▎      | 131302/400000 [00:17<00:36, 7265.43it/s] 33%|███▎      | 132041/400000 [00:17<00:36, 7300.52it/s] 33%|███▎      | 132772/400000 [00:17<00:36, 7272.11it/s] 33%|███▎      | 133503/400000 [00:17<00:36, 7282.88it/s] 34%|███▎      | 134232/400000 [00:17<00:37, 7169.15it/s] 34%|███▎      | 134950/400000 [00:18<00:38, 6885.21it/s] 34%|███▍      | 135692/400000 [00:18<00:37, 7034.66it/s] 34%|███▍      | 136428/400000 [00:18<00:36, 7127.20it/s] 34%|███▍      | 137179/400000 [00:18<00:36, 7236.98it/s] 34%|███▍      | 137912/400000 [00:18<00:36, 7263.03it/s] 35%|███▍      | 138671/400000 [00:18<00:35, 7356.16it/s] 35%|███▍      | 139430/400000 [00:18<00:35, 7423.16it/s] 35%|███▌      | 140174/400000 [00:18<00:34, 7426.51it/s] 35%|███▌      | 140921/400000 [00:18<00:34, 7439.43it/s] 35%|███▌      | 141671/400000 [00:18<00:34, 7456.10it/s] 36%|███▌      | 142417/400000 [00:19<00:35, 7325.21it/s] 36%|███▌      | 143162/400000 [00:19<00:34, 7359.82it/s] 36%|███▌      | 143903/400000 [00:19<00:34, 7373.15it/s] 36%|███▌      | 144641/400000 [00:19<00:34, 7354.19it/s] 36%|███▋      | 145377/400000 [00:19<00:35, 7217.62it/s] 37%|███▋      | 146116/400000 [00:19<00:34, 7268.07it/s] 37%|███▋      | 146864/400000 [00:19<00:34, 7330.32it/s] 37%|███▋      | 147598/400000 [00:19<00:34, 7329.46it/s] 37%|███▋      | 148349/400000 [00:19<00:34, 7380.90it/s] 37%|███▋      | 149099/400000 [00:19<00:33, 7414.82it/s] 37%|███▋      | 149841/400000 [00:20<00:33, 7373.86it/s] 38%|███▊      | 150610/400000 [00:20<00:33, 7465.03it/s] 38%|███▊      | 151357/400000 [00:20<00:33, 7451.78it/s] 38%|███▊      | 152103/400000 [00:20<00:33, 7441.79it/s] 38%|███▊      | 152860/400000 [00:20<00:33, 7479.14it/s] 38%|███▊      | 153609/400000 [00:20<00:32, 7477.77it/s] 39%|███▊      | 154357/400000 [00:20<00:32, 7451.41it/s] 39%|███▉      | 155103/400000 [00:20<00:33, 7275.57it/s] 39%|███▉      | 155832/400000 [00:20<00:33, 7267.81it/s] 39%|███▉      | 156582/400000 [00:20<00:33, 7333.24it/s] 39%|███▉      | 157323/400000 [00:21<00:33, 7353.79it/s] 40%|███▉      | 158102/400000 [00:21<00:32, 7478.39it/s] 40%|███▉      | 158888/400000 [00:21<00:31, 7588.43it/s] 40%|███▉      | 159648/400000 [00:21<00:31, 7547.67it/s] 40%|████      | 160430/400000 [00:21<00:31, 7624.80it/s] 40%|████      | 161194/400000 [00:21<00:31, 7615.12it/s] 40%|████      | 161974/400000 [00:21<00:31, 7666.97it/s] 41%|████      | 162749/400000 [00:21<00:30, 7688.93it/s] 41%|████      | 163519/400000 [00:21<00:30, 7665.02it/s] 41%|████      | 164293/400000 [00:21<00:30, 7685.85it/s] 41%|████▏     | 165062/400000 [00:22<00:30, 7635.54it/s] 41%|████▏     | 165826/400000 [00:22<00:30, 7629.78it/s] 42%|████▏     | 166600/400000 [00:22<00:30, 7662.04it/s] 42%|████▏     | 167367/400000 [00:22<00:30, 7617.17it/s] 42%|████▏     | 168129/400000 [00:22<00:30, 7565.57it/s] 42%|████▏     | 168886/400000 [00:22<00:30, 7498.05it/s] 42%|████▏     | 169669/400000 [00:22<00:30, 7592.21it/s] 43%|████▎     | 170452/400000 [00:22<00:29, 7660.01it/s] 43%|████▎     | 171225/400000 [00:22<00:29, 7678.07it/s] 43%|████▎     | 172010/400000 [00:22<00:29, 7728.10it/s] 43%|████▎     | 172784/400000 [00:23<00:30, 7339.41it/s] 43%|████▎     | 173538/400000 [00:23<00:30, 7396.10it/s] 44%|████▎     | 174281/400000 [00:23<00:30, 7397.23it/s] 44%|████▍     | 175023/400000 [00:23<00:30, 7351.49it/s] 44%|████▍     | 175785/400000 [00:23<00:30, 7429.72it/s] 44%|████▍     | 176554/400000 [00:23<00:29, 7505.34it/s] 44%|████▍     | 177306/400000 [00:23<00:29, 7498.53it/s] 45%|████▍     | 178085/400000 [00:23<00:29, 7582.17it/s] 45%|████▍     | 178844/400000 [00:23<00:29, 7544.71it/s] 45%|████▍     | 179619/400000 [00:23<00:28, 7600.65it/s] 45%|████▌     | 180380/400000 [00:24<00:28, 7600.18it/s] 45%|████▌     | 181158/400000 [00:24<00:28, 7652.25it/s] 45%|████▌     | 181940/400000 [00:24<00:28, 7701.37it/s] 46%|████▌     | 182711/400000 [00:24<00:28, 7644.53it/s] 46%|████▌     | 183476/400000 [00:24<00:28, 7629.49it/s] 46%|████▌     | 184244/400000 [00:24<00:28, 7644.01it/s] 46%|████▋     | 185015/400000 [00:24<00:28, 7663.09it/s] 46%|████▋     | 185793/400000 [00:24<00:27, 7696.68it/s] 47%|████▋     | 186563/400000 [00:24<00:27, 7688.85it/s] 47%|████▋     | 187332/400000 [00:24<00:27, 7661.80it/s] 47%|████▋     | 188104/400000 [00:25<00:27, 7678.21it/s] 47%|████▋     | 188888/400000 [00:25<00:27, 7725.00it/s] 47%|████▋     | 189669/400000 [00:25<00:27, 7749.82it/s] 48%|████▊     | 190445/400000 [00:25<00:27, 7731.05it/s] 48%|████▊     | 191219/400000 [00:25<00:27, 7678.02it/s] 48%|████▊     | 191987/400000 [00:25<00:27, 7663.75it/s] 48%|████▊     | 192754/400000 [00:25<00:27, 7478.63it/s] 48%|████▊     | 193503/400000 [00:25<00:27, 7392.85it/s] 49%|████▊     | 194262/400000 [00:25<00:27, 7450.91it/s] 49%|████▉     | 195035/400000 [00:26<00:27, 7530.21it/s] 49%|████▉     | 195811/400000 [00:26<00:26, 7596.66it/s] 49%|████▉     | 196581/400000 [00:26<00:26, 7625.23it/s] 49%|████▉     | 197357/400000 [00:26<00:26, 7663.91it/s] 50%|████▉     | 198124/400000 [00:26<00:26, 7654.85it/s] 50%|████▉     | 198890/400000 [00:26<00:26, 7574.71it/s] 50%|████▉     | 199677/400000 [00:26<00:26, 7660.16it/s] 50%|█████     | 200467/400000 [00:26<00:25, 7728.06it/s] 50%|█████     | 201262/400000 [00:26<00:25, 7791.97it/s] 51%|█████     | 202042/400000 [00:26<00:25, 7732.37it/s] 51%|█████     | 202816/400000 [00:27<00:25, 7717.60it/s] 51%|█████     | 203601/400000 [00:27<00:25, 7754.47it/s] 51%|█████     | 204377/400000 [00:27<00:25, 7643.33it/s] 51%|█████▏    | 205142/400000 [00:27<00:25, 7572.12it/s] 51%|█████▏    | 205900/400000 [00:27<00:26, 7449.00it/s] 52%|█████▏    | 206678/400000 [00:27<00:25, 7545.08it/s] 52%|█████▏    | 207434/400000 [00:27<00:25, 7538.24it/s] 52%|█████▏    | 208189/400000 [00:27<00:25, 7540.95it/s] 52%|█████▏    | 208947/400000 [00:27<00:25, 7551.10it/s] 52%|█████▏    | 209703/400000 [00:27<00:25, 7496.18it/s] 53%|█████▎    | 210456/400000 [00:28<00:25, 7505.03it/s] 53%|█████▎    | 211218/400000 [00:28<00:25, 7538.56it/s] 53%|█████▎    | 211973/400000 [00:28<00:24, 7541.19it/s] 53%|█████▎    | 212728/400000 [00:28<00:24, 7506.28it/s] 53%|█████▎    | 213479/400000 [00:28<00:25, 7380.46it/s] 54%|█████▎    | 214218/400000 [00:28<00:25, 7337.88it/s] 54%|█████▎    | 214969/400000 [00:28<00:25, 7388.61it/s] 54%|█████▍    | 215720/400000 [00:28<00:24, 7424.29it/s] 54%|█████▍    | 216463/400000 [00:28<00:24, 7401.52it/s] 54%|█████▍    | 217218/400000 [00:28<00:24, 7445.02it/s] 54%|█████▍    | 217963/400000 [00:29<00:24, 7390.91it/s] 55%|█████▍    | 218719/400000 [00:29<00:24, 7440.44it/s] 55%|█████▍    | 219479/400000 [00:29<00:24, 7487.29it/s] 55%|█████▌    | 220228/400000 [00:29<00:24, 7367.49it/s] 55%|█████▌    | 220966/400000 [00:29<00:24, 7283.09it/s] 55%|█████▌    | 221719/400000 [00:29<00:24, 7355.07it/s] 56%|█████▌    | 222464/400000 [00:29<00:24, 7382.26it/s] 56%|█████▌    | 223242/400000 [00:29<00:23, 7495.51it/s] 56%|█████▌    | 224006/400000 [00:29<00:23, 7537.43it/s] 56%|█████▌    | 224762/400000 [00:29<00:23, 7541.18it/s] 56%|█████▋    | 225517/400000 [00:30<00:23, 7516.30it/s] 57%|█████▋    | 226269/400000 [00:30<00:23, 7506.10it/s] 57%|█████▋    | 227040/400000 [00:30<00:22, 7562.43it/s] 57%|█████▋    | 227810/400000 [00:30<00:22, 7601.32it/s] 57%|█████▋    | 228571/400000 [00:30<00:22, 7576.27it/s] 57%|█████▋    | 229329/400000 [00:30<00:22, 7492.86it/s] 58%|█████▊    | 230079/400000 [00:30<00:23, 7355.71it/s] 58%|█████▊    | 230816/400000 [00:30<00:23, 7319.06it/s] 58%|█████▊    | 231552/400000 [00:30<00:22, 7328.81it/s] 58%|█████▊    | 232320/400000 [00:30<00:22, 7430.50it/s] 58%|█████▊    | 233064/400000 [00:31<00:22, 7258.82it/s] 58%|█████▊    | 233801/400000 [00:31<00:22, 7289.16it/s] 59%|█████▊    | 234560/400000 [00:31<00:22, 7375.16it/s] 59%|█████▉    | 235312/400000 [00:31<00:22, 7414.75it/s] 59%|█████▉    | 236057/400000 [00:31<00:22, 7425.26it/s] 59%|█████▉    | 236801/400000 [00:31<00:22, 7265.12it/s] 59%|█████▉    | 237545/400000 [00:31<00:22, 7314.70it/s] 60%|█████▉    | 238296/400000 [00:31<00:21, 7369.83it/s] 60%|█████▉    | 239049/400000 [00:31<00:21, 7416.84it/s] 60%|█████▉    | 239806/400000 [00:31<00:21, 7461.25it/s] 60%|██████    | 240553/400000 [00:32<00:21, 7428.03it/s] 60%|██████    | 241316/400000 [00:32<00:21, 7483.95it/s] 61%|██████    | 242082/400000 [00:32<00:20, 7529.19it/s] 61%|██████    | 242836/400000 [00:32<00:20, 7496.34it/s] 61%|██████    | 243589/400000 [00:32<00:20, 7503.22it/s] 61%|██████    | 244340/400000 [00:32<00:20, 7438.31it/s] 61%|██████▏   | 245098/400000 [00:32<00:20, 7476.93it/s] 61%|██████▏   | 245862/400000 [00:32<00:20, 7524.18it/s] 62%|██████▏   | 246615/400000 [00:32<00:20, 7513.67it/s] 62%|██████▏   | 247370/400000 [00:32<00:20, 7522.80it/s] 62%|██████▏   | 248123/400000 [00:33<00:20, 7470.67it/s] 62%|██████▏   | 248871/400000 [00:33<00:20, 7463.73it/s] 62%|██████▏   | 249638/400000 [00:33<00:19, 7522.90it/s] 63%|██████▎   | 250391/400000 [00:33<00:19, 7513.61it/s] 63%|██████▎   | 251160/400000 [00:33<00:19, 7565.10it/s] 63%|██████▎   | 251917/400000 [00:33<00:19, 7495.36it/s] 63%|██████▎   | 252667/400000 [00:33<00:19, 7486.37it/s] 63%|██████▎   | 253443/400000 [00:33<00:19, 7563.37it/s] 64%|██████▎   | 254200/400000 [00:33<00:19, 7552.12it/s] 64%|██████▎   | 254976/400000 [00:34<00:19, 7611.11it/s] 64%|██████▍   | 255738/400000 [00:34<00:19, 7231.89it/s] 64%|██████▍   | 256467/400000 [00:34<00:19, 7248.56it/s] 64%|██████▍   | 257220/400000 [00:34<00:19, 7329.50it/s] 64%|██████▍   | 257956/400000 [00:34<00:19, 7168.64it/s] 65%|██████▍   | 258709/400000 [00:34<00:19, 7271.94it/s] 65%|██████▍   | 259454/400000 [00:34<00:19, 7323.67it/s] 65%|██████▌   | 260226/400000 [00:34<00:18, 7435.85it/s] 65%|██████▌   | 260999/400000 [00:34<00:18, 7517.71it/s] 65%|██████▌   | 261761/400000 [00:34<00:18, 7546.27it/s] 66%|██████▌   | 262529/400000 [00:35<00:18, 7585.51it/s] 66%|██████▌   | 263293/400000 [00:35<00:17, 7600.87it/s] 66%|██████▌   | 264057/400000 [00:35<00:17, 7610.38it/s] 66%|██████▌   | 264819/400000 [00:35<00:17, 7579.92it/s] 66%|██████▋   | 265609/400000 [00:35<00:17, 7671.73it/s] 67%|██████▋   | 266390/400000 [00:35<00:17, 7711.61it/s] 67%|██████▋   | 267162/400000 [00:35<00:17, 7637.12it/s] 67%|██████▋   | 267927/400000 [00:35<00:17, 7581.20it/s] 67%|██████▋   | 268686/400000 [00:35<00:17, 7543.28it/s] 67%|██████▋   | 269444/400000 [00:35<00:17, 7553.09it/s] 68%|██████▊   | 270200/400000 [00:36<00:17, 7317.74it/s] 68%|██████▊   | 270943/400000 [00:36<00:17, 7350.12it/s] 68%|██████▊   | 271714/400000 [00:36<00:17, 7451.73it/s] 68%|██████▊   | 272486/400000 [00:36<00:16, 7528.45it/s] 68%|██████▊   | 273263/400000 [00:36<00:16, 7596.89it/s] 69%|██████▊   | 274024/400000 [00:36<00:16, 7596.37it/s] 69%|██████▊   | 274785/400000 [00:36<00:16, 7520.91it/s] 69%|██████▉   | 275538/400000 [00:36<00:17, 7318.00it/s] 69%|██████▉   | 276272/400000 [00:36<00:17, 7266.55it/s] 69%|██████▉   | 277006/400000 [00:36<00:16, 7288.05it/s] 69%|██████▉   | 277738/400000 [00:37<00:16, 7294.57it/s] 70%|██████▉   | 278469/400000 [00:37<00:16, 7269.48it/s] 70%|██████▉   | 279231/400000 [00:37<00:16, 7370.60it/s] 70%|██████▉   | 279972/400000 [00:37<00:16, 7379.95it/s] 70%|███████   | 280730/400000 [00:37<00:16, 7437.63it/s] 70%|███████   | 281501/400000 [00:37<00:15, 7517.19it/s] 71%|███████   | 282254/400000 [00:37<00:15, 7435.42it/s] 71%|███████   | 283001/400000 [00:37<00:15, 7445.29it/s] 71%|███████   | 283766/400000 [00:37<00:15, 7504.71it/s] 71%|███████   | 284533/400000 [00:37<00:15, 7551.35it/s] 71%|███████▏  | 285295/400000 [00:38<00:15, 7570.72it/s] 72%|███████▏  | 286053/400000 [00:38<00:15, 7493.84it/s] 72%|███████▏  | 286803/400000 [00:38<00:15, 7474.72it/s] 72%|███████▏  | 287555/400000 [00:38<00:15, 7485.69it/s] 72%|███████▏  | 288304/400000 [00:38<00:14, 7485.12it/s] 72%|███████▏  | 289061/400000 [00:38<00:14, 7507.73it/s] 72%|███████▏  | 289812/400000 [00:38<00:15, 7315.02it/s] 73%|███████▎  | 290568/400000 [00:38<00:14, 7385.96it/s] 73%|███████▎  | 291321/400000 [00:38<00:14, 7426.64it/s] 73%|███████▎  | 292065/400000 [00:38<00:14, 7394.12it/s] 73%|███████▎  | 292813/400000 [00:39<00:14, 7417.44it/s] 73%|███████▎  | 293556/400000 [00:39<00:14, 7378.65it/s] 74%|███████▎  | 294295/400000 [00:39<00:14, 7325.53it/s] 74%|███████▍  | 295053/400000 [00:39<00:14, 7399.58it/s] 74%|███████▍  | 295813/400000 [00:39<00:13, 7458.05it/s] 74%|███████▍  | 296586/400000 [00:39<00:13, 7535.44it/s] 74%|███████▍  | 297341/400000 [00:39<00:13, 7431.37it/s] 75%|███████▍  | 298085/400000 [00:39<00:13, 7405.97it/s] 75%|███████▍  | 298827/400000 [00:39<00:13, 7374.27it/s] 75%|███████▍  | 299565/400000 [00:40<00:13, 7334.53it/s] 75%|███████▌  | 300299/400000 [00:40<00:13, 7294.14it/s] 75%|███████▌  | 301029/400000 [00:40<00:13, 7253.04it/s] 75%|███████▌  | 301778/400000 [00:40<00:13, 7315.51it/s] 76%|███████▌  | 302540/400000 [00:40<00:13, 7403.84it/s] 76%|███████▌  | 303298/400000 [00:40<00:12, 7453.72it/s] 76%|███████▌  | 304056/400000 [00:40<00:12, 7488.94it/s] 76%|███████▌  | 304806/400000 [00:40<00:12, 7372.99it/s] 76%|███████▋  | 305552/400000 [00:40<00:12, 7397.21it/s] 77%|███████▋  | 306326/400000 [00:40<00:12, 7494.38it/s] 77%|███████▋  | 307077/400000 [00:41<00:12, 7498.38it/s] 77%|███████▋  | 307853/400000 [00:41<00:12, 7573.80it/s] 77%|███████▋  | 308611/400000 [00:41<00:12, 7523.15it/s] 77%|███████▋  | 309381/400000 [00:41<00:11, 7575.24it/s] 78%|███████▊  | 310170/400000 [00:41<00:11, 7664.27it/s] 78%|███████▊  | 310950/400000 [00:41<00:11, 7703.27it/s] 78%|███████▊  | 311721/400000 [00:41<00:11, 7690.72it/s] 78%|███████▊  | 312491/400000 [00:41<00:11, 7596.13it/s] 78%|███████▊  | 313252/400000 [00:41<00:11, 7585.14it/s] 79%|███████▊  | 314011/400000 [00:41<00:11, 7577.22it/s] 79%|███████▊  | 314769/400000 [00:42<00:11, 7552.15it/s] 79%|███████▉  | 315525/400000 [00:42<00:11, 7515.41it/s] 79%|███████▉  | 316277/400000 [00:42<00:11, 7440.36it/s] 79%|███████▉  | 317022/400000 [00:42<00:11, 7202.82it/s] 79%|███████▉  | 317770/400000 [00:42<00:11, 7282.91it/s] 80%|███████▉  | 318543/400000 [00:42<00:10, 7410.56it/s] 80%|███████▉  | 319325/400000 [00:42<00:10, 7526.25it/s] 80%|████████  | 320080/400000 [00:42<00:10, 7486.86it/s] 80%|████████  | 320850/400000 [00:42<00:10, 7548.80it/s] 80%|████████  | 321623/400000 [00:42<00:10, 7600.49it/s] 81%|████████  | 322393/400000 [00:43<00:10, 7627.66it/s] 81%|████████  | 323157/400000 [00:43<00:10, 7595.92it/s] 81%|████████  | 323917/400000 [00:43<00:10, 7539.97it/s] 81%|████████  | 324691/400000 [00:43<00:09, 7598.72it/s] 81%|████████▏ | 325469/400000 [00:43<00:09, 7650.24it/s] 82%|████████▏ | 326238/400000 [00:43<00:09, 7660.30it/s] 82%|████████▏ | 327005/400000 [00:43<00:09, 7512.35it/s] 82%|████████▏ | 327758/400000 [00:43<00:09, 7408.67it/s] 82%|████████▏ | 328519/400000 [00:43<00:09, 7466.73it/s] 82%|████████▏ | 329297/400000 [00:43<00:09, 7555.93it/s] 83%|████████▎ | 330054/400000 [00:44<00:09, 7507.82it/s] 83%|████████▎ | 330826/400000 [00:44<00:09, 7569.94it/s] 83%|████████▎ | 331584/400000 [00:44<00:09, 7472.08it/s] 83%|████████▎ | 332332/400000 [00:44<00:09, 7391.36it/s] 83%|████████▎ | 333077/400000 [00:44<00:09, 7407.29it/s] 83%|████████▎ | 333857/400000 [00:44<00:08, 7518.26it/s] 84%|████████▎ | 334633/400000 [00:44<00:08, 7588.53it/s] 84%|████████▍ | 335393/400000 [00:44<00:08, 7542.65it/s] 84%|████████▍ | 336159/400000 [00:44<00:08, 7575.70it/s] 84%|████████▍ | 336922/400000 [00:44<00:08, 7589.95it/s] 84%|████████▍ | 337701/400000 [00:45<00:08, 7647.36it/s] 85%|████████▍ | 338467/400000 [00:45<00:08, 7630.42it/s] 85%|████████▍ | 339231/400000 [00:45<00:08, 7554.42it/s] 85%|████████▍ | 339987/400000 [00:45<00:07, 7546.57it/s] 85%|████████▌ | 340764/400000 [00:45<00:07, 7610.92it/s] 85%|████████▌ | 341526/400000 [00:45<00:07, 7603.57it/s] 86%|████████▌ | 342287/400000 [00:45<00:07, 7552.49it/s] 86%|████████▌ | 343043/400000 [00:45<00:07, 7278.96it/s] 86%|████████▌ | 343791/400000 [00:45<00:07, 7337.70it/s] 86%|████████▌ | 344564/400000 [00:45<00:07, 7449.59it/s] 86%|████████▋ | 345314/400000 [00:46<00:07, 7464.47it/s] 87%|████████▋ | 346097/400000 [00:46<00:07, 7570.05it/s] 87%|████████▋ | 346856/400000 [00:46<00:07, 7534.92it/s] 87%|████████▋ | 347632/400000 [00:46<00:06, 7600.78it/s] 87%|████████▋ | 348406/400000 [00:46<00:06, 7641.09it/s] 87%|████████▋ | 349180/400000 [00:46<00:06, 7668.61it/s] 87%|████████▋ | 349979/400000 [00:46<00:06, 7761.97it/s] 88%|████████▊ | 350756/400000 [00:46<00:06, 7722.58it/s] 88%|████████▊ | 351536/400000 [00:46<00:06, 7743.36it/s] 88%|████████▊ | 352325/400000 [00:46<00:06, 7785.64it/s] 88%|████████▊ | 353132/400000 [00:47<00:05, 7867.08it/s] 88%|████████▊ | 353935/400000 [00:47<00:05, 7914.96it/s] 89%|████████▊ | 354727/400000 [00:47<00:05, 7764.61it/s] 89%|████████▉ | 355505/400000 [00:47<00:05, 7726.25it/s] 89%|████████▉ | 356298/400000 [00:47<00:05, 7783.65it/s] 89%|████████▉ | 357102/400000 [00:47<00:05, 7858.13it/s] 89%|████████▉ | 357899/400000 [00:47<00:05, 7889.96it/s] 90%|████████▉ | 358689/400000 [00:47<00:05, 7780.87it/s] 90%|████████▉ | 359484/400000 [00:47<00:05, 7830.76it/s] 90%|█████████ | 360268/400000 [00:48<00:05, 7833.10it/s] 90%|█████████ | 361052/400000 [00:48<00:04, 7799.95it/s] 90%|█████████ | 361833/400000 [00:48<00:04, 7706.64it/s] 91%|█████████ | 362605/400000 [00:48<00:04, 7648.19it/s] 91%|█████████ | 363391/400000 [00:48<00:04, 7707.35it/s] 91%|█████████ | 364178/400000 [00:48<00:04, 7755.35it/s] 91%|█████████ | 364954/400000 [00:48<00:04, 7499.42it/s] 91%|█████████▏| 365713/400000 [00:48<00:04, 7522.80it/s] 92%|█████████▏| 366467/400000 [00:48<00:04, 7506.40it/s] 92%|█████████▏| 367238/400000 [00:48<00:04, 7566.18it/s] 92%|█████████▏| 367996/400000 [00:49<00:04, 7559.01it/s] 92%|█████████▏| 368766/400000 [00:49<00:04, 7598.55it/s] 92%|█████████▏| 369532/400000 [00:49<00:04, 7614.95it/s] 93%|█████████▎| 370294/400000 [00:49<00:03, 7594.30it/s] 93%|█████████▎| 371054/400000 [00:49<00:03, 7503.28it/s] 93%|█████████▎| 371824/400000 [00:49<00:03, 7560.91it/s] 93%|█████████▎| 372613/400000 [00:49<00:03, 7654.71it/s] 93%|█████████▎| 373398/400000 [00:49<00:03, 7711.73it/s] 94%|█████████▎| 374170/400000 [00:49<00:03, 7699.57it/s] 94%|█████████▎| 374957/400000 [00:49<00:03, 7746.72it/s] 94%|█████████▍| 375741/400000 [00:50<00:03, 7773.82it/s] 94%|█████████▍| 376529/400000 [00:50<00:03, 7803.55it/s] 94%|█████████▍| 377320/400000 [00:50<00:02, 7834.43it/s] 95%|█████████▍| 378104/400000 [00:50<00:02, 7675.91it/s] 95%|█████████▍| 378885/400000 [00:50<00:02, 7714.40it/s] 95%|█████████▍| 379662/400000 [00:50<00:02, 7729.14it/s] 95%|█████████▌| 380436/400000 [00:50<00:02, 7463.56it/s] 95%|█████████▌| 381185/400000 [00:50<00:02, 7433.28it/s] 95%|█████████▌| 381930/400000 [00:50<00:02, 7404.37it/s] 96%|█████████▌| 382677/400000 [00:50<00:02, 7421.72it/s] 96%|█████████▌| 383462/400000 [00:51<00:02, 7543.77it/s] 96%|█████████▌| 384248/400000 [00:51<00:02, 7634.73it/s] 96%|█████████▋| 385021/400000 [00:51<00:01, 7662.45it/s] 96%|█████████▋| 385788/400000 [00:51<00:01, 7612.94it/s] 97%|█████████▋| 386557/400000 [00:51<00:01, 7635.61it/s] 97%|█████████▋| 387328/400000 [00:51<00:01, 7656.64it/s] 97%|█████████▋| 388094/400000 [00:51<00:01, 7616.61it/s] 97%|█████████▋| 388856/400000 [00:51<00:01, 7390.11it/s] 97%|█████████▋| 389597/400000 [00:51<00:01, 7391.24it/s] 98%|█████████▊| 390366/400000 [00:51<00:01, 7476.57it/s] 98%|█████████▊| 391119/400000 [00:52<00:01, 7490.01it/s] 98%|█████████▊| 391891/400000 [00:52<00:01, 7554.91it/s] 98%|█████████▊| 392648/400000 [00:52<00:00, 7429.06it/s] 98%|█████████▊| 393392/400000 [00:52<00:00, 7330.65it/s] 99%|█████████▊| 394127/400000 [00:52<00:00, 7264.64it/s] 99%|█████████▊| 394855/400000 [00:52<00:00, 7156.71it/s] 99%|█████████▉| 395602/400000 [00:52<00:00, 7247.38it/s] 99%|█████████▉| 396355/400000 [00:52<00:00, 7329.07it/s] 99%|█████████▉| 397090/400000 [00:52<00:00, 7333.84it/s] 99%|█████████▉| 397857/400000 [00:52<00:00, 7430.56it/s]100%|█████████▉| 398619/400000 [00:53<00:00, 7486.14it/s]100%|█████████▉| 399378/400000 [00:53<00:00, 7514.09it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7507.56it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f95816fbbe0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011446361889654985 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011349301274404876 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15960 out of table with 15838 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15960 out of table with 15838 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb5883b2fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 01:15:14.652642
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 01:15:14.659209
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 01:15:14.665346
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 01:15:14.670384
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb5943ca4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354083.4688
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 256726.0469
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 159611.2188
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 85147.9688
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 43939.5508
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 23997.6836
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 14576.0166
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 9643.9609
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 6882.8687
Epoch 10/10

1/1 [==============================] - 0s 87ms/step - loss: 5185.1743

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.80787921e-01  7.52040720e+00  9.44233704e+00  1.07711058e+01
   9.17455864e+00  6.95110607e+00  1.14145584e+01  1.06034021e+01
   1.07391853e+01  7.79919529e+00  8.26246166e+00  9.35336494e+00
   9.46431255e+00  1.05039587e+01  8.04717636e+00  9.53972912e+00
   9.87671280e+00  9.11230183e+00  8.85066223e+00  8.33891582e+00
   8.69220066e+00  1.10789471e+01  1.00108242e+01  9.36486340e+00
   9.22116756e+00  1.20145712e+01  9.08728886e+00  7.56740046e+00
   8.07522392e+00  9.42145157e+00  9.75564575e+00  1.04484873e+01
   1.07477846e+01  9.63156223e+00  9.93306637e+00  9.93328190e+00
   9.91854572e+00  9.89242363e+00  8.14416313e+00  9.09090996e+00
   8.68591213e+00  1.04488535e+01  9.00595760e+00  9.89713192e+00
   1.07281551e+01  1.15916967e+01  9.63251114e+00  6.92385530e+00
   9.84403133e+00  1.00748596e+01  1.04227867e+01  9.15160847e+00
   8.54246330e+00  9.03328705e+00  7.78296089e+00  8.95687389e+00
   1.02178383e+01  1.15516253e+01  8.87447166e+00  8.59942818e+00
  -3.16268802e-02 -6.48307800e-03  3.61118257e-01 -2.09480762e+00
  -1.16871238e-01 -8.92884851e-01  7.45144248e-01 -1.66995478e+00
  -9.48299170e-01  1.64220423e-01  2.34791517e+00 -2.54732192e-01
   9.67427135e-01  1.66444850e+00  1.59036815e-02  5.53831697e-01
   4.88676518e-01  1.83201170e+00 -1.66168833e+00 -7.35040069e-01
   5.75939119e-01  4.91043329e-01  5.82942486e-01  2.73263127e-01
   1.13946331e+00 -2.41716433e+00  3.40265900e-01 -2.32056665e+00
  -6.47589922e-01 -1.95684159e+00 -1.67139745e+00 -2.85500169e-01
  -1.19221067e+00 -1.16811478e+00  6.15928173e-01 -1.51604652e+00
   2.35027224e-01 -2.37930156e-02  6.73847675e-01 -9.61728930e-01
  -3.05180359e+00 -1.26136518e+00  9.94471252e-01 -7.32407272e-01
   3.98508489e-01 -2.54264057e-01 -1.17532814e+00 -4.01508927e-01
  -7.18950629e-02 -2.57741034e-01  2.71876663e-01 -4.55410212e-01
   9.13587451e-01 -1.85951495e+00 -2.97165990e-01  4.34632003e-02
  -2.39587831e+00 -4.92357463e-01  6.04045570e-01  1.09521770e+00
   3.04356515e-02  3.36023867e-01  3.03735673e-01  1.17250532e-01
   1.75895810e+00  1.65470481e-01 -1.79968894e+00  1.00417209e+00
   8.15890193e-01 -5.03770411e-01 -1.36245739e+00  9.63411212e-01
   1.10689211e+00 -1.86615682e+00 -1.50793481e+00  1.81356013e-01
   1.66728306e+00  2.83624887e-01 -4.76212204e-02  3.13597620e-01
   1.71611741e-01  7.65193462e-01 -4.57989573e-02 -1.59376061e+00
   6.28136516e-01 -5.77953935e-01 -1.73302686e+00 -7.87038743e-01
  -3.38035166e-01 -4.68231738e-02  2.13694620e+00 -2.86449134e-01
  -6.09525740e-01  1.63004279e-01  1.08067119e+00 -1.22803807e+00
   1.53082657e+00  5.24623811e-01 -2.78966874e-01 -2.84097850e-01
   1.01520908e+00 -2.00052094e+00  3.84028971e-01  1.86381853e+00
  -7.72414684e-01  1.30579376e+00  1.50524676e-01 -7.37336278e-01
  -1.11537933e+00  6.67870820e-01 -6.32718980e-01  9.50935856e-03
  -1.06507301e+00  6.22566879e-01 -3.03449869e-01  8.12958658e-01
  -7.11378336e-01  7.74081409e-01  9.48869944e-01 -1.51056862e+00
   1.88599050e-01  9.22228146e+00  9.67908764e+00  1.06149521e+01
   1.03935900e+01  9.58920288e+00  9.26879120e+00  1.13320208e+01
   9.49011993e+00  9.86385822e+00  1.04582939e+01  9.56974983e+00
   8.21303558e+00  9.43742943e+00  9.36349964e+00  9.82659435e+00
   9.90040493e+00  1.00123043e+01  8.56684971e+00  9.51528072e+00
   8.91192818e+00  9.82813263e+00  1.02766409e+01  9.86081982e+00
   9.41376305e+00  9.99518585e+00  7.01536131e+00  1.02170591e+01
   1.02052746e+01  9.42646503e+00  1.15374746e+01  7.71057844e+00
   9.42093182e+00  1.04226351e+01  9.00929546e+00  9.82439137e+00
   1.07850609e+01  8.73135662e+00  8.29291916e+00  1.15741510e+01
   8.94714451e+00  1.06471910e+01  1.11898108e+01  9.99355602e+00
   9.61442471e+00  1.01419735e+01  1.02282019e+01  1.20751371e+01
   1.03775692e+01  8.59246922e+00  1.05595675e+01  8.84782124e+00
   1.04336376e+01  1.04596701e+01  9.12178230e+00  8.86067486e+00
   1.13150206e+01  1.02067099e+01  8.94719410e+00  1.10240822e+01
   1.43845534e+00  5.95924258e-01  3.82649243e-01  1.74740553e+00
   2.22743511e+00  1.21082067e-01  8.55027437e-02  1.49831188e+00
   5.59678793e-01  1.56153786e+00  6.04386628e-01  2.30853379e-01
   9.39794362e-01  9.18707490e-01  1.65767860e+00  1.16103721e+00
   5.37999272e-01  5.26972175e-01  6.66407704e-01  1.08464932e+00
   1.06584668e-01  9.53926861e-01  5.25471687e-01  1.70774913e+00
   2.74536669e-01  4.04317796e-01  2.29393291e+00  1.68460846e-01
   3.27008128e-01  9.04234529e-01  7.57569313e-01  1.45465410e+00
   8.83602977e-01  1.94339073e+00  2.54914641e-01  9.12113667e-01
   5.13444960e-01  1.94192493e+00  5.17131925e-01  2.28890514e+00
   1.68550193e+00  2.75168896e-01  6.92100525e-01  2.33181334e+00
   4.19145882e-01  2.71495581e-01  7.57660985e-01  1.65920794e+00
   3.69213283e-01  4.02487230e+00  1.70351624e+00  1.32421851e+00
   1.69335055e+00  3.25972652e+00  1.01359046e+00  2.00612307e+00
   1.52183819e+00  2.34263849e+00  7.70413280e-01  2.42143965e+00
   1.64045453e+00  1.20409775e+00  1.39795971e+00  3.85283828e-01
   2.32452631e+00  1.37336707e+00  2.33110714e+00  1.38582981e+00
   2.07603121e+00  4.89590287e-01  1.04623437e-01  4.96767342e-01
   6.76905334e-01  2.42941165e+00  1.47748649e+00  1.96873403e+00
   1.54378510e+00  6.78875208e-01  5.38984179e-01  1.29427779e+00
   4.52445626e-01  1.70664513e+00  3.17078233e-01  1.35983062e+00
   5.39140821e-01  2.90077031e-01  1.07475066e+00  2.01665401e+00
   6.02273166e-01  1.31058788e+00  1.67682409e-01  2.41629124e-01
   1.67300868e+00  4.20174897e-01  1.49205041e+00  3.54527175e-01
   6.48095667e-01  1.96507061e+00  2.13106513e+00  2.27612305e+00
   1.36282229e+00  1.83306813e-01  2.82785892e-01  1.52698278e+00
   1.66148257e+00  2.57554317e+00  9.37156141e-01  1.03444993e+00
   2.26035833e+00  6.82816803e-01  1.04863369e+00  2.94953227e-01
   4.75939751e-01  5.34073353e-01  1.50574028e+00  4.09665823e-01
   1.20796609e+00  2.73607969e-01  1.17603397e+00  3.55903387e-01
   7.70162868e+00 -3.45834637e+00 -3.23503828e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 01:15:24.169907
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.8933
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 01:15:24.174179
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8648.51
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 01:15:24.177848
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5947
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 01:15:24.181517
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -773.535
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140417294673008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140414782038200
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140414782038704
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414782039208
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414782039712
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140414782040216

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb573fdefd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.693316
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.647875
grad_step = 000002, loss = 0.620910
grad_step = 000003, loss = 0.594786
grad_step = 000004, loss = 0.571222
grad_step = 000005, loss = 0.550989
grad_step = 000006, loss = 0.529398
grad_step = 000007, loss = 0.505909
grad_step = 000008, loss = 0.489668
grad_step = 000009, loss = 0.480820
grad_step = 000010, loss = 0.467265
grad_step = 000011, loss = 0.451837
grad_step = 000012, loss = 0.438820
grad_step = 000013, loss = 0.427638
grad_step = 000014, loss = 0.415598
grad_step = 000015, loss = 0.403521
grad_step = 000016, loss = 0.390727
grad_step = 000017, loss = 0.377108
grad_step = 000018, loss = 0.363139
grad_step = 000019, loss = 0.349653
grad_step = 000020, loss = 0.337175
grad_step = 000021, loss = 0.326168
grad_step = 000022, loss = 0.315338
grad_step = 000023, loss = 0.303633
grad_step = 000024, loss = 0.292220
grad_step = 000025, loss = 0.281699
grad_step = 000026, loss = 0.271788
grad_step = 000027, loss = 0.261825
grad_step = 000028, loss = 0.251505
grad_step = 000029, loss = 0.241419
grad_step = 000030, loss = 0.231916
grad_step = 000031, loss = 0.222468
grad_step = 000032, loss = 0.213195
grad_step = 000033, loss = 0.204169
grad_step = 000034, loss = 0.195138
grad_step = 000035, loss = 0.186400
grad_step = 000036, loss = 0.178312
grad_step = 000037, loss = 0.170487
grad_step = 000038, loss = 0.162675
grad_step = 000039, loss = 0.155037
grad_step = 000040, loss = 0.147872
grad_step = 000041, loss = 0.141132
grad_step = 000042, loss = 0.134455
grad_step = 000043, loss = 0.128064
grad_step = 000044, loss = 0.121991
grad_step = 000045, loss = 0.116067
grad_step = 000046, loss = 0.110454
grad_step = 000047, loss = 0.105060
grad_step = 000048, loss = 0.099829
grad_step = 000049, loss = 0.094931
grad_step = 000050, loss = 0.090279
grad_step = 000051, loss = 0.085830
grad_step = 000052, loss = 0.081608
grad_step = 000053, loss = 0.077603
grad_step = 000054, loss = 0.073724
grad_step = 000055, loss = 0.070082
grad_step = 000056, loss = 0.066606
grad_step = 000057, loss = 0.063231
grad_step = 000058, loss = 0.060056
grad_step = 000059, loss = 0.057045
grad_step = 000060, loss = 0.054177
grad_step = 000061, loss = 0.051420
grad_step = 000062, loss = 0.048801
grad_step = 000063, loss = 0.046329
grad_step = 000064, loss = 0.043941
grad_step = 000065, loss = 0.041654
grad_step = 000066, loss = 0.039499
grad_step = 000067, loss = 0.037421
grad_step = 000068, loss = 0.035438
grad_step = 000069, loss = 0.033538
grad_step = 000070, loss = 0.031724
grad_step = 000071, loss = 0.029988
grad_step = 000072, loss = 0.028332
grad_step = 000073, loss = 0.026759
grad_step = 000074, loss = 0.025248
grad_step = 000075, loss = 0.023824
grad_step = 000076, loss = 0.022453
grad_step = 000077, loss = 0.021145
grad_step = 000078, loss = 0.019904
grad_step = 000079, loss = 0.018724
grad_step = 000080, loss = 0.017603
grad_step = 000081, loss = 0.016543
grad_step = 000082, loss = 0.015542
grad_step = 000083, loss = 0.014592
grad_step = 000084, loss = 0.013698
grad_step = 000085, loss = 0.012853
grad_step = 000086, loss = 0.012060
grad_step = 000087, loss = 0.011314
grad_step = 000088, loss = 0.010613
grad_step = 000089, loss = 0.009956
grad_step = 000090, loss = 0.009343
grad_step = 000091, loss = 0.008767
grad_step = 000092, loss = 0.008233
grad_step = 000093, loss = 0.007735
grad_step = 000094, loss = 0.007272
grad_step = 000095, loss = 0.006841
grad_step = 000096, loss = 0.006442
grad_step = 000097, loss = 0.006072
grad_step = 000098, loss = 0.005729
grad_step = 000099, loss = 0.005412
grad_step = 000100, loss = 0.005121
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004851
grad_step = 000102, loss = 0.004603
grad_step = 000103, loss = 0.004376
grad_step = 000104, loss = 0.004166
grad_step = 000105, loss = 0.003973
grad_step = 000106, loss = 0.003796
grad_step = 000107, loss = 0.003635
grad_step = 000108, loss = 0.003487
grad_step = 000109, loss = 0.003353
grad_step = 000110, loss = 0.003234
grad_step = 000111, loss = 0.003132
grad_step = 000112, loss = 0.003039
grad_step = 000113, loss = 0.002959
grad_step = 000114, loss = 0.002891
grad_step = 000115, loss = 0.002812
grad_step = 000116, loss = 0.002725
grad_step = 000117, loss = 0.002638
grad_step = 000118, loss = 0.002562
grad_step = 000119, loss = 0.002503
grad_step = 000120, loss = 0.002467
grad_step = 000121, loss = 0.002441
grad_step = 000122, loss = 0.002408
grad_step = 000123, loss = 0.002367
grad_step = 000124, loss = 0.002324
grad_step = 000125, loss = 0.002279
grad_step = 000126, loss = 0.002241
grad_step = 000127, loss = 0.002216
grad_step = 000128, loss = 0.002202
grad_step = 000129, loss = 0.002188
grad_step = 000130, loss = 0.002173
grad_step = 000131, loss = 0.002161
grad_step = 000132, loss = 0.002154
grad_step = 000133, loss = 0.002145
grad_step = 000134, loss = 0.002113
grad_step = 000135, loss = 0.002091
grad_step = 000136, loss = 0.002081
grad_step = 000137, loss = 0.002068
grad_step = 000138, loss = 0.002051
grad_step = 000139, loss = 0.002039
grad_step = 000140, loss = 0.002037
grad_step = 000141, loss = 0.002042
grad_step = 000142, loss = 0.002042
grad_step = 000143, loss = 0.002038
grad_step = 000144, loss = 0.002029
grad_step = 000145, loss = 0.002027
grad_step = 000146, loss = 0.002043
grad_step = 000147, loss = 0.002078
grad_step = 000148, loss = 0.002108
grad_step = 000149, loss = 0.002050
grad_step = 000150, loss = 0.002004
grad_step = 000151, loss = 0.002023
grad_step = 000152, loss = 0.002031
grad_step = 000153, loss = 0.001991
grad_step = 000154, loss = 0.001965
grad_step = 000155, loss = 0.001987
grad_step = 000156, loss = 0.001998
grad_step = 000157, loss = 0.001972
grad_step = 000158, loss = 0.001968
grad_step = 000159, loss = 0.001989
grad_step = 000160, loss = 0.001988
grad_step = 000161, loss = 0.001967
grad_step = 000162, loss = 0.001958
grad_step = 000163, loss = 0.001966
grad_step = 000164, loss = 0.001967
grad_step = 000165, loss = 0.001951
grad_step = 000166, loss = 0.001939
grad_step = 000167, loss = 0.001942
grad_step = 000168, loss = 0.001947
grad_step = 000169, loss = 0.001943
grad_step = 000170, loss = 0.001933
grad_step = 000171, loss = 0.001928
grad_step = 000172, loss = 0.001931
grad_step = 000173, loss = 0.001935
grad_step = 000174, loss = 0.001934
grad_step = 000175, loss = 0.001930
grad_step = 000176, loss = 0.001929
grad_step = 000177, loss = 0.001935
grad_step = 000178, loss = 0.001951
grad_step = 000179, loss = 0.001980
grad_step = 000180, loss = 0.002018
grad_step = 000181, loss = 0.002076
grad_step = 000182, loss = 0.002111
grad_step = 000183, loss = 0.002133
grad_step = 000184, loss = 0.002093
grad_step = 000185, loss = 0.002024
grad_step = 000186, loss = 0.001951
grad_step = 000187, loss = 0.001920
grad_step = 000188, loss = 0.001979
grad_step = 000189, loss = 0.002031
grad_step = 000190, loss = 0.001972
grad_step = 000191, loss = 0.001908
grad_step = 000192, loss = 0.001920
grad_step = 000193, loss = 0.001935
grad_step = 000194, loss = 0.001934
grad_step = 000195, loss = 0.001945
grad_step = 000196, loss = 0.001923
grad_step = 000197, loss = 0.001888
grad_step = 000198, loss = 0.001894
grad_step = 000199, loss = 0.001918
grad_step = 000200, loss = 0.001914
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001892
grad_step = 000202, loss = 0.001887
grad_step = 000203, loss = 0.001893
grad_step = 000204, loss = 0.001890
grad_step = 000205, loss = 0.001881
grad_step = 000206, loss = 0.001883
grad_step = 000207, loss = 0.001888
grad_step = 000208, loss = 0.001883
grad_step = 000209, loss = 0.001871
grad_step = 000210, loss = 0.001865
grad_step = 000211, loss = 0.001869
grad_step = 000212, loss = 0.001875
grad_step = 000213, loss = 0.001875
grad_step = 000214, loss = 0.001868
grad_step = 000215, loss = 0.001862
grad_step = 000216, loss = 0.001861
grad_step = 000217, loss = 0.001860
grad_step = 000218, loss = 0.001857
grad_step = 000219, loss = 0.001854
grad_step = 000220, loss = 0.001853
grad_step = 000221, loss = 0.001854
grad_step = 000222, loss = 0.001856
grad_step = 000223, loss = 0.001858
grad_step = 000224, loss = 0.001858
grad_step = 000225, loss = 0.001860
grad_step = 000226, loss = 0.001864
grad_step = 000227, loss = 0.001875
grad_step = 000228, loss = 0.001892
grad_step = 000229, loss = 0.001915
grad_step = 000230, loss = 0.001938
grad_step = 000231, loss = 0.001953
grad_step = 000232, loss = 0.001951
grad_step = 000233, loss = 0.001938
grad_step = 000234, loss = 0.001910
grad_step = 000235, loss = 0.001882
grad_step = 000236, loss = 0.001852
grad_step = 000237, loss = 0.001836
grad_step = 000238, loss = 0.001838
grad_step = 000239, loss = 0.001856
grad_step = 000240, loss = 0.001876
grad_step = 000241, loss = 0.001882
grad_step = 000242, loss = 0.001875
grad_step = 000243, loss = 0.001858
grad_step = 000244, loss = 0.001844
grad_step = 000245, loss = 0.001838
grad_step = 000246, loss = 0.001837
grad_step = 000247, loss = 0.001835
grad_step = 000248, loss = 0.001829
grad_step = 000249, loss = 0.001823
grad_step = 000250, loss = 0.001820
grad_step = 000251, loss = 0.001823
grad_step = 000252, loss = 0.001830
grad_step = 000253, loss = 0.001839
grad_step = 000254, loss = 0.001849
grad_step = 000255, loss = 0.001861
grad_step = 000256, loss = 0.001877
grad_step = 000257, loss = 0.001899
grad_step = 000258, loss = 0.001920
grad_step = 000259, loss = 0.001936
grad_step = 000260, loss = 0.001934
grad_step = 000261, loss = 0.001916
grad_step = 000262, loss = 0.001879
grad_step = 000263, loss = 0.001843
grad_step = 000264, loss = 0.001817
grad_step = 000265, loss = 0.001810
grad_step = 000266, loss = 0.001820
grad_step = 000267, loss = 0.001840
grad_step = 000268, loss = 0.001859
grad_step = 000269, loss = 0.001862
grad_step = 000270, loss = 0.001854
grad_step = 000271, loss = 0.001834
grad_step = 000272, loss = 0.001815
grad_step = 000273, loss = 0.001804
grad_step = 000274, loss = 0.001802
grad_step = 000275, loss = 0.001805
grad_step = 000276, loss = 0.001810
grad_step = 000277, loss = 0.001813
grad_step = 000278, loss = 0.001816
grad_step = 000279, loss = 0.001819
grad_step = 000280, loss = 0.001824
grad_step = 000281, loss = 0.001830
grad_step = 000282, loss = 0.001837
grad_step = 000283, loss = 0.001841
grad_step = 000284, loss = 0.001842
grad_step = 000285, loss = 0.001839
grad_step = 000286, loss = 0.001831
grad_step = 000287, loss = 0.001821
grad_step = 000288, loss = 0.001810
grad_step = 000289, loss = 0.001803
grad_step = 000290, loss = 0.001799
grad_step = 000291, loss = 0.001797
grad_step = 000292, loss = 0.001796
grad_step = 000293, loss = 0.001795
grad_step = 000294, loss = 0.001793
grad_step = 000295, loss = 0.001791
grad_step = 000296, loss = 0.001788
grad_step = 000297, loss = 0.001787
grad_step = 000298, loss = 0.001787
grad_step = 000299, loss = 0.001788
grad_step = 000300, loss = 0.001792
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001800
grad_step = 000302, loss = 0.001813
grad_step = 000303, loss = 0.001836
grad_step = 000304, loss = 0.001876
grad_step = 000305, loss = 0.001930
grad_step = 000306, loss = 0.002011
grad_step = 000307, loss = 0.002074
grad_step = 000308, loss = 0.002101
grad_step = 000309, loss = 0.002038
grad_step = 000310, loss = 0.001914
grad_step = 000311, loss = 0.001810
grad_step = 000312, loss = 0.001797
grad_step = 000313, loss = 0.001846
grad_step = 000314, loss = 0.001891
grad_step = 000315, loss = 0.001883
grad_step = 000316, loss = 0.001840
grad_step = 000317, loss = 0.001814
grad_step = 000318, loss = 0.001816
grad_step = 000319, loss = 0.001825
grad_step = 000320, loss = 0.001812
grad_step = 000321, loss = 0.001795
grad_step = 000322, loss = 0.001793
grad_step = 000323, loss = 0.001807
grad_step = 000324, loss = 0.001815
grad_step = 000325, loss = 0.001795
grad_step = 000326, loss = 0.001772
grad_step = 000327, loss = 0.001767
grad_step = 000328, loss = 0.001782
grad_step = 000329, loss = 0.001796
grad_step = 000330, loss = 0.001789
grad_step = 000331, loss = 0.001774
grad_step = 000332, loss = 0.001764
grad_step = 000333, loss = 0.001766
grad_step = 000334, loss = 0.001773
grad_step = 000335, loss = 0.001775
grad_step = 000336, loss = 0.001771
grad_step = 000337, loss = 0.001765
grad_step = 000338, loss = 0.001763
grad_step = 000339, loss = 0.001764
grad_step = 000340, loss = 0.001765
grad_step = 000341, loss = 0.001765
grad_step = 000342, loss = 0.001762
grad_step = 000343, loss = 0.001758
grad_step = 000344, loss = 0.001755
grad_step = 000345, loss = 0.001755
grad_step = 000346, loss = 0.001756
grad_step = 000347, loss = 0.001758
grad_step = 000348, loss = 0.001759
grad_step = 000349, loss = 0.001758
grad_step = 000350, loss = 0.001756
grad_step = 000351, loss = 0.001754
grad_step = 000352, loss = 0.001751
grad_step = 000353, loss = 0.001749
grad_step = 000354, loss = 0.001747
grad_step = 000355, loss = 0.001746
grad_step = 000356, loss = 0.001746
grad_step = 000357, loss = 0.001746
grad_step = 000358, loss = 0.001746
grad_step = 000359, loss = 0.001746
grad_step = 000360, loss = 0.001746
grad_step = 000361, loss = 0.001747
grad_step = 000362, loss = 0.001748
grad_step = 000363, loss = 0.001749
grad_step = 000364, loss = 0.001751
grad_step = 000365, loss = 0.001756
grad_step = 000366, loss = 0.001764
grad_step = 000367, loss = 0.001777
grad_step = 000368, loss = 0.001801
grad_step = 000369, loss = 0.001830
grad_step = 000370, loss = 0.001870
grad_step = 000371, loss = 0.001902
grad_step = 000372, loss = 0.001925
grad_step = 000373, loss = 0.001934
grad_step = 000374, loss = 0.001958
grad_step = 000375, loss = 0.001982
grad_step = 000376, loss = 0.002043
grad_step = 000377, loss = 0.001985
grad_step = 000378, loss = 0.001885
grad_step = 000379, loss = 0.001761
grad_step = 000380, loss = 0.001751
grad_step = 000381, loss = 0.001834
grad_step = 000382, loss = 0.001878
grad_step = 000383, loss = 0.001838
grad_step = 000384, loss = 0.001751
grad_step = 000385, loss = 0.001737
grad_step = 000386, loss = 0.001789
grad_step = 000387, loss = 0.001810
grad_step = 000388, loss = 0.001776
grad_step = 000389, loss = 0.001732
grad_step = 000390, loss = 0.001740
grad_step = 000391, loss = 0.001776
grad_step = 000392, loss = 0.001776
grad_step = 000393, loss = 0.001749
grad_step = 000394, loss = 0.001728
grad_step = 000395, loss = 0.001738
grad_step = 000396, loss = 0.001753
grad_step = 000397, loss = 0.001748
grad_step = 000398, loss = 0.001731
grad_step = 000399, loss = 0.001722
grad_step = 000400, loss = 0.001729
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001738
grad_step = 000402, loss = 0.001735
grad_step = 000403, loss = 0.001725
grad_step = 000404, loss = 0.001720
grad_step = 000405, loss = 0.001724
grad_step = 000406, loss = 0.001731
grad_step = 000407, loss = 0.001731
grad_step = 000408, loss = 0.001726
grad_step = 000409, loss = 0.001723
grad_step = 000410, loss = 0.001728
grad_step = 000411, loss = 0.001738
grad_step = 000412, loss = 0.001748
grad_step = 000413, loss = 0.001759
grad_step = 000414, loss = 0.001771
grad_step = 000415, loss = 0.001792
grad_step = 000416, loss = 0.001825
grad_step = 000417, loss = 0.001841
grad_step = 000418, loss = 0.001847
grad_step = 000419, loss = 0.001814
grad_step = 000420, loss = 0.001770
grad_step = 000421, loss = 0.001732
grad_step = 000422, loss = 0.001717
grad_step = 000423, loss = 0.001723
grad_step = 000424, loss = 0.001735
grad_step = 000425, loss = 0.001742
grad_step = 000426, loss = 0.001745
grad_step = 000427, loss = 0.001741
grad_step = 000428, loss = 0.001728
grad_step = 000429, loss = 0.001711
grad_step = 000430, loss = 0.001702
grad_step = 000431, loss = 0.001707
grad_step = 000432, loss = 0.001717
grad_step = 000433, loss = 0.001724
grad_step = 000434, loss = 0.001723
grad_step = 000435, loss = 0.001717
grad_step = 000436, loss = 0.001711
grad_step = 000437, loss = 0.001709
grad_step = 000438, loss = 0.001711
grad_step = 000439, loss = 0.001710
grad_step = 000440, loss = 0.001707
grad_step = 000441, loss = 0.001701
grad_step = 000442, loss = 0.001696
grad_step = 000443, loss = 0.001694
grad_step = 000444, loss = 0.001695
grad_step = 000445, loss = 0.001696
grad_step = 000446, loss = 0.001695
grad_step = 000447, loss = 0.001693
grad_step = 000448, loss = 0.001691
grad_step = 000449, loss = 0.001690
grad_step = 000450, loss = 0.001690
grad_step = 000451, loss = 0.001691
grad_step = 000452, loss = 0.001692
grad_step = 000453, loss = 0.001695
grad_step = 000454, loss = 0.001698
grad_step = 000455, loss = 0.001704
grad_step = 000456, loss = 0.001715
grad_step = 000457, loss = 0.001736
grad_step = 000458, loss = 0.001769
grad_step = 000459, loss = 0.001823
grad_step = 000460, loss = 0.001890
grad_step = 000461, loss = 0.001962
grad_step = 000462, loss = 0.001994
grad_step = 000463, loss = 0.001961
grad_step = 000464, loss = 0.001869
grad_step = 000465, loss = 0.001767
grad_step = 000466, loss = 0.001716
grad_step = 000467, loss = 0.001727
grad_step = 000468, loss = 0.001745
grad_step = 000469, loss = 0.001747
grad_step = 000470, loss = 0.001745
grad_step = 000471, loss = 0.001755
grad_step = 000472, loss = 0.001762
grad_step = 000473, loss = 0.001733
grad_step = 000474, loss = 0.001697
grad_step = 000475, loss = 0.001677
grad_step = 000476, loss = 0.001690
grad_step = 000477, loss = 0.001718
grad_step = 000478, loss = 0.001725
grad_step = 000479, loss = 0.001710
grad_step = 000480, loss = 0.001682
grad_step = 000481, loss = 0.001669
grad_step = 000482, loss = 0.001675
grad_step = 000483, loss = 0.001689
grad_step = 000484, loss = 0.001697
grad_step = 000485, loss = 0.001689
grad_step = 000486, loss = 0.001674
grad_step = 000487, loss = 0.001664
grad_step = 000488, loss = 0.001665
grad_step = 000489, loss = 0.001673
grad_step = 000490, loss = 0.001678
grad_step = 000491, loss = 0.001675
grad_step = 000492, loss = 0.001668
grad_step = 000493, loss = 0.001661
grad_step = 000494, loss = 0.001659
grad_step = 000495, loss = 0.001662
grad_step = 000496, loss = 0.001665
grad_step = 000497, loss = 0.001665
grad_step = 000498, loss = 0.001662
grad_step = 000499, loss = 0.001657
grad_step = 000500, loss = 0.001654
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001654
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

  date_run                              2020-05-12 01:15:47.896674
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.270531
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 01:15:47.904129
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.180292
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 01:15:47.911082
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.164965
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 01:15:47.917388
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.7396
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
0   2020-05-12 01:15:14.652642  ...    mean_absolute_error
1   2020-05-12 01:15:14.659209  ...     mean_squared_error
2   2020-05-12 01:15:14.665346  ...  median_absolute_error
3   2020-05-12 01:15:14.670384  ...               r2_score
4   2020-05-12 01:15:24.169907  ...    mean_absolute_error
5   2020-05-12 01:15:24.174179  ...     mean_squared_error
6   2020-05-12 01:15:24.177848  ...  median_absolute_error
7   2020-05-12 01:15:24.181517  ...               r2_score
8   2020-05-12 01:15:47.896674  ...    mean_absolute_error
9   2020-05-12 01:15:47.904129  ...     mean_squared_error
10  2020-05-12 01:15:47.911082  ...  median_absolute_error
11  2020-05-12 01:15:47.917388  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 319129.21it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 411447.96it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 569394.30it/s] 36%|███▌      | 3522560/9912422 [00:00<00:07, 804453.92it/s] 77%|███████▋  | 7675904/9912422 [00:00<00:01, 1137456.34it/s]9920512it [00:00, 10971164.17it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 144160.28it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 332936.21it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 428602.31it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 591989.15it/s]1654784it [00:00, 2671929.10it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53144.85it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355e2b38> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52d2d37da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355ee7b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355a6e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355e2b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355a6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355e2b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52e7f9fdd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355ea9b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52e7f9fdd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53355ea9b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f03418091d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=70c7853473cea17977b600b38271951211ef95189a1ec7d5e537f49b1ea7d107
  Stored in directory: /tmp/pip-ephem-wheel-cache-ycn57wg7/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f033968af98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 49s
   57344/17464789 [..............................] - ETA: 42s
  106496/17464789 [..............................] - ETA: 34s
  245760/17464789 [..............................] - ETA: 19s
  507904/17464789 [..............................] - ETA: 11s
 1032192/17464789 [>.............................] - ETA: 6s 
 2072576/17464789 [==>...........................] - ETA: 3s
 4153344/17464789 [======>.......................] - ETA: 1s
 7102464/17464789 [===========>..................] - ETA: 0s
 9920512/17464789 [================>.............] - ETA: 0s
12820480/17464789 [=====================>........] - ETA: 0s
15851520/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 01:17:22.002894: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 01:17:22.006773: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-12 01:17:22.006909: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562e008b6fa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 01:17:22.006924: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.2033 - accuracy: 0.4650
 2000/25000 [=>............................] - ETA: 9s - loss: 8.0270 - accuracy: 0.4765 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9120 - accuracy: 0.4840
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8775 - accuracy: 0.4863
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8476 - accuracy: 0.4882
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8200 - accuracy: 0.4900
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8441 - accuracy: 0.4884
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8104 - accuracy: 0.4906
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.8165 - accuracy: 0.4902
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7832 - accuracy: 0.4924
11000/25000 [============>.................] - ETA: 4s - loss: 7.7963 - accuracy: 0.4915
12000/25000 [=============>................] - ETA: 4s - loss: 7.7420 - accuracy: 0.4951
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7362 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7247 - accuracy: 0.4962
15000/25000 [=================>............] - ETA: 3s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6752 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6475 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
25000/25000 [==============================] - 9s 369us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 01:17:38.574082
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 01:17:38.574082  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 01:17:45.439122: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 01:17:45.444049: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-12 01:17:45.444194: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b07308e2f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 01:17:45.444210: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f96f41debe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1425 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.1555 - val_crf_viterbi_accuracy: 0.2800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f96cfcdde48> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7356 - accuracy: 0.4955
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7484 - accuracy: 0.4947 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6644 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6615 - accuracy: 0.5003
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 4s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 4s - loss: 7.6245 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6239 - accuracy: 0.5028
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6693 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6803 - accuracy: 0.4991
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6866 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6542 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 9s 366us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f96c40b1860> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<37:08:29, 6.45kB/s].vector_cache/glove.6B.zip:   0%|          | 303k/862M [00:01<26:00:52, 9.20kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.92M/862M [00:01<18:08:08, 13.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<12:31:49, 18.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<8:44:25, 26.8kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<6:02:20, 38.3kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:01<4:10:24, 54.7kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:01<2:52:54, 78.2kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<2:03:11, 110kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<1:26:18, 156kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<10:46:22, 20.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<7:31:46, 29.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<5:18:16, 42.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<3:44:09, 59.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:06<2:36:32, 85.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<1:55:30, 115kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<1:21:37, 163kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<58:57, 225kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<41:50, 316kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<31:17, 421kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<22:28, 586kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<17:46, 737kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<13:09, 994kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<11:12, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<08:26, 1.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<07:58, 1.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<06:25, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<05:03, 2.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<6:17:24, 34.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<4:24:37, 48.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:21<3:06:19, 69.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:21<2:09:57, 98.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<1:47:47, 119kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<1:16:27, 167kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<55:09, 231kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<39:08, 325kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<29:19, 431kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<21:07, 598kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<16:46, 750kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<12:14, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:35, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<07:37, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:32<05:54, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:06, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:01, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<04:04, 3.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<6:06:33, 33.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<4:15:26, 48.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<3:11:46, 63.9kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<2:16:05, 90.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<1:34:58, 128kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<1:14:45, 163kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<53:19, 228kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<38:56, 311kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<28:19, 428kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<21:30, 560kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<16:04, 749kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<12:58, 923kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<10:07, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:49, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<07:13, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:47, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<05:49, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:34, 2.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<5:41:31, 34.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<3:59:21, 49.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<2:48:24, 69.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<1:59:01, 98.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:23:52, 139kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<1:00:14, 193kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<43:03, 269kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<31:42, 364kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<22:50, 504kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<17:40, 648kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<13:05, 875kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<10:51, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<08:08, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<07:31, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<06:03, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<04:43, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<5:33:51, 33.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<3:52:41, 48.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<2:49:53, 66.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<1:59:48, 93.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<1:25:03, 131kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<1:00:09, 185kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<43:35, 254kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<31:26, 352kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<23:31, 468kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<17:09, 641kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<12:05, 906kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<13:59, 781kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<10:37, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<09:01, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:29, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<05:27, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:17, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<5:11:22, 34.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<3:38:05, 49.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<2:33:31, 69.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<1:46:55, 99.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<2:23:32, 74.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:40:48, 105kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<1:11:54, 147kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<50:47, 207kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<37:05, 283kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<26:26, 396kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<20:04, 519kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<14:31, 716kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<11:46, 879kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<08:44, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<07:45, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:50, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:36, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<5:03:02, 33.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<3:30:47, 48.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<5:20:45, 31.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<3:45:10, 45.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<2:36:50, 64.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<2:03:37, 81.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:26:54, 116kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<1:02:05, 161kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<43:54, 228kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<32:08, 310kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<22:56, 433kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<17:35, 562kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<12:46, 773kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<10:30, 935kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<07:46, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<06:59, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<05:22, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:19, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:32, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:38, 2.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<4:25:51, 36.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<3:06:14, 51.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<2:09:51, 73.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<10:48:26, 14.8kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<7:33:10, 21.1kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<5:17:25, 29.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<3:42:43, 42.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<2:36:18, 60.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:49:41, 85.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<1:17:50, 120kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<54:54, 170kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<39:39, 234kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<28:06, 330kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<19:50, 465kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<9:20:25, 16.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<6:31:54, 23.5kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<4:34:21, 33.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<3:12:30, 47.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<2:15:13, 67.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<1:35:03, 95.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<1:07:30, 133kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<47:33, 189kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<34:33, 259kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<24:40, 362kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<18:32, 479kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<13:22, 663kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<10:43, 821kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<07:51, 1.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<06:54, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<05:20, 1.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:03, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:55, 2.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:07, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:14, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:00, 2.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:29, 3.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<4:08:59, 34.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<2:54:05, 48.3kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<2:02:27, 68.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<1:26:20, 96.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<1:00:45, 137kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<43:33, 190kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<31:00, 267kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<22:48, 360kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<16:16, 503kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<12:39, 643kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<09:14, 880kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<07:42, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<05:49, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<05:18, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:05, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:14, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<3:56:36, 33.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<2:44:22, 48.0kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<9:51:27, 13.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<6:54:13, 19.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<4:48:51, 27.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<3:22:42, 38.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<2:21:54, 54.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<1:39:54, 77.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<1:10:30, 109kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<50:00, 154kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<34:53, 218kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<7:59:41, 15.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<5:35:41, 22.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<3:54:23, 32.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<2:44:08, 45.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<1:55:14, 64.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<1:20:58, 92.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<57:23, 129kB/s]   .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<40:25, 183kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<29:16, 250kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<20:59, 349kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<15:40, 464kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<11:20, 639kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<09:00, 799kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<06:35, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<05:44, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<04:34, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:14, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:30, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:45, 2.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<3:20:32, 35.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<2:19:59, 49.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<1:38:29, 70.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:09:21, 99.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<48:47, 141kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<34:02, 200kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<7:11:39, 15.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<5:02:01, 22.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<3:30:40, 32.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<2:27:31, 45.6kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<1:43:29, 64.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<1:12:32, 91.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<51:28, 128kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<36:26, 181kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<26:19, 248kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<19:06, 338kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<13:51, 466kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<10:33, 605kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<07:40, 831kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<06:20, 996kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<04:44, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<3:03:27, 34.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<2:07:14, 48.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<1:35:39, 64.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<1:07:22, 92.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<47:36, 129kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<33:42, 182kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<24:16, 250kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<17:13, 352kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<12:55, 465kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<09:22, 640kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<07:25, 800kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<05:25, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:43, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:36, 1.62MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:25, 1.70MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:46, 2.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:47, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:14, 2.55MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:50, 3.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<2:46:19, 34.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<1:55:45, 48.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<1:22:12, 68.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<58:00, 96.8kB/s]  .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<40:11, 138kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<1:57:53, 47.1kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<1:22:50, 66.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<58:09, 94.2kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<41:28, 132kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<28:45, 188kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<29:18, 185kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<21:06, 256kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<14:38, 365kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<33:17, 160kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<24:02, 222kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:55<16:41, 316kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<20:53, 253kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<15:06, 349kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<10:45, 486kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<2:40:14, 32.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<1:51:24, 46.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<1:18:40, 65.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<54:32, 93.6kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<42:18, 120kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<30:01, 170kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<21:30, 234kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<15:26, 325kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<11:25, 434kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<08:24, 589kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<06:32, 747kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<05:00, 974kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:08<03:31, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<05:03, 953kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<03:56, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:57, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<2:12:58, 35.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<1:32:01, 51.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<1:07:51, 69.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<47:45, 98.5kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<33:03, 140kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<33:51, 137kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<23:59, 193kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<16:38, 275kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<16:43, 273kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<12:02, 379kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<09:00, 500kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<06:39, 675kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:15, 842kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<04:00, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:25, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:45, 1.58MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:07, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<1:56:21, 37.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<1:20:25, 53.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<56:33, 75.2kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<4:56:59, 14.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<3:28:19, 20.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<2:23:59, 29.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<1:44:00, 40.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<1:13:05, 57.2kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<50:26, 81.6kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<54:19, 75.8kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<38:18, 107kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<27:01, 150kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<19:24, 208kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<13:24, 297kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<16:23, 243kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<11:46, 337kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<08:42, 449kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<06:22, 611kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<04:58, 771kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:47, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<03:10, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:45, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:56, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:56, 937kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<03:01, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:37, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<02:15, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:35, 2.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<18:52, 189kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<13:28, 264kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<09:46, 357kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<08:44, 399kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<06:09, 563kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<04:33, 756kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:28:58, 38.8kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<1:01:31, 55.0kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<43:24, 77.8kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<29:53, 111kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<25:00, 132kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<17:51, 185kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<12:18, 264kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<12:25, 261kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<08:55, 363kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<06:36, 481kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<05:02, 629kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<03:30, 891kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<04:24, 704kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<03:27, 898kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<02:24, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<03:17, 923kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:45, 1.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:02, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:22:44, 36.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<56:52, 51.7kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<41:42, 70.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<29:27, 99.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<20:15, 141kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<16:31, 173kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<11:42, 243kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<08:02, 347kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:05:48, 42.4kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<46:16, 60.1kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<31:42, 85.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<30:32, 89.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<21:34, 126kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<15:08, 175kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<10:53, 243kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<07:28, 346kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<10:36, 243kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<07:42, 334kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<05:17, 475kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<10:45, 233kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<07:43, 324kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<05:19, 461kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<06:13, 393kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<04:32, 536kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<03:14, 740kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:09:13, 34.7kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<47:17, 49.5kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<35:12, 66.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<24:53, 93.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<17:02, 133kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<13:58, 162kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<09:55, 228kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<07:04, 310kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<05:07, 427kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<03:48, 559kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<02:54, 732kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<01:59, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<23:57, 85.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<16:53, 121kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<11:45, 169kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<08:19, 238kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<05:56, 323kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<04:18, 444kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<03:02, 618kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<55:18, 33.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<37:34, 48.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<27:25, 66.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<19:15, 93.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<13:17, 131kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<09:24, 184kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<06:36, 253kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<04:43, 352kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:25, 468kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<02:29, 639kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:54, 800kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:33, 979kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<01:03, 1.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<03:14, 453kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<02:21, 619kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<01:40, 850kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<41:13, 34.5kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<27:51, 49.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<19:53, 68.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<13:58, 96.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<09:23, 138kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<08:02, 160kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<05:42, 224kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<03:49, 319kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<06:15, 195kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<04:32, 267kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<03:02, 380kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<03:03, 376kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<02:13, 514kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:37, 662kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<01:17, 834kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:52, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<01:19, 761kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<01:03, 952kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:45, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<27:44, 34.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<18:22, 49.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<13:08, 68.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<09:11, 97.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<06:07, 136kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<04:18, 191kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<02:49, 272kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<02:42, 282kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<01:56, 389kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<01:21, 513kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:59, 696kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:38, 985kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:55, 679kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:41, 899kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:30, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:24, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:17, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<14:37, 35.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<09:13, 50.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<06:26, 69.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<04:28, 98.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<02:43, 140kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<02:54, 130kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<02:01, 183kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<01:13, 251kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:51, 348kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:30, 463kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:22, 625kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:12, 886kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:20, 491kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:14, 668kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:23<00:06, 946kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:19, 319kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:12, 439kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:05, 610kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:44, 34.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:10, 48.6kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 755/400000 [00:00<00:52, 7548.91it/s]  0%|          | 1485/400000 [00:00<00:53, 7470.35it/s]  1%|          | 2181/400000 [00:00<00:54, 7309.26it/s]  1%|          | 2846/400000 [00:00<00:55, 7095.97it/s]  1%|          | 3514/400000 [00:00<00:56, 6964.47it/s]  1%|          | 4184/400000 [00:00<00:57, 6881.93it/s]  1%|          | 4840/400000 [00:00<00:58, 6780.27it/s]  1%|▏         | 5583/400000 [00:00<00:56, 6959.63it/s]  2%|▏         | 6312/400000 [00:00<00:55, 7055.28it/s]  2%|▏         | 7110/400000 [00:01<00:53, 7308.21it/s]  2%|▏         | 7862/400000 [00:01<00:53, 7368.65it/s]  2%|▏         | 8587/400000 [00:01<00:54, 7202.58it/s]  2%|▏         | 9339/400000 [00:01<00:53, 7292.08it/s]  3%|▎         | 10066/400000 [00:01<00:53, 7283.40it/s]  3%|▎         | 10814/400000 [00:01<00:53, 7339.92it/s]  3%|▎         | 11546/400000 [00:01<00:53, 7243.07it/s]  3%|▎         | 12269/400000 [00:01<00:56, 6838.66it/s]  3%|▎         | 12996/400000 [00:01<00:55, 6962.39it/s]  3%|▎         | 13751/400000 [00:01<00:54, 7126.45it/s]  4%|▎         | 14536/400000 [00:02<00:52, 7327.18it/s]  4%|▍         | 15312/400000 [00:02<00:51, 7449.89it/s]  4%|▍         | 16061/400000 [00:02<00:54, 7082.35it/s]  4%|▍         | 16776/400000 [00:02<00:54, 7056.49it/s]  4%|▍         | 17486/400000 [00:02<00:54, 7046.09it/s]  5%|▍         | 18255/400000 [00:02<00:52, 7227.48it/s]  5%|▍         | 19012/400000 [00:02<00:52, 7325.48it/s]  5%|▍         | 19748/400000 [00:02<00:52, 7215.95it/s]  5%|▌         | 20472/400000 [00:02<00:52, 7188.39it/s]  5%|▌         | 21193/400000 [00:02<00:52, 7147.80it/s]  5%|▌         | 21909/400000 [00:03<00:53, 7012.61it/s]  6%|▌         | 22612/400000 [00:03<00:55, 6841.89it/s]  6%|▌         | 23299/400000 [00:03<00:56, 6716.61it/s]  6%|▌         | 24011/400000 [00:03<00:55, 6821.89it/s]  6%|▌         | 24830/400000 [00:03<00:52, 7180.86it/s]  6%|▋         | 25636/400000 [00:03<00:50, 7423.72it/s]  7%|▋         | 26406/400000 [00:03<00:49, 7501.57it/s]  7%|▋         | 27161/400000 [00:03<00:51, 7253.75it/s]  7%|▋         | 27892/400000 [00:03<00:54, 6868.35it/s]  7%|▋         | 28657/400000 [00:04<00:52, 7084.44it/s]  7%|▋         | 29503/400000 [00:04<00:49, 7445.87it/s]  8%|▊         | 30343/400000 [00:04<00:47, 7708.05it/s]  8%|▊         | 31155/400000 [00:04<00:47, 7825.08it/s]  8%|▊         | 31964/400000 [00:04<00:46, 7902.33it/s]  8%|▊         | 32760/400000 [00:04<00:46, 7891.66it/s]  8%|▊         | 33584/400000 [00:04<00:45, 7991.60it/s]  9%|▊         | 34386/400000 [00:04<00:45, 7999.77it/s]  9%|▉         | 35188/400000 [00:04<00:46, 7865.75it/s]  9%|▉         | 36012/400000 [00:04<00:45, 7971.86it/s]  9%|▉         | 36829/400000 [00:05<00:45, 8029.86it/s]  9%|▉         | 37634/400000 [00:05<00:45, 7997.34it/s] 10%|▉         | 38435/400000 [00:05<00:45, 7973.37it/s] 10%|▉         | 39233/400000 [00:05<00:48, 7493.26it/s] 10%|▉         | 39989/400000 [00:05<00:51, 7039.62it/s] 10%|█         | 40704/400000 [00:05<00:52, 6889.58it/s] 10%|█         | 41435/400000 [00:05<00:51, 7008.71it/s] 11%|█         | 42154/400000 [00:05<00:50, 7060.74it/s] 11%|█         | 42925/400000 [00:05<00:49, 7231.42it/s] 11%|█         | 43653/400000 [00:05<00:51, 6983.38it/s] 11%|█         | 44395/400000 [00:06<00:50, 7107.71it/s] 11%|█▏        | 45142/400000 [00:06<00:49, 7211.36it/s] 11%|█▏        | 45867/400000 [00:06<00:49, 7127.46it/s] 12%|█▏        | 46625/400000 [00:06<00:48, 7255.23it/s] 12%|█▏        | 47413/400000 [00:06<00:47, 7431.18it/s] 12%|█▏        | 48199/400000 [00:06<00:46, 7552.53it/s] 12%|█▏        | 48984/400000 [00:06<00:45, 7636.82it/s] 12%|█▏        | 49750/400000 [00:06<00:46, 7493.75it/s] 13%|█▎        | 50510/400000 [00:06<00:46, 7522.79it/s] 13%|█▎        | 51299/400000 [00:07<00:45, 7627.48it/s] 13%|█▎        | 52064/400000 [00:07<00:47, 7359.64it/s] 13%|█▎        | 52860/400000 [00:07<00:46, 7529.05it/s] 13%|█▎        | 53629/400000 [00:07<00:45, 7575.92it/s] 14%|█▎        | 54435/400000 [00:07<00:44, 7714.23it/s] 14%|█▍        | 55267/400000 [00:07<00:43, 7885.44it/s] 14%|█▍        | 56117/400000 [00:07<00:42, 8057.79it/s] 14%|█▍        | 56952/400000 [00:07<00:42, 8142.21it/s] 14%|█▍        | 57769/400000 [00:07<00:42, 8001.82it/s] 15%|█▍        | 58572/400000 [00:07<00:43, 7884.52it/s] 15%|█▍        | 59363/400000 [00:08<00:45, 7556.34it/s] 15%|█▌        | 60123/400000 [00:08<00:46, 7246.97it/s] 15%|█▌        | 60854/400000 [00:08<00:48, 7052.84it/s] 15%|█▌        | 61565/400000 [00:08<00:48, 7028.76it/s] 16%|█▌        | 62307/400000 [00:08<00:47, 7139.51it/s] 16%|█▌        | 63024/400000 [00:08<00:47, 7141.09it/s] 16%|█▌        | 63741/400000 [00:08<00:47, 7140.73it/s] 16%|█▌        | 64457/400000 [00:08<00:46, 7144.26it/s] 16%|█▋        | 65173/400000 [00:08<00:47, 7063.15it/s] 16%|█▋        | 65943/400000 [00:08<00:46, 7241.38it/s] 17%|█▋        | 66707/400000 [00:09<00:45, 7354.38it/s] 17%|█▋        | 67497/400000 [00:09<00:44, 7507.66it/s] 17%|█▋        | 68259/400000 [00:09<00:44, 7539.22it/s] 17%|█▋        | 69015/400000 [00:09<00:45, 7343.54it/s] 17%|█▋        | 69759/400000 [00:09<00:44, 7370.72it/s] 18%|█▊        | 70498/400000 [00:09<00:44, 7338.27it/s] 18%|█▊        | 71275/400000 [00:09<00:44, 7461.99it/s] 18%|█▊        | 72029/400000 [00:09<00:43, 7483.53it/s] 18%|█▊        | 72779/400000 [00:09<00:47, 6893.14it/s] 18%|█▊        | 73479/400000 [00:10<00:48, 6744.61it/s] 19%|█▊        | 74166/400000 [00:10<00:48, 6780.58it/s] 19%|█▊        | 74905/400000 [00:10<00:46, 6952.58it/s] 19%|█▉        | 75627/400000 [00:10<00:46, 7030.40it/s] 19%|█▉        | 76334/400000 [00:10<00:47, 6788.66it/s] 19%|█▉        | 77080/400000 [00:10<00:46, 6975.90it/s] 19%|█▉        | 77861/400000 [00:10<00:44, 7202.29it/s] 20%|█▉        | 78587/400000 [00:10<00:45, 7043.78it/s] 20%|█▉        | 79296/400000 [00:10<00:46, 6921.81it/s] 20%|█▉        | 79992/400000 [00:10<00:46, 6841.07it/s] 20%|██        | 80679/400000 [00:11<00:46, 6813.45it/s] 20%|██        | 81363/400000 [00:11<00:47, 6762.17it/s] 21%|██        | 82043/400000 [00:11<00:46, 6772.51it/s] 21%|██        | 82734/400000 [00:11<00:46, 6812.81it/s] 21%|██        | 83434/400000 [00:11<00:46, 6867.14it/s] 21%|██        | 84176/400000 [00:11<00:44, 7022.33it/s] 21%|██        | 84970/400000 [00:11<00:43, 7272.90it/s] 21%|██▏       | 85701/400000 [00:11<00:43, 7248.31it/s] 22%|██▏       | 86497/400000 [00:11<00:42, 7445.76it/s] 22%|██▏       | 87299/400000 [00:11<00:41, 7607.36it/s] 22%|██▏       | 88063/400000 [00:12<00:42, 7260.26it/s] 22%|██▏       | 88795/400000 [00:12<00:43, 7175.42it/s] 22%|██▏       | 89545/400000 [00:12<00:42, 7268.67it/s] 23%|██▎       | 90275/400000 [00:12<00:42, 7275.72it/s] 23%|██▎       | 91022/400000 [00:12<00:42, 7331.79it/s] 23%|██▎       | 91764/400000 [00:12<00:41, 7357.36it/s] 23%|██▎       | 92510/400000 [00:12<00:41, 7387.26it/s] 23%|██▎       | 93273/400000 [00:12<00:41, 7456.76it/s] 24%|██▎       | 94020/400000 [00:12<00:41, 7452.22it/s] 24%|██▎       | 94791/400000 [00:12<00:40, 7526.01it/s] 24%|██▍       | 95545/400000 [00:13<00:40, 7440.17it/s] 24%|██▍       | 96290/400000 [00:13<00:42, 7156.28it/s] 24%|██▍       | 97009/400000 [00:13<00:43, 6976.80it/s] 24%|██▍       | 97747/400000 [00:13<00:42, 7091.01it/s] 25%|██▍       | 98518/400000 [00:13<00:41, 7264.17it/s] 25%|██▍       | 99303/400000 [00:13<00:40, 7430.12it/s] 25%|██▌       | 100049/400000 [00:13<00:41, 7197.89it/s] 25%|██▌       | 100773/400000 [00:13<00:41, 7132.30it/s] 25%|██▌       | 101529/400000 [00:13<00:41, 7254.94it/s] 26%|██▌       | 102257/400000 [00:14<00:41, 7235.22it/s] 26%|██▌       | 103027/400000 [00:14<00:40, 7367.89it/s] 26%|██▌       | 103769/400000 [00:14<00:40, 7381.35it/s] 26%|██▌       | 104509/400000 [00:14<00:40, 7320.54it/s] 26%|██▋       | 105256/400000 [00:14<00:40, 7362.96it/s] 27%|██▋       | 106005/400000 [00:14<00:39, 7398.59it/s] 27%|██▋       | 106769/400000 [00:14<00:39, 7468.30it/s] 27%|██▋       | 107560/400000 [00:14<00:38, 7595.03it/s] 27%|██▋       | 108338/400000 [00:14<00:38, 7649.37it/s] 27%|██▋       | 109104/400000 [00:14<00:38, 7574.11it/s] 27%|██▋       | 109863/400000 [00:15<00:38, 7545.93it/s] 28%|██▊       | 110630/400000 [00:15<00:38, 7581.60it/s] 28%|██▊       | 111418/400000 [00:15<00:37, 7666.70it/s] 28%|██▊       | 112186/400000 [00:15<00:39, 7299.86it/s] 28%|██▊       | 112921/400000 [00:15<00:40, 7006.43it/s] 28%|██▊       | 113682/400000 [00:15<00:39, 7175.21it/s] 29%|██▊       | 114442/400000 [00:15<00:39, 7295.71it/s] 29%|██▉       | 115211/400000 [00:15<00:38, 7407.82it/s] 29%|██▉       | 115955/400000 [00:15<00:39, 7175.68it/s] 29%|██▉       | 116677/400000 [00:15<00:40, 7017.61it/s] 29%|██▉       | 117383/400000 [00:16<00:40, 6904.39it/s] 30%|██▉       | 118112/400000 [00:16<00:40, 7015.06it/s] 30%|██▉       | 118892/400000 [00:16<00:38, 7233.24it/s] 30%|██▉       | 119619/400000 [00:16<00:39, 7176.49it/s] 30%|███       | 120349/400000 [00:16<00:38, 7212.17it/s] 30%|███       | 121072/400000 [00:16<00:39, 7052.94it/s] 30%|███       | 121780/400000 [00:16<00:40, 6938.94it/s] 31%|███       | 122476/400000 [00:16<00:40, 6853.76it/s] 31%|███       | 123163/400000 [00:16<00:41, 6699.43it/s] 31%|███       | 123835/400000 [00:17<00:42, 6555.51it/s] 31%|███       | 124539/400000 [00:17<00:41, 6693.70it/s] 31%|███▏      | 125288/400000 [00:17<00:39, 6913.28it/s] 32%|███▏      | 126036/400000 [00:17<00:38, 7071.81it/s] 32%|███▏      | 126801/400000 [00:17<00:37, 7235.50it/s] 32%|███▏      | 127585/400000 [00:17<00:36, 7405.00it/s] 32%|███▏      | 128394/400000 [00:17<00:35, 7595.98it/s] 32%|███▏      | 129195/400000 [00:17<00:35, 7714.66it/s] 32%|███▏      | 129970/400000 [00:17<00:34, 7724.96it/s] 33%|███▎      | 130745/400000 [00:17<00:34, 7718.42it/s] 33%|███▎      | 131519/400000 [00:18<00:35, 7477.35it/s] 33%|███▎      | 132270/400000 [00:18<00:36, 7418.12it/s] 33%|███▎      | 133038/400000 [00:18<00:35, 7492.79it/s] 33%|███▎      | 133821/400000 [00:18<00:35, 7589.40it/s] 34%|███▎      | 134582/400000 [00:18<00:35, 7478.42it/s] 34%|███▍      | 135332/400000 [00:18<00:36, 7174.66it/s] 34%|███▍      | 136054/400000 [00:18<00:36, 7146.16it/s] 34%|███▍      | 136894/400000 [00:18<00:35, 7479.60it/s] 34%|███▍      | 137648/400000 [00:18<00:35, 7299.15it/s] 35%|███▍      | 138383/400000 [00:18<00:36, 7129.11it/s] 35%|███▍      | 139198/400000 [00:19<00:35, 7405.35it/s] 35%|███▌      | 140018/400000 [00:19<00:34, 7625.05it/s] 35%|███▌      | 140787/400000 [00:19<00:34, 7600.77it/s] 35%|███▌      | 141567/400000 [00:19<00:33, 7657.28it/s] 36%|███▌      | 142336/400000 [00:19<00:34, 7546.87it/s] 36%|███▌      | 143094/400000 [00:19<00:35, 7268.63it/s] 36%|███▌      | 143825/400000 [00:19<00:36, 7033.39it/s] 36%|███▌      | 144533/400000 [00:19<00:37, 6902.95it/s] 36%|███▋      | 145293/400000 [00:19<00:35, 7097.53it/s] 37%|███▋      | 146038/400000 [00:20<00:35, 7199.08it/s] 37%|███▋      | 146795/400000 [00:20<00:34, 7306.20it/s] 37%|███▋      | 147529/400000 [00:20<00:36, 6971.32it/s] 37%|███▋      | 148341/400000 [00:20<00:34, 7279.69it/s] 37%|███▋      | 149157/400000 [00:20<00:33, 7522.28it/s] 37%|███▋      | 149917/400000 [00:20<00:33, 7402.01it/s] 38%|███▊      | 150663/400000 [00:20<00:34, 7157.12it/s] 38%|███▊      | 151385/400000 [00:20<00:35, 7008.22it/s] 38%|███▊      | 152091/400000 [00:20<00:35, 6936.11it/s] 38%|███▊      | 152863/400000 [00:20<00:34, 7153.41it/s] 38%|███▊      | 153583/400000 [00:21<00:34, 7149.93it/s] 39%|███▊      | 154332/400000 [00:21<00:33, 7246.86it/s] 39%|███▉      | 155059/400000 [00:21<00:33, 7220.76it/s] 39%|███▉      | 155816/400000 [00:21<00:33, 7319.24it/s] 39%|███▉      | 156580/400000 [00:21<00:32, 7410.36it/s] 39%|███▉      | 157323/400000 [00:21<00:33, 7211.63it/s] 40%|███▉      | 158110/400000 [00:21<00:32, 7395.66it/s] 40%|███▉      | 158860/400000 [00:21<00:32, 7417.60it/s] 40%|███▉      | 159604/400000 [00:21<00:32, 7358.48it/s] 40%|████      | 160429/400000 [00:21<00:31, 7600.42it/s] 40%|████      | 161218/400000 [00:22<00:31, 7682.56it/s] 41%|████      | 162042/400000 [00:22<00:30, 7839.98it/s] 41%|████      | 162829/400000 [00:22<00:30, 7792.75it/s] 41%|████      | 163611/400000 [00:22<00:30, 7770.03it/s] 41%|████      | 164390/400000 [00:22<00:30, 7743.14it/s] 41%|████▏     | 165166/400000 [00:22<00:31, 7567.04it/s] 41%|████▏     | 165934/400000 [00:22<00:30, 7599.95it/s] 42%|████▏     | 166706/400000 [00:22<00:30, 7635.52it/s] 42%|████▏     | 167471/400000 [00:22<00:30, 7604.45it/s] 42%|████▏     | 168245/400000 [00:23<00:30, 7641.74it/s] 42%|████▏     | 169025/400000 [00:23<00:30, 7687.13it/s] 42%|████▏     | 169831/400000 [00:23<00:29, 7793.04it/s] 43%|████▎     | 170611/400000 [00:23<00:29, 7777.47it/s] 43%|████▎     | 171390/400000 [00:23<00:29, 7691.37it/s] 43%|████▎     | 172160/400000 [00:23<00:30, 7555.49it/s] 43%|████▎     | 172917/400000 [00:23<00:30, 7405.22it/s] 43%|████▎     | 173659/400000 [00:23<00:30, 7334.10it/s] 44%|████▎     | 174403/400000 [00:23<00:30, 7365.50it/s] 44%|████▍     | 175161/400000 [00:23<00:30, 7428.55it/s] 44%|████▍     | 175923/400000 [00:24<00:29, 7483.69it/s] 44%|████▍     | 176679/400000 [00:24<00:29, 7503.80it/s] 44%|████▍     | 177430/400000 [00:24<00:29, 7497.39it/s] 45%|████▍     | 178208/400000 [00:24<00:29, 7579.45it/s] 45%|████▍     | 179046/400000 [00:24<00:28, 7801.75it/s] 45%|████▍     | 179829/400000 [00:24<00:28, 7807.03it/s] 45%|████▌     | 180612/400000 [00:24<00:28, 7665.17it/s] 45%|████▌     | 181392/400000 [00:24<00:28, 7705.05it/s] 46%|████▌     | 182167/400000 [00:24<00:28, 7716.81it/s] 46%|████▌     | 182982/400000 [00:24<00:27, 7841.74it/s] 46%|████▌     | 183768/400000 [00:25<00:28, 7714.57it/s] 46%|████▌     | 184541/400000 [00:25<00:29, 7289.79it/s] 46%|████▋     | 185276/400000 [00:25<00:30, 7081.79it/s] 46%|████▋     | 185990/400000 [00:25<00:30, 6951.42it/s] 47%|████▋     | 186690/400000 [00:25<00:31, 6765.38it/s] 47%|████▋     | 187371/400000 [00:25<00:31, 6677.50it/s] 47%|████▋     | 188042/400000 [00:25<00:31, 6639.11it/s] 47%|████▋     | 188709/400000 [00:25<00:31, 6640.80it/s] 47%|████▋     | 189375/400000 [00:25<00:31, 6613.11it/s] 48%|████▊     | 190038/400000 [00:25<00:31, 6590.73it/s] 48%|████▊     | 190726/400000 [00:26<00:31, 6672.93it/s] 48%|████▊     | 191462/400000 [00:26<00:30, 6864.08it/s] 48%|████▊     | 192235/400000 [00:26<00:29, 7101.88it/s] 48%|████▊     | 192949/400000 [00:26<00:29, 7014.82it/s] 48%|████▊     | 193654/400000 [00:26<00:29, 6885.24it/s] 49%|████▊     | 194418/400000 [00:26<00:28, 7094.28it/s] 49%|████▉     | 195131/400000 [00:26<00:29, 6846.59it/s] 49%|████▉     | 195845/400000 [00:26<00:29, 6929.87it/s] 49%|████▉     | 196605/400000 [00:26<00:28, 7116.29it/s] 49%|████▉     | 197386/400000 [00:27<00:27, 7309.11it/s] 50%|████▉     | 198188/400000 [00:27<00:26, 7506.34it/s] 50%|████▉     | 198943/400000 [00:27<00:27, 7427.43it/s] 50%|████▉     | 199769/400000 [00:27<00:26, 7657.64it/s] 50%|█████     | 200571/400000 [00:27<00:25, 7760.45it/s] 50%|█████     | 201351/400000 [00:27<00:25, 7717.37it/s] 51%|█████     | 202129/400000 [00:27<00:25, 7733.28it/s] 51%|█████     | 202904/400000 [00:27<00:26, 7332.55it/s] 51%|█████     | 203643/400000 [00:27<00:26, 7274.95it/s] 51%|█████     | 204375/400000 [00:27<00:27, 7237.47it/s] 51%|█████▏    | 205114/400000 [00:28<00:26, 7282.12it/s] 51%|█████▏    | 205850/400000 [00:28<00:26, 7304.77it/s] 52%|█████▏    | 206607/400000 [00:28<00:26, 7381.75it/s] 52%|█████▏    | 207359/400000 [00:28<00:25, 7422.29it/s] 52%|█████▏    | 208136/400000 [00:28<00:25, 7521.14it/s] 52%|█████▏    | 208964/400000 [00:28<00:24, 7731.29it/s] 52%|█████▏    | 209773/400000 [00:28<00:24, 7833.60it/s] 53%|█████▎    | 210559/400000 [00:28<00:24, 7786.19it/s] 53%|█████▎    | 211339/400000 [00:28<00:24, 7726.58it/s] 53%|█████▎    | 212117/400000 [00:28<00:24, 7741.77it/s] 53%|█████▎    | 212900/400000 [00:29<00:24, 7766.08it/s] 53%|█████▎    | 213678/400000 [00:29<00:24, 7681.75it/s] 54%|█████▎    | 214447/400000 [00:29<00:24, 7580.53it/s] 54%|█████▍    | 215206/400000 [00:29<00:24, 7576.12it/s] 54%|█████▍    | 215965/400000 [00:29<00:25, 7304.47it/s] 54%|█████▍    | 216698/400000 [00:29<00:25, 7263.35it/s] 54%|█████▍    | 217455/400000 [00:29<00:24, 7351.15it/s] 55%|█████▍    | 218192/400000 [00:29<00:24, 7309.15it/s] 55%|█████▍    | 218925/400000 [00:29<00:24, 7269.55it/s] 55%|█████▍    | 219653/400000 [00:29<00:24, 7234.21it/s] 55%|█████▌    | 220423/400000 [00:30<00:24, 7366.64it/s] 55%|█████▌    | 221223/400000 [00:30<00:23, 7545.22it/s] 55%|█████▌    | 221980/400000 [00:30<00:23, 7504.63it/s] 56%|█████▌    | 222743/400000 [00:30<00:23, 7540.13it/s] 56%|█████▌    | 223518/400000 [00:30<00:23, 7599.41it/s] 56%|█████▌    | 224343/400000 [00:30<00:22, 7781.35it/s] 56%|█████▋    | 225167/400000 [00:30<00:22, 7910.69it/s] 56%|█████▋    | 225960/400000 [00:30<00:22, 7854.13it/s] 57%|█████▋    | 226747/400000 [00:30<00:22, 7784.65it/s] 57%|█████▋    | 227528/400000 [00:30<00:22, 7790.81it/s] 57%|█████▋    | 228308/400000 [00:31<00:22, 7653.46it/s] 57%|█████▋    | 229090/400000 [00:31<00:22, 7700.84it/s] 57%|█████▋    | 229867/400000 [00:31<00:22, 7717.63it/s] 58%|█████▊    | 230694/400000 [00:31<00:21, 7874.97it/s] 58%|█████▊    | 231483/400000 [00:31<00:22, 7537.66it/s] 58%|█████▊    | 232241/400000 [00:31<00:22, 7480.35it/s] 58%|█████▊    | 232992/400000 [00:31<00:22, 7430.20it/s] 58%|█████▊    | 233738/400000 [00:31<00:22, 7378.32it/s] 59%|█████▊    | 234478/400000 [00:31<00:22, 7308.02it/s] 59%|█████▉    | 235211/400000 [00:32<00:22, 7224.55it/s] 59%|█████▉    | 235941/400000 [00:32<00:22, 7246.90it/s] 59%|█████▉    | 236700/400000 [00:32<00:22, 7344.34it/s] 59%|█████▉    | 237436/400000 [00:32<00:22, 7267.14it/s] 60%|█████▉    | 238164/400000 [00:32<00:22, 7227.72it/s] 60%|█████▉    | 238917/400000 [00:32<00:22, 7315.50it/s] 60%|█████▉    | 239736/400000 [00:32<00:21, 7556.54it/s] 60%|██████    | 240547/400000 [00:32<00:20, 7712.53it/s] 60%|██████    | 241355/400000 [00:32<00:20, 7816.72it/s] 61%|██████    | 242139/400000 [00:32<00:20, 7686.31it/s] 61%|██████    | 242910/400000 [00:33<00:20, 7601.33it/s] 61%|██████    | 243672/400000 [00:33<00:20, 7508.96it/s] 61%|██████    | 244425/400000 [00:33<00:20, 7427.99it/s] 61%|██████▏   | 245169/400000 [00:33<00:20, 7406.48it/s] 61%|██████▏   | 245927/400000 [00:33<00:20, 7454.94it/s] 62%|██████▏   | 246686/400000 [00:33<00:20, 7492.41it/s] 62%|██████▏   | 247461/400000 [00:33<00:20, 7565.67it/s] 62%|██████▏   | 248219/400000 [00:33<00:20, 7554.84it/s] 62%|██████▏   | 248979/400000 [00:33<00:19, 7567.88it/s] 62%|██████▏   | 249737/400000 [00:33<00:19, 7566.56it/s] 63%|██████▎   | 250568/400000 [00:34<00:19, 7773.12it/s] 63%|██████▎   | 251378/400000 [00:34<00:18, 7866.29it/s] 63%|██████▎   | 252166/400000 [00:34<00:18, 7785.83it/s] 63%|██████▎   | 252946/400000 [00:34<00:19, 7736.70it/s] 63%|██████▎   | 253721/400000 [00:34<00:19, 7681.70it/s] 64%|██████▎   | 254490/400000 [00:34<00:19, 7510.19it/s] 64%|██████▍   | 255243/400000 [00:34<00:19, 7306.47it/s] 64%|██████▍   | 256027/400000 [00:34<00:19, 7455.74it/s] 64%|██████▍   | 256793/400000 [00:34<00:19, 7513.76it/s] 64%|██████▍   | 257609/400000 [00:34<00:18, 7694.13it/s] 65%|██████▍   | 258394/400000 [00:35<00:18, 7739.42it/s] 65%|██████▍   | 259170/400000 [00:35<00:18, 7694.72it/s] 65%|██████▍   | 259954/400000 [00:35<00:18, 7737.15it/s] 65%|██████▌   | 260766/400000 [00:35<00:17, 7847.80it/s] 65%|██████▌   | 261573/400000 [00:35<00:17, 7912.30it/s] 66%|██████▌   | 262395/400000 [00:35<00:17, 7999.31it/s] 66%|██████▌   | 263196/400000 [00:35<00:17, 7893.51it/s] 66%|██████▌   | 263987/400000 [00:35<00:17, 7697.23it/s] 66%|██████▌   | 264759/400000 [00:35<00:17, 7597.33it/s] 66%|██████▋   | 265521/400000 [00:36<00:17, 7520.45it/s] 67%|██████▋   | 266283/400000 [00:36<00:17, 7547.86it/s] 67%|██████▋   | 267046/400000 [00:36<00:17, 7570.35it/s] 67%|██████▋   | 267848/400000 [00:36<00:17, 7698.40it/s] 67%|██████▋   | 268619/400000 [00:36<00:17, 7644.44it/s] 67%|██████▋   | 269385/400000 [00:36<00:17, 7306.63it/s] 68%|██████▊   | 270120/400000 [00:36<00:17, 7246.36it/s] 68%|██████▊   | 270848/400000 [00:36<00:18, 7026.57it/s] 68%|██████▊   | 271554/400000 [00:36<00:18, 6797.56it/s] 68%|██████▊   | 272341/400000 [00:36<00:18, 7085.77it/s] 68%|██████▊   | 273191/400000 [00:37<00:17, 7456.16it/s] 69%|██████▊   | 274025/400000 [00:37<00:16, 7698.54it/s] 69%|██████▊   | 274809/400000 [00:37<00:16, 7732.12it/s] 69%|██████▉   | 275589/400000 [00:37<00:16, 7724.78it/s] 69%|██████▉   | 276425/400000 [00:37<00:15, 7902.65it/s] 69%|██████▉   | 277270/400000 [00:37<00:15, 8057.66it/s] 70%|██████▉   | 278082/400000 [00:37<00:15, 8074.88it/s] 70%|██████▉   | 278892/400000 [00:37<00:15, 7800.15it/s] 70%|██████▉   | 279676/400000 [00:37<00:16, 7271.28it/s] 70%|███████   | 280413/400000 [00:38<00:16, 7189.61it/s] 70%|███████   | 281140/400000 [00:38<00:16, 7140.63it/s] 70%|███████   | 281860/400000 [00:38<00:16, 7033.18it/s] 71%|███████   | 282598/400000 [00:38<00:16, 7132.96it/s] 71%|███████   | 283394/400000 [00:38<00:15, 7360.13it/s] 71%|███████   | 284238/400000 [00:38<00:15, 7652.12it/s] 71%|███████▏  | 285051/400000 [00:38<00:14, 7787.18it/s] 71%|███████▏  | 285860/400000 [00:38<00:14, 7875.49it/s] 72%|███████▏  | 286685/400000 [00:38<00:14, 7983.10it/s] 72%|███████▏  | 287504/400000 [00:38<00:13, 8043.18it/s] 72%|███████▏  | 288334/400000 [00:39<00:13, 8116.66it/s] 72%|███████▏  | 289148/400000 [00:39<00:13, 8041.40it/s] 72%|███████▏  | 289954/400000 [00:39<00:13, 8044.14it/s] 73%|███████▎  | 290760/400000 [00:39<00:13, 7872.45it/s] 73%|███████▎  | 291549/400000 [00:39<00:13, 7853.20it/s] 73%|███████▎  | 292340/400000 [00:39<00:13, 7868.01it/s] 73%|███████▎  | 293150/400000 [00:39<00:13, 7934.59it/s] 73%|███████▎  | 293947/400000 [00:39<00:13, 7944.58it/s] 74%|███████▎  | 294742/400000 [00:39<00:13, 7914.00it/s] 74%|███████▍  | 295534/400000 [00:39<00:13, 7904.47it/s] 74%|███████▍  | 296325/400000 [00:40<00:13, 7840.76it/s] 74%|███████▍  | 297125/400000 [00:40<00:13, 7886.59it/s] 74%|███████▍  | 297914/400000 [00:40<00:12, 7874.29it/s] 75%|███████▍  | 298713/400000 [00:40<00:12, 7906.48it/s] 75%|███████▍  | 299504/400000 [00:40<00:12, 7848.47it/s] 75%|███████▌  | 300347/400000 [00:40<00:12, 8011.67it/s] 75%|███████▌  | 301162/400000 [00:40<00:12, 8049.37it/s] 75%|███████▌  | 301968/400000 [00:40<00:12, 7800.49it/s] 76%|███████▌  | 302751/400000 [00:40<00:13, 7385.78it/s] 76%|███████▌  | 303496/400000 [00:40<00:13, 7162.47it/s] 76%|███████▌  | 304218/400000 [00:41<00:13, 7029.94it/s] 76%|███████▌  | 304955/400000 [00:41<00:13, 7128.55it/s] 76%|███████▋  | 305742/400000 [00:41<00:12, 7334.35it/s] 77%|███████▋  | 306480/400000 [00:41<00:12, 7312.23it/s] 77%|███████▋  | 307269/400000 [00:41<00:12, 7476.15it/s] 77%|███████▋  | 308120/400000 [00:41<00:11, 7756.92it/s] 77%|███████▋  | 308954/400000 [00:41<00:11, 7921.92it/s] 77%|███████▋  | 309802/400000 [00:41<00:11, 8079.16it/s] 78%|███████▊  | 310614/400000 [00:41<00:11, 8060.80it/s] 78%|███████▊  | 311446/400000 [00:41<00:10, 8136.74it/s] 78%|███████▊  | 312262/400000 [00:42<00:10, 7996.29it/s] 78%|███████▊  | 313064/400000 [00:42<00:11, 7363.64it/s] 78%|███████▊  | 313812/400000 [00:42<00:12, 6989.42it/s] 79%|███████▊  | 314523/400000 [00:42<00:12, 6656.31it/s] 79%|███████▉  | 315201/400000 [00:42<00:12, 6645.24it/s] 79%|███████▉  | 316044/400000 [00:42<00:11, 7094.87it/s] 79%|███████▉  | 316881/400000 [00:42<00:11, 7433.47it/s] 79%|███████▉  | 317732/400000 [00:42<00:10, 7725.55it/s] 80%|███████▉  | 318525/400000 [00:42<00:10, 7784.00it/s] 80%|███████▉  | 319334/400000 [00:43<00:10, 7873.30it/s] 80%|████████  | 320172/400000 [00:43<00:09, 8016.64it/s] 80%|████████  | 321027/400000 [00:43<00:09, 8166.92it/s] 80%|████████  | 321849/400000 [00:43<00:09, 7893.12it/s] 81%|████████  | 322644/400000 [00:43<00:10, 7619.00it/s] 81%|████████  | 323412/400000 [00:43<00:10, 7338.28it/s] 81%|████████  | 324152/400000 [00:43<00:10, 7188.00it/s] 81%|████████  | 324944/400000 [00:43<00:10, 7392.16it/s] 81%|████████▏ | 325744/400000 [00:43<00:09, 7562.94it/s] 82%|████████▏ | 326537/400000 [00:44<00:09, 7667.42it/s] 82%|████████▏ | 327359/400000 [00:44<00:09, 7824.60it/s] 82%|████████▏ | 328145/400000 [00:44<00:09, 7803.73it/s] 82%|████████▏ | 328928/400000 [00:44<00:09, 7459.03it/s] 82%|████████▏ | 329679/400000 [00:44<00:09, 7183.67it/s] 83%|████████▎ | 330403/400000 [00:44<00:09, 7034.08it/s] 83%|████████▎ | 331111/400000 [00:44<00:09, 7016.51it/s] 83%|████████▎ | 331890/400000 [00:44<00:09, 7230.87it/s] 83%|████████▎ | 332670/400000 [00:44<00:09, 7391.01it/s] 83%|████████▎ | 333461/400000 [00:44<00:08, 7536.92it/s] 84%|████████▎ | 334218/400000 [00:45<00:08, 7511.04it/s] 84%|████████▎ | 334996/400000 [00:45<00:08, 7587.97it/s] 84%|████████▍ | 335794/400000 [00:45<00:08, 7701.40it/s] 84%|████████▍ | 336587/400000 [00:45<00:08, 7768.27it/s] 84%|████████▍ | 337370/400000 [00:45<00:08, 7786.40it/s] 85%|████████▍ | 338187/400000 [00:45<00:07, 7895.69it/s] 85%|████████▍ | 338997/400000 [00:45<00:07, 7953.86it/s] 85%|████████▍ | 339798/400000 [00:45<00:07, 7968.46it/s] 85%|████████▌ | 340596/400000 [00:45<00:07, 7648.36it/s] 85%|████████▌ | 341364/400000 [00:45<00:08, 7312.66it/s] 86%|████████▌ | 342101/400000 [00:46<00:08, 7018.77it/s] 86%|████████▌ | 342864/400000 [00:46<00:07, 7189.70it/s] 86%|████████▌ | 343589/400000 [00:46<00:08, 6967.19it/s] 86%|████████▌ | 344291/400000 [00:46<00:08, 6825.55it/s] 86%|████████▌ | 344998/400000 [00:46<00:07, 6894.62it/s] 86%|████████▋ | 345775/400000 [00:46<00:07, 7134.84it/s] 87%|████████▋ | 346580/400000 [00:46<00:07, 7385.56it/s] 87%|████████▋ | 347369/400000 [00:46<00:06, 7528.40it/s] 87%|████████▋ | 348161/400000 [00:46<00:06, 7639.25it/s] 87%|████████▋ | 348929/400000 [00:47<00:06, 7311.65it/s] 87%|████████▋ | 349666/400000 [00:47<00:07, 7006.32it/s] 88%|████████▊ | 350374/400000 [00:47<00:07, 6937.88it/s] 88%|████████▊ | 351073/400000 [00:47<00:07, 6880.61it/s] 88%|████████▊ | 351813/400000 [00:47<00:06, 7026.19it/s] 88%|████████▊ | 352537/400000 [00:47<00:06, 7087.73it/s] 88%|████████▊ | 353295/400000 [00:47<00:06, 7227.31it/s] 89%|████████▊ | 354076/400000 [00:47<00:06, 7391.01it/s] 89%|████████▊ | 354833/400000 [00:47<00:06, 7441.92it/s] 89%|████████▉ | 355580/400000 [00:47<00:06, 7094.71it/s] 89%|████████▉ | 356295/400000 [00:48<00:06, 7059.99it/s] 89%|████████▉ | 357111/400000 [00:48<00:05, 7356.99it/s] 89%|████████▉ | 357933/400000 [00:48<00:05, 7595.30it/s] 90%|████████▉ | 358771/400000 [00:48<00:05, 7812.74it/s] 90%|████████▉ | 359558/400000 [00:48<00:05, 7823.63it/s] 90%|█████████ | 360345/400000 [00:48<00:05, 7428.80it/s] 90%|█████████ | 361095/400000 [00:48<00:05, 7055.72it/s] 90%|█████████ | 361810/400000 [00:48<00:05, 6920.40it/s] 91%|█████████ | 362569/400000 [00:48<00:05, 7107.07it/s] 91%|█████████ | 363350/400000 [00:49<00:05, 7301.97it/s] 91%|█████████ | 364162/400000 [00:49<00:04, 7527.89it/s] 91%|█████████ | 364966/400000 [00:49<00:04, 7673.00it/s] 91%|█████████▏| 365797/400000 [00:49<00:04, 7851.19it/s] 92%|█████████▏| 366606/400000 [00:49<00:04, 7919.22it/s] 92%|█████████▏| 367402/400000 [00:49<00:04, 7530.48it/s] 92%|█████████▏| 368213/400000 [00:49<00:04, 7694.16it/s] 92%|█████████▏| 369054/400000 [00:49<00:03, 7895.37it/s] 92%|█████████▏| 369888/400000 [00:49<00:03, 8021.51it/s] 93%|█████████▎| 370736/400000 [00:49<00:03, 8151.41it/s] 93%|█████████▎| 371555/400000 [00:50<00:03, 8083.98it/s] 93%|█████████▎| 372366/400000 [00:50<00:03, 8017.55it/s] 93%|█████████▎| 373172/400000 [00:50<00:03, 8028.71it/s] 93%|█████████▎| 373977/400000 [00:50<00:03, 8026.33it/s] 94%|█████████▎| 374809/400000 [00:50<00:03, 8109.90it/s] 94%|█████████▍| 375621/400000 [00:50<00:03, 8056.49it/s] 94%|█████████▍| 376457/400000 [00:50<00:02, 8145.06it/s] 94%|█████████▍| 377288/400000 [00:50<00:02, 8190.79it/s] 95%|█████████▍| 378108/400000 [00:50<00:02, 8182.63it/s] 95%|█████████▍| 378945/400000 [00:50<00:02, 8235.61it/s] 95%|█████████▍| 379769/400000 [00:51<00:02, 8202.81it/s] 95%|█████████▌| 380590/400000 [00:51<00:02, 7772.35it/s] 95%|█████████▌| 381373/400000 [00:51<00:02, 7374.14it/s] 96%|█████████▌| 382119/400000 [00:51<00:02, 7094.65it/s] 96%|█████████▌| 382837/400000 [00:51<00:02, 7111.23it/s] 96%|█████████▌| 383660/400000 [00:51<00:02, 7412.09it/s] 96%|█████████▌| 384486/400000 [00:51<00:02, 7647.41it/s] 96%|█████████▋| 385313/400000 [00:51<00:01, 7823.69it/s] 97%|█████████▋| 386138/400000 [00:51<00:01, 7946.80it/s] 97%|█████████▋| 386963/400000 [00:52<00:01, 8035.04it/s] 97%|█████████▋| 387789/400000 [00:52<00:01, 8100.55it/s] 97%|█████████▋| 388602/400000 [00:52<00:01, 7746.60it/s] 97%|█████████▋| 389382/400000 [00:52<00:01, 7494.00it/s] 98%|█████████▊| 390199/400000 [00:52<00:01, 7680.89it/s] 98%|█████████▊| 391001/400000 [00:52<00:01, 7777.41it/s] 98%|█████████▊| 391783/400000 [00:52<00:01, 7730.70it/s] 98%|█████████▊| 392559/400000 [00:52<00:01, 7052.71it/s] 98%|█████████▊| 393278/400000 [00:52<00:00, 6939.46it/s] 99%|█████████▊| 394112/400000 [00:52<00:00, 7305.81it/s] 99%|█████████▊| 394876/400000 [00:53<00:00, 7400.73it/s] 99%|█████████▉| 395711/400000 [00:53<00:00, 7661.88it/s] 99%|█████████▉| 396522/400000 [00:53<00:00, 7790.74it/s] 99%|█████████▉| 397313/400000 [00:53<00:00, 7825.13it/s]100%|█████████▉| 398105/400000 [00:53<00:00, 7851.73it/s]100%|█████████▉| 398904/400000 [00:53<00:00, 7892.26it/s]100%|█████████▉| 399710/400000 [00:53<00:00, 7941.34it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7447.91it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f96cfcdde48> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010851628875868855 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.011053101075532843 	 Accuracy: 65

  model saves at 65% accuracy 

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

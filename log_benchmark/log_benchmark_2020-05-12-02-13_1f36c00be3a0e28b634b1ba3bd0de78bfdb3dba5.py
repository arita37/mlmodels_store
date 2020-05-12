
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f518f631f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 02:13:19.144647
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 02:13:19.149413
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 02:13:19.153361
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 02:13:19.157301
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f519b3f5438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351097.9062
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 212252.4219
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 105915.9375
Epoch 4/10

1/1 [==============================] - 0s 116ms/step - loss: 49023.0547
Epoch 5/10

1/1 [==============================] - 0s 115ms/step - loss: 24199.7930
Epoch 6/10

1/1 [==============================] - 0s 145ms/step - loss: 13459.8896
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 8352.1719
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 5690.3047
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 4205.0474
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 3329.8311

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.49889934e+00  1.33875430e+00  2.48849249e+00  7.20270157e-01
   1.46913075e+00  1.06586540e+00 -9.76144075e-01 -2.07649398e+00
  -2.09092116e+00  3.67919028e-01 -1.77656913e+00 -1.13640517e-01
   2.60764003e-01  1.35756958e+00 -7.71726489e-01 -4.90079194e-01
  -2.03944778e+00  6.43398464e-02 -7.92688489e-01 -1.32856262e+00
   1.46077800e+00  2.22072029e+00  1.60012573e-01 -9.76920545e-01
  -2.79641479e-01  3.16672325e-02  9.11800802e-01 -1.30239260e+00
  -7.34704137e-02 -8.63459945e-01  3.15893126e+00 -7.51255155e-01
  -7.07991958e-01 -1.33345461e+00 -2.34707189e+00  3.32249284e-01
   3.13528061e-01 -1.13375759e+00  2.05745769e+00  4.13070023e-01
   6.90341294e-01  3.53432238e-01  4.53928113e-01 -1.92274618e+00
  -6.45147085e-01  2.07178140e+00 -8.32542896e-01  2.72502351e+00
   5.51275790e-01 -1.31032670e+00  4.68456745e-03  2.00336099e+00
  -2.25184011e+00 -3.75959814e-01 -2.12022400e+00  6.16934836e-01
  -1.89864683e+00 -1.29795671e+00 -2.76290894e+00  1.86913025e+00
  -8.83383334e-01  1.41674213e+01  1.17826223e+01  1.15238028e+01
   1.15972242e+01  1.40407667e+01  1.11315250e+01  1.27826576e+01
   1.15262356e+01  1.43795090e+01  1.34722176e+01  1.22724648e+01
   1.16958151e+01  1.38770638e+01  1.26560612e+01  1.29242201e+01
   1.32427063e+01  1.24198160e+01  1.30532322e+01  1.29009724e+01
   1.20875006e+01  1.22335062e+01  1.01832590e+01  1.24521580e+01
   1.31290302e+01  1.38438816e+01  1.36759377e+01  1.31811695e+01
   1.47992353e+01  1.11399088e+01  1.18295746e+01  1.28069687e+01
   1.19460230e+01  1.30340719e+01  1.24190664e+01  1.34486265e+01
   1.31161356e+01  1.37833776e+01  1.34843359e+01  1.27094240e+01
   1.30312595e+01  1.20459204e+01  1.22621593e+01  1.23393908e+01
   1.00414429e+01  1.27008028e+01  1.15162220e+01  1.36107960e+01
   1.22402582e+01  1.31382189e+01  1.09846506e+01  1.41663837e+01
   1.39964695e+01  1.15223951e+01  1.04996338e+01  1.27430840e+01
   1.30039368e+01  1.31760044e+01  1.12942238e+01  1.49676085e+01
  -7.57401884e-02 -2.07412213e-01  4.77043837e-01 -1.23242342e+00
  -1.80874825e+00 -1.96210766e+00 -1.48852229e-01  2.14926481e+00
  -1.75599468e+00  8.70437086e-01 -1.50968766e+00 -1.40587851e-01
  -1.99291611e+00 -1.42145801e+00  1.08149266e+00  4.00397271e-01
  -3.00934732e-01 -1.16337705e+00  2.97878124e-02  6.41157776e-02
  -1.66523504e+00  2.30628774e-01  1.92314970e+00 -1.14189970e+00
   1.08877206e+00  3.27336878e-01  9.27466333e-01 -1.16174984e+00
  -1.76046491e+00  6.22781277e-01 -3.57314825e-01 -3.28108847e-01
   6.78321719e-01  4.22071815e-02  2.10603285e+00  6.44436777e-01
  -6.82219744e-01  3.91762137e-01 -1.23116505e+00 -1.19768524e+00
  -2.55967349e-01 -3.03798795e+00  2.32393599e+00  1.66321886e+00
   6.35839283e-01 -6.12878680e-01 -1.48212790e-01  4.02402759e-01
   1.68865502e+00 -6.26948595e-01 -5.08846343e-02  6.16732657e-01
  -1.97229075e+00  1.16718620e-01 -1.66239524e+00 -6.81381464e-01
  -8.50681543e-01  2.77408481e+00 -2.51004982e+00 -2.61264467e+00
   9.72473860e-01  2.56129789e+00  3.79848957e-01  2.61334896e-01
   1.73829722e+00  1.89718437e+00  1.51522422e+00  2.75461054e+00
   5.63677669e-01  1.37588620e+00  1.77569211e-01  1.35651827e-01
   5.89576304e-01  2.28326130e+00  1.38890588e+00  4.66669977e-01
   3.21431732e+00  6.18577242e-01  3.92128706e+00  6.48011208e-01
   2.43567562e+00  3.50719512e-01  2.32120156e-01  3.08759356e+00
   1.97193980e-01  5.47270119e-01  7.92013168e-01  2.71010399e-02
   2.18059015e+00  3.98820543e+00  4.65427041e-01  9.33906436e-02
   2.47305870e-01  1.39352608e+00  2.24366724e-01  8.92594934e-01
   2.45439768e-01  1.77477372e+00  4.34206903e-01  9.19506848e-01
   1.12728548e+00  1.52819920e+00  2.14332342e-01  4.04980659e-01
   5.49138188e-01  2.24641085e+00  3.00269902e-01  1.70094478e+00
   4.83578324e-01  2.26749372e+00  1.67456257e+00  2.00294352e+00
   1.89502954e+00  7.90481210e-01  2.11798096e+00  2.31766367e+00
   2.93938017e+00  6.30785227e-01  4.76425290e-01  3.11068892e-01
   9.76104319e-01  1.15004368e+01  1.06468630e+01  1.38361340e+01
   1.14992399e+01  1.32515726e+01  1.11503363e+01  1.23041945e+01
   1.46788044e+01  1.55415239e+01  1.24127197e+01  1.54671593e+01
   8.65182209e+00  1.42877998e+01  1.37445383e+01  1.36728745e+01
   1.27575493e+01  1.14275780e+01  9.58844376e+00  1.48734999e+01
   9.61308765e+00  9.70424366e+00  1.40543804e+01  1.33861599e+01
   1.18774281e+01  1.13727236e+01  1.14970713e+01  1.05245457e+01
   1.50085907e+01  1.07083187e+01  1.31328831e+01  1.19152250e+01
   1.51866760e+01  1.59392481e+01  1.26498604e+01  1.45478287e+01
   1.31207809e+01  1.00234556e+01  1.12982349e+01  1.41533108e+01
   1.48007574e+01  1.51179190e+01  1.41658697e+01  1.34965248e+01
   1.25790634e+01  1.09380722e+01  1.24602757e+01  1.36731958e+01
   1.23609657e+01  1.00677528e+01  1.30839100e+01  1.26450672e+01
   1.34341955e+01  9.57705307e+00  8.02799320e+00  1.26441822e+01
   1.27243252e+01  1.28859062e+01  9.53874588e+00  1.49697065e+01
   2.14426851e+00  2.27965474e-01  1.77271247e+00  2.77647734e+00
   2.01820660e+00  2.04203224e+00  1.86314535e+00  2.91944504e+00
   5.06173551e-01  1.70626938e+00  6.56866193e-01  1.67697692e+00
   2.74562788e+00  3.65878761e-01  2.18668509e+00  1.09610438e-01
   7.07728148e-01  1.86069465e+00  1.74202585e+00  4.14975166e-01
   8.58778596e-01  2.48462868e+00  1.28677917e+00  2.61360002e+00
   7.27894545e-01  8.72753263e-01  4.18707752e+00  8.83114159e-01
   1.48926818e+00  3.97495568e-01  2.59738505e-01  6.68513775e-01
   2.40058517e+00  5.62798619e-01  1.70488930e+00  4.40604925e-01
   6.13743007e-01  3.97219419e-01  7.73802638e-01  3.10559154e-01
   1.50540257e+00  6.72674179e-02  2.43123245e+00  1.03695548e+00
   1.39393997e+00  2.84364891e+00  8.54192376e-02  3.77857089e-01
   2.30775595e+00  1.91920590e+00  1.23810279e+00  3.55361366e+00
   1.21083558e+00  5.62732041e-01  4.66257751e-01  2.07590914e+00
   1.02960777e+00  4.43934917e-01  2.15909421e-01  1.23688030e+00
  -6.72651958e+00  1.86754513e+01 -6.58012533e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 02:13:29.300169
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.0801
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 02:13:29.304835
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   7970.31
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 02:13:29.309045
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.0442
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 02:13:29.313179
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -712.797
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139987932720656
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139986705866824
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139986705867328
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139986705867832
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139986705868336
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139986705868840

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5197278e80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.885916
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.840396
grad_step = 000002, loss = 0.804891
grad_step = 000003, loss = 0.766708
grad_step = 000004, loss = 0.725579
grad_step = 000005, loss = 0.686480
grad_step = 000006, loss = 0.657993
grad_step = 000007, loss = 0.634624
grad_step = 000008, loss = 0.607595
grad_step = 000009, loss = 0.576804
grad_step = 000010, loss = 0.549136
grad_step = 000011, loss = 0.528660
grad_step = 000012, loss = 0.511195
grad_step = 000013, loss = 0.492830
grad_step = 000014, loss = 0.475237
grad_step = 000015, loss = 0.458007
grad_step = 000016, loss = 0.439632
grad_step = 000017, loss = 0.421681
grad_step = 000018, loss = 0.404159
grad_step = 000019, loss = 0.386137
grad_step = 000020, loss = 0.369582
grad_step = 000021, loss = 0.355090
grad_step = 000022, loss = 0.340803
grad_step = 000023, loss = 0.325982
grad_step = 000024, loss = 0.311167
grad_step = 000025, loss = 0.297306
grad_step = 000026, loss = 0.284633
grad_step = 000027, loss = 0.271744
grad_step = 000028, loss = 0.258366
grad_step = 000029, loss = 0.245564
grad_step = 000030, loss = 0.233745
grad_step = 000031, loss = 0.222798
grad_step = 000032, loss = 0.212066
grad_step = 000033, loss = 0.201282
grad_step = 000034, loss = 0.191168
grad_step = 000035, loss = 0.181758
grad_step = 000036, loss = 0.172524
grad_step = 000037, loss = 0.163567
grad_step = 000038, loss = 0.154870
grad_step = 000039, loss = 0.146623
grad_step = 000040, loss = 0.138841
grad_step = 000041, loss = 0.131333
grad_step = 000042, loss = 0.124179
grad_step = 000043, loss = 0.117416
grad_step = 000044, loss = 0.110941
grad_step = 000045, loss = 0.104574
grad_step = 000046, loss = 0.098527
grad_step = 000047, loss = 0.092921
grad_step = 000048, loss = 0.087502
grad_step = 000049, loss = 0.082247
grad_step = 000050, loss = 0.077375
grad_step = 000051, loss = 0.072857
grad_step = 000052, loss = 0.068446
grad_step = 000053, loss = 0.064283
grad_step = 000054, loss = 0.060417
grad_step = 000055, loss = 0.056702
grad_step = 000056, loss = 0.053191
grad_step = 000057, loss = 0.049872
grad_step = 000058, loss = 0.046761
grad_step = 000059, loss = 0.043774
grad_step = 000060, loss = 0.040921
grad_step = 000061, loss = 0.038272
grad_step = 000062, loss = 0.035730
grad_step = 000063, loss = 0.033282
grad_step = 000064, loss = 0.030999
grad_step = 000065, loss = 0.028844
grad_step = 000066, loss = 0.026784
grad_step = 000067, loss = 0.024850
grad_step = 000068, loss = 0.023050
grad_step = 000069, loss = 0.021338
grad_step = 000070, loss = 0.019753
grad_step = 000071, loss = 0.018271
grad_step = 000072, loss = 0.016885
grad_step = 000073, loss = 0.015593
grad_step = 000074, loss = 0.014413
grad_step = 000075, loss = 0.013313
grad_step = 000076, loss = 0.012292
grad_step = 000077, loss = 0.011366
grad_step = 000078, loss = 0.010507
grad_step = 000079, loss = 0.009715
grad_step = 000080, loss = 0.008999
grad_step = 000081, loss = 0.008341
grad_step = 000082, loss = 0.007740
grad_step = 000083, loss = 0.007197
grad_step = 000084, loss = 0.006696
grad_step = 000085, loss = 0.006239
grad_step = 000086, loss = 0.005828
grad_step = 000087, loss = 0.005448
grad_step = 000088, loss = 0.005103
grad_step = 000089, loss = 0.004793
grad_step = 000090, loss = 0.004507
grad_step = 000091, loss = 0.004247
grad_step = 000092, loss = 0.004012
grad_step = 000093, loss = 0.003796
grad_step = 000094, loss = 0.003600
grad_step = 000095, loss = 0.003423
grad_step = 000096, loss = 0.003260
grad_step = 000097, loss = 0.003113
grad_step = 000098, loss = 0.002979
grad_step = 000099, loss = 0.002855
grad_step = 000100, loss = 0.002745
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002644
grad_step = 000102, loss = 0.002552
grad_step = 000103, loss = 0.002468
grad_step = 000104, loss = 0.002392
grad_step = 000105, loss = 0.002323
grad_step = 000106, loss = 0.002261
grad_step = 000107, loss = 0.002205
grad_step = 000108, loss = 0.002155
grad_step = 000109, loss = 0.002110
grad_step = 000110, loss = 0.002070
grad_step = 000111, loss = 0.002034
grad_step = 000112, loss = 0.002002
grad_step = 000113, loss = 0.001974
grad_step = 000114, loss = 0.001949
grad_step = 000115, loss = 0.001927
grad_step = 000116, loss = 0.001908
grad_step = 000117, loss = 0.001891
grad_step = 000118, loss = 0.001876
grad_step = 000119, loss = 0.001864
grad_step = 000120, loss = 0.001854
grad_step = 000121, loss = 0.001849
grad_step = 000122, loss = 0.001848
grad_step = 000123, loss = 0.001843
grad_step = 000124, loss = 0.001833
grad_step = 000125, loss = 0.001817
grad_step = 000126, loss = 0.001805
grad_step = 000127, loss = 0.001801
grad_step = 000128, loss = 0.001802
grad_step = 000129, loss = 0.001802
grad_step = 000130, loss = 0.001798
grad_step = 000131, loss = 0.001790
grad_step = 000132, loss = 0.001781
grad_step = 000133, loss = 0.001776
grad_step = 000134, loss = 0.001775
grad_step = 000135, loss = 0.001775
grad_step = 000136, loss = 0.001773
grad_step = 000137, loss = 0.001769
grad_step = 000138, loss = 0.001764
grad_step = 000139, loss = 0.001758
grad_step = 000140, loss = 0.001753
grad_step = 000141, loss = 0.001749
grad_step = 000142, loss = 0.001747
grad_step = 000143, loss = 0.001747
grad_step = 000144, loss = 0.001746
grad_step = 000145, loss = 0.001746
grad_step = 000146, loss = 0.001747
grad_step = 000147, loss = 0.001748
grad_step = 000148, loss = 0.001748
grad_step = 000149, loss = 0.001747
grad_step = 000150, loss = 0.001745
grad_step = 000151, loss = 0.001743
grad_step = 000152, loss = 0.001739
grad_step = 000153, loss = 0.001735
grad_step = 000154, loss = 0.001728
grad_step = 000155, loss = 0.001721
grad_step = 000156, loss = 0.001716
grad_step = 000157, loss = 0.001713
grad_step = 000158, loss = 0.001713
grad_step = 000159, loss = 0.001714
grad_step = 000160, loss = 0.001718
grad_step = 000161, loss = 0.001723
grad_step = 000162, loss = 0.001729
grad_step = 000163, loss = 0.001732
grad_step = 000164, loss = 0.001737
grad_step = 000165, loss = 0.001747
grad_step = 000166, loss = 0.001768
grad_step = 000167, loss = 0.001807
grad_step = 000168, loss = 0.001832
grad_step = 000169, loss = 0.001841
grad_step = 000170, loss = 0.001780
grad_step = 000171, loss = 0.001710
grad_step = 000172, loss = 0.001669
grad_step = 000173, loss = 0.001679
grad_step = 000174, loss = 0.001710
grad_step = 000175, loss = 0.001715
grad_step = 000176, loss = 0.001686
grad_step = 000177, loss = 0.001644
grad_step = 000178, loss = 0.001624
grad_step = 000179, loss = 0.001635
grad_step = 000180, loss = 0.001656
grad_step = 000181, loss = 0.001667
grad_step = 000182, loss = 0.001655
grad_step = 000183, loss = 0.001636
grad_step = 000184, loss = 0.001625
grad_step = 000185, loss = 0.001634
grad_step = 000186, loss = 0.001659
grad_step = 000187, loss = 0.001695
grad_step = 000188, loss = 0.001743
grad_step = 000189, loss = 0.001812
grad_step = 000190, loss = 0.001912
grad_step = 000191, loss = 0.002027
grad_step = 000192, loss = 0.002089
grad_step = 000193, loss = 0.001992
grad_step = 000194, loss = 0.001780
grad_step = 000195, loss = 0.001619
grad_step = 000196, loss = 0.001625
grad_step = 000197, loss = 0.001729
grad_step = 000198, loss = 0.001778
grad_step = 000199, loss = 0.001727
grad_step = 000200, loss = 0.001650
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001648
grad_step = 000202, loss = 0.001686
grad_step = 000203, loss = 0.001687
grad_step = 000204, loss = 0.001629
grad_step = 000205, loss = 0.001580
grad_step = 000206, loss = 0.001588
grad_step = 000207, loss = 0.001631
grad_step = 000208, loss = 0.001655
grad_step = 000209, loss = 0.001617
grad_step = 000210, loss = 0.001569
grad_step = 000211, loss = 0.001554
grad_step = 000212, loss = 0.001575
grad_step = 000213, loss = 0.001595
grad_step = 000214, loss = 0.001588
grad_step = 000215, loss = 0.001569
grad_step = 000216, loss = 0.001559
grad_step = 000217, loss = 0.001566
grad_step = 000218, loss = 0.001577
grad_step = 000219, loss = 0.001579
grad_step = 000220, loss = 0.001571
grad_step = 000221, loss = 0.001569
grad_step = 000222, loss = 0.001582
grad_step = 000223, loss = 0.001618
grad_step = 000224, loss = 0.001638
grad_step = 000225, loss = 0.001643
grad_step = 000226, loss = 0.001604
grad_step = 000227, loss = 0.001569
grad_step = 000228, loss = 0.001569
grad_step = 000229, loss = 0.001604
grad_step = 000230, loss = 0.001656
grad_step = 000231, loss = 0.001673
grad_step = 000232, loss = 0.001685
grad_step = 000233, loss = 0.001691
grad_step = 000234, loss = 0.001732
grad_step = 000235, loss = 0.001776
grad_step = 000236, loss = 0.001795
grad_step = 000237, loss = 0.001758
grad_step = 000238, loss = 0.001683
grad_step = 000239, loss = 0.001602
grad_step = 000240, loss = 0.001548
grad_step = 000241, loss = 0.001526
grad_step = 000242, loss = 0.001528
grad_step = 000243, loss = 0.001536
grad_step = 000244, loss = 0.001550
grad_step = 000245, loss = 0.001572
grad_step = 000246, loss = 0.001594
grad_step = 000247, loss = 0.001607
grad_step = 000248, loss = 0.001594
grad_step = 000249, loss = 0.001565
grad_step = 000250, loss = 0.001524
grad_step = 000251, loss = 0.001495
grad_step = 000252, loss = 0.001483
grad_step = 000253, loss = 0.001485
grad_step = 000254, loss = 0.001489
grad_step = 000255, loss = 0.001491
grad_step = 000256, loss = 0.001492
grad_step = 000257, loss = 0.001498
grad_step = 000258, loss = 0.001509
grad_step = 000259, loss = 0.001524
grad_step = 000260, loss = 0.001538
grad_step = 000261, loss = 0.001549
grad_step = 000262, loss = 0.001560
grad_step = 000263, loss = 0.001572
grad_step = 000264, loss = 0.001590
grad_step = 000265, loss = 0.001611
grad_step = 000266, loss = 0.001638
grad_step = 000267, loss = 0.001659
grad_step = 000268, loss = 0.001675
grad_step = 000269, loss = 0.001669
grad_step = 000270, loss = 0.001649
grad_step = 000271, loss = 0.001610
grad_step = 000272, loss = 0.001563
grad_step = 000273, loss = 0.001515
grad_step = 000274, loss = 0.001476
grad_step = 000275, loss = 0.001450
grad_step = 000276, loss = 0.001440
grad_step = 000277, loss = 0.001442
grad_step = 000278, loss = 0.001455
grad_step = 000279, loss = 0.001473
grad_step = 000280, loss = 0.001493
grad_step = 000281, loss = 0.001511
grad_step = 000282, loss = 0.001524
grad_step = 000283, loss = 0.001533
grad_step = 000284, loss = 0.001536
grad_step = 000285, loss = 0.001535
grad_step = 000286, loss = 0.001530
grad_step = 000287, loss = 0.001525
grad_step = 000288, loss = 0.001515
grad_step = 000289, loss = 0.001505
grad_step = 000290, loss = 0.001490
grad_step = 000291, loss = 0.001475
grad_step = 000292, loss = 0.001459
grad_step = 000293, loss = 0.001446
grad_step = 000294, loss = 0.001435
grad_step = 000295, loss = 0.001429
grad_step = 000296, loss = 0.001425
grad_step = 000297, loss = 0.001424
grad_step = 000298, loss = 0.001423
grad_step = 000299, loss = 0.001425
grad_step = 000300, loss = 0.001427
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001433
grad_step = 000302, loss = 0.001439
grad_step = 000303, loss = 0.001453
grad_step = 000304, loss = 0.001464
grad_step = 000305, loss = 0.001484
grad_step = 000306, loss = 0.001484
grad_step = 000307, loss = 0.001479
grad_step = 000308, loss = 0.001459
grad_step = 000309, loss = 0.001445
grad_step = 000310, loss = 0.001450
grad_step = 000311, loss = 0.001472
grad_step = 000312, loss = 0.001509
grad_step = 000313, loss = 0.001535
grad_step = 000314, loss = 0.001582
grad_step = 000315, loss = 0.001665
grad_step = 000316, loss = 0.001874
grad_step = 000317, loss = 0.002289
grad_step = 000318, loss = 0.002983
grad_step = 000319, loss = 0.003408
grad_step = 000320, loss = 0.003076
grad_step = 000321, loss = 0.001905
grad_step = 000322, loss = 0.001504
grad_step = 000323, loss = 0.002088
grad_step = 000324, loss = 0.002309
grad_step = 000325, loss = 0.001856
grad_step = 000326, loss = 0.001617
grad_step = 000327, loss = 0.001856
grad_step = 000328, loss = 0.001807
grad_step = 000329, loss = 0.001604
grad_step = 000330, loss = 0.001759
grad_step = 000331, loss = 0.001754
grad_step = 000332, loss = 0.001468
grad_step = 000333, loss = 0.001586
grad_step = 000334, loss = 0.001722
grad_step = 000335, loss = 0.001468
grad_step = 000336, loss = 0.001406
grad_step = 000337, loss = 0.001607
grad_step = 000338, loss = 0.001540
grad_step = 000339, loss = 0.001372
grad_step = 000340, loss = 0.001486
grad_step = 000341, loss = 0.001529
grad_step = 000342, loss = 0.001400
grad_step = 000343, loss = 0.001421
grad_step = 000344, loss = 0.001466
grad_step = 000345, loss = 0.001396
grad_step = 000346, loss = 0.001401
grad_step = 000347, loss = 0.001429
grad_step = 000348, loss = 0.001380
grad_step = 000349, loss = 0.001375
grad_step = 000350, loss = 0.001416
grad_step = 000351, loss = 0.001384
grad_step = 000352, loss = 0.001350
grad_step = 000353, loss = 0.001386
grad_step = 000354, loss = 0.001390
grad_step = 000355, loss = 0.001350
grad_step = 000356, loss = 0.001359
grad_step = 000357, loss = 0.001374
grad_step = 000358, loss = 0.001354
grad_step = 000359, loss = 0.001351
grad_step = 000360, loss = 0.001360
grad_step = 000361, loss = 0.001348
grad_step = 000362, loss = 0.001343
grad_step = 000363, loss = 0.001354
grad_step = 000364, loss = 0.001350
grad_step = 000365, loss = 0.001339
grad_step = 000366, loss = 0.001347
grad_step = 000367, loss = 0.001362
grad_step = 000368, loss = 0.001366
grad_step = 000369, loss = 0.001390
grad_step = 000370, loss = 0.001434
grad_step = 000371, loss = 0.001504
grad_step = 000372, loss = 0.001538
grad_step = 000373, loss = 0.001577
grad_step = 000374, loss = 0.001474
grad_step = 000375, loss = 0.001367
grad_step = 000376, loss = 0.001328
grad_step = 000377, loss = 0.001375
grad_step = 000378, loss = 0.001433
grad_step = 000379, loss = 0.001419
grad_step = 000380, loss = 0.001367
grad_step = 000381, loss = 0.001324
grad_step = 000382, loss = 0.001330
grad_step = 000383, loss = 0.001370
grad_step = 000384, loss = 0.001393
grad_step = 000385, loss = 0.001384
grad_step = 000386, loss = 0.001342
grad_step = 000387, loss = 0.001316
grad_step = 000388, loss = 0.001316
grad_step = 000389, loss = 0.001332
grad_step = 000390, loss = 0.001349
grad_step = 000391, loss = 0.001344
grad_step = 000392, loss = 0.001326
grad_step = 000393, loss = 0.001309
grad_step = 000394, loss = 0.001309
grad_step = 000395, loss = 0.001320
grad_step = 000396, loss = 0.001323
grad_step = 000397, loss = 0.001316
grad_step = 000398, loss = 0.001306
grad_step = 000399, loss = 0.001302
grad_step = 000400, loss = 0.001304
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001309
grad_step = 000402, loss = 0.001310
grad_step = 000403, loss = 0.001305
grad_step = 000404, loss = 0.001299
grad_step = 000405, loss = 0.001295
grad_step = 000406, loss = 0.001295
grad_step = 000407, loss = 0.001298
grad_step = 000408, loss = 0.001298
grad_step = 000409, loss = 0.001297
grad_step = 000410, loss = 0.001295
grad_step = 000411, loss = 0.001292
grad_step = 000412, loss = 0.001289
grad_step = 000413, loss = 0.001289
grad_step = 000414, loss = 0.001290
grad_step = 000415, loss = 0.001290
grad_step = 000416, loss = 0.001290
grad_step = 000417, loss = 0.001288
grad_step = 000418, loss = 0.001287
grad_step = 000419, loss = 0.001287
grad_step = 000420, loss = 0.001288
grad_step = 000421, loss = 0.001291
grad_step = 000422, loss = 0.001297
grad_step = 000423, loss = 0.001309
grad_step = 000424, loss = 0.001329
grad_step = 000425, loss = 0.001369
grad_step = 000426, loss = 0.001436
grad_step = 000427, loss = 0.001562
grad_step = 000428, loss = 0.001714
grad_step = 000429, loss = 0.001918
grad_step = 000430, loss = 0.001949
grad_step = 000431, loss = 0.001858
grad_step = 000432, loss = 0.001570
grad_step = 000433, loss = 0.001360
grad_step = 000434, loss = 0.001336
grad_step = 000435, loss = 0.001451
grad_step = 000436, loss = 0.001558
grad_step = 000437, loss = 0.001516
grad_step = 000438, loss = 0.001401
grad_step = 000439, loss = 0.001313
grad_step = 000440, loss = 0.001326
grad_step = 000441, loss = 0.001402
grad_step = 000442, loss = 0.001446
grad_step = 000443, loss = 0.001395
grad_step = 000444, loss = 0.001309
grad_step = 000445, loss = 0.001278
grad_step = 000446, loss = 0.001314
grad_step = 000447, loss = 0.001362
grad_step = 000448, loss = 0.001364
grad_step = 000449, loss = 0.001325
grad_step = 000450, loss = 0.001283
grad_step = 000451, loss = 0.001273
grad_step = 000452, loss = 0.001296
grad_step = 000453, loss = 0.001318
grad_step = 000454, loss = 0.001317
grad_step = 000455, loss = 0.001296
grad_step = 000456, loss = 0.001275
grad_step = 000457, loss = 0.001268
grad_step = 000458, loss = 0.001278
grad_step = 000459, loss = 0.001292
grad_step = 000460, loss = 0.001294
grad_step = 000461, loss = 0.001282
grad_step = 000462, loss = 0.001268
grad_step = 000463, loss = 0.001263
grad_step = 000464, loss = 0.001266
grad_step = 000465, loss = 0.001273
grad_step = 000466, loss = 0.001277
grad_step = 000467, loss = 0.001275
grad_step = 000468, loss = 0.001268
grad_step = 000469, loss = 0.001262
grad_step = 000470, loss = 0.001259
grad_step = 000471, loss = 0.001260
grad_step = 000472, loss = 0.001262
grad_step = 000473, loss = 0.001264
grad_step = 000474, loss = 0.001264
grad_step = 000475, loss = 0.001262
grad_step = 000476, loss = 0.001258
grad_step = 000477, loss = 0.001255
grad_step = 000478, loss = 0.001254
grad_step = 000479, loss = 0.001253
grad_step = 000480, loss = 0.001254
grad_step = 000481, loss = 0.001255
grad_step = 000482, loss = 0.001256
grad_step = 000483, loss = 0.001255
grad_step = 000484, loss = 0.001254
grad_step = 000485, loss = 0.001252
grad_step = 000486, loss = 0.001251
grad_step = 000487, loss = 0.001249
grad_step = 000488, loss = 0.001248
grad_step = 000489, loss = 0.001248
grad_step = 000490, loss = 0.001248
grad_step = 000491, loss = 0.001249
grad_step = 000492, loss = 0.001251
grad_step = 000493, loss = 0.001255
grad_step = 000494, loss = 0.001261
grad_step = 000495, loss = 0.001271
grad_step = 000496, loss = 0.001291
grad_step = 000497, loss = 0.001325
grad_step = 000498, loss = 0.001390
grad_step = 000499, loss = 0.001499
grad_step = 000500, loss = 0.001702
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001992
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

  date_run                              2020-05-12 02:13:54.843065
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.260743
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 02:13:54.849905
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.162078
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 02:13:54.858431
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.157397
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 02:13:54.864656
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.46284
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
0   2020-05-12 02:13:19.144647  ...    mean_absolute_error
1   2020-05-12 02:13:19.149413  ...     mean_squared_error
2   2020-05-12 02:13:19.153361  ...  median_absolute_error
3   2020-05-12 02:13:19.157301  ...               r2_score
4   2020-05-12 02:13:29.300169  ...    mean_absolute_error
5   2020-05-12 02:13:29.304835  ...     mean_squared_error
6   2020-05-12 02:13:29.309045  ...  median_absolute_error
7   2020-05-12 02:13:29.313179  ...               r2_score
8   2020-05-12 02:13:54.843065  ...    mean_absolute_error
9   2020-05-12 02:13:54.849905  ...     mean_squared_error
10  2020-05-12 02:13:54.858431  ...  median_absolute_error
11  2020-05-12 02:13:54.864656  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 317372.86it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 410861.74it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 568521.02it/s] 36%|███▌      | 3522560/9912422 [00:00<00:07, 803083.68it/s] 77%|███████▋  | 7667712/9912422 [00:00<00:01, 1135665.02it/s]9920512it [00:00, 11035474.58it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 150393.55it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 313279.74it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 404548.43it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 560442.05it/s]1654784it [00:00, 2870884.32it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52741.78it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f95c3a7cfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9561199be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f95c3a07ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9560c70048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9576410cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9576400e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9561199e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9576400e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9561199e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9561199be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9561199e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc728d441d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=31475d5951ee9190b7f3dfc04c7fd271130b983be7fe91bb339ed029e9ee008e
  Stored in directory: /tmp/pip-ephem-wheel-cache-8t7iecoj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc6c092c198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
  106496/17464789 [..............................] - ETA: 32s
  196608/17464789 [..............................] - ETA: 23s
  417792/17464789 [..............................] - ETA: 13s
  835584/17464789 [>.............................] - ETA: 7s 
 1695744/17464789 [=>............................] - ETA: 4s
 3366912/17464789 [====>.........................] - ETA: 2s
 6250496/17464789 [=========>....................] - ETA: 1s
 9183232/17464789 [==============>...............] - ETA: 0s
12197888/17464789 [===================>..........] - ETA: 0s
15310848/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 02:15:27.770965: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 02:15:27.775460: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 02:15:27.776058: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557013242c10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 02:15:27.776075: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5900 - accuracy: 0.5050
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5746 - accuracy: 0.5060 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5440 - accuracy: 0.5080
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5838 - accuracy: 0.5054
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5797 - accuracy: 0.5057
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6184 - accuracy: 0.5031
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6530 - accuracy: 0.5009
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6590 - accuracy: 0.5005
11000/25000 [============>.................] - ETA: 4s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6961 - accuracy: 0.4981
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7017 - accuracy: 0.4977
15000/25000 [=================>............] - ETA: 3s - loss: 7.7167 - accuracy: 0.4967
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7343 - accuracy: 0.4956
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7186 - accuracy: 0.4966
19000/25000 [=====================>........] - ETA: 2s - loss: 7.7239 - accuracy: 0.4963
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7142 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7141 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 1s - loss: 7.7029 - accuracy: 0.4976
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6906 - accuracy: 0.4984
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 10s 405us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 02:15:45.922362
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 02:15:45.922362  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 02:15:53.209551: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 02:15:53.215455: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 02:15:53.215690: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55efe3becba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 02:15:53.215709: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fa0b16c8be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1482 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.1469 - val_crf_viterbi_accuracy: 0.2800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa08d1c7e48> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5992 - accuracy: 0.5044
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5637 - accuracy: 0.5067
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6245 - accuracy: 0.5027
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6087 - accuracy: 0.5038
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6222 - accuracy: 0.5029
11000/25000 [============>.................] - ETA: 4s - loss: 7.6332 - accuracy: 0.5022
12000/25000 [=============>................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 3s - loss: 7.6370 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6837 - accuracy: 0.4989
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6942 - accuracy: 0.4982
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6931 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6866 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 10s 414us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fa08c4a7a58> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:37:07, 24.9kB/s].vector_cache/glove.6B.zip:   0%|          | 352k/862M [00:00<6:45:04, 35.5kB/s] .vector_cache/glove.6B.zip:   1%|          | 4.62M/862M [00:00<4:42:15, 50.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:00<3:15:50, 72.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.1M/862M [00:00<2:16:04, 103kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:00<1:34:52, 147kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:00<1:06:04, 210kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:01<45:45, 300kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 47.4M/862M [00:01<31:43, 428kB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.4M/862M [00:02<22:54, 588kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:02<16:33, 812kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<12:10, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<8:31:05, 26.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:04<5:56:30, 37.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<4:16:27, 52.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:06<3:00:28, 73.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<2:07:44, 104kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:08<1:30:01, 147kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<1:04:52, 204kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<46:02, 287kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<34:12, 384kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<24:37, 533kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<19:14, 679kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:14<14:06, 925kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<11:56, 1.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<08:54, 1.46MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:17<08:19, 1.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:18<06:26, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:19<06:35, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:20<05:13, 2.46MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:21<05:43, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<04:36, 2.77MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:23<05:17, 2.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<04:22, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<05:04, 2.49MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<04:10, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<04:57, 2.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<04:05, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<04:52, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<04:02, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<04:50, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<04:00, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<04:47, 2.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<03:58, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<04:45, 2.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<03:56, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:37<04:43, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<03:54, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<04:42, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<04:01, 3.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:15, 3.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<7:50:06, 25.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:41<5:27:32, 36.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<4:00:50, 50.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<2:49:29, 71.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:59:49, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<1:24:23, 142kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<1:00:43, 196kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<43:05, 276kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<31:55, 371kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<22:54, 516kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<17:51, 659kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<13:07, 897kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<10:58, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<08:17, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<07:38, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<06:02, 1.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<05:59, 1.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:54, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<05:10, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<04:39, 2.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [00:59<03:20, 3.43MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<4:09:15, 45.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<2:55:03, 65.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<2:03:38, 91.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<1:27:04, 130kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<1:02:24, 181kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<44:14, 255kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<32:37, 344kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<23:19, 481kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<18:04, 617kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<13:18, 838kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<10:58, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<08:14, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<07:30, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<05:48, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<05:48, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:40, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:40, 2.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<6:58:17, 26.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<4:51:41, 37.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<3:28:04, 52.0kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<2:26:58, 73.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:18<1:42:29, 105kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<1:19:40, 135kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<56:40, 190kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<41:01, 261kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<29:28, 362kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<22:06, 480kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<16:21, 649kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<12:56, 815kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<09:36, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<08:22, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<06:24, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:53, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:30<03:31, 2.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<09:22, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<07:13, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:32<05:07, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:33<20:39, 497kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<15:22, 668kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:34<10:48, 945kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<15:08, 674kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<11:07, 916kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<09:20, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<07:03, 1.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<06:32, 1.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<05:06, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<05:09, 1.94MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<04:24, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<04:02, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:43<02:55, 3.38MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<08:07, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<06:11, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<05:53, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<04:41, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<04:47, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<04:07, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<03:13, 2.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<6:09:10, 26.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<4:17:46, 37.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<3:02:02, 52.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<2:08:06, 74.9kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:30:31, 105kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<1:04:07, 149kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<46:00, 206kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<32:57, 287kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<24:20, 386kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<17:47, 528kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<13:46, 677kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<10:18, 905kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:01<07:14, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<23:47, 389kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<17:34, 527kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:03<12:19, 747kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<15:17, 601kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<11:31, 797kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:05<08:05, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<23:33, 387kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<17:12, 530kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<13:19, 679kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<09:58, 907kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:09<07:00, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<3:18:09, 45.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<2:19:18, 64.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:10<1:37:01, 92.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<1:15:44, 118kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<53:44, 166kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<38:40, 229kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<27:46, 318kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<20:37, 426kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<15:10, 578kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<11:51, 735kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<08:59, 969kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<06:36, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<5:13:51, 27.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<3:39:01, 39.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<2:34:52, 55.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<1:48:59, 78.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<1:17:02, 111kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<54:33, 156kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<39:11, 216kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<28:04, 301kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<20:46, 404kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<15:11, 551kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<11:48, 704kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<09:23, 885kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:30<06:35, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<15:43, 524kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<11:39, 706kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<09:20, 876kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<07:11, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<06:13, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<04:59, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<04:40, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<03:57, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:23, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<03:32, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<03:07, 2.53MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:20, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<02:58, 2.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<03:13, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<02:52, 2.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:09, 2.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<02:48, 2.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<02:14, 3.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<4:44:13, 26.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<3:18:18, 38.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<2:19:55, 54.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<1:38:27, 76.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<1:09:29, 108kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<49:12, 152kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<35:16, 211kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<25:15, 294kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<18:39, 395kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<13:43, 537kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:57<09:36, 761kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<13:05, 557kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<09:44, 748kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<07:51, 921kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<06:04, 1.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<05:17, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<04:16, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<04:01, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<03:20, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:05<02:23, 2.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<06:57, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<05:25, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<04:48, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<03:45, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<03:19, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:10<02:22, 2.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<27:37, 247kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<19:52, 343kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<14:11, 477kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<12:28, 541kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<09:15, 728kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<07:25, 899kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<05:44, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<04:14, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<4:06:10, 26.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<2:51:36, 38.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<2:01:04, 54.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<1:25:11, 76.9kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:00:03, 108kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<42:31, 153kB/s]  .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<30:26, 211kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<22:08, 290kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:24<15:27, 412kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<14:04, 452kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<10:33, 602kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:26<07:23, 852kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<08:36, 730kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<06:56, 904kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:28<04:52, 1.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<08:46, 709kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<06:37, 938kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<05:31, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<04:20, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:32<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<58:20, 104kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<41:30, 146kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:34<28:52, 209kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<24:03, 250kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<17:18, 347kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<12:52, 461kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<09:29, 625kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:38<06:38, 886kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<12:28, 471kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<09:20, 629kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:39<06:32, 890kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<07:55, 733kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<05:56, 976kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:41<04:10, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<08:54, 644kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<06:40, 858kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:43<04:40, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<20:33, 276kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<14:46, 383kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<10:28, 536kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:03:26, 23.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<2:49:56, 32.9kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<1:58:53, 46.7kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<1:23:36, 66.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<58:30, 94.4kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<41:40, 131kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<29:34, 185kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<21:17, 254kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<15:17, 353kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<11:23, 469kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<08:23, 635kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<06:36, 799kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<05:01, 1.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<04:15, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:24, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<03:07, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:35, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<02:33, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<02:11, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:16, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<01:58, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:52, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<02:01, 2.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:45, 2.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:56, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:47, 2.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:10<01:16, 3.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<09:09, 516kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<06:46, 696kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<05:23, 864kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<04:13, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<03:35, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<03:05, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<02:20, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<2:43:33, 27.7kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<1:54:02, 39.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:19:53, 55.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<56:11, 79.3kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<39:27, 111kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<27:55, 157kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:22<19:21, 224kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<17:12, 252kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<12:21, 349kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<09:10, 464kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<06:51, 620kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:26<04:46, 879kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<10:28, 400kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<07:39, 546kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<05:54, 698kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<04:27, 922kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<03:07, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<04:32, 891kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<03:29, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<03:00, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<02:25, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<02:15, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<01:54, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:48, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:37<01:17, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<03:35, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<02:49, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<02:29, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<02:08, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:41<01:30, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<09:09, 397kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<06:52, 529kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:43<04:45, 751kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<08:08, 438kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<06:10, 578kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<04:24, 798kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<2:05:50, 28.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:27:23, 39.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:01:20, 56.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<43:08, 79.8kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<30:09, 112kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<21:19, 158kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<15:09, 219kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<10:51, 305kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<07:56, 408kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<05:49, 556kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<04:28, 710kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<03:27, 917kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:57<02:23, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<11:43, 265kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<08:26, 367kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<06:14, 486kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<04:37, 656kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<03:36, 822kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<02:48, 1.05MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:03<01:57, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<05:38, 514kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<04:09, 695kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:05<02:53, 984kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<06:42, 422kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<05:05, 556kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:06<03:31, 788kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<05:12, 531kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<04:02, 684kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:08<02:47, 968kB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:10<03:58, 678kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<02:58, 901kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<02:26, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:54, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:12<01:19, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<03:26, 742kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<02:36, 978kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<01:53, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<1:35:04, 26.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<1:05:59, 37.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<45:50, 53.2kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<32:14, 75.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:18<22:03, 108kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<18:28, 128kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<13:13, 179kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:20<09:03, 255kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<08:11, 281kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<05:53, 389kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<04:21, 513kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<03:11, 695kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<02:30, 862kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<01:55, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:26<01:19, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<04:41, 445kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<03:24, 611kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<02:38, 769kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<02:00, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:39, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:18, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<01:10, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<00:56, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:55, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<00:45, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:47, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<00:41, 2.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:36, 2.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:41<00:39, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:35, 2.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:37, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:33, 2.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<00:26, 3.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<52:48, 28.3kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<36:15, 40.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<25:01, 56.9kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<17:35, 80.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:47<11:51, 115kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<09:28, 143kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<06:42, 201kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<04:40, 275kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<03:21, 381kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<02:25, 503kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<01:51, 652kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:53<01:15, 924kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<01:57, 585kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<01:27, 782kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<01:07, 957kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:50, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<00:43, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<00:31, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:01<00:17, 2.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:51, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:39, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:33, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<00:25, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:23, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<00:19, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:18, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:15, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:15, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:13, 2.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:13, 2.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<00:11, 2.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:11, 2.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:09, 2.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:06, 3.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<15:27, 26.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<09:44, 37.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<06:25, 53.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<04:26, 75.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:18<02:32, 107kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<02:52, 94.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<01:55, 134kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<01:05, 186kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:44, 263kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:22, 354kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:15, 492kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:06, 633kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:03, 859kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 711/400000 [00:00<00:56, 7106.48it/s]  0%|          | 1424/400000 [00:00<00:56, 7113.03it/s]  1%|          | 2067/400000 [00:00<00:57, 6892.38it/s]  1%|          | 2786/400000 [00:00<00:56, 6977.20it/s]  1%|          | 3478/400000 [00:00<00:56, 6958.25it/s]  1%|          | 4141/400000 [00:00<00:57, 6854.63it/s]  1%|          | 4844/400000 [00:00<00:57, 6905.92it/s]  1%|▏         | 5544/400000 [00:00<00:56, 6933.84it/s]  2%|▏         | 6283/400000 [00:00<00:55, 7061.10it/s]  2%|▏         | 6992/400000 [00:01<00:55, 7067.28it/s]  2%|▏         | 7720/400000 [00:01<00:55, 7126.98it/s]  2%|▏         | 8424/400000 [00:01<00:55, 7098.52it/s]  2%|▏         | 9127/400000 [00:01<00:55, 7076.90it/s]  2%|▏         | 9851/400000 [00:01<00:54, 7124.19it/s]  3%|▎         | 10569/400000 [00:01<00:54, 7138.78it/s]  3%|▎         | 11280/400000 [00:01<00:54, 7123.56it/s]  3%|▎         | 11991/400000 [00:01<00:54, 7111.68it/s]  3%|▎         | 12701/400000 [00:01<00:54, 7042.74it/s]  3%|▎         | 13420/400000 [00:01<00:54, 7083.25it/s]  4%|▎         | 14128/400000 [00:02<00:54, 7078.83it/s]  4%|▎         | 14853/400000 [00:02<00:54, 7126.82it/s]  4%|▍         | 15585/400000 [00:02<00:53, 7182.95it/s]  4%|▍         | 16304/400000 [00:02<00:56, 6838.33it/s]  4%|▍         | 16992/400000 [00:02<00:56, 6773.81it/s]  4%|▍         | 17672/400000 [00:02<00:57, 6697.85it/s]  5%|▍         | 18359/400000 [00:02<00:56, 6746.58it/s]  5%|▍         | 19093/400000 [00:02<00:55, 6913.54it/s]  5%|▍         | 19828/400000 [00:02<00:54, 7036.52it/s]  5%|▌         | 20547/400000 [00:02<00:53, 7079.54it/s]  5%|▌         | 21257/400000 [00:03<00:53, 7068.86it/s]  5%|▌         | 21965/400000 [00:03<00:53, 7071.70it/s]  6%|▌         | 22681/400000 [00:03<00:53, 7097.51it/s]  6%|▌         | 23392/400000 [00:03<00:53, 7070.48it/s]  6%|▌         | 24100/400000 [00:03<00:54, 6957.64it/s]  6%|▌         | 24797/400000 [00:03<00:54, 6831.83it/s]  6%|▋         | 25504/400000 [00:03<00:54, 6899.33it/s]  7%|▋         | 26247/400000 [00:03<00:53, 7047.89it/s]  7%|▋         | 26989/400000 [00:03<00:52, 7155.19it/s]  7%|▋         | 27733/400000 [00:03<00:51, 7237.28it/s]  7%|▋         | 28463/400000 [00:04<00:51, 7254.20it/s]  7%|▋         | 29208/400000 [00:04<00:50, 7309.71it/s]  7%|▋         | 29940/400000 [00:04<00:50, 7303.89it/s]  8%|▊         | 30671/400000 [00:04<00:50, 7275.86it/s]  8%|▊         | 31399/400000 [00:04<00:51, 7215.34it/s]  8%|▊         | 32121/400000 [00:04<00:51, 7143.37it/s]  8%|▊         | 32851/400000 [00:04<00:51, 7188.91it/s]  8%|▊         | 33571/400000 [00:04<00:50, 7185.50it/s]  9%|▊         | 34290/400000 [00:04<00:51, 7099.35it/s]  9%|▉         | 35001/400000 [00:04<00:51, 7025.21it/s]  9%|▉         | 35704/400000 [00:05<00:52, 6969.13it/s]  9%|▉         | 36408/400000 [00:05<00:52, 6989.00it/s]  9%|▉         | 37111/400000 [00:05<00:51, 6997.90it/s]  9%|▉         | 37812/400000 [00:05<00:51, 6993.12it/s] 10%|▉         | 38512/400000 [00:05<00:52, 6911.45it/s] 10%|▉         | 39204/400000 [00:05<00:52, 6851.86it/s] 10%|▉         | 39890/400000 [00:05<00:52, 6849.16it/s] 10%|█         | 40614/400000 [00:05<00:51, 6960.30it/s] 10%|█         | 41311/400000 [00:05<00:51, 6907.96it/s] 11%|█         | 42019/400000 [00:05<00:51, 6956.44it/s] 11%|█         | 42735/400000 [00:06<00:50, 7013.78it/s] 11%|█         | 43455/400000 [00:06<00:50, 7067.59it/s] 11%|█         | 44163/400000 [00:06<00:50, 7050.19it/s] 11%|█         | 44869/400000 [00:06<00:51, 6894.33it/s] 11%|█▏        | 45585/400000 [00:06<00:50, 6970.03it/s] 12%|█▏        | 46287/400000 [00:06<00:50, 6984.18it/s] 12%|█▏        | 47014/400000 [00:06<00:49, 7066.07it/s] 12%|█▏        | 47754/400000 [00:06<00:49, 7161.46it/s] 12%|█▏        | 48484/400000 [00:06<00:48, 7199.43it/s] 12%|█▏        | 49213/400000 [00:06<00:48, 7225.44it/s] 12%|█▏        | 49936/400000 [00:07<00:48, 7199.30it/s] 13%|█▎        | 50674/400000 [00:07<00:48, 7250.43it/s] 13%|█▎        | 51406/400000 [00:07<00:47, 7269.06it/s] 13%|█▎        | 52142/400000 [00:07<00:47, 7293.93it/s] 13%|█▎        | 52874/400000 [00:07<00:47, 7300.25it/s] 13%|█▎        | 53605/400000 [00:07<00:47, 7271.20it/s] 14%|█▎        | 54333/400000 [00:07<00:47, 7273.36it/s] 14%|█▍        | 55067/400000 [00:07<00:47, 7291.27it/s] 14%|█▍        | 55797/400000 [00:07<00:47, 7271.79it/s] 14%|█▍        | 56525/400000 [00:07<00:47, 7261.32it/s] 14%|█▍        | 57252/400000 [00:08<00:48, 7062.56it/s] 14%|█▍        | 57960/400000 [00:08<00:48, 7024.38it/s] 15%|█▍        | 58664/400000 [00:08<00:49, 6859.04it/s] 15%|█▍        | 59382/400000 [00:08<00:49, 6950.55it/s] 15%|█▌        | 60080/400000 [00:08<00:48, 6958.99it/s] 15%|█▌        | 60785/400000 [00:08<00:48, 6986.03it/s] 15%|█▌        | 61485/400000 [00:08<00:49, 6844.56it/s] 16%|█▌        | 62177/400000 [00:08<00:49, 6864.78it/s] 16%|█▌        | 62867/400000 [00:08<00:49, 6874.05it/s] 16%|█▌        | 63579/400000 [00:09<00:48, 6945.80it/s] 16%|█▌        | 64275/400000 [00:09<00:48, 6905.06it/s] 16%|█▌        | 64966/400000 [00:09<00:49, 6818.04it/s] 16%|█▋        | 65649/400000 [00:09<00:49, 6704.01it/s] 17%|█▋        | 66345/400000 [00:09<00:49, 6776.82it/s] 17%|█▋        | 67035/400000 [00:09<00:48, 6812.17it/s] 17%|█▋        | 67720/400000 [00:09<00:48, 6822.85it/s] 17%|█▋        | 68434/400000 [00:09<00:47, 6914.40it/s] 17%|█▋        | 69127/400000 [00:09<00:48, 6879.76it/s] 17%|█▋        | 69849/400000 [00:09<00:47, 6975.73it/s] 18%|█▊        | 70548/400000 [00:10<00:48, 6752.63it/s] 18%|█▊        | 71240/400000 [00:10<00:48, 6801.78it/s] 18%|█▊        | 71929/400000 [00:10<00:48, 6826.71it/s] 18%|█▊        | 72627/400000 [00:10<00:47, 6869.59it/s] 18%|█▊        | 73356/400000 [00:10<00:46, 6989.08it/s] 19%|█▊        | 74076/400000 [00:10<00:46, 7049.65it/s] 19%|█▊        | 74793/400000 [00:10<00:45, 7083.94it/s] 19%|█▉        | 75525/400000 [00:10<00:45, 7149.66it/s] 19%|█▉        | 76241/400000 [00:10<00:46, 6934.94it/s] 19%|█▉        | 76949/400000 [00:10<00:46, 6977.75it/s] 19%|█▉        | 77661/400000 [00:11<00:45, 7018.36it/s] 20%|█▉        | 78364/400000 [00:11<00:45, 7021.40it/s] 20%|█▉        | 79102/400000 [00:11<00:45, 7124.77it/s] 20%|█▉        | 79833/400000 [00:11<00:44, 7171.49it/s] 20%|██        | 80554/400000 [00:11<00:44, 7180.08it/s] 20%|██        | 81273/400000 [00:11<00:44, 7168.80it/s] 20%|██        | 81991/400000 [00:11<00:44, 7066.99it/s] 21%|██        | 82699/400000 [00:11<00:44, 7069.37it/s] 21%|██        | 83411/400000 [00:11<00:44, 7082.84it/s] 21%|██        | 84123/400000 [00:11<00:44, 7093.34it/s] 21%|██        | 84833/400000 [00:12<00:45, 7002.67it/s] 21%|██▏       | 85534/400000 [00:12<00:45, 6916.75it/s] 22%|██▏       | 86227/400000 [00:12<00:45, 6881.71it/s] 22%|██▏       | 86916/400000 [00:12<00:46, 6775.33it/s] 22%|██▏       | 87595/400000 [00:12<00:46, 6734.96it/s] 22%|██▏       | 88295/400000 [00:12<00:45, 6811.35it/s] 22%|██▏       | 88996/400000 [00:12<00:45, 6869.56it/s] 22%|██▏       | 89730/400000 [00:12<00:44, 7003.23it/s] 23%|██▎       | 90463/400000 [00:12<00:43, 7096.07it/s] 23%|██▎       | 91186/400000 [00:12<00:43, 7133.88it/s] 23%|██▎       | 91901/400000 [00:13<00:43, 7103.84it/s] 23%|██▎       | 92612/400000 [00:13<00:44, 6978.95it/s] 23%|██▎       | 93329/400000 [00:13<00:43, 7033.27it/s] 24%|██▎       | 94058/400000 [00:13<00:43, 7106.80it/s] 24%|██▎       | 94780/400000 [00:13<00:42, 7139.80it/s] 24%|██▍       | 95517/400000 [00:13<00:42, 7204.71it/s] 24%|██▍       | 96238/400000 [00:13<00:42, 7203.50it/s] 24%|██▍       | 96968/400000 [00:13<00:41, 7230.88it/s] 24%|██▍       | 97692/400000 [00:13<00:43, 6923.53it/s] 25%|██▍       | 98421/400000 [00:14<00:42, 7027.42it/s] 25%|██▍       | 99154/400000 [00:14<00:42, 7114.35it/s] 25%|██▍       | 99869/400000 [00:14<00:42, 7124.78it/s] 25%|██▌       | 100589/400000 [00:14<00:41, 7146.55it/s] 25%|██▌       | 101313/400000 [00:14<00:41, 7174.06it/s] 26%|██▌       | 102032/400000 [00:14<00:41, 7122.84it/s] 26%|██▌       | 102745/400000 [00:14<00:42, 7053.47it/s] 26%|██▌       | 103451/400000 [00:14<00:42, 6961.75it/s] 26%|██▌       | 104148/400000 [00:14<00:42, 6938.83it/s] 26%|██▌       | 104843/400000 [00:14<00:42, 6893.48it/s] 26%|██▋       | 105533/400000 [00:15<00:43, 6827.77it/s] 27%|██▋       | 106232/400000 [00:15<00:42, 6868.52it/s] 27%|██▋       | 106920/400000 [00:15<00:42, 6854.13it/s] 27%|██▋       | 107618/400000 [00:15<00:42, 6890.51it/s] 27%|██▋       | 108339/400000 [00:15<00:41, 6981.44it/s] 27%|██▋       | 109077/400000 [00:15<00:41, 7095.27it/s] 27%|██▋       | 109804/400000 [00:15<00:40, 7145.03it/s] 28%|██▊       | 110533/400000 [00:15<00:40, 7186.28it/s] 28%|██▊       | 111253/400000 [00:15<00:40, 7081.64it/s] 28%|██▊       | 111962/400000 [00:15<00:40, 7080.40it/s] 28%|██▊       | 112676/400000 [00:16<00:40, 7097.07it/s] 28%|██▊       | 113396/400000 [00:16<00:40, 7125.68it/s] 29%|██▊       | 114111/400000 [00:16<00:40, 7131.59it/s] 29%|██▊       | 114825/400000 [00:16<00:40, 7054.55it/s] 29%|██▉       | 115531/400000 [00:16<00:40, 6943.37it/s] 29%|██▉       | 116249/400000 [00:16<00:40, 7011.82it/s] 29%|██▉       | 116976/400000 [00:16<00:39, 7085.24it/s] 29%|██▉       | 117698/400000 [00:16<00:39, 7123.02it/s] 30%|██▉       | 118443/400000 [00:16<00:39, 7217.40it/s] 30%|██▉       | 119179/400000 [00:16<00:38, 7259.00it/s] 30%|██▉       | 119924/400000 [00:17<00:38, 7314.90it/s] 30%|███       | 120659/400000 [00:17<00:38, 7324.94it/s] 30%|███       | 121392/400000 [00:17<00:38, 7200.28it/s] 31%|███       | 122113/400000 [00:17<00:38, 7190.27it/s] 31%|███       | 122833/400000 [00:17<00:40, 6911.70it/s] 31%|███       | 123568/400000 [00:17<00:39, 7036.50it/s] 31%|███       | 124295/400000 [00:17<00:38, 7103.91it/s] 31%|███▏      | 125027/400000 [00:17<00:38, 7165.14it/s] 31%|███▏      | 125760/400000 [00:17<00:38, 7212.68it/s] 32%|███▏      | 126502/400000 [00:17<00:37, 7272.78it/s] 32%|███▏      | 127238/400000 [00:18<00:37, 7292.57it/s] 32%|███▏      | 127968/400000 [00:18<00:37, 7266.53it/s] 32%|███▏      | 128696/400000 [00:18<00:37, 7195.30it/s] 32%|███▏      | 129417/400000 [00:18<00:38, 6980.43it/s] 33%|███▎      | 130142/400000 [00:18<00:38, 7057.42it/s] 33%|███▎      | 130850/400000 [00:18<00:38, 6965.39it/s] 33%|███▎      | 131548/400000 [00:18<00:38, 6903.09it/s] 33%|███▎      | 132246/400000 [00:18<00:38, 6923.82it/s] 33%|███▎      | 132965/400000 [00:18<00:38, 6999.96it/s] 33%|███▎      | 133684/400000 [00:18<00:37, 7053.23it/s] 34%|███▎      | 134390/400000 [00:19<00:37, 7022.93it/s] 34%|███▍      | 135093/400000 [00:19<00:38, 6814.35it/s] 34%|███▍      | 135795/400000 [00:19<00:38, 6874.18it/s] 34%|███▍      | 136484/400000 [00:19<00:38, 6869.02it/s] 34%|███▍      | 137183/400000 [00:19<00:38, 6902.37it/s] 34%|███▍      | 137879/400000 [00:19<00:37, 6917.27it/s] 35%|███▍      | 138603/400000 [00:19<00:37, 7009.52it/s] 35%|███▍      | 139305/400000 [00:19<00:38, 6846.04it/s] 35%|███▌      | 140047/400000 [00:19<00:37, 7007.53it/s] 35%|███▌      | 140782/400000 [00:20<00:36, 7106.28it/s] 35%|███▌      | 141516/400000 [00:20<00:36, 7174.58it/s] 36%|███▌      | 142257/400000 [00:20<00:35, 7241.97it/s] 36%|███▌      | 142991/400000 [00:20<00:35, 7270.05it/s] 36%|███▌      | 143733/400000 [00:20<00:35, 7314.15it/s] 36%|███▌      | 144479/400000 [00:20<00:34, 7357.18it/s] 36%|███▋      | 145224/400000 [00:20<00:34, 7382.24it/s] 36%|███▋      | 145963/400000 [00:20<00:34, 7383.32it/s] 37%|███▋      | 146702/400000 [00:20<00:34, 7372.58it/s] 37%|███▋      | 147442/400000 [00:20<00:34, 7379.45it/s] 37%|███▋      | 148187/400000 [00:21<00:34, 7400.02it/s] 37%|███▋      | 148934/400000 [00:21<00:33, 7418.20it/s] 37%|███▋      | 149678/400000 [00:21<00:33, 7424.08it/s] 38%|███▊      | 150421/400000 [00:21<00:33, 7392.06it/s] 38%|███▊      | 151161/400000 [00:21<00:33, 7355.57it/s] 38%|███▊      | 151903/400000 [00:21<00:33, 7373.07it/s] 38%|███▊      | 152641/400000 [00:21<00:33, 7365.69it/s] 38%|███▊      | 153379/400000 [00:21<00:33, 7367.80it/s] 39%|███▊      | 154116/400000 [00:21<00:33, 7345.25it/s] 39%|███▊      | 154857/400000 [00:21<00:33, 7363.41it/s] 39%|███▉      | 155599/400000 [00:22<00:33, 7378.84it/s] 39%|███▉      | 156337/400000 [00:22<00:33, 7367.97it/s] 39%|███▉      | 157079/400000 [00:22<00:32, 7383.31it/s] 39%|███▉      | 157818/400000 [00:22<00:33, 7310.86it/s] 40%|███▉      | 158550/400000 [00:22<00:34, 7024.46it/s] 40%|███▉      | 159274/400000 [00:22<00:33, 7087.34it/s] 40%|████      | 160013/400000 [00:22<00:33, 7174.40it/s] 40%|████      | 160756/400000 [00:22<00:33, 7247.81it/s] 40%|████      | 161483/400000 [00:22<00:33, 7224.99it/s] 41%|████      | 162207/400000 [00:22<00:32, 7209.60it/s] 41%|████      | 162950/400000 [00:23<00:32, 7272.72it/s] 41%|████      | 163690/400000 [00:23<00:32, 7309.47it/s] 41%|████      | 164430/400000 [00:23<00:32, 7334.34it/s] 41%|████▏     | 165164/400000 [00:23<00:32, 7317.80it/s] 41%|████▏     | 165907/400000 [00:23<00:31, 7348.31it/s] 42%|████▏     | 166649/400000 [00:23<00:31, 7368.16it/s] 42%|████▏     | 167392/400000 [00:23<00:31, 7385.35it/s] 42%|████▏     | 168137/400000 [00:23<00:31, 7401.27it/s] 42%|████▏     | 168878/400000 [00:23<00:31, 7360.91it/s] 42%|████▏     | 169625/400000 [00:23<00:31, 7390.54it/s] 43%|████▎     | 170368/400000 [00:24<00:31, 7399.85it/s] 43%|████▎     | 171109/400000 [00:24<00:30, 7398.40it/s] 43%|████▎     | 171852/400000 [00:24<00:30, 7406.02it/s] 43%|████▎     | 172593/400000 [00:24<00:30, 7370.91it/s] 43%|████▎     | 173331/400000 [00:24<00:32, 7041.37it/s] 44%|████▎     | 174069/400000 [00:24<00:31, 7138.13it/s] 44%|████▎     | 174806/400000 [00:24<00:31, 7205.57it/s] 44%|████▍     | 175543/400000 [00:24<00:30, 7252.53it/s] 44%|████▍     | 176274/400000 [00:24<00:30, 7267.58it/s] 44%|████▍     | 177013/400000 [00:24<00:30, 7303.76it/s] 44%|████▍     | 177755/400000 [00:25<00:30, 7338.16it/s] 45%|████▍     | 178500/400000 [00:25<00:30, 7370.80it/s] 45%|████▍     | 179247/400000 [00:25<00:29, 7397.96it/s] 45%|████▍     | 179988/400000 [00:25<00:29, 7385.93it/s] 45%|████▌     | 180728/400000 [00:25<00:29, 7387.38it/s] 45%|████▌     | 181467/400000 [00:25<00:29, 7386.99it/s] 46%|████▌     | 182211/400000 [00:25<00:29, 7400.70it/s] 46%|████▌     | 182955/400000 [00:25<00:29, 7412.05it/s] 46%|████▌     | 183697/400000 [00:25<00:29, 7382.17it/s] 46%|████▌     | 184441/400000 [00:25<00:29, 7397.98it/s] 46%|████▋     | 185183/400000 [00:26<00:29, 7402.34it/s] 46%|████▋     | 185924/400000 [00:26<00:28, 7386.95it/s] 47%|████▋     | 186669/400000 [00:26<00:28, 7403.08it/s] 47%|████▋     | 187410/400000 [00:26<00:28, 7383.52it/s] 47%|████▋     | 188154/400000 [00:26<00:28, 7398.95it/s] 47%|████▋     | 188897/400000 [00:26<00:28, 7408.00it/s] 47%|████▋     | 189638/400000 [00:26<00:28, 7405.82it/s] 48%|████▊     | 190379/400000 [00:26<00:28, 7321.37it/s] 48%|████▊     | 191112/400000 [00:26<00:28, 7299.86it/s] 48%|████▊     | 191845/400000 [00:26<00:28, 7307.01it/s] 48%|████▊     | 192576/400000 [00:27<00:28, 7297.24it/s] 48%|████▊     | 193306/400000 [00:27<00:28, 7280.54it/s] 49%|████▊     | 194035/400000 [00:27<00:28, 7270.38it/s] 49%|████▊     | 194763/400000 [00:27<00:28, 7198.15it/s] 49%|████▉     | 195484/400000 [00:27<00:28, 7151.85it/s] 49%|████▉     | 196224/400000 [00:27<00:28, 7222.68it/s] 49%|████▉     | 196962/400000 [00:27<00:27, 7267.91it/s] 49%|████▉     | 197700/400000 [00:27<00:27, 7299.53it/s] 50%|████▉     | 198431/400000 [00:27<00:27, 7230.51it/s] 50%|████▉     | 199155/400000 [00:27<00:27, 7215.07it/s] 50%|████▉     | 199877/400000 [00:28<00:27, 7187.92it/s] 50%|█████     | 200596/400000 [00:28<00:27, 7136.86it/s] 50%|█████     | 201310/400000 [00:28<00:27, 7128.29it/s] 51%|█████     | 202023/400000 [00:28<00:27, 7078.96it/s] 51%|█████     | 202732/400000 [00:28<00:28, 6851.25it/s] 51%|█████     | 203429/400000 [00:28<00:28, 6884.08it/s] 51%|█████     | 204119/400000 [00:28<00:28, 6843.50it/s] 51%|█████     | 204817/400000 [00:28<00:28, 6880.88it/s] 51%|█████▏    | 205506/400000 [00:28<00:28, 6811.33it/s] 52%|█████▏    | 206215/400000 [00:28<00:28, 6890.65it/s] 52%|█████▏    | 206913/400000 [00:29<00:27, 6917.14it/s] 52%|█████▏    | 207606/400000 [00:29<00:28, 6828.65it/s] 52%|█████▏    | 208294/400000 [00:29<00:28, 6841.55it/s] 52%|█████▏    | 209020/400000 [00:29<00:27, 6961.09it/s] 52%|█████▏    | 209717/400000 [00:29<00:27, 6929.40it/s] 53%|█████▎    | 210422/400000 [00:29<00:27, 6963.93it/s] 53%|█████▎    | 211119/400000 [00:29<00:28, 6647.05it/s] 53%|█████▎    | 211803/400000 [00:29<00:28, 6703.71it/s] 53%|█████▎    | 212488/400000 [00:29<00:27, 6746.53it/s] 53%|█████▎    | 213201/400000 [00:30<00:27, 6854.75it/s] 53%|█████▎    | 213913/400000 [00:30<00:26, 6930.21it/s] 54%|█████▎    | 214638/400000 [00:30<00:26, 7022.68it/s] 54%|█████▍    | 215349/400000 [00:30<00:26, 7047.37it/s] 54%|█████▍    | 216055/400000 [00:30<00:26, 6884.81it/s] 54%|█████▍    | 216793/400000 [00:30<00:26, 7024.05it/s] 54%|█████▍    | 217529/400000 [00:30<00:25, 7120.24it/s] 55%|█████▍    | 218266/400000 [00:30<00:25, 7192.58it/s] 55%|█████▍    | 219009/400000 [00:30<00:24, 7261.53it/s] 55%|█████▍    | 219737/400000 [00:30<00:24, 7249.24it/s] 55%|█████▌    | 220466/400000 [00:31<00:24, 7259.33it/s] 55%|█████▌    | 221204/400000 [00:31<00:24, 7293.95it/s] 55%|█████▌    | 221949/400000 [00:31<00:24, 7337.33it/s] 56%|█████▌    | 222687/400000 [00:31<00:24, 7349.38it/s] 56%|█████▌    | 223423/400000 [00:31<00:24, 7301.78it/s] 56%|█████▌    | 224154/400000 [00:31<00:24, 7222.32it/s] 56%|█████▌    | 224877/400000 [00:31<00:24, 7090.00it/s] 56%|█████▋    | 225587/400000 [00:31<00:24, 7071.00it/s] 57%|█████▋    | 226313/400000 [00:31<00:24, 7125.95it/s] 57%|█████▋    | 227027/400000 [00:31<00:24, 7097.94it/s] 57%|█████▋    | 227752/400000 [00:32<00:24, 7140.39it/s] 57%|█████▋    | 228495/400000 [00:32<00:23, 7222.85it/s] 57%|█████▋    | 229235/400000 [00:32<00:23, 7272.97it/s] 57%|█████▋    | 229963/400000 [00:32<00:23, 7238.67it/s] 58%|█████▊    | 230688/400000 [00:32<00:25, 6727.26it/s] 58%|█████▊    | 231416/400000 [00:32<00:24, 6883.04it/s] 58%|█████▊    | 232138/400000 [00:32<00:24, 6980.71it/s] 58%|█████▊    | 232864/400000 [00:32<00:23, 7061.32it/s] 58%|█████▊    | 233591/400000 [00:32<00:23, 7121.70it/s] 59%|█████▊    | 234307/400000 [00:32<00:23, 7132.69it/s] 59%|█████▉    | 235049/400000 [00:33<00:22, 7214.78it/s] 59%|█████▉    | 235772/400000 [00:33<00:22, 7173.02it/s] 59%|█████▉    | 236491/400000 [00:33<00:22, 7176.88it/s] 59%|█████▉    | 237210/400000 [00:33<00:22, 7159.44it/s] 59%|█████▉    | 237927/400000 [00:33<00:22, 7105.45it/s] 60%|█████▉    | 238643/400000 [00:33<00:22, 7120.45it/s] 60%|█████▉    | 239356/400000 [00:33<00:23, 6982.50it/s] 60%|██████    | 240056/400000 [00:33<00:23, 6795.45it/s] 60%|██████    | 240743/400000 [00:33<00:23, 6816.91it/s] 60%|██████    | 241437/400000 [00:33<00:23, 6852.93it/s] 61%|██████    | 242124/400000 [00:34<00:23, 6822.48it/s] 61%|██████    | 242807/400000 [00:34<00:23, 6770.37it/s] 61%|██████    | 243485/400000 [00:34<00:23, 6712.28it/s] 61%|██████    | 244189/400000 [00:34<00:22, 6803.78it/s] 61%|██████    | 244882/400000 [00:34<00:22, 6839.97it/s] 61%|██████▏   | 245570/400000 [00:34<00:22, 6849.91it/s] 62%|██████▏   | 246256/400000 [00:34<00:22, 6833.53it/s] 62%|██████▏   | 246945/400000 [00:34<00:22, 6849.02it/s] 62%|██████▏   | 247674/400000 [00:34<00:21, 6974.46it/s] 62%|██████▏   | 248386/400000 [00:35<00:21, 7017.01it/s] 62%|██████▏   | 249107/400000 [00:35<00:21, 7071.30it/s] 62%|██████▏   | 249831/400000 [00:35<00:21, 7120.03it/s] 63%|██████▎   | 250549/400000 [00:35<00:20, 7136.54it/s] 63%|██████▎   | 251266/400000 [00:35<00:20, 7143.93it/s] 63%|██████▎   | 251981/400000 [00:35<00:20, 7139.25it/s] 63%|██████▎   | 252710/400000 [00:35<00:20, 7181.18it/s] 63%|██████▎   | 253429/400000 [00:35<00:20, 7106.66it/s] 64%|██████▎   | 254153/400000 [00:35<00:20, 7145.80it/s] 64%|██████▎   | 254877/400000 [00:35<00:20, 7171.89it/s] 64%|██████▍   | 255600/400000 [00:36<00:20, 7188.05it/s] 64%|██████▍   | 256319/400000 [00:36<00:19, 7187.39it/s] 64%|██████▍   | 257038/400000 [00:36<00:19, 7181.39it/s] 64%|██████▍   | 257761/400000 [00:36<00:19, 7194.78it/s] 65%|██████▍   | 258494/400000 [00:36<00:19, 7232.33it/s] 65%|██████▍   | 259218/400000 [00:36<00:19, 7217.41it/s] 65%|██████▍   | 259940/400000 [00:36<00:19, 7135.41it/s] 65%|██████▌   | 260654/400000 [00:36<00:20, 6830.59it/s] 65%|██████▌   | 261341/400000 [00:36<00:20, 6776.23it/s] 66%|██████▌   | 262021/400000 [00:36<00:20, 6723.55it/s] 66%|██████▌   | 262698/400000 [00:37<00:20, 6734.83it/s] 66%|██████▌   | 263373/400000 [00:37<00:20, 6716.39it/s] 66%|██████▌   | 264051/400000 [00:37<00:20, 6733.32it/s] 66%|██████▌   | 264778/400000 [00:37<00:19, 6885.80it/s] 66%|██████▋   | 265493/400000 [00:37<00:19, 6960.77it/s] 67%|██████▋   | 266218/400000 [00:37<00:18, 7043.92it/s] 67%|██████▋   | 266924/400000 [00:37<00:18, 7035.48it/s] 67%|██████▋   | 267629/400000 [00:37<00:18, 7036.74it/s] 67%|██████▋   | 268349/400000 [00:37<00:18, 7083.74it/s] 67%|██████▋   | 269058/400000 [00:37<00:18, 7044.85it/s] 67%|██████▋   | 269763/400000 [00:38<00:18, 7043.75it/s] 68%|██████▊   | 270468/400000 [00:38<00:18, 7043.48it/s] 68%|██████▊   | 271174/400000 [00:38<00:18, 7048.20it/s] 68%|██████▊   | 271879/400000 [00:38<00:18, 7011.55it/s] 68%|██████▊   | 272593/400000 [00:38<00:18, 7047.48it/s] 68%|██████▊   | 273317/400000 [00:38<00:17, 7102.09it/s] 69%|██████▊   | 274028/400000 [00:38<00:17, 7075.52it/s] 69%|██████▊   | 274736/400000 [00:38<00:18, 6846.62it/s] 69%|██████▉   | 275423/400000 [00:38<00:18, 6826.43it/s] 69%|██████▉   | 276107/400000 [00:38<00:18, 6830.48it/s] 69%|██████▉   | 276808/400000 [00:39<00:17, 6882.62it/s] 69%|██████▉   | 277514/400000 [00:39<00:17, 6933.31it/s] 70%|██████▉   | 278241/400000 [00:39<00:17, 7028.68it/s] 70%|██████▉   | 278969/400000 [00:39<00:17, 7101.68it/s] 70%|██████▉   | 279680/400000 [00:39<00:17, 6944.70it/s] 70%|███████   | 280376/400000 [00:39<00:17, 6903.78it/s] 70%|███████   | 281068/400000 [00:39<00:17, 6886.04it/s] 70%|███████   | 281759/400000 [00:39<00:17, 6892.58it/s] 71%|███████   | 282451/400000 [00:39<00:17, 6898.58it/s] 71%|███████   | 283159/400000 [00:39<00:16, 6949.78it/s] 71%|███████   | 283855/400000 [00:40<00:16, 6886.46it/s] 71%|███████   | 284545/400000 [00:40<00:17, 6776.80it/s] 71%|███████▏  | 285231/400000 [00:40<00:16, 6799.48it/s] 71%|███████▏  | 285912/400000 [00:40<00:16, 6795.48it/s] 72%|███████▏  | 286592/400000 [00:40<00:17, 6533.32it/s] 72%|███████▏  | 287286/400000 [00:40<00:16, 6648.14it/s] 72%|███████▏  | 287980/400000 [00:40<00:16, 6731.53it/s] 72%|███████▏  | 288655/400000 [00:40<00:16, 6673.56it/s] 72%|███████▏  | 289347/400000 [00:40<00:16, 6743.67it/s] 73%|███████▎  | 290045/400000 [00:40<00:16, 6811.37it/s] 73%|███████▎  | 290734/400000 [00:41<00:15, 6831.55it/s] 73%|███████▎  | 291421/400000 [00:41<00:15, 6841.99it/s] 73%|███████▎  | 292106/400000 [00:41<00:15, 6837.84it/s] 73%|███████▎  | 292791/400000 [00:41<00:15, 6825.14it/s] 73%|███████▎  | 293474/400000 [00:41<00:15, 6824.69it/s] 74%|███████▎  | 294160/400000 [00:41<00:15, 6834.62it/s] 74%|███████▎  | 294844/400000 [00:41<00:15, 6741.59it/s] 74%|███████▍  | 295535/400000 [00:41<00:15, 6790.77it/s] 74%|███████▍  | 296217/400000 [00:41<00:15, 6797.74it/s] 74%|███████▍  | 296908/400000 [00:41<00:15, 6830.52it/s] 74%|███████▍  | 297620/400000 [00:42<00:14, 6914.25it/s] 75%|███████▍  | 298312/400000 [00:42<00:14, 6837.97it/s] 75%|███████▍  | 298997/400000 [00:42<00:14, 6803.03it/s] 75%|███████▍  | 299693/400000 [00:42<00:14, 6847.80it/s] 75%|███████▌  | 300379/400000 [00:42<00:14, 6791.30it/s] 75%|███████▌  | 301074/400000 [00:42<00:14, 6834.68it/s] 75%|███████▌  | 301771/400000 [00:42<00:14, 6873.85it/s] 76%|███████▌  | 302459/400000 [00:42<00:14, 6734.19it/s] 76%|███████▌  | 303150/400000 [00:42<00:14, 6784.02it/s] 76%|███████▌  | 303869/400000 [00:43<00:13, 6899.70it/s] 76%|███████▌  | 304574/400000 [00:43<00:13, 6943.88it/s] 76%|███████▋  | 305285/400000 [00:43<00:13, 6990.29it/s] 76%|███████▋  | 305985/400000 [00:43<00:13, 6868.91it/s] 77%|███████▋  | 306679/400000 [00:43<00:13, 6889.32it/s] 77%|███████▋  | 307369/400000 [00:43<00:13, 6702.75it/s] 77%|███████▋  | 308054/400000 [00:43<00:13, 6745.44it/s] 77%|███████▋  | 308744/400000 [00:43<00:13, 6788.65it/s] 77%|███████▋  | 309436/400000 [00:43<00:13, 6824.07it/s] 78%|███████▊  | 310137/400000 [00:43<00:13, 6876.98it/s] 78%|███████▊  | 310836/400000 [00:44<00:12, 6909.13it/s] 78%|███████▊  | 311545/400000 [00:44<00:12, 6961.02it/s] 78%|███████▊  | 312263/400000 [00:44<00:12, 7023.52it/s] 78%|███████▊  | 312992/400000 [00:44<00:12, 7098.13it/s] 78%|███████▊  | 313728/400000 [00:44<00:12, 7172.01it/s] 79%|███████▊  | 314463/400000 [00:44<00:11, 7224.08it/s] 79%|███████▉  | 315189/400000 [00:44<00:11, 7233.79it/s] 79%|███████▉  | 315913/400000 [00:44<00:11, 7205.96it/s] 79%|███████▉  | 316640/400000 [00:44<00:11, 7223.75it/s] 79%|███████▉  | 317363/400000 [00:44<00:11, 7054.69it/s] 80%|███████▉  | 318070/400000 [00:45<00:11, 7026.70it/s] 80%|███████▉  | 318790/400000 [00:45<00:11, 7077.50it/s] 80%|███████▉  | 319521/400000 [00:45<00:11, 7143.47it/s] 80%|████████  | 320255/400000 [00:45<00:11, 7199.23it/s] 80%|████████  | 320988/400000 [00:45<00:10, 7236.58it/s] 80%|████████  | 321724/400000 [00:45<00:10, 7270.95it/s] 81%|████████  | 322455/400000 [00:45<00:10, 7281.68it/s] 81%|████████  | 323184/400000 [00:45<00:10, 7267.00it/s] 81%|████████  | 323921/400000 [00:45<00:10, 7296.82it/s] 81%|████████  | 324655/400000 [00:45<00:10, 7307.43it/s] 81%|████████▏ | 325386/400000 [00:46<00:10, 7302.52it/s] 82%|████████▏ | 326117/400000 [00:46<00:10, 7282.70it/s] 82%|████████▏ | 326846/400000 [00:46<00:10, 7238.33it/s] 82%|████████▏ | 327570/400000 [00:46<00:10, 7216.79it/s] 82%|████████▏ | 328296/400000 [00:46<00:09, 7227.59it/s] 82%|████████▏ | 329035/400000 [00:46<00:09, 7273.76it/s] 82%|████████▏ | 329773/400000 [00:46<00:09, 7303.25it/s] 83%|████████▎ | 330504/400000 [00:46<00:09, 7303.53it/s] 83%|████████▎ | 331242/400000 [00:46<00:09, 7325.47it/s] 83%|████████▎ | 331975/400000 [00:46<00:09, 7302.56it/s] 83%|████████▎ | 332706/400000 [00:47<00:09, 7297.75it/s] 83%|████████▎ | 333436/400000 [00:47<00:09, 7297.90it/s] 84%|████████▎ | 334166/400000 [00:47<00:09, 7229.68it/s] 84%|████████▎ | 334890/400000 [00:47<00:09, 7107.46it/s] 84%|████████▍ | 335616/400000 [00:47<00:09, 7149.98it/s] 84%|████████▍ | 336332/400000 [00:47<00:08, 7088.25it/s] 84%|████████▍ | 337042/400000 [00:47<00:09, 6929.28it/s] 84%|████████▍ | 337737/400000 [00:47<00:09, 6900.11it/s] 85%|████████▍ | 338428/400000 [00:47<00:08, 6886.92it/s] 85%|████████▍ | 339118/400000 [00:47<00:08, 6870.40it/s] 85%|████████▍ | 339819/400000 [00:48<00:08, 6910.47it/s] 85%|████████▌ | 340520/400000 [00:48<00:08, 6938.43it/s] 85%|████████▌ | 341215/400000 [00:48<00:08, 6900.17it/s] 85%|████████▌ | 341917/400000 [00:48<00:08, 6934.32it/s] 86%|████████▌ | 342623/400000 [00:48<00:08, 6971.50it/s] 86%|████████▌ | 343321/400000 [00:48<00:08, 6860.08it/s] 86%|████████▌ | 344008/400000 [00:48<00:08, 6701.77it/s] 86%|████████▌ | 344685/400000 [00:48<00:08, 6722.09it/s] 86%|████████▋ | 345365/400000 [00:48<00:08, 6743.02it/s] 87%|████████▋ | 346051/400000 [00:48<00:07, 6777.54it/s] 87%|████████▋ | 346758/400000 [00:49<00:07, 6860.64it/s] 87%|████████▋ | 347451/400000 [00:49<00:07, 6881.27it/s] 87%|████████▋ | 348149/400000 [00:49<00:07, 6907.88it/s] 87%|████████▋ | 348843/400000 [00:49<00:07, 6915.01it/s] 87%|████████▋ | 349536/400000 [00:49<00:07, 6918.69it/s] 88%|████████▊ | 350229/400000 [00:49<00:07, 6729.35it/s] 88%|████████▊ | 350935/400000 [00:49<00:07, 6821.53it/s] 88%|████████▊ | 351655/400000 [00:49<00:06, 6929.09it/s] 88%|████████▊ | 352374/400000 [00:49<00:06, 7004.80it/s] 88%|████████▊ | 353078/400000 [00:49<00:06, 7012.54it/s] 88%|████████▊ | 353802/400000 [00:50<00:06, 7077.83it/s] 89%|████████▊ | 354527/400000 [00:50<00:06, 7124.92it/s] 89%|████████▉ | 355241/400000 [00:50<00:06, 7125.67it/s] 89%|████████▉ | 355959/400000 [00:50<00:06, 7140.49it/s] 89%|████████▉ | 356681/400000 [00:50<00:06, 7162.27it/s] 89%|████████▉ | 357419/400000 [00:50<00:05, 7225.88it/s] 90%|████████▉ | 358153/400000 [00:50<00:05, 7259.14it/s] 90%|████████▉ | 358882/400000 [00:50<00:05, 7267.75it/s] 90%|████████▉ | 359609/400000 [00:50<00:05, 6999.39it/s] 90%|█████████ | 360312/400000 [00:51<00:05, 6919.14it/s] 90%|█████████ | 361006/400000 [00:51<00:05, 6858.36it/s] 90%|█████████ | 361705/400000 [00:51<00:05, 6894.85it/s] 91%|█████████ | 362396/400000 [00:51<00:05, 6841.30it/s] 91%|█████████ | 363094/400000 [00:51<00:05, 6881.39it/s] 91%|█████████ | 363795/400000 [00:51<00:05, 6918.14it/s] 91%|█████████ | 364488/400000 [00:51<00:05, 6874.32it/s] 91%|█████████▏| 365176/400000 [00:51<00:05, 6825.48it/s] 91%|█████████▏| 365865/400000 [00:51<00:04, 6843.39it/s] 92%|█████████▏| 366580/400000 [00:51<00:04, 6930.28it/s] 92%|█████████▏| 367275/400000 [00:52<00:04, 6935.81it/s] 92%|█████████▏| 368011/400000 [00:52<00:04, 7057.79it/s] 92%|█████████▏| 368748/400000 [00:52<00:04, 7147.83it/s] 92%|█████████▏| 369464/400000 [00:52<00:04, 7124.76it/s] 93%|█████████▎| 370185/400000 [00:52<00:04, 7149.12it/s] 93%|█████████▎| 370923/400000 [00:52<00:04, 7215.75it/s] 93%|█████████▎| 371659/400000 [00:52<00:03, 7256.09it/s] 93%|█████████▎| 372391/400000 [00:52<00:03, 7274.17it/s] 93%|█████████▎| 373119/400000 [00:52<00:03, 7198.75it/s] 93%|█████████▎| 373840/400000 [00:52<00:03, 7057.49it/s] 94%|█████████▎| 374576/400000 [00:53<00:03, 7145.35it/s] 94%|█████████▍| 375292/400000 [00:53<00:03, 7139.79it/s] 94%|█████████▍| 376014/400000 [00:53<00:03, 7162.61it/s] 94%|█████████▍| 376731/400000 [00:53<00:03, 7064.09it/s] 94%|█████████▍| 377439/400000 [00:53<00:03, 7015.39it/s] 95%|█████████▍| 378142/400000 [00:53<00:03, 6874.40it/s] 95%|█████████▍| 378831/400000 [00:53<00:03, 6878.13it/s] 95%|█████████▍| 379520/400000 [00:53<00:02, 6876.20it/s] 95%|█████████▌| 380209/400000 [00:53<00:02, 6873.75it/s] 95%|█████████▌| 380897/400000 [00:53<00:02, 6848.67it/s] 95%|█████████▌| 381590/400000 [00:54<00:02, 6872.02it/s] 96%|█████████▌| 382292/400000 [00:54<00:02, 6914.57it/s] 96%|█████████▌| 383000/400000 [00:54<00:02, 6960.80it/s] 96%|█████████▌| 383697/400000 [00:54<00:02, 6865.58it/s] 96%|█████████▌| 384385/400000 [00:54<00:02, 6783.36it/s] 96%|█████████▋| 385124/400000 [00:54<00:02, 6953.45it/s] 96%|█████████▋| 385859/400000 [00:54<00:02, 7066.12it/s] 97%|█████████▋| 386568/400000 [00:54<00:01, 6878.38it/s] 97%|█████████▋| 387258/400000 [00:54<00:01, 6824.09it/s] 97%|█████████▋| 387953/400000 [00:54<00:01, 6860.97it/s] 97%|█████████▋| 388641/400000 [00:55<00:01, 6818.32it/s] 97%|█████████▋| 389343/400000 [00:55<00:01, 6875.73it/s] 98%|█████████▊| 390054/400000 [00:55<00:01, 6941.84it/s] 98%|█████████▊| 390749/400000 [00:55<00:01, 6924.23it/s] 98%|█████████▊| 391460/400000 [00:55<00:01, 6978.38it/s] 98%|█████████▊| 392203/400000 [00:55<00:01, 7105.27it/s] 98%|█████████▊| 392915/400000 [00:55<00:01, 7063.45it/s] 98%|█████████▊| 393629/400000 [00:55<00:00, 7085.85it/s] 99%|█████████▊| 394347/400000 [00:55<00:00, 7113.54it/s] 99%|█████████▉| 395061/400000 [00:55<00:00, 7121.11it/s] 99%|█████████▉| 395774/400000 [00:56<00:00, 7072.80it/s] 99%|█████████▉| 396516/400000 [00:56<00:00, 7170.85it/s] 99%|█████████▉| 397253/400000 [00:56<00:00, 7228.31it/s] 99%|█████████▉| 397982/400000 [00:56<00:00, 7246.04it/s]100%|█████████▉| 398717/400000 [00:56<00:00, 7274.50it/s]100%|█████████▉| 399445/400000 [00:56<00:00, 7270.52it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7059.44it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa052125f28> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011425291519819922 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.011832757737724279 	 Accuracy: 49

  model saves at 49% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15710 out of table with 15686 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15710 out of table with 15686 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

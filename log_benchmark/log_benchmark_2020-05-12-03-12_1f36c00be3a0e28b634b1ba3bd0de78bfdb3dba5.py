
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f58837affd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 03:13:08.437059
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 03:13:08.442520
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 03:13:08.447530
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 03:13:08.451962
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f588f7c74a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353259.0000
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 226217.6562
Epoch 3/10

1/1 [==============================] - 0s 119ms/step - loss: 114817.1328
Epoch 4/10

1/1 [==============================] - 0s 117ms/step - loss: 47982.8867
Epoch 5/10

1/1 [==============================] - 0s 112ms/step - loss: 22259.0137
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 12373.9531
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 7852.7935
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 5517.5352
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 4174.2241
Epoch 10/10

1/1 [==============================] - 0s 108ms/step - loss: 3359.0251

  #### Inference Need return ypred, ytrue ######################### 
[[-5.56716919e-01  1.15477304e+01  1.14553614e+01  1.23501863e+01
   1.10678253e+01  1.43109121e+01  1.16190405e+01  1.53085070e+01
   1.38229218e+01  1.33800249e+01  1.27297535e+01  1.22406588e+01
   1.42613306e+01  1.09169493e+01  1.35974350e+01  1.33433990e+01
   1.10253296e+01  1.18033657e+01  1.36490278e+01  1.15005674e+01
   1.26777849e+01  1.43213053e+01  1.23580360e+01  1.17918253e+01
   1.23472157e+01  9.46341324e+00  1.26860819e+01  1.04336157e+01
   1.37555189e+01  1.33227749e+01  1.36271086e+01  1.00947447e+01
   1.11324196e+01  1.05241585e+01  1.24008455e+01  1.19234638e+01
   1.18386211e+01  1.36379128e+01  1.38181419e+01  1.17205496e+01
   1.18758211e+01  1.36772079e+01  1.06037264e+01  1.09604378e+01
   1.09894018e+01  1.16632900e+01  1.15430326e+01  1.15255098e+01
   1.27650356e+01  1.20024948e+01  1.32584686e+01  1.49055843e+01
   1.22079620e+01  1.15745640e+01  1.41280394e+01  1.05654325e+01
   1.16095428e+01  1.19069090e+01  1.27522078e+01  1.30736856e+01
  -1.37953854e+00  2.69015265e+00  1.89149648e-01 -1.70874023e+00
   2.92481065e+00 -2.25313354e+00  1.93817127e+00 -2.78708309e-01
  -1.54195452e+00  2.21206546e+00 -1.78481030e+00 -1.15871310e-01
  -1.27278280e+00 -1.49885511e+00 -5.16711473e-02  1.50518870e+00
  -3.25296223e-01  2.13673353e-01 -4.82891738e-01 -2.52756000e-01
  -2.78366804e-01  2.68266296e+00 -2.16513562e+00 -4.85471189e-01
  -4.93435383e-01 -7.41271526e-02  2.19444609e+00  6.83397412e-01
   2.02929282e+00 -1.50684103e-01  2.35864949e+00 -1.13786352e+00
   1.38612270e+00  2.48557734e+00  2.06136203e+00  1.17480731e+00
  -5.61801672e-01 -1.58327293e+00  1.27120590e+00 -1.21817350e+00
  -7.05911279e-01  1.43839455e+00  4.50609326e-01  1.26689708e+00
  -1.86738837e+00 -1.35119104e+00 -9.50643957e-01  2.15059137e+00
   5.42798042e-01 -2.06632543e+00  2.80117393e+00 -8.49425495e-01
   1.19261122e+00 -4.25456017e-01 -7.30377674e-01 -2.39458323e-01
   1.88459086e+00 -1.98614562e+00  9.01900679e-02  1.58485854e+00
  -2.39028549e+00  7.22256750e-02 -1.07834888e+00  1.19659543e+00
  -1.46810460e+00  1.86865878e+00  5.87870359e-01  2.27745056e+00
  -1.84443498e+00 -5.94496727e-03 -1.51228011e-01 -2.25999999e+00
   4.37323719e-01  1.37110186e+00  4.03485894e-02  3.58039021e-01
  -1.69896185e-01  2.32753062e+00  4.28585768e-01 -4.12096143e-01
  -1.41942430e+00  1.21598744e+00  1.59015000e+00 -1.40992594e+00
   8.00017834e-01  1.01934814e+00  8.44864368e-01  1.38570261e+00
  -8.77040923e-02  1.84996057e+00  1.35544384e+00 -3.44308674e-01
   1.00257349e+00  9.12033319e-02  1.71856236e+00 -2.31190801e-01
   1.63662219e+00 -1.95213699e+00 -1.58047366e+00  6.43095493e-01
  -2.32542396e-01  4.75569010e-01  2.26596689e+00  3.38542104e+00
   2.34276676e+00 -8.20497096e-01  8.69691312e-01  2.64924693e+00
   1.13398337e+00  7.75598645e-01 -2.48594332e+00 -1.71272784e-01
  -6.42227650e-01 -4.04475272e-01 -6.60711288e-01  5.08105755e-01
  -1.29360795e+00 -5.52301049e-01  1.54403162e+00 -1.64478540e+00
   9.28798854e-01  1.11119089e+01  1.12774668e+01  1.36991453e+01
   1.38419809e+01  1.13988495e+01  1.39985256e+01  1.23706551e+01
   1.22819672e+01  1.29437761e+01  1.15668020e+01  1.31271238e+01
   1.37033911e+01  1.34875383e+01  1.33186159e+01  1.43865223e+01
   1.19549179e+01  1.30282927e+01  1.32429047e+01  1.19286480e+01
   1.25468454e+01  9.20967388e+00  1.20815353e+01  9.48416328e+00
   1.21342506e+01  1.25120373e+01  1.11208868e+01  1.01836014e+01
   1.36742115e+01  1.19681234e+01  1.30260534e+01  1.00342150e+01
   1.11561785e+01  1.11395636e+01  1.30187883e+01  1.31395741e+01
   1.14113293e+01  1.15116739e+01  1.12120523e+01  1.26691732e+01
   1.03906889e+01  1.16136885e+01  1.23560677e+01  1.22242546e+01
   1.43064404e+01  1.32915010e+01  1.26904345e+01  1.11694117e+01
   1.31652079e+01  1.31365767e+01  1.16452713e+01  1.16224184e+01
   1.24145384e+01  1.04246387e+01  1.27106228e+01  1.24853916e+01
   1.23903227e+01  1.36038685e+01  1.31026449e+01  1.07307310e+01
   2.24819803e+00  2.88366377e-01  2.67617083e+00  2.89842319e+00
   8.39288473e-01  7.64852762e-02  2.60048437e+00  3.34196448e-01
   1.55699015e+00  1.08732867e+00  3.27464044e-01  1.26474428e+00
   2.41136253e-01  1.44400907e+00  2.04939365e-01  2.96015072e+00
   6.33301675e-01  5.03883481e-01  1.50555968e-01  3.13512897e+00
   2.99648285e-01  2.00138187e+00  7.62889266e-01  3.53826857e+00
   2.23980248e-01  8.31603885e-01  1.51038361e+00  1.07154560e+00
   1.33340776e+00  7.11727023e-01  1.42172933e+00  2.13233173e-01
   2.93749511e-01  2.51246500e+00  3.17757368e-01  4.52860951e-01
   9.95209455e-01  9.14214849e-01  1.43799520e+00  1.61177754e-01
   6.64845705e-02  2.79001093e+00  3.35178971e-01  2.49223757e+00
   1.09661984e+00  1.01966023e-01  3.42554808e-01  2.14911163e-01
   1.84181535e+00  4.14262414e-01  2.59382725e+00  2.01155365e-01
   4.78844523e-01  1.82740450e-01  1.41811848e-01  2.73954582e+00
   5.14581740e-01  5.97642720e-01  1.07640624e-01  3.39849710e-01
   3.31129551e-01  2.15255594e+00  3.53060293e+00  1.83844948e+00
   1.33055592e+00  2.40763605e-01  1.95572472e+00  6.99806452e-01
   1.66867495e-01  2.72529185e-01  9.77578282e-01  2.72391415e+00
   2.68022108e+00  2.07061529e-01  1.06727481e-01  2.33901739e+00
   7.34884262e-01  1.70703876e+00  2.36379504e-02  1.81081319e+00
   1.41497850e-01  2.19119167e+00  2.53374100e-01  2.07428277e-01
   1.98236656e+00  8.47744942e-01  5.98927200e-01  4.76028860e-01
   1.40984476e+00  8.84899318e-01  1.29003501e+00  5.66831946e-01
   1.12947583e-01  1.35269046e-01  4.48728800e-01  4.59227502e-01
   2.91533947e-01  2.61979985e+00  1.90878212e-01  4.42078233e-01
   2.37326765e+00  7.26699769e-01  1.79676008e+00  1.31794965e+00
   2.87806690e-01  1.34883785e+00  2.75130987e-01  2.82553911e+00
   1.53329670e-01  2.92437339e+00  1.12277246e+00  4.68158305e-01
   1.62686765e-01  1.16196823e+00  3.04879427e+00  2.74893463e-01
   2.35705256e-01  2.46256530e-01  3.34215212e+00  1.58425236e+00
   4.13840580e+00 -1.73461437e+01 -1.69585457e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 03:13:17.743471
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.9388
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 03:13:17.748219
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8115.05
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 03:13:17.752173
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.5254
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 03:13:17.756119
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -725.76
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140017782981744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140016573141176
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140016573141680
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016573142184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016573142688
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016573143192

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f586f3dcfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.470830
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.436736
grad_step = 000002, loss = 0.408501
grad_step = 000003, loss = 0.374554
grad_step = 000004, loss = 0.335910
grad_step = 000005, loss = 0.295225
grad_step = 000006, loss = 0.265203
grad_step = 000007, loss = 0.266110
grad_step = 000008, loss = 0.259212
grad_step = 000009, loss = 0.237064
grad_step = 000010, loss = 0.217130
grad_step = 000011, loss = 0.205656
grad_step = 000012, loss = 0.199435
grad_step = 000013, loss = 0.194012
grad_step = 000014, loss = 0.186899
grad_step = 000015, loss = 0.177822
grad_step = 000016, loss = 0.167739
grad_step = 000017, loss = 0.157980
grad_step = 000018, loss = 0.149827
grad_step = 000019, loss = 0.143549
grad_step = 000020, loss = 0.137482
grad_step = 000021, loss = 0.130016
grad_step = 000022, loss = 0.121703
grad_step = 000023, loss = 0.114020
grad_step = 000024, loss = 0.107597
grad_step = 000025, loss = 0.102098
grad_step = 000026, loss = 0.096876
grad_step = 000027, loss = 0.091433
grad_step = 000028, loss = 0.085600
grad_step = 000029, loss = 0.079650
grad_step = 000030, loss = 0.074267
grad_step = 000031, loss = 0.069871
grad_step = 000032, loss = 0.066007
grad_step = 000033, loss = 0.061935
grad_step = 000034, loss = 0.057563
grad_step = 000035, loss = 0.053327
grad_step = 000036, loss = 0.049561
grad_step = 000037, loss = 0.046214
grad_step = 000038, loss = 0.042977
grad_step = 000039, loss = 0.039670
grad_step = 000040, loss = 0.036430
grad_step = 000041, loss = 0.033497
grad_step = 000042, loss = 0.030940
grad_step = 000043, loss = 0.028625
grad_step = 000044, loss = 0.026341
grad_step = 000045, loss = 0.024011
grad_step = 000046, loss = 0.021832
grad_step = 000047, loss = 0.019979
grad_step = 000048, loss = 0.018344
grad_step = 000049, loss = 0.016757
grad_step = 000050, loss = 0.015194
grad_step = 000051, loss = 0.013723
grad_step = 000052, loss = 0.012437
grad_step = 000053, loss = 0.011342
grad_step = 000054, loss = 0.010331
grad_step = 000055, loss = 0.009366
grad_step = 000056, loss = 0.008482
grad_step = 000057, loss = 0.007708
grad_step = 000058, loss = 0.007058
grad_step = 000059, loss = 0.006484
grad_step = 000060, loss = 0.005950
grad_step = 000061, loss = 0.005474
grad_step = 000062, loss = 0.005063
grad_step = 000063, loss = 0.004715
grad_step = 000064, loss = 0.004415
grad_step = 000065, loss = 0.004147
grad_step = 000066, loss = 0.003917
grad_step = 000067, loss = 0.003724
grad_step = 000068, loss = 0.003560
grad_step = 000069, loss = 0.003416
grad_step = 000070, loss = 0.003287
grad_step = 000071, loss = 0.003181
grad_step = 000072, loss = 0.003092
grad_step = 000073, loss = 0.003009
grad_step = 000074, loss = 0.002933
grad_step = 000075, loss = 0.002864
grad_step = 000076, loss = 0.002809
grad_step = 000077, loss = 0.002758
grad_step = 000078, loss = 0.002705
grad_step = 000079, loss = 0.002653
grad_step = 000080, loss = 0.002609
grad_step = 000081, loss = 0.002573
grad_step = 000082, loss = 0.002535
grad_step = 000083, loss = 0.002495
grad_step = 000084, loss = 0.002460
grad_step = 000085, loss = 0.002432
grad_step = 000086, loss = 0.002406
grad_step = 000087, loss = 0.002377
grad_step = 000088, loss = 0.002350
grad_step = 000089, loss = 0.002328
grad_step = 000090, loss = 0.002310
grad_step = 000091, loss = 0.002291
grad_step = 000092, loss = 0.002273
grad_step = 000093, loss = 0.002259
grad_step = 000094, loss = 0.002247
grad_step = 000095, loss = 0.002236
grad_step = 000096, loss = 0.002225
grad_step = 000097, loss = 0.002217
grad_step = 000098, loss = 0.002210
grad_step = 000099, loss = 0.002203
grad_step = 000100, loss = 0.002197
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002192
grad_step = 000102, loss = 0.002188
grad_step = 000103, loss = 0.002184
grad_step = 000104, loss = 0.002180
grad_step = 000105, loss = 0.002176
grad_step = 000106, loss = 0.002173
grad_step = 000107, loss = 0.002169
grad_step = 000108, loss = 0.002166
grad_step = 000109, loss = 0.002163
grad_step = 000110, loss = 0.002159
grad_step = 000111, loss = 0.002156
grad_step = 000112, loss = 0.002152
grad_step = 000113, loss = 0.002149
grad_step = 000114, loss = 0.002145
grad_step = 000115, loss = 0.002141
grad_step = 000116, loss = 0.002137
grad_step = 000117, loss = 0.002133
grad_step = 000118, loss = 0.002129
grad_step = 000119, loss = 0.002125
grad_step = 000120, loss = 0.002121
grad_step = 000121, loss = 0.002117
grad_step = 000122, loss = 0.002113
grad_step = 000123, loss = 0.002109
grad_step = 000124, loss = 0.002104
grad_step = 000125, loss = 0.002100
grad_step = 000126, loss = 0.002096
grad_step = 000127, loss = 0.002092
grad_step = 000128, loss = 0.002088
grad_step = 000129, loss = 0.002084
grad_step = 000130, loss = 0.002080
grad_step = 000131, loss = 0.002076
grad_step = 000132, loss = 0.002073
grad_step = 000133, loss = 0.002069
grad_step = 000134, loss = 0.002065
grad_step = 000135, loss = 0.002061
grad_step = 000136, loss = 0.002057
grad_step = 000137, loss = 0.002054
grad_step = 000138, loss = 0.002050
grad_step = 000139, loss = 0.002046
grad_step = 000140, loss = 0.002042
grad_step = 000141, loss = 0.002039
grad_step = 000142, loss = 0.002035
grad_step = 000143, loss = 0.002031
grad_step = 000144, loss = 0.002027
grad_step = 000145, loss = 0.002024
grad_step = 000146, loss = 0.002020
grad_step = 000147, loss = 0.002016
grad_step = 000148, loss = 0.002012
grad_step = 000149, loss = 0.002008
grad_step = 000150, loss = 0.002005
grad_step = 000151, loss = 0.002001
grad_step = 000152, loss = 0.001997
grad_step = 000153, loss = 0.001993
grad_step = 000154, loss = 0.001989
grad_step = 000155, loss = 0.001985
grad_step = 000156, loss = 0.001982
grad_step = 000157, loss = 0.001978
grad_step = 000158, loss = 0.001974
grad_step = 000159, loss = 0.001970
grad_step = 000160, loss = 0.001966
grad_step = 000161, loss = 0.001962
grad_step = 000162, loss = 0.001959
grad_step = 000163, loss = 0.001955
grad_step = 000164, loss = 0.001951
grad_step = 000165, loss = 0.001947
grad_step = 000166, loss = 0.001944
grad_step = 000167, loss = 0.001940
grad_step = 000168, loss = 0.001936
grad_step = 000169, loss = 0.001932
grad_step = 000170, loss = 0.001929
grad_step = 000171, loss = 0.001925
grad_step = 000172, loss = 0.001921
grad_step = 000173, loss = 0.001918
grad_step = 000174, loss = 0.001914
grad_step = 000175, loss = 0.001910
grad_step = 000176, loss = 0.001906
grad_step = 000177, loss = 0.001903
grad_step = 000178, loss = 0.001899
grad_step = 000179, loss = 0.001895
grad_step = 000180, loss = 0.001892
grad_step = 000181, loss = 0.001889
grad_step = 000182, loss = 0.001886
grad_step = 000183, loss = 0.001887
grad_step = 000184, loss = 0.001898
grad_step = 000185, loss = 0.001926
grad_step = 000186, loss = 0.001956
grad_step = 000187, loss = 0.001926
grad_step = 000188, loss = 0.001873
grad_step = 000189, loss = 0.001866
grad_step = 000190, loss = 0.001897
grad_step = 000191, loss = 0.001897
grad_step = 000192, loss = 0.001857
grad_step = 000193, loss = 0.001848
grad_step = 000194, loss = 0.001870
grad_step = 000195, loss = 0.001865
grad_step = 000196, loss = 0.001838
grad_step = 000197, loss = 0.001830
grad_step = 000198, loss = 0.001843
grad_step = 000199, loss = 0.001842
grad_step = 000200, loss = 0.001821
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001810
grad_step = 000202, loss = 0.001816
grad_step = 000203, loss = 0.001817
grad_step = 000204, loss = 0.001805
grad_step = 000205, loss = 0.001791
grad_step = 000206, loss = 0.001788
grad_step = 000207, loss = 0.001790
grad_step = 000208, loss = 0.001785
grad_step = 000209, loss = 0.001774
grad_step = 000210, loss = 0.001763
grad_step = 000211, loss = 0.001758
grad_step = 000212, loss = 0.001756
grad_step = 000213, loss = 0.001753
grad_step = 000214, loss = 0.001748
grad_step = 000215, loss = 0.001742
grad_step = 000216, loss = 0.001733
grad_step = 000217, loss = 0.001722
grad_step = 000218, loss = 0.001714
grad_step = 000219, loss = 0.001708
grad_step = 000220, loss = 0.001705
grad_step = 000221, loss = 0.001708
grad_step = 000222, loss = 0.001732
grad_step = 000223, loss = 0.001784
grad_step = 000224, loss = 0.001771
grad_step = 000225, loss = 0.001730
grad_step = 000226, loss = 0.001708
grad_step = 000227, loss = 0.001711
grad_step = 000228, loss = 0.001695
grad_step = 000229, loss = 0.001673
grad_step = 000230, loss = 0.001697
grad_step = 000231, loss = 0.001708
grad_step = 000232, loss = 0.001668
grad_step = 000233, loss = 0.001638
grad_step = 000234, loss = 0.001654
grad_step = 000235, loss = 0.001675
grad_step = 000236, loss = 0.001655
grad_step = 000237, loss = 0.001623
grad_step = 000238, loss = 0.001619
grad_step = 000239, loss = 0.001637
grad_step = 000240, loss = 0.001647
grad_step = 000241, loss = 0.001651
grad_step = 000242, loss = 0.001682
grad_step = 000243, loss = 0.001820
grad_step = 000244, loss = 0.001842
grad_step = 000245, loss = 0.001819
grad_step = 000246, loss = 0.001644
grad_step = 000247, loss = 0.001648
grad_step = 000248, loss = 0.001734
grad_step = 000249, loss = 0.001676
grad_step = 000250, loss = 0.001674
grad_step = 000251, loss = 0.001668
grad_step = 000252, loss = 0.001613
grad_step = 000253, loss = 0.001692
grad_step = 000254, loss = 0.001696
grad_step = 000255, loss = 0.001572
grad_step = 000256, loss = 0.001702
grad_step = 000257, loss = 0.001714
grad_step = 000258, loss = 0.001566
grad_step = 000259, loss = 0.001725
grad_step = 000260, loss = 0.001657
grad_step = 000261, loss = 0.001604
grad_step = 000262, loss = 0.001695
grad_step = 000263, loss = 0.001559
grad_step = 000264, loss = 0.001631
grad_step = 000265, loss = 0.001579
grad_step = 000266, loss = 0.001572
grad_step = 000267, loss = 0.001593
grad_step = 000268, loss = 0.001545
grad_step = 000269, loss = 0.001590
grad_step = 000270, loss = 0.001543
grad_step = 000271, loss = 0.001551
grad_step = 000272, loss = 0.001557
grad_step = 000273, loss = 0.001519
grad_step = 000274, loss = 0.001549
grad_step = 000275, loss = 0.001515
grad_step = 000276, loss = 0.001524
grad_step = 000277, loss = 0.001526
grad_step = 000278, loss = 0.001503
grad_step = 000279, loss = 0.001516
grad_step = 000280, loss = 0.001504
grad_step = 000281, loss = 0.001496
grad_step = 000282, loss = 0.001502
grad_step = 000283, loss = 0.001489
grad_step = 000284, loss = 0.001482
grad_step = 000285, loss = 0.001491
grad_step = 000286, loss = 0.001474
grad_step = 000287, loss = 0.001471
grad_step = 000288, loss = 0.001476
grad_step = 000289, loss = 0.001462
grad_step = 000290, loss = 0.001457
grad_step = 000291, loss = 0.001458
grad_step = 000292, loss = 0.001451
grad_step = 000293, loss = 0.001442
grad_step = 000294, loss = 0.001441
grad_step = 000295, loss = 0.001438
grad_step = 000296, loss = 0.001428
grad_step = 000297, loss = 0.001424
grad_step = 000298, loss = 0.001422
grad_step = 000299, loss = 0.001416
grad_step = 000300, loss = 0.001408
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001401
grad_step = 000302, loss = 0.001399
grad_step = 000303, loss = 0.001394
grad_step = 000304, loss = 0.001389
grad_step = 000305, loss = 0.001384
grad_step = 000306, loss = 0.001381
grad_step = 000307, loss = 0.001381
grad_step = 000308, loss = 0.001383
grad_step = 000309, loss = 0.001394
grad_step = 000310, loss = 0.001419
grad_step = 000311, loss = 0.001480
grad_step = 000312, loss = 0.001524
grad_step = 000313, loss = 0.001595
grad_step = 000314, loss = 0.001520
grad_step = 000315, loss = 0.001431
grad_step = 000316, loss = 0.001329
grad_step = 000317, loss = 0.001321
grad_step = 000318, loss = 0.001387
grad_step = 000319, loss = 0.001393
grad_step = 000320, loss = 0.001330
grad_step = 000321, loss = 0.001276
grad_step = 000322, loss = 0.001293
grad_step = 000323, loss = 0.001315
grad_step = 000324, loss = 0.001296
grad_step = 000325, loss = 0.001303
grad_step = 000326, loss = 0.001332
grad_step = 000327, loss = 0.001351
grad_step = 000328, loss = 0.001291
grad_step = 000329, loss = 0.001276
grad_step = 000330, loss = 0.001274
grad_step = 000331, loss = 0.001243
grad_step = 000332, loss = 0.001208
grad_step = 000333, loss = 0.001216
grad_step = 000334, loss = 0.001223
grad_step = 000335, loss = 0.001206
grad_step = 000336, loss = 0.001209
grad_step = 000337, loss = 0.001249
grad_step = 000338, loss = 0.001338
grad_step = 000339, loss = 0.001375
grad_step = 000340, loss = 0.001518
grad_step = 000341, loss = 0.001436
grad_step = 000342, loss = 0.001323
grad_step = 000343, loss = 0.001185
grad_step = 000344, loss = 0.001207
grad_step = 000345, loss = 0.001312
grad_step = 000346, loss = 0.001286
grad_step = 000347, loss = 0.001204
grad_step = 000348, loss = 0.001171
grad_step = 000349, loss = 0.001220
grad_step = 000350, loss = 0.001273
grad_step = 000351, loss = 0.001221
grad_step = 000352, loss = 0.001178
grad_step = 000353, loss = 0.001162
grad_step = 000354, loss = 0.001190
grad_step = 000355, loss = 0.001234
grad_step = 000356, loss = 0.001189
grad_step = 000357, loss = 0.001150
grad_step = 000358, loss = 0.001150
grad_step = 000359, loss = 0.001174
grad_step = 000360, loss = 0.001186
grad_step = 000361, loss = 0.001154
grad_step = 000362, loss = 0.001142
grad_step = 000363, loss = 0.001150
grad_step = 000364, loss = 0.001151
grad_step = 000365, loss = 0.001152
grad_step = 000366, loss = 0.001146
grad_step = 000367, loss = 0.001134
grad_step = 000368, loss = 0.001127
grad_step = 000369, loss = 0.001132
grad_step = 000370, loss = 0.001140
grad_step = 000371, loss = 0.001139
grad_step = 000372, loss = 0.001134
grad_step = 000373, loss = 0.001129
grad_step = 000374, loss = 0.001126
grad_step = 000375, loss = 0.001118
grad_step = 000376, loss = 0.001112
grad_step = 000377, loss = 0.001111
grad_step = 000378, loss = 0.001111
grad_step = 000379, loss = 0.001110
grad_step = 000380, loss = 0.001109
grad_step = 000381, loss = 0.001111
grad_step = 000382, loss = 0.001113
grad_step = 000383, loss = 0.001116
grad_step = 000384, loss = 0.001117
grad_step = 000385, loss = 0.001126
grad_step = 000386, loss = 0.001130
grad_step = 000387, loss = 0.001141
grad_step = 000388, loss = 0.001136
grad_step = 000389, loss = 0.001136
grad_step = 000390, loss = 0.001121
grad_step = 000391, loss = 0.001107
grad_step = 000392, loss = 0.001092
grad_step = 000393, loss = 0.001085
grad_step = 000394, loss = 0.001087
grad_step = 000395, loss = 0.001092
grad_step = 000396, loss = 0.001098
grad_step = 000397, loss = 0.001099
grad_step = 000398, loss = 0.001102
grad_step = 000399, loss = 0.001098
grad_step = 000400, loss = 0.001100
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001096
grad_step = 000402, loss = 0.001098
grad_step = 000403, loss = 0.001096
grad_step = 000404, loss = 0.001100
grad_step = 000405, loss = 0.001098
grad_step = 000406, loss = 0.001101
grad_step = 000407, loss = 0.001094
grad_step = 000408, loss = 0.001092
grad_step = 000409, loss = 0.001081
grad_step = 000410, loss = 0.001071
grad_step = 000411, loss = 0.001061
grad_step = 000412, loss = 0.001055
grad_step = 000413, loss = 0.001053
grad_step = 000414, loss = 0.001053
grad_step = 000415, loss = 0.001056
grad_step = 000416, loss = 0.001060
grad_step = 000417, loss = 0.001068
grad_step = 000418, loss = 0.001075
grad_step = 000419, loss = 0.001099
grad_step = 000420, loss = 0.001119
grad_step = 000421, loss = 0.001175
grad_step = 000422, loss = 0.001183
grad_step = 000423, loss = 0.001219
grad_step = 000424, loss = 0.001156
grad_step = 000425, loss = 0.001099
grad_step = 000426, loss = 0.001051
grad_step = 000427, loss = 0.001046
grad_step = 000428, loss = 0.001075
grad_step = 000429, loss = 0.001092
grad_step = 000430, loss = 0.001091
grad_step = 000431, loss = 0.001060
grad_step = 000432, loss = 0.001031
grad_step = 000433, loss = 0.001027
grad_step = 000434, loss = 0.001046
grad_step = 000435, loss = 0.001067
grad_step = 000436, loss = 0.001067
grad_step = 000437, loss = 0.001061
grad_step = 000438, loss = 0.001055
grad_step = 000439, loss = 0.001064
grad_step = 000440, loss = 0.001055
grad_step = 000441, loss = 0.001046
grad_step = 000442, loss = 0.001025
grad_step = 000443, loss = 0.001016
grad_step = 000444, loss = 0.001014
grad_step = 000445, loss = 0.001012
grad_step = 000446, loss = 0.001007
grad_step = 000447, loss = 0.001003
grad_step = 000448, loss = 0.001005
grad_step = 000449, loss = 0.001011
grad_step = 000450, loss = 0.001016
grad_step = 000451, loss = 0.001018
grad_step = 000452, loss = 0.001025
grad_step = 000453, loss = 0.001035
grad_step = 000454, loss = 0.001064
grad_step = 000455, loss = 0.001075
grad_step = 000456, loss = 0.001105
grad_step = 000457, loss = 0.001080
grad_step = 000458, loss = 0.001068
grad_step = 000459, loss = 0.001037
grad_step = 000460, loss = 0.001017
grad_step = 000461, loss = 0.001000
grad_step = 000462, loss = 0.000988
grad_step = 000463, loss = 0.000988
grad_step = 000464, loss = 0.000998
grad_step = 000465, loss = 0.001013
grad_step = 000466, loss = 0.001010
grad_step = 000467, loss = 0.000994
grad_step = 000468, loss = 0.000974
grad_step = 000469, loss = 0.000966
grad_step = 000470, loss = 0.000970
grad_step = 000471, loss = 0.000976
grad_step = 000472, loss = 0.000976
grad_step = 000473, loss = 0.000970
grad_step = 000474, loss = 0.000966
grad_step = 000475, loss = 0.000969
grad_step = 000476, loss = 0.000981
grad_step = 000477, loss = 0.000986
grad_step = 000478, loss = 0.000997
grad_step = 000479, loss = 0.000994
grad_step = 000480, loss = 0.001006
grad_step = 000481, loss = 0.001010
grad_step = 000482, loss = 0.001024
grad_step = 000483, loss = 0.001017
grad_step = 000484, loss = 0.001006
grad_step = 000485, loss = 0.000980
grad_step = 000486, loss = 0.000959
grad_step = 000487, loss = 0.000944
grad_step = 000488, loss = 0.000941
grad_step = 000489, loss = 0.000946
grad_step = 000490, loss = 0.000953
grad_step = 000491, loss = 0.000962
grad_step = 000492, loss = 0.000968
grad_step = 000493, loss = 0.000982
grad_step = 000494, loss = 0.000987
grad_step = 000495, loss = 0.001007
grad_step = 000496, loss = 0.000996
grad_step = 000497, loss = 0.000995
grad_step = 000498, loss = 0.000966
grad_step = 000499, loss = 0.000947
grad_step = 000500, loss = 0.000936
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000934
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

  date_run                              2020-05-12 03:13:41.871789
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.25895
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 03:13:41.877388
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.169376
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 03:13:41.884641
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.146103
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 03:13:41.890588
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.57373
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
0   2020-05-12 03:13:08.437059  ...    mean_absolute_error
1   2020-05-12 03:13:08.442520  ...     mean_squared_error
2   2020-05-12 03:13:08.447530  ...  median_absolute_error
3   2020-05-12 03:13:08.451962  ...               r2_score
4   2020-05-12 03:13:17.743471  ...    mean_absolute_error
5   2020-05-12 03:13:17.748219  ...     mean_squared_error
6   2020-05-12 03:13:17.752173  ...  median_absolute_error
7   2020-05-12 03:13:17.756119  ...               r2_score
8   2020-05-12 03:13:41.871789  ...    mean_absolute_error
9   2020-05-12 03:13:41.877388  ...     mean_squared_error
10  2020-05-12 03:13:41.884641  ...  median_absolute_error
11  2020-05-12 03:13:41.890588  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 297483.05it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 386863.87it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 534740.58it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 756048.02it/s] 77%|███████▋  | 7659520/9912422 [00:00<00:02, 1068900.36it/s]9920512it [00:00, 10136378.99it/s]                            
0it [00:00, ?it/s]32768it [00:00, 355154.10it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 305755.09it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 395264.77it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 547132.79it/s]1654784it [00:00, 2760931.54it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51220.96it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbf374cecc0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbed4c1eba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbf37492eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbed46fa080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbf374cecc0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbee9e89e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbf37492eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbed4c1f0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbed4c1f080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbee9e89e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbf374da828> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4e7b061208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=1bb1c810f7f77691f56e0fac0c0a988eeac157faf2289116713239e617cafbd7
  Stored in directory: /tmp/pip-ephem-wheel-cache-ccwrplmx/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4e12c490f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
  106496/17464789 [..............................] - ETA: 32s
  245760/17464789 [..............................] - ETA: 18s
  507904/17464789 [..............................] - ETA: 11s
 1032192/17464789 [>.............................] - ETA: 6s 
 2088960/17464789 [==>...........................] - ETA: 3s
 4202496/17464789 [======>.......................] - ETA: 1s
 7315456/17464789 [===========>..................] - ETA: 0s
10428416/17464789 [================>.............] - ETA: 0s
13524992/17464789 [======================>.......] - ETA: 0s
16621568/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 03:15:13.556513: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 03:15:13.561432: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 03:15:13.561852: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5620ab268330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 03:15:13.562143: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5440 - accuracy: 0.5080
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6768 - accuracy: 0.4993 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6858 - accuracy: 0.4988
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6907 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6114 - accuracy: 0.5036
11000/25000 [============>.................] - ETA: 4s - loss: 7.5983 - accuracy: 0.5045
12000/25000 [=============>................] - ETA: 4s - loss: 7.6091 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6065 - accuracy: 0.5039
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6020 - accuracy: 0.5042
15000/25000 [=================>............] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6468 - accuracy: 0.5013
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6384 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6637 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 03:15:31.163436
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 03:15:31.163436  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 03:15:37.899302: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 03:15:37.905255: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 03:15:37.905410: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fd00697740 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 03:15:37.905430: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fdf2591cc18> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3439 - crf_viterbi_accuracy: 0.0667 - val_loss: 1.3270 - val_crf_viterbi_accuracy: 0.1067

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdf2591cd30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6871 - accuracy: 0.4987
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7214 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7471 - accuracy: 0.4947
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 4s - loss: 7.6889 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.6922 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6961 - accuracy: 0.4981
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7225 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 3s - loss: 7.7259 - accuracy: 0.4961
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7280 - accuracy: 0.4960
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7198 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7186 - accuracy: 0.4966
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7021 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7111 - accuracy: 0.4971
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7177 - accuracy: 0.4967
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7001 - accuracy: 0.4978
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fdef86c5a58> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<12:08:41, 19.7kB/s].vector_cache/glove.6B.zip:   0%|          | 369k/862M [00:00<8:31:04, 28.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 4.10M/862M [00:00<5:56:19, 40.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.8M/862M [00:00<4:07:31, 57.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:00<2:51:19, 81.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:00<1:58:40, 117kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:01<1:22:42, 167kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:01<57:27, 238kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:01<39:48, 340kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:01<29:00, 465kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<20:37, 652kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<9:51:30, 22.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:03<6:53:19, 32.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<4:53:34, 45.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<8:41:42, 25.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:04:51, 36.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<9:50:13, 22.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<6:52:39, 32.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:28:52, 21.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:19:39, 30.3kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:45:03, 20.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:30:58, 29.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:44:49, 20.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:30:47, 29.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:49:08, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:33:48, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:47:04, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:32:26, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:27:10, 21.0kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:18:27, 30.0kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:36:59, 20.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:25:18, 29.4kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:42:38, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:29:21, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:12:47, 21.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:08:23, 30.4kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:32:35, 20.6kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:22:16, 29.4kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:21:52, 20.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:14:42, 29.8kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:34:31, 20.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:23:33, 29.1kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:34:17, 20.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:23:22, 29.1kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:36:37, 20.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:25:00, 28.9kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:33:25, 20.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:22:44, 28.9kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:38:26, 20.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:26:15, 28.6kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:35:02, 20.1kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:23:55, 28.7kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:26:16, 20.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:17:45, 29.0kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:29:44, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:20:16, 28.8kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:06:45, 20.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:04:12, 29.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<9:58:12, 21.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<6:58:07, 30.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:17:00, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:11:16, 29.2kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:19:57, 20.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:13:21, 28.9kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:11:42, 20.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<7:07:32, 29.2kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:19:29, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<7:12:59, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:17:16, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:11:25, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:18:52, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:12:37, 28.7kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<9:56:48, 20.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<6:57:08, 29.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:06:43, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<7:04:03, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:09:05, 20.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<7:05:44, 28.9kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:02:01, 20.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<7:00:45, 29.1kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<10:09:31, 20.1kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<7:06:01, 28.7kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:02:22, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<7:00:59, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:08:13, 20.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<7:05:09, 28.6kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<9:42:48, 20.9kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:47:19, 29.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:54:44, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:55:39, 29.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:53:42, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:54:55, 29.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:59:16, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:58:55, 28.7kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:27:07, 21.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:36:27, 30.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:20:16, 21.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:31:34, 30.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:36:52, 20.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:43:09, 29.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:44:10, 20.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:48:18, 29.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:34:36, 20.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:41:34, 29.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:41:26, 20.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:46:22, 29.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:33:25, 20.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:40:43, 29.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:43:21, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:47:44, 28.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<4:46:39, 40.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<7:56:19, 24.6kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<5:32:57, 35.1kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<8:51:18, 22.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:11:20, 31.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:07:45, 21.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:22:47, 30.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:25:41, 20.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:35:18, 29.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:28:22, 20.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:37:10, 29.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:34:31, 20.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:41:33, 28.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:09:35, 21.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:24:03, 29.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:21:37, 20.5kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:32:28, 29.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:18:32, 20.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:30:21, 29.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:06:36, 20.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:21:57, 29.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:18:30, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:30:16, 29.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:20:26, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:31:37, 28.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:15:54, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:28:25, 29.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:21:07, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:32:04, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<9:19:20, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:30:51, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:09:19, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:23:48, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<9:15:26, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:28:05, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:12:00, 20.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:25:42, 28.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:04:10, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:20:17, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<8:49:36, 20.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:10:01, 29.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:01:27, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:18:19, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<9:05:58, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:21:29, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<8:58:44, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:16:23, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<9:02:49, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:19:15, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:59:08, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:16:40, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:59:36, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:17:01, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:50:39, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:10:49, 29.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:32:31, 21.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<5:58:04, 30.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:46:42, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:07:58, 29.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:47:56, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<6:08:51, 29.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:42:30, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:05:01, 29.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:48:49, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:09:30, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:28:46, 20.9kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<5:55:26, 29.8kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:39:03, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<6:02:36, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:39:42, 20.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<6:03:09, 28.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:16:01, 21.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<5:46:31, 30.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:30:00, 20.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<5:56:18, 29.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:27:16, 20.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<5:54:21, 29.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:34:15, 20.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:59:14, 28.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:31:00, 20.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:56:57, 29.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:32:24, 20.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:57:56, 28.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:31:00, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:56:58, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:22:51, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:51:15, 29.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:26:13, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:53:37, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:21:55, 20.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:50:35, 29.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:25:16, 20.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:52:55, 28.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:23:50, 20.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:51:56, 28.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:16:09, 20.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:46:33, 29.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:18:09, 20.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:47:56, 28.8kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:18:31, 20.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:48:17, 28.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<7:52:54, 21.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:30:18, 30.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:05:33, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:39:09, 29.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<8:04:01, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:38:03, 29.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<8:09:52, 20.2kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:42:08, 28.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<8:05:53, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:39:20, 28.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<8:09:19, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:41:44, 28.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<8:05:53, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:39:25, 28.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<7:42:10, 21.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:22:47, 30.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:53:47, 20.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:30:57, 29.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:38:59, 21.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:20:37, 30.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:33:51, 21.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:17:02, 30.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:31:43, 21.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:15:32, 30.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:29:24, 21.3kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:13:50, 30.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:43:41, 20.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:23:49, 29.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:43:49, 20.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:23:54, 29.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:46:28, 20.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:25:46, 28.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:38:10, 20.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:19:56, 29.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:44:06, 20.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:24:11, 28.9kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:18:32, 21.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:06:15, 30.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:26:42, 20.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:12:01, 29.8kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:12:56, 21.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:02:19, 30.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:28:02, 20.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:12:52, 29.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:27:07, 20.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:12:12, 29.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:31:41, 20.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<5:15:25, 29.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:26:57, 20.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<5:12:05, 29.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:27:47, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<5:12:38, 29.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:31:29, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<5:15:18, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:07:28, 21.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<4:58:28, 30.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:19:43, 20.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:07:02, 29.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:16:30, 20.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<5:04:45, 29.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:19:34, 20.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<5:06:55, 29.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<7:12:31, 20.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<5:01:57, 29.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<7:17:51, 20.3kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<5:05:45, 28.9kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<6:59:44, 21.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<4:53:02, 30.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<7:10:52, 20.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<5:00:52, 29.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<6:55:53, 21.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<4:50:20, 30.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<7:07:26, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<4:58:24, 29.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<7:06:44, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<4:57:54, 29.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<7:09:20, 20.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<4:59:43, 28.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<7:05:38, 20.3kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<4:57:07, 28.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<7:06:58, 20.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:58:03, 28.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<7:03:58, 20.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<4:55:56, 28.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<7:03:56, 20.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<4:55:58, 28.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<6:46:03, 20.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<4:43:26, 29.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:54:44, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<4:49:21, 29.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<3:26:18, 40.8kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<5:40:38, 24.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<3:57:51, 35.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:10:47, 22.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<4:18:51, 32.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<6:25:01, 21.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<4:28:49, 30.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<6:23:11, 21.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<4:27:28, 30.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<6:36:06, 20.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<4:36:28, 29.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<6:42:17, 20.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<4:40:51, 29.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<6:23:19, 21.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<4:27:35, 30.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<6:21:46, 21.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<4:26:27, 30.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<6:34:17, 20.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<4:35:14, 29.4kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:23:02, 21.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<4:27:19, 30.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<6:33:09, 20.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<4:34:26, 29.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<6:20:39, 21.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<4:25:42, 30.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<6:16:27, 21.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<4:22:42, 30.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<6:28:52, 20.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<4:31:23, 29.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<6:27:59, 20.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<4:30:47, 29.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<6:17:28, 20.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<4:23:24, 29.8kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<6:25:14, 20.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<4:28:49, 29.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<6:26:00, 20.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<4:29:28, 28.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<3:09:35, 41.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<5:09:09, 25.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<3:35:47, 35.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<5:47:16, 22.3kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<4:02:08, 31.9kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<3:07:40, 41.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<5:12:37, 24.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<3:38:11, 35.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<5:47:23, 22.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<4:02:26, 31.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<5:55:51, 21.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<4:08:17, 30.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<6:08:24, 20.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<4:17:07, 29.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<5:52:22, 21.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<4:05:51, 30.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<6:06:12, 20.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<4:15:31, 29.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<6:01:42, 20.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<4:12:20, 29.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<6:08:35, 20.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<4:17:08, 28.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<6:04:36, 20.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<4:14:20, 29.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<6:07:58, 20.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<4:16:42, 28.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<6:17:40, 19.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<4:23:28, 27.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<6:05:10, 20.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<4:14:43, 28.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<6:04:52, 20.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<4:14:30, 28.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<6:01:48, 20.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<4:12:23, 28.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:55:22, 20.3kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<4:07:52, 29.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<5:57:45, 20.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<4:09:32, 28.7kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<5:53:53, 20.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<4:06:49, 28.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<5:55:27, 20.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<4:07:55, 28.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<5:52:23, 20.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<4:05:50, 28.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:33:46, 21.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<3:52:47, 30.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<5:41:18, 20.6kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<3:58:02, 29.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:43:14, 20.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<3:59:23, 29.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:39:21, 20.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<3:56:39, 29.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:42:45, 20.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<3:59:01, 28.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:40:24, 20.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:57:22, 28.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<5:41:26, 20.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<3:58:06, 28.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<5:38:00, 20.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<3:55:41, 28.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<5:38:37, 20.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<3:56:07, 28.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<5:32:27, 20.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<3:51:51, 28.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:22:00, 20.8kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<3:44:31, 29.7kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<5:27:36, 20.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<3:48:38, 29.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<2:40:58, 41.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<4:06:32, 26.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<2:51:57, 38.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<4:46:41, 23.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<3:19:55, 32.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<5:05:49, 21.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<3:33:16, 30.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<5:03:00, 21.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<3:31:18, 30.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<5:03:01, 21.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<3:31:15, 30.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<5:13:05, 20.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<3:38:18, 29.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<5:04:26, 21.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<3:32:16, 30.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<5:00:09, 21.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<3:29:14, 30.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<5:09:23, 20.5kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<3:35:42, 29.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<4:59:07, 21.1kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<3:28:33, 30.1kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<4:54:34, 21.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<3:25:20, 30.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<5:03:29, 20.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<3:31:32, 29.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<5:04:31, 20.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<3:32:15, 29.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<5:00:51, 20.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<3:29:41, 29.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<5:03:40, 20.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<3:31:38, 28.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<5:02:44, 20.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<3:30:59, 28.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<4:57:22, 20.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<3:27:14, 29.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<4:56:59, 20.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<3:26:57, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<4:57:33, 20.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<3:27:20, 28.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<4:54:32, 20.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<3:25:15, 28.9kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<4:48:19, 20.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<3:20:54, 29.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<4:49:56, 20.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<3:22:01, 29.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<4:50:01, 20.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<3:22:04, 28.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<4:48:29, 20.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<3:21:00, 28.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<4:44:33, 20.4kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<3:18:15, 29.0kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<4:44:44, 20.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<3:18:24, 28.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<4:34:13, 20.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<3:11:02, 29.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<4:38:09, 20.4kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<3:13:48, 29.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<4:28:14, 21.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<3:06:54, 30.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:24:30, 21.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<3:04:14, 30.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<4:32:26, 20.5kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<3:09:47, 29.2kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<4:27:03, 20.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<3:06:00, 29.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<4:30:38, 20.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<3:08:33, 29.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<4:15:00, 21.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<2:57:38, 30.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<4:13:18, 21.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<2:56:25, 30.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<4:18:30, 20.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<3:00:01, 29.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<4:23:35, 20.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<3:03:34, 29.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<4:19:20, 20.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<3:00:38, 29.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<4:11:07, 21.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<2:54:54, 30.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<4:06:20, 21.4kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<2:51:31, 30.5kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<4:12:54, 20.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<2:56:06, 29.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<4:12:35, 20.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<2:55:51, 29.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<4:15:24, 20.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<2:57:50, 28.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<4:04:25, 21.0kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<2:50:09, 29.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<4:08:52, 20.5kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<2:53:17, 29.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<3:59:48, 21.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<2:46:56, 30.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<4:00:41, 20.9kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<2:47:32, 29.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<4:04:15, 20.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<2:50:01, 29.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<3:59:57, 20.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<2:47:00, 29.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<4:02:16, 20.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<2:48:37, 29.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<4:01:50, 20.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:08<2:48:20, 28.8kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<3:50:07, 21.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:09<2:40:08, 30.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<3:50:59, 20.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<2:40:43, 29.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<3:55:04, 20.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<2:43:34, 29.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<3:50:37, 20.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<2:40:27, 29.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<3:52:08, 20.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<2:41:30, 28.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<3:49:52, 20.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<2:39:57, 29.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<3:41:05, 21.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<2:33:48, 29.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<3:45:00, 20.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<2:36:31, 29.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<3:43:00, 20.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<2:35:06, 29.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<3:44:43, 20.2kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<2:36:20, 28.8kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<3:34:35, 21.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<2:29:14, 29.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<3:38:01, 20.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<2:31:39, 29.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<3:30:24, 21.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<2:26:18, 30.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<3:34:15, 20.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<2:28:58, 29.3kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<3:34:20, 20.3kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<2:29:03, 29.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<3:28:41, 20.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<2:25:05, 29.6kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<3:30:53, 20.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<2:26:37, 29.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<3:27:25, 20.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<2:24:11, 29.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<3:28:31, 20.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<2:24:57, 28.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<3:24:28, 20.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<2:22:07, 29.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<3:26:00, 20.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<2:22:24, 28.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<1:40:10, 40.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<1:09:46, 58.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<2:16:34, 29.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<1:34:57, 42.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<2:49:04, 23.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<1:57:30, 33.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<3:02:22, 21.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<2:06:43, 31.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<3:06:35, 21.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<2:09:37, 30.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<3:09:44, 20.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<2:11:48, 29.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<3:10:26, 20.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<2:12:17, 29.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<3:07:34, 20.4kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<2:10:16, 29.2kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<3:08:39, 20.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<2:11:01, 28.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<3:05:55, 20.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<2:09:06, 28.9kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<3:05:19, 20.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<2:08:39, 28.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<1:32:04, 40.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<2:24:37, 25.5kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<1:40:26, 36.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<2:43:13, 22.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<1:53:18, 32.0kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<1:19:27, 45.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<55:45, 64.3kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<38:54, 91.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<1:48:45, 32.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<1:15:09, 46.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<52:47, 65.9kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<36:51, 93.6kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:37:33, 35.4kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:49<1:07:13, 50.5kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<49:13, 68.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<34:58, 96.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<24:26, 137kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<1:24:02, 39.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<58:05, 56.4kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<40:41, 80.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<28:29, 114kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:28:49, 36.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<1:01:19, 51.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<43:19, 73.1kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<30:11, 104kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<1:25:11, 36.8kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<58:46, 52.2kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<41:17, 74.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<28:47, 105kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<1:31:13, 33.2kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:03:15, 47.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:57:37, 25.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<1:21:30, 36.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<2:13:08, 22.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:03<1:31:17, 31.8kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:07:19, 43.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<47:06, 61.2kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<32:51, 86.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<1:27:55, 32.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:00:28, 46.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<42:32, 65.4kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<29:36, 92.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<1:16:58, 35.7kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<52:55, 50.7kB/s]  .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<37:10, 72.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<25:52, 102kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<1:17:39, 34.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<53:19, 48.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<37:27, 68.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<26:02, 97.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<1:15:21, 33.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<51:41, 47.8kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<36:17, 68.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<25:13, 96.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<1:11:59, 33.8kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:18<49:00, 48.3kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<2:58:32, 13.3kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<2:04:40, 18.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<1:25:23, 26.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<59:39, 38.4kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<41:25, 54.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:16:26, 29.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<52:46, 42.2kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<1:33:42, 23.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:24<1:04:40, 33.9kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<1:40:45, 21.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<1:09:30, 31.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<1:41:57, 21.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<1:10:18, 30.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<1:43:02, 20.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<1:11:02, 29.4kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<1:50:57, 18.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:28<1:15:26, 26.9kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<54:51, 36.8kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<38:26, 52.3kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:30<26:17, 74.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<19:24, 100kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<13:36, 142kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<09:32, 197kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<06:44, 278kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<04:51, 373kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<03:28, 519kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<02:37, 664kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:54, 908kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<01:24, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<44:36, 38.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<30:11, 54.1kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<21:13, 76.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:41<14:21, 110kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<12:04, 130kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<08:29, 183kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<05:56, 252kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<04:10, 355kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<03:03, 467kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<02:12, 644kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:41, 804kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:14, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:54, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<36:37, 36.0kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<25:02, 51.4kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:50<17:03, 73.1kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<1:23:25, 15.0kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<57:55, 21.3kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<38:50, 30.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<27:06, 43.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<18:08, 61.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<12:42, 86.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<08:32, 122kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<06:02, 171kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:58<03:59, 244kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<04:38, 209kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<03:19, 291kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:00<02:11, 414kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<03:46, 239kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<02:42, 331kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:02<01:46, 471kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<02:55, 285kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<02:06, 393kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<01:27, 545kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<21:51, 36.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<14:05, 51.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<09:47, 73.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<06:22, 103kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<04:26, 146kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<02:54, 202kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<02:03, 282kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<01:21, 380kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:57, 528kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:40, 674kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:29, 899kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:20, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<11:21, 36.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:16<06:42, 51.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<05:21, 63.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:45, 89.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:18<02:16, 128kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:37, 168kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:08, 234kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:20<00:36, 334kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<02:04, 98.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<01:24, 139kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:41, 192kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:28, 268kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:10, 362kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:07, 498kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:02, 686kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:38, 37.9kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 691/400000 [00:00<00:57, 6898.90it/s]  0%|          | 1403/400000 [00:00<00:57, 6962.53it/s]  1%|          | 2112/400000 [00:00<00:56, 6999.20it/s]  1%|          | 2782/400000 [00:00<00:57, 6899.54it/s]  1%|          | 3523/400000 [00:00<00:56, 7042.28it/s]  1%|          | 4268/400000 [00:00<00:55, 7159.11it/s]  1%|▏         | 5000/400000 [00:00<00:54, 7206.53it/s]  1%|▏         | 5711/400000 [00:00<00:54, 7176.70it/s]  2%|▏         | 6434/400000 [00:00<00:54, 7186.75it/s]  2%|▏         | 7147/400000 [00:01<00:54, 7168.00it/s]  2%|▏         | 7845/400000 [00:01<00:55, 7067.55it/s]  2%|▏         | 8540/400000 [00:01<00:55, 7029.62it/s]  2%|▏         | 9234/400000 [00:01<00:56, 6961.33it/s]  2%|▏         | 9930/400000 [00:01<00:56, 6959.60it/s]  3%|▎         | 10634/400000 [00:01<00:55, 6982.34it/s]  3%|▎         | 11347/400000 [00:01<00:55, 7009.40it/s]  3%|▎         | 12055/400000 [00:01<00:55, 7028.50it/s]  3%|▎         | 12778/400000 [00:01<00:54, 7085.06it/s]  3%|▎         | 13486/400000 [00:01<00:55, 6941.81it/s]  4%|▎         | 14217/400000 [00:02<00:54, 7047.75it/s]  4%|▎         | 14931/400000 [00:02<00:54, 7073.01it/s]  4%|▍         | 15670/400000 [00:02<00:53, 7162.79it/s]  4%|▍         | 16387/400000 [00:02<00:53, 7158.43it/s]  4%|▍         | 17115/400000 [00:02<00:53, 7192.76it/s]  4%|▍         | 17861/400000 [00:02<00:52, 7267.86it/s]  5%|▍         | 18589/400000 [00:02<00:52, 7237.68it/s]  5%|▍         | 19314/400000 [00:02<00:53, 7141.32it/s]  5%|▌         | 20029/400000 [00:02<00:53, 7060.08it/s]  5%|▌         | 20772/400000 [00:02<00:52, 7165.99it/s]  5%|▌         | 21503/400000 [00:03<00:52, 7206.29it/s]  6%|▌         | 22237/400000 [00:03<00:52, 7244.64it/s]  6%|▌         | 22980/400000 [00:03<00:51, 7299.27it/s]  6%|▌         | 23711/400000 [00:03<00:52, 7182.99it/s]  6%|▌         | 24431/400000 [00:03<00:52, 7177.34it/s]  6%|▋         | 25150/400000 [00:03<00:52, 7163.84it/s]  6%|▋         | 25867/400000 [00:03<00:53, 6989.25it/s]  7%|▋         | 26583/400000 [00:03<00:53, 7039.13it/s]  7%|▋         | 27288/400000 [00:03<00:53, 6935.41it/s]  7%|▋         | 27983/400000 [00:03<00:54, 6851.16it/s]  7%|▋         | 28670/400000 [00:04<00:54, 6851.57it/s]  7%|▋         | 29377/400000 [00:04<00:53, 6912.62it/s]  8%|▊         | 30069/400000 [00:04<00:54, 6830.27it/s]  8%|▊         | 30763/400000 [00:04<00:53, 6859.59it/s]  8%|▊         | 31470/400000 [00:04<00:53, 6919.95it/s]  8%|▊         | 32166/400000 [00:04<00:53, 6929.46it/s]  8%|▊         | 32872/400000 [00:04<00:52, 6967.13it/s]  8%|▊         | 33586/400000 [00:04<00:52, 7015.99it/s]  9%|▊         | 34303/400000 [00:04<00:51, 7060.08it/s]  9%|▉         | 35010/400000 [00:04<00:52, 6999.67it/s]  9%|▉         | 35728/400000 [00:05<00:51, 7049.48it/s]  9%|▉         | 36440/400000 [00:05<00:51, 7068.91it/s]  9%|▉         | 37155/400000 [00:05<00:51, 7092.95it/s]  9%|▉         | 37865/400000 [00:05<00:51, 7020.64it/s] 10%|▉         | 38568/400000 [00:05<00:51, 6972.04it/s] 10%|▉         | 39266/400000 [00:05<00:52, 6902.44it/s] 10%|▉         | 39973/400000 [00:05<00:51, 6949.34it/s] 10%|█         | 40679/400000 [00:05<00:51, 6979.95it/s] 10%|█         | 41389/400000 [00:05<00:51, 7014.84it/s] 11%|█         | 42091/400000 [00:05<00:51, 6971.01it/s] 11%|█         | 42789/400000 [00:06<00:52, 6854.59it/s] 11%|█         | 43501/400000 [00:06<00:51, 6930.46it/s] 11%|█         | 44195/400000 [00:06<00:51, 6907.59it/s] 11%|█         | 44901/400000 [00:06<00:51, 6950.68it/s] 11%|█▏        | 45618/400000 [00:06<00:50, 7013.22it/s] 12%|█▏        | 46320/400000 [00:06<00:50, 6971.42it/s] 12%|█▏        | 47039/400000 [00:06<00:50, 7034.64it/s] 12%|█▏        | 47752/400000 [00:06<00:49, 7062.62it/s] 12%|█▏        | 48466/400000 [00:06<00:49, 7083.74it/s] 12%|█▏        | 49175/400000 [00:06<00:49, 7040.49it/s] 12%|█▏        | 49880/400000 [00:07<00:50, 6984.57it/s] 13%|█▎        | 50579/400000 [00:07<00:50, 6950.15it/s] 13%|█▎        | 51314/400000 [00:07<00:49, 7062.87it/s] 13%|█▎        | 52024/400000 [00:07<00:49, 7070.58it/s] 13%|█▎        | 52741/400000 [00:07<00:48, 7098.68it/s] 13%|█▎        | 53452/400000 [00:07<00:49, 7045.60it/s] 14%|█▎        | 54200/400000 [00:07<00:48, 7169.79it/s] 14%|█▎        | 54918/400000 [00:07<00:49, 7041.75it/s] 14%|█▍        | 55665/400000 [00:07<00:48, 7162.28it/s] 14%|█▍        | 56383/400000 [00:07<00:48, 7114.48it/s] 14%|█▍        | 57110/400000 [00:08<00:47, 7158.70it/s] 14%|█▍        | 57864/400000 [00:08<00:47, 7267.32it/s] 15%|█▍        | 58602/400000 [00:08<00:46, 7298.39it/s] 15%|█▍        | 59337/400000 [00:08<00:46, 7313.06it/s] 15%|█▌        | 60072/400000 [00:08<00:46, 7323.70it/s] 15%|█▌        | 60805/400000 [00:08<00:47, 7174.42it/s] 15%|█▌        | 61546/400000 [00:08<00:46, 7243.38it/s] 16%|█▌        | 62302/400000 [00:08<00:46, 7332.16it/s] 16%|█▌        | 63037/400000 [00:08<00:46, 7316.30it/s] 16%|█▌        | 63770/400000 [00:09<00:46, 7288.75it/s] 16%|█▌        | 64500/400000 [00:09<00:46, 7252.16it/s] 16%|█▋        | 65233/400000 [00:09<00:46, 7275.05it/s] 17%|█▋        | 66002/400000 [00:09<00:45, 7392.88it/s] 17%|█▋        | 66742/400000 [00:09<00:45, 7377.98it/s] 17%|█▋        | 67481/400000 [00:09<00:45, 7381.44it/s] 17%|█▋        | 68220/400000 [00:09<00:45, 7322.23it/s] 17%|█▋        | 68981/400000 [00:09<00:44, 7404.30it/s] 17%|█▋        | 69725/400000 [00:09<00:44, 7414.30it/s] 18%|█▊        | 70467/400000 [00:09<00:44, 7345.41it/s] 18%|█▊        | 71202/400000 [00:10<00:45, 7243.90it/s] 18%|█▊        | 71927/400000 [00:10<00:45, 7202.98it/s] 18%|█▊        | 72648/400000 [00:10<00:45, 7184.87it/s] 18%|█▊        | 73367/400000 [00:10<00:45, 7147.75it/s] 19%|█▊        | 74083/400000 [00:10<00:45, 7124.27it/s] 19%|█▊        | 74796/400000 [00:10<00:45, 7089.94it/s] 19%|█▉        | 75506/400000 [00:10<00:46, 7051.50it/s] 19%|█▉        | 76212/400000 [00:10<00:46, 6976.29it/s] 19%|█▉        | 76917/400000 [00:10<00:46, 6997.95it/s] 19%|█▉        | 77620/400000 [00:10<00:46, 7006.89it/s] 20%|█▉        | 78334/400000 [00:11<00:45, 7045.88it/s] 20%|█▉        | 79049/400000 [00:11<00:45, 7074.23it/s] 20%|█▉        | 79757/400000 [00:11<00:47, 6775.37it/s] 20%|██        | 80451/400000 [00:11<00:46, 6821.90it/s] 20%|██        | 81156/400000 [00:11<00:46, 6887.40it/s] 20%|██        | 81853/400000 [00:11<00:46, 6910.54it/s] 21%|██        | 82546/400000 [00:11<00:45, 6913.88it/s] 21%|██        | 83239/400000 [00:11<00:46, 6820.35it/s] 21%|██        | 83922/400000 [00:11<00:46, 6786.47it/s] 21%|██        | 84602/400000 [00:11<00:46, 6764.98it/s] 21%|██▏       | 85279/400000 [00:12<00:46, 6739.25it/s] 21%|██▏       | 85977/400000 [00:12<00:46, 6807.21it/s] 22%|██▏       | 86704/400000 [00:12<00:45, 6939.44it/s] 22%|██▏       | 87399/400000 [00:12<00:45, 6895.78it/s] 22%|██▏       | 88091/400000 [00:12<00:45, 6902.07it/s] 22%|██▏       | 88782/400000 [00:12<00:46, 6712.51it/s] 22%|██▏       | 89499/400000 [00:12<00:45, 6841.68it/s] 23%|██▎       | 90185/400000 [00:12<00:45, 6838.30it/s] 23%|██▎       | 90876/400000 [00:12<00:45, 6857.54it/s] 23%|██▎       | 91570/400000 [00:12<00:44, 6881.03it/s] 23%|██▎       | 92259/400000 [00:13<00:45, 6824.39it/s] 23%|██▎       | 92945/400000 [00:13<00:44, 6834.57it/s] 23%|██▎       | 93637/400000 [00:13<00:44, 6859.25it/s] 24%|██▎       | 94326/400000 [00:13<00:44, 6866.05it/s] 24%|██▍       | 95013/400000 [00:13<00:46, 6586.62it/s] 24%|██▍       | 95701/400000 [00:13<00:45, 6671.69it/s] 24%|██▍       | 96371/400000 [00:13<00:45, 6625.73it/s] 24%|██▍       | 97072/400000 [00:13<00:44, 6734.17it/s] 24%|██▍       | 97793/400000 [00:13<00:43, 6869.69it/s] 25%|██▍       | 98517/400000 [00:13<00:43, 6975.63it/s] 25%|██▍       | 99217/400000 [00:14<00:43, 6942.65it/s] 25%|██▍       | 99913/400000 [00:14<00:43, 6905.90it/s] 25%|██▌       | 100615/400000 [00:14<00:43, 6938.46it/s] 25%|██▌       | 101315/400000 [00:14<00:42, 6954.41it/s] 26%|██▌       | 102011/400000 [00:14<00:42, 6948.69it/s] 26%|██▌       | 102707/400000 [00:14<00:42, 6931.19it/s] 26%|██▌       | 103401/400000 [00:14<00:43, 6888.40it/s] 26%|██▌       | 104101/400000 [00:14<00:42, 6919.56it/s] 26%|██▌       | 104794/400000 [00:14<00:43, 6854.67it/s] 26%|██▋       | 105506/400000 [00:15<00:42, 6931.77it/s] 27%|██▋       | 106200/400000 [00:15<00:42, 6891.19it/s] 27%|██▋       | 106890/400000 [00:15<00:42, 6879.30it/s] 27%|██▋       | 107579/400000 [00:15<00:42, 6866.71it/s] 27%|██▋       | 108266/400000 [00:15<00:42, 6814.34it/s] 27%|██▋       | 108961/400000 [00:15<00:42, 6852.42it/s] 27%|██▋       | 109647/400000 [00:15<00:42, 6847.39it/s] 28%|██▊       | 110348/400000 [00:15<00:42, 6894.14it/s] 28%|██▊       | 111038/400000 [00:15<00:42, 6850.45it/s] 28%|██▊       | 111729/400000 [00:15<00:41, 6866.96it/s] 28%|██▊       | 112416/400000 [00:16<00:42, 6788.63it/s] 28%|██▊       | 113103/400000 [00:16<00:42, 6812.15it/s] 28%|██▊       | 113805/400000 [00:16<00:41, 6871.83it/s] 29%|██▊       | 114522/400000 [00:16<00:41, 6956.31it/s] 29%|██▉       | 115219/400000 [00:16<00:41, 6797.81it/s] 29%|██▉       | 115900/400000 [00:16<00:42, 6624.67it/s] 29%|██▉       | 116625/400000 [00:16<00:41, 6800.18it/s] 29%|██▉       | 117308/400000 [00:16<00:41, 6797.05it/s] 30%|██▉       | 118048/400000 [00:16<00:40, 6966.58it/s] 30%|██▉       | 118750/400000 [00:16<00:40, 6982.14it/s] 30%|██▉       | 119450/400000 [00:17<00:40, 6847.97it/s] 30%|███       | 120137/400000 [00:17<00:41, 6709.76it/s] 30%|███       | 120839/400000 [00:17<00:41, 6798.82it/s] 30%|███       | 121525/400000 [00:17<00:40, 6816.89it/s] 31%|███       | 122208/400000 [00:17<00:41, 6766.89it/s] 31%|███       | 122930/400000 [00:17<00:40, 6895.23it/s] 31%|███       | 123621/400000 [00:17<00:40, 6864.33it/s] 31%|███       | 124335/400000 [00:17<00:39, 6942.48it/s] 31%|███▏      | 125077/400000 [00:17<00:38, 7077.11it/s] 31%|███▏      | 125786/400000 [00:17<00:38, 7071.67it/s] 32%|███▏      | 126507/400000 [00:18<00:38, 7111.66it/s] 32%|███▏      | 127219/400000 [00:18<00:38, 7056.13it/s] 32%|███▏      | 127928/400000 [00:18<00:38, 7065.05it/s] 32%|███▏      | 128635/400000 [00:18<00:38, 6967.36it/s] 32%|███▏      | 129355/400000 [00:18<00:38, 7033.77it/s] 33%|███▎      | 130059/400000 [00:18<00:38, 6990.84it/s] 33%|███▎      | 130759/400000 [00:18<00:38, 6945.29it/s] 33%|███▎      | 131457/400000 [00:18<00:38, 6953.72it/s] 33%|███▎      | 132182/400000 [00:18<00:38, 7039.66it/s] 33%|███▎      | 132891/400000 [00:18<00:37, 7051.94it/s] 33%|███▎      | 133597/400000 [00:19<00:38, 6961.21it/s] 34%|███▎      | 134294/400000 [00:19<00:38, 6917.87it/s] 34%|███▎      | 134991/400000 [00:19<00:38, 6933.28it/s] 34%|███▍      | 135704/400000 [00:19<00:37, 6990.91it/s] 34%|███▍      | 136432/400000 [00:19<00:37, 7072.52it/s] 34%|███▍      | 137177/400000 [00:19<00:36, 7179.08it/s] 34%|███▍      | 137896/400000 [00:19<00:36, 7160.71it/s] 35%|███▍      | 138613/400000 [00:19<00:36, 7153.38it/s] 35%|███▍      | 139329/400000 [00:19<00:36, 7088.32it/s] 35%|███▌      | 140144/400000 [00:19<00:35, 7373.86it/s] 35%|███▌      | 140896/400000 [00:20<00:34, 7416.09it/s] 35%|███▌      | 141640/400000 [00:20<00:35, 7308.87it/s] 36%|███▌      | 142373/400000 [00:20<00:35, 7306.57it/s] 36%|███▌      | 143106/400000 [00:20<00:35, 7227.15it/s] 36%|███▌      | 143830/400000 [00:20<00:35, 7161.73it/s] 36%|███▌      | 144548/400000 [00:20<00:35, 7102.63it/s] 36%|███▋      | 145260/400000 [00:20<00:36, 7053.14it/s] 36%|███▋      | 145966/400000 [00:20<00:36, 7055.15it/s] 37%|███▋      | 146685/400000 [00:20<00:35, 7094.71it/s] 37%|███▋      | 147395/400000 [00:20<00:35, 7052.96it/s] 37%|███▋      | 148112/400000 [00:21<00:35, 7084.71it/s] 37%|███▋      | 148821/400000 [00:21<00:35, 7012.16it/s] 37%|███▋      | 149541/400000 [00:21<00:35, 7064.08it/s] 38%|███▊      | 150270/400000 [00:21<00:35, 7129.08it/s] 38%|███▊      | 150988/400000 [00:21<00:34, 7144.09it/s] 38%|███▊      | 151723/400000 [00:21<00:34, 7204.37it/s] 38%|███▊      | 152455/400000 [00:21<00:34, 7238.54it/s] 38%|███▊      | 153181/400000 [00:21<00:34, 7242.78it/s] 38%|███▊      | 153906/400000 [00:21<00:34, 7120.41it/s] 39%|███▊      | 154619/400000 [00:22<00:35, 6966.58it/s] 39%|███▉      | 155317/400000 [00:22<00:35, 6924.85it/s] 39%|███▉      | 156011/400000 [00:22<00:35, 6872.02it/s] 39%|███▉      | 156750/400000 [00:22<00:34, 7017.70it/s] 39%|███▉      | 157454/400000 [00:22<00:34, 6970.42it/s] 40%|███▉      | 158156/400000 [00:22<00:34, 6982.57it/s] 40%|███▉      | 158855/400000 [00:22<00:34, 6956.65it/s] 40%|███▉      | 159556/400000 [00:22<00:34, 6972.13it/s] 40%|████      | 160278/400000 [00:22<00:34, 7042.85it/s] 40%|████      | 160983/400000 [00:22<00:34, 7015.22it/s] 40%|████      | 161708/400000 [00:23<00:33, 7083.87it/s] 41%|████      | 162417/400000 [00:23<00:33, 7043.02it/s] 41%|████      | 163125/400000 [00:23<00:33, 7053.41it/s] 41%|████      | 163831/400000 [00:23<00:33, 7048.35it/s] 41%|████      | 164537/400000 [00:23<00:33, 7051.39it/s] 41%|████▏     | 165246/400000 [00:23<00:33, 7061.03it/s] 41%|████▏     | 165968/400000 [00:23<00:32, 7106.11it/s] 42%|████▏     | 166679/400000 [00:23<00:32, 7093.28it/s] 42%|████▏     | 167429/400000 [00:23<00:32, 7209.34it/s] 42%|████▏     | 168151/400000 [00:23<00:32, 7142.82it/s] 42%|████▏     | 168866/400000 [00:24<00:32, 7134.09it/s] 42%|████▏     | 169580/400000 [00:24<00:32, 7106.94it/s] 43%|████▎     | 170291/400000 [00:24<00:32, 7095.68it/s] 43%|████▎     | 171012/400000 [00:24<00:32, 7129.37it/s] 43%|████▎     | 171735/400000 [00:24<00:31, 7158.58it/s] 43%|████▎     | 172483/400000 [00:24<00:31, 7249.91it/s] 43%|████▎     | 173216/400000 [00:24<00:31, 7266.43it/s] 43%|████▎     | 173943/400000 [00:24<00:31, 7221.51it/s] 44%|████▎     | 174666/400000 [00:24<00:31, 7185.00it/s] 44%|████▍     | 175412/400000 [00:24<00:30, 7265.13it/s] 44%|████▍     | 176149/400000 [00:25<00:30, 7293.26it/s] 44%|████▍     | 176879/400000 [00:25<00:30, 7198.59it/s] 44%|████▍     | 177600/400000 [00:25<00:31, 7112.32it/s] 45%|████▍     | 178312/400000 [00:25<00:31, 6947.22it/s] 45%|████▍     | 179021/400000 [00:25<00:31, 6986.83it/s] 45%|████▍     | 179732/400000 [00:25<00:31, 7021.72it/s] 45%|████▌     | 180435/400000 [00:25<00:31, 7013.15it/s] 45%|████▌     | 181139/400000 [00:25<00:31, 7019.37it/s] 45%|████▌     | 181842/400000 [00:25<00:31, 6909.22it/s] 46%|████▌     | 182552/400000 [00:25<00:31, 6964.67it/s] 46%|████▌     | 183289/400000 [00:26<00:30, 7081.14it/s] 46%|████▌     | 183998/400000 [00:26<00:30, 7080.89it/s] 46%|████▌     | 184707/400000 [00:26<00:30, 6950.24it/s] 46%|████▋     | 185418/400000 [00:26<00:30, 6997.29it/s] 47%|████▋     | 186151/400000 [00:26<00:30, 7091.41it/s] 47%|████▋     | 186874/400000 [00:26<00:29, 7131.01it/s] 47%|████▋     | 187588/400000 [00:26<00:29, 7094.93it/s] 47%|████▋     | 188299/400000 [00:26<00:29, 7098.82it/s] 47%|████▋     | 189010/400000 [00:26<00:29, 7034.24it/s] 47%|████▋     | 189714/400000 [00:26<00:30, 6938.12it/s] 48%|████▊     | 190409/400000 [00:27<00:30, 6927.92it/s] 48%|████▊     | 191108/400000 [00:27<00:30, 6945.74it/s] 48%|████▊     | 191803/400000 [00:27<00:30, 6888.87it/s] 48%|████▊     | 192502/400000 [00:27<00:30, 6916.51it/s] 48%|████▊     | 193207/400000 [00:27<00:29, 6955.20it/s] 48%|████▊     | 193922/400000 [00:27<00:29, 7011.84it/s] 49%|████▊     | 194639/400000 [00:27<00:29, 7056.30it/s] 49%|████▉     | 195376/400000 [00:27<00:28, 7144.70it/s] 49%|████▉     | 196109/400000 [00:27<00:28, 7198.98it/s] 49%|████▉     | 196830/400000 [00:27<00:28, 7122.11it/s] 49%|████▉     | 197543/400000 [00:28<00:29, 6896.27it/s] 50%|████▉     | 198235/400000 [00:28<00:29, 6853.61it/s] 50%|████▉     | 198928/400000 [00:28<00:29, 6876.17it/s] 50%|████▉     | 199660/400000 [00:28<00:28, 7003.41it/s] 50%|█████     | 200362/400000 [00:28<00:28, 6991.70it/s] 50%|█████     | 201063/400000 [00:28<00:29, 6763.74it/s] 50%|█████     | 201742/400000 [00:28<00:29, 6679.92it/s] 51%|█████     | 202448/400000 [00:28<00:29, 6789.07it/s] 51%|█████     | 203188/400000 [00:28<00:28, 6958.36it/s] 51%|█████     | 203913/400000 [00:29<00:27, 7042.28it/s] 51%|█████     | 204644/400000 [00:29<00:27, 7120.45it/s] 51%|█████▏    | 205364/400000 [00:29<00:27, 7143.21it/s] 52%|█████▏    | 206084/400000 [00:29<00:27, 7159.24it/s] 52%|█████▏    | 206801/400000 [00:29<00:26, 7157.55it/s] 52%|█████▏    | 207530/400000 [00:29<00:26, 7196.73it/s] 52%|█████▏    | 208253/400000 [00:29<00:26, 7204.89it/s] 52%|█████▏    | 208990/400000 [00:29<00:26, 7253.16it/s] 52%|█████▏    | 209716/400000 [00:29<00:26, 7184.40it/s] 53%|█████▎    | 210451/400000 [00:29<00:26, 7231.44it/s] 53%|█████▎    | 211175/400000 [00:30<00:26, 7139.91it/s] 53%|█████▎    | 211890/400000 [00:30<00:26, 7124.40it/s] 53%|█████▎    | 212603/400000 [00:30<00:26, 7084.93it/s] 53%|█████▎    | 213312/400000 [00:30<00:26, 7049.41it/s] 54%|█████▎    | 214018/400000 [00:30<00:26, 6986.21it/s] 54%|█████▎    | 214740/400000 [00:30<00:26, 7053.97it/s] 54%|█████▍    | 215485/400000 [00:30<00:25, 7167.65it/s] 54%|█████▍    | 216210/400000 [00:30<00:25, 7190.91it/s] 54%|█████▍    | 216930/400000 [00:30<00:25, 7186.25it/s] 54%|█████▍    | 217662/400000 [00:30<00:25, 7224.67it/s] 55%|█████▍    | 218385/400000 [00:31<00:25, 7205.01it/s] 55%|█████▍    | 219116/400000 [00:31<00:25, 7232.00it/s] 55%|█████▍    | 219840/400000 [00:31<00:24, 7212.05it/s] 55%|█████▌    | 220562/400000 [00:31<00:25, 7160.95it/s] 55%|█████▌    | 221310/400000 [00:31<00:24, 7253.35it/s] 56%|█████▌    | 222036/400000 [00:31<00:24, 7207.93it/s] 56%|█████▌    | 222758/400000 [00:31<00:24, 7186.80it/s] 56%|█████▌    | 223503/400000 [00:31<00:24, 7263.25it/s] 56%|█████▌    | 224230/400000 [00:31<00:24, 7192.32it/s] 56%|█████▌    | 224950/400000 [00:31<00:24, 7170.76it/s] 56%|█████▋    | 225668/400000 [00:32<00:24, 7166.51it/s] 57%|█████▋    | 226385/400000 [00:32<00:24, 7098.57it/s] 57%|█████▋    | 227096/400000 [00:32<00:24, 7068.47it/s] 57%|█████▋    | 227804/400000 [00:32<00:24, 7030.81it/s] 57%|█████▋    | 228508/400000 [00:32<00:24, 7030.22it/s] 57%|█████▋    | 229212/400000 [00:32<00:24, 7033.03it/s] 57%|█████▋    | 229922/400000 [00:32<00:24, 7050.64it/s] 58%|█████▊    | 230641/400000 [00:32<00:23, 7089.14it/s] 58%|█████▊    | 231351/400000 [00:32<00:23, 7077.01it/s] 58%|█████▊    | 232072/400000 [00:32<00:23, 7115.91it/s] 58%|█████▊    | 232800/400000 [00:33<00:23, 7161.94it/s] 58%|█████▊    | 233517/400000 [00:33<00:23, 7086.05it/s] 59%|█████▊    | 234232/400000 [00:33<00:23, 7102.48it/s] 59%|█████▊    | 234943/400000 [00:33<00:23, 6991.98it/s] 59%|█████▉    | 235643/400000 [00:33<00:23, 6985.77it/s] 59%|█████▉    | 236385/400000 [00:33<00:23, 7108.78it/s] 59%|█████▉    | 237137/400000 [00:33<00:22, 7224.25it/s] 59%|█████▉    | 237863/400000 [00:33<00:22, 7234.18it/s] 60%|█████▉    | 238597/400000 [00:33<00:22, 7263.99it/s] 60%|█████▉    | 239352/400000 [00:33<00:21, 7344.99it/s] 60%|██████    | 240088/400000 [00:34<00:22, 7260.96it/s] 60%|██████    | 240815/400000 [00:34<00:22, 7178.05it/s] 60%|██████    | 241547/400000 [00:34<00:21, 7216.67it/s] 61%|██████    | 242283/400000 [00:34<00:21, 7256.10it/s] 61%|██████    | 243060/400000 [00:34<00:21, 7401.62it/s] 61%|██████    | 243816/400000 [00:34<00:20, 7446.94it/s] 61%|██████    | 244562/400000 [00:34<00:21, 7304.94it/s] 61%|██████▏   | 245294/400000 [00:34<00:21, 7098.43it/s] 62%|██████▏   | 246006/400000 [00:34<00:22, 6956.08it/s] 62%|██████▏   | 246704/400000 [00:34<00:22, 6953.49it/s] 62%|██████▏   | 247421/400000 [00:35<00:21, 7016.29it/s] 62%|██████▏   | 248137/400000 [00:35<00:21, 7057.91it/s] 62%|██████▏   | 248844/400000 [00:35<00:21, 7013.12it/s] 62%|██████▏   | 249546/400000 [00:35<00:21, 6855.89it/s] 63%|██████▎   | 250249/400000 [00:35<00:21, 6906.94it/s] 63%|██████▎   | 250941/400000 [00:35<00:21, 6906.52it/s] 63%|██████▎   | 251662/400000 [00:35<00:21, 6993.66it/s] 63%|██████▎   | 252363/400000 [00:35<00:21, 6961.47it/s] 63%|██████▎   | 253072/400000 [00:35<00:20, 6997.34it/s] 63%|██████▎   | 253773/400000 [00:36<00:21, 6909.73it/s] 64%|██████▎   | 254496/400000 [00:36<00:20, 7002.64it/s] 64%|██████▍   | 255197/400000 [00:36<00:20, 6974.85it/s] 64%|██████▍   | 255912/400000 [00:36<00:20, 7024.17it/s] 64%|██████▍   | 256615/400000 [00:36<00:21, 6693.93it/s] 64%|██████▍   | 257299/400000 [00:36<00:21, 6736.53it/s] 65%|██████▍   | 258003/400000 [00:36<00:20, 6823.49it/s] 65%|██████▍   | 258690/400000 [00:36<00:20, 6835.53it/s] 65%|██████▍   | 259423/400000 [00:36<00:20, 6974.97it/s] 65%|██████▌   | 260145/400000 [00:36<00:19, 7043.51it/s] 65%|██████▌   | 260866/400000 [00:37<00:19, 7092.30it/s] 65%|██████▌   | 261609/400000 [00:37<00:19, 7186.76it/s] 66%|██████▌   | 262329/400000 [00:37<00:19, 7189.21it/s] 66%|██████▌   | 263080/400000 [00:37<00:18, 7281.95it/s] 66%|██████▌   | 263809/400000 [00:37<00:19, 7129.18it/s] 66%|██████▌   | 264551/400000 [00:37<00:18, 7211.79it/s] 66%|██████▋   | 265277/400000 [00:37<00:18, 7225.18it/s] 67%|██████▋   | 266058/400000 [00:37<00:18, 7389.11it/s] 67%|██████▋   | 266813/400000 [00:37<00:17, 7434.65it/s] 67%|██████▋   | 267558/400000 [00:37<00:17, 7420.74it/s] 67%|██████▋   | 268301/400000 [00:38<00:17, 7349.05it/s] 67%|██████▋   | 269037/400000 [00:38<00:18, 7107.88it/s] 67%|██████▋   | 269750/400000 [00:38<00:18, 7078.78it/s] 68%|██████▊   | 270467/400000 [00:38<00:18, 7102.29it/s] 68%|██████▊   | 271188/400000 [00:38<00:18, 7133.18it/s] 68%|██████▊   | 271920/400000 [00:38<00:17, 7187.83it/s] 68%|██████▊   | 272640/400000 [00:38<00:17, 7190.01it/s] 68%|██████▊   | 273386/400000 [00:38<00:17, 7266.20it/s] 69%|██████▊   | 274114/400000 [00:38<00:17, 7092.82it/s] 69%|██████▊   | 274825/400000 [00:38<00:17, 6957.02it/s] 69%|██████▉   | 275523/400000 [00:39<00:18, 6823.61it/s] 69%|██████▉   | 276215/400000 [00:39<00:18, 6851.60it/s] 69%|██████▉   | 276902/400000 [00:39<00:18, 6786.95it/s] 69%|██████▉   | 277595/400000 [00:39<00:17, 6827.64it/s] 70%|██████▉   | 278309/400000 [00:39<00:17, 6918.08it/s] 70%|██████▉   | 279002/400000 [00:39<00:17, 6826.56it/s] 70%|██████▉   | 279686/400000 [00:39<00:17, 6697.10it/s] 70%|███████   | 280361/400000 [00:39<00:17, 6711.61it/s] 70%|███████   | 281115/400000 [00:39<00:17, 6937.35it/s] 70%|███████   | 281831/400000 [00:39<00:16, 7000.41it/s] 71%|███████   | 282540/400000 [00:40<00:16, 7026.32it/s] 71%|███████   | 283280/400000 [00:40<00:16, 7134.17it/s] 71%|███████   | 283996/400000 [00:40<00:16, 7141.22it/s] 71%|███████   | 284712/400000 [00:40<00:16, 7075.55it/s] 71%|███████▏  | 285430/400000 [00:40<00:16, 7105.80it/s] 72%|███████▏  | 286142/400000 [00:40<00:16, 6906.73it/s] 72%|███████▏  | 286891/400000 [00:40<00:15, 7071.05it/s] 72%|███████▏  | 287650/400000 [00:40<00:15, 7217.41it/s] 72%|███████▏  | 288374/400000 [00:40<00:15, 7184.59it/s] 72%|███████▏  | 289119/400000 [00:41<00:15, 7259.01it/s] 72%|███████▏  | 289877/400000 [00:41<00:14, 7351.18it/s] 73%|███████▎  | 290614/400000 [00:41<00:14, 7311.76it/s] 73%|███████▎  | 291348/400000 [00:41<00:14, 7319.21it/s] 73%|███████▎  | 292081/400000 [00:41<00:14, 7204.39it/s] 73%|███████▎  | 292803/400000 [00:41<00:14, 7158.40it/s] 73%|███████▎  | 293520/400000 [00:41<00:15, 7073.11it/s] 74%|███████▎  | 294228/400000 [00:41<00:15, 7051.20it/s] 74%|███████▎  | 294934/400000 [00:41<00:15, 6972.90it/s] 74%|███████▍  | 295658/400000 [00:41<00:14, 7048.61it/s] 74%|███████▍  | 296369/400000 [00:42<00:14, 7064.43it/s] 74%|███████▍  | 297086/400000 [00:42<00:14, 7094.34it/s] 74%|███████▍  | 297824/400000 [00:42<00:14, 7177.65it/s] 75%|███████▍  | 298565/400000 [00:42<00:14, 7244.46it/s] 75%|███████▍  | 299290/400000 [00:42<00:14, 7158.42it/s] 75%|███████▌  | 300007/400000 [00:42<00:14, 7007.91it/s] 75%|███████▌  | 300731/400000 [00:42<00:14, 7074.92it/s] 75%|███████▌  | 301449/400000 [00:42<00:13, 7103.97it/s] 76%|███████▌  | 302220/400000 [00:42<00:13, 7274.89it/s] 76%|███████▌  | 303005/400000 [00:42<00:13, 7436.18it/s] 76%|███████▌  | 303751/400000 [00:43<00:13, 7322.73it/s] 76%|███████▌  | 304486/400000 [00:43<00:13, 7259.94it/s] 76%|███████▋  | 305214/400000 [00:43<00:13, 7173.59it/s] 76%|███████▋  | 305933/400000 [00:43<00:13, 7122.23it/s] 77%|███████▋  | 306647/400000 [00:43<00:13, 7115.78it/s] 77%|███████▋  | 307360/400000 [00:43<00:13, 7071.19it/s] 77%|███████▋  | 308068/400000 [00:43<00:13, 7056.37it/s] 77%|███████▋  | 308797/400000 [00:43<00:12, 7123.42it/s] 77%|███████▋  | 309557/400000 [00:43<00:12, 7259.72it/s] 78%|███████▊  | 310284/400000 [00:43<00:12, 7186.96it/s] 78%|███████▊  | 311026/400000 [00:44<00:12, 7253.54it/s] 78%|███████▊  | 311753/400000 [00:44<00:12, 7206.92it/s] 78%|███████▊  | 312492/400000 [00:44<00:12, 7257.92it/s] 78%|███████▊  | 313241/400000 [00:44<00:11, 7323.12it/s] 78%|███████▊  | 313974/400000 [00:44<00:12, 7002.38it/s] 79%|███████▊  | 314678/400000 [00:44<00:12, 6986.20it/s] 79%|███████▉  | 315409/400000 [00:44<00:11, 7079.29it/s] 79%|███████▉  | 316143/400000 [00:44<00:11, 7154.97it/s] 79%|███████▉  | 316884/400000 [00:44<00:11, 7226.73it/s] 79%|███████▉  | 317608/400000 [00:44<00:11, 7173.51it/s] 80%|███████▉  | 318328/400000 [00:45<00:11, 7178.36it/s] 80%|███████▉  | 319047/400000 [00:45<00:11, 7079.80it/s] 80%|███████▉  | 319778/400000 [00:45<00:11, 7147.12it/s] 80%|████████  | 320498/400000 [00:45<00:11, 7162.53it/s] 80%|████████  | 321225/400000 [00:45<00:10, 7193.93it/s] 80%|████████  | 321945/400000 [00:45<00:10, 7156.91it/s] 81%|████████  | 322662/400000 [00:45<00:10, 7136.48it/s] 81%|████████  | 323394/400000 [00:45<00:10, 7189.56it/s] 81%|████████  | 324114/400000 [00:45<00:10, 7174.02it/s] 81%|████████  | 324832/400000 [00:45<00:10, 7157.06it/s] 81%|████████▏ | 325549/400000 [00:46<00:10, 7158.39it/s] 82%|████████▏ | 326265/400000 [00:46<00:10, 7131.24it/s] 82%|████████▏ | 326979/400000 [00:46<00:10, 7125.28it/s] 82%|████████▏ | 327718/400000 [00:46<00:10, 7201.95it/s] 82%|████████▏ | 328439/400000 [00:46<00:09, 7178.15it/s] 82%|████████▏ | 329158/400000 [00:46<00:09, 7157.93it/s] 82%|████████▏ | 329879/400000 [00:46<00:09, 7172.42it/s] 83%|████████▎ | 330597/400000 [00:46<00:09, 7135.53it/s] 83%|████████▎ | 331311/400000 [00:46<00:09, 7083.66it/s] 83%|████████▎ | 332020/400000 [00:47<00:09, 6867.98it/s] 83%|████████▎ | 332740/400000 [00:47<00:09, 6962.49it/s] 83%|████████▎ | 333438/400000 [00:47<00:09, 6959.36it/s] 84%|████████▎ | 334160/400000 [00:47<00:09, 7035.48it/s] 84%|████████▎ | 334865/400000 [00:47<00:09, 7036.97it/s] 84%|████████▍ | 335570/400000 [00:47<00:09, 7019.22it/s] 84%|████████▍ | 336277/400000 [00:47<00:09, 7031.23it/s] 84%|████████▍ | 337017/400000 [00:47<00:08, 7137.74it/s] 84%|████████▍ | 337732/400000 [00:47<00:08, 7141.35it/s] 85%|████████▍ | 338451/400000 [00:47<00:08, 7154.61it/s] 85%|████████▍ | 339171/400000 [00:48<00:08, 7166.12it/s] 85%|████████▍ | 339893/400000 [00:48<00:08, 7182.12it/s] 85%|████████▌ | 340612/400000 [00:48<00:08, 7163.25it/s] 85%|████████▌ | 341329/400000 [00:48<00:08, 7143.85it/s] 86%|████████▌ | 342070/400000 [00:48<00:08, 7221.35it/s] 86%|████████▌ | 342844/400000 [00:48<00:07, 7366.29it/s] 86%|████████▌ | 343582/400000 [00:48<00:07, 7315.13it/s] 86%|████████▌ | 344315/400000 [00:48<00:07, 7177.69it/s] 86%|████████▋ | 345077/400000 [00:48<00:07, 7304.76it/s] 86%|████████▋ | 345809/400000 [00:48<00:07, 7239.67it/s] 87%|████████▋ | 346535/400000 [00:49<00:07, 7147.96it/s] 87%|████████▋ | 347251/400000 [00:49<00:07, 7049.45it/s] 87%|████████▋ | 347976/400000 [00:49<00:07, 7107.15it/s] 87%|████████▋ | 348734/400000 [00:49<00:07, 7240.44it/s] 87%|████████▋ | 349482/400000 [00:49<00:06, 7309.86it/s] 88%|████████▊ | 350234/400000 [00:49<00:06, 7369.73it/s] 88%|████████▊ | 350985/400000 [00:49<00:06, 7410.68it/s] 88%|████████▊ | 351727/400000 [00:49<00:06, 7246.25it/s] 88%|████████▊ | 352476/400000 [00:49<00:06, 7315.25it/s] 88%|████████▊ | 353222/400000 [00:49<00:06, 7357.63it/s] 88%|████████▊ | 353959/400000 [00:50<00:06, 7325.92it/s] 89%|████████▊ | 354693/400000 [00:50<00:06, 7228.26it/s] 89%|████████▉ | 355454/400000 [00:50<00:06, 7337.13it/s] 89%|████████▉ | 356189/400000 [00:50<00:06, 7292.72it/s] 89%|████████▉ | 356919/400000 [00:50<00:05, 7212.65it/s] 89%|████████▉ | 357641/400000 [00:50<00:05, 7208.54it/s] 90%|████████▉ | 358363/400000 [00:50<00:06, 6856.34it/s] 90%|████████▉ | 359089/400000 [00:50<00:05, 6971.32it/s] 90%|████████▉ | 359809/400000 [00:50<00:05, 7035.79it/s] 90%|█████████ | 360530/400000 [00:50<00:05, 7086.04it/s] 90%|█████████ | 361258/400000 [00:51<00:05, 7140.94it/s] 90%|█████████ | 361982/400000 [00:51<00:05, 7170.12it/s] 91%|█████████ | 362718/400000 [00:51<00:05, 7225.23it/s] 91%|█████████ | 363442/400000 [00:51<00:05, 7219.39it/s] 91%|█████████ | 364175/400000 [00:51<00:04, 7248.77it/s] 91%|█████████ | 364901/400000 [00:51<00:04, 7249.57it/s] 91%|█████████▏| 365627/400000 [00:51<00:04, 7188.72it/s] 92%|█████████▏| 366400/400000 [00:51<00:04, 7340.76it/s] 92%|█████████▏| 367136/400000 [00:51<00:04, 7321.64it/s] 92%|█████████▏| 367874/400000 [00:51<00:04, 7338.39it/s] 92%|█████████▏| 368609/400000 [00:52<00:04, 7310.33it/s] 92%|█████████▏| 369358/400000 [00:52<00:04, 7361.91it/s] 93%|█████████▎| 370112/400000 [00:52<00:04, 7412.18it/s] 93%|█████████▎| 370868/400000 [00:52<00:03, 7455.02it/s] 93%|█████████▎| 371616/400000 [00:52<00:03, 7458.34it/s] 93%|█████████▎| 372363/400000 [00:52<00:03, 7342.90it/s] 93%|█████████▎| 373098/400000 [00:52<00:03, 7289.10it/s] 93%|█████████▎| 373828/400000 [00:52<00:03, 7254.76it/s] 94%|█████████▎| 374554/400000 [00:52<00:03, 7170.80it/s] 94%|█████████▍| 375277/400000 [00:53<00:03, 7188.10it/s] 94%|█████████▍| 375997/400000 [00:53<00:03, 6994.22it/s] 94%|█████████▍| 376698/400000 [00:53<00:03, 6956.94it/s] 94%|█████████▍| 377413/400000 [00:53<00:03, 7012.32it/s] 95%|█████████▍| 378118/400000 [00:53<00:03, 7020.97it/s] 95%|█████████▍| 378821/400000 [00:53<00:03, 6967.02it/s] 95%|█████████▍| 379544/400000 [00:53<00:02, 7041.59it/s] 95%|█████████▌| 380249/400000 [00:53<00:02, 7040.19it/s] 95%|█████████▌| 380985/400000 [00:53<00:02, 7130.14it/s] 95%|█████████▌| 381702/400000 [00:53<00:02, 7140.24it/s] 96%|█████████▌| 382432/400000 [00:54<00:02, 7186.98it/s] 96%|█████████▌| 383155/400000 [00:54<00:02, 7199.07it/s] 96%|█████████▌| 383876/400000 [00:54<00:02, 6886.55it/s] 96%|█████████▌| 384568/400000 [00:54<00:02, 6851.93it/s] 96%|█████████▋| 385274/400000 [00:54<00:02, 6912.32it/s] 97%|█████████▋| 386019/400000 [00:54<00:01, 7065.14it/s] 97%|█████████▋| 386741/400000 [00:54<00:01, 7110.61it/s] 97%|█████████▋| 387475/400000 [00:54<00:01, 7176.38it/s] 97%|█████████▋| 388211/400000 [00:54<00:01, 7229.50it/s] 97%|█████████▋| 388935/400000 [00:54<00:01, 7196.76it/s] 97%|█████████▋| 389668/400000 [00:55<00:01, 7234.59it/s] 98%|█████████▊| 390411/400000 [00:55<00:01, 7290.73it/s] 98%|█████████▊| 391141/400000 [00:55<00:01, 7204.67it/s] 98%|█████████▊| 391872/400000 [00:55<00:01, 7234.08it/s] 98%|█████████▊| 392596/400000 [00:55<00:01, 7200.62it/s] 98%|█████████▊| 393325/400000 [00:55<00:00, 7226.49it/s] 99%|█████████▊| 394048/400000 [00:55<00:00, 7223.43it/s] 99%|█████████▊| 394771/400000 [00:55<00:00, 7142.29it/s] 99%|█████████▉| 395486/400000 [00:55<00:00, 7095.36it/s] 99%|█████████▉| 396208/400000 [00:55<00:00, 7131.26it/s] 99%|█████████▉| 396931/400000 [00:56<00:00, 7158.06it/s] 99%|█████████▉| 397673/400000 [00:56<00:00, 7234.23it/s]100%|█████████▉| 398397/400000 [00:56<00:00, 7117.85it/s]100%|█████████▉| 399110/400000 [00:56<00:00, 6752.75it/s]100%|█████████▉| 399817/400000 [00:56<00:00, 6844.75it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7079.92it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdeca32ef28> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0110086350140824 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011214534574527804 	 Accuracy: 51

  model saves at 51% accuracy 

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

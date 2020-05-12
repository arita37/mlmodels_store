
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5f42ab3f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 04:13:17.453308
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 04:13:17.457394
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 04:13:17.461004
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 04:13:17.464698
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5f4e877400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353291.0938
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 222849.0000
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 128055.5938
Epoch 4/10

1/1 [==============================] - 0s 109ms/step - loss: 64921.3203
Epoch 5/10

1/1 [==============================] - 0s 112ms/step - loss: 32270.4668
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 17853.0977
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 11074.8828
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 7515.0244
Epoch 9/10

1/1 [==============================] - 0s 121ms/step - loss: 5474.7627
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 4231.6367

  #### Inference Need return ypred, ytrue ######################### 
[[-1.97470975e+00 -1.52138186e+00  1.95228767e+00 -1.45775485e+00
  -1.74208248e+00  2.09995657e-01  1.46843046e-01  9.74410892e-01
  -6.97139800e-01  1.86011708e+00  1.46702504e+00  2.12296844e-02
  -8.14136505e-01  2.29377031e-01  2.01818013e+00 -2.30869591e-01
  -2.24788237e+00  6.16086125e-01  8.73360336e-01  1.54699039e+00
   2.25088310e+00 -1.24608755e-01  2.80442506e-01  4.94566679e-01
   6.51045322e-01  9.47460175e-01 -2.65354633e+00  2.08720112e+00
  -1.21691513e+00  1.42456484e+00 -2.95824575e+00 -5.36766648e-01
  -2.70557344e-01  2.22301483e-03  1.94073844e+00  1.38861513e+00
   2.95521355e+00 -7.61238933e-01  1.24241233e+00 -2.44732666e+00
  -9.70926940e-01 -1.12782383e+00 -1.20870399e+00 -6.38625264e-01
   1.07025015e+00  1.56156731e+00  7.14422345e-01  1.26881552e+00
  -8.53563249e-01 -6.74919009e-01  1.73091400e+00  4.50938344e-01
   4.97798324e-01  8.12126756e-01  4.24888790e-01  2.91376090e+00
   5.36920071e-01  1.08384764e+00  1.77661467e+00  1.33227259e-01
  -8.20268750e-01  8.93048191e+00  9.18673515e+00  1.04021235e+01
   8.47541332e+00  9.94421577e+00  1.01689901e+01  9.43973637e+00
   1.31796818e+01  9.67363930e+00  1.07975454e+01  1.22574196e+01
   1.05715475e+01  1.31160183e+01  1.00705519e+01  9.25255680e+00
   1.14018278e+01  1.07683506e+01  1.12253551e+01  1.19068527e+01
   1.02113523e+01  1.16462812e+01  9.33645153e+00  9.44044971e+00
   1.00920410e+01  9.22607231e+00  1.11184692e+01  8.76061344e+00
   8.48004627e+00  1.17350197e+01  1.09100323e+01  7.87687397e+00
   1.01612577e+01  9.26798153e+00  1.06207590e+01  1.16771564e+01
   1.16890459e+01  1.34310322e+01  1.09924183e+01  1.12290592e+01
   8.99853802e+00  1.25101681e+01  1.09174223e+01  1.08647842e+01
   1.25883045e+01  1.00813675e+01  1.15384722e+01  8.36899281e+00
   1.22196074e+01  9.85523510e+00  9.19942093e+00  1.03035822e+01
   1.03294840e+01  9.71436787e+00  8.49291039e+00  1.06172886e+01
   1.06528835e+01  9.84043312e+00  1.16459017e+01  1.06907053e+01
   2.14576793e+00  2.45450807e+00  7.01739669e-01  1.02101910e+00
   7.25425899e-01  9.92711484e-01  8.34186971e-01 -1.48548394e-01
  -1.02373743e+00  1.60760164e+00  9.32463109e-01 -3.27394605e-02
  -1.20730531e+00 -2.74301291e+00  2.57060230e-02  9.44267094e-01
  -3.21400434e-01 -2.90320009e-01 -3.21156025e-01 -1.05734754e+00
  -7.16468096e-01  7.97892034e-01 -3.60982418e-02 -1.16254961e+00
   1.02253652e+00 -6.02655172e-01  7.95748770e-01  6.47556543e-01
   9.60417688e-01 -4.56806779e-01 -2.82740188e+00 -1.81052756e+00
  -1.25986981e+00  2.24662924e+00 -1.75446248e+00 -1.84770823e-01
   2.33199430e+00  6.83646083e-01 -5.39119661e-01  1.17198431e+00
  -2.51277852e+00  1.30567741e+00 -6.34830952e-01 -2.37386823e-02
  -4.08617854e-02  1.69671702e+00 -2.10181475e-02 -9.12062287e-01
   1.09143138e+00  1.19309354e+00  8.70211422e-01  9.27624643e-01
   1.83234954e+00  7.78910160e-01 -2.06197500e+00  4.21823859e-01
   5.77185512e-01 -1.56178522e+00 -2.67311096e+00 -1.70060897e+00
   7.00032830e-01  4.22393739e-01  4.16360915e-01  8.07684541e-01
   1.86612654e+00  2.28597403e-01  1.16731262e+00  1.03254354e+00
   2.47813106e-01  1.32688344e-01  2.76405096e+00  9.95680273e-01
   2.68051767e+00  2.77112913e+00  3.85264575e-01  4.25783515e-01
   2.20447540e+00  6.52828813e-01  4.25381303e-01  1.89996123e-01
   1.83217573e+00  1.33448815e+00  7.52219737e-01  1.81469893e+00
   4.38566923e-01  1.06854200e-01  7.27644205e-01  4.11879897e-01
   3.22286272e+00  2.14635670e-01  9.90045488e-01  6.63857698e-01
   2.14819050e+00  2.15291166e+00  6.57311201e-01  2.33257127e+00
   7.86222577e-01  1.62550437e+00  2.82594681e+00  1.54420686e+00
   6.47420526e-01  1.25751281e+00  1.30457950e+00  5.93805254e-01
   2.50373125e+00  2.06270289e+00  3.43945563e-01  3.09041381e-01
   3.38182211e-01  1.60493374e-01  8.92982125e-01  1.05664897e+00
   5.05469739e-01  1.45370483e-01  1.20252776e+00  1.62352979e+00
   6.72005415e-02  2.86728144e-01  2.57778764e-01  1.20352721e+00
   6.97904229e-01  1.09629087e+01  1.23340712e+01  1.04126129e+01
   1.15615406e+01  9.63106823e+00  9.93852997e+00  8.92493057e+00
   1.12779255e+01  1.20087624e+01  1.09708271e+01  1.07141733e+01
   1.08041639e+01  9.25259113e+00  1.16028061e+01  1.09497099e+01
   1.31344042e+01  1.00051003e+01  1.19898758e+01  1.00359488e+01
   1.19800482e+01  9.56004810e+00  1.18099079e+01  1.12776098e+01
   9.55943108e+00  1.11689930e+01  1.07114410e+01  8.95371723e+00
   1.06668425e+01  1.12357674e+01  1.13075361e+01  9.11784935e+00
   1.27767344e+01  1.07266397e+01  9.92068481e+00  1.28650684e+01
   9.10233307e+00  1.13680267e+01  1.03389006e+01  1.06871386e+01
   1.00881186e+01  1.14826260e+01  1.12625380e+01  1.13952589e+01
   1.11684980e+01  1.22209921e+01  1.02463131e+01  1.32509727e+01
   1.04984598e+01  1.19669056e+01  1.22847738e+01  1.07427893e+01
   1.23571091e+01  9.74756527e+00  9.34778786e+00  1.08466797e+01
   1.21351929e+01  9.75775337e+00  8.81539154e+00  1.09257679e+01
   2.06134796e-01  3.34198534e-01  2.54925489e+00  6.83462620e-02
   3.77776623e-01  1.84524369e+00  1.27262473e+00  1.54180074e+00
   1.23261845e+00  1.75726235e+00  1.47608960e+00  2.30859160e-01
   6.60981774e-01  3.43096375e-01  1.61800122e+00  5.98961651e-01
   2.85181379e+00  1.29806399e-01  1.67654419e+00  9.02713478e-01
   9.88527358e-01  8.74272943e-01  1.29117393e+00  1.05033135e+00
   9.41216469e-01  1.45549858e+00  3.89200091e-01  4.21644866e-01
   7.52238095e-01  2.23262143e+00  2.20517969e+00  3.03820908e-01
   2.03963852e+00  1.91425490e+00  3.90869915e-01  8.13798308e-01
   1.09349370e+00  6.86426282e-01  1.67782331e+00  5.18469691e-01
   8.40969682e-01  2.17105150e-01  7.12268353e-02  8.61193895e-01
   2.08639145e+00  1.74093235e+00  1.55455947e+00  1.27417135e+00
   3.01177621e-01  3.95057917e-01  1.58470750e+00  2.79042482e+00
   2.55074978e+00  2.41908216e+00  2.04745436e+00  1.26145399e+00
   1.03056407e+00  3.53747725e-01  6.70354843e-01  1.94063807e+00
  -4.72510958e+00  7.72154474e+00 -6.39259434e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 04:13:26.130798
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0296
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 04:13:26.135548
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8496.08
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 04:13:26.139216
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.2801
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 04:13:26.143078
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -759.884
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140046757950632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140045816701224
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140045816701728
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140045816702232
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140045816702736
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140045816703240

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5f2e48bfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.532825
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.496690
grad_step = 000002, loss = 0.467597
grad_step = 000003, loss = 0.436690
grad_step = 000004, loss = 0.403407
grad_step = 000005, loss = 0.373560
grad_step = 000006, loss = 0.352191
grad_step = 000007, loss = 0.337324
grad_step = 000008, loss = 0.322377
grad_step = 000009, loss = 0.301703
grad_step = 000010, loss = 0.281699
grad_step = 000011, loss = 0.267406
grad_step = 000012, loss = 0.258432
grad_step = 000013, loss = 0.251259
grad_step = 000014, loss = 0.243435
grad_step = 000015, loss = 0.233618
grad_step = 000016, loss = 0.222794
grad_step = 000017, loss = 0.212323
grad_step = 000018, loss = 0.202706
grad_step = 000019, loss = 0.194154
grad_step = 000020, loss = 0.186756
grad_step = 000021, loss = 0.179100
grad_step = 000022, loss = 0.170660
grad_step = 000023, loss = 0.162850
grad_step = 000024, loss = 0.155889
grad_step = 000025, loss = 0.149514
grad_step = 000026, loss = 0.143295
grad_step = 000027, loss = 0.136849
grad_step = 000028, loss = 0.130399
grad_step = 000029, loss = 0.124358
grad_step = 000030, loss = 0.118845
grad_step = 000031, loss = 0.113522
grad_step = 000032, loss = 0.108251
grad_step = 000033, loss = 0.103268
grad_step = 000034, loss = 0.098488
grad_step = 000035, loss = 0.093915
grad_step = 000036, loss = 0.089659
grad_step = 000037, loss = 0.085522
grad_step = 000038, loss = 0.081457
grad_step = 000039, loss = 0.077559
grad_step = 000040, loss = 0.073819
grad_step = 000041, loss = 0.070307
grad_step = 000042, loss = 0.066976
grad_step = 000043, loss = 0.063733
grad_step = 000044, loss = 0.060624
grad_step = 000045, loss = 0.057681
grad_step = 000046, loss = 0.054894
grad_step = 000047, loss = 0.052210
grad_step = 000048, loss = 0.049618
grad_step = 000049, loss = 0.047127
grad_step = 000050, loss = 0.044739
grad_step = 000051, loss = 0.042493
grad_step = 000052, loss = 0.040330
grad_step = 000053, loss = 0.038255
grad_step = 000054, loss = 0.036269
grad_step = 000055, loss = 0.034394
grad_step = 000056, loss = 0.032605
grad_step = 000057, loss = 0.030871
grad_step = 000058, loss = 0.029202
grad_step = 000059, loss = 0.027627
grad_step = 000060, loss = 0.026132
grad_step = 000061, loss = 0.024703
grad_step = 000062, loss = 0.023319
grad_step = 000063, loss = 0.022005
grad_step = 000064, loss = 0.020768
grad_step = 000065, loss = 0.019594
grad_step = 000066, loss = 0.018459
grad_step = 000067, loss = 0.017374
grad_step = 000068, loss = 0.016349
grad_step = 000069, loss = 0.015387
grad_step = 000070, loss = 0.014475
grad_step = 000071, loss = 0.013599
grad_step = 000072, loss = 0.012770
grad_step = 000073, loss = 0.011996
grad_step = 000074, loss = 0.011262
grad_step = 000075, loss = 0.010566
grad_step = 000076, loss = 0.009907
grad_step = 000077, loss = 0.009292
grad_step = 000078, loss = 0.008717
grad_step = 000079, loss = 0.008175
grad_step = 000080, loss = 0.007666
grad_step = 000081, loss = 0.007193
grad_step = 000082, loss = 0.006753
grad_step = 000083, loss = 0.006338
grad_step = 000084, loss = 0.005953
grad_step = 000085, loss = 0.005599
grad_step = 000086, loss = 0.005270
grad_step = 000087, loss = 0.004965
grad_step = 000088, loss = 0.004683
grad_step = 000089, loss = 0.004426
grad_step = 000090, loss = 0.004188
grad_step = 000091, loss = 0.003970
grad_step = 000092, loss = 0.003776
grad_step = 000093, loss = 0.003602
grad_step = 000094, loss = 0.003447
grad_step = 000095, loss = 0.003299
grad_step = 000096, loss = 0.003152
grad_step = 000097, loss = 0.003012
grad_step = 000098, loss = 0.002903
grad_step = 000099, loss = 0.002813
grad_step = 000100, loss = 0.002727
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002634
grad_step = 000102, loss = 0.002552
grad_step = 000103, loss = 0.002493
grad_step = 000104, loss = 0.002442
grad_step = 000105, loss = 0.002390
grad_step = 000106, loss = 0.002338
grad_step = 000107, loss = 0.002290
grad_step = 000108, loss = 0.002256
grad_step = 000109, loss = 0.002230
grad_step = 000110, loss = 0.002202
grad_step = 000111, loss = 0.002176
grad_step = 000112, loss = 0.002148
grad_step = 000113, loss = 0.002124
grad_step = 000114, loss = 0.002107
grad_step = 000115, loss = 0.002093
grad_step = 000116, loss = 0.002081
grad_step = 000117, loss = 0.002073
grad_step = 000118, loss = 0.002068
grad_step = 000119, loss = 0.002061
grad_step = 000120, loss = 0.002058
grad_step = 000121, loss = 0.002056
grad_step = 000122, loss = 0.002047
grad_step = 000123, loss = 0.002034
grad_step = 000124, loss = 0.002019
grad_step = 000125, loss = 0.002002
grad_step = 000126, loss = 0.001990
grad_step = 000127, loss = 0.001985
grad_step = 000128, loss = 0.001983
grad_step = 000129, loss = 0.001985
grad_step = 000130, loss = 0.001990
grad_step = 000131, loss = 0.002000
grad_step = 000132, loss = 0.002013
grad_step = 000133, loss = 0.002026
grad_step = 000134, loss = 0.002024
grad_step = 000135, loss = 0.002004
grad_step = 000136, loss = 0.001967
grad_step = 000137, loss = 0.001942
grad_step = 000138, loss = 0.001937
grad_step = 000139, loss = 0.001950
grad_step = 000140, loss = 0.001966
grad_step = 000141, loss = 0.001971
grad_step = 000142, loss = 0.001958
grad_step = 000143, loss = 0.001936
grad_step = 000144, loss = 0.001917
grad_step = 000145, loss = 0.001905
grad_step = 000146, loss = 0.001906
grad_step = 000147, loss = 0.001915
grad_step = 000148, loss = 0.001922
grad_step = 000149, loss = 0.001926
grad_step = 000150, loss = 0.001922
grad_step = 000151, loss = 0.001913
grad_step = 000152, loss = 0.001898
grad_step = 000153, loss = 0.001884
grad_step = 000154, loss = 0.001874
grad_step = 000155, loss = 0.001869
grad_step = 000156, loss = 0.001866
grad_step = 000157, loss = 0.001865
grad_step = 000158, loss = 0.001865
grad_step = 000159, loss = 0.001870
grad_step = 000160, loss = 0.001882
grad_step = 000161, loss = 0.001896
grad_step = 000162, loss = 0.001917
grad_step = 000163, loss = 0.001937
grad_step = 000164, loss = 0.001949
grad_step = 000165, loss = 0.001941
grad_step = 000166, loss = 0.001890
grad_step = 000167, loss = 0.001842
grad_step = 000168, loss = 0.001828
grad_step = 000169, loss = 0.001843
grad_step = 000170, loss = 0.001865
grad_step = 000171, loss = 0.001875
grad_step = 000172, loss = 0.001872
grad_step = 000173, loss = 0.001861
grad_step = 000174, loss = 0.001828
grad_step = 000175, loss = 0.001805
grad_step = 000176, loss = 0.001810
grad_step = 000177, loss = 0.001824
grad_step = 000178, loss = 0.001828
grad_step = 000179, loss = 0.001821
grad_step = 000180, loss = 0.001815
grad_step = 000181, loss = 0.001803
grad_step = 000182, loss = 0.001787
grad_step = 000183, loss = 0.001783
grad_step = 000184, loss = 0.001789
grad_step = 000185, loss = 0.001792
grad_step = 000186, loss = 0.001790
grad_step = 000187, loss = 0.001787
grad_step = 000188, loss = 0.001786
grad_step = 000189, loss = 0.001785
grad_step = 000190, loss = 0.001778
grad_step = 000191, loss = 0.001770
grad_step = 000192, loss = 0.001764
grad_step = 000193, loss = 0.001760
grad_step = 000194, loss = 0.001755
grad_step = 000195, loss = 0.001749
grad_step = 000196, loss = 0.001743
grad_step = 000197, loss = 0.001740
grad_step = 000198, loss = 0.001738
grad_step = 000199, loss = 0.001736
grad_step = 000200, loss = 0.001732
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001731
grad_step = 000202, loss = 0.001735
grad_step = 000203, loss = 0.001745
grad_step = 000204, loss = 0.001768
grad_step = 000205, loss = 0.001814
grad_step = 000206, loss = 0.001877
grad_step = 000207, loss = 0.001939
grad_step = 000208, loss = 0.001932
grad_step = 000209, loss = 0.001818
grad_step = 000210, loss = 0.001712
grad_step = 000211, loss = 0.001701
grad_step = 000212, loss = 0.001756
grad_step = 000213, loss = 0.001789
grad_step = 000214, loss = 0.001756
grad_step = 000215, loss = 0.001696
grad_step = 000216, loss = 0.001673
grad_step = 000217, loss = 0.001700
grad_step = 000218, loss = 0.001725
grad_step = 000219, loss = 0.001706
grad_step = 000220, loss = 0.001660
grad_step = 000221, loss = 0.001643
grad_step = 000222, loss = 0.001665
grad_step = 000223, loss = 0.001684
grad_step = 000224, loss = 0.001670
grad_step = 000225, loss = 0.001638
grad_step = 000226, loss = 0.001623
grad_step = 000227, loss = 0.001628
grad_step = 000228, loss = 0.001634
grad_step = 000229, loss = 0.001626
grad_step = 000230, loss = 0.001608
grad_step = 000231, loss = 0.001598
grad_step = 000232, loss = 0.001602
grad_step = 000233, loss = 0.001609
grad_step = 000234, loss = 0.001608
grad_step = 000235, loss = 0.001595
grad_step = 000236, loss = 0.001588
grad_step = 000237, loss = 0.001593
grad_step = 000238, loss = 0.001601
grad_step = 000239, loss = 0.001604
grad_step = 000240, loss = 0.001598
grad_step = 000241, loss = 0.001589
grad_step = 000242, loss = 0.001583
grad_step = 000243, loss = 0.001574
grad_step = 000244, loss = 0.001561
grad_step = 000245, loss = 0.001546
grad_step = 000246, loss = 0.001538
grad_step = 000247, loss = 0.001540
grad_step = 000248, loss = 0.001544
grad_step = 000249, loss = 0.001544
grad_step = 000250, loss = 0.001544
grad_step = 000251, loss = 0.001547
grad_step = 000252, loss = 0.001556
grad_step = 000253, loss = 0.001564
grad_step = 000254, loss = 0.001567
grad_step = 000255, loss = 0.001565
grad_step = 000256, loss = 0.001560
grad_step = 000257, loss = 0.001547
grad_step = 000258, loss = 0.001531
grad_step = 000259, loss = 0.001513
grad_step = 000260, loss = 0.001502
grad_step = 000261, loss = 0.001497
grad_step = 000262, loss = 0.001498
grad_step = 000263, loss = 0.001501
grad_step = 000264, loss = 0.001504
grad_step = 000265, loss = 0.001507
grad_step = 000266, loss = 0.001511
grad_step = 000267, loss = 0.001514
grad_step = 000268, loss = 0.001515
grad_step = 000269, loss = 0.001514
grad_step = 000270, loss = 0.001511
grad_step = 000271, loss = 0.001508
grad_step = 000272, loss = 0.001504
grad_step = 000273, loss = 0.001498
grad_step = 000274, loss = 0.001490
grad_step = 000275, loss = 0.001484
grad_step = 000276, loss = 0.001477
grad_step = 000277, loss = 0.001472
grad_step = 000278, loss = 0.001467
grad_step = 000279, loss = 0.001462
grad_step = 000280, loss = 0.001458
grad_step = 000281, loss = 0.001455
grad_step = 000282, loss = 0.001453
grad_step = 000283, loss = 0.001450
grad_step = 000284, loss = 0.001448
grad_step = 000285, loss = 0.001446
grad_step = 000286, loss = 0.001444
grad_step = 000287, loss = 0.001442
grad_step = 000288, loss = 0.001442
grad_step = 000289, loss = 0.001443
grad_step = 000290, loss = 0.001448
grad_step = 000291, loss = 0.001460
grad_step = 000292, loss = 0.001483
grad_step = 000293, loss = 0.001535
grad_step = 000294, loss = 0.001611
grad_step = 000295, loss = 0.001737
grad_step = 000296, loss = 0.001817
grad_step = 000297, loss = 0.001783
grad_step = 000298, loss = 0.001650
grad_step = 000299, loss = 0.001494
grad_step = 000300, loss = 0.001460
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001510
grad_step = 000302, loss = 0.001545
grad_step = 000303, loss = 0.001547
grad_step = 000304, loss = 0.001513
grad_step = 000305, loss = 0.001445
grad_step = 000306, loss = 0.001425
grad_step = 000307, loss = 0.001480
grad_step = 000308, loss = 0.001506
grad_step = 000309, loss = 0.001454
grad_step = 000310, loss = 0.001404
grad_step = 000311, loss = 0.001417
grad_step = 000312, loss = 0.001448
grad_step = 000313, loss = 0.001445
grad_step = 000314, loss = 0.001410
grad_step = 000315, loss = 0.001393
grad_step = 000316, loss = 0.001407
grad_step = 000317, loss = 0.001418
grad_step = 000318, loss = 0.001402
grad_step = 000319, loss = 0.001385
grad_step = 000320, loss = 0.001388
grad_step = 000321, loss = 0.001396
grad_step = 000322, loss = 0.001388
grad_step = 000323, loss = 0.001375
grad_step = 000324, loss = 0.001374
grad_step = 000325, loss = 0.001382
grad_step = 000326, loss = 0.001382
grad_step = 000327, loss = 0.001369
grad_step = 000328, loss = 0.001359
grad_step = 000329, loss = 0.001361
grad_step = 000330, loss = 0.001366
grad_step = 000331, loss = 0.001363
grad_step = 000332, loss = 0.001354
grad_step = 000333, loss = 0.001349
grad_step = 000334, loss = 0.001350
grad_step = 000335, loss = 0.001352
grad_step = 000336, loss = 0.001348
grad_step = 000337, loss = 0.001342
grad_step = 000338, loss = 0.001339
grad_step = 000339, loss = 0.001340
grad_step = 000340, loss = 0.001339
grad_step = 000341, loss = 0.001336
grad_step = 000342, loss = 0.001332
grad_step = 000343, loss = 0.001328
grad_step = 000344, loss = 0.001326
grad_step = 000345, loss = 0.001325
grad_step = 000346, loss = 0.001323
grad_step = 000347, loss = 0.001320
grad_step = 000348, loss = 0.001317
grad_step = 000349, loss = 0.001315
grad_step = 000350, loss = 0.001313
grad_step = 000351, loss = 0.001311
grad_step = 000352, loss = 0.001309
grad_step = 000353, loss = 0.001307
grad_step = 000354, loss = 0.001306
grad_step = 000355, loss = 0.001308
grad_step = 000356, loss = 0.001312
grad_step = 000357, loss = 0.001328
grad_step = 000358, loss = 0.001353
grad_step = 000359, loss = 0.001415
grad_step = 000360, loss = 0.001422
grad_step = 000361, loss = 0.001420
grad_step = 000362, loss = 0.001367
grad_step = 000363, loss = 0.001361
grad_step = 000364, loss = 0.001367
grad_step = 000365, loss = 0.001317
grad_step = 000366, loss = 0.001280
grad_step = 000367, loss = 0.001298
grad_step = 000368, loss = 0.001326
grad_step = 000369, loss = 0.001326
grad_step = 000370, loss = 0.001301
grad_step = 000371, loss = 0.001290
grad_step = 000372, loss = 0.001301
grad_step = 000373, loss = 0.001297
grad_step = 000374, loss = 0.001281
grad_step = 000375, loss = 0.001264
grad_step = 000376, loss = 0.001269
grad_step = 000377, loss = 0.001281
grad_step = 000378, loss = 0.001275
grad_step = 000379, loss = 0.001261
grad_step = 000380, loss = 0.001258
grad_step = 000381, loss = 0.001262
grad_step = 000382, loss = 0.001260
grad_step = 000383, loss = 0.001248
grad_step = 000384, loss = 0.001241
grad_step = 000385, loss = 0.001243
grad_step = 000386, loss = 0.001246
grad_step = 000387, loss = 0.001243
grad_step = 000388, loss = 0.001237
grad_step = 000389, loss = 0.001233
grad_step = 000390, loss = 0.001232
grad_step = 000391, loss = 0.001234
grad_step = 000392, loss = 0.001234
grad_step = 000393, loss = 0.001232
grad_step = 000394, loss = 0.001228
grad_step = 000395, loss = 0.001225
grad_step = 000396, loss = 0.001222
grad_step = 000397, loss = 0.001224
grad_step = 000398, loss = 0.001231
grad_step = 000399, loss = 0.001251
grad_step = 000400, loss = 0.001286
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001354
grad_step = 000402, loss = 0.001414
grad_step = 000403, loss = 0.001501
grad_step = 000404, loss = 0.001525
grad_step = 000405, loss = 0.001561
grad_step = 000406, loss = 0.001575
grad_step = 000407, loss = 0.001543
grad_step = 000408, loss = 0.001434
grad_step = 000409, loss = 0.001282
grad_step = 000410, loss = 0.001222
grad_step = 000411, loss = 0.001290
grad_step = 000412, loss = 0.001354
grad_step = 000413, loss = 0.001311
grad_step = 000414, loss = 0.001233
grad_step = 000415, loss = 0.001230
grad_step = 000416, loss = 0.001260
grad_step = 000417, loss = 0.001234
grad_step = 000418, loss = 0.001207
grad_step = 000419, loss = 0.001230
grad_step = 000420, loss = 0.001232
grad_step = 000421, loss = 0.001195
grad_step = 000422, loss = 0.001174
grad_step = 000423, loss = 0.001197
grad_step = 000424, loss = 0.001215
grad_step = 000425, loss = 0.001183
grad_step = 000426, loss = 0.001159
grad_step = 000427, loss = 0.001167
grad_step = 000428, loss = 0.001182
grad_step = 000429, loss = 0.001179
grad_step = 000430, loss = 0.001157
grad_step = 000431, loss = 0.001148
grad_step = 000432, loss = 0.001154
grad_step = 000433, loss = 0.001157
grad_step = 000434, loss = 0.001152
grad_step = 000435, loss = 0.001143
grad_step = 000436, loss = 0.001140
grad_step = 000437, loss = 0.001141
grad_step = 000438, loss = 0.001137
grad_step = 000439, loss = 0.001134
grad_step = 000440, loss = 0.001132
grad_step = 000441, loss = 0.001131
grad_step = 000442, loss = 0.001129
grad_step = 000443, loss = 0.001125
grad_step = 000444, loss = 0.001121
grad_step = 000445, loss = 0.001119
grad_step = 000446, loss = 0.001118
grad_step = 000447, loss = 0.001117
grad_step = 000448, loss = 0.001114
grad_step = 000449, loss = 0.001111
grad_step = 000450, loss = 0.001108
grad_step = 000451, loss = 0.001106
grad_step = 000452, loss = 0.001105
grad_step = 000453, loss = 0.001104
grad_step = 000454, loss = 0.001101
grad_step = 000455, loss = 0.001098
grad_step = 000456, loss = 0.001095
grad_step = 000457, loss = 0.001093
grad_step = 000458, loss = 0.001091
grad_step = 000459, loss = 0.001090
grad_step = 000460, loss = 0.001089
grad_step = 000461, loss = 0.001087
grad_step = 000462, loss = 0.001085
grad_step = 000463, loss = 0.001082
grad_step = 000464, loss = 0.001080
grad_step = 000465, loss = 0.001078
grad_step = 000466, loss = 0.001076
grad_step = 000467, loss = 0.001075
grad_step = 000468, loss = 0.001074
grad_step = 000469, loss = 0.001073
grad_step = 000470, loss = 0.001073
grad_step = 000471, loss = 0.001075
grad_step = 000472, loss = 0.001080
grad_step = 000473, loss = 0.001087
grad_step = 000474, loss = 0.001104
grad_step = 000475, loss = 0.001115
grad_step = 000476, loss = 0.001141
grad_step = 000477, loss = 0.001136
grad_step = 000478, loss = 0.001136
grad_step = 000479, loss = 0.001117
grad_step = 000480, loss = 0.001110
grad_step = 000481, loss = 0.001106
grad_step = 000482, loss = 0.001100
grad_step = 000483, loss = 0.001081
grad_step = 000484, loss = 0.001060
grad_step = 000485, loss = 0.001049
grad_step = 000486, loss = 0.001052
grad_step = 000487, loss = 0.001061
grad_step = 000488, loss = 0.001062
grad_step = 000489, loss = 0.001055
grad_step = 000490, loss = 0.001044
grad_step = 000491, loss = 0.001038
grad_step = 000492, loss = 0.001040
grad_step = 000493, loss = 0.001043
grad_step = 000494, loss = 0.001044
grad_step = 000495, loss = 0.001038
grad_step = 000496, loss = 0.001028
grad_step = 000497, loss = 0.001019
grad_step = 000498, loss = 0.001014
grad_step = 000499, loss = 0.001014
grad_step = 000500, loss = 0.001016
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001017
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

  date_run                              2020-05-12 04:13:50.371445
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.258801
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 04:13:50.377415
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.177218
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 04:13:50.385021
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138236
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 04:13:50.391288
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.6929
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
0   2020-05-12 04:13:17.453308  ...    mean_absolute_error
1   2020-05-12 04:13:17.457394  ...     mean_squared_error
2   2020-05-12 04:13:17.461004  ...  median_absolute_error
3   2020-05-12 04:13:17.464698  ...               r2_score
4   2020-05-12 04:13:26.130798  ...    mean_absolute_error
5   2020-05-12 04:13:26.135548  ...     mean_squared_error
6   2020-05-12 04:13:26.139216  ...  median_absolute_error
7   2020-05-12 04:13:26.143078  ...               r2_score
8   2020-05-12 04:13:50.371445  ...    mean_absolute_error
9   2020-05-12 04:13:50.377415  ...     mean_squared_error
10  2020-05-12 04:13:50.385021  ...  median_absolute_error
11  2020-05-12 04:13:50.391288  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 149670.96it/s] 77%|███████▋  | 7618560/9912422 [00:00<00:10, 213634.58it/s]9920512it [00:00, 43207140.78it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1304767.16it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163381.68it/s]1654784it [00:00, 11601093.06it/s]                         
0it [00:00, ?it/s]8192it [00:00, 211006.95it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03c623bfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0363957ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03c61c6ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03607520f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03c623bfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0378bc0e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03c623bfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f036d072748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f03c6203ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0378bc0e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0363957fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8b0f567208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=982d4035470919e0863e3221bb42b7d122ec21dccb42d6d3360d512e4e9c6a42
  Stored in directory: /tmp/pip-ephem-wheel-cache-_b4bf4q2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8aa824c240> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2424832/17464789 [===>..........................] - ETA: 0s
 8527872/17464789 [=============>................] - ETA: 0s
14819328/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 04:15:17.433482: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 04:15:17.437707: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 04:15:17.437847: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5630ed9033b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 04:15:17.437862: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8353 - accuracy: 0.4890
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7701 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7484 - accuracy: 0.4947
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7510 - accuracy: 0.4945
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 4s - loss: 7.6569 - accuracy: 0.5006
12000/25000 [=============>................] - ETA: 4s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6612 - accuracy: 0.5004
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6494 - accuracy: 0.5011
25000/25000 [==============================] - 10s 388us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 04:15:34.535905
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 04:15:34.535905  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 04:15:41.039851: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 04:15:41.045618: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 04:15:41.046167: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559a0f224040 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 04:15:41.046322: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f8f60881358> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4578 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.4896 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8f5eb7f5c0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6973 - accuracy: 0.4980
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6425 - accuracy: 0.5016
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6496 - accuracy: 0.5011
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 4s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6040 - accuracy: 0.5041
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5864 - accuracy: 0.5052
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
15000/25000 [=================>............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6168 - accuracy: 0.5033
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6161 - accuracy: 0.5033
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6172 - accuracy: 0.5032
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6421 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6781 - accuracy: 0.4992
25000/25000 [==============================] - 10s 399us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f8ef58d1e80> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:46:36, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:46:32, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:23:48, 23.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:17:09, 32.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:05:14, 46.9kB/s].vector_cache/glove.6B.zip:   1%|          | 7.86M/862M [00:01<3:32:42, 66.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:28:06, 95.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.5M/862M [00:01<1:43:21, 136kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:12:02, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:01<50:18, 277kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<35:05, 395kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:01<24:33, 562kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:02<17:11, 799kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:02<12:04, 1.13MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<08:29, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:02<06:01, 2.24MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<06:05, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:03<04:21, 3.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<14:14, 942kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<12:14, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<09:08, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<08:40, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<07:42, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:44, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:43, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<05:52, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<04:24, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:09<03:15, 4.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<28:03, 471kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<20:58, 630kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<14:56, 882kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<13:33, 970kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<10:36, 1.24MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<07:46, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<53:18, 245kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<38:24, 340kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<27:09, 480kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<22:04, 589kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<18:06, 718kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<13:20, 974kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:17<09:26, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<56:16, 230kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<40:41, 318kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<28:45, 449kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<23:05, 558kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<18:47, 685kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<13:42, 938kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<09:42, 1.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<17:26, 734kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<13:31, 946kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:22<09:46, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<09:49, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<09:29, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<07:17, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:12, 2.43MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:17:27, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<8:37:14, 24.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<6:01:39, 34.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<4:15:23, 49.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<2:59:57, 70.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<2:06:01, 99.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:30:56, 138kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<1:04:53, 193kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<45:39, 274kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<34:48, 358kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<26:55, 463kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<19:28, 639kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<15:35, 795kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:09, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<08:48, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:02, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:49, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:43, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<04:49, 2.54MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<25:18, 484kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<18:58, 646kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<13:34, 901kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:19, 988kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:14, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<08:29, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<06:04, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:30:36, 134kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:04:37, 187kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<45:27, 266kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<34:33, 349kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<25:23, 474kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<18:00, 667kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:24, 777kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<12:00, 997kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:39, 1.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:53, 1.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:26, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:38, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:03, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:26, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:56, 2.99MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:02, 978kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:37, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<07:01, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:38, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:32, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<04:51, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:40, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:10, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:50, 3.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:12, 1.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:32, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:10, 2.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:36, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:10, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:49, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:31, 3.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<07:28, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<06:25, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:46, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:00, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:02, 2.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:29, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:59, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:46, 2.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:17, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:59, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:45, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:07, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:45, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<03:34, 3.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:03, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:48, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:35, 2.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:19, 3.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<11:51, 930kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:24, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<06:51, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:21, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:22, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:40, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:04, 2.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:44, 927kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:20, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<06:48, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:17, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:19, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:35, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:02, 2.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:35, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:06, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<05:12, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:08, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:22, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:01, 2.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:19, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:35, 2.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:18, 3.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<17:31, 601kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<13:22, 788kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<09:36, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:09, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:37, 1.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:30, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<04:39, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:56, 950kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:42, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<06:21, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:51, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:50, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:20, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:26, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:55, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:39, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:54, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:29, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:24, 2.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:45, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:22, 2.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<05:21, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:11, 2.39MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:07, 3.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:13, 1.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:41, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:31, 2.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:47, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:21, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:15, 3.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:37, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:06, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:08, 3.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<02:18, 4.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<38:33, 254kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<29:00, 337kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<20:42, 472kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<14:34, 668kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<15:13, 638kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<11:38, 834kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<08:22, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:06, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:22, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:39, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:04, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:12, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:04, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:29, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:21, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:31, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<03:42, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<02:43, 3.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:40, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:39, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:13, 2.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<06:35, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:27, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:04, 3.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:15, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:44, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:57, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:34, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<1:20:52, 114kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<57:30, 161kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<40:23, 228kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<30:19, 302kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<22:09, 414kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<15:41, 582kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<13:06, 695kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<10:05, 901kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<07:17, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:19, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:30, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:57, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:15, 1.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:58, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:27, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<03:12, 2.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:08:20, 130kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<48:43, 182kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<34:14, 259kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<25:57, 340kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<19:02, 463kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<13:27, 653kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:16<09:30, 921kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<1:50:50, 79.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<1:18:25, 112kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<54:55, 159kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<40:21, 215kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<29:08, 298kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<20:33, 421kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<16:21, 527kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<12:19, 699kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<08:48, 976kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:10, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:35, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:49, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:22, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:37, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:26, 2.45MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:23, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:56, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:57, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:02, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:40, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:46, 2.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:54, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:33, 2.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:41, 3.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:49, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:20, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:24, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:29, 3.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:54, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:57, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:38, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:26, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:45, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:54, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:32, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:39, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:42, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<02:33, 3.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:39, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<02:32, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:37, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:20, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:31, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:17, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:29, 3.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:33, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:03, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:10, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:18, 3.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<07:12, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:49, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:15, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:43, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:50, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:45, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:49, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:27, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:35, 2.86MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:59, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:09, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:16, 3.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<12:00, 610kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<09:08, 800kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<06:34, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:16, 1.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:45, 1.92MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:19, 1.66MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:28, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:27, 2.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:30, 2.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<04:50, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:07, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:03, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:47, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:05, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:13, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:22, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:19, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:14, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:57, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:12, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:36, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:52, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:05, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:52, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:09, 3.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:03, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:47, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<02:01, 3.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<49:14, 135kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<35:07, 189kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<24:39, 268kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<18:42, 351kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<14:22, 457kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<10:22, 631kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<08:15, 787kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:26, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<04:39, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:45, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:58, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:54, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:32, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:07, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:20, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:07, 2.02MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:49, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:07, 2.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:36, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<01:56, 3.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<01:26, 4.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<22:52, 269kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<16:37, 370kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<11:43, 522kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<09:35, 634kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<07:57, 764kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:50, 1.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<04:07, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<09:33, 630kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:18, 823kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<05:14, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:02, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:44, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:27, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:01, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:15, 2.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:56, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:58, 2.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:44, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:05, 1.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:23, 2.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:44, 3.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:58, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:22, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:28, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:42, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:00, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:43, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:01, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:23, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:21, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:46, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:30, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:52, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:15, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<01:38, 3.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:06, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:01, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:04, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<22:45, 231kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<17:00, 309kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:58<12:06, 433kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<08:28, 614kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<09:39, 538kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<07:17, 712kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<05:12, 991kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:49, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:26, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:19, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:24, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:17, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:49, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:04, 2.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:37, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:45, 2.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:23, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:39, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:05, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:29, 3.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<21:27, 226kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<15:30, 312kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<10:54, 441kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<08:42, 549kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<06:34, 727kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:41, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:22, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:32, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:33, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:53, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:29, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:51, 2.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:22, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:35, 2.85MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:10, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:58, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:29, 3.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:05, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:21, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:20, 3.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<32:49, 133kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<23:23, 187kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<16:23, 265kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<12:35, 341kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<08:43, 486kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<12:34, 337kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<09:15, 457kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:32, 642kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<05:26, 765kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<04:38, 895kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:25, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:27, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:59, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:30, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:50, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:13, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:58, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:28, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:56, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:09, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:40, 2.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:12, 3.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:25, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:47, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:02, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:19, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:23, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:53, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:42, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:17, 2.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:45, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:11, 3.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:41, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:29, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:08, 3.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<00:50, 4.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<26:57, 132kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<19:34, 181kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<13:49, 255kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<10:06, 344kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<07:47, 446kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<05:35, 620kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<03:54, 876kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:02, 677kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:51, 880kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<02:46, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:34, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:23, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:28, 943kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:45, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:59, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:07, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:08, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:38, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<01:09, 2.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:32, 886kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:47, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:01, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:06, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:05, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:36, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:35, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:04, 2.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:24, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:36, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:16, 2.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<00:54, 3.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<21:02, 136kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<15:00, 190kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<10:28, 270kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<07:53, 354kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<06:05, 458kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:23, 633kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<03:02, 895kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<2:39:08, 17.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<1:51:24, 24.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:17:18, 34.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<54:00, 49.1kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<37:59, 69.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<26:23, 99.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<18:50, 137kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<13:25, 192kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<09:21, 272kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<07:03, 356kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<05:10, 484kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:39, 680kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:06, 789kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:24, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:43, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:45, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:28, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:03, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<00:45, 3.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<23:27, 98.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<16:52, 137kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<11:51, 193kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<08:28, 264kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<06:08, 364kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<04:17, 514kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:28, 626kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:51, 757kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:06, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<01:27, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<2:04:07, 17.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<1:26:50, 24.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<1:00:05, 34.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<41:48, 48.7kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<29:22, 69.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<20:20, 98.5kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<14:27, 136kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<10:16, 191kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<07:09, 270kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<05:21, 354kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<03:56, 480kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:45, 677kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:19, 785kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:48, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:17, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:18, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:16, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:57, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:41, 2.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:09, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:58, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:43, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:51, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:56, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:43, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:30, 3.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:48, 553kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:05, 740kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:29, 1.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<01:01, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:31, 421kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<02:36, 567kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<01:49, 793kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:34, 895kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:23, 1.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:02, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:42, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<10:14, 132kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<07:16, 184kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<05:01, 262kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<03:43, 344kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:43, 468kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:53, 657kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:34, 768kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:20, 897kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:59, 1.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<00:40, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<1:06:23, 17.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<46:19, 24.5kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<31:43, 35.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<21:43, 49.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<15:22, 69.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<10:41, 98.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<07:20, 141kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<05:15, 191kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<03:45, 265kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<02:35, 376kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:57, 477kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<01:26, 644kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:00, 897kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:41, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<05:39, 153kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<04:07, 210kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<02:53, 295kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<01:54, 420kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<03:28, 230kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<02:29, 318kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:42, 451kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:18, 559kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:03, 687kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:45, 935kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:30, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<38:34, 17.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<26:48, 24.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<18:02, 34.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<12:02, 49.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<08:23, 69.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<05:37, 99.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<03:48, 138kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<02:45, 189kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:54, 267kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<01:14, 379kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:06, 408kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:49, 550kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:32, 773kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:26, 875kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:22, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:10, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:18, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.18MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 863/400000 [00:00<00:46, 8626.22it/s]  0%|          | 1644/400000 [00:00<00:47, 8361.38it/s]  1%|          | 2472/400000 [00:00<00:47, 8334.39it/s]  1%|          | 3235/400000 [00:00<00:48, 8106.66it/s]  1%|          | 4003/400000 [00:00<00:49, 7973.13it/s]  1%|          | 4832/400000 [00:00<00:49, 8063.52it/s]  1%|▏         | 5604/400000 [00:00<00:49, 7955.90it/s]  2%|▏         | 6460/400000 [00:00<00:48, 8125.63it/s]  2%|▏         | 7308/400000 [00:00<00:47, 8227.13it/s]  2%|▏         | 8172/400000 [00:01<00:46, 8339.54it/s]  2%|▏         | 8984/400000 [00:01<00:47, 8262.25it/s]  2%|▏         | 9795/400000 [00:01<00:49, 7942.55it/s]  3%|▎         | 10609/400000 [00:01<00:48, 8000.08it/s]  3%|▎         | 11418/400000 [00:01<00:48, 8025.37it/s]  3%|▎         | 12217/400000 [00:01<00:48, 7991.13it/s]  3%|▎         | 13014/400000 [00:01<00:48, 7937.44it/s]  3%|▎         | 13806/400000 [00:01<00:48, 7921.42it/s]  4%|▎         | 14629/400000 [00:01<00:48, 8009.87it/s]  4%|▍         | 15430/400000 [00:01<00:48, 8005.91it/s]  4%|▍         | 16236/400000 [00:02<00:47, 8018.81it/s]  4%|▍         | 17080/400000 [00:02<00:47, 8138.39it/s]  4%|▍         | 17895/400000 [00:02<00:47, 8053.60it/s]  5%|▍         | 18701/400000 [00:02<00:47, 7962.23it/s]  5%|▍         | 19498/400000 [00:02<00:47, 7927.49it/s]  5%|▌         | 20300/400000 [00:02<00:47, 7952.37it/s]  5%|▌         | 21136/400000 [00:02<00:46, 8067.78it/s]  5%|▌         | 21944/400000 [00:02<00:46, 8068.53it/s]  6%|▌         | 22754/400000 [00:02<00:46, 8075.96it/s]  6%|▌         | 23630/400000 [00:02<00:45, 8267.62it/s]  6%|▌         | 24538/400000 [00:03<00:44, 8494.93it/s]  6%|▋         | 25390/400000 [00:03<00:44, 8497.28it/s]  7%|▋         | 26242/400000 [00:03<00:44, 8307.23it/s]  7%|▋         | 27084/400000 [00:03<00:44, 8340.28it/s]  7%|▋         | 27920/400000 [00:03<00:44, 8345.73it/s]  7%|▋         | 28756/400000 [00:03<00:45, 8194.14it/s]  7%|▋         | 29581/400000 [00:03<00:45, 8209.45it/s]  8%|▊         | 30403/400000 [00:03<00:45, 8096.86it/s]  8%|▊         | 31216/400000 [00:03<00:45, 8105.03it/s]  8%|▊         | 32050/400000 [00:03<00:45, 8172.93it/s]  8%|▊         | 32868/400000 [00:04<00:45, 8130.55it/s]  8%|▊         | 33692/400000 [00:04<00:44, 8162.68it/s]  9%|▊         | 34509/400000 [00:04<00:45, 8007.52it/s]  9%|▉         | 35311/400000 [00:04<00:45, 8006.42it/s]  9%|▉         | 36170/400000 [00:04<00:44, 8169.86it/s]  9%|▉         | 36990/400000 [00:04<00:44, 8178.42it/s]  9%|▉         | 37849/400000 [00:04<00:43, 8296.97it/s] 10%|▉         | 38680/400000 [00:04<00:43, 8214.46it/s] 10%|▉         | 39529/400000 [00:04<00:43, 8293.98it/s] 10%|█         | 40369/400000 [00:04<00:43, 8325.47it/s] 10%|█         | 41234/400000 [00:05<00:42, 8418.91it/s] 11%|█         | 42135/400000 [00:05<00:41, 8586.76it/s] 11%|█         | 42995/400000 [00:05<00:42, 8320.12it/s] 11%|█         | 43830/400000 [00:05<00:44, 8000.77it/s] 11%|█         | 44635/400000 [00:05<00:44, 7956.58it/s] 11%|█▏        | 45434/400000 [00:05<00:44, 7952.54it/s] 12%|█▏        | 46254/400000 [00:05<00:44, 8022.30it/s] 12%|█▏        | 47089/400000 [00:05<00:43, 8115.30it/s] 12%|█▏        | 47958/400000 [00:05<00:42, 8279.06it/s] 12%|█▏        | 48863/400000 [00:05<00:41, 8492.64it/s] 12%|█▏        | 49729/400000 [00:06<00:41, 8540.91it/s] 13%|█▎        | 50586/400000 [00:06<00:41, 8518.64it/s] 13%|█▎        | 51440/400000 [00:06<00:41, 8451.10it/s] 13%|█▎        | 52287/400000 [00:06<00:41, 8447.27it/s] 13%|█▎        | 53133/400000 [00:06<00:41, 8331.90it/s] 13%|█▎        | 53968/400000 [00:06<00:42, 8233.48it/s] 14%|█▎        | 54795/400000 [00:06<00:41, 8242.38it/s] 14%|█▍        | 55620/400000 [00:06<00:42, 8080.59it/s] 14%|█▍        | 56444/400000 [00:06<00:42, 8127.37it/s] 14%|█▍        | 57281/400000 [00:07<00:41, 8198.00it/s] 15%|█▍        | 58105/400000 [00:07<00:41, 8209.14it/s] 15%|█▍        | 58927/400000 [00:07<00:41, 8151.56it/s] 15%|█▍        | 59743/400000 [00:07<00:41, 8151.11it/s] 15%|█▌        | 60567/400000 [00:07<00:41, 8176.48it/s] 15%|█▌        | 61404/400000 [00:07<00:41, 8231.82it/s] 16%|█▌        | 62228/400000 [00:07<00:41, 8160.33it/s] 16%|█▌        | 63045/400000 [00:07<00:42, 7950.03it/s] 16%|█▌        | 63842/400000 [00:07<00:42, 7818.94it/s] 16%|█▌        | 64626/400000 [00:07<00:43, 7685.34it/s] 16%|█▋        | 65473/400000 [00:08<00:42, 7902.97it/s] 17%|█▋        | 66308/400000 [00:08<00:41, 8031.70it/s] 17%|█▋        | 67114/400000 [00:08<00:41, 8027.14it/s] 17%|█▋        | 67926/400000 [00:08<00:41, 8054.32it/s] 17%|█▋        | 68753/400000 [00:08<00:40, 8117.43it/s] 17%|█▋        | 69576/400000 [00:08<00:40, 8147.35it/s] 18%|█▊        | 70465/400000 [00:08<00:39, 8355.64it/s] 18%|█▊        | 71327/400000 [00:08<00:38, 8430.98it/s] 18%|█▊        | 72172/400000 [00:08<00:39, 8314.52it/s] 18%|█▊        | 73005/400000 [00:08<00:39, 8314.84it/s] 18%|█▊        | 73838/400000 [00:09<00:39, 8254.15it/s] 19%|█▊        | 74665/400000 [00:09<00:39, 8150.67it/s] 19%|█▉        | 75508/400000 [00:09<00:39, 8230.51it/s] 19%|█▉        | 76332/400000 [00:09<00:40, 8064.02it/s] 19%|█▉        | 77148/400000 [00:09<00:39, 8091.16it/s] 19%|█▉        | 77991/400000 [00:09<00:39, 8187.78it/s] 20%|█▉        | 78811/400000 [00:09<00:39, 8092.70it/s] 20%|█▉        | 79622/400000 [00:09<00:40, 7970.12it/s] 20%|██        | 80421/400000 [00:09<00:40, 7940.65it/s] 20%|██        | 81216/400000 [00:09<00:40, 7858.70it/s] 21%|██        | 82026/400000 [00:10<00:40, 7927.74it/s] 21%|██        | 82868/400000 [00:10<00:39, 8067.97it/s] 21%|██        | 83691/400000 [00:10<00:38, 8114.15it/s] 21%|██        | 84504/400000 [00:10<00:39, 8058.77it/s] 21%|██▏       | 85313/400000 [00:10<00:39, 8064.95it/s] 22%|██▏       | 86120/400000 [00:10<00:39, 7993.31it/s] 22%|██▏       | 86936/400000 [00:10<00:38, 8041.83it/s] 22%|██▏       | 87777/400000 [00:10<00:38, 8148.19it/s] 22%|██▏       | 88593/400000 [00:10<00:38, 8097.23it/s] 22%|██▏       | 89411/400000 [00:10<00:38, 8121.75it/s] 23%|██▎       | 90227/400000 [00:11<00:38, 8131.02it/s] 23%|██▎       | 91097/400000 [00:11<00:37, 8292.96it/s] 23%|██▎       | 91928/400000 [00:11<00:37, 8265.68it/s] 23%|██▎       | 92756/400000 [00:11<00:37, 8107.22it/s] 23%|██▎       | 93586/400000 [00:11<00:37, 8162.40it/s] 24%|██▎       | 94404/400000 [00:11<00:37, 8070.38it/s] 24%|██▍       | 95248/400000 [00:11<00:37, 8176.87it/s] 24%|██▍       | 96088/400000 [00:11<00:36, 8240.95it/s] 24%|██▍       | 96935/400000 [00:11<00:36, 8308.20it/s] 24%|██▍       | 97767/400000 [00:11<00:36, 8187.26it/s] 25%|██▍       | 98607/400000 [00:12<00:36, 8247.32it/s] 25%|██▍       | 99504/400000 [00:12<00:35, 8449.75it/s] 25%|██▌       | 100351/400000 [00:12<00:36, 8256.59it/s] 25%|██▌       | 101179/400000 [00:12<00:36, 8177.66it/s] 25%|██▌       | 101999/400000 [00:12<00:36, 8171.29it/s] 26%|██▌       | 102821/400000 [00:12<00:36, 8185.21it/s] 26%|██▌       | 103641/400000 [00:12<00:36, 8107.12it/s] 26%|██▌       | 104467/400000 [00:12<00:36, 8150.50it/s] 26%|██▋       | 105283/400000 [00:12<00:37, 7940.41it/s] 27%|██▋       | 106086/400000 [00:13<00:36, 7965.56it/s] 27%|██▋       | 106884/400000 [00:13<00:37, 7914.85it/s] 27%|██▋       | 107720/400000 [00:13<00:36, 8043.15it/s] 27%|██▋       | 108608/400000 [00:13<00:35, 8275.39it/s] 27%|██▋       | 109438/400000 [00:13<00:35, 8182.72it/s] 28%|██▊       | 110259/400000 [00:13<00:35, 8178.11it/s] 28%|██▊       | 111079/400000 [00:13<00:35, 8138.84it/s] 28%|██▊       | 111942/400000 [00:13<00:34, 8279.44it/s] 28%|██▊       | 112821/400000 [00:13<00:34, 8425.48it/s] 28%|██▊       | 113666/400000 [00:13<00:34, 8297.44it/s] 29%|██▊       | 114498/400000 [00:14<00:34, 8250.37it/s] 29%|██▉       | 115330/400000 [00:14<00:34, 8270.80it/s] 29%|██▉       | 116224/400000 [00:14<00:33, 8457.77it/s] 29%|██▉       | 117072/400000 [00:14<00:33, 8391.92it/s] 29%|██▉       | 117913/400000 [00:14<00:34, 8151.98it/s] 30%|██▉       | 118766/400000 [00:14<00:34, 8261.23it/s] 30%|██▉       | 119612/400000 [00:14<00:33, 8318.07it/s] 30%|███       | 120461/400000 [00:14<00:33, 8368.18it/s] 30%|███       | 121299/400000 [00:14<00:34, 8122.19it/s] 31%|███       | 122114/400000 [00:14<00:34, 8044.68it/s] 31%|███       | 122933/400000 [00:15<00:34, 8087.17it/s] 31%|███       | 123757/400000 [00:15<00:33, 8129.85it/s] 31%|███       | 124571/400000 [00:15<00:34, 8091.29it/s] 31%|███▏      | 125395/400000 [00:15<00:33, 8133.68it/s] 32%|███▏      | 126248/400000 [00:15<00:33, 8246.70it/s] 32%|███▏      | 127146/400000 [00:15<00:32, 8451.95it/s] 32%|███▏      | 127995/400000 [00:15<00:32, 8460.36it/s] 32%|███▏      | 128843/400000 [00:15<00:32, 8325.15it/s] 32%|███▏      | 129677/400000 [00:15<00:34, 7882.10it/s] 33%|███▎      | 130509/400000 [00:15<00:33, 8003.94it/s] 33%|███▎      | 131314/400000 [00:16<00:33, 7963.78it/s] 33%|███▎      | 132152/400000 [00:16<00:33, 8083.67it/s] 33%|███▎      | 132964/400000 [00:16<00:33, 8054.99it/s] 33%|███▎      | 133786/400000 [00:16<00:32, 8097.98it/s] 34%|███▎      | 134598/400000 [00:16<00:33, 8020.53it/s] 34%|███▍      | 135402/400000 [00:16<00:33, 7854.03it/s] 34%|███▍      | 136226/400000 [00:16<00:33, 7963.67it/s] 34%|███▍      | 137024/400000 [00:16<00:33, 7954.74it/s] 34%|███▍      | 137821/400000 [00:16<00:34, 7701.82it/s] 35%|███▍      | 138594/400000 [00:17<00:34, 7671.54it/s] 35%|███▍      | 139380/400000 [00:17<00:33, 7721.74it/s] 35%|███▌      | 140177/400000 [00:17<00:33, 7793.09it/s] 35%|███▌      | 140958/400000 [00:17<00:34, 7483.89it/s] 35%|███▌      | 141724/400000 [00:17<00:34, 7535.12it/s] 36%|███▌      | 142534/400000 [00:17<00:33, 7694.67it/s] 36%|███▌      | 143388/400000 [00:17<00:32, 7929.53it/s] 36%|███▌      | 144192/400000 [00:17<00:32, 7961.96it/s] 36%|███▌      | 144991/400000 [00:17<00:32, 7934.36it/s] 36%|███▋      | 145869/400000 [00:17<00:31, 8169.33it/s] 37%|███▋      | 146689/400000 [00:18<00:31, 8127.61it/s] 37%|███▋      | 147504/400000 [00:18<00:31, 8016.69it/s] 37%|███▋      | 148308/400000 [00:18<00:32, 7844.29it/s] 37%|███▋      | 149123/400000 [00:18<00:31, 7932.58it/s] 37%|███▋      | 149920/400000 [00:18<00:31, 7942.63it/s] 38%|███▊      | 150716/400000 [00:18<00:31, 7910.02it/s] 38%|███▊      | 151567/400000 [00:18<00:30, 8079.26it/s] 38%|███▊      | 152414/400000 [00:18<00:30, 8191.07it/s] 38%|███▊      | 153240/400000 [00:18<00:30, 8209.19it/s] 39%|███▊      | 154079/400000 [00:18<00:29, 8261.24it/s] 39%|███▊      | 154919/400000 [00:19<00:29, 8302.12it/s] 39%|███▉      | 155789/400000 [00:19<00:29, 8416.72it/s] 39%|███▉      | 156658/400000 [00:19<00:28, 8492.61it/s] 39%|███▉      | 157541/400000 [00:19<00:28, 8588.53it/s] 40%|███▉      | 158401/400000 [00:19<00:28, 8469.06it/s] 40%|███▉      | 159249/400000 [00:19<00:28, 8341.57it/s] 40%|████      | 160085/400000 [00:19<00:29, 8174.03it/s] 40%|████      | 160946/400000 [00:19<00:28, 8299.23it/s] 40%|████      | 161779/400000 [00:19<00:28, 8306.31it/s] 41%|████      | 162611/400000 [00:19<00:29, 8173.36it/s] 41%|████      | 163430/400000 [00:20<00:29, 8123.13it/s] 41%|████      | 164255/400000 [00:20<00:28, 8160.19it/s] 41%|████▏     | 165105/400000 [00:20<00:28, 8257.72it/s] 41%|████▏     | 165991/400000 [00:20<00:27, 8429.57it/s] 42%|████▏     | 166839/400000 [00:20<00:27, 8443.12it/s] 42%|████▏     | 167685/400000 [00:20<00:27, 8433.55it/s] 42%|████▏     | 168579/400000 [00:20<00:26, 8577.48it/s] 42%|████▏     | 169468/400000 [00:20<00:26, 8666.39it/s] 43%|████▎     | 170336/400000 [00:20<00:26, 8617.96it/s] 43%|████▎     | 171199/400000 [00:20<00:27, 8362.33it/s] 43%|████▎     | 172038/400000 [00:21<00:27, 8282.50it/s] 43%|████▎     | 172939/400000 [00:21<00:26, 8486.07it/s] 43%|████▎     | 173791/400000 [00:21<00:26, 8449.50it/s] 44%|████▎     | 174641/400000 [00:21<00:26, 8461.71it/s] 44%|████▍     | 175515/400000 [00:21<00:26, 8541.78it/s] 44%|████▍     | 176371/400000 [00:21<00:26, 8345.95it/s] 44%|████▍     | 177213/400000 [00:21<00:26, 8366.45it/s] 45%|████▍     | 178051/400000 [00:21<00:26, 8241.02it/s] 45%|████▍     | 178877/400000 [00:21<00:26, 8234.16it/s] 45%|████▍     | 179702/400000 [00:22<00:27, 8101.52it/s] 45%|████▌     | 180514/400000 [00:22<00:27, 7892.33it/s] 45%|████▌     | 181306/400000 [00:22<00:27, 7862.56it/s] 46%|████▌     | 182143/400000 [00:22<00:27, 8007.80it/s] 46%|████▌     | 182946/400000 [00:22<00:27, 7979.90it/s] 46%|████▌     | 183746/400000 [00:22<00:27, 7872.61it/s] 46%|████▌     | 184570/400000 [00:22<00:27, 7978.41it/s] 46%|████▋     | 185431/400000 [00:22<00:26, 8157.91it/s] 47%|████▋     | 186275/400000 [00:22<00:25, 8240.20it/s] 47%|████▋     | 187117/400000 [00:22<00:25, 8291.19it/s] 47%|████▋     | 187948/400000 [00:23<00:26, 8038.46it/s] 47%|████▋     | 188813/400000 [00:23<00:25, 8211.57it/s] 47%|████▋     | 189700/400000 [00:23<00:25, 8396.46it/s] 48%|████▊     | 190572/400000 [00:23<00:24, 8489.45it/s] 48%|████▊     | 191433/400000 [00:23<00:24, 8525.14it/s] 48%|████▊     | 192288/400000 [00:23<00:25, 8097.25it/s] 48%|████▊     | 193104/400000 [00:23<00:25, 8042.95it/s] 48%|████▊     | 193935/400000 [00:23<00:25, 8120.00it/s] 49%|████▊     | 194807/400000 [00:23<00:24, 8288.83it/s] 49%|████▉     | 195706/400000 [00:23<00:24, 8486.69it/s] 49%|████▉     | 196558/400000 [00:24<00:24, 8378.94it/s] 49%|████▉     | 197401/400000 [00:24<00:24, 8393.36it/s] 50%|████▉     | 198243/400000 [00:24<00:24, 8314.11it/s] 50%|████▉     | 199100/400000 [00:24<00:23, 8389.12it/s] 50%|████▉     | 199944/400000 [00:24<00:23, 8402.80it/s] 50%|█████     | 200786/400000 [00:24<00:24, 8150.72it/s] 50%|█████     | 201604/400000 [00:24<00:24, 8023.38it/s] 51%|█████     | 202409/400000 [00:24<00:25, 7854.76it/s] 51%|█████     | 203233/400000 [00:24<00:24, 7964.52it/s] 51%|█████     | 204104/400000 [00:24<00:23, 8170.95it/s] 51%|█████     | 204924/400000 [00:25<00:25, 7758.57it/s] 51%|█████▏    | 205746/400000 [00:25<00:24, 7887.67it/s] 52%|█████▏    | 206564/400000 [00:25<00:24, 7968.24it/s] 52%|█████▏    | 207424/400000 [00:25<00:23, 8146.83it/s] 52%|█████▏    | 208243/400000 [00:25<00:23, 8156.23it/s] 52%|█████▏    | 209062/400000 [00:25<00:23, 7972.42it/s] 52%|█████▏    | 209935/400000 [00:25<00:23, 8183.36it/s] 53%|█████▎    | 210757/400000 [00:25<00:23, 8172.98it/s] 53%|█████▎    | 211605/400000 [00:25<00:22, 8260.20it/s] 53%|█████▎    | 212433/400000 [00:26<00:22, 8263.17it/s] 53%|█████▎    | 213261/400000 [00:26<00:23, 7942.21it/s] 54%|█████▎    | 214069/400000 [00:26<00:23, 7979.94it/s] 54%|█████▎    | 214870/400000 [00:26<00:23, 7973.35it/s] 54%|█████▍    | 215670/400000 [00:26<00:23, 7919.48it/s] 54%|█████▍    | 216512/400000 [00:26<00:22, 8060.65it/s] 54%|█████▍    | 217320/400000 [00:26<00:22, 7963.37it/s] 55%|█████▍    | 218192/400000 [00:26<00:22, 8174.85it/s] 55%|█████▍    | 219012/400000 [00:26<00:22, 8129.87it/s] 55%|█████▍    | 219845/400000 [00:26<00:22, 8188.07it/s] 55%|█████▌    | 220732/400000 [00:27<00:21, 8379.76it/s] 55%|█████▌    | 221572/400000 [00:27<00:21, 8365.16it/s] 56%|█████▌    | 222469/400000 [00:27<00:20, 8537.48it/s] 56%|█████▌    | 223342/400000 [00:27<00:20, 8594.15it/s] 56%|█████▌    | 224203/400000 [00:27<00:20, 8593.02it/s] 56%|█████▋    | 225073/400000 [00:27<00:20, 8622.94it/s] 56%|█████▋    | 225936/400000 [00:27<00:20, 8409.80it/s] 57%|█████▋    | 226779/400000 [00:27<00:20, 8281.47it/s] 57%|█████▋    | 227609/400000 [00:27<00:21, 8167.74it/s] 57%|█████▋    | 228428/400000 [00:27<00:21, 8004.29it/s] 57%|█████▋    | 229254/400000 [00:28<00:21, 8077.74it/s] 58%|█████▊    | 230064/400000 [00:28<00:21, 8054.73it/s] 58%|█████▊    | 230947/400000 [00:28<00:20, 8270.30it/s] 58%|█████▊    | 231784/400000 [00:28<00:20, 8278.07it/s] 58%|█████▊    | 232627/400000 [00:28<00:20, 8321.70it/s] 58%|█████▊    | 233461/400000 [00:28<00:20, 8306.47it/s] 59%|█████▊    | 234293/400000 [00:28<00:20, 8254.87it/s] 59%|█████▉    | 235173/400000 [00:28<00:19, 8410.56it/s] 59%|█████▉    | 236016/400000 [00:28<00:19, 8223.46it/s] 59%|█████▉    | 236841/400000 [00:28<00:20, 8071.64it/s] 59%|█████▉    | 237650/400000 [00:29<00:20, 8006.77it/s] 60%|█████▉    | 238453/400000 [00:29<00:20, 7837.37it/s] 60%|█████▉    | 239308/400000 [00:29<00:19, 8038.34it/s] 60%|██████    | 240158/400000 [00:29<00:19, 8170.35it/s] 60%|██████    | 240978/400000 [00:29<00:19, 8132.85it/s] 60%|██████    | 241793/400000 [00:29<00:19, 8039.89it/s] 61%|██████    | 242599/400000 [00:29<00:19, 7997.81it/s] 61%|██████    | 243494/400000 [00:29<00:18, 8260.45it/s] 61%|██████    | 244326/400000 [00:29<00:18, 8275.21it/s] 61%|██████▏   | 245156/400000 [00:30<00:19, 8023.64it/s] 61%|██████▏   | 245962/400000 [00:30<00:19, 7830.77it/s] 62%|██████▏   | 246828/400000 [00:30<00:19, 8060.75it/s] 62%|██████▏   | 247699/400000 [00:30<00:18, 8244.90it/s] 62%|██████▏   | 248585/400000 [00:30<00:17, 8418.75it/s] 62%|██████▏   | 249431/400000 [00:30<00:17, 8412.41it/s] 63%|██████▎   | 250275/400000 [00:30<00:18, 8290.93it/s] 63%|██████▎   | 251107/400000 [00:30<00:18, 8073.41it/s] 63%|██████▎   | 251918/400000 [00:30<00:18, 7984.08it/s] 63%|██████▎   | 252719/400000 [00:30<00:18, 7943.84it/s] 63%|██████▎   | 253539/400000 [00:31<00:18, 8018.64it/s] 64%|██████▎   | 254343/400000 [00:31<00:18, 7981.52it/s] 64%|██████▍   | 255143/400000 [00:31<00:18, 7846.30it/s] 64%|██████▍   | 255932/400000 [00:31<00:18, 7857.05it/s] 64%|██████▍   | 256773/400000 [00:31<00:17, 8014.13it/s] 64%|██████▍   | 257576/400000 [00:31<00:18, 7866.24it/s] 65%|██████▍   | 258365/400000 [00:31<00:18, 7813.70it/s] 65%|██████▍   | 259247/400000 [00:31<00:17, 8088.27it/s] 65%|██████▌   | 260119/400000 [00:31<00:16, 8265.73it/s] 65%|██████▌   | 260978/400000 [00:31<00:16, 8360.39it/s] 65%|██████▌   | 261856/400000 [00:32<00:16, 8479.87it/s] 66%|██████▌   | 262707/400000 [00:32<00:16, 8449.84it/s] 66%|██████▌   | 263554/400000 [00:32<00:16, 8453.72it/s] 66%|██████▌   | 264407/400000 [00:32<00:15, 8476.02it/s] 66%|██████▋   | 265279/400000 [00:32<00:15, 8546.92it/s] 67%|██████▋   | 266181/400000 [00:32<00:15, 8681.20it/s] 67%|██████▋   | 267051/400000 [00:32<00:15, 8659.45it/s] 67%|██████▋   | 267918/400000 [00:32<00:15, 8436.25it/s] 67%|██████▋   | 268764/400000 [00:32<00:15, 8355.31it/s] 67%|██████▋   | 269601/400000 [00:32<00:15, 8319.99it/s] 68%|██████▊   | 270435/400000 [00:33<00:16, 8059.04it/s] 68%|██████▊   | 271294/400000 [00:33<00:15, 8210.80it/s] 68%|██████▊   | 272148/400000 [00:33<00:15, 8306.00it/s] 68%|██████▊   | 273032/400000 [00:33<00:15, 8457.06it/s] 68%|██████▊   | 273882/400000 [00:33<00:14, 8467.15it/s] 69%|██████▊   | 274731/400000 [00:33<00:14, 8382.81it/s] 69%|██████▉   | 275571/400000 [00:33<00:15, 8229.75it/s] 69%|██████▉   | 276396/400000 [00:33<00:15, 8159.97it/s] 69%|██████▉   | 277214/400000 [00:33<00:15, 8037.84it/s] 70%|██████▉   | 278055/400000 [00:34<00:14, 8144.36it/s] 70%|██████▉   | 278871/400000 [00:34<00:15, 7978.91it/s] 70%|██████▉   | 279671/400000 [00:34<00:15, 7980.03it/s] 70%|███████   | 280471/400000 [00:34<00:15, 7909.85it/s] 70%|███████   | 281297/400000 [00:34<00:14, 8011.11it/s] 71%|███████   | 282124/400000 [00:34<00:14, 8086.01it/s] 71%|███████   | 282950/400000 [00:34<00:14, 8131.85it/s] 71%|███████   | 283792/400000 [00:34<00:14, 8214.80it/s] 71%|███████   | 284618/400000 [00:34<00:14, 8225.15it/s] 71%|███████▏  | 285488/400000 [00:34<00:13, 8360.22it/s] 72%|███████▏  | 286325/400000 [00:35<00:13, 8357.15it/s] 72%|███████▏  | 287162/400000 [00:35<00:13, 8312.44it/s] 72%|███████▏  | 287994/400000 [00:35<00:13, 8160.53it/s] 72%|███████▏  | 288832/400000 [00:35<00:13, 8223.25it/s] 72%|███████▏  | 289676/400000 [00:35<00:13, 8286.45it/s] 73%|███████▎  | 290552/400000 [00:35<00:12, 8420.41it/s] 73%|███████▎  | 291461/400000 [00:35<00:12, 8610.56it/s] 73%|███████▎  | 292324/400000 [00:35<00:12, 8423.78it/s] 73%|███████▎  | 293203/400000 [00:35<00:12, 8528.45it/s] 74%|███████▎  | 294082/400000 [00:35<00:12, 8603.77it/s] 74%|███████▎  | 294975/400000 [00:36<00:12, 8698.37it/s] 74%|███████▍  | 295847/400000 [00:36<00:12, 8624.35it/s] 74%|███████▍  | 296711/400000 [00:36<00:12, 8554.82it/s] 74%|███████▍  | 297568/400000 [00:36<00:12, 8182.87it/s] 75%|███████▍  | 298391/400000 [00:36<00:12, 8167.60it/s] 75%|███████▍  | 299211/400000 [00:36<00:12, 8168.08it/s] 75%|███████▌  | 300054/400000 [00:36<00:12, 8243.07it/s] 75%|███████▌  | 300880/400000 [00:36<00:12, 8206.88it/s] 75%|███████▌  | 301702/400000 [00:36<00:12, 8101.24it/s] 76%|███████▌  | 302514/400000 [00:36<00:12, 8029.64it/s] 76%|███████▌  | 303318/400000 [00:37<00:12, 7913.73it/s] 76%|███████▌  | 304140/400000 [00:37<00:11, 8003.06it/s] 76%|███████▌  | 304962/400000 [00:37<00:11, 8064.23it/s] 76%|███████▋  | 305770/400000 [00:37<00:11, 7949.13it/s] 77%|███████▋  | 306587/400000 [00:37<00:11, 8013.04it/s] 77%|███████▋  | 307390/400000 [00:37<00:11, 7859.46it/s] 77%|███████▋  | 308179/400000 [00:37<00:11, 7865.47it/s] 77%|███████▋  | 308976/400000 [00:37<00:11, 7894.29it/s] 77%|███████▋  | 309782/400000 [00:37<00:11, 7943.10it/s] 78%|███████▊  | 310577/400000 [00:37<00:11, 7920.21it/s] 78%|███████▊  | 311370/400000 [00:38<00:11, 7887.90it/s] 78%|███████▊  | 312187/400000 [00:38<00:11, 7969.35it/s] 78%|███████▊  | 312985/400000 [00:38<00:11, 7767.54it/s] 78%|███████▊  | 313781/400000 [00:38<00:11, 7821.81it/s] 79%|███████▊  | 314644/400000 [00:38<00:10, 8046.27it/s] 79%|███████▉  | 315481/400000 [00:38<00:10, 8140.60it/s] 79%|███████▉  | 316361/400000 [00:38<00:10, 8324.43it/s] 79%|███████▉  | 317199/400000 [00:38<00:09, 8339.99it/s] 80%|███████▉  | 318050/400000 [00:38<00:09, 8388.26it/s] 80%|███████▉  | 318891/400000 [00:39<00:09, 8302.80it/s] 80%|███████▉  | 319728/400000 [00:39<00:09, 8322.37it/s] 80%|████████  | 320581/400000 [00:39<00:09, 8382.65it/s] 80%|████████  | 321481/400000 [00:39<00:09, 8556.28it/s] 81%|████████  | 322338/400000 [00:39<00:09, 8235.65it/s] 81%|████████  | 323166/400000 [00:39<00:09, 8175.62it/s] 81%|████████  | 323987/400000 [00:39<00:09, 8112.61it/s] 81%|████████  | 324801/400000 [00:39<00:09, 8107.02it/s] 81%|████████▏ | 325614/400000 [00:39<00:09, 8086.12it/s] 82%|████████▏ | 326461/400000 [00:39<00:08, 8196.48it/s] 82%|████████▏ | 327336/400000 [00:40<00:08, 8354.31it/s] 82%|████████▏ | 328173/400000 [00:40<00:08, 8300.92it/s] 82%|████████▏ | 329005/400000 [00:40<00:08, 8262.68it/s] 82%|████████▏ | 329833/400000 [00:40<00:08, 8205.48it/s] 83%|████████▎ | 330655/400000 [00:40<00:08, 8200.69it/s] 83%|████████▎ | 331486/400000 [00:40<00:08, 8231.88it/s] 83%|████████▎ | 332341/400000 [00:40<00:08, 8324.78it/s] 83%|████████▎ | 333178/400000 [00:40<00:08, 8336.29it/s] 84%|████████▎ | 334012/400000 [00:40<00:07, 8275.83it/s] 84%|████████▎ | 334882/400000 [00:40<00:07, 8397.36it/s] 84%|████████▍ | 335729/400000 [00:41<00:07, 8418.56it/s] 84%|████████▍ | 336584/400000 [00:41<00:07, 8455.40it/s] 84%|████████▍ | 337430/400000 [00:41<00:07, 8452.53it/s] 85%|████████▍ | 338276/400000 [00:41<00:07, 8145.07it/s] 85%|████████▍ | 339140/400000 [00:41<00:07, 8286.85it/s] 85%|████████▍ | 339985/400000 [00:41<00:07, 8334.82it/s] 85%|████████▌ | 340821/400000 [00:41<00:07, 8217.83it/s] 85%|████████▌ | 341691/400000 [00:41<00:06, 8356.51it/s] 86%|████████▌ | 342554/400000 [00:41<00:06, 8435.52it/s] 86%|████████▌ | 343399/400000 [00:41<00:06, 8431.02it/s] 86%|████████▌ | 344295/400000 [00:42<00:06, 8580.62it/s] 86%|████████▋ | 345155/400000 [00:42<00:06, 8377.47it/s] 86%|████████▋ | 346000/400000 [00:42<00:06, 8398.21it/s] 87%|████████▋ | 346842/400000 [00:42<00:06, 8178.00it/s] 87%|████████▋ | 347663/400000 [00:42<00:06, 8086.84it/s] 87%|████████▋ | 348527/400000 [00:42<00:06, 8243.45it/s] 87%|████████▋ | 349366/400000 [00:42<00:06, 8285.95it/s] 88%|████████▊ | 350227/400000 [00:42<00:05, 8379.80it/s] 88%|████████▊ | 351095/400000 [00:42<00:05, 8465.60it/s] 88%|████████▊ | 351943/400000 [00:42<00:05, 8413.80it/s] 88%|████████▊ | 352789/400000 [00:43<00:05, 8427.20it/s] 88%|████████▊ | 353639/400000 [00:43<00:05, 8447.49it/s] 89%|████████▊ | 354485/400000 [00:43<00:05, 8418.38it/s] 89%|████████▉ | 355328/400000 [00:43<00:05, 8347.99it/s] 89%|████████▉ | 356164/400000 [00:43<00:05, 8304.04it/s] 89%|████████▉ | 357013/400000 [00:43<00:05, 8358.06it/s] 89%|████████▉ | 357850/400000 [00:43<00:05, 8279.84it/s] 90%|████████▉ | 358719/400000 [00:43<00:04, 8397.28it/s] 90%|████████▉ | 359585/400000 [00:43<00:04, 8472.16it/s] 90%|█████████ | 360433/400000 [00:43<00:04, 8418.85it/s] 90%|█████████ | 361286/400000 [00:44<00:04, 8449.75it/s] 91%|█████████ | 362132/400000 [00:44<00:04, 8412.67it/s] 91%|█████████ | 362995/400000 [00:44<00:04, 8476.37it/s] 91%|█████████ | 363843/400000 [00:44<00:04, 8205.00it/s] 91%|█████████ | 364666/400000 [00:44<00:04, 8035.06it/s] 91%|█████████▏| 365501/400000 [00:44<00:04, 8126.18it/s] 92%|█████████▏| 366357/400000 [00:44<00:04, 8250.47it/s] 92%|█████████▏| 367184/400000 [00:44<00:03, 8214.53it/s] 92%|█████████▏| 368007/400000 [00:44<00:03, 8154.40it/s] 92%|█████████▏| 368824/400000 [00:45<00:03, 8056.92it/s] 92%|█████████▏| 369649/400000 [00:45<00:03, 8108.99it/s] 93%|█████████▎| 370461/400000 [00:45<00:03, 8043.67it/s] 93%|█████████▎| 371267/400000 [00:45<00:03, 8019.25it/s] 93%|█████████▎| 372090/400000 [00:45<00:03, 8080.03it/s] 93%|█████████▎| 372899/400000 [00:45<00:03, 7957.12it/s] 93%|█████████▎| 373696/400000 [00:45<00:03, 7909.88it/s] 94%|█████████▎| 374498/400000 [00:45<00:03, 7940.40it/s] 94%|█████████▍| 375293/400000 [00:45<00:03, 7912.85it/s] 94%|█████████▍| 376106/400000 [00:45<00:02, 7976.59it/s] 94%|█████████▍| 376905/400000 [00:46<00:02, 7971.59it/s] 94%|█████████▍| 377751/400000 [00:46<00:02, 8110.67it/s] 95%|█████████▍| 378587/400000 [00:46<00:02, 8182.90it/s] 95%|█████████▍| 379444/400000 [00:46<00:02, 8293.83it/s] 95%|█████████▌| 380296/400000 [00:46<00:02, 8360.27it/s] 95%|█████████▌| 381145/400000 [00:46<00:02, 8395.38it/s] 96%|█████████▌| 382010/400000 [00:46<00:02, 8468.70it/s] 96%|█████████▌| 382863/400000 [00:46<00:02, 8484.86it/s] 96%|█████████▌| 383712/400000 [00:46<00:01, 8444.56it/s] 96%|█████████▌| 384557/400000 [00:46<00:01, 8339.99it/s] 96%|█████████▋| 385392/400000 [00:47<00:01, 8298.96it/s] 97%|█████████▋| 386223/400000 [00:47<00:01, 8231.98it/s] 97%|█████████▋| 387065/400000 [00:47<00:01, 8286.68it/s] 97%|█████████▋| 387895/400000 [00:47<00:01, 8050.44it/s] 97%|█████████▋| 388710/400000 [00:47<00:01, 8077.01it/s] 97%|█████████▋| 389524/400000 [00:47<00:01, 8094.88it/s] 98%|█████████▊| 390393/400000 [00:47<00:01, 8262.37it/s] 98%|█████████▊| 391228/400000 [00:47<00:01, 8288.46it/s] 98%|█████████▊| 392108/400000 [00:47<00:00, 8432.82it/s] 98%|█████████▊| 392955/400000 [00:47<00:00, 8443.05it/s] 98%|█████████▊| 393801/400000 [00:48<00:00, 8394.82it/s] 99%|█████████▊| 394683/400000 [00:48<00:00, 8516.51it/s] 99%|█████████▉| 395575/400000 [00:48<00:00, 8631.01it/s] 99%|█████████▉| 396495/400000 [00:48<00:00, 8792.34it/s] 99%|█████████▉| 397391/400000 [00:48<00:00, 8839.65it/s]100%|█████████▉| 398277/400000 [00:48<00:00, 8740.98it/s]100%|█████████▉| 399153/400000 [00:48<00:00, 8697.55it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8202.29it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8eff0b4dd8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010847074791085567 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.010979356175681021 	 Accuracy: 61

  model saves at 61% accuracy 

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

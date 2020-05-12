
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f90e23aff28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 05:12:57.212706
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 05:12:57.216710
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 05:12:57.219936
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 05:12:57.223223
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f90ee174390> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351218.2188
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 226142.8125
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 123122.0781
Epoch 4/10

1/1 [==============================] - 0s 117ms/step - loss: 57673.6172
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 27503.7148
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 14620.1025
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 8729.7051
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 5795.3726
Epoch 9/10

1/1 [==============================] - 0s 108ms/step - loss: 4132.7363
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 3206.3318

  #### Inference Need return ypred, ytrue ######################### 
[[-2.27494812e+00  9.78118420e-01 -7.73878455e-01 -1.04549778e+00
   1.13289607e+00 -6.06374502e-01 -5.12717426e-01  5.01362801e-01
  -8.86241317e-01 -1.67799258e+00 -4.75018173e-01  1.45533001e+00
  -2.21781278e+00  5.42192340e-01 -7.73136735e-01  3.85109633e-01
   5.17632544e-01  4.12354708e-01  3.45215350e-01  1.20166026e-01
   9.02823806e-01  9.07469332e-01 -3.15842748e-01  1.41578794e+00
  -1.08594835e+00  2.49226069e+00  6.94157064e-01  1.87629497e+00
  -1.50805497e+00  4.58843082e-01 -2.04184666e-01  3.04580808e-01
   7.45062113e-01 -2.07796395e-01 -1.09884262e-01 -4.18562561e-01
   1.26473880e+00 -1.37521553e+00 -3.67356151e-01 -4.10452545e-01
  -7.94035912e-01  5.83496988e-01 -2.92755771e+00  2.65636265e-01
   2.52566791e+00 -2.24721646e+00  2.80402601e-03 -2.31177831e+00
   1.85261935e-01  2.82676071e-02 -2.22929984e-01 -1.78453803e+00
  -1.15701997e+00  1.27794802e+00  6.07119679e-01 -8.01364407e-02
  -1.50905848e+00 -2.52152801e-01  5.47374606e-01  5.15236378e-01
   6.03877723e-01  1.14857025e+01  1.18922634e+01  1.09397564e+01
   1.48704920e+01  1.25092239e+01  1.35653658e+01  1.32079077e+01
   1.27590389e+01  1.05441675e+01  1.21379871e+01  1.39505749e+01
   1.39937038e+01  1.12132282e+01  1.35580091e+01  1.21651201e+01
   1.54781618e+01  1.26250658e+01  1.23396454e+01  1.27168522e+01
   1.41615829e+01  1.14448204e+01  1.25641146e+01  1.32887621e+01
   1.07215214e+01  1.21285267e+01  1.50621538e+01  1.44574156e+01
   1.33726025e+01  1.05723152e+01  1.29757671e+01  1.43872032e+01
   1.20288277e+01  1.25185223e+01  9.98582458e+00  1.24750814e+01
   1.24915419e+01  1.38572245e+01  1.16686411e+01  1.30842934e+01
   1.43378620e+01  9.97465801e+00  1.35012493e+01  1.25555449e+01
   1.05803556e+01  1.37059135e+01  1.34047680e+01  1.36275139e+01
   1.30655832e+01  1.31333055e+01  1.11725006e+01  1.54415321e+01
   1.48509197e+01  1.44658842e+01  1.27206678e+01  1.06915483e+01
   1.20092773e+01  1.30348940e+01  1.38595886e+01  1.35191975e+01
   5.21666408e-01  1.83861256e+00 -6.12497568e-01 -1.31153679e+00
  -9.87100303e-02  1.81557155e+00  1.03944111e+00  7.35176563e-01
   4.45822626e-01  5.21808863e-01  1.03592694e+00  1.67999315e+00
   4.79982883e-01 -9.24160033e-02 -2.38488436e+00  9.30683851e-01
   1.00424957e+00 -2.58570099e+00 -7.27149725e-01 -1.76933360e+00
  -1.03501213e+00 -7.95170546e-01 -1.19634593e+00  9.04496193e-01
  -7.74291456e-01 -2.07732868e+00 -1.19962204e+00 -1.65834653e+00
  -1.01795888e+00 -7.45414019e-01 -4.26657915e-01 -1.16248345e+00
  -2.99408853e-01  9.87238884e-01  4.21286076e-01 -1.07092309e+00
  -2.11204696e+00 -2.25404233e-01 -1.25835848e+00  1.07024521e-01
  -6.68573260e-01 -1.12070787e+00 -1.45006537e-01 -3.86274457e-01
  -4.55102324e-01 -8.41097295e-01 -5.51408172e-01 -7.88356960e-01
   1.27096009e+00 -4.92510438e-01 -9.46339488e-01 -1.13790619e+00
   4.62338746e-01  1.55791938e-01  2.27273536e+00 -2.05524993e+00
  -5.11995733e-01 -5.86936533e-01 -3.65737200e-01 -1.31667674e+00
   8.55702162e-02  1.49747980e+00  2.53138185e-01  5.65696061e-01
   4.90851641e-01  1.00222898e+00  9.77854848e-01  1.15648115e+00
   1.40508711e-01  7.70153522e-01  1.61212838e+00  1.45455873e+00
   7.01589942e-01  2.88488626e+00  1.98846602e+00  1.52332306e+00
   8.91914308e-01  1.39300466e-01  4.80348468e-01  1.56037104e+00
   4.96828198e-01  5.47205567e-01  1.21829295e+00  2.26564002e+00
   2.73820877e+00  1.39774275e+00  8.40484500e-02  7.54957795e-02
   1.94121158e+00  1.78643322e+00  1.65308976e+00  2.38064170e-01
   1.16883540e+00  1.24173474e+00  3.89189482e-01  3.61868620e+00
   3.02948284e+00  4.60022068e+00  4.45904851e-01  3.07449961e+00
   1.05463648e+00  1.39425290e+00  1.75921679e+00  1.34406853e+00
   1.65646493e-01  3.91824067e-01  1.56518281e-01  1.93099117e+00
   1.83991218e+00  1.16311574e+00  2.56768990e+00  2.89570391e-01
   2.91110516e+00  9.54048514e-01  1.06970906e+00  1.38539648e+00
   1.63462424e+00  1.20774925e+00  2.32856274e+00  2.30908203e+00
   3.17934692e-01  1.30099516e+01  1.21888380e+01  1.20689297e+01
   1.27808714e+01  1.22620125e+01  1.13261271e+01  1.21354933e+01
   1.42640114e+01  1.24316273e+01  1.44746208e+01  1.27007856e+01
   1.20790920e+01  1.26483345e+01  1.53964090e+01  1.30556049e+01
   1.50332727e+01  1.37448092e+01  1.54314861e+01  1.46934328e+01
   1.28555079e+01  1.22460899e+01  1.23514013e+01  1.03785486e+01
   1.06770220e+01  1.37237978e+01  1.24028063e+01  1.14582491e+01
   1.25036469e+01  1.09123802e+01  1.32729168e+01  1.01865911e+01
   1.23855896e+01  1.25192862e+01  1.42817307e+01  1.30367012e+01
   1.35002155e+01  1.42989311e+01  1.14289389e+01  1.08878984e+01
   1.17656612e+01  1.31065054e+01  1.12550936e+01  1.40398321e+01
   1.10651026e+01  1.31733646e+01  1.24126663e+01  1.46521816e+01
   1.30929632e+01  1.04307308e+01  1.56801262e+01  1.38479681e+01
   1.30398865e+01  1.21922884e+01  1.48001595e+01  1.31313515e+01
   1.25166321e+01  1.31777382e+01  1.34582911e+01  1.29863710e+01
   1.25591636e-01  3.69305706e+00  4.11907792e-01  4.91432333e+00
   1.72489405e+00  4.87483323e-01  6.29250944e-01  1.42018068e+00
   2.35518646e+00  7.33308792e-01  1.31032467e+00  2.47406423e-01
   2.93297625e+00  4.04725671e-01  1.79813552e+00  2.25085354e+00
   1.07547355e+00  3.07209110e+00  5.77699244e-01  1.38073349e+00
   5.80032945e-01  2.00228691e+00  1.19749713e+00  3.29996109e-01
   2.42377043e-01  7.68421829e-01  3.08471203e-01  3.22393227e+00
   1.99851239e+00  5.21460533e-01  6.83612883e-01  6.89125001e-01
   1.51317155e+00  3.02044153e-01  2.02030611e+00  1.75014734e-01
   2.69178152e+00  3.30875826e+00  1.39854252e-01  3.99977088e-01
   3.86484385e-01  5.81222832e-01  3.62690115e+00  1.33507848e-01
   1.43578291e+00  1.04746318e+00  1.33744419e-01  1.64943171e+00
   2.01711392e+00  2.56054831e+00  2.36758280e+00  1.06250584e-01
   1.51424837e+00  1.84477854e+00  2.59761763e+00  8.91003370e-01
   1.55445457e+00  2.82569766e-01  9.13103104e-01  6.03937089e-01
  -1.44588909e+01  1.00162401e+01 -1.76176529e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 05:13:07.850783
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.2023
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 05:13:07.855532
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   7990.17
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 05:13:07.859693
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.4403
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 05:13:07.863652
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -714.576
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140259888403008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140258946850944
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140258946851448
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140258946851952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140258946852456
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140258946852960

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f90cdd88f98> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.509720
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.482167
grad_step = 000002, loss = 0.457620
grad_step = 000003, loss = 0.431175
grad_step = 000004, loss = 0.403002
grad_step = 000005, loss = 0.378198
grad_step = 000006, loss = 0.362982
grad_step = 000007, loss = 0.359656
grad_step = 000008, loss = 0.348116
grad_step = 000009, loss = 0.332332
grad_step = 000010, loss = 0.320337
grad_step = 000011, loss = 0.312823
grad_step = 000012, loss = 0.306579
grad_step = 000013, loss = 0.298942
grad_step = 000014, loss = 0.289283
grad_step = 000015, loss = 0.278409
grad_step = 000016, loss = 0.267710
grad_step = 000017, loss = 0.258505
grad_step = 000018, loss = 0.250957
grad_step = 000019, loss = 0.243546
grad_step = 000020, loss = 0.234552
grad_step = 000021, loss = 0.224530
grad_step = 000022, loss = 0.214947
grad_step = 000023, loss = 0.206621
grad_step = 000024, loss = 0.199131
grad_step = 000025, loss = 0.191525
grad_step = 000026, loss = 0.183372
grad_step = 000027, loss = 0.175019
grad_step = 000028, loss = 0.167131
grad_step = 000029, loss = 0.160010
grad_step = 000030, loss = 0.153108
grad_step = 000031, loss = 0.145764
grad_step = 000032, loss = 0.138255
grad_step = 000033, loss = 0.131185
grad_step = 000034, loss = 0.124672
grad_step = 000035, loss = 0.118352
grad_step = 000036, loss = 0.111891
grad_step = 000037, loss = 0.105468
grad_step = 000038, loss = 0.099426
grad_step = 000039, loss = 0.093808
grad_step = 000040, loss = 0.088336
grad_step = 000041, loss = 0.082834
grad_step = 000042, loss = 0.077470
grad_step = 000043, loss = 0.072429
grad_step = 000044, loss = 0.067698
grad_step = 000045, loss = 0.063090
grad_step = 000046, loss = 0.058563
grad_step = 000047, loss = 0.054277
grad_step = 000048, loss = 0.050336
grad_step = 000049, loss = 0.046600
grad_step = 000050, loss = 0.042895
grad_step = 000051, loss = 0.039379
grad_step = 000052, loss = 0.036196
grad_step = 000053, loss = 0.033191
grad_step = 000054, loss = 0.030262
grad_step = 000055, loss = 0.027542
grad_step = 000056, loss = 0.025094
grad_step = 000057, loss = 0.022773
grad_step = 000058, loss = 0.020579
grad_step = 000059, loss = 0.018489
grad_step = 000060, loss = 0.016613
grad_step = 000061, loss = 0.014947
grad_step = 000062, loss = 0.013372
grad_step = 000063, loss = 0.011919
grad_step = 000064, loss = 0.010636
grad_step = 000065, loss = 0.009479
grad_step = 000066, loss = 0.008421
grad_step = 000067, loss = 0.007505
grad_step = 000068, loss = 0.006701
grad_step = 000069, loss = 0.005969
grad_step = 000070, loss = 0.005350
grad_step = 000071, loss = 0.004843
grad_step = 000072, loss = 0.004435
grad_step = 000073, loss = 0.004148
grad_step = 000074, loss = 0.003857
grad_step = 000075, loss = 0.003510
grad_step = 000076, loss = 0.003270
grad_step = 000077, loss = 0.003209
grad_step = 000078, loss = 0.003040
grad_step = 000079, loss = 0.002883
grad_step = 000080, loss = 0.002904
grad_step = 000081, loss = 0.002790
grad_step = 000082, loss = 0.002713
grad_step = 000083, loss = 0.002740
grad_step = 000084, loss = 0.002640
grad_step = 000085, loss = 0.002635
grad_step = 000086, loss = 0.002619
grad_step = 000087, loss = 0.002547
grad_step = 000088, loss = 0.002564
grad_step = 000089, loss = 0.002510
grad_step = 000090, loss = 0.002483
grad_step = 000091, loss = 0.002480
grad_step = 000092, loss = 0.002424
grad_step = 000093, loss = 0.002421
grad_step = 000094, loss = 0.002390
grad_step = 000095, loss = 0.002357
grad_step = 000096, loss = 0.002352
grad_step = 000097, loss = 0.002311
grad_step = 000098, loss = 0.002299
grad_step = 000099, loss = 0.002282
grad_step = 000100, loss = 0.002250
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002244
grad_step = 000102, loss = 0.002219
grad_step = 000103, loss = 0.002200
grad_step = 000104, loss = 0.002191
grad_step = 000105, loss = 0.002167
grad_step = 000106, loss = 0.002154
grad_step = 000107, loss = 0.002141
grad_step = 000108, loss = 0.002121
grad_step = 000109, loss = 0.002111
grad_step = 000110, loss = 0.002093
grad_step = 000111, loss = 0.002075
grad_step = 000112, loss = 0.002065
grad_step = 000113, loss = 0.002046
grad_step = 000114, loss = 0.002033
grad_step = 000115, loss = 0.002024
grad_step = 000116, loss = 0.002019
grad_step = 000117, loss = 0.002017
grad_step = 000118, loss = 0.001983
grad_step = 000119, loss = 0.001953
grad_step = 000120, loss = 0.001954
grad_step = 000121, loss = 0.001969
grad_step = 000122, loss = 0.002000
grad_step = 000123, loss = 0.001971
grad_step = 000124, loss = 0.001907
grad_step = 000125, loss = 0.001882
grad_step = 000126, loss = 0.001915
grad_step = 000127, loss = 0.001963
grad_step = 000128, loss = 0.001941
grad_step = 000129, loss = 0.001873
grad_step = 000130, loss = 0.001825
grad_step = 000131, loss = 0.001844
grad_step = 000132, loss = 0.001916
grad_step = 000133, loss = 0.001961
grad_step = 000134, loss = 0.001926
grad_step = 000135, loss = 0.001829
grad_step = 000136, loss = 0.001768
grad_step = 000137, loss = 0.001766
grad_step = 000138, loss = 0.001809
grad_step = 000139, loss = 0.001922
grad_step = 000140, loss = 0.002005
grad_step = 000141, loss = 0.001941
grad_step = 000142, loss = 0.001798
grad_step = 000143, loss = 0.001719
grad_step = 000144, loss = 0.001747
grad_step = 000145, loss = 0.001846
grad_step = 000146, loss = 0.001877
grad_step = 000147, loss = 0.001808
grad_step = 000148, loss = 0.001702
grad_step = 000149, loss = 0.001692
grad_step = 000150, loss = 0.001748
grad_step = 000151, loss = 0.001785
grad_step = 000152, loss = 0.001768
grad_step = 000153, loss = 0.001698
grad_step = 000154, loss = 0.001659
grad_step = 000155, loss = 0.001672
grad_step = 000156, loss = 0.001697
grad_step = 000157, loss = 0.001735
grad_step = 000158, loss = 0.001706
grad_step = 000159, loss = 0.001677
grad_step = 000160, loss = 0.001643
grad_step = 000161, loss = 0.001638
grad_step = 000162, loss = 0.001661
grad_step = 000163, loss = 0.001678
grad_step = 000164, loss = 0.001694
grad_step = 000165, loss = 0.001676
grad_step = 000166, loss = 0.001657
grad_step = 000167, loss = 0.001627
grad_step = 000168, loss = 0.001617
grad_step = 000169, loss = 0.001626
grad_step = 000170, loss = 0.001638
grad_step = 000171, loss = 0.001654
grad_step = 000172, loss = 0.001649
grad_step = 000173, loss = 0.001644
grad_step = 000174, loss = 0.001621
grad_step = 000175, loss = 0.001604
grad_step = 000176, loss = 0.001600
grad_step = 000177, loss = 0.001606
grad_step = 000178, loss = 0.001622
grad_step = 000179, loss = 0.001635
grad_step = 000180, loss = 0.001655
grad_step = 000181, loss = 0.001652
grad_step = 000182, loss = 0.001642
grad_step = 000183, loss = 0.001610
grad_step = 000184, loss = 0.001588
grad_step = 000185, loss = 0.001586
grad_step = 000186, loss = 0.001602
grad_step = 000187, loss = 0.001624
grad_step = 000188, loss = 0.001629
grad_step = 000189, loss = 0.001634
grad_step = 000190, loss = 0.001604
grad_step = 000191, loss = 0.001581
grad_step = 000192, loss = 0.001573
grad_step = 000193, loss = 0.001584
grad_step = 000194, loss = 0.001604
grad_step = 000195, loss = 0.001617
grad_step = 000196, loss = 0.001644
grad_step = 000197, loss = 0.001629
grad_step = 000198, loss = 0.001612
grad_step = 000199, loss = 0.001577
grad_step = 000200, loss = 0.001560
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001572
grad_step = 000202, loss = 0.001594
grad_step = 000203, loss = 0.001621
grad_step = 000204, loss = 0.001623
grad_step = 000205, loss = 0.001625
grad_step = 000206, loss = 0.001593
grad_step = 000207, loss = 0.001563
grad_step = 000208, loss = 0.001548
grad_step = 000209, loss = 0.001561
grad_step = 000210, loss = 0.001589
grad_step = 000211, loss = 0.001614
grad_step = 000212, loss = 0.001662
grad_step = 000213, loss = 0.001665
grad_step = 000214, loss = 0.001651
grad_step = 000215, loss = 0.001590
grad_step = 000216, loss = 0.001559
grad_step = 000217, loss = 0.001574
grad_step = 000218, loss = 0.001610
grad_step = 000219, loss = 0.001631
grad_step = 000220, loss = 0.001677
grad_step = 000221, loss = 0.001642
grad_step = 000222, loss = 0.001563
grad_step = 000223, loss = 0.001530
grad_step = 000224, loss = 0.001570
grad_step = 000225, loss = 0.001608
grad_step = 000226, loss = 0.001582
grad_step = 000227, loss = 0.001542
grad_step = 000228, loss = 0.001517
grad_step = 000229, loss = 0.001534
grad_step = 000230, loss = 0.001561
grad_step = 000231, loss = 0.001560
grad_step = 000232, loss = 0.001546
grad_step = 000233, loss = 0.001519
grad_step = 000234, loss = 0.001505
grad_step = 000235, loss = 0.001504
grad_step = 000236, loss = 0.001515
grad_step = 000237, loss = 0.001535
grad_step = 000238, loss = 0.001556
grad_step = 000239, loss = 0.001590
grad_step = 000240, loss = 0.001598
grad_step = 000241, loss = 0.001602
grad_step = 000242, loss = 0.001552
grad_step = 000243, loss = 0.001506
grad_step = 000244, loss = 0.001486
grad_step = 000245, loss = 0.001512
grad_step = 000246, loss = 0.001562
grad_step = 000247, loss = 0.001590
grad_step = 000248, loss = 0.001602
grad_step = 000249, loss = 0.001551
grad_step = 000250, loss = 0.001491
grad_step = 000251, loss = 0.001481
grad_step = 000252, loss = 0.001513
grad_step = 000253, loss = 0.001533
grad_step = 000254, loss = 0.001505
grad_step = 000255, loss = 0.001472
grad_step = 000256, loss = 0.001464
grad_step = 000257, loss = 0.001477
grad_step = 000258, loss = 0.001492
grad_step = 000259, loss = 0.001494
grad_step = 000260, loss = 0.001485
grad_step = 000261, loss = 0.001469
grad_step = 000262, loss = 0.001460
grad_step = 000263, loss = 0.001448
grad_step = 000264, loss = 0.001442
grad_step = 000265, loss = 0.001437
grad_step = 000266, loss = 0.001433
grad_step = 000267, loss = 0.001432
grad_step = 000268, loss = 0.001431
grad_step = 000269, loss = 0.001436
grad_step = 000270, loss = 0.001453
grad_step = 000271, loss = 0.001525
grad_step = 000272, loss = 0.001754
grad_step = 000273, loss = 0.002298
grad_step = 000274, loss = 0.003022
grad_step = 000275, loss = 0.002362
grad_step = 000276, loss = 0.001539
grad_step = 000277, loss = 0.001652
grad_step = 000278, loss = 0.002021
grad_step = 000279, loss = 0.001599
grad_step = 000280, loss = 0.001573
grad_step = 000281, loss = 0.001816
grad_step = 000282, loss = 0.001502
grad_step = 000283, loss = 0.001623
grad_step = 000284, loss = 0.001673
grad_step = 000285, loss = 0.001476
grad_step = 000286, loss = 0.001639
grad_step = 000287, loss = 0.001549
grad_step = 000288, loss = 0.001496
grad_step = 000289, loss = 0.001590
grad_step = 000290, loss = 0.001470
grad_step = 000291, loss = 0.001502
grad_step = 000292, loss = 0.001513
grad_step = 000293, loss = 0.001465
grad_step = 000294, loss = 0.001459
grad_step = 000295, loss = 0.001480
grad_step = 000296, loss = 0.001453
grad_step = 000297, loss = 0.001424
grad_step = 000298, loss = 0.001463
grad_step = 000299, loss = 0.001428
grad_step = 000300, loss = 0.001404
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001435
grad_step = 000302, loss = 0.001435
grad_step = 000303, loss = 0.001388
grad_step = 000304, loss = 0.001387
grad_step = 000305, loss = 0.001429
grad_step = 000306, loss = 0.001393
grad_step = 000307, loss = 0.001366
grad_step = 000308, loss = 0.001383
grad_step = 000309, loss = 0.001392
grad_step = 000310, loss = 0.001385
grad_step = 000311, loss = 0.001364
grad_step = 000312, loss = 0.001352
grad_step = 000313, loss = 0.001358
grad_step = 000314, loss = 0.001370
grad_step = 000315, loss = 0.001357
grad_step = 000316, loss = 0.001339
grad_step = 000317, loss = 0.001336
grad_step = 000318, loss = 0.001333
grad_step = 000319, loss = 0.001340
grad_step = 000320, loss = 0.001345
grad_step = 000321, loss = 0.001341
grad_step = 000322, loss = 0.001334
grad_step = 000323, loss = 0.001326
grad_step = 000324, loss = 0.001316
grad_step = 000325, loss = 0.001306
grad_step = 000326, loss = 0.001301
grad_step = 000327, loss = 0.001297
grad_step = 000328, loss = 0.001294
grad_step = 000329, loss = 0.001295
grad_step = 000330, loss = 0.001298
grad_step = 000331, loss = 0.001307
grad_step = 000332, loss = 0.001334
grad_step = 000333, loss = 0.001388
grad_step = 000334, loss = 0.001486
grad_step = 000335, loss = 0.001583
grad_step = 000336, loss = 0.001575
grad_step = 000337, loss = 0.001419
grad_step = 000338, loss = 0.001274
grad_step = 000339, loss = 0.001319
grad_step = 000340, loss = 0.001411
grad_step = 000341, loss = 0.001343
grad_step = 000342, loss = 0.001256
grad_step = 000343, loss = 0.001297
grad_step = 000344, loss = 0.001329
grad_step = 000345, loss = 0.001269
grad_step = 000346, loss = 0.001242
grad_step = 000347, loss = 0.001284
grad_step = 000348, loss = 0.001281
grad_step = 000349, loss = 0.001233
grad_step = 000350, loss = 0.001229
grad_step = 000351, loss = 0.001254
grad_step = 000352, loss = 0.001252
grad_step = 000353, loss = 0.001227
grad_step = 000354, loss = 0.001204
grad_step = 000355, loss = 0.001206
grad_step = 000356, loss = 0.001221
grad_step = 000357, loss = 0.001225
grad_step = 000358, loss = 0.001211
grad_step = 000359, loss = 0.001195
grad_step = 000360, loss = 0.001182
grad_step = 000361, loss = 0.001171
grad_step = 000362, loss = 0.001165
grad_step = 000363, loss = 0.001163
grad_step = 000364, loss = 0.001163
grad_step = 000365, loss = 0.001167
grad_step = 000366, loss = 0.001184
grad_step = 000367, loss = 0.001228
grad_step = 000368, loss = 0.001329
grad_step = 000369, loss = 0.001514
grad_step = 000370, loss = 0.001631
grad_step = 000371, loss = 0.001509
grad_step = 000372, loss = 0.001239
grad_step = 000373, loss = 0.001143
grad_step = 000374, loss = 0.001314
grad_step = 000375, loss = 0.001333
grad_step = 000376, loss = 0.001156
grad_step = 000377, loss = 0.001179
grad_step = 000378, loss = 0.001259
grad_step = 000379, loss = 0.001165
grad_step = 000380, loss = 0.001138
grad_step = 000381, loss = 0.001189
grad_step = 000382, loss = 0.001137
grad_step = 000383, loss = 0.001114
grad_step = 000384, loss = 0.001143
grad_step = 000385, loss = 0.001133
grad_step = 000386, loss = 0.001085
grad_step = 000387, loss = 0.001085
grad_step = 000388, loss = 0.001116
grad_step = 000389, loss = 0.001095
grad_step = 000390, loss = 0.001051
grad_step = 000391, loss = 0.001051
grad_step = 000392, loss = 0.001073
grad_step = 000393, loss = 0.001073
grad_step = 000394, loss = 0.001070
grad_step = 000395, loss = 0.001064
grad_step = 000396, loss = 0.001040
grad_step = 000397, loss = 0.001017
grad_step = 000398, loss = 0.001011
grad_step = 000399, loss = 0.001003
grad_step = 000400, loss = 0.000997
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001004
grad_step = 000402, loss = 0.001009
grad_step = 000403, loss = 0.001012
grad_step = 000404, loss = 0.001022
grad_step = 000405, loss = 0.001040
grad_step = 000406, loss = 0.001053
grad_step = 000407, loss = 0.001073
grad_step = 000408, loss = 0.001076
grad_step = 000409, loss = 0.001050
grad_step = 000410, loss = 0.000994
grad_step = 000411, loss = 0.000950
grad_step = 000412, loss = 0.000936
grad_step = 000413, loss = 0.000952
grad_step = 000414, loss = 0.000970
grad_step = 000415, loss = 0.000960
grad_step = 000416, loss = 0.000932
grad_step = 000417, loss = 0.000910
grad_step = 000418, loss = 0.000907
grad_step = 000419, loss = 0.000916
grad_step = 000420, loss = 0.000925
grad_step = 000421, loss = 0.000923
grad_step = 000422, loss = 0.000913
grad_step = 000423, loss = 0.000896
grad_step = 000424, loss = 0.000879
grad_step = 000425, loss = 0.000867
grad_step = 000426, loss = 0.000860
grad_step = 000427, loss = 0.000858
grad_step = 000428, loss = 0.000859
grad_step = 000429, loss = 0.000863
grad_step = 000430, loss = 0.000876
grad_step = 000431, loss = 0.000899
grad_step = 000432, loss = 0.000959
grad_step = 000433, loss = 0.001058
grad_step = 000434, loss = 0.001221
grad_step = 000435, loss = 0.001279
grad_step = 000436, loss = 0.001223
grad_step = 000437, loss = 0.000915
grad_step = 000438, loss = 0.000858
grad_step = 000439, loss = 0.001052
grad_step = 000440, loss = 0.001057
grad_step = 000441, loss = 0.000879
grad_step = 000442, loss = 0.000888
grad_step = 000443, loss = 0.000935
grad_step = 000444, loss = 0.000874
grad_step = 000445, loss = 0.000876
grad_step = 000446, loss = 0.000837
grad_step = 000447, loss = 0.000838
grad_step = 000448, loss = 0.000869
grad_step = 000449, loss = 0.000783
grad_step = 000450, loss = 0.000805
grad_step = 000451, loss = 0.000839
grad_step = 000452, loss = 0.000770
grad_step = 000453, loss = 0.000772
grad_step = 000454, loss = 0.000786
grad_step = 000455, loss = 0.000765
grad_step = 000456, loss = 0.000747
grad_step = 000457, loss = 0.000749
grad_step = 000458, loss = 0.000754
grad_step = 000459, loss = 0.000735
grad_step = 000460, loss = 0.000730
grad_step = 000461, loss = 0.000762
grad_step = 000462, loss = 0.000794
grad_step = 000463, loss = 0.000895
grad_step = 000464, loss = 0.001159
grad_step = 000465, loss = 0.001516
grad_step = 000466, loss = 0.001277
grad_step = 000467, loss = 0.001018
grad_step = 000468, loss = 0.000773
grad_step = 000469, loss = 0.000830
grad_step = 000470, loss = 0.001046
grad_step = 000471, loss = 0.000979
grad_step = 000472, loss = 0.000740
grad_step = 000473, loss = 0.000741
grad_step = 000474, loss = 0.000876
grad_step = 000475, loss = 0.000802
grad_step = 000476, loss = 0.000687
grad_step = 000477, loss = 0.000736
grad_step = 000478, loss = 0.000782
grad_step = 000479, loss = 0.000737
grad_step = 000480, loss = 0.000674
grad_step = 000481, loss = 0.000671
grad_step = 000482, loss = 0.000733
grad_step = 000483, loss = 0.000774
grad_step = 000484, loss = 0.000738
grad_step = 000485, loss = 0.000651
grad_step = 000486, loss = 0.000623
grad_step = 000487, loss = 0.000654
grad_step = 000488, loss = 0.000679
grad_step = 000489, loss = 0.000681
grad_step = 000490, loss = 0.000652
grad_step = 000491, loss = 0.000605
grad_step = 000492, loss = 0.000598
grad_step = 000493, loss = 0.000620
grad_step = 000494, loss = 0.000627
grad_step = 000495, loss = 0.000618
grad_step = 000496, loss = 0.000596
grad_step = 000497, loss = 0.000576
grad_step = 000498, loss = 0.000579
grad_step = 000499, loss = 0.000591
grad_step = 000500, loss = 0.000592
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000582
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

  date_run                              2020-05-12 05:13:31.933128
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.172689
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 05:13:31.939938
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0660746
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 05:13:31.947700
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.106621
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 05:13:31.953460
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                               -0.00402645
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
0   2020-05-12 05:12:57.212706  ...    mean_absolute_error
1   2020-05-12 05:12:57.216710  ...     mean_squared_error
2   2020-05-12 05:12:57.219936  ...  median_absolute_error
3   2020-05-12 05:12:57.223223  ...               r2_score
4   2020-05-12 05:13:07.850783  ...    mean_absolute_error
5   2020-05-12 05:13:07.855532  ...     mean_squared_error
6   2020-05-12 05:13:07.859693  ...  median_absolute_error
7   2020-05-12 05:13:07.863652  ...               r2_score
8   2020-05-12 05:13:31.933128  ...    mean_absolute_error
9   2020-05-12 05:13:31.939938  ...     mean_squared_error
10  2020-05-12 05:13:31.947700  ...  median_absolute_error
11  2020-05-12 05:13:31.953460  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:01<?, ?it/s] 38%|███▊      | 3809280/9912422 [00:01<00:00, 38087297.08it/s]9920512it [00:01, 7728625.26it/s]                              
0it [00:00, ?it/s]32768it [00:00, 744995.22it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 887054.63it/s]1654784it [00:00, 12703677.95it/s]                         
0it [00:00, ?it/s]8192it [00:00, 272020.60it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ef204f128> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e8e6b2ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ef0f22eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e8e18a0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ef204f128> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ea391ae80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ef204f128> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e979ca748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ef0f5ec18> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ea391ae80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2e8e6b2ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f22830021d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=66015d9ca1dbc574209eb4eacc7367773ca30348dfa894685d8b58a8696fa871
  Stored in directory: /tmp/pip-ephem-wheel-cache-g1c7v9wv/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2279170048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2334720/17464789 [===>..........................] - ETA: 0s
10543104/17464789 [=================>............] - ETA: 0s
16695296/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 05:14:57.961269: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 05:14:57.965909: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 05:14:57.966034: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560eb9f7e390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 05:14:57.966048: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4980 - accuracy: 0.5110
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5325 - accuracy: 0.5088
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5337 - accuracy: 0.5087
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5155 - accuracy: 0.5099
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5631 - accuracy: 0.5067
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5508 - accuracy: 0.5076
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5823 - accuracy: 0.5055
11000/25000 [============>.................] - ETA: 4s - loss: 7.5830 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 3s - loss: 7.5721 - accuracy: 0.5062
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6112 - accuracy: 0.5036
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 2s - loss: 7.6257 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6321 - accuracy: 0.5023
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6450 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6731 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 9s 358us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 05:15:13.675747
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 05:15:13.675747  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 05:15:19.334062: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 05:15:19.339693: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 05:15:19.339887: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5558fa962150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 05:15:19.339900: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fe463b6ebe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 909ms/step - loss: 1.5745 - crf_viterbi_accuracy: 0.0933 - val_loss: 1.5426 - val_crf_viterbi_accuracy: 0.0800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe46b2cb128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5951 - accuracy: 0.5047
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6250 - accuracy: 0.5027
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5976 - accuracy: 0.5045
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5884 - accuracy: 0.5051
11000/25000 [============>.................] - ETA: 4s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6383 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7002 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7000 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7024 - accuracy: 0.4977
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6876 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6422 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fe3fa843ac8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:26:21, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:38:32, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:42:13, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:12:02, 29.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:43:32, 41.7kB/s].vector_cache/glove.6B.zip:   1%|          | 8.30M/862M [00:01<3:59:16, 59.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:46:40, 84.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<1:56:15, 121kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<1:21:00, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<56:25, 247kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:01<39:28, 351kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:02<27:31, 500kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<19:18, 710kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<13:31, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:02<09:32, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<07:19, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:03<05:12, 2.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<20:17, 662kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<16:30, 813kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<12:08, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:05<08:36, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<2:16:19, 98.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<1:36:47, 138kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<1:07:58, 196kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<50:33, 263kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<36:46, 361kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<25:58, 510kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<21:16, 621kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<16:14, 814kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<11:38, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<11:15, 1.17MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<09:14, 1.42MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<06:45, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:49, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:15<06:49, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<05:06, 2.56MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:38, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<05:59, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<04:31, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:13, 2.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:42, 2.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<04:19, 2.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<05:33, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<04:12, 3.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<04:09, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<05:28, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<04:09, 3.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:25, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:03, 3.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:22, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:02, 3.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:03, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:46, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:20, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:03, 3.06MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:34, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:07, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:47, 3.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:54, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:06, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:35, 2.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:30, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:09, 2.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:47, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:05, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:53, 3.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<02:52, 4.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<50:47, 239kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<36:47, 329kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<25:57, 466kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<20:59, 574kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<17:09, 702kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<12:31, 961kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<09:00, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:43, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:04, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:56, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:05, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:33, 2.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:59, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:04, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:37, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:09, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:54, 3.01MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:29, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:03, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:50, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:25, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:59, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:45, 3.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:22, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:56, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:45, 3.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:20, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:56, 2.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:44, 3.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:55, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:23, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:10, 3.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:22, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:14, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<05:27, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:54, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:16, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:41, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:31, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:08, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<03:51, 2.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:08, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:53, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:36, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:25, 3.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:56, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:19, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:59, 2.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:22, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:55, 2.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<02:52, 3.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:58, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:25, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:20, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:34, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:10, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:23, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:00, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:44, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:03, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:39, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<03:30, 3.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:53, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:31, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:25, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:54, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:46, 2.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<02:48, 3.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:31, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:04, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:46, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:01, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:40, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:32, 2.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:28, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:23, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:48, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:27, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:19, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:43, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:25, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:19, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:39, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:11, 2.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:11, 3.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<02:21, 4.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<25:56, 390kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<20:16, 498kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<14:42, 686kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<10:31, 957kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:41, 1.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:48, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:41, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<04:06, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<42:16, 236kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<30:36, 326kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<21:37, 460kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<17:25, 568kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<14:13, 696kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<10:24, 950kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<07:25, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:59, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<07:09, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:46, 2.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<39:21, 248kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<29:34, 330kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<21:10, 460kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<16:19, 594kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<12:24, 781kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<08:54, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:27, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:45, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:07, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<03:40, 2.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<38:33, 248kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<27:58, 341kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<19:46, 482kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<16:02, 592kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<12:11, 778kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<08:45, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:20, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:47, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<04:59, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:41, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:57, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:42, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:46, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:16, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:13, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:03, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:04, 2.99MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:18, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:53, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:10, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:51, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<02:55, 3.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:08, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:50, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<02:53, 3.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:06, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:48, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<02:53, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:06, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:47, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<02:52, 3.08MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:45, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<02:51, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:03, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:45, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:50, 3.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:47, 3.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:03, 4.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<58:10, 148kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<41:35, 207kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<29:14, 293kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<22:23, 381kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<16:32, 515kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<11:43, 725kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<08:17, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<1:08:41, 123kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<48:55, 173kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<34:25, 245kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<24:04, 349kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<1:11:58, 117kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<51:12, 164kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<35:56, 233kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<25:09, 331kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<1:19:27, 105kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<56:25, 147kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<39:35, 209kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<29:32, 280kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<21:30, 384kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<15:13, 540kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<10:42, 765kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<1:08:16, 120kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<49:28, 165kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<34:59, 233kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<25:39, 316kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<18:47, 432kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<13:17, 609kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<11:09, 721kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:37, 933kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:11, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:13, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:09, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:48, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:31, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:58, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<02:58, 2.65MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:55, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:22, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:23, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:30, 3.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:24, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:54, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:55, 2.65MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:50, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:28, 2.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:37, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:38, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:31, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:15, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:28, 3.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:29, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:12, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:24, 3.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:03, 2.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:17, 3.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<01:42, 4.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<28:56, 254kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<20:59, 350kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<14:49, 494kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<12:03, 605kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<09:11, 794kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<06:33, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:15, 1.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:48, 1.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:24, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:14, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:44, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:46, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:32, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:00, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:06, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:16, 3.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:20, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:46, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:48, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:35, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:06, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:19, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<01:42, 4.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<28:57, 238kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<20:58, 328kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<14:48, 463kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<11:54, 572kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<09:02, 753kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<06:27, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:06, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:57, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:38, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:06, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:15, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:18, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:23, 2.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<6:20:32, 17.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<4:26:50, 24.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<3:06:13, 35.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<2:11:08, 49.9kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<1:33:07, 70.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<1:05:23, 99.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<45:29, 142kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<6:43:43, 16.0kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<4:43:00, 22.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<3:17:27, 32.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<2:17:34, 46.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<2:00:00, 53.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<1:24:35, 75.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<59:08, 108kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<42:35, 149kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<31:21, 202kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<22:15, 284kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<15:37, 402kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<12:42, 493kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<09:33, 654kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<06:50, 912kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:09, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:31, 1.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:10, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:21, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:28, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:08, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:07, 2.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:53, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:38, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<01:59, 2.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:47, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:32, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<01:55, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:30, 2.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<01:53, 3.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:41, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:28, 2.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<01:51, 3.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:38, 2.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:26, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<01:49, 3.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:36, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:24, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<01:49, 3.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:35, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:22, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<01:48, 3.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:33, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:16, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:42, 3.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:15, 4.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<21:22, 254kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<15:30, 350kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<10:56, 494kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<08:52, 604kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<06:44, 794kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<04:49, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:36, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:46, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:45, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:09, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:45, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:03, 2.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:38, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:47, 2.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:26, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:09, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:36, 3.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<01:11, 4.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<21:02, 239kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<15:13, 330kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<10:43, 465kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<08:37, 574kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:04, 700kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<05:10, 956kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<03:38, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<07:10, 681kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:31, 885kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<03:58, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:53, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:13, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:22, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:46, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:26, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:49, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:22, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:08, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:36, 2.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:12, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:37, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:03, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:27, 3.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<12:15, 371kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<09:01, 503kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:23, 707kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<04:29, 996kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<47:27, 94.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<34:07, 131kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<24:00, 186kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<16:43, 264kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<14:22, 306kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<10:31, 418kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<07:25, 589kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<06:10, 703kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:45, 910kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:25, 1.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:23, 1.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:49, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:04, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:25, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:07, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:35, 2.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:04, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:52, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:24, 2.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:56, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:46, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:20, 3.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:52, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:10, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:41, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:13, 3.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:38, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:15, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:40, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:03, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:50, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:22, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:50, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:04, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:36, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:11, 3.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:45, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:19, 2.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:45, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:36, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:12, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:32, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:09, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:38, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:30, 2.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:08, 3.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:36, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:28, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:06, 3.07MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:34, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:26, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:32, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:21, 2.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:04, 3.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<00:46, 4.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<13:33, 239kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<09:48, 330kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<06:53, 465kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<05:31, 574kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:11, 756kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:59, 1.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:48, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:16, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:39, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:56, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:04, 2.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:58, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:41, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:14, 2.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:32, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:01, 2.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:22, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:15, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<00:56, 2.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:18, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:11, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:53, 3.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:15, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:09, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<00:52, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:13, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:07, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:50, 3.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:10, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:05, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:48, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:09, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:03, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:07, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:01, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:46, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:05, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:00, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:44, 3.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:58, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:43, 3.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:01, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:10, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:55, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:58, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:55, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:41, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:50, 2.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:38, 3.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:28, 4.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<08:23, 239kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<06:03, 330kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<04:13, 466kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<03:22, 575kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:33, 757kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:48, 1.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:41, 1.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:22, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:59, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:06, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:57, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:42, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:53, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:58, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:46, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:47, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:41, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:31, 3.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:22, 4.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<06:40, 239kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<04:49, 329kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<03:21, 466kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:39, 574kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:00, 755kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:25, 1.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:18, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:03, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:46, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:51, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:44, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:32, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:40, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:36, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:26, 2.89MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:36, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:32, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:24, 3.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:33, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:30, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:22, 3.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:31, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:28, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:21, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:29, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:21, 2.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:21, 2.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:16, 3.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:22, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:20, 2.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:20, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:19, 2.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:14, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:17, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:11, 3.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<09:26, 67.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<06:36, 95.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<04:26, 135kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<03:03, 185kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<02:14, 251kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:33, 352kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<01:04, 464kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:47, 620kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:32, 866kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:26, 960kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:21, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:14, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:08, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:03, 3.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 3.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.32MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 893/400000 [00:00<00:44, 8910.23it/s]  0%|          | 1822/400000 [00:00<00:44, 9019.35it/s]  1%|          | 2600/400000 [00:00<00:46, 8605.71it/s]  1%|          | 3401/400000 [00:00<00:47, 8416.88it/s]  1%|          | 4228/400000 [00:00<00:47, 8370.09it/s]  1%|▏         | 5116/400000 [00:00<00:46, 8516.33it/s]  1%|▏         | 5937/400000 [00:00<00:46, 8419.29it/s]  2%|▏         | 6758/400000 [00:00<00:47, 8355.04it/s]  2%|▏         | 7544/400000 [00:00<00:48, 8150.02it/s]  2%|▏         | 8386/400000 [00:01<00:47, 8226.97it/s]  2%|▏         | 9249/400000 [00:01<00:46, 8342.59it/s]  3%|▎         | 10182/400000 [00:01<00:45, 8615.83it/s]  3%|▎         | 11035/400000 [00:01<00:45, 8561.24it/s]  3%|▎         | 11910/400000 [00:01<00:45, 8615.87it/s]  3%|▎         | 12849/400000 [00:01<00:43, 8833.89it/s]  3%|▎         | 13731/400000 [00:01<00:44, 8658.04it/s]  4%|▎         | 14650/400000 [00:01<00:43, 8807.14it/s]  4%|▍         | 15532/400000 [00:01<00:43, 8796.68it/s]  4%|▍         | 16412/400000 [00:01<00:43, 8760.58it/s]  4%|▍         | 17289/400000 [00:02<00:43, 8757.17it/s]  5%|▍         | 18165/400000 [00:02<00:44, 8615.23it/s]  5%|▍         | 19028/400000 [00:02<00:44, 8524.69it/s]  5%|▍         | 19882/400000 [00:02<00:44, 8465.92it/s]  5%|▌         | 20738/400000 [00:02<00:44, 8492.18it/s]  5%|▌         | 21622/400000 [00:02<00:44, 8592.09it/s]  6%|▌         | 22482/400000 [00:02<00:44, 8539.60it/s]  6%|▌         | 23337/400000 [00:02<00:45, 8290.48it/s]  6%|▌         | 24168/400000 [00:02<00:46, 8062.78it/s]  6%|▋         | 25043/400000 [00:02<00:45, 8255.21it/s]  6%|▋         | 25930/400000 [00:03<00:44, 8430.14it/s]  7%|▋         | 26777/400000 [00:03<00:44, 8326.26it/s]  7%|▋         | 27619/400000 [00:03<00:44, 8353.76it/s]  7%|▋         | 28457/400000 [00:03<00:46, 7990.07it/s]  7%|▋         | 29270/400000 [00:03<00:46, 8028.57it/s]  8%|▊         | 30077/400000 [00:03<00:47, 7852.80it/s]  8%|▊         | 30866/400000 [00:03<00:47, 7703.25it/s]  8%|▊         | 31640/400000 [00:03<00:50, 7366.93it/s]  8%|▊         | 32382/400000 [00:03<00:50, 7215.37it/s]  8%|▊         | 33108/400000 [00:04<00:51, 7140.57it/s]  8%|▊         | 33896/400000 [00:04<00:49, 7345.87it/s]  9%|▊         | 34674/400000 [00:04<00:48, 7469.81it/s]  9%|▉         | 35435/400000 [00:04<00:48, 7510.85it/s]  9%|▉         | 36189/400000 [00:04<00:49, 7310.91it/s]  9%|▉         | 36973/400000 [00:04<00:48, 7460.76it/s]  9%|▉         | 37840/400000 [00:04<00:46, 7786.50it/s] 10%|▉         | 38778/400000 [00:04<00:44, 8203.30it/s] 10%|▉         | 39730/400000 [00:04<00:42, 8557.33it/s] 10%|█         | 40600/400000 [00:04<00:41, 8597.96it/s] 10%|█         | 41468/400000 [00:05<00:42, 8531.42it/s] 11%|█         | 42327/400000 [00:05<00:42, 8386.99it/s] 11%|█         | 43171/400000 [00:05<00:43, 8292.11it/s] 11%|█         | 44004/400000 [00:05<00:43, 8203.94it/s] 11%|█         | 44827/400000 [00:05<00:44, 7901.62it/s] 11%|█▏        | 45622/400000 [00:05<00:45, 7779.55it/s] 12%|█▏        | 46404/400000 [00:05<00:46, 7567.37it/s] 12%|█▏        | 47183/400000 [00:05<00:46, 7629.92it/s] 12%|█▏        | 47952/400000 [00:05<00:46, 7647.14it/s] 12%|█▏        | 48814/400000 [00:05<00:44, 7914.15it/s] 12%|█▏        | 49682/400000 [00:06<00:43, 8129.06it/s] 13%|█▎        | 50591/400000 [00:06<00:41, 8394.25it/s] 13%|█▎        | 51538/400000 [00:06<00:40, 8687.41it/s] 13%|█▎        | 52413/400000 [00:06<00:40, 8608.02it/s] 13%|█▎        | 53279/400000 [00:06<00:40, 8472.56it/s] 14%|█▎        | 54149/400000 [00:06<00:40, 8537.22it/s] 14%|█▍        | 55050/400000 [00:06<00:39, 8673.25it/s] 14%|█▍        | 55972/400000 [00:06<00:38, 8829.68it/s] 14%|█▍        | 56858/400000 [00:06<00:39, 8745.26it/s] 14%|█▍        | 57735/400000 [00:06<00:40, 8532.36it/s] 15%|█▍        | 58625/400000 [00:07<00:39, 8638.47it/s] 15%|█▍        | 59505/400000 [00:07<00:39, 8684.84it/s] 15%|█▌        | 60440/400000 [00:07<00:38, 8873.70it/s] 15%|█▌        | 61330/400000 [00:07<00:38, 8881.07it/s] 16%|█▌        | 62260/400000 [00:07<00:37, 9000.31it/s] 16%|█▌        | 63163/400000 [00:07<00:37, 9008.77it/s] 16%|█▌        | 64065/400000 [00:07<00:38, 8797.16it/s] 16%|█▌        | 64947/400000 [00:07<00:38, 8734.97it/s] 16%|█▋        | 65880/400000 [00:07<00:37, 8904.46it/s] 17%|█▋        | 66779/400000 [00:08<00:37, 8928.60it/s] 17%|█▋        | 67674/400000 [00:08<00:37, 8782.51it/s] 17%|█▋        | 68577/400000 [00:08<00:37, 8853.15it/s] 17%|█▋        | 69464/400000 [00:08<00:37, 8720.61it/s] 18%|█▊        | 70338/400000 [00:08<00:38, 8554.55it/s] 18%|█▊        | 71300/400000 [00:08<00:37, 8844.39it/s] 18%|█▊        | 72189/400000 [00:08<00:37, 8729.61it/s] 18%|█▊        | 73065/400000 [00:08<00:37, 8731.17it/s] 18%|█▊        | 73961/400000 [00:08<00:37, 8796.47it/s] 19%|█▊        | 74843/400000 [00:08<00:38, 8526.62it/s] 19%|█▉        | 75699/400000 [00:09<00:38, 8505.90it/s] 19%|█▉        | 76666/400000 [00:09<00:36, 8822.87it/s] 19%|█▉        | 77599/400000 [00:09<00:35, 8968.81it/s] 20%|█▉        | 78530/400000 [00:09<00:35, 9067.03it/s] 20%|█▉        | 79440/400000 [00:09<00:37, 8658.45it/s] 20%|██        | 80338/400000 [00:09<00:36, 8749.03it/s] 20%|██        | 81218/400000 [00:09<00:36, 8669.74it/s] 21%|██        | 82138/400000 [00:09<00:36, 8822.15it/s] 21%|██        | 83084/400000 [00:09<00:35, 9003.05it/s] 21%|██        | 83988/400000 [00:09<00:35, 8864.03it/s] 21%|██        | 84877/400000 [00:10<00:36, 8688.33it/s] 21%|██▏       | 85749/400000 [00:10<00:36, 8673.43it/s] 22%|██▏       | 86635/400000 [00:10<00:35, 8727.30it/s] 22%|██▏       | 87510/400000 [00:10<00:35, 8700.76it/s] 22%|██▏       | 88382/400000 [00:10<00:36, 8468.90it/s] 22%|██▏       | 89244/400000 [00:10<00:36, 8511.37it/s] 23%|██▎       | 90097/400000 [00:10<00:37, 8354.71it/s] 23%|██▎       | 90973/400000 [00:10<00:36, 8472.24it/s] 23%|██▎       | 91853/400000 [00:10<00:35, 8564.98it/s] 23%|██▎       | 92711/400000 [00:10<00:36, 8405.51it/s] 23%|██▎       | 93554/400000 [00:11<00:36, 8405.22it/s] 24%|██▎       | 94459/400000 [00:11<00:35, 8588.59it/s] 24%|██▍       | 95364/400000 [00:11<00:34, 8720.74it/s] 24%|██▍       | 96266/400000 [00:11<00:34, 8805.06it/s] 24%|██▍       | 97148/400000 [00:11<00:35, 8602.86it/s] 25%|██▍       | 98026/400000 [00:11<00:34, 8655.11it/s] 25%|██▍       | 98893/400000 [00:11<00:34, 8614.25it/s] 25%|██▍       | 99833/400000 [00:11<00:33, 8834.67it/s] 25%|██▌       | 100719/400000 [00:11<00:33, 8814.72it/s] 25%|██▌       | 101602/400000 [00:12<00:34, 8599.50it/s] 26%|██▌       | 102474/400000 [00:12<00:34, 8634.80it/s] 26%|██▌       | 103355/400000 [00:12<00:34, 8684.08it/s] 26%|██▌       | 104244/400000 [00:12<00:33, 8743.87it/s] 26%|██▋       | 105137/400000 [00:12<00:33, 8796.74it/s] 27%|██▋       | 106018/400000 [00:12<00:34, 8550.22it/s] 27%|██▋       | 106908/400000 [00:12<00:33, 8650.07it/s] 27%|██▋       | 107775/400000 [00:12<00:34, 8519.98it/s] 27%|██▋       | 108715/400000 [00:12<00:33, 8764.56it/s] 27%|██▋       | 109595/400000 [00:12<00:33, 8686.59it/s] 28%|██▊       | 110466/400000 [00:13<00:34, 8443.25it/s] 28%|██▊       | 111314/400000 [00:13<00:34, 8328.31it/s] 28%|██▊       | 112194/400000 [00:13<00:34, 8462.75it/s] 28%|██▊       | 113043/400000 [00:13<00:34, 8428.73it/s] 29%|██▊       | 114006/400000 [00:13<00:32, 8755.51it/s] 29%|██▊       | 114886/400000 [00:13<00:33, 8455.99it/s] 29%|██▉       | 115737/400000 [00:13<00:33, 8431.70it/s] 29%|██▉       | 116584/400000 [00:13<00:33, 8395.14it/s] 29%|██▉       | 117434/400000 [00:13<00:33, 8424.06it/s] 30%|██▉       | 118378/400000 [00:13<00:32, 8703.07it/s] 30%|██▉       | 119290/400000 [00:14<00:31, 8823.83it/s] 30%|███       | 120272/400000 [00:14<00:30, 9100.24it/s] 30%|███       | 121187/400000 [00:14<00:31, 8891.65it/s] 31%|███       | 122081/400000 [00:14<00:31, 8716.20it/s] 31%|███       | 122960/400000 [00:14<00:31, 8736.74it/s] 31%|███       | 123854/400000 [00:14<00:31, 8794.85it/s] 31%|███       | 124736/400000 [00:14<00:31, 8658.21it/s] 31%|███▏      | 125616/400000 [00:14<00:31, 8699.99it/s] 32%|███▏      | 126550/400000 [00:14<00:30, 8879.96it/s] 32%|███▏      | 127440/400000 [00:14<00:31, 8576.26it/s] 32%|███▏      | 128302/400000 [00:15<00:31, 8545.00it/s] 32%|███▏      | 129219/400000 [00:15<00:31, 8721.00it/s] 33%|███▎      | 130154/400000 [00:15<00:30, 8898.35it/s] 33%|███▎      | 131107/400000 [00:15<00:29, 9077.05it/s] 33%|███▎      | 132055/400000 [00:15<00:29, 9193.72it/s] 33%|███▎      | 132977/400000 [00:15<00:30, 8716.21it/s] 33%|███▎      | 133856/400000 [00:15<00:30, 8730.48it/s] 34%|███▎      | 134795/400000 [00:15<00:29, 8918.05it/s] 34%|███▍      | 135734/400000 [00:15<00:29, 9053.65it/s] 34%|███▍      | 136643/400000 [00:16<00:30, 8702.94it/s] 34%|███▍      | 137519/400000 [00:16<00:30, 8554.73it/s] 35%|███▍      | 138417/400000 [00:16<00:30, 8676.04it/s] 35%|███▍      | 139353/400000 [00:16<00:29, 8868.87it/s] 35%|███▌      | 140252/400000 [00:16<00:29, 8904.06it/s] 35%|███▌      | 141145/400000 [00:16<00:29, 8632.77it/s] 36%|███▌      | 142012/400000 [00:16<00:29, 8599.90it/s] 36%|███▌      | 142931/400000 [00:16<00:29, 8767.38it/s] 36%|███▌      | 143876/400000 [00:16<00:28, 8959.37it/s] 36%|███▌      | 144808/400000 [00:16<00:28, 9062.19it/s] 36%|███▋      | 145717/400000 [00:17<00:29, 8754.89it/s] 37%|███▋      | 146649/400000 [00:17<00:28, 8914.72it/s] 37%|███▋      | 147563/400000 [00:17<00:28, 8978.75it/s] 37%|███▋      | 148499/400000 [00:17<00:27, 9088.18it/s] 37%|███▋      | 149410/400000 [00:17<00:27, 8987.88it/s] 38%|███▊      | 150311/400000 [00:17<00:29, 8583.21it/s] 38%|███▊      | 151175/400000 [00:17<00:29, 8568.74it/s] 38%|███▊      | 152079/400000 [00:17<00:28, 8704.56it/s] 38%|███▊      | 153002/400000 [00:17<00:27, 8855.58it/s] 38%|███▊      | 153913/400000 [00:17<00:27, 8929.50it/s] 39%|███▊      | 154809/400000 [00:18<00:28, 8651.44it/s] 39%|███▉      | 155695/400000 [00:18<00:28, 8711.97it/s] 39%|███▉      | 156586/400000 [00:18<00:27, 8770.27it/s] 39%|███▉      | 157465/400000 [00:18<00:27, 8769.38it/s] 40%|███▉      | 158344/400000 [00:18<00:27, 8731.30it/s] 40%|███▉      | 159219/400000 [00:18<00:28, 8562.84it/s] 40%|████      | 160158/400000 [00:18<00:27, 8791.66it/s] 40%|████      | 161040/400000 [00:18<00:27, 8739.25it/s] 40%|████      | 161916/400000 [00:18<00:27, 8576.06it/s] 41%|████      | 162776/400000 [00:19<00:28, 8330.15it/s] 41%|████      | 163612/400000 [00:19<00:28, 8206.12it/s] 41%|████      | 164464/400000 [00:19<00:28, 8295.65it/s] 41%|████▏     | 165296/400000 [00:19<00:28, 8215.68it/s] 42%|████▏     | 166120/400000 [00:19<00:28, 8177.29it/s] 42%|████▏     | 166996/400000 [00:19<00:27, 8342.05it/s] 42%|████▏     | 167865/400000 [00:19<00:27, 8442.88it/s] 42%|████▏     | 168772/400000 [00:19<00:26, 8620.39it/s] 42%|████▏     | 169636/400000 [00:19<00:26, 8585.73it/s] 43%|████▎     | 170567/400000 [00:19<00:26, 8788.06it/s] 43%|████▎     | 171471/400000 [00:20<00:25, 8859.44it/s] 43%|████▎     | 172359/400000 [00:20<00:25, 8806.79it/s] 43%|████▎     | 173243/400000 [00:20<00:25, 8815.99it/s] 44%|████▎     | 174126/400000 [00:20<00:25, 8788.82it/s] 44%|████▍     | 175010/400000 [00:20<00:25, 8800.44it/s] 44%|████▍     | 175897/400000 [00:20<00:25, 8819.74it/s] 44%|████▍     | 176780/400000 [00:20<00:25, 8744.52it/s] 44%|████▍     | 177672/400000 [00:20<00:25, 8796.30it/s] 45%|████▍     | 178592/400000 [00:20<00:24, 8910.99it/s] 45%|████▍     | 179484/400000 [00:20<00:25, 8630.20it/s] 45%|████▌     | 180350/400000 [00:21<00:25, 8595.73it/s] 45%|████▌     | 181212/400000 [00:21<00:25, 8545.20it/s] 46%|████▌     | 182078/400000 [00:21<00:25, 8578.76it/s] 46%|████▌     | 182937/400000 [00:21<00:25, 8549.55it/s] 46%|████▌     | 183855/400000 [00:21<00:24, 8728.97it/s] 46%|████▌     | 184779/400000 [00:21<00:24, 8874.01it/s] 46%|████▋     | 185668/400000 [00:21<00:24, 8688.12it/s] 47%|████▋     | 186542/400000 [00:21<00:24, 8701.73it/s] 47%|████▋     | 187448/400000 [00:21<00:24, 8805.41it/s] 47%|████▋     | 188330/400000 [00:21<00:24, 8728.54it/s] 47%|████▋     | 189220/400000 [00:22<00:24, 8777.62it/s] 48%|████▊     | 190099/400000 [00:22<00:24, 8575.36it/s] 48%|████▊     | 191000/400000 [00:22<00:24, 8699.69it/s] 48%|████▊     | 191900/400000 [00:22<00:23, 8785.37it/s] 48%|████▊     | 192780/400000 [00:22<00:24, 8490.82it/s] 48%|████▊     | 193633/400000 [00:22<00:24, 8402.02it/s] 49%|████▊     | 194476/400000 [00:22<00:24, 8285.45it/s] 49%|████▉     | 195341/400000 [00:22<00:24, 8389.19it/s] 49%|████▉     | 196284/400000 [00:22<00:23, 8674.99it/s] 49%|████▉     | 197229/400000 [00:23<00:22, 8891.44it/s] 50%|████▉     | 198123/400000 [00:23<00:23, 8635.05it/s] 50%|████▉     | 198991/400000 [00:23<00:23, 8554.62it/s] 50%|████▉     | 199850/400000 [00:23<00:23, 8485.03it/s] 50%|█████     | 200701/400000 [00:23<00:24, 8123.19it/s] 50%|█████     | 201519/400000 [00:23<00:24, 8104.79it/s] 51%|█████     | 202404/400000 [00:23<00:23, 8314.22it/s] 51%|█████     | 203261/400000 [00:23<00:23, 8388.68it/s] 51%|█████     | 204108/400000 [00:23<00:23, 8412.89it/s] 51%|█████     | 204952/400000 [00:23<00:23, 8383.45it/s] 51%|█████▏    | 205818/400000 [00:24<00:22, 8462.54it/s] 52%|█████▏    | 206666/400000 [00:24<00:23, 8295.20it/s] 52%|█████▏    | 207498/400000 [00:24<00:23, 8286.77it/s] 52%|█████▏    | 208328/400000 [00:24<00:23, 8160.85it/s] 52%|█████▏    | 209179/400000 [00:24<00:23, 8260.25it/s] 53%|█████▎    | 210066/400000 [00:24<00:22, 8431.75it/s] 53%|█████▎    | 210941/400000 [00:24<00:22, 8523.87it/s] 53%|█████▎    | 211865/400000 [00:24<00:21, 8724.44it/s] 53%|█████▎    | 212812/400000 [00:24<00:20, 8933.38it/s] 53%|█████▎    | 213709/400000 [00:24<00:20, 8940.23it/s] 54%|█████▎    | 214606/400000 [00:25<00:20, 8948.81it/s] 54%|█████▍    | 215503/400000 [00:25<00:20, 8886.83it/s] 54%|█████▍    | 216396/400000 [00:25<00:20, 8898.21it/s] 54%|█████▍    | 217360/400000 [00:25<00:20, 9107.47it/s] 55%|█████▍    | 218273/400000 [00:25<00:20, 8941.48it/s] 55%|█████▍    | 219170/400000 [00:25<00:20, 8912.11it/s] 55%|█████▌    | 220063/400000 [00:25<00:20, 8820.90it/s] 55%|█████▌    | 220947/400000 [00:25<00:20, 8759.58it/s] 55%|█████▌    | 221824/400000 [00:25<00:20, 8714.45it/s] 56%|█████▌    | 222697/400000 [00:25<00:20, 8508.16it/s] 56%|█████▌    | 223550/400000 [00:26<00:20, 8511.06it/s] 56%|█████▌    | 224403/400000 [00:26<00:21, 8350.51it/s] 56%|█████▋    | 225269/400000 [00:26<00:20, 8440.00it/s] 57%|█████▋    | 226209/400000 [00:26<00:19, 8706.70it/s] 57%|█████▋    | 227083/400000 [00:26<00:20, 8604.76it/s] 57%|█████▋    | 227980/400000 [00:26<00:19, 8709.17it/s] 57%|█████▋    | 228853/400000 [00:26<00:20, 8412.85it/s] 57%|█████▋    | 229708/400000 [00:26<00:20, 8451.24it/s] 58%|█████▊    | 230556/400000 [00:26<00:20, 8341.54it/s] 58%|█████▊    | 231457/400000 [00:27<00:19, 8529.72it/s] 58%|█████▊    | 232329/400000 [00:27<00:19, 8584.53it/s] 58%|█████▊    | 233213/400000 [00:27<00:19, 8658.67it/s] 59%|█████▊    | 234119/400000 [00:27<00:18, 8773.91it/s] 59%|█████▊    | 234998/400000 [00:27<00:18, 8719.12it/s] 59%|█████▉    | 235871/400000 [00:27<00:18, 8684.90it/s] 59%|█████▉    | 236741/400000 [00:27<00:18, 8644.96it/s] 59%|█████▉    | 237612/400000 [00:27<00:18, 8663.29it/s] 60%|█████▉    | 238558/400000 [00:27<00:18, 8887.12it/s] 60%|█████▉    | 239506/400000 [00:27<00:17, 9054.23it/s] 60%|██████    | 240414/400000 [00:28<00:18, 8830.43it/s] 60%|██████    | 241347/400000 [00:28<00:17, 8972.97it/s] 61%|██████    | 242258/400000 [00:28<00:17, 9010.46it/s] 61%|██████    | 243161/400000 [00:28<00:17, 8919.13it/s] 61%|██████    | 244116/400000 [00:28<00:17, 9096.49it/s] 61%|██████▏   | 245028/400000 [00:28<00:17, 8926.55it/s] 61%|██████▏   | 245923/400000 [00:28<00:18, 8558.86it/s] 62%|██████▏   | 246784/400000 [00:28<00:18, 8396.96it/s] 62%|██████▏   | 247701/400000 [00:28<00:17, 8612.42it/s] 62%|██████▏   | 248567/400000 [00:28<00:17, 8545.83it/s] 62%|██████▏   | 249425/400000 [00:29<00:17, 8534.06it/s] 63%|██████▎   | 250287/400000 [00:29<00:17, 8555.76it/s] 63%|██████▎   | 251155/400000 [00:29<00:17, 8590.33it/s] 63%|██████▎   | 252053/400000 [00:29<00:16, 8702.98it/s] 63%|██████▎   | 252955/400000 [00:29<00:16, 8793.45it/s] 63%|██████▎   | 253894/400000 [00:29<00:16, 8964.12it/s] 64%|██████▎   | 254847/400000 [00:29<00:15, 9124.38it/s] 64%|██████▍   | 255762/400000 [00:29<00:15, 9035.91it/s] 64%|██████▍   | 256700/400000 [00:29<00:15, 9136.06it/s] 64%|██████▍   | 257630/400000 [00:29<00:15, 9184.13it/s] 65%|██████▍   | 258550/400000 [00:30<00:15, 9091.52it/s] 65%|██████▍   | 259461/400000 [00:30<00:15, 9042.98it/s] 65%|██████▌   | 260366/400000 [00:30<00:15, 8856.28it/s] 65%|██████▌   | 261324/400000 [00:30<00:15, 9059.21it/s] 66%|██████▌   | 262254/400000 [00:30<00:15, 9129.33it/s] 66%|██████▌   | 263185/400000 [00:30<00:14, 9181.48it/s] 66%|██████▌   | 264112/400000 [00:30<00:14, 9207.56it/s] 66%|██████▋   | 265034/400000 [00:30<00:15, 8952.42it/s] 66%|██████▋   | 265946/400000 [00:30<00:14, 8999.82it/s] 67%|██████▋   | 266848/400000 [00:30<00:14, 8925.02it/s] 67%|██████▋   | 267742/400000 [00:31<00:15, 8779.93it/s] 67%|██████▋   | 268628/400000 [00:31<00:14, 8801.50it/s] 67%|██████▋   | 269510/400000 [00:31<00:15, 8468.04it/s] 68%|██████▊   | 270419/400000 [00:31<00:14, 8644.96it/s] 68%|██████▊   | 271317/400000 [00:31<00:14, 8740.73it/s] 68%|██████▊   | 272206/400000 [00:31<00:14, 8782.56it/s] 68%|██████▊   | 273087/400000 [00:31<00:14, 8779.72it/s] 68%|██████▊   | 273967/400000 [00:31<00:14, 8769.10it/s] 69%|██████▊   | 274904/400000 [00:31<00:13, 8939.93it/s] 69%|██████▉   | 275819/400000 [00:32<00:13, 8999.48it/s] 69%|██████▉   | 276721/400000 [00:32<00:13, 8909.26it/s] 69%|██████▉   | 277613/400000 [00:32<00:13, 8836.67it/s] 70%|██████▉   | 278498/400000 [00:32<00:13, 8808.09it/s] 70%|██████▉   | 279380/400000 [00:32<00:13, 8730.60it/s] 70%|███████   | 280277/400000 [00:32<00:13, 8800.18it/s] 70%|███████   | 281204/400000 [00:32<00:13, 8934.26it/s] 71%|███████   | 282146/400000 [00:32<00:12, 9072.81it/s] 71%|███████   | 283055/400000 [00:32<00:13, 8801.55it/s] 71%|███████   | 283946/400000 [00:32<00:13, 8828.62it/s] 71%|███████   | 284835/400000 [00:33<00:13, 8846.69it/s] 71%|███████▏  | 285780/400000 [00:33<00:12, 9018.23it/s] 72%|███████▏  | 286684/400000 [00:33<00:12, 9006.25it/s] 72%|███████▏  | 287586/400000 [00:33<00:12, 8839.03it/s] 72%|███████▏  | 288518/400000 [00:33<00:12, 8976.02it/s] 72%|███████▏  | 289418/400000 [00:33<00:12, 8846.19it/s] 73%|███████▎  | 290348/400000 [00:33<00:12, 8977.32it/s] 73%|███████▎  | 291266/400000 [00:33<00:12, 9034.99it/s] 73%|███████▎  | 292171/400000 [00:33<00:12, 8852.75it/s] 73%|███████▎  | 293058/400000 [00:33<00:12, 8689.74it/s] 73%|███████▎  | 293962/400000 [00:34<00:12, 8791.34it/s] 74%|███████▎  | 294846/400000 [00:34<00:11, 8804.53it/s] 74%|███████▍  | 295728/400000 [00:34<00:11, 8744.46it/s] 74%|███████▍  | 296604/400000 [00:34<00:11, 8722.86it/s] 74%|███████▍  | 297482/400000 [00:34<00:11, 8737.77it/s] 75%|███████▍  | 298357/400000 [00:34<00:11, 8580.81it/s] 75%|███████▍  | 299226/400000 [00:34<00:11, 8611.05it/s] 75%|███████▌  | 300139/400000 [00:34<00:11, 8760.27it/s] 75%|███████▌  | 301017/400000 [00:34<00:11, 8623.46it/s] 75%|███████▌  | 301975/400000 [00:34<00:11, 8887.04it/s] 76%|███████▌  | 302885/400000 [00:35<00:10, 8949.77it/s] 76%|███████▌  | 303783/400000 [00:35<00:10, 8812.76it/s] 76%|███████▌  | 304667/400000 [00:35<00:10, 8771.48it/s] 76%|███████▋  | 305546/400000 [00:35<00:10, 8678.73it/s] 77%|███████▋  | 306530/400000 [00:35<00:10, 8994.89it/s] 77%|███████▋  | 307485/400000 [00:35<00:10, 9152.40it/s] 77%|███████▋  | 308408/400000 [00:35<00:09, 9174.19it/s] 77%|███████▋  | 309339/400000 [00:35<00:09, 9213.60it/s] 78%|███████▊  | 310263/400000 [00:35<00:09, 9006.61it/s] 78%|███████▊  | 311166/400000 [00:35<00:10, 8871.98it/s] 78%|███████▊  | 312108/400000 [00:36<00:09, 9028.90it/s] 78%|███████▊  | 313013/400000 [00:36<00:09, 8864.28it/s] 78%|███████▊  | 313908/400000 [00:36<00:09, 8887.49it/s] 79%|███████▊  | 314799/400000 [00:36<00:09, 8688.47it/s] 79%|███████▉  | 315670/400000 [00:36<00:09, 8645.94it/s] 79%|███████▉  | 316537/400000 [00:36<00:09, 8422.43it/s] 79%|███████▉  | 317463/400000 [00:36<00:09, 8655.13it/s] 80%|███████▉  | 318408/400000 [00:36<00:09, 8879.00it/s] 80%|███████▉  | 319300/400000 [00:36<00:09, 8792.32it/s] 80%|████████  | 320231/400000 [00:37<00:08, 8939.49it/s] 80%|████████  | 321128/400000 [00:37<00:08, 8834.30it/s] 81%|████████  | 322051/400000 [00:37<00:08, 8947.13it/s] 81%|████████  | 322957/400000 [00:37<00:08, 8979.25it/s] 81%|████████  | 323857/400000 [00:37<00:08, 8884.01it/s] 81%|████████  | 324749/400000 [00:37<00:08, 8894.16it/s] 81%|████████▏ | 325670/400000 [00:37<00:08, 8984.59it/s] 82%|████████▏ | 326570/400000 [00:37<00:08, 8896.60it/s] 82%|████████▏ | 327470/400000 [00:37<00:08, 8927.14it/s] 82%|████████▏ | 328364/400000 [00:37<00:08, 8851.48it/s] 82%|████████▏ | 329250/400000 [00:38<00:08, 8834.57it/s] 83%|████████▎ | 330157/400000 [00:38<00:07, 8903.66it/s] 83%|████████▎ | 331074/400000 [00:38<00:07, 8980.49it/s] 83%|████████▎ | 332034/400000 [00:38<00:07, 9156.34it/s] 83%|████████▎ | 332951/400000 [00:38<00:07, 9023.16it/s] 83%|████████▎ | 333855/400000 [00:38<00:07, 8873.89it/s] 84%|████████▎ | 334746/400000 [00:38<00:07, 8880.03it/s] 84%|████████▍ | 335635/400000 [00:38<00:07, 8841.81it/s] 84%|████████▍ | 336562/400000 [00:38<00:07, 8965.90it/s] 84%|████████▍ | 337460/400000 [00:38<00:07, 8917.55it/s] 85%|████████▍ | 338401/400000 [00:39<00:06, 9059.17it/s] 85%|████████▍ | 339308/400000 [00:39<00:06, 8967.62it/s] 85%|████████▌ | 340206/400000 [00:39<00:06, 8834.90it/s] 85%|████████▌ | 341138/400000 [00:39<00:06, 8974.91it/s] 86%|████████▌ | 342037/400000 [00:39<00:06, 8684.61it/s] 86%|████████▌ | 342970/400000 [00:39<00:06, 8867.03it/s] 86%|████████▌ | 343917/400000 [00:39<00:06, 9037.49it/s] 86%|████████▌ | 344835/400000 [00:39<00:06, 9077.34it/s] 86%|████████▋ | 345778/400000 [00:39<00:05, 9178.35it/s] 87%|████████▋ | 346698/400000 [00:39<00:05, 8927.29it/s] 87%|████████▋ | 347624/400000 [00:40<00:05, 9022.86it/s] 87%|████████▋ | 348555/400000 [00:40<00:05, 9104.33it/s] 87%|████████▋ | 349468/400000 [00:40<00:05, 9021.29it/s] 88%|████████▊ | 350393/400000 [00:40<00:05, 9086.24it/s] 88%|████████▊ | 351303/400000 [00:40<00:05, 8852.93it/s] 88%|████████▊ | 352197/400000 [00:40<00:05, 8876.90it/s] 88%|████████▊ | 353087/400000 [00:40<00:05, 8617.39it/s] 88%|████████▊ | 353952/400000 [00:40<00:05, 8372.07it/s] 89%|████████▊ | 354793/400000 [00:40<00:05, 8271.77it/s] 89%|████████▉ | 355623/400000 [00:41<00:05, 8178.14it/s] 89%|████████▉ | 356500/400000 [00:41<00:05, 8346.42it/s] 89%|████████▉ | 357341/400000 [00:41<00:05, 8364.54it/s] 90%|████████▉ | 358239/400000 [00:41<00:04, 8539.85it/s] 90%|████████▉ | 359096/400000 [00:41<00:04, 8477.87it/s] 90%|████████▉ | 359946/400000 [00:41<00:04, 8454.19it/s] 90%|█████████ | 360813/400000 [00:41<00:04, 8516.16it/s] 90%|█████████ | 361712/400000 [00:41<00:04, 8652.18it/s] 91%|█████████ | 362606/400000 [00:41<00:04, 8736.45it/s] 91%|█████████ | 363539/400000 [00:41<00:04, 8905.74it/s] 91%|█████████ | 364432/400000 [00:42<00:04, 8577.53it/s] 91%|█████████▏| 365371/400000 [00:42<00:03, 8804.40it/s] 92%|█████████▏| 366337/400000 [00:42<00:03, 9043.28it/s] 92%|█████████▏| 367246/400000 [00:42<00:03, 8985.22it/s] 92%|█████████▏| 368148/400000 [00:42<00:03, 8955.83it/s] 92%|█████████▏| 369046/400000 [00:42<00:03, 8740.87it/s] 92%|█████████▏| 369923/400000 [00:42<00:03, 8669.33it/s] 93%|█████████▎| 370792/400000 [00:42<00:03, 8576.16it/s] 93%|█████████▎| 371652/400000 [00:42<00:03, 8482.30it/s] 93%|█████████▎| 372523/400000 [00:42<00:03, 8546.90it/s] 93%|█████████▎| 373433/400000 [00:43<00:03, 8704.81it/s] 94%|█████████▎| 374356/400000 [00:43<00:02, 8854.53it/s] 94%|█████████▍| 375244/400000 [00:43<00:02, 8825.59it/s] 94%|█████████▍| 376140/400000 [00:43<00:02, 8863.00it/s] 94%|█████████▍| 377067/400000 [00:43<00:02, 8978.94it/s] 94%|█████████▍| 377966/400000 [00:43<00:02, 8821.61it/s] 95%|█████████▍| 378850/400000 [00:43<00:02, 8801.47it/s] 95%|█████████▍| 379732/400000 [00:43<00:02, 8706.99it/s] 95%|█████████▌| 380604/400000 [00:43<00:02, 8646.67it/s] 95%|█████████▌| 381572/400000 [00:43<00:02, 8930.49it/s] 96%|█████████▌| 382468/400000 [00:44<00:01, 8793.10it/s] 96%|█████████▌| 383365/400000 [00:44<00:01, 8843.67it/s] 96%|█████████▌| 384252/400000 [00:44<00:01, 8672.63it/s] 96%|█████████▋| 385141/400000 [00:44<00:01, 8735.08it/s] 97%|█████████▋| 386025/400000 [00:44<00:01, 8764.99it/s] 97%|█████████▋| 386903/400000 [00:44<00:01, 8575.35it/s] 97%|█████████▋| 387878/400000 [00:44<00:01, 8894.03it/s] 97%|█████████▋| 388784/400000 [00:44<00:01, 8942.44it/s] 97%|█████████▋| 389725/400000 [00:44<00:01, 9075.38it/s] 98%|█████████▊| 390663/400000 [00:45<00:01, 9164.25it/s] 98%|█████████▊| 391582/400000 [00:45<00:00, 8884.87it/s] 98%|█████████▊| 392532/400000 [00:45<00:00, 9058.70it/s] 98%|█████████▊| 393469/400000 [00:45<00:00, 9148.78it/s] 99%|█████████▊| 394387/400000 [00:45<00:00, 9118.44it/s] 99%|█████████▉| 395301/400000 [00:45<00:00, 9029.99it/s] 99%|█████████▉| 396206/400000 [00:45<00:00, 8710.31it/s] 99%|█████████▉| 397081/400000 [00:45<00:00, 8712.74it/s] 99%|█████████▉| 397955/400000 [00:45<00:00, 8501.95it/s]100%|█████████▉| 398818/400000 [00:45<00:00, 8537.72it/s]100%|█████████▉| 399715/400000 [00:46<00:00, 8660.34it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8680.98it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe43ef54be0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011130592938997545 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.01127979428473125 	 Accuracy: 52

  model saves at 52% accuracy 

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

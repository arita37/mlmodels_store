
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f74ea89df98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 06:13:58.397885
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 06:13:58.402255
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 06:13:58.405988
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 06:13:58.409892
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f74f6661470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357408.3125
Epoch 2/10

1/1 [==============================] - 0s 112ms/step - loss: 308573.3750
Epoch 3/10

1/1 [==============================] - 0s 114ms/step - loss: 240514.2188
Epoch 4/10

1/1 [==============================] - 0s 109ms/step - loss: 176104.8750
Epoch 5/10

1/1 [==============================] - 0s 108ms/step - loss: 119064.7500
Epoch 6/10

1/1 [==============================] - 0s 114ms/step - loss: 76132.4922
Epoch 7/10

1/1 [==============================] - 0s 121ms/step - loss: 47344.6602
Epoch 8/10

1/1 [==============================] - 0s 106ms/step - loss: 30416.3164
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 20732.4844
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 14975.0498

  #### Inference Need return ypred, ytrue ######################### 
[[-0.6586721  -0.7106016   0.30384743  0.4495679   0.63513744 -0.09758662
  -0.09171191  1.0238833  -0.14316711  0.99349463  0.45654583 -0.9711599
  -0.26694864  0.0781281   0.44732487 -0.6131529  -0.264623   -0.2571224
  -0.5003278   0.72880226 -1.286974   -0.3413366  -1.3399701  -0.3577755
   0.16746894  0.71368444 -0.6174742   0.8968551  -0.8618375   0.11872448
   0.5575986   0.13899025 -0.12407169 -0.45705676  0.3030001  -0.16978428
  -0.99953645 -0.5179879   0.17425719 -0.8635746  -0.5035149  -0.19297662
   0.22616917 -0.2611755   0.2539313   0.38744116 -0.23734516  1.0630376
   1.2303162  -0.5704343  -0.6455514  -0.06331147  0.27272925  0.09637482
  -0.01585391  0.46746096  1.3398911  -1.0867107   0.60883504 -0.29341447
  -1.4402637  -0.07652912 -0.5363585  -0.5771426  -0.49635008  0.49512073
  -0.19768298 -0.61280364 -0.28594318  0.55878156  0.18532945 -0.06081642
   0.50284714  0.39926484 -0.70340884  1.8405154   0.14167148  0.17004403
  -0.22307947  0.67107785  0.2722058   0.04237354  1.3475666   0.13260396
  -0.83082855  0.14697036  1.0921322   0.50684094 -0.4858447   0.1851041
  -1.0293875  -0.2258228   0.88336885  0.38307834  0.9651775   0.3916808
  -0.02160076 -0.08518264  1.0964372   0.6239639   0.5883825   0.5148297
  -0.01248818  0.90490323 -0.725546    0.27048084 -0.6946924   0.01272629
   0.19235416  0.5722492   0.12128459 -0.1936928  -0.25473586 -0.4833906
  -0.08181038  0.2044057   0.10341024  0.22466558 -0.5558853  -0.49713907
   0.10100265  5.277607    4.9913793   6.2159553   4.9343634   5.251333
   5.531483    5.2584376   4.931797    5.5887313   4.9491553   5.6603603
   6.17821     4.5019984   5.357213    4.7420483   5.4609647   5.5933995
   4.9848647   6.348236    5.012397    4.291131    4.446061    4.5348043
   4.7233496   6.1660542   3.7940042   5.3610506   5.892298    3.5818074
   5.908322    4.23382     5.6962013   5.7684894   3.6175802   4.7978992
   6.144047    6.590043    4.8977876   4.2834725   3.793882    4.977764
   4.6986446   4.8531084   4.147652    5.071148    5.2552357   5.997019
   4.816326    5.8611927   4.7611074   4.6197124   6.3340106   5.3309965
   4.733876    5.550009    4.434001    5.587805    5.391349    4.457532
   1.3697602   0.69024575  1.0528177   0.9074404   0.8154571   1.5099556
   2.017815    0.88467604  1.0452877   1.4288408   0.89854693  0.29904675
   0.42761403  0.8203833   0.54863876  1.1967838   1.1335611   0.30183822
   0.6211822   0.6856817   0.9291572   1.8827965   1.4555091   0.68161553
   0.60964245  2.3563335   0.8081973   0.88092726  1.3941258   0.42775667
   2.5189238   0.55937606  1.2544281   0.45152658  0.97739035  1.8247617
   1.4750948   0.6117924   0.5737836   1.9967484   1.0425966   1.2138004
   0.5135618   1.085455    0.45616913  0.38975847  1.0448726   0.88871783
   1.2553154   0.65296435  0.50218594  0.8748805   1.4962995   1.567703
   1.7272835   0.80659676  0.43762624  0.39865386  1.7173207   1.5266565
   0.94527423  0.47362268  0.58189535  1.6298892   1.0172609   0.5321931
   1.342217    1.2254747   1.6904728   1.0150102   0.6497735   0.8967495
   0.54678786  0.9927409   1.1413931   1.3278272   1.5456427   1.0311072
   2.2308278   0.3548892   0.43293273  2.2847192   2.2431068   1.0706365
   1.0507009   0.55964094  0.50435203  0.329347    0.6980702   1.0064154
   0.86370325  1.3885427   1.7259185   1.2317101   1.0523889   0.9410712
   1.8105794   1.8035741   0.5880562   0.7914251   0.84294474  0.5139425
   0.5844038   1.3970864   0.6227139   0.7247796   1.2479762   0.33235484
   0.4290164   0.41352886  2.22481     1.5233612   1.774796    0.25674736
   0.80133617  2.0994353   1.0591652   1.0001438   0.9488044   1.3575976
   0.0336864   5.084748    4.828634    6.541066    5.9570856   5.2113495
   5.582563    6.210556    4.8783817   5.556286    6.129514    5.361549
   5.697452    5.980413    6.106014    4.9794946   6.116129    5.5144963
   6.68754     5.4085627   6.2241206   4.8909063   5.815449    6.1343107
   4.9581013   5.8947115   5.812392    6.0785775   6.7766786   6.4843507
   5.957846    5.2190385   4.9306307   5.998604    4.9658327   5.0614142
   5.611787    4.313715    6.0998163   5.7193947   5.160599    5.0887413
   6.1341133   4.701297    4.5156317   5.326332    5.841127    5.448508
   5.5365605   5.8420205   6.112286    6.4152904   4.885709    6.057825
   5.561391    5.7581296   5.7933583   5.0433087   5.7430935   6.1608257
  -7.3534636  -3.4315352   3.3050752 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 06:14:08.213191
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.9018
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 06:14:08.217696
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9410.1
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 06:14:08.221526
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    97.809
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 06:14:08.225403
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -841.741
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140139768783032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140139095560432
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140139095560936
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140139095561440
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140139095561944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140139095562448

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f74f24f7518> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.524813
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.496463
grad_step = 000002, loss = 0.476342
grad_step = 000003, loss = 0.454168
grad_step = 000004, loss = 0.427159
grad_step = 000005, loss = 0.400205
grad_step = 000006, loss = 0.378646
grad_step = 000007, loss = 0.362012
grad_step = 000008, loss = 0.345840
grad_step = 000009, loss = 0.330557
grad_step = 000010, loss = 0.315087
grad_step = 000011, loss = 0.302566
grad_step = 000012, loss = 0.293058
grad_step = 000013, loss = 0.284596
grad_step = 000014, loss = 0.275718
grad_step = 000015, loss = 0.266037
grad_step = 000016, loss = 0.256013
grad_step = 000017, loss = 0.245924
grad_step = 000018, loss = 0.236178
grad_step = 000019, loss = 0.226625
grad_step = 000020, loss = 0.217306
grad_step = 000021, loss = 0.208237
grad_step = 000022, loss = 0.199199
grad_step = 000023, loss = 0.190363
grad_step = 000024, loss = 0.182127
grad_step = 000025, loss = 0.174496
grad_step = 000026, loss = 0.167171
grad_step = 000027, loss = 0.159866
grad_step = 000028, loss = 0.152487
grad_step = 000029, loss = 0.145222
grad_step = 000030, loss = 0.138201
grad_step = 000031, loss = 0.131278
grad_step = 000032, loss = 0.124476
grad_step = 000033, loss = 0.118004
grad_step = 000034, loss = 0.111894
grad_step = 000035, loss = 0.106044
grad_step = 000036, loss = 0.100406
grad_step = 000037, loss = 0.094967
grad_step = 000038, loss = 0.089637
grad_step = 000039, loss = 0.084518
grad_step = 000040, loss = 0.079608
grad_step = 000041, loss = 0.074861
grad_step = 000042, loss = 0.070372
grad_step = 000043, loss = 0.066106
grad_step = 000044, loss = 0.061974
grad_step = 000045, loss = 0.058031
grad_step = 000046, loss = 0.054341
grad_step = 000047, loss = 0.050815
grad_step = 000048, loss = 0.047427
grad_step = 000049, loss = 0.044238
grad_step = 000050, loss = 0.041219
grad_step = 000051, loss = 0.038342
grad_step = 000052, loss = 0.035620
grad_step = 000053, loss = 0.033063
grad_step = 000054, loss = 0.030632
grad_step = 000055, loss = 0.028356
grad_step = 000056, loss = 0.026220
grad_step = 000057, loss = 0.024182
grad_step = 000058, loss = 0.022291
grad_step = 000059, loss = 0.020511
grad_step = 000060, loss = 0.018827
grad_step = 000061, loss = 0.017265
grad_step = 000062, loss = 0.015796
grad_step = 000063, loss = 0.014450
grad_step = 000064, loss = 0.013194
grad_step = 000065, loss = 0.012024
grad_step = 000066, loss = 0.010952
grad_step = 000067, loss = 0.009960
grad_step = 000068, loss = 0.009054
grad_step = 000069, loss = 0.008221
grad_step = 000070, loss = 0.007462
grad_step = 000071, loss = 0.006781
grad_step = 000072, loss = 0.006161
grad_step = 000073, loss = 0.005613
grad_step = 000074, loss = 0.005117
grad_step = 000075, loss = 0.004684
grad_step = 000076, loss = 0.004295
grad_step = 000077, loss = 0.003959
grad_step = 000078, loss = 0.003660
grad_step = 000079, loss = 0.003405
grad_step = 000080, loss = 0.003187
grad_step = 000081, loss = 0.003005
grad_step = 000082, loss = 0.002848
grad_step = 000083, loss = 0.002718
grad_step = 000084, loss = 0.002611
grad_step = 000085, loss = 0.002525
grad_step = 000086, loss = 0.002453
grad_step = 000087, loss = 0.002398
grad_step = 000088, loss = 0.002355
grad_step = 000089, loss = 0.002329
grad_step = 000090, loss = 0.002322
grad_step = 000091, loss = 0.002332
grad_step = 000092, loss = 0.002327
grad_step = 000093, loss = 0.002299
grad_step = 000094, loss = 0.002261
grad_step = 000095, loss = 0.002231
grad_step = 000096, loss = 0.002224
grad_step = 000097, loss = 0.002234
grad_step = 000098, loss = 0.002246
grad_step = 000099, loss = 0.002245
grad_step = 000100, loss = 0.002228
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002202
grad_step = 000102, loss = 0.002180
grad_step = 000103, loss = 0.002167
grad_step = 000104, loss = 0.002164
grad_step = 000105, loss = 0.002165
grad_step = 000106, loss = 0.002164
grad_step = 000107, loss = 0.002158
grad_step = 000108, loss = 0.002146
grad_step = 000109, loss = 0.002127
grad_step = 000110, loss = 0.002105
grad_step = 000111, loss = 0.002083
grad_step = 000112, loss = 0.002064
grad_step = 000113, loss = 0.002048
grad_step = 000114, loss = 0.002034
grad_step = 000115, loss = 0.002023
grad_step = 000116, loss = 0.002020
grad_step = 000117, loss = 0.002029
grad_step = 000118, loss = 0.002056
grad_step = 000119, loss = 0.002055
grad_step = 000120, loss = 0.002022
grad_step = 000121, loss = 0.001991
grad_step = 000122, loss = 0.002017
grad_step = 000123, loss = 0.002118
grad_step = 000124, loss = 0.002208
grad_step = 000125, loss = 0.002228
grad_step = 000126, loss = 0.002127
grad_step = 000127, loss = 0.002120
grad_step = 000128, loss = 0.002115
grad_step = 000129, loss = 0.001960
grad_step = 000130, loss = 0.001924
grad_step = 000131, loss = 0.002064
grad_step = 000132, loss = 0.002051
grad_step = 000133, loss = 0.001934
grad_step = 000134, loss = 0.001923
grad_step = 000135, loss = 0.001970
grad_step = 000136, loss = 0.001949
grad_step = 000137, loss = 0.001892
grad_step = 000138, loss = 0.001939
grad_step = 000139, loss = 0.001971
grad_step = 000140, loss = 0.001886
grad_step = 000141, loss = 0.001849
grad_step = 000142, loss = 0.001903
grad_step = 000143, loss = 0.001926
grad_step = 000144, loss = 0.001888
grad_step = 000145, loss = 0.001841
grad_step = 000146, loss = 0.001843
grad_step = 000147, loss = 0.001889
grad_step = 000148, loss = 0.001891
grad_step = 000149, loss = 0.001863
grad_step = 000150, loss = 0.001823
grad_step = 000151, loss = 0.001810
grad_step = 000152, loss = 0.001824
grad_step = 000153, loss = 0.001843
grad_step = 000154, loss = 0.001873
grad_step = 000155, loss = 0.001882
grad_step = 000156, loss = 0.001878
grad_step = 000157, loss = 0.001854
grad_step = 000158, loss = 0.001831
grad_step = 000159, loss = 0.001821
grad_step = 000160, loss = 0.001821
grad_step = 000161, loss = 0.001826
grad_step = 000162, loss = 0.001818
grad_step = 000163, loss = 0.001802
grad_step = 000164, loss = 0.001787
grad_step = 000165, loss = 0.001778
grad_step = 000166, loss = 0.001781
grad_step = 000167, loss = 0.001792
grad_step = 000168, loss = 0.001806
grad_step = 000169, loss = 0.001819
grad_step = 000170, loss = 0.001824
grad_step = 000171, loss = 0.001820
grad_step = 000172, loss = 0.001807
grad_step = 000173, loss = 0.001792
grad_step = 000174, loss = 0.001786
grad_step = 000175, loss = 0.001821
grad_step = 000176, loss = 0.001912
grad_step = 000177, loss = 0.002129
grad_step = 000178, loss = 0.002027
grad_step = 000179, loss = 0.001843
grad_step = 000180, loss = 0.001749
grad_step = 000181, loss = 0.001889
grad_step = 000182, loss = 0.001928
grad_step = 000183, loss = 0.001747
grad_step = 000184, loss = 0.001812
grad_step = 000185, loss = 0.001914
grad_step = 000186, loss = 0.001762
grad_step = 000187, loss = 0.001761
grad_step = 000188, loss = 0.001865
grad_step = 000189, loss = 0.001762
grad_step = 000190, loss = 0.001732
grad_step = 000191, loss = 0.001809
grad_step = 000192, loss = 0.001756
grad_step = 000193, loss = 0.001722
grad_step = 000194, loss = 0.001775
grad_step = 000195, loss = 0.001749
grad_step = 000196, loss = 0.001714
grad_step = 000197, loss = 0.001742
grad_step = 000198, loss = 0.001743
grad_step = 000199, loss = 0.001709
grad_step = 000200, loss = 0.001717
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001732
grad_step = 000202, loss = 0.001709
grad_step = 000203, loss = 0.001700
grad_step = 000204, loss = 0.001715
grad_step = 000205, loss = 0.001712
grad_step = 000206, loss = 0.001694
grad_step = 000207, loss = 0.001697
grad_step = 000208, loss = 0.001707
grad_step = 000209, loss = 0.001699
grad_step = 000210, loss = 0.001687
grad_step = 000211, loss = 0.001690
grad_step = 000212, loss = 0.001697
grad_step = 000213, loss = 0.001692
grad_step = 000214, loss = 0.001683
grad_step = 000215, loss = 0.001681
grad_step = 000216, loss = 0.001684
grad_step = 000217, loss = 0.001685
grad_step = 000218, loss = 0.001680
grad_step = 000219, loss = 0.001675
grad_step = 000220, loss = 0.001675
grad_step = 000221, loss = 0.001678
grad_step = 000222, loss = 0.001679
grad_step = 000223, loss = 0.001679
grad_step = 000224, loss = 0.001683
grad_step = 000225, loss = 0.001707
grad_step = 000226, loss = 0.001777
grad_step = 000227, loss = 0.001964
grad_step = 000228, loss = 0.002378
grad_step = 000229, loss = 0.003097
grad_step = 000230, loss = 0.003268
grad_step = 000231, loss = 0.002444
grad_step = 000232, loss = 0.001692
grad_step = 000233, loss = 0.002131
grad_step = 000234, loss = 0.002578
grad_step = 000235, loss = 0.001977
grad_step = 000236, loss = 0.001848
grad_step = 000237, loss = 0.002315
grad_step = 000238, loss = 0.001970
grad_step = 000239, loss = 0.001836
grad_step = 000240, loss = 0.002153
grad_step = 000241, loss = 0.001844
grad_step = 000242, loss = 0.001860
grad_step = 000243, loss = 0.002026
grad_step = 000244, loss = 0.001786
grad_step = 000245, loss = 0.001894
grad_step = 000246, loss = 0.001870
grad_step = 000247, loss = 0.001815
grad_step = 000248, loss = 0.001780
grad_step = 000249, loss = 0.001824
grad_step = 000250, loss = 0.001801
grad_step = 000251, loss = 0.001724
grad_step = 000252, loss = 0.001816
grad_step = 000253, loss = 0.001760
grad_step = 000254, loss = 0.001719
grad_step = 000255, loss = 0.001761
grad_step = 000256, loss = 0.001734
grad_step = 000257, loss = 0.001664
grad_step = 000258, loss = 0.001749
grad_step = 000259, loss = 0.001660
grad_step = 000260, loss = 0.001692
grad_step = 000261, loss = 0.001684
grad_step = 000262, loss = 0.001667
grad_step = 000263, loss = 0.001660
grad_step = 000264, loss = 0.001654
grad_step = 000265, loss = 0.001663
grad_step = 000266, loss = 0.001627
grad_step = 000267, loss = 0.001662
grad_step = 000268, loss = 0.001626
grad_step = 000269, loss = 0.001636
grad_step = 000270, loss = 0.001626
grad_step = 000271, loss = 0.001634
grad_step = 000272, loss = 0.001615
grad_step = 000273, loss = 0.001625
grad_step = 000274, loss = 0.001617
grad_step = 000275, loss = 0.001608
grad_step = 000276, loss = 0.001615
grad_step = 000277, loss = 0.001607
grad_step = 000278, loss = 0.001605
grad_step = 000279, loss = 0.001600
grad_step = 000280, loss = 0.001605
grad_step = 000281, loss = 0.001592
grad_step = 000282, loss = 0.001597
grad_step = 000283, loss = 0.001592
grad_step = 000284, loss = 0.001591
grad_step = 000285, loss = 0.001587
grad_step = 000286, loss = 0.001586
grad_step = 000287, loss = 0.001581
grad_step = 000288, loss = 0.001578
grad_step = 000289, loss = 0.001580
grad_step = 000290, loss = 0.001575
grad_step = 000291, loss = 0.001575
grad_step = 000292, loss = 0.001574
grad_step = 000293, loss = 0.001578
grad_step = 000294, loss = 0.001576
grad_step = 000295, loss = 0.001581
grad_step = 000296, loss = 0.001588
grad_step = 000297, loss = 0.001597
grad_step = 000298, loss = 0.001618
grad_step = 000299, loss = 0.001631
grad_step = 000300, loss = 0.001646
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001610
grad_step = 000302, loss = 0.001579
grad_step = 000303, loss = 0.001548
grad_step = 000304, loss = 0.001551
grad_step = 000305, loss = 0.001572
grad_step = 000306, loss = 0.001576
grad_step = 000307, loss = 0.001560
grad_step = 000308, loss = 0.001537
grad_step = 000309, loss = 0.001541
grad_step = 000310, loss = 0.001554
grad_step = 000311, loss = 0.001548
grad_step = 000312, loss = 0.001532
grad_step = 000313, loss = 0.001526
grad_step = 000314, loss = 0.001531
grad_step = 000315, loss = 0.001534
grad_step = 000316, loss = 0.001530
grad_step = 000317, loss = 0.001522
grad_step = 000318, loss = 0.001516
grad_step = 000319, loss = 0.001516
grad_step = 000320, loss = 0.001519
grad_step = 000321, loss = 0.001521
grad_step = 000322, loss = 0.001519
grad_step = 000323, loss = 0.001517
grad_step = 000324, loss = 0.001516
grad_step = 000325, loss = 0.001515
grad_step = 000326, loss = 0.001515
grad_step = 000327, loss = 0.001518
grad_step = 000328, loss = 0.001525
grad_step = 000329, loss = 0.001541
grad_step = 000330, loss = 0.001593
grad_step = 000331, loss = 0.001647
grad_step = 000332, loss = 0.001761
grad_step = 000333, loss = 0.001658
grad_step = 000334, loss = 0.001575
grad_step = 000335, loss = 0.001516
grad_step = 000336, loss = 0.001565
grad_step = 000337, loss = 0.001602
grad_step = 000338, loss = 0.001527
grad_step = 000339, loss = 0.001514
grad_step = 000340, loss = 0.001566
grad_step = 000341, loss = 0.001539
grad_step = 000342, loss = 0.001499
grad_step = 000343, loss = 0.001511
grad_step = 000344, loss = 0.001522
grad_step = 000345, loss = 0.001499
grad_step = 000346, loss = 0.001489
grad_step = 000347, loss = 0.001505
grad_step = 000348, loss = 0.001505
grad_step = 000349, loss = 0.001484
grad_step = 000350, loss = 0.001481
grad_step = 000351, loss = 0.001493
grad_step = 000352, loss = 0.001491
grad_step = 000353, loss = 0.001481
grad_step = 000354, loss = 0.001472
grad_step = 000355, loss = 0.001475
grad_step = 000356, loss = 0.001479
grad_step = 000357, loss = 0.001477
grad_step = 000358, loss = 0.001470
grad_step = 000359, loss = 0.001467
grad_step = 000360, loss = 0.001465
grad_step = 000361, loss = 0.001463
grad_step = 000362, loss = 0.001461
grad_step = 000363, loss = 0.001465
grad_step = 000364, loss = 0.001473
grad_step = 000365, loss = 0.001491
grad_step = 000366, loss = 0.001522
grad_step = 000367, loss = 0.001608
grad_step = 000368, loss = 0.001633
grad_step = 000369, loss = 0.001691
grad_step = 000370, loss = 0.001540
grad_step = 000371, loss = 0.001458
grad_step = 000372, loss = 0.001495
grad_step = 000373, loss = 0.001548
grad_step = 000374, loss = 0.001568
grad_step = 000375, loss = 0.001474
grad_step = 000376, loss = 0.001446
grad_step = 000377, loss = 0.001485
grad_step = 000378, loss = 0.001490
grad_step = 000379, loss = 0.001458
grad_step = 000380, loss = 0.001442
grad_step = 000381, loss = 0.001460
grad_step = 000382, loss = 0.001469
grad_step = 000383, loss = 0.001448
grad_step = 000384, loss = 0.001433
grad_step = 000385, loss = 0.001444
grad_step = 000386, loss = 0.001449
grad_step = 000387, loss = 0.001438
grad_step = 000388, loss = 0.001426
grad_step = 000389, loss = 0.001427
grad_step = 000390, loss = 0.001435
grad_step = 000391, loss = 0.001439
grad_step = 000392, loss = 0.001431
grad_step = 000393, loss = 0.001421
grad_step = 000394, loss = 0.001415
grad_step = 000395, loss = 0.001417
grad_step = 000396, loss = 0.001421
grad_step = 000397, loss = 0.001423
grad_step = 000398, loss = 0.001420
grad_step = 000399, loss = 0.001417
grad_step = 000400, loss = 0.001414
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001411
grad_step = 000402, loss = 0.001408
grad_step = 000403, loss = 0.001404
grad_step = 000404, loss = 0.001401
grad_step = 000405, loss = 0.001399
grad_step = 000406, loss = 0.001399
grad_step = 000407, loss = 0.001405
grad_step = 000408, loss = 0.001423
grad_step = 000409, loss = 0.001481
grad_step = 000410, loss = 0.001582
grad_step = 000411, loss = 0.001831
grad_step = 000412, loss = 0.001775
grad_step = 000413, loss = 0.001720
grad_step = 000414, loss = 0.001438
grad_step = 000415, loss = 0.001611
grad_step = 000416, loss = 0.001756
grad_step = 000417, loss = 0.001483
grad_step = 000418, loss = 0.001564
grad_step = 000419, loss = 0.001744
grad_step = 000420, loss = 0.001479
grad_step = 000421, loss = 0.001500
grad_step = 000422, loss = 0.001687
grad_step = 000423, loss = 0.001517
grad_step = 000424, loss = 0.001422
grad_step = 000425, loss = 0.001583
grad_step = 000426, loss = 0.001521
grad_step = 000427, loss = 0.001409
grad_step = 000428, loss = 0.001475
grad_step = 000429, loss = 0.001484
grad_step = 000430, loss = 0.001451
grad_step = 000431, loss = 0.001386
grad_step = 000432, loss = 0.001443
grad_step = 000433, loss = 0.001446
grad_step = 000434, loss = 0.001389
grad_step = 000435, loss = 0.001399
grad_step = 000436, loss = 0.001428
grad_step = 000437, loss = 0.001401
grad_step = 000438, loss = 0.001380
grad_step = 000439, loss = 0.001383
grad_step = 000440, loss = 0.001410
grad_step = 000441, loss = 0.001378
grad_step = 000442, loss = 0.001367
grad_step = 000443, loss = 0.001373
grad_step = 000444, loss = 0.001384
grad_step = 000445, loss = 0.001378
grad_step = 000446, loss = 0.001357
grad_step = 000447, loss = 0.001360
grad_step = 000448, loss = 0.001368
grad_step = 000449, loss = 0.001365
grad_step = 000450, loss = 0.001355
grad_step = 000451, loss = 0.001346
grad_step = 000452, loss = 0.001351
grad_step = 000453, loss = 0.001356
grad_step = 000454, loss = 0.001353
grad_step = 000455, loss = 0.001348
grad_step = 000456, loss = 0.001338
grad_step = 000457, loss = 0.001339
grad_step = 000458, loss = 0.001338
grad_step = 000459, loss = 0.001340
grad_step = 000460, loss = 0.001339
grad_step = 000461, loss = 0.001335
grad_step = 000462, loss = 0.001332
grad_step = 000463, loss = 0.001328
grad_step = 000464, loss = 0.001327
grad_step = 000465, loss = 0.001326
grad_step = 000466, loss = 0.001326
grad_step = 000467, loss = 0.001326
grad_step = 000468, loss = 0.001325
grad_step = 000469, loss = 0.001322
grad_step = 000470, loss = 0.001321
grad_step = 000471, loss = 0.001318
grad_step = 000472, loss = 0.001317
grad_step = 000473, loss = 0.001315
grad_step = 000474, loss = 0.001314
grad_step = 000475, loss = 0.001314
grad_step = 000476, loss = 0.001314
grad_step = 000477, loss = 0.001317
grad_step = 000478, loss = 0.001322
grad_step = 000479, loss = 0.001333
grad_step = 000480, loss = 0.001356
grad_step = 000481, loss = 0.001391
grad_step = 000482, loss = 0.001466
grad_step = 000483, loss = 0.001552
grad_step = 000484, loss = 0.001715
grad_step = 000485, loss = 0.001726
grad_step = 000486, loss = 0.001725
grad_step = 000487, loss = 0.001591
grad_step = 000488, loss = 0.001635
grad_step = 000489, loss = 0.002047
grad_step = 000490, loss = 0.001888
grad_step = 000491, loss = 0.001722
grad_step = 000492, loss = 0.001538
grad_step = 000493, loss = 0.001577
grad_step = 000494, loss = 0.001480
grad_step = 000495, loss = 0.001334
grad_step = 000496, loss = 0.001537
grad_step = 000497, loss = 0.001601
grad_step = 000498, loss = 0.001414
grad_step = 000499, loss = 0.001446
grad_step = 000500, loss = 0.001417
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001350
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

  date_run                              2020-05-12 06:14:32.703783
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.199386
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 06:14:32.710768
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0913222
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 06:14:32.718767
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12749
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 06:14:32.725215
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.387674
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
0   2020-05-12 06:13:58.397885  ...    mean_absolute_error
1   2020-05-12 06:13:58.402255  ...     mean_squared_error
2   2020-05-12 06:13:58.405988  ...  median_absolute_error
3   2020-05-12 06:13:58.409892  ...               r2_score
4   2020-05-12 06:14:08.213191  ...    mean_absolute_error
5   2020-05-12 06:14:08.217696  ...     mean_squared_error
6   2020-05-12 06:14:08.221526  ...  median_absolute_error
7   2020-05-12 06:14:08.225403  ...               r2_score
8   2020-05-12 06:14:32.703783  ...    mean_absolute_error
9   2020-05-12 06:14:32.710768  ...     mean_squared_error
10  2020-05-12 06:14:32.718767  ...  median_absolute_error
11  2020-05-12 06:14:32.725215  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 147986.55it/s] 83%|████████▎ | 8208384/9912422 [00:00<00:08, 211244.87it/s]9920512it [00:00, 43516901.94it/s]                           
0it [00:00, ?it/s]32768it [00:00, 606572.22it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486858.84it/s] 85%|████████▍ | 1400832/1648877 [00:00<00:00, 684620.19it/s]1654784it [00:00, 7808486.75it/s]                            
0it [00:00, ?it/s]8192it [00:00, 207918.25it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b8b34afd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b28a66f60> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b8b2d5ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b2853e0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b8b34afd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b3dccee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b8b34afd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b31d7f748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b8b312ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b3dccee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3b28a66be0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd8d5755208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e1a713c650500298abdf7a46572a1be1c4311da8d273a13911927b9180d36c6a
  Stored in directory: /tmp/pip-ephem-wheel-cache-pcefnln5/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd86d33d1d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3670016/17464789 [=====>........................] - ETA: 0s
12705792/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 06:16:01.091023: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 06:16:01.095688: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 06:16:01.095845: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b657d3dc40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 06:16:01.095862: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7510 - accuracy: 0.4945
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5900 - accuracy: 0.5050 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5777 - accuracy: 0.5058
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5491 - accuracy: 0.5077
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5968 - accuracy: 0.5046
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5470 - accuracy: 0.5078
11000/25000 [============>.................] - ETA: 4s - loss: 7.5467 - accuracy: 0.5078
12000/25000 [=============>................] - ETA: 4s - loss: 7.5542 - accuracy: 0.5073
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5782 - accuracy: 0.5058
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5790 - accuracy: 0.5057
15000/25000 [=================>............] - ETA: 3s - loss: 7.5910 - accuracy: 0.5049
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6005 - accuracy: 0.5043
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6291 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6545 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 10s 398us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 06:16:18.794796
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 06:16:18.794796  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 06:16:25.608525: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 06:16:25.613740: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 06:16:25.613928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd9469bd00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 06:16:25.613945: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc23487ba58> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5923 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.5173 - val_crf_viterbi_accuracy: 0.0267

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc23bfd1128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8123 - accuracy: 0.4905
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6922 - accuracy: 0.4983 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6329 - accuracy: 0.5022
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5995 - accuracy: 0.5044
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5917 - accuracy: 0.5049
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
11000/25000 [============>.................] - ETA: 4s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
15000/25000 [=================>............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6323 - accuracy: 0.5022
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6444 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6781 - accuracy: 0.4992
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc20c610ac8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:37:30, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:52:38, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:22:05, 28.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<5:52:02, 40.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:05:51, 58.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.52M/862M [00:01<2:51:12, 83.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<1:59:27, 119kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:01<1:23:16, 169kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<58:03, 241kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:01<40:33, 344kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<28:19, 490kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:01<19:50, 696kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:01<13:54, 988kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<09:47, 1.40MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<06:54, 1.97MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:02<04:55, 2.75MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<05:07, 2.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<05:29, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:01, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<04:45, 2.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:40, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<05:21, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<04:05, 3.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:56, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<05:34, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<04:15, 3.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<05:55, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<06:59, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<04:05, 3.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<23:29, 560kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<17:52, 736kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:12<12:50, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<11:53, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<09:42, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:14<07:08, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:53, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<08:24, 1.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<06:29, 2.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<04:42, 2.75MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<09:16, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<07:52, 1.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:18<05:48, 2.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<06:56, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<06:07, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:21<04:26, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<23:12, 552kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<17:36, 727kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:22<12:38, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<11:40, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<10:52, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<08:12, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:26, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<09:23, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:53, 1.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:37, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:02, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:13, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:31, 2.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:19, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:12, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<05:19, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:27, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:18, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:43, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:08, 2.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:57, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:16, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:07, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:01, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:56, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:21, 2.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:00, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:12, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:39, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:58, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:53, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:29, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:59, 3.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<21:47, 556kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<16:31, 732kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<11:52, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<10:59, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<08:58, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<06:33, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:15, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:44, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:02, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:21, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:52, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<08:10, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<06:01, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:51, 1.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:26, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:50, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<04:14, 2.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<21:26, 548kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<16:15, 723kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<11:40, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<10:45, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<10:07, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<07:39, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<05:30, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:22, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<07:47, 1.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<05:45, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:36, 1.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:05, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:29, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:59, 2.88MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:26, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<07:09, 1.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:18, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:15, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:47, 1.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:22, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<03:53, 2.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<20:33, 552kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<15:36, 726kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<11:09, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:19, 1.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:37, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<07:14, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:25, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:49, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:25, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:05, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<03:41, 3.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<35:46, 309kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<26:14, 421kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<18:37, 592kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<15:27, 712kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<13:09, 835kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:48, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<06:58, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<22:05, 495kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<16:38, 656kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<11:55, 914kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<10:44, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<09:49, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:27, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<05:20, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<20:15, 533kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<15:18, 704kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<10:58, 979kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<10:04, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<09:20, 1.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:01, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<05:02, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:27, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:46, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:40, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:35, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<7:58:45, 22.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<5:35:18, 31.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<3:53:51, 45.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<2:50:53, 61.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<2:01:56, 86.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<1:25:45, 122kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<59:59, 174kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<45:58, 227kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<33:15, 314kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<23:30, 443kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<18:43, 554kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<15:21, 675kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<11:12, 924kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<08:02, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:22, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:59, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<05:09, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:52, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:12, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:52, 2.63MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:57, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:43, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:31, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:23, 2.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:54, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:20, 2.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:18, 3.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<02:27, 4.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<24:56, 402kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<18:29, 542kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<13:10, 759kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<11:29, 866kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<09:08, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:48, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:50, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:20, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:13, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:49, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:35, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:20, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:56, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:01, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:06, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<02:59, 3.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<17:02, 568kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<13:59, 691kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<10:17, 939kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<07:17, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<1:13:36, 131kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<52:32, 183kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<36:56, 259kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<27:53, 342kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<21:32, 443kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<15:32, 613kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<10:57, 866kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<13:03, 725kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<10:09, 931kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<07:20, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:14, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:03, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:28, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:16, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:40, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<03:13, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<24:20, 381kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<17:59, 515kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<12:48, 721kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<11:02, 833kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<09:40, 950kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:10, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<05:08, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<07:48, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:27, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:44, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:20, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:39, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:24, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:09, 2.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<13:12, 680kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<10:13, 879kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<07:22, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:08, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<06:53, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<05:13, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:46, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:18, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:22, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:59, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:11, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:55, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<09:57, 874kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<07:53, 1.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:44, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:59, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:01, 1.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<04:40, 1.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:20, 2.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<20:42, 414kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<15:23, 556kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<10:58, 778kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<09:32, 891kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<08:34, 992kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<06:24, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<04:34, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<13:31, 624kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:20, 815kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<07:24, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:05, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:42, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<04:12, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:48, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:07, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:02, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<02:55, 2.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<14:58, 549kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<11:22, 723kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<08:09, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:30, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:59, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<05:16, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<03:46, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:01, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<05:47, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<04:15, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:45, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:02, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:57, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<02:51, 2.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<14:10, 560kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<10:46, 737kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<07:44, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<07:09, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:41, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:06, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<03:38, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<12:19, 633kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<09:28, 824kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<06:49, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:28, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:20, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:55, 1.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:31, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:42, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<02:41, 2.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<19:57, 381kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<14:46, 514kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<10:28, 723kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<09:02, 833kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<07:08, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:09, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:15, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:15, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:02, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<02:53, 2.56MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<08:57, 825kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<07:04, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<05:07, 1.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:15, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:15, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:04, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:55, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<06:27, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<5:17:32, 22.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<3:42:14, 32.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<2:34:44, 46.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<1:52:34, 63.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<1:20:22, 89.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<56:33, 127kB/s]   .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<39:24, 181kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<38:46, 184kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<27:44, 256kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<19:31, 363kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<13:41, 515kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<18:48, 375kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<13:54, 507kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<09:53, 710kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<08:26, 827kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:23, 944kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<05:29, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:01, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:22, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:48, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:50, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:30, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:54, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:03, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:13, 3.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:26, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:50, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:49, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:28, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:51, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<02:13, 2.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<11:42, 567kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:53, 746kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:22, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:57, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:50, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:32, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:58, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:21, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:31, 2.57MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:40, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:51, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:05, 3.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:43, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:17, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:26, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:30, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:47, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:01, 3.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<11:12, 555kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<08:29, 731kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:03, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:38, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:19, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:06, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:01, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:16, 2.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:55, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:23, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:41, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:56, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<10:42, 555kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<08:00, 741kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:42, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<04:03, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<09:32, 616kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:55, 741kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:50, 1.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<04:07, 1.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<11:39, 498kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<08:46, 660kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<06:15, 924kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:38, 1.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:32, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:19, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:34, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:42, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:53, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:04, 2.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<41:24, 135kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<29:33, 189kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<20:45, 268kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<15:39, 353kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<11:32, 479kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<08:11, 671kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<06:55, 789kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:59, 909kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:29, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<03:11, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<10:42, 503kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<08:04, 666kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:46, 928kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:11, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:43, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:34, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:32, 2.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<44:36, 118kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<31:44, 165kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<22:13, 235kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<16:38, 311kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<12:10, 425kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<08:37, 597kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:08, 716kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<06:08, 833kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<04:31, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<03:12, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:47, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:52, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:49, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:06, 1.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:17, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<02:32, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:49, 2.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:49, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:20, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:42, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:49, 2.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:19, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:35, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:03, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:29, 3.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<34:34, 136kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<24:39, 190kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<17:17, 270kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<13:04, 354kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<09:38, 480kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<06:50, 673kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:45, 791kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:00, 909kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:45, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:18<02:39, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<08:54, 504kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<06:42, 668kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<04:46, 934kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:17, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:54, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:55, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<02:05, 2.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:16, 1.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:27, 1.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:30, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:41, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:50, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:12, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:35, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<06:21, 662kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:53, 860kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<03:30, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:23, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:15, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:28, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:45, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:30, 741kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:17, 949kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<03:05, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:02, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:57, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:16, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:37, 2.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<29:12, 135kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<20:49, 189kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<14:34, 268kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<10:21, 373kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<2:57:09, 21.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<2:03:47, 31.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<1:25:57, 44.5kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<1:01:16, 62.0kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<43:41, 86.9kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<30:39, 123kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<21:16, 176kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<17:12, 217kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<12:25, 300kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<08:44, 423kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<06:53, 531kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:37, 651kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:06, 889kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<02:53, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:52, 929kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:06, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:15, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:20, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:01, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:28, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:04, 3.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<05:12, 664kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:25, 781kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:14, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:19, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:34, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:09, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:35, 2.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:51, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:39, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:14, 2.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:34, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:45, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:22, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:00, 3.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:49, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:37, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:12, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:33, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<05:08, 605kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:36, 851kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<03:07, 973kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:30, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:49, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:56, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:40, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:15, 2.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<00:54, 3.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<07:00, 414kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<05:09, 562kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:44, 772kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<02:36, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<05:06, 556kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:52, 730kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:45, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:56, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<05:31, 500kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:26, 622kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:13, 854kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:17, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:21, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:55, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:29, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:05, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:27, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<00:58, 2.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:14, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:08, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:50, 2.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:37, 4.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<04:44, 525kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:49, 649kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:47, 884kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:56, 1.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<18:34, 130kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<13:13, 182kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<09:13, 259kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<06:23, 368kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<11:06, 212kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<08:16, 284kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<05:52, 398kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<04:07, 561kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<03:26, 663kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:38, 860kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:53, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:48, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:45, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:19, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:57, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:13, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:05, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:48, 2.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:09, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:55, 2.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<00:39, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:06, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:48, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:03, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:56, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:41, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:30, 3.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<05:56, 314kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<04:20, 428kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<03:02, 602kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:30, 718kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:08, 836kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:35, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<01:06, 1.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:38, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:57, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:02, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:05, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:49, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<00:36, 2.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:53, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:47, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:35, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:40, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:30, 2.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<00:21, 3.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<08:11, 177kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<06:02, 240kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<04:16, 336kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<02:56, 477kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<02:37, 528kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:58, 697kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:23, 970kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:14, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:08, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:51, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:43, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:38, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:28, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:35, 2.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<00:21, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<08:07, 136kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<05:46, 190kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<03:58, 270kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<02:45, 377kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<49:45, 20.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<34:28, 29.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<22:58, 42.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<17:12, 56.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<12:12, 79.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<08:28, 112kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<05:47, 160kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<04:13, 214kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<03:01, 295kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<02:05, 417kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:34, 526kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:17, 644kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:55, 878kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:38, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:39, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:32, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:24, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:19, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:13, 2.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:53, 331kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:22, 449kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:56, 631kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:44, 750kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:37, 880kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:27, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:18, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:20, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 2.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:13, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:07, 2.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:04, 3.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:22, 555kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:16, 732kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:10, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:07, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.64MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.36MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 707/400000 [00:00<00:56, 7062.97it/s]  0%|          | 1365/400000 [00:00<00:57, 6880.66it/s]  1%|          | 2052/400000 [00:00<00:57, 6877.34it/s]  1%|          | 2759/400000 [00:00<00:57, 6933.28it/s]  1%|          | 3497/400000 [00:00<00:56, 7060.68it/s]  1%|          | 4213/400000 [00:00<00:55, 7089.39it/s]  1%|          | 4930/400000 [00:00<00:55, 7110.33it/s]  1%|▏         | 5661/400000 [00:00<00:55, 7166.34it/s]  2%|▏         | 6370/400000 [00:00<00:55, 7141.24it/s]  2%|▏         | 7109/400000 [00:01<00:54, 7213.08it/s]  2%|▏         | 7867/400000 [00:01<00:53, 7317.80it/s]  2%|▏         | 8637/400000 [00:01<00:52, 7427.82it/s]  2%|▏         | 9395/400000 [00:01<00:52, 7470.19it/s]  3%|▎         | 10136/400000 [00:01<00:53, 7305.05it/s]  3%|▎         | 10863/400000 [00:01<00:54, 7122.98it/s]  3%|▎         | 11578/400000 [00:01<00:54, 7130.97it/s]  3%|▎         | 12364/400000 [00:01<00:52, 7332.46it/s]  3%|▎         | 13109/400000 [00:01<00:52, 7367.20it/s]  3%|▎         | 13872/400000 [00:01<00:51, 7443.83it/s]  4%|▎         | 14617/400000 [00:02<00:52, 7405.57it/s]  4%|▍         | 15389/400000 [00:02<00:51, 7496.83it/s]  4%|▍         | 16183/400000 [00:02<00:50, 7623.22it/s]  4%|▍         | 16995/400000 [00:02<00:49, 7764.01it/s]  4%|▍         | 17773/400000 [00:02<00:49, 7651.97it/s]  5%|▍         | 18540/400000 [00:02<00:50, 7567.05it/s]  5%|▍         | 19298/400000 [00:02<00:50, 7514.31it/s]  5%|▌         | 20051/400000 [00:02<00:51, 7429.82it/s]  5%|▌         | 20795/400000 [00:02<00:51, 7344.96it/s]  5%|▌         | 21531/400000 [00:02<00:52, 7267.18it/s]  6%|▌         | 22307/400000 [00:03<00:50, 7405.94it/s]  6%|▌         | 23049/400000 [00:03<00:51, 7295.03it/s]  6%|▌         | 23780/400000 [00:03<00:51, 7289.25it/s]  6%|▌         | 24510/400000 [00:03<00:51, 7260.73it/s]  6%|▋         | 25237/400000 [00:03<00:52, 7140.05it/s]  6%|▋         | 25997/400000 [00:03<00:51, 7270.45it/s]  7%|▋         | 26741/400000 [00:03<00:51, 7318.76it/s]  7%|▋         | 27510/400000 [00:03<00:50, 7425.93it/s]  7%|▋         | 28259/400000 [00:03<00:49, 7444.29it/s]  7%|▋         | 29005/400000 [00:03<00:50, 7349.48it/s]  7%|▋         | 29773/400000 [00:04<00:49, 7444.67it/s]  8%|▊         | 30519/400000 [00:04<00:49, 7432.42it/s]  8%|▊         | 31287/400000 [00:04<00:49, 7503.57it/s]  8%|▊         | 32038/400000 [00:04<00:49, 7477.62it/s]  8%|▊         | 32787/400000 [00:04<00:49, 7403.50it/s]  8%|▊         | 33528/400000 [00:04<00:49, 7391.01it/s]  9%|▊         | 34268/400000 [00:04<00:49, 7360.72it/s]  9%|▉         | 35005/400000 [00:04<00:49, 7310.25it/s]  9%|▉         | 35750/400000 [00:04<00:49, 7351.49it/s]  9%|▉         | 36486/400000 [00:04<00:49, 7299.33it/s]  9%|▉         | 37217/400000 [00:05<00:49, 7275.76it/s]  9%|▉         | 37960/400000 [00:05<00:49, 7320.81it/s] 10%|▉         | 38693/400000 [00:05<00:50, 7119.75it/s] 10%|▉         | 39407/400000 [00:05<00:51, 7043.21it/s] 10%|█         | 40124/400000 [00:05<00:50, 7078.78it/s] 10%|█         | 40845/400000 [00:05<00:50, 7114.72it/s] 10%|█         | 41600/400000 [00:05<00:49, 7237.95it/s] 11%|█         | 42347/400000 [00:05<00:48, 7305.21it/s] 11%|█         | 43116/400000 [00:05<00:48, 7415.83it/s] 11%|█         | 43876/400000 [00:05<00:47, 7467.63it/s] 11%|█         | 44660/400000 [00:06<00:46, 7574.85it/s] 11%|█▏        | 45419/400000 [00:06<00:46, 7554.36it/s] 12%|█▏        | 46222/400000 [00:06<00:46, 7689.30it/s] 12%|█▏        | 46992/400000 [00:06<00:46, 7571.11it/s] 12%|█▏        | 47751/400000 [00:06<00:47, 7436.00it/s] 12%|█▏        | 48496/400000 [00:06<00:47, 7423.42it/s] 12%|█▏        | 49248/400000 [00:06<00:47, 7450.49it/s] 12%|█▏        | 49994/400000 [00:06<00:48, 7215.61it/s] 13%|█▎        | 50718/400000 [00:06<00:48, 7190.42it/s] 13%|█▎        | 51439/400000 [00:07<00:49, 6975.38it/s] 13%|█▎        | 52139/400000 [00:07<00:50, 6867.55it/s] 13%|█▎        | 52872/400000 [00:07<00:49, 6998.28it/s] 13%|█▎        | 53583/400000 [00:07<00:49, 7029.23it/s] 14%|█▎        | 54310/400000 [00:07<00:48, 7097.64it/s] 14%|█▍        | 55041/400000 [00:07<00:48, 7159.05it/s] 14%|█▍        | 55815/400000 [00:07<00:47, 7322.17it/s] 14%|█▍        | 56549/400000 [00:07<00:47, 7287.75it/s] 14%|█▍        | 57332/400000 [00:07<00:46, 7440.81it/s] 15%|█▍        | 58130/400000 [00:07<00:45, 7592.27it/s] 15%|█▍        | 58902/400000 [00:08<00:44, 7630.01it/s] 15%|█▍        | 59694/400000 [00:08<00:44, 7714.30it/s] 15%|█▌        | 60467/400000 [00:08<00:44, 7713.13it/s] 15%|█▌        | 61244/400000 [00:08<00:43, 7729.03it/s] 16%|█▌        | 62020/400000 [00:08<00:43, 7736.71it/s] 16%|█▌        | 62795/400000 [00:08<00:43, 7666.61it/s] 16%|█▌        | 63572/400000 [00:08<00:43, 7695.96it/s] 16%|█▌        | 64342/400000 [00:08<00:44, 7606.44it/s] 16%|█▋        | 65104/400000 [00:08<00:44, 7529.23it/s] 16%|█▋        | 65878/400000 [00:08<00:44, 7589.29it/s] 17%|█▋        | 66638/400000 [00:09<00:44, 7491.02it/s] 17%|█▋        | 67391/400000 [00:09<00:44, 7501.84it/s] 17%|█▋        | 68142/400000 [00:09<00:44, 7386.81it/s] 17%|█▋        | 68882/400000 [00:09<00:45, 7248.84it/s] 17%|█▋        | 69608/400000 [00:09<00:46, 7103.21it/s] 18%|█▊        | 70320/400000 [00:09<00:47, 6983.85it/s] 18%|█▊        | 71033/400000 [00:09<00:46, 7024.16it/s] 18%|█▊        | 71737/400000 [00:09<00:46, 7019.82it/s] 18%|█▊        | 72483/400000 [00:09<00:45, 7145.51it/s] 18%|█▊        | 73238/400000 [00:09<00:45, 7259.82it/s] 18%|█▊        | 73994/400000 [00:10<00:44, 7346.09it/s] 19%|█▊        | 74756/400000 [00:10<00:43, 7423.00it/s] 19%|█▉        | 75500/400000 [00:10<00:43, 7421.85it/s] 19%|█▉        | 76243/400000 [00:10<00:43, 7390.62it/s] 19%|█▉        | 76983/400000 [00:10<00:44, 7269.59it/s] 19%|█▉        | 77716/400000 [00:10<00:44, 7285.60it/s] 20%|█▉        | 78446/400000 [00:10<00:44, 7285.41it/s] 20%|█▉        | 79175/400000 [00:10<00:44, 7260.63it/s] 20%|█▉        | 79902/400000 [00:10<00:44, 7169.94it/s] 20%|██        | 80620/400000 [00:10<00:44, 7159.76it/s] 20%|██        | 81344/400000 [00:11<00:44, 7182.69it/s] 21%|██        | 82070/400000 [00:11<00:44, 7205.21it/s] 21%|██        | 82795/400000 [00:11<00:43, 7218.52it/s] 21%|██        | 83526/400000 [00:11<00:43, 7243.04it/s] 21%|██        | 84251/400000 [00:11<00:43, 7186.25it/s] 21%|██        | 84972/400000 [00:11<00:43, 7192.48it/s] 21%|██▏       | 85692/400000 [00:11<00:43, 7166.45it/s] 22%|██▏       | 86411/400000 [00:11<00:43, 7170.49it/s] 22%|██▏       | 87137/400000 [00:11<00:43, 7194.88it/s] 22%|██▏       | 87857/400000 [00:11<00:43, 7193.72it/s] 22%|██▏       | 88577/400000 [00:12<00:43, 7166.95it/s] 22%|██▏       | 89294/400000 [00:12<00:43, 7091.40it/s] 23%|██▎       | 90028/400000 [00:12<00:43, 7162.50it/s] 23%|██▎       | 90771/400000 [00:12<00:42, 7239.40it/s] 23%|██▎       | 91496/400000 [00:12<00:43, 7053.06it/s] 23%|██▎       | 92203/400000 [00:12<00:45, 6836.10it/s] 23%|██▎       | 92928/400000 [00:12<00:44, 6953.05it/s] 23%|██▎       | 93675/400000 [00:12<00:43, 7097.96it/s] 24%|██▎       | 94427/400000 [00:12<00:42, 7217.13it/s] 24%|██▍       | 95181/400000 [00:13<00:41, 7307.70it/s] 24%|██▍       | 95957/400000 [00:13<00:40, 7436.30it/s] 24%|██▍       | 96748/400000 [00:13<00:40, 7570.02it/s] 24%|██▍       | 97507/400000 [00:13<00:39, 7572.34it/s] 25%|██▍       | 98266/400000 [00:13<00:40, 7513.68it/s] 25%|██▍       | 99019/400000 [00:13<00:40, 7445.40it/s] 25%|██▍       | 99777/400000 [00:13<00:40, 7483.76it/s] 25%|██▌       | 100547/400000 [00:13<00:39, 7545.78it/s] 25%|██▌       | 101303/400000 [00:13<00:39, 7479.95it/s] 26%|██▌       | 102085/400000 [00:13<00:39, 7576.37it/s] 26%|██▌       | 102888/400000 [00:14<00:38, 7705.35it/s] 26%|██▌       | 103660/400000 [00:14<00:38, 7619.02it/s] 26%|██▌       | 104465/400000 [00:14<00:38, 7742.51it/s] 26%|██▋       | 105246/400000 [00:14<00:37, 7761.64it/s] 27%|██▋       | 106033/400000 [00:14<00:37, 7791.38it/s] 27%|██▋       | 106813/400000 [00:14<00:37, 7771.19it/s] 27%|██▋       | 107591/400000 [00:14<00:38, 7636.35it/s] 27%|██▋       | 108368/400000 [00:14<00:38, 7674.28it/s] 27%|██▋       | 109137/400000 [00:14<00:38, 7627.21it/s] 27%|██▋       | 109906/400000 [00:14<00:37, 7644.77it/s] 28%|██▊       | 110680/400000 [00:15<00:37, 7672.43it/s] 28%|██▊       | 111448/400000 [00:15<00:37, 7602.39it/s] 28%|██▊       | 112214/400000 [00:15<00:37, 7618.23it/s] 28%|██▊       | 112977/400000 [00:15<00:37, 7596.47it/s] 28%|██▊       | 113737/400000 [00:15<00:38, 7460.55it/s] 29%|██▊       | 114484/400000 [00:15<00:38, 7371.63it/s] 29%|██▉       | 115228/400000 [00:15<00:38, 7390.64it/s] 29%|██▉       | 116011/400000 [00:15<00:37, 7515.05it/s] 29%|██▉       | 116781/400000 [00:15<00:37, 7567.60it/s] 29%|██▉       | 117570/400000 [00:15<00:36, 7660.07it/s] 30%|██▉       | 118337/400000 [00:16<00:36, 7612.62it/s] 30%|██▉       | 119099/400000 [00:16<00:37, 7470.31it/s] 30%|██▉       | 119848/400000 [00:16<00:37, 7462.65it/s] 30%|███       | 120595/400000 [00:16<00:37, 7429.65it/s] 30%|███       | 121349/400000 [00:16<00:37, 7460.53it/s] 31%|███       | 122135/400000 [00:16<00:36, 7574.08it/s] 31%|███       | 122919/400000 [00:16<00:36, 7651.15it/s] 31%|███       | 123726/400000 [00:16<00:35, 7770.97it/s] 31%|███       | 124545/400000 [00:16<00:34, 7889.73it/s] 31%|███▏      | 125336/400000 [00:16<00:34, 7883.20it/s] 32%|███▏      | 126133/400000 [00:17<00:34, 7907.83it/s] 32%|███▏      | 126925/400000 [00:17<00:35, 7744.40it/s] 32%|███▏      | 127701/400000 [00:17<00:35, 7594.41it/s] 32%|███▏      | 128477/400000 [00:17<00:35, 7641.99it/s] 32%|███▏      | 129243/400000 [00:17<00:35, 7613.83it/s] 33%|███▎      | 130028/400000 [00:17<00:35, 7680.83it/s] 33%|███▎      | 130797/400000 [00:17<00:35, 7650.45it/s] 33%|███▎      | 131584/400000 [00:17<00:34, 7712.87it/s] 33%|███▎      | 132381/400000 [00:17<00:34, 7788.06it/s] 33%|███▎      | 133170/400000 [00:17<00:34, 7816.37it/s] 33%|███▎      | 133955/400000 [00:18<00:33, 7825.71it/s] 34%|███▎      | 134738/400000 [00:18<00:34, 7677.91it/s] 34%|███▍      | 135507/400000 [00:18<00:34, 7624.42it/s] 34%|███▍      | 136276/400000 [00:18<00:34, 7642.48it/s] 34%|███▍      | 137050/400000 [00:18<00:34, 7671.13it/s] 34%|███▍      | 137836/400000 [00:18<00:33, 7724.13it/s] 35%|███▍      | 138609/400000 [00:18<00:33, 7709.08it/s] 35%|███▍      | 139381/400000 [00:18<00:33, 7695.96it/s] 35%|███▌      | 140151/400000 [00:18<00:34, 7577.07it/s] 35%|███▌      | 140937/400000 [00:19<00:33, 7657.16it/s] 35%|███▌      | 141713/400000 [00:19<00:33, 7686.73it/s] 36%|███▌      | 142484/400000 [00:19<00:33, 7693.39it/s] 36%|███▌      | 143256/400000 [00:19<00:33, 7698.56it/s] 36%|███▌      | 144075/400000 [00:19<00:32, 7838.67it/s] 36%|███▌      | 144860/400000 [00:19<00:32, 7806.70it/s] 36%|███▋      | 145696/400000 [00:19<00:31, 7962.34it/s] 37%|███▋      | 146494/400000 [00:19<00:32, 7881.89it/s] 37%|███▋      | 147284/400000 [00:19<00:32, 7713.48it/s] 37%|███▋      | 148057/400000 [00:19<00:32, 7635.26it/s] 37%|███▋      | 148826/400000 [00:20<00:32, 7650.71it/s] 37%|███▋      | 149636/400000 [00:20<00:32, 7778.49it/s] 38%|███▊      | 150435/400000 [00:20<00:31, 7840.15it/s] 38%|███▊      | 151257/400000 [00:20<00:31, 7948.45it/s] 38%|███▊      | 152053/400000 [00:20<00:31, 7931.92it/s] 38%|███▊      | 152847/400000 [00:20<00:31, 7816.59it/s] 38%|███▊      | 153650/400000 [00:20<00:31, 7876.61it/s] 39%|███▊      | 154441/400000 [00:20<00:31, 7885.17it/s] 39%|███▉      | 155233/400000 [00:20<00:31, 7893.02it/s] 39%|███▉      | 156023/400000 [00:20<00:31, 7841.80it/s] 39%|███▉      | 156816/400000 [00:21<00:30, 7867.63it/s] 39%|███▉      | 157604/400000 [00:21<00:30, 7828.05it/s] 40%|███▉      | 158388/400000 [00:21<00:31, 7759.81it/s] 40%|███▉      | 159188/400000 [00:21<00:30, 7827.96it/s] 40%|████      | 160003/400000 [00:21<00:30, 7920.54it/s] 40%|████      | 160796/400000 [00:21<00:30, 7898.75it/s] 40%|████      | 161591/400000 [00:21<00:30, 7911.91it/s] 41%|████      | 162383/400000 [00:21<00:30, 7793.78it/s] 41%|████      | 163174/400000 [00:21<00:30, 7828.14it/s] 41%|████      | 163964/400000 [00:21<00:30, 7848.32it/s] 41%|████      | 164763/400000 [00:22<00:29, 7888.84it/s] 41%|████▏     | 165553/400000 [00:22<00:29, 7823.51it/s] 42%|████▏     | 166336/400000 [00:22<00:30, 7605.51it/s] 42%|████▏     | 167113/400000 [00:22<00:30, 7653.53it/s] 42%|████▏     | 167886/400000 [00:22<00:30, 7673.91it/s] 42%|████▏     | 168655/400000 [00:22<00:30, 7590.81it/s] 42%|████▏     | 169433/400000 [00:22<00:30, 7645.37it/s] 43%|████▎     | 170199/400000 [00:22<00:30, 7624.98it/s] 43%|████▎     | 170984/400000 [00:22<00:29, 7690.06it/s] 43%|████▎     | 171756/400000 [00:22<00:29, 7696.76it/s] 43%|████▎     | 172526/400000 [00:23<00:29, 7696.49it/s] 43%|████▎     | 173296/400000 [00:23<00:29, 7645.54it/s] 44%|████▎     | 174061/400000 [00:23<00:29, 7615.65it/s] 44%|████▎     | 174863/400000 [00:23<00:29, 7730.73it/s] 44%|████▍     | 175668/400000 [00:23<00:28, 7823.73it/s] 44%|████▍     | 176452/400000 [00:23<00:28, 7732.66it/s] 44%|████▍     | 177226/400000 [00:23<00:29, 7667.62it/s] 44%|████▍     | 177994/400000 [00:23<00:30, 7343.50it/s] 45%|████▍     | 178732/400000 [00:23<00:31, 7113.99it/s] 45%|████▍     | 179473/400000 [00:23<00:30, 7198.52it/s] 45%|████▌     | 180238/400000 [00:24<00:29, 7327.59it/s] 45%|████▌     | 181013/400000 [00:24<00:29, 7449.33it/s] 45%|████▌     | 181785/400000 [00:24<00:28, 7525.81it/s] 46%|████▌     | 182614/400000 [00:24<00:28, 7737.89it/s] 46%|████▌     | 183428/400000 [00:24<00:27, 7853.01it/s] 46%|████▌     | 184216/400000 [00:24<00:27, 7763.64it/s] 46%|████▌     | 184995/400000 [00:24<00:27, 7742.66it/s] 46%|████▋     | 185771/400000 [00:24<00:28, 7637.20it/s] 47%|████▋     | 186536/400000 [00:24<00:29, 7257.29it/s] 47%|████▋     | 187272/400000 [00:25<00:29, 7287.20it/s] 47%|████▋     | 188050/400000 [00:25<00:28, 7426.03it/s] 47%|████▋     | 188845/400000 [00:25<00:27, 7573.50it/s] 47%|████▋     | 189606/400000 [00:25<00:27, 7534.59it/s] 48%|████▊     | 190406/400000 [00:25<00:27, 7666.34it/s] 48%|████▊     | 191185/400000 [00:25<00:27, 7683.65it/s] 48%|████▊     | 191961/400000 [00:25<00:27, 7704.35it/s] 48%|████▊     | 192733/400000 [00:25<00:27, 7529.60it/s] 48%|████▊     | 193541/400000 [00:25<00:26, 7685.06it/s] 49%|████▊     | 194384/400000 [00:25<00:26, 7892.82it/s] 49%|████▉     | 195228/400000 [00:26<00:25, 8047.56it/s] 49%|████▉     | 196054/400000 [00:26<00:25, 8109.97it/s] 49%|████▉     | 196867/400000 [00:26<00:25, 8070.96it/s] 49%|████▉     | 197677/400000 [00:26<00:25, 8077.48it/s] 50%|████▉     | 198486/400000 [00:26<00:25, 7996.10it/s] 50%|████▉     | 199287/400000 [00:26<00:25, 7970.18it/s] 50%|█████     | 200105/400000 [00:26<00:24, 8031.20it/s] 50%|█████     | 200943/400000 [00:26<00:24, 8130.75it/s] 50%|█████     | 201757/400000 [00:26<00:25, 7725.29it/s] 51%|█████     | 202535/400000 [00:26<00:25, 7683.17it/s] 51%|█████     | 203307/400000 [00:27<00:26, 7523.70it/s] 51%|█████     | 204063/400000 [00:27<00:26, 7299.44it/s] 51%|█████     | 204797/400000 [00:27<00:28, 6840.81it/s] 51%|█████▏    | 205518/400000 [00:27<00:27, 6947.22it/s] 52%|█████▏    | 206219/400000 [00:27<00:27, 6926.23it/s] 52%|█████▏    | 206931/400000 [00:27<00:27, 6979.54it/s] 52%|█████▏    | 207650/400000 [00:27<00:27, 7038.17it/s] 52%|█████▏    | 208385/400000 [00:27<00:26, 7128.36it/s] 52%|█████▏    | 209104/400000 [00:27<00:26, 7144.39it/s] 52%|█████▏    | 209892/400000 [00:28<00:25, 7348.35it/s] 53%|█████▎    | 210656/400000 [00:28<00:25, 7432.05it/s] 53%|█████▎    | 211465/400000 [00:28<00:24, 7613.86it/s] 53%|█████▎    | 212229/400000 [00:28<00:24, 7605.69it/s] 53%|█████▎    | 213059/400000 [00:28<00:23, 7801.26it/s] 53%|█████▎    | 213842/400000 [00:28<00:24, 7500.83it/s] 54%|█████▎    | 214597/400000 [00:28<00:25, 7319.12it/s] 54%|█████▍    | 215333/400000 [00:28<00:25, 7203.38it/s] 54%|█████▍    | 216110/400000 [00:28<00:24, 7362.61it/s] 54%|█████▍    | 216905/400000 [00:28<00:24, 7526.81it/s] 54%|█████▍    | 217722/400000 [00:29<00:23, 7706.79it/s] 55%|█████▍    | 218496/400000 [00:29<00:23, 7620.99it/s] 55%|█████▍    | 219359/400000 [00:29<00:22, 7896.22it/s] 55%|█████▌    | 220168/400000 [00:29<00:22, 7951.96it/s] 55%|█████▌    | 220980/400000 [00:29<00:22, 7999.66it/s] 55%|█████▌    | 221804/400000 [00:29<00:22, 8069.40it/s] 56%|█████▌    | 222653/400000 [00:29<00:21, 8189.63it/s] 56%|█████▌    | 223526/400000 [00:29<00:21, 8343.11it/s] 56%|█████▌    | 224363/400000 [00:29<00:21, 8189.35it/s] 56%|█████▋    | 225184/400000 [00:29<00:21, 8129.93it/s] 57%|█████▋    | 226012/400000 [00:30<00:21, 8170.58it/s] 57%|█████▋    | 226831/400000 [00:30<00:21, 8118.42it/s] 57%|█████▋    | 227662/400000 [00:30<00:21, 8173.01it/s] 57%|█████▋    | 228480/400000 [00:30<00:21, 8071.40it/s] 57%|█████▋    | 229308/400000 [00:30<00:20, 8129.17it/s] 58%|█████▊    | 230122/400000 [00:30<00:21, 8038.24it/s] 58%|█████▊    | 230929/400000 [00:30<00:21, 8045.96it/s] 58%|█████▊    | 231735/400000 [00:30<00:21, 7881.04it/s] 58%|█████▊    | 232525/400000 [00:30<00:22, 7555.09it/s] 58%|█████▊    | 233285/400000 [00:30<00:22, 7541.84it/s] 59%|█████▊    | 234042/400000 [00:31<00:22, 7533.04it/s] 59%|█████▊    | 234848/400000 [00:31<00:21, 7680.55it/s] 59%|█████▉    | 235627/400000 [00:31<00:21, 7712.79it/s] 59%|█████▉    | 236400/400000 [00:31<00:21, 7693.56it/s] 59%|█████▉    | 237171/400000 [00:31<00:21, 7648.53it/s] 59%|█████▉    | 237937/400000 [00:31<00:21, 7649.43it/s] 60%|█████▉    | 238717/400000 [00:31<00:20, 7693.60it/s] 60%|█████▉    | 239514/400000 [00:31<00:20, 7774.45it/s] 60%|██████    | 240298/400000 [00:31<00:20, 7792.26it/s] 60%|██████    | 241078/400000 [00:31<00:20, 7716.45it/s] 60%|██████    | 241851/400000 [00:32<00:20, 7562.50it/s] 61%|██████    | 242609/400000 [00:32<00:21, 7449.49it/s] 61%|██████    | 243356/400000 [00:32<00:21, 7371.91it/s] 61%|██████    | 244115/400000 [00:32<00:20, 7433.91it/s] 61%|██████    | 244881/400000 [00:32<00:20, 7500.06it/s] 61%|██████▏   | 245632/400000 [00:32<00:20, 7460.89it/s] 62%|██████▏   | 246382/400000 [00:32<00:20, 7471.86it/s] 62%|██████▏   | 247139/400000 [00:32<00:20, 7500.89it/s] 62%|██████▏   | 247890/400000 [00:32<00:20, 7457.99it/s] 62%|██████▏   | 248637/400000 [00:33<00:20, 7408.47it/s] 62%|██████▏   | 249379/400000 [00:33<00:20, 7378.46it/s] 63%|██████▎   | 250118/400000 [00:33<00:20, 7362.60it/s] 63%|██████▎   | 250868/400000 [00:33<00:20, 7395.78it/s] 63%|██████▎   | 251608/400000 [00:33<00:20, 7132.81it/s] 63%|██████▎   | 252324/400000 [00:33<00:21, 6988.42it/s] 63%|██████▎   | 253025/400000 [00:33<00:21, 6972.85it/s] 63%|██████▎   | 253724/400000 [00:33<00:21, 6802.10it/s] 64%|██████▎   | 254407/400000 [00:33<00:21, 6777.02it/s] 64%|██████▍   | 255087/400000 [00:33<00:21, 6691.40it/s] 64%|██████▍   | 255758/400000 [00:34<00:21, 6632.97it/s] 64%|██████▍   | 256457/400000 [00:34<00:21, 6736.04it/s] 64%|██████▍   | 257189/400000 [00:34<00:20, 6900.06it/s] 64%|██████▍   | 257920/400000 [00:34<00:20, 7015.72it/s] 65%|██████▍   | 258630/400000 [00:34<00:20, 7040.71it/s] 65%|██████▍   | 259360/400000 [00:34<00:19, 7115.37it/s] 65%|██████▌   | 260094/400000 [00:34<00:19, 7179.78it/s] 65%|██████▌   | 260813/400000 [00:34<00:19, 7168.90it/s] 65%|██████▌   | 261571/400000 [00:34<00:18, 7287.46it/s] 66%|██████▌   | 262351/400000 [00:34<00:18, 7432.93it/s] 66%|██████▌   | 263096/400000 [00:35<00:18, 7426.36it/s] 66%|██████▌   | 263840/400000 [00:35<00:18, 7422.96it/s] 66%|██████▌   | 264583/400000 [00:35<00:18, 7316.81it/s] 66%|██████▋   | 265316/400000 [00:35<00:18, 7181.46it/s] 67%|██████▋   | 266055/400000 [00:35<00:18, 7240.54it/s] 67%|██████▋   | 266780/400000 [00:35<00:18, 7125.64it/s] 67%|██████▋   | 267517/400000 [00:35<00:18, 7195.67it/s] 67%|██████▋   | 268280/400000 [00:35<00:17, 7319.84it/s] 67%|██████▋   | 269031/400000 [00:35<00:17, 7374.68it/s] 67%|██████▋   | 269770/400000 [00:35<00:17, 7320.50it/s] 68%|██████▊   | 270524/400000 [00:36<00:17, 7384.34it/s] 68%|██████▊   | 271279/400000 [00:36<00:17, 7430.41it/s] 68%|██████▊   | 272072/400000 [00:36<00:16, 7572.38it/s] 68%|██████▊   | 272871/400000 [00:36<00:16, 7685.66it/s] 68%|██████▊   | 273641/400000 [00:36<00:16, 7590.07it/s] 69%|██████▊   | 274402/400000 [00:36<00:16, 7504.72it/s] 69%|██████▉   | 275154/400000 [00:36<00:17, 7313.76it/s] 69%|██████▉   | 275888/400000 [00:36<00:16, 7300.97it/s] 69%|██████▉   | 276639/400000 [00:36<00:16, 7359.81it/s] 69%|██████▉   | 277376/400000 [00:36<00:16, 7350.37it/s] 70%|██████▉   | 278125/400000 [00:37<00:16, 7391.42it/s] 70%|██████▉   | 278865/400000 [00:37<00:16, 7181.83it/s] 70%|██████▉   | 279624/400000 [00:37<00:16, 7298.08it/s] 70%|███████   | 280391/400000 [00:37<00:16, 7403.31it/s] 70%|███████   | 281133/400000 [00:37<00:16, 7083.30it/s] 70%|███████   | 281846/400000 [00:37<00:16, 6974.78it/s] 71%|███████   | 282547/400000 [00:37<00:16, 6916.96it/s] 71%|███████   | 283278/400000 [00:37<00:16, 7028.74it/s] 71%|███████   | 284030/400000 [00:37<00:16, 7168.34it/s] 71%|███████   | 284749/400000 [00:38<00:16, 7086.65it/s] 71%|███████▏  | 285460/400000 [00:38<00:16, 7054.92it/s] 72%|███████▏  | 286167/400000 [00:38<00:16, 7044.30it/s] 72%|███████▏  | 286873/400000 [00:38<00:16, 7007.79it/s] 72%|███████▏  | 287583/400000 [00:38<00:15, 7032.90it/s] 72%|███████▏  | 288371/400000 [00:38<00:15, 7265.01it/s] 72%|███████▏  | 289135/400000 [00:38<00:15, 7371.77it/s] 72%|███████▏  | 289889/400000 [00:38<00:14, 7420.49it/s] 73%|███████▎  | 290662/400000 [00:38<00:14, 7509.85it/s] 73%|███████▎  | 291415/400000 [00:38<00:14, 7476.81it/s] 73%|███████▎  | 292164/400000 [00:39<00:15, 7052.54it/s] 73%|███████▎  | 292875/400000 [00:39<00:15, 6804.98it/s] 73%|███████▎  | 293562/400000 [00:39<00:15, 6788.10it/s] 74%|███████▎  | 294285/400000 [00:39<00:15, 6913.29it/s] 74%|███████▎  | 294980/400000 [00:39<00:15, 6863.89it/s] 74%|███████▍  | 295669/400000 [00:39<00:15, 6784.41it/s] 74%|███████▍  | 296350/400000 [00:39<00:15, 6751.61it/s] 74%|███████▍  | 297099/400000 [00:39<00:14, 6956.84it/s] 74%|███████▍  | 297798/400000 [00:39<00:14, 6867.13it/s] 75%|███████▍  | 298496/400000 [00:39<00:14, 6900.05it/s] 75%|███████▍  | 299251/400000 [00:40<00:14, 7077.91it/s] 75%|███████▌  | 300057/400000 [00:40<00:13, 7345.54it/s] 75%|███████▌  | 300863/400000 [00:40<00:13, 7545.53it/s] 75%|███████▌  | 301679/400000 [00:40<00:12, 7718.01it/s] 76%|███████▌  | 302455/400000 [00:40<00:12, 7682.33it/s] 76%|███████▌  | 303227/400000 [00:40<00:12, 7521.20it/s] 76%|███████▌  | 303982/400000 [00:40<00:13, 7316.97it/s] 76%|███████▌  | 304733/400000 [00:40<00:12, 7371.71it/s] 76%|███████▋  | 305487/400000 [00:40<00:12, 7419.91it/s] 77%|███████▋  | 306231/400000 [00:41<00:12, 7371.18it/s] 77%|███████▋  | 306987/400000 [00:41<00:12, 7425.37it/s] 77%|███████▋  | 307784/400000 [00:41<00:12, 7579.83it/s] 77%|███████▋  | 308564/400000 [00:41<00:11, 7643.19it/s] 77%|███████▋  | 309370/400000 [00:41<00:11, 7762.78it/s] 78%|███████▊  | 310155/400000 [00:41<00:11, 7788.20it/s] 78%|███████▊  | 310935/400000 [00:41<00:11, 7717.01it/s] 78%|███████▊  | 311709/400000 [00:41<00:11, 7722.32it/s] 78%|███████▊  | 312482/400000 [00:41<00:11, 7723.96it/s] 78%|███████▊  | 313255/400000 [00:41<00:11, 7719.30it/s] 79%|███████▊  | 314028/400000 [00:42<00:11, 7502.78it/s] 79%|███████▊  | 314802/400000 [00:42<00:11, 7572.38it/s] 79%|███████▉  | 315561/400000 [00:42<00:11, 7510.15it/s] 79%|███████▉  | 316324/400000 [00:42<00:11, 7545.01it/s] 79%|███████▉  | 317086/400000 [00:42<00:10, 7566.63it/s] 79%|███████▉  | 317853/400000 [00:42<00:10, 7593.99it/s] 80%|███████▉  | 318613/400000 [00:42<00:10, 7525.10it/s] 80%|███████▉  | 319366/400000 [00:42<00:10, 7437.45it/s] 80%|████████  | 320111/400000 [00:42<00:10, 7422.93it/s] 80%|████████  | 320854/400000 [00:42<00:11, 7172.80it/s] 80%|████████  | 321574/400000 [00:43<00:11, 6994.62it/s] 81%|████████  | 322276/400000 [00:43<00:11, 6942.79it/s] 81%|████████  | 322979/400000 [00:43<00:11, 6966.94it/s] 81%|████████  | 323677/400000 [00:43<00:11, 6891.73it/s] 81%|████████  | 324373/400000 [00:43<00:10, 6910.30it/s] 81%|████████▏ | 325065/400000 [00:43<00:10, 6825.36it/s] 81%|████████▏ | 325749/400000 [00:43<00:11, 6268.86it/s] 82%|████████▏ | 326386/400000 [00:43<00:11, 6150.04it/s] 82%|████████▏ | 327009/400000 [00:43<00:12, 6054.88it/s] 82%|████████▏ | 327620/400000 [00:44<00:12, 5739.66it/s] 82%|████████▏ | 328229/400000 [00:44<00:12, 5839.55it/s] 82%|████████▏ | 328828/400000 [00:44<00:12, 5883.43it/s] 82%|████████▏ | 329525/400000 [00:44<00:11, 6171.48it/s] 83%|████████▎ | 330208/400000 [00:44<00:10, 6353.85it/s] 83%|████████▎ | 330897/400000 [00:44<00:10, 6503.57it/s] 83%|████████▎ | 331553/400000 [00:44<00:10, 6467.76it/s] 83%|████████▎ | 332257/400000 [00:44<00:10, 6627.16it/s] 83%|████████▎ | 333002/400000 [00:44<00:09, 6851.95it/s] 83%|████████▎ | 333741/400000 [00:44<00:09, 7004.95it/s] 84%|████████▎ | 334485/400000 [00:45<00:09, 7129.04it/s] 84%|████████▍ | 335215/400000 [00:45<00:09, 7178.17it/s] 84%|████████▍ | 335953/400000 [00:45<00:08, 7236.57it/s] 84%|████████▍ | 336713/400000 [00:45<00:08, 7339.09it/s] 84%|████████▍ | 337456/400000 [00:45<00:08, 7365.61it/s] 85%|████████▍ | 338194/400000 [00:45<00:08, 7285.96it/s] 85%|████████▍ | 338924/400000 [00:45<00:08, 6849.59it/s] 85%|████████▍ | 339658/400000 [00:45<00:08, 6988.06it/s] 85%|████████▌ | 340394/400000 [00:45<00:08, 7094.48it/s] 85%|████████▌ | 341134/400000 [00:45<00:08, 7181.35it/s] 85%|████████▌ | 341890/400000 [00:46<00:07, 7288.30it/s] 86%|████████▌ | 342640/400000 [00:46<00:07, 7350.30it/s] 86%|████████▌ | 343401/400000 [00:46<00:07, 7426.20it/s] 86%|████████▌ | 344168/400000 [00:46<00:07, 7495.94it/s] 86%|████████▌ | 344919/400000 [00:46<00:07, 7266.55it/s] 86%|████████▋ | 345667/400000 [00:46<00:07, 7327.95it/s] 87%|████████▋ | 346402/400000 [00:46<00:07, 6771.27it/s] 87%|████████▋ | 347089/400000 [00:46<00:07, 6639.13it/s] 87%|████████▋ | 347761/400000 [00:46<00:07, 6576.44it/s] 87%|████████▋ | 348424/400000 [00:47<00:08, 6333.23it/s] 87%|████████▋ | 349079/400000 [00:47<00:07, 6394.51it/s] 87%|████████▋ | 349825/400000 [00:47<00:07, 6679.60it/s] 88%|████████▊ | 350500/400000 [00:47<00:07, 6678.93it/s] 88%|████████▊ | 351220/400000 [00:47<00:07, 6824.76it/s] 88%|████████▊ | 352040/400000 [00:47<00:06, 7185.69it/s] 88%|████████▊ | 352859/400000 [00:47<00:06, 7459.23it/s] 88%|████████▊ | 353728/400000 [00:47<00:05, 7790.12it/s] 89%|████████▊ | 354568/400000 [00:47<00:05, 7962.81it/s] 89%|████████▉ | 355427/400000 [00:47<00:05, 8141.06it/s] 89%|████████▉ | 356275/400000 [00:48<00:05, 8237.80it/s] 89%|████████▉ | 357104/400000 [00:48<00:05, 8248.89it/s] 89%|████████▉ | 357933/400000 [00:48<00:05, 8233.43it/s] 90%|████████▉ | 358759/400000 [00:48<00:05, 8240.64it/s] 90%|████████▉ | 359600/400000 [00:48<00:04, 8289.64it/s] 90%|█████████ | 360431/400000 [00:48<00:04, 8118.68it/s] 90%|█████████ | 361245/400000 [00:48<00:04, 8084.19it/s] 91%|█████████ | 362077/400000 [00:48<00:04, 8150.69it/s] 91%|█████████ | 362896/400000 [00:48<00:04, 8159.73it/s] 91%|█████████ | 363713/400000 [00:48<00:04, 7987.69it/s] 91%|█████████ | 364514/400000 [00:49<00:04, 7899.64it/s] 91%|█████████▏| 365306/400000 [00:49<00:04, 7615.75it/s] 92%|█████████▏| 366104/400000 [00:49<00:04, 7720.78it/s] 92%|█████████▏| 366995/400000 [00:49<00:04, 8041.26it/s] 92%|█████████▏| 367855/400000 [00:49<00:03, 8199.45it/s] 92%|█████████▏| 368732/400000 [00:49<00:03, 8362.35it/s] 92%|█████████▏| 369576/400000 [00:49<00:03, 8385.18it/s] 93%|█████████▎| 370477/400000 [00:49<00:03, 8561.92it/s] 93%|█████████▎| 371350/400000 [00:49<00:03, 8609.06it/s] 93%|█████████▎| 372213/400000 [00:49<00:03, 8314.54it/s] 93%|█████████▎| 373049/400000 [00:50<00:03, 8033.75it/s] 93%|█████████▎| 373857/400000 [00:50<00:03, 7956.94it/s] 94%|█████████▎| 374657/400000 [00:50<00:03, 7947.17it/s] 94%|█████████▍| 375467/400000 [00:50<00:03, 7991.20it/s] 94%|█████████▍| 376280/400000 [00:50<00:02, 8029.78it/s] 94%|█████████▍| 377085/400000 [00:50<00:02, 7908.70it/s] 94%|█████████▍| 377878/400000 [00:50<00:02, 7668.44it/s] 95%|█████████▍| 378648/400000 [00:50<00:02, 7659.46it/s] 95%|█████████▍| 379416/400000 [00:50<00:02, 7651.65it/s] 95%|█████████▌| 380183/400000 [00:51<00:02, 7333.66it/s] 95%|█████████▌| 380921/400000 [00:51<00:02, 7148.13it/s] 95%|█████████▌| 381640/400000 [00:51<00:02, 6965.56it/s] 96%|█████████▌| 382341/400000 [00:51<00:02, 6893.72it/s] 96%|█████████▌| 383034/400000 [00:51<00:02, 6659.21it/s] 96%|█████████▌| 383782/400000 [00:51<00:02, 6883.12it/s] 96%|█████████▌| 384522/400000 [00:51<00:02, 7030.40it/s] 96%|█████████▋| 385277/400000 [00:51<00:02, 7177.57it/s] 97%|█████████▋| 386020/400000 [00:51<00:01, 7249.83it/s] 97%|█████████▋| 386748/400000 [00:51<00:01, 7200.29it/s] 97%|█████████▋| 387487/400000 [00:52<00:01, 7253.79it/s] 97%|█████████▋| 388233/400000 [00:52<00:01, 7312.54it/s] 97%|█████████▋| 388970/400000 [00:52<00:01, 7327.58it/s] 97%|█████████▋| 389704/400000 [00:52<00:01, 7299.02it/s] 98%|█████████▊| 390441/400000 [00:52<00:01, 7317.51it/s] 98%|█████████▊| 391210/400000 [00:52<00:01, 7425.23it/s] 98%|█████████▊| 391954/400000 [00:52<00:01, 7323.24it/s] 98%|█████████▊| 392688/400000 [00:52<00:01, 7182.45it/s] 98%|█████████▊| 393408/400000 [00:52<00:00, 7025.16it/s] 99%|█████████▊| 394113/400000 [00:52<00:00, 6909.93it/s] 99%|█████████▊| 394828/400000 [00:53<00:00, 6979.23it/s] 99%|█████████▉| 395552/400000 [00:53<00:00, 7053.97it/s] 99%|█████████▉| 396302/400000 [00:53<00:00, 7181.42it/s] 99%|█████████▉| 397057/400000 [00:53<00:00, 7285.44it/s] 99%|█████████▉| 397801/400000 [00:53<00:00, 7329.86it/s]100%|█████████▉| 398563/400000 [00:53<00:00, 7414.44it/s]100%|█████████▉| 399332/400000 [00:53<00:00, 7493.14it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7437.15it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc20fb5abe0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011342468265129602 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.01081535848088089 	 Accuracy: 69

  model saves at 69% accuracy 

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

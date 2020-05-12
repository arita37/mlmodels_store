
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff2d317af60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 07:13:03.753683
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 07:13:03.758316
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 07:13:03.762176
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 07:13:03.766054
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff2d2e38128> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353959.5312
Epoch 2/10

1/1 [==============================] - 0s 113ms/step - loss: 256283.7969
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 158199.3438
Epoch 4/10

1/1 [==============================] - 0s 112ms/step - loss: 84411.9453
Epoch 5/10

1/1 [==============================] - 0s 109ms/step - loss: 44193.7109
Epoch 6/10

1/1 [==============================] - 0s 124ms/step - loss: 24526.5469
Epoch 7/10

1/1 [==============================] - 0s 105ms/step - loss: 14910.2100
Epoch 8/10

1/1 [==============================] - 0s 106ms/step - loss: 9857.5146
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 7003.1836
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 5288.9590

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.68548760e-02  8.83710670e+00  9.45300770e+00  9.50227261e+00
   9.25491810e+00  8.84976959e+00  8.82051849e+00  9.82501411e+00
   9.40731525e+00  8.39137554e+00  1.01296368e+01  8.25040150e+00
   8.12848663e+00  9.10130882e+00  9.48227406e+00  1.00044117e+01
   9.08614826e+00  9.84361458e+00  9.93443108e+00  9.57998466e+00
   9.86720467e+00  1.13748493e+01  8.72981548e+00  8.96326160e+00
   1.03383512e+01  8.35955334e+00  9.76754951e+00  9.88236332e+00
   8.51119900e+00  7.94126558e+00  9.65817642e+00  1.08868647e+01
   9.71277523e+00  9.11648464e+00  1.13321371e+01  9.36744785e+00
   8.51020336e+00  9.87729073e+00  1.01585340e+01  9.98896694e+00
   1.00308619e+01  9.68055344e+00  8.33102703e+00  9.32223034e+00
   9.22133350e+00  8.68904305e+00  9.66490841e+00  9.14786816e+00
   8.87948895e+00  7.75808096e+00  7.70990705e+00  1.06475458e+01
   7.87601614e+00  9.11349487e+00  1.05292015e+01  8.30653000e+00
   9.29515743e+00  1.12000360e+01  8.48825645e+00  9.46758270e+00
  -8.64861608e-02 -2.03844160e-03  1.25021863e+00 -1.26376796e+00
  -5.82568169e-01  6.78525031e-01  8.33682179e-01 -8.77649307e-01
  -1.39507961e+00  4.28743631e-01 -1.50470984e+00  1.15914130e+00
  -1.59749782e+00 -1.41213632e+00 -6.78883612e-01 -8.88246238e-01
   2.44114310e-01  1.03071916e+00 -9.46388483e-01 -1.13527155e+00
  -2.20676959e-02  6.08288236e-02  1.92301428e+00  1.73719689e-01
   4.59068656e-01  6.38419032e-01 -1.28284380e-01 -1.10706478e-01
  -1.63060308e+00  3.27013850e-01  1.38577878e+00 -6.19709492e-01
  -2.08754539e-02  1.72044128e-01 -2.02075869e-01  1.60534203e-01
  -1.35908508e+00  1.52700734e+00 -6.43621325e-01  1.65796608e-01
   4.73554134e-02  7.06949711e-01 -6.03921652e-01  6.75857067e-01
   2.52130723e+00 -9.29153860e-01  1.00129032e+00 -1.21912614e-01
   1.02911282e+00 -1.47260773e+00  2.11858630e-01 -1.46425664e-01
   2.19713449e-02  1.11912227e+00  5.42700529e-01 -1.51840121e-01
  -3.63069892e-01  5.91974974e-01 -3.34617853e-01 -4.47309881e-01
   1.41798651e+00 -5.77793360e-01 -9.83361185e-01 -5.46694994e-01
   9.65368748e-03 -5.56461215e-01  9.37698662e-01  1.07172847e+00
   4.72012311e-01  2.67656267e-01 -1.44377244e+00 -4.44649130e-01
  -7.08603978e-01 -1.49232006e+00 -8.23979139e-01 -1.82754755e+00
  -4.52914894e-01  7.90305674e-01  1.19079483e+00 -1.81464419e-01
   1.04825532e+00  1.38065386e+00  2.04796106e-01 -2.80389041e-01
  -1.34724629e+00 -3.35769922e-01 -5.69669008e-01 -8.77106667e-01
   1.99782038e+00 -3.25001985e-01 -6.29490912e-01  5.37216663e-02
   4.06111091e-01 -1.63923502e+00  1.16347468e+00  8.44268620e-01
   5.41070104e-03 -1.49999702e+00 -1.34017146e+00  3.13691586e-01
  -9.38473046e-02 -1.94021356e+00  1.11109042e+00  1.62988782e+00
  -1.02763224e+00  1.20495105e+00 -1.01200417e-01  1.05262089e+00
   3.49953324e-02  1.47997606e+00  1.29384828e+00  6.27354741e-01
   1.02600718e+00 -3.81030887e-01 -3.02332938e-01  3.96979064e-01
   1.86166406e-01  4.26546276e-01 -1.41542196e-01 -2.40023717e-01
   3.96429300e-01  8.85666847e+00  9.50799656e+00  7.87429762e+00
   8.61367416e+00  1.01667042e+01  9.27783680e+00  9.35615063e+00
   9.06728745e+00  9.64604950e+00  8.75685978e+00  1.01075315e+01
   8.55070877e+00  9.14799118e+00  8.74042320e+00  8.49900723e+00
   9.41170692e+00  1.04859552e+01  9.73618984e+00  1.09803019e+01
   1.00125666e+01  9.05017281e+00  9.85029888e+00  9.89879036e+00
   1.00342293e+01  1.09183731e+01  7.65465021e+00  9.75584316e+00
   1.09370470e+01  9.91557026e+00  9.69448853e+00  1.00130568e+01
   1.10739298e+01  9.05525589e+00  1.00850191e+01  1.05849504e+01
   1.06650448e+01  9.89645386e+00  9.03483868e+00  9.68328285e+00
   9.64712524e+00  1.04998055e+01  9.09013081e+00  9.57434082e+00
   1.09231939e+01  9.36528492e+00  1.03324680e+01  8.84669018e+00
   9.69083691e+00  1.09829149e+01  9.35956669e+00  1.01491241e+01
   8.40455627e+00  1.03757849e+01  9.51420212e+00  9.25101852e+00
   9.18696594e+00  1.04589520e+01  8.03598976e+00  8.27152443e+00
   8.51044059e-02  1.72748327e-01  1.43391430e+00  1.35507500e+00
   2.90124559e+00  8.21773171e-01  8.17261696e-01  1.41093946e+00
   4.86115038e-01  1.80156946e-01  1.05159307e+00  1.83164227e+00
   9.59406376e-01  7.43996501e-02  1.73301280e+00  3.08520436e-01
   2.71027207e-01  1.18318999e+00  9.60891366e-01  3.02939701e+00
   8.58999372e-01  1.46514809e+00  2.45673394e+00  7.05297112e-01
   6.81942999e-01  5.21633267e-01  9.92474735e-01  1.56821346e+00
   8.91903460e-01  1.39442956e+00  2.96570897e-01  1.12835002e+00
   2.45487142e+00  8.52611899e-01  7.22278237e-01  4.52317297e-01
   5.87077916e-01  5.03175795e-01  1.88927650e+00  1.38786054e+00
   9.18368638e-01  7.14532733e-02  9.96136665e-01  2.75866246e+00
   6.85117424e-01  2.72008848e+00  2.82164240e+00  3.10455024e-01
   1.29721129e+00  2.67671883e-01  1.56954598e+00  1.05529416e+00
   2.99085140e-01  2.82115936e-01  2.12638330e+00  9.96668220e-01
   8.49441290e-01  1.89638901e+00  2.56814814e+00  1.61108494e-01
   1.52875876e+00  2.47023988e+00  9.96241450e-01  1.87915206e-01
   9.10698175e-02  1.48326039e-01  2.61877108e+00  7.49691725e-01
   8.60210478e-01  6.68428779e-01  1.12143803e+00  2.45736241e-01
   1.38676786e+00  1.09217918e+00  2.43455553e+00  7.51312494e-01
   4.59820509e-01  3.77278686e-01  1.00126934e+00  1.49178696e+00
   1.69677961e+00  1.23946774e+00  1.52489424e+00  1.06869948e+00
   1.07350862e+00  4.84064102e-01  9.84669209e-01  2.64885378e+00
   4.86219168e-01  4.01387751e-01  2.86567402e+00  1.26265144e+00
   1.77697849e+00  4.07563686e-01  2.74105978e+00  8.07511806e-01
   2.16839314e+00  8.83660376e-01  1.37999630e+00  1.05438530e-01
   2.10196257e-01  1.62621939e+00  2.92777300e-01  1.02251613e+00
   1.47543538e+00  4.56653714e-01  1.25883257e+00  1.67505562e+00
   5.35555124e-01  3.65289688e-01  1.65885222e+00  2.66421270e+00
   1.59279656e+00  2.04716301e+00  1.41441429e+00  1.57569921e+00
   5.45619607e-01  7.45235205e-01  1.96744668e+00  1.15516269e+00
   9.69861889e+00 -9.06876278e+00 -7.75297117e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 07:13:13.718112
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9044
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 07:13:13.722449
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8654.02
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 07:13:13.726638
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1891
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 07:13:13.730893
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -774.028
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140680558362128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140679599812680
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140679599813184
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140679599813688
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140679599814192
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140679599814696

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff2dadc1eb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.556279
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.507878
grad_step = 000002, loss = 0.469936
grad_step = 000003, loss = 0.429056
grad_step = 000004, loss = 0.384676
grad_step = 000005, loss = 0.344398
grad_step = 000006, loss = 0.329556
grad_step = 000007, loss = 0.321656
grad_step = 000008, loss = 0.302848
grad_step = 000009, loss = 0.285820
grad_step = 000010, loss = 0.270615
grad_step = 000011, loss = 0.257788
grad_step = 000012, loss = 0.247155
grad_step = 000013, loss = 0.238438
grad_step = 000014, loss = 0.230009
grad_step = 000015, loss = 0.221375
grad_step = 000016, loss = 0.211984
grad_step = 000017, loss = 0.201173
grad_step = 000018, loss = 0.191037
grad_step = 000019, loss = 0.183761
grad_step = 000020, loss = 0.176653
grad_step = 000021, loss = 0.167219
grad_step = 000022, loss = 0.157443
grad_step = 000023, loss = 0.149744
grad_step = 000024, loss = 0.143790
grad_step = 000025, loss = 0.137609
grad_step = 000026, loss = 0.130381
grad_step = 000027, loss = 0.122942
grad_step = 000028, loss = 0.116215
grad_step = 000029, loss = 0.109835
grad_step = 000030, loss = 0.103096
grad_step = 000031, loss = 0.096703
grad_step = 000032, loss = 0.091217
grad_step = 000033, loss = 0.085724
grad_step = 000034, loss = 0.080106
grad_step = 000035, loss = 0.075113
grad_step = 000036, loss = 0.070547
grad_step = 000037, loss = 0.065750
grad_step = 000038, loss = 0.060933
grad_step = 000039, loss = 0.056696
grad_step = 000040, loss = 0.052840
grad_step = 000041, loss = 0.048900
grad_step = 000042, loss = 0.044927
grad_step = 000043, loss = 0.041194
grad_step = 000044, loss = 0.037721
grad_step = 000045, loss = 0.034731
grad_step = 000046, loss = 0.032122
grad_step = 000047, loss = 0.029487
grad_step = 000048, loss = 0.026831
grad_step = 000049, loss = 0.024482
grad_step = 000050, loss = 0.022410
grad_step = 000051, loss = 0.020474
grad_step = 000052, loss = 0.018757
grad_step = 000053, loss = 0.017106
grad_step = 000054, loss = 0.015524
grad_step = 000055, loss = 0.014161
grad_step = 000056, loss = 0.012884
grad_step = 000057, loss = 0.011683
grad_step = 000058, loss = 0.010672
grad_step = 000059, loss = 0.009758
grad_step = 000060, loss = 0.008918
grad_step = 000061, loss = 0.008164
grad_step = 000062, loss = 0.007445
grad_step = 000063, loss = 0.006833
grad_step = 000064, loss = 0.006295
grad_step = 000065, loss = 0.005784
grad_step = 000066, loss = 0.005364
grad_step = 000067, loss = 0.004995
grad_step = 000068, loss = 0.004652
grad_step = 000069, loss = 0.004353
grad_step = 000070, loss = 0.004078
grad_step = 000071, loss = 0.003866
grad_step = 000072, loss = 0.003683
grad_step = 000073, loss = 0.003501
grad_step = 000074, loss = 0.003346
grad_step = 000075, loss = 0.003210
grad_step = 000076, loss = 0.003111
grad_step = 000077, loss = 0.003023
grad_step = 000078, loss = 0.002934
grad_step = 000079, loss = 0.002861
grad_step = 000080, loss = 0.002798
grad_step = 000081, loss = 0.002749
grad_step = 000082, loss = 0.002699
grad_step = 000083, loss = 0.002652
grad_step = 000084, loss = 0.002610
grad_step = 000085, loss = 0.002567
grad_step = 000086, loss = 0.002537
grad_step = 000087, loss = 0.002510
grad_step = 000088, loss = 0.002484
grad_step = 000089, loss = 0.002456
grad_step = 000090, loss = 0.002429
grad_step = 000091, loss = 0.002409
grad_step = 000092, loss = 0.002390
grad_step = 000093, loss = 0.002372
grad_step = 000094, loss = 0.002353
grad_step = 000095, loss = 0.002340
grad_step = 000096, loss = 0.002327
grad_step = 000097, loss = 0.002315
grad_step = 000098, loss = 0.002303
grad_step = 000099, loss = 0.002293
grad_step = 000100, loss = 0.002284
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002275
grad_step = 000102, loss = 0.002268
grad_step = 000103, loss = 0.002261
grad_step = 000104, loss = 0.002255
grad_step = 000105, loss = 0.002248
grad_step = 000106, loss = 0.002242
grad_step = 000107, loss = 0.002237
grad_step = 000108, loss = 0.002232
grad_step = 000109, loss = 0.002227
grad_step = 000110, loss = 0.002222
grad_step = 000111, loss = 0.002217
grad_step = 000112, loss = 0.002212
grad_step = 000113, loss = 0.002208
grad_step = 000114, loss = 0.002204
grad_step = 000115, loss = 0.002199
grad_step = 000116, loss = 0.002195
grad_step = 000117, loss = 0.002191
grad_step = 000118, loss = 0.002187
grad_step = 000119, loss = 0.002183
grad_step = 000120, loss = 0.002179
grad_step = 000121, loss = 0.002175
grad_step = 000122, loss = 0.002170
grad_step = 000123, loss = 0.002167
grad_step = 000124, loss = 0.002162
grad_step = 000125, loss = 0.002158
grad_step = 000126, loss = 0.002154
grad_step = 000127, loss = 0.002150
grad_step = 000128, loss = 0.002147
grad_step = 000129, loss = 0.002145
grad_step = 000130, loss = 0.002149
grad_step = 000131, loss = 0.002166
grad_step = 000132, loss = 0.002208
grad_step = 000133, loss = 0.002260
grad_step = 000134, loss = 0.002272
grad_step = 000135, loss = 0.002214
grad_step = 000136, loss = 0.002137
grad_step = 000137, loss = 0.002117
grad_step = 000138, loss = 0.002158
grad_step = 000139, loss = 0.002189
grad_step = 000140, loss = 0.002166
grad_step = 000141, loss = 0.002119
grad_step = 000142, loss = 0.002099
grad_step = 000143, loss = 0.002124
grad_step = 000144, loss = 0.002146
grad_step = 000145, loss = 0.002132
grad_step = 000146, loss = 0.002101
grad_step = 000147, loss = 0.002082
grad_step = 000148, loss = 0.002092
grad_step = 000149, loss = 0.002109
grad_step = 000150, loss = 0.002105
grad_step = 000151, loss = 0.002089
grad_step = 000152, loss = 0.002071
grad_step = 000153, loss = 0.002064
grad_step = 000154, loss = 0.002071
grad_step = 000155, loss = 0.002079
grad_step = 000156, loss = 0.002078
grad_step = 000157, loss = 0.002069
grad_step = 000158, loss = 0.002056
grad_step = 000159, loss = 0.002045
grad_step = 000160, loss = 0.002042
grad_step = 000161, loss = 0.002044
grad_step = 000162, loss = 0.002046
grad_step = 000163, loss = 0.002048
grad_step = 000164, loss = 0.002049
grad_step = 000165, loss = 0.002048
grad_step = 000166, loss = 0.002043
grad_step = 000167, loss = 0.002038
grad_step = 000168, loss = 0.002031
grad_step = 000169, loss = 0.002023
grad_step = 000170, loss = 0.002017
grad_step = 000171, loss = 0.002012
grad_step = 000172, loss = 0.002007
grad_step = 000173, loss = 0.002003
grad_step = 000174, loss = 0.002000
grad_step = 000175, loss = 0.001999
grad_step = 000176, loss = 0.001999
grad_step = 000177, loss = 0.002004
grad_step = 000178, loss = 0.002019
grad_step = 000179, loss = 0.002052
grad_step = 000180, loss = 0.002111
grad_step = 000181, loss = 0.002206
grad_step = 000182, loss = 0.002297
grad_step = 000183, loss = 0.002312
grad_step = 000184, loss = 0.002164
grad_step = 000185, loss = 0.001994
grad_step = 000186, loss = 0.001967
grad_step = 000187, loss = 0.002071
grad_step = 000188, loss = 0.002138
grad_step = 000189, loss = 0.002061
grad_step = 000190, loss = 0.001954
grad_step = 000191, loss = 0.001957
grad_step = 000192, loss = 0.002032
grad_step = 000193, loss = 0.002042
grad_step = 000194, loss = 0.001968
grad_step = 000195, loss = 0.001924
grad_step = 000196, loss = 0.001955
grad_step = 000197, loss = 0.001989
grad_step = 000198, loss = 0.001966
grad_step = 000199, loss = 0.001918
grad_step = 000200, loss = 0.001911
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001939
grad_step = 000202, loss = 0.001946
grad_step = 000203, loss = 0.001918
grad_step = 000204, loss = 0.001892
grad_step = 000205, loss = 0.001896
grad_step = 000206, loss = 0.001913
grad_step = 000207, loss = 0.001912
grad_step = 000208, loss = 0.001893
grad_step = 000209, loss = 0.001875
grad_step = 000210, loss = 0.001874
grad_step = 000211, loss = 0.001882
grad_step = 000212, loss = 0.001884
grad_step = 000213, loss = 0.001875
grad_step = 000214, loss = 0.001861
grad_step = 000215, loss = 0.001852
grad_step = 000216, loss = 0.001852
grad_step = 000217, loss = 0.001855
grad_step = 000218, loss = 0.001856
grad_step = 000219, loss = 0.001852
grad_step = 000220, loss = 0.001845
grad_step = 000221, loss = 0.001837
grad_step = 000222, loss = 0.001830
grad_step = 000223, loss = 0.001825
grad_step = 000224, loss = 0.001821
grad_step = 000225, loss = 0.001819
grad_step = 000226, loss = 0.001818
grad_step = 000227, loss = 0.001818
grad_step = 000228, loss = 0.001819
grad_step = 000229, loss = 0.001825
grad_step = 000230, loss = 0.001837
grad_step = 000231, loss = 0.001857
grad_step = 000232, loss = 0.001893
grad_step = 000233, loss = 0.001955
grad_step = 000234, loss = 0.002015
grad_step = 000235, loss = 0.002070
grad_step = 000236, loss = 0.002039
grad_step = 000237, loss = 0.001925
grad_step = 000238, loss = 0.001810
grad_step = 000239, loss = 0.001782
grad_step = 000240, loss = 0.001838
grad_step = 000241, loss = 0.001902
grad_step = 000242, loss = 0.001919
grad_step = 000243, loss = 0.001861
grad_step = 000244, loss = 0.001786
grad_step = 000245, loss = 0.001764
grad_step = 000246, loss = 0.001799
grad_step = 000247, loss = 0.001841
grad_step = 000248, loss = 0.001848
grad_step = 000249, loss = 0.001817
grad_step = 000250, loss = 0.001768
grad_step = 000251, loss = 0.001746
grad_step = 000252, loss = 0.001757
grad_step = 000253, loss = 0.001781
grad_step = 000254, loss = 0.001797
grad_step = 000255, loss = 0.001796
grad_step = 000256, loss = 0.001773
grad_step = 000257, loss = 0.001784
grad_step = 000258, loss = 0.001803
grad_step = 000259, loss = 0.001746
grad_step = 000260, loss = 0.001798
grad_step = 000261, loss = 0.001787
grad_step = 000262, loss = 0.001795
grad_step = 000263, loss = 0.001766
grad_step = 000264, loss = 0.001751
grad_step = 000265, loss = 0.001800
grad_step = 000266, loss = 0.001761
grad_step = 000267, loss = 0.001776
grad_step = 000268, loss = 0.001741
grad_step = 000269, loss = 0.001763
grad_step = 000270, loss = 0.001723
grad_step = 000271, loss = 0.001756
grad_step = 000272, loss = 0.001761
grad_step = 000273, loss = 0.001760
grad_step = 000274, loss = 0.001738
grad_step = 000275, loss = 0.001754
grad_step = 000276, loss = 0.001794
grad_step = 000277, loss = 0.001781
grad_step = 000278, loss = 0.001840
grad_step = 000279, loss = 0.001854
grad_step = 000280, loss = 0.001940
grad_step = 000281, loss = 0.001968
grad_step = 000282, loss = 0.001947
grad_step = 000283, loss = 0.001824
grad_step = 000284, loss = 0.001737
grad_step = 000285, loss = 0.001735
grad_step = 000286, loss = 0.001813
grad_step = 000287, loss = 0.001867
grad_step = 000288, loss = 0.001816
grad_step = 000289, loss = 0.001733
grad_step = 000290, loss = 0.001709
grad_step = 000291, loss = 0.001749
grad_step = 000292, loss = 0.001770
grad_step = 000293, loss = 0.001792
grad_step = 000294, loss = 0.001753
grad_step = 000295, loss = 0.001744
grad_step = 000296, loss = 0.001692
grad_step = 000297, loss = 0.001697
grad_step = 000298, loss = 0.001694
grad_step = 000299, loss = 0.001709
grad_step = 000300, loss = 0.001710
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001721
grad_step = 000302, loss = 0.001709
grad_step = 000303, loss = 0.001693
grad_step = 000304, loss = 0.001676
grad_step = 000305, loss = 0.001675
grad_step = 000306, loss = 0.001680
grad_step = 000307, loss = 0.001683
grad_step = 000308, loss = 0.001696
grad_step = 000309, loss = 0.001694
grad_step = 000310, loss = 0.001699
grad_step = 000311, loss = 0.001691
grad_step = 000312, loss = 0.001692
grad_step = 000313, loss = 0.001683
grad_step = 000314, loss = 0.001676
grad_step = 000315, loss = 0.001670
grad_step = 000316, loss = 0.001663
grad_step = 000317, loss = 0.001659
grad_step = 000318, loss = 0.001654
grad_step = 000319, loss = 0.001653
grad_step = 000320, loss = 0.001649
grad_step = 000321, loss = 0.001650
grad_step = 000322, loss = 0.001649
grad_step = 000323, loss = 0.001651
grad_step = 000324, loss = 0.001651
grad_step = 000325, loss = 0.001655
grad_step = 000326, loss = 0.001661
grad_step = 000327, loss = 0.001672
grad_step = 000328, loss = 0.001696
grad_step = 000329, loss = 0.001731
grad_step = 000330, loss = 0.001808
grad_step = 000331, loss = 0.001894
grad_step = 000332, loss = 0.002037
grad_step = 000333, loss = 0.002020
grad_step = 000334, loss = 0.001929
grad_step = 000335, loss = 0.001734
grad_step = 000336, loss = 0.001646
grad_step = 000337, loss = 0.001706
grad_step = 000338, loss = 0.001803
grad_step = 000339, loss = 0.001813
grad_step = 000340, loss = 0.001699
grad_step = 000341, loss = 0.001630
grad_step = 000342, loss = 0.001671
grad_step = 000343, loss = 0.001732
grad_step = 000344, loss = 0.001747
grad_step = 000345, loss = 0.001704
grad_step = 000346, loss = 0.001661
grad_step = 000347, loss = 0.001627
grad_step = 000348, loss = 0.001634
grad_step = 000349, loss = 0.001673
grad_step = 000350, loss = 0.001687
grad_step = 000351, loss = 0.001666
grad_step = 000352, loss = 0.001634
grad_step = 000353, loss = 0.001620
grad_step = 000354, loss = 0.001619
grad_step = 000355, loss = 0.001630
grad_step = 000356, loss = 0.001644
grad_step = 000357, loss = 0.001638
grad_step = 000358, loss = 0.001617
grad_step = 000359, loss = 0.001604
grad_step = 000360, loss = 0.001605
grad_step = 000361, loss = 0.001609
grad_step = 000362, loss = 0.001613
grad_step = 000363, loss = 0.001617
grad_step = 000364, loss = 0.001612
grad_step = 000365, loss = 0.001601
grad_step = 000366, loss = 0.001593
grad_step = 000367, loss = 0.001589
grad_step = 000368, loss = 0.001587
grad_step = 000369, loss = 0.001587
grad_step = 000370, loss = 0.001591
grad_step = 000371, loss = 0.001593
grad_step = 000372, loss = 0.001593
grad_step = 000373, loss = 0.001591
grad_step = 000374, loss = 0.001590
grad_step = 000375, loss = 0.001588
grad_step = 000376, loss = 0.001585
grad_step = 000377, loss = 0.001583
grad_step = 000378, loss = 0.001582
grad_step = 000379, loss = 0.001582
grad_step = 000380, loss = 0.001583
grad_step = 000381, loss = 0.001588
grad_step = 000382, loss = 0.001597
grad_step = 000383, loss = 0.001610
grad_step = 000384, loss = 0.001629
grad_step = 000385, loss = 0.001653
grad_step = 000386, loss = 0.001680
grad_step = 000387, loss = 0.001686
grad_step = 000388, loss = 0.001676
grad_step = 000389, loss = 0.001635
grad_step = 000390, loss = 0.001589
grad_step = 000391, loss = 0.001554
grad_step = 000392, loss = 0.001548
grad_step = 000393, loss = 0.001565
grad_step = 000394, loss = 0.001587
grad_step = 000395, loss = 0.001598
grad_step = 000396, loss = 0.001591
grad_step = 000397, loss = 0.001573
grad_step = 000398, loss = 0.001553
grad_step = 000399, loss = 0.001541
grad_step = 000400, loss = 0.001537
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001541
grad_step = 000402, loss = 0.001549
grad_step = 000403, loss = 0.001560
grad_step = 000404, loss = 0.001572
grad_step = 000405, loss = 0.001588
grad_step = 000406, loss = 0.001608
grad_step = 000407, loss = 0.001628
grad_step = 000408, loss = 0.001654
grad_step = 000409, loss = 0.001666
grad_step = 000410, loss = 0.001672
grad_step = 000411, loss = 0.001646
grad_step = 000412, loss = 0.001608
grad_step = 000413, loss = 0.001559
grad_step = 000414, loss = 0.001531
grad_step = 000415, loss = 0.001530
grad_step = 000416, loss = 0.001549
grad_step = 000417, loss = 0.001570
grad_step = 000418, loss = 0.001574
grad_step = 000419, loss = 0.001562
grad_step = 000420, loss = 0.001540
grad_step = 000421, loss = 0.001524
grad_step = 000422, loss = 0.001520
grad_step = 000423, loss = 0.001527
grad_step = 000424, loss = 0.001537
grad_step = 000425, loss = 0.001545
grad_step = 000426, loss = 0.001547
grad_step = 000427, loss = 0.001542
grad_step = 000428, loss = 0.001535
grad_step = 000429, loss = 0.001526
grad_step = 000430, loss = 0.001518
grad_step = 000431, loss = 0.001513
grad_step = 000432, loss = 0.001511
grad_step = 000433, loss = 0.001511
grad_step = 000434, loss = 0.001512
grad_step = 000435, loss = 0.001514
grad_step = 000436, loss = 0.001517
grad_step = 000437, loss = 0.001521
grad_step = 000438, loss = 0.001526
grad_step = 000439, loss = 0.001532
grad_step = 000440, loss = 0.001537
grad_step = 000441, loss = 0.001546
grad_step = 000442, loss = 0.001553
grad_step = 000443, loss = 0.001565
grad_step = 000444, loss = 0.001572
grad_step = 000445, loss = 0.001582
grad_step = 000446, loss = 0.001583
grad_step = 000447, loss = 0.001585
grad_step = 000448, loss = 0.001573
grad_step = 000449, loss = 0.001559
grad_step = 000450, loss = 0.001537
grad_step = 000451, loss = 0.001517
grad_step = 000452, loss = 0.001502
grad_step = 000453, loss = 0.001495
grad_step = 000454, loss = 0.001495
grad_step = 000455, loss = 0.001500
grad_step = 000456, loss = 0.001508
grad_step = 000457, loss = 0.001514
grad_step = 000458, loss = 0.001520
grad_step = 000459, loss = 0.001522
grad_step = 000460, loss = 0.001524
grad_step = 000461, loss = 0.001521
grad_step = 000462, loss = 0.001518
grad_step = 000463, loss = 0.001513
grad_step = 000464, loss = 0.001508
grad_step = 000465, loss = 0.001502
grad_step = 000466, loss = 0.001496
grad_step = 000467, loss = 0.001491
grad_step = 000468, loss = 0.001487
grad_step = 000469, loss = 0.001484
grad_step = 000470, loss = 0.001482
grad_step = 000471, loss = 0.001480
grad_step = 000472, loss = 0.001478
grad_step = 000473, loss = 0.001477
grad_step = 000474, loss = 0.001476
grad_step = 000475, loss = 0.001476
grad_step = 000476, loss = 0.001475
grad_step = 000477, loss = 0.001475
grad_step = 000478, loss = 0.001476
grad_step = 000479, loss = 0.001480
grad_step = 000480, loss = 0.001490
grad_step = 000481, loss = 0.001515
grad_step = 000482, loss = 0.001563
grad_step = 000483, loss = 0.001657
grad_step = 000484, loss = 0.001774
grad_step = 000485, loss = 0.001932
grad_step = 000486, loss = 0.001936
grad_step = 000487, loss = 0.001837
grad_step = 000488, loss = 0.001605
grad_step = 000489, loss = 0.001484
grad_step = 000490, loss = 0.001542
grad_step = 000491, loss = 0.001642
grad_step = 000492, loss = 0.001652
grad_step = 000493, loss = 0.001549
grad_step = 000494, loss = 0.001483
grad_step = 000495, loss = 0.001523
grad_step = 000496, loss = 0.001573
grad_step = 000497, loss = 0.001560
grad_step = 000498, loss = 0.001510
grad_step = 000499, loss = 0.001473
grad_step = 000500, loss = 0.001532
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001589
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

  date_run                              2020-05-12 07:13:38.991509
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.19881
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 07:13:38.998395
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0865692
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 07:13:39.006337
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.126013
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 07:13:39.013903
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -0.31545
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
0   2020-05-12 07:13:03.753683  ...    mean_absolute_error
1   2020-05-12 07:13:03.758316  ...     mean_squared_error
2   2020-05-12 07:13:03.762176  ...  median_absolute_error
3   2020-05-12 07:13:03.766054  ...               r2_score
4   2020-05-12 07:13:13.718112  ...    mean_absolute_error
5   2020-05-12 07:13:13.722449  ...     mean_squared_error
6   2020-05-12 07:13:13.726638  ...  median_absolute_error
7   2020-05-12 07:13:13.730893  ...               r2_score
8   2020-05-12 07:13:38.991509  ...    mean_absolute_error
9   2020-05-12 07:13:38.998395  ...     mean_squared_error
10  2020-05-12 07:13:39.006337  ...  median_absolute_error
11  2020-05-12 07:13:39.013903  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 306312.51it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 395899.71it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 547648.22it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 773817.51it/s] 77%|███████▋  | 7675904/9912422 [00:00<00:02, 1094027.93it/s]9920512it [00:00, 10429402.08it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 144389.73it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 307679.40it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 397255.28it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 549815.07it/s]1654784it [00:00, 2766669.84it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54489.71it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7523e2fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb6efafbbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7523e2fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb6ec8f90b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb704d76e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb704d65e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb704d76e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb704d65e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb704d76e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb6efafbbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7523aaba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb55ce6e1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=116532d2762ad0d7f504dcf28313884553245ff7ae9e2b4821811390f97d6149
  Stored in directory: /tmp/pip-ephem-wheel-cache-90s5vciv/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb4f4a56198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  212992/17464789 [..............................] - ETA: 21s
  417792/17464789 [..............................] - ETA: 13s
  835584/17464789 [>.............................] - ETA: 7s 
 1703936/17464789 [=>............................] - ETA: 4s
 3358720/17464789 [====>.........................] - ETA: 2s
 6094848/17464789 [=========>....................] - ETA: 1s
 8863744/17464789 [==============>...............] - ETA: 0s
11894784/17464789 [===================>..........] - ETA: 0s
14942208/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 07:15:10.950740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 07:15:10.954313: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 07:15:10.954927: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56238aa55290 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 07:15:10.954942: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6935 - accuracy: 0.4983
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7862 - accuracy: 0.4922
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7254 - accuracy: 0.4962
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7041 - accuracy: 0.4976
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7019 - accuracy: 0.4977
11000/25000 [============>.................] - ETA: 4s - loss: 7.6610 - accuracy: 0.5004
12000/25000 [=============>................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6419 - accuracy: 0.5016
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6392 - accuracy: 0.5018
15000/25000 [=================>............] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6862 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6613 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6446 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 07:15:28.700116
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 07:15:28.700116  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 07:15:35.554630: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 07:15:35.560383: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 07:15:35.560546: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566898e48d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 07:15:35.560571: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fda83e45be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0876 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.9313 - val_crf_viterbi_accuracy: 0.0133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fda83e45cf8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8276 - accuracy: 0.4895
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7893 - accuracy: 0.4920 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7855 - accuracy: 0.4922
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8384 - accuracy: 0.4888
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8276 - accuracy: 0.4895
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
11000/25000 [============>.................] - ETA: 4s - loss: 7.7084 - accuracy: 0.4973
12000/25000 [=============>................] - ETA: 4s - loss: 7.7254 - accuracy: 0.4962
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7386 - accuracy: 0.4953
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7510 - accuracy: 0.4945
15000/25000 [=================>............] - ETA: 3s - loss: 7.7494 - accuracy: 0.4946
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7251 - accuracy: 0.4962
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7081 - accuracy: 0.4973
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7143 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6941 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 10s 404us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fda5c438a20> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:24:09, 25.5kB/s].vector_cache/glove.6B.zip:   0%|          | 360k/862M [00:00<6:35:58, 36.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.49M/862M [00:00<4:36:18, 51.8kB/s].vector_cache/glove.6B.zip:   1%|          | 7.88M/862M [00:00<3:12:31, 74.0kB/s].vector_cache/glove.6B.zip:   1%|          | 10.8M/862M [00:00<2:14:27, 106kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:00<1:33:30, 151kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:00<1:04:51, 215kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:01<45:10, 307kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:01<31:34, 437kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:01<21:57, 622kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:01<15:16, 886kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:01<11:40, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:03<10:02, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<08:57, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:04<06:23, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:05<11:09, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<09:43, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:06<06:56, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:07<10:50, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:08<09:17, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:08<06:34, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:09<34:06, 388kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<25:51, 511kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:10<18:21, 719kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:11<15:32, 846kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:11<12:39, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.7M/862M [00:12<09:01, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:13<10:17, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:13<09:08, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:14<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:15<08:02, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:15<07:28, 1.74MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:15<05:21, 2.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:17<09:41, 1.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:17<08:41, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:17<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:19<08:25, 1.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:19<07:53, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:19<05:38, 2.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:21<09:12, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:21<08:29, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:21<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:23<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:23<07:56, 1.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<05:59, 2.12MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<9:08:52, 23.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<6:23:58, 33.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<4:30:03, 46.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<3:09:58, 66.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<2:14:12, 93.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<1:35:01, 132kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<1:08:13, 183kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<49:05, 253kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<35:32, 349kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:33<24:51, 496kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:04:19, 192kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<46:06, 267kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<33:55, 361kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<24:36, 498kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:37<17:15, 707kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<33:24, 365kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<24:22, 500kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:39<17:04, 710kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<1:01:01, 199kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<43:42, 277kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<32:14, 374kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<23:38, 510kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:43<16:33, 724kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:16:19, 157kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<54:55, 218kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<38:23, 311kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<36:14, 329kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<26:38, 447kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:47<18:39, 635kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<45:02, 263kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<32:24, 365kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:49<22:40, 520kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<47:39, 247kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<34:53, 338kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<24:33, 479kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<20:06, 582kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<15:43, 745kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<11:05, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<13:14, 879kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<10:36, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<07:29, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<16:46, 690kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<12:59, 891kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:57<09:08, 1.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<47:46, 241kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<34:36, 332kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<24:13, 473kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<25:20, 451kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<19:21, 591kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:00<13:36, 836kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<14:50, 766kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<11:17, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:02<08:00, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<10:44, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<08:25, 1.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:04<05:57, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<30:09, 373kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<21:55, 512kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<16:57, 658kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<12:38, 883kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:08<08:56, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<08:40, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<8:22:51, 22.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<5:51:53, 31.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<4:07:02, 44.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<2:54:02, 63.4kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:12<2:01:18, 90.5kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<1:36:28, 114kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<1:08:41, 160kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<49:23, 221kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<35:23, 308kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<26:15, 412kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<19:27, 556kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<15:06, 712kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<11:23, 944kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:20<08:01, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<42:02, 254kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<30:52, 346kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:22<21:34, 493kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<35:20, 301kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<25:26, 417kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:24<17:47, 593kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<1:09:23, 152kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<50:03, 211kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<34:57, 300kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<33:01, 317kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<23:59, 437kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<16:49, 620kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<18:25, 565kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<14:01, 743kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:30<09:51, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<15:15, 678kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<11:28, 900kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<09:32, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<08:04, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:34<05:42, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<14:19, 713kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<10:41, 955kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<08:58, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:08, 1.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<06:28, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<05:29, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:40<03:54, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<20:18, 493kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<15:37, 640kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:42<10:57, 907kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<20:32, 484kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<15:37, 636kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<10:57, 901kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<20:25, 483kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<15:08, 651kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:45<10:38, 922kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<16:45, 585kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<13:07, 746kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:47<09:13, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<13:18, 731kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<10:40, 911kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:49<07:31, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<12:52, 750kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<10:24, 928kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:51<07:20, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<11:04, 866kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<08:36, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<06:18, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:13:53, 22.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<5:03:31, 31.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<3:32:55, 44.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<2:30:23, 62.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<1:44:55, 89.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<1:16:46, 122kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<55:03, 171kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<38:28, 243kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<30:14, 308kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<22:31, 414kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:01<15:47, 587kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<15:22, 602kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<12:05, 765kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:03<08:32, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<09:44, 943kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<08:09, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<05:49, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:45, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:04, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<04:22, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<05:19, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<05:01, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<03:38, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:45, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:27, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:55, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:40, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:22, 2.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:07, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:49, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<03:28, 2.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:19, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:59, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<03:35, 2.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:25, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:04, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:40, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:01, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:11, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:24, 1.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:34, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:35, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<05:49, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<04:19, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:25<03:05, 2.72MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<44:52, 188kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<33:15, 253kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<23:23, 359kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<18:00, 464kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<14:04, 593kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<10:03, 828kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:29<07:05, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<50:54, 163kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<36:55, 224kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:31<25:58, 318kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<19:45, 416kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<15:01, 547kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<10:48, 759kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:33<07:37, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<12:19, 662kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<09:51, 827kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<07:14, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<05:13, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<05:52, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<05:18, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:54, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:37<02:49, 2.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<16:20, 490kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<12:10, 658kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<08:39, 923kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<07:53, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<06:39, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<04:49, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<03:28, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<10:03, 783kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<08:12, 959kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<05:50, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<06:12, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<05:52, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:44<04:13, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:44<03:04, 2.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<12:22, 625kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<09:48, 789kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:46<06:57, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<05:03, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<08:28, 905kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<06:29, 1.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:48<04:37, 1.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<04:00, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<5:12:13, 24.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<3:38:15, 34.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:50<2:31:56, 49.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<1:58:54, 63.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<1:23:46, 89.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:52<58:24, 128kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<44:04, 170kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<31:26, 237kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:54<21:58, 338kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<18:25, 402kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<13:31, 547kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:56<09:29, 775kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<10:35, 692kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<07:55, 924kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<05:36, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<06:50, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<05:22, 1.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:00<03:48, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<07:10, 1.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<05:38, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:02<04:00, 1.79MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<06:39, 1.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<05:15, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:04<03:44, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<06:25, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:03, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<03:36, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:11, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:08, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:08<02:58, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:47, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:54, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<02:47, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<05:51, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:41, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<03:20, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<06:04, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<03:24, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<06:13, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<04:53, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:16<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:17<06:09, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<04:50, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:18<03:26, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<06:24, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<05:15, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<03:44, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<04:43, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<03:45, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<02:42, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<04:13, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<03:44, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:23<02:41, 2.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<03:18, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:25<02:21, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<05:20, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<04:11, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:27<03:00, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<04:12, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:29<02:31, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:43, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<03:06, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:31<02:15, 2.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<02:00, 3.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<4:21:51, 23.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<3:03:09, 33.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:33<2:07:16, 47.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<1:35:17, 63.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<1:07:04, 89.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:35<46:43, 128kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<34:48, 171kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<24:49, 240kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:37<17:17, 342kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<16:13, 364kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<12:10, 484kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:39<08:32, 686kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<07:56, 734kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<06:23, 912kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<04:31, 1.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:03, 1.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:03, 1.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<02:52, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<04:26, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:51, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:44, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<04:13, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:43, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<02:39, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:45, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:09, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:49<02:15, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:37, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:57, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<02:07, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:17, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<02:23, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:53<01:44, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<19:19, 277kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<14:00, 381kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<09:51, 540kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:55<06:55, 762kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<31:40, 167kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<22:34, 234kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<15:46, 332kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<12:38, 412kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<09:27, 550kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<06:40, 775kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [03:59<04:42, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<1:06:03, 77.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<46:35, 110kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<32:28, 157kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<24:14, 209kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<17:23, 291kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<12:10, 414kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<10:01, 499kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<07:24, 675kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:04<05:14, 948kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:05<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<1:42:52, 48.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<1:12:15, 68.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:06<50:22, 97.3kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<36:18, 134kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<25:48, 188kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:08<18:02, 268kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<13:45, 349kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<09:58, 481kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<07:01, 678kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<06:04, 779kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<04:38, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:12<03:17, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<02:51, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<3:08:06, 24.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<2:11:30, 35.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:14<1:31:22, 50.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<1:05:51, 69.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<46:24, 99.1kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:16<32:15, 141kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<24:15, 187kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<17:16, 262kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<12:06, 372kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<09:33, 468kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<07:02, 633kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<04:57, 893kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<04:53, 901kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<03:56, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<02:48, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:02, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:47, 1.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<02:01, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:24<01:27, 2.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<33:26, 127kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<24:01, 177kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<16:46, 252kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<12:40, 331kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<09:31, 440kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<06:41, 622kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<05:38, 730kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<04:33, 904kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<03:14, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:30<02:18, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:30<02:14, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<4:08:47, 16.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<2:54:36, 23.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<2:01:24, 33.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<1:25:11, 46.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<1:00:09, 66.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<41:52, 94.2kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<29:58, 130kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<21:21, 183kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<14:54, 260kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<11:20, 338kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<08:14, 464kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<05:46, 658kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<05:19, 708kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<04:15, 885kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<03:01, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:01, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:37, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<01:53, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:42<01:21, 2.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<3:42:32, 16.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<2:35:58, 23.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<1:48:20, 33.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<1:16:02, 46.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<53:43, 66.3kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<37:22, 94.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:46<26:01, 135kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<20:48, 168kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<15:06, 231kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<10:32, 328kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:48<07:22, 465kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<09:23, 365kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<07:29, 457kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<05:17, 642kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:50<03:42, 907kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<06:00, 559kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<04:42, 711kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<03:19, 999kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<03:08, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:39, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:53, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<02:11, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:22, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:43, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:56<01:13, 2.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<48:45, 64.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<35:04, 89.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<24:27, 128kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:58<16:56, 182kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<16:12, 190kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<12:00, 256kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:00<08:23, 364kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<06:27, 466kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<05:11, 579kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<03:39, 815kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<03:10, 928kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<03:10, 928kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:16, 1.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<01:36, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<04:17, 669kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<03:35, 799kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:05<02:32, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<02:24, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<02:14, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:07<01:35, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:44, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:45, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:09<01:15, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:30, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:40, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<01:13, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<00:58, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:44:01, 25.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:12:41, 35.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<50:33, 51.1kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<35:25, 71.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<25:08, 101kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:15<17:28, 144kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<12:35, 197kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<09:08, 270kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<06:24, 382kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<04:53, 491kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<03:37, 661kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<02:35, 920kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:14, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:47, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:19, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:21<00:56, 2.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<04:11, 540kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<03:06, 729kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<02:12, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:58, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:33, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<01:07, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:19, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:08, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<00:49, 2.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:27<00:36, 3.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<04:54, 420kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<03:37, 566kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<02:31, 799kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<02:22, 839kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:48, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:31<01:15, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<01:39, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:18, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<00:55, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:24, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:11, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:16, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:05, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<01:12, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<01:00, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:42, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<01:18, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:47, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<01:15, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<01:02, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<00:43, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<01:17, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<01:02, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:44<00:43, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<01:19, 1.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:46<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<01:13, 1.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<01:01, 1.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:48<00:42, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<01:06, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:56, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:50<00:38, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<01:00, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:48, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:34, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<49:57, 23.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<34:43, 33.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<23:15, 47.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<16:20, 67.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:56<10:55, 96.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<08:12, 127kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<05:51, 177kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:58<03:53, 252kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<03:27, 281kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<02:31, 384kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:00<01:40, 545kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<01:43, 522kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<01:18, 683kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:02<00:52, 967kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:05, 765kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:53, 940kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:04<00:35, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:44, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:37, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:06<00:25, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:37, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:30, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:08<00:20, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:31, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:27, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:10<00:18, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:28, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:24, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:12<00:15, 1.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:25, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:20, 1.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:16, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:13, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<00:08, 2.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:13, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:11, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:07, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:07, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:07, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:25, 510kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:13, 668kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:09, 899kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:23<00:03, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:15, 297kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:10, 411kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:25<00:03, 582kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 634kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 844kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 732/400000 [00:00<00:54, 7315.75it/s]  0%|          | 1463/400000 [00:00<00:54, 7313.63it/s]  1%|          | 2204/400000 [00:00<00:54, 7340.67it/s]  1%|          | 2924/400000 [00:00<00:54, 7297.02it/s]  1%|          | 3634/400000 [00:00<00:54, 7235.52it/s]  1%|          | 4357/400000 [00:00<00:54, 7231.54it/s]  1%|▏         | 5078/400000 [00:00<00:54, 7222.53it/s]  1%|▏         | 5796/400000 [00:00<00:54, 7207.21it/s]  2%|▏         | 6475/400000 [00:00<00:56, 6912.03it/s]  2%|▏         | 7209/400000 [00:01<00:55, 7034.36it/s]  2%|▏         | 7945/400000 [00:01<00:55, 7126.45it/s]  2%|▏         | 8666/400000 [00:01<00:54, 7148.88it/s]  2%|▏         | 9398/400000 [00:01<00:54, 7198.02it/s]  3%|▎         | 10157/400000 [00:01<00:53, 7309.93it/s]  3%|▎         | 10899/400000 [00:01<00:53, 7341.23it/s]  3%|▎         | 11702/400000 [00:01<00:51, 7534.29it/s]  3%|▎         | 12455/400000 [00:01<00:52, 7355.09it/s]  3%|▎         | 13192/400000 [00:01<00:53, 7268.61it/s]  3%|▎         | 13920/400000 [00:01<00:54, 7099.77it/s]  4%|▎         | 14658/400000 [00:02<00:53, 7179.70it/s]  4%|▍         | 15395/400000 [00:02<00:53, 7235.22it/s]  4%|▍         | 16120/400000 [00:02<00:53, 7172.06it/s]  4%|▍         | 16838/400000 [00:02<00:53, 7113.91it/s]  4%|▍         | 17600/400000 [00:02<00:52, 7256.42it/s]  5%|▍         | 18353/400000 [00:02<00:52, 7334.75it/s]  5%|▍         | 19144/400000 [00:02<00:50, 7497.39it/s]  5%|▍         | 19921/400000 [00:02<00:50, 7574.99it/s]  5%|▌         | 20680/400000 [00:02<00:50, 7477.11it/s]  5%|▌         | 21429/400000 [00:02<00:52, 7270.28it/s]  6%|▌         | 22192/400000 [00:03<00:51, 7373.18it/s]  6%|▌         | 22978/400000 [00:03<00:50, 7512.49it/s]  6%|▌         | 23732/400000 [00:03<00:50, 7397.88it/s]  6%|▌         | 24482/400000 [00:03<00:50, 7424.85it/s]  6%|▋         | 25226/400000 [00:03<00:51, 7261.53it/s]  6%|▋         | 25954/400000 [00:03<00:52, 7096.14it/s]  7%|▋         | 26666/400000 [00:03<00:53, 6975.54it/s]  7%|▋         | 27400/400000 [00:03<00:52, 7080.73it/s]  7%|▋         | 28110/400000 [00:03<00:52, 7084.33it/s]  7%|▋         | 28855/400000 [00:03<00:51, 7186.93it/s]  7%|▋         | 29598/400000 [00:04<00:51, 7256.07it/s]  8%|▊         | 30354/400000 [00:04<00:50, 7342.58it/s]  8%|▊         | 31090/400000 [00:04<00:51, 7233.42it/s]  8%|▊         | 31835/400000 [00:04<00:50, 7296.85it/s]  8%|▊         | 32575/400000 [00:04<00:50, 7325.69it/s]  8%|▊         | 33350/400000 [00:04<00:49, 7447.37it/s]  9%|▊         | 34109/400000 [00:04<00:48, 7489.19it/s]  9%|▊         | 34859/400000 [00:04<00:48, 7475.19it/s]  9%|▉         | 35608/400000 [00:04<00:49, 7384.12it/s]  9%|▉         | 36348/400000 [00:04<00:50, 7230.66it/s]  9%|▉         | 37073/400000 [00:05<00:50, 7219.80it/s]  9%|▉         | 37809/400000 [00:05<00:49, 7252.15it/s] 10%|▉         | 38553/400000 [00:05<00:49, 7306.33it/s] 10%|▉         | 39285/400000 [00:05<00:49, 7221.09it/s] 10%|█         | 40011/400000 [00:05<00:49, 7232.36it/s] 10%|█         | 40802/400000 [00:05<00:48, 7421.90it/s] 10%|█         | 41557/400000 [00:05<00:48, 7459.66it/s] 11%|█         | 42305/400000 [00:05<00:47, 7461.84it/s] 11%|█         | 43078/400000 [00:05<00:47, 7539.24it/s] 11%|█         | 43833/400000 [00:06<00:47, 7422.76it/s] 11%|█         | 44608/400000 [00:06<00:47, 7517.09it/s] 11%|█▏        | 45390/400000 [00:06<00:46, 7602.44it/s] 12%|█▏        | 46152/400000 [00:06<00:46, 7591.64it/s] 12%|█▏        | 46912/400000 [00:06<00:46, 7553.07it/s] 12%|█▏        | 47668/400000 [00:06<00:48, 7334.63it/s] 12%|█▏        | 48404/400000 [00:06<00:48, 7253.72it/s] 12%|█▏        | 49137/400000 [00:06<00:48, 7273.50it/s] 12%|█▏        | 49869/400000 [00:06<00:48, 7287.09it/s] 13%|█▎        | 50624/400000 [00:06<00:47, 7362.16it/s] 13%|█▎        | 51361/400000 [00:07<00:47, 7327.26it/s] 13%|█▎        | 52128/400000 [00:07<00:46, 7425.23it/s] 13%|█▎        | 52903/400000 [00:07<00:46, 7514.28it/s] 13%|█▎        | 53656/400000 [00:07<00:46, 7471.92it/s] 14%|█▎        | 54465/400000 [00:07<00:45, 7647.00it/s] 14%|█▍        | 55232/400000 [00:07<00:45, 7526.76it/s] 14%|█▍        | 55987/400000 [00:07<00:46, 7390.97it/s] 14%|█▍        | 56728/400000 [00:07<00:46, 7383.15it/s] 14%|█▍        | 57468/400000 [00:07<00:47, 7221.30it/s] 15%|█▍        | 58192/400000 [00:07<00:47, 7186.68it/s] 15%|█▍        | 58912/400000 [00:08<00:47, 7141.50it/s] 15%|█▍        | 59635/400000 [00:08<00:47, 7166.02it/s] 15%|█▌        | 60370/400000 [00:08<00:47, 7217.75it/s] 15%|█▌        | 61106/400000 [00:08<00:46, 7258.20it/s] 15%|█▌        | 61868/400000 [00:08<00:45, 7361.48it/s] 16%|█▌        | 62605/400000 [00:08<00:46, 7258.00it/s] 16%|█▌        | 63386/400000 [00:08<00:45, 7412.29it/s] 16%|█▌        | 64129/400000 [00:08<00:47, 7115.39it/s] 16%|█▌        | 64845/400000 [00:08<00:48, 6951.56it/s] 16%|█▋        | 65603/400000 [00:08<00:46, 7126.87it/s] 17%|█▋        | 66358/400000 [00:09<00:46, 7246.50it/s] 17%|█▋        | 67108/400000 [00:09<00:45, 7318.62it/s] 17%|█▋        | 67843/400000 [00:09<00:47, 7009.81it/s] 17%|█▋        | 68549/400000 [00:09<00:47, 6934.32it/s] 17%|█▋        | 69269/400000 [00:09<00:47, 7011.75it/s] 17%|█▋        | 69984/400000 [00:09<00:46, 7052.64it/s] 18%|█▊        | 70722/400000 [00:09<00:46, 7146.73it/s] 18%|█▊        | 71442/400000 [00:09<00:45, 7162.20it/s] 18%|█▊        | 72274/400000 [00:09<00:43, 7472.56it/s] 18%|█▊        | 73066/400000 [00:10<00:43, 7599.69it/s] 18%|█▊        | 73830/400000 [00:10<00:42, 7611.08it/s] 19%|█▊        | 74606/400000 [00:10<00:42, 7654.89it/s] 19%|█▉        | 75397/400000 [00:10<00:42, 7727.51it/s] 19%|█▉        | 76177/400000 [00:10<00:41, 7747.34it/s] 19%|█▉        | 76953/400000 [00:10<00:42, 7651.84it/s] 19%|█▉        | 77720/400000 [00:10<00:42, 7571.72it/s] 20%|█▉        | 78479/400000 [00:10<00:42, 7557.94it/s] 20%|█▉        | 79236/400000 [00:10<00:43, 7455.83it/s] 20%|█▉        | 79983/400000 [00:10<00:43, 7350.57it/s] 20%|██        | 80719/400000 [00:11<00:43, 7285.89it/s] 20%|██        | 81449/400000 [00:11<00:44, 7191.79it/s] 21%|██        | 82169/400000 [00:11<00:44, 7125.09it/s] 21%|██        | 82929/400000 [00:11<00:43, 7260.13it/s] 21%|██        | 83668/400000 [00:11<00:43, 7296.16it/s] 21%|██        | 84399/400000 [00:11<00:43, 7212.66it/s] 21%|██▏       | 85135/400000 [00:11<00:43, 7253.80it/s] 21%|██▏       | 85878/400000 [00:11<00:43, 7304.89it/s] 22%|██▏       | 86610/400000 [00:11<00:42, 7294.24it/s] 22%|██▏       | 87392/400000 [00:11<00:41, 7444.05it/s] 22%|██▏       | 88180/400000 [00:12<00:41, 7568.88it/s] 22%|██▏       | 88939/400000 [00:12<00:41, 7507.99it/s] 22%|██▏       | 89691/400000 [00:12<00:42, 7385.10it/s] 23%|██▎       | 90431/400000 [00:12<00:42, 7224.91it/s] 23%|██▎       | 91156/400000 [00:12<00:42, 7219.34it/s] 23%|██▎       | 91880/400000 [00:12<00:42, 7176.64it/s] 23%|██▎       | 92620/400000 [00:12<00:42, 7241.99it/s] 23%|██▎       | 93365/400000 [00:12<00:41, 7301.06it/s] 24%|██▎       | 94096/400000 [00:12<00:42, 7253.17it/s] 24%|██▎       | 94841/400000 [00:12<00:41, 7309.14it/s] 24%|██▍       | 95594/400000 [00:13<00:41, 7373.37it/s] 24%|██▍       | 96375/400000 [00:13<00:40, 7498.23it/s] 24%|██▍       | 97126/400000 [00:13<00:40, 7486.65it/s] 24%|██▍       | 97882/400000 [00:13<00:40, 7507.58it/s] 25%|██▍       | 98646/400000 [00:13<00:39, 7544.70it/s] 25%|██▍       | 99401/400000 [00:13<00:39, 7545.99it/s] 25%|██▌       | 100156/400000 [00:13<00:39, 7544.05it/s] 25%|██▌       | 100911/400000 [00:13<00:40, 7365.18it/s] 25%|██▌       | 101649/400000 [00:13<00:41, 7155.16it/s] 26%|██▌       | 102367/400000 [00:13<00:41, 7107.99it/s] 26%|██▌       | 103080/400000 [00:14<00:41, 7088.27it/s] 26%|██▌       | 103850/400000 [00:14<00:40, 7260.47it/s] 26%|██▌       | 104578/400000 [00:14<00:40, 7221.09it/s] 26%|██▋       | 105316/400000 [00:14<00:40, 7267.36it/s] 27%|██▋       | 106066/400000 [00:14<00:40, 7333.38it/s] 27%|██▋       | 106850/400000 [00:14<00:39, 7477.07it/s] 27%|██▋       | 107632/400000 [00:14<00:38, 7574.91it/s] 27%|██▋       | 108391/400000 [00:14<00:38, 7544.29it/s] 27%|██▋       | 109147/400000 [00:14<00:39, 7319.57it/s] 27%|██▋       | 109882/400000 [00:14<00:39, 7322.26it/s] 28%|██▊       | 110616/400000 [00:15<00:39, 7297.11it/s] 28%|██▊       | 111357/400000 [00:15<00:39, 7328.83it/s] 28%|██▊       | 112091/400000 [00:15<00:39, 7211.53it/s] 28%|██▊       | 112820/400000 [00:15<00:39, 7234.53it/s] 28%|██▊       | 113545/400000 [00:15<00:39, 7167.36it/s] 29%|██▊       | 114308/400000 [00:15<00:39, 7298.99it/s] 29%|██▉       | 115096/400000 [00:15<00:38, 7461.56it/s] 29%|██▉       | 115851/400000 [00:15<00:37, 7485.50it/s] 29%|██▉       | 116619/400000 [00:15<00:37, 7542.76it/s] 29%|██▉       | 117375/400000 [00:16<00:37, 7541.98it/s] 30%|██▉       | 118200/400000 [00:16<00:36, 7740.20it/s] 30%|██▉       | 118976/400000 [00:16<00:36, 7662.65it/s] 30%|██▉       | 119744/400000 [00:16<00:36, 7645.43it/s] 30%|███       | 120544/400000 [00:16<00:36, 7748.06it/s] 30%|███       | 121320/400000 [00:16<00:36, 7590.52it/s] 31%|███       | 122081/400000 [00:16<00:38, 7285.81it/s] 31%|███       | 122814/400000 [00:16<00:38, 7230.67it/s] 31%|███       | 123540/400000 [00:16<00:39, 7054.83it/s] 31%|███       | 124260/400000 [00:16<00:38, 7096.84it/s] 31%|███       | 124993/400000 [00:17<00:38, 7164.22it/s] 31%|███▏      | 125733/400000 [00:17<00:37, 7232.72it/s] 32%|███▏      | 126507/400000 [00:17<00:37, 7375.59it/s] 32%|███▏      | 127331/400000 [00:17<00:35, 7613.73it/s] 32%|███▏      | 128102/400000 [00:17<00:35, 7641.18it/s] 32%|███▏      | 128869/400000 [00:17<00:35, 7575.80it/s] 32%|███▏      | 129629/400000 [00:17<00:35, 7577.68it/s] 33%|███▎      | 130388/400000 [00:17<00:35, 7560.24it/s] 33%|███▎      | 131145/400000 [00:17<00:36, 7444.58it/s] 33%|███▎      | 131891/400000 [00:17<00:36, 7424.08it/s] 33%|███▎      | 132662/400000 [00:18<00:35, 7505.39it/s] 33%|███▎      | 133414/400000 [00:18<00:35, 7408.26it/s] 34%|███▎      | 134156/400000 [00:18<00:36, 7379.23it/s] 34%|███▎      | 134909/400000 [00:18<00:35, 7422.98it/s] 34%|███▍      | 135669/400000 [00:18<00:35, 7475.02it/s] 34%|███▍      | 136434/400000 [00:18<00:35, 7523.85it/s] 34%|███▍      | 137224/400000 [00:18<00:34, 7630.14it/s] 34%|███▍      | 137988/400000 [00:18<00:34, 7545.91it/s] 35%|███▍      | 138744/400000 [00:18<00:34, 7540.54it/s] 35%|███▍      | 139506/400000 [00:18<00:34, 7562.48it/s] 35%|███▌      | 140284/400000 [00:19<00:34, 7625.28it/s] 35%|███▌      | 141047/400000 [00:19<00:33, 7618.82it/s] 35%|███▌      | 141810/400000 [00:19<00:34, 7532.84it/s] 36%|███▌      | 142564/400000 [00:19<00:34, 7451.24it/s] 36%|███▌      | 143335/400000 [00:19<00:34, 7527.01it/s] 36%|███▌      | 144089/400000 [00:19<00:34, 7442.86it/s] 36%|███▌      | 144834/400000 [00:19<00:34, 7363.51it/s] 36%|███▋      | 145571/400000 [00:19<00:34, 7336.21it/s] 37%|███▋      | 146306/400000 [00:19<00:34, 7289.09it/s] 37%|███▋      | 147043/400000 [00:19<00:34, 7311.79it/s] 37%|███▋      | 147793/400000 [00:20<00:34, 7365.08it/s] 37%|███▋      | 148533/400000 [00:20<00:34, 7375.12it/s] 37%|███▋      | 149292/400000 [00:20<00:33, 7436.85it/s] 38%|███▊      | 150070/400000 [00:20<00:33, 7534.03it/s] 38%|███▊      | 150824/400000 [00:20<00:33, 7479.75it/s] 38%|███▊      | 151593/400000 [00:20<00:32, 7539.81it/s] 38%|███▊      | 152370/400000 [00:20<00:32, 7606.48it/s] 38%|███▊      | 153132/400000 [00:20<00:32, 7569.89it/s] 38%|███▊      | 153890/400000 [00:20<00:32, 7496.60it/s] 39%|███▊      | 154641/400000 [00:20<00:32, 7451.75it/s] 39%|███▉      | 155387/400000 [00:21<00:33, 7407.52it/s] 39%|███▉      | 156130/400000 [00:21<00:32, 7414.08it/s] 39%|███▉      | 156882/400000 [00:21<00:32, 7442.45it/s] 39%|███▉      | 157699/400000 [00:21<00:31, 7646.00it/s] 40%|███▉      | 158466/400000 [00:21<00:31, 7624.63it/s] 40%|███▉      | 159281/400000 [00:21<00:30, 7772.26it/s] 40%|████      | 160060/400000 [00:21<00:31, 7725.72it/s] 40%|████      | 160834/400000 [00:21<00:31, 7679.49it/s] 40%|████      | 161605/400000 [00:21<00:31, 7686.55it/s] 41%|████      | 162405/400000 [00:22<00:30, 7777.84it/s] 41%|████      | 163200/400000 [00:22<00:30, 7828.46it/s] 41%|████      | 163984/400000 [00:22<00:30, 7783.03it/s] 41%|████      | 164763/400000 [00:22<00:30, 7770.28it/s] 41%|████▏     | 165541/400000 [00:22<00:30, 7600.37it/s] 42%|████▏     | 166303/400000 [00:22<00:31, 7474.77it/s] 42%|████▏     | 167052/400000 [00:22<00:31, 7399.54it/s] 42%|████▏     | 167793/400000 [00:22<00:32, 7255.51it/s] 42%|████▏     | 168537/400000 [00:22<00:31, 7308.87it/s] 42%|████▏     | 169317/400000 [00:22<00:30, 7445.14it/s] 43%|████▎     | 170064/400000 [00:23<00:30, 7451.08it/s] 43%|████▎     | 170850/400000 [00:23<00:30, 7567.55it/s] 43%|████▎     | 171608/400000 [00:23<00:30, 7443.05it/s] 43%|████▎     | 172387/400000 [00:23<00:30, 7542.82it/s] 43%|████▎     | 173156/400000 [00:23<00:29, 7586.03it/s] 43%|████▎     | 173934/400000 [00:23<00:29, 7643.04it/s] 44%|████▎     | 174756/400000 [00:23<00:28, 7806.39it/s] 44%|████▍     | 175539/400000 [00:23<00:29, 7551.15it/s] 44%|████▍     | 176297/400000 [00:23<00:30, 7451.77it/s] 44%|████▍     | 177045/400000 [00:23<00:30, 7285.96it/s] 44%|████▍     | 177776/400000 [00:24<00:30, 7276.79it/s] 45%|████▍     | 178506/400000 [00:24<00:30, 7236.40it/s] 45%|████▍     | 179233/400000 [00:24<00:30, 7245.26it/s] 45%|████▍     | 179978/400000 [00:24<00:30, 7303.49it/s] 45%|████▌     | 180727/400000 [00:24<00:29, 7357.41it/s] 45%|████▌     | 181520/400000 [00:24<00:29, 7519.03it/s] 46%|████▌     | 182302/400000 [00:24<00:28, 7604.91it/s] 46%|████▌     | 183064/400000 [00:24<00:28, 7562.25it/s] 46%|████▌     | 183825/400000 [00:24<00:28, 7575.19it/s] 46%|████▌     | 184584/400000 [00:24<00:28, 7484.43it/s] 46%|████▋     | 185340/400000 [00:25<00:28, 7506.58it/s] 47%|████▋     | 186096/400000 [00:25<00:28, 7520.70it/s] 47%|████▋     | 186849/400000 [00:25<00:28, 7481.58it/s] 47%|████▋     | 187598/400000 [00:25<00:28, 7398.48it/s] 47%|████▋     | 188339/400000 [00:25<00:29, 7169.43it/s] 47%|████▋     | 189072/400000 [00:25<00:29, 7216.02it/s] 47%|████▋     | 189812/400000 [00:25<00:28, 7268.95it/s] 48%|████▊     | 190560/400000 [00:25<00:28, 7329.43it/s] 48%|████▊     | 191354/400000 [00:25<00:27, 7500.92it/s] 48%|████▊     | 192128/400000 [00:25<00:27, 7570.62it/s] 48%|████▊     | 192923/400000 [00:26<00:26, 7679.44it/s] 48%|████▊     | 193711/400000 [00:26<00:26, 7738.52it/s] 49%|████▊     | 194486/400000 [00:26<00:26, 7632.20it/s] 49%|████▉     | 195256/400000 [00:26<00:26, 7652.36it/s] 49%|████▉     | 196022/400000 [00:26<00:26, 7633.89it/s] 49%|████▉     | 196786/400000 [00:26<00:26, 7581.97it/s] 49%|████▉     | 197553/400000 [00:26<00:26, 7602.81it/s] 50%|████▉     | 198314/400000 [00:26<00:27, 7297.31it/s] 50%|████▉     | 199047/400000 [00:26<00:27, 7276.26it/s] 50%|████▉     | 199782/400000 [00:27<00:27, 7296.36it/s] 50%|█████     | 200514/400000 [00:27<00:27, 7285.72it/s] 50%|█████     | 201287/400000 [00:27<00:26, 7411.27it/s] 51%|█████     | 202036/400000 [00:27<00:26, 7432.69it/s] 51%|█████     | 202841/400000 [00:27<00:25, 7607.45it/s] 51%|█████     | 203604/400000 [00:27<00:25, 7569.10it/s] 51%|█████     | 204409/400000 [00:27<00:25, 7705.98it/s] 51%|█████▏    | 205182/400000 [00:27<00:25, 7554.31it/s] 51%|█████▏    | 205940/400000 [00:27<00:26, 7363.58it/s] 52%|█████▏    | 206702/400000 [00:27<00:25, 7436.44it/s] 52%|█████▏    | 207448/400000 [00:28<00:26, 7326.21it/s] 52%|█████▏    | 208216/400000 [00:28<00:25, 7428.45it/s] 52%|█████▏    | 208970/400000 [00:28<00:25, 7460.82it/s] 52%|█████▏    | 209718/400000 [00:28<00:26, 7268.45it/s] 53%|█████▎    | 210447/400000 [00:28<00:26, 7212.00it/s] 53%|█████▎    | 211180/400000 [00:28<00:26, 7244.28it/s] 53%|█████▎    | 211943/400000 [00:28<00:25, 7355.56it/s] 53%|█████▎    | 212680/400000 [00:28<00:26, 7090.83it/s] 53%|█████▎    | 213409/400000 [00:28<00:26, 7145.66it/s] 54%|█████▎    | 214159/400000 [00:28<00:25, 7246.49it/s] 54%|█████▎    | 214973/400000 [00:29<00:24, 7491.29it/s] 54%|█████▍    | 215726/400000 [00:29<00:24, 7493.33it/s] 54%|█████▍    | 216493/400000 [00:29<00:24, 7545.00it/s] 54%|█████▍    | 217250/400000 [00:29<00:24, 7495.36it/s] 55%|█████▍    | 218049/400000 [00:29<00:23, 7635.58it/s] 55%|█████▍    | 218844/400000 [00:29<00:23, 7722.14it/s] 55%|█████▍    | 219618/400000 [00:29<00:23, 7591.78it/s] 55%|█████▌    | 220379/400000 [00:29<00:23, 7501.65it/s] 55%|█████▌    | 221131/400000 [00:29<00:24, 7245.31it/s] 55%|█████▌    | 221861/400000 [00:29<00:24, 7261.35it/s] 56%|█████▌    | 222590/400000 [00:30<00:24, 7241.03it/s] 56%|█████▌    | 223341/400000 [00:30<00:24, 7316.87it/s] 56%|█████▌    | 224118/400000 [00:30<00:23, 7446.52it/s] 56%|█████▌    | 224875/400000 [00:30<00:23, 7481.14it/s] 56%|█████▋    | 225625/400000 [00:30<00:23, 7443.35it/s] 57%|█████▋    | 226416/400000 [00:30<00:22, 7576.53it/s] 57%|█████▋    | 227248/400000 [00:30<00:22, 7782.83it/s] 57%|█████▋    | 228029/400000 [00:30<00:22, 7690.78it/s] 57%|█████▋    | 228800/400000 [00:30<00:22, 7489.02it/s] 57%|█████▋    | 229552/400000 [00:31<00:23, 7401.78it/s] 58%|█████▊    | 230304/400000 [00:31<00:22, 7434.82it/s] 58%|█████▊    | 231052/400000 [00:31<00:22, 7446.59it/s] 58%|█████▊    | 231798/400000 [00:31<00:22, 7338.87it/s] 58%|█████▊    | 232533/400000 [00:31<00:22, 7293.89it/s] 58%|█████▊    | 233264/400000 [00:31<00:23, 7244.57it/s] 59%|█████▊    | 234057/400000 [00:31<00:22, 7436.26it/s] 59%|█████▊    | 234816/400000 [00:31<00:22, 7480.55it/s] 59%|█████▉    | 235566/400000 [00:31<00:22, 7437.34it/s] 59%|█████▉    | 236311/400000 [00:31<00:22, 7405.50it/s] 59%|█████▉    | 237071/400000 [00:32<00:21, 7461.83it/s] 59%|█████▉    | 237838/400000 [00:32<00:21, 7521.30it/s] 60%|█████▉    | 238594/400000 [00:32<00:21, 7531.83it/s] 60%|█████▉    | 239348/400000 [00:32<00:21, 7400.01it/s] 60%|██████    | 240101/400000 [00:32<00:21, 7437.30it/s] 60%|██████    | 240856/400000 [00:32<00:21, 7469.56it/s] 60%|██████    | 241604/400000 [00:32<00:21, 7465.50it/s] 61%|██████    | 242351/400000 [00:32<00:21, 7405.35it/s] 61%|██████    | 243092/400000 [00:32<00:21, 7347.11it/s] 61%|██████    | 243828/400000 [00:32<00:21, 7181.93it/s] 61%|██████    | 244548/400000 [00:33<00:21, 7134.59it/s] 61%|██████▏   | 245318/400000 [00:33<00:21, 7295.27it/s] 62%|██████▏   | 246104/400000 [00:33<00:20, 7454.84it/s] 62%|██████▏   | 246957/400000 [00:33<00:19, 7745.63it/s] 62%|██████▏   | 247736/400000 [00:33<00:19, 7645.68it/s] 62%|██████▏   | 248532/400000 [00:33<00:19, 7735.70it/s] 62%|██████▏   | 249309/400000 [00:33<00:19, 7733.18it/s] 63%|██████▎   | 250085/400000 [00:33<00:19, 7609.97it/s] 63%|██████▎   | 250865/400000 [00:33<00:19, 7662.32it/s] 63%|██████▎   | 251633/400000 [00:33<00:19, 7578.49it/s] 63%|██████▎   | 252392/400000 [00:34<00:19, 7468.44it/s] 63%|██████▎   | 253140/400000 [00:34<00:19, 7390.79it/s] 63%|██████▎   | 253881/400000 [00:34<00:19, 7394.89it/s] 64%|██████▎   | 254622/400000 [00:34<00:20, 7195.88it/s] 64%|██████▍   | 255344/400000 [00:34<00:20, 7172.55it/s] 64%|██████▍   | 256128/400000 [00:34<00:19, 7359.20it/s] 64%|██████▍   | 256960/400000 [00:34<00:18, 7622.09it/s] 64%|██████▍   | 257727/400000 [00:34<00:18, 7562.54it/s] 65%|██████▍   | 258487/400000 [00:34<00:18, 7512.10it/s] 65%|██████▍   | 259241/400000 [00:34<00:18, 7497.75it/s] 65%|██████▌   | 260020/400000 [00:35<00:18, 7581.01it/s] 65%|██████▌   | 260783/400000 [00:35<00:18, 7594.76it/s] 65%|██████▌   | 261576/400000 [00:35<00:17, 7690.66it/s] 66%|██████▌   | 262346/400000 [00:35<00:17, 7648.73it/s] 66%|██████▌   | 263112/400000 [00:35<00:18, 7578.98it/s] 66%|██████▌   | 263871/400000 [00:35<00:18, 7472.57it/s] 66%|██████▌   | 264620/400000 [00:35<00:18, 7438.36it/s] 66%|██████▋   | 265365/400000 [00:35<00:18, 7422.07it/s] 67%|██████▋   | 266145/400000 [00:35<00:17, 7531.37it/s] 67%|██████▋   | 266899/400000 [00:36<00:17, 7474.29it/s] 67%|██████▋   | 267689/400000 [00:36<00:17, 7595.03it/s] 67%|██████▋   | 268465/400000 [00:36<00:17, 7642.74it/s] 67%|██████▋   | 269249/400000 [00:36<00:16, 7698.78it/s] 68%|██████▊   | 270029/400000 [00:36<00:16, 7728.82it/s] 68%|██████▊   | 270803/400000 [00:36<00:17, 7535.29it/s] 68%|██████▊   | 271626/400000 [00:36<00:16, 7729.31it/s] 68%|██████▊   | 272426/400000 [00:36<00:16, 7808.15it/s] 68%|██████▊   | 273209/400000 [00:36<00:16, 7623.17it/s] 68%|██████▊   | 273974/400000 [00:36<00:16, 7500.78it/s] 69%|██████▊   | 274727/400000 [00:37<00:16, 7428.38it/s] 69%|██████▉   | 275472/400000 [00:37<00:16, 7397.00it/s] 69%|██████▉   | 276213/400000 [00:37<00:16, 7339.83it/s] 69%|██████▉   | 276948/400000 [00:37<00:16, 7324.15it/s] 69%|██████▉   | 277699/400000 [00:37<00:16, 7378.53it/s] 70%|██████▉   | 278438/400000 [00:37<00:16, 7380.91it/s] 70%|██████▉   | 279255/400000 [00:37<00:15, 7599.13it/s] 70%|███████   | 280017/400000 [00:37<00:15, 7508.18it/s] 70%|███████   | 280770/400000 [00:37<00:16, 7445.32it/s] 70%|███████   | 281516/400000 [00:37<00:15, 7442.09it/s] 71%|███████   | 282262/400000 [00:38<00:15, 7446.04it/s] 71%|███████   | 283050/400000 [00:38<00:15, 7570.27it/s] 71%|███████   | 283860/400000 [00:38<00:15, 7720.36it/s] 71%|███████   | 284659/400000 [00:38<00:14, 7797.49it/s] 71%|███████▏  | 285440/400000 [00:38<00:15, 7547.45it/s] 72%|███████▏  | 286198/400000 [00:38<00:15, 7327.07it/s] 72%|███████▏  | 286934/400000 [00:38<00:15, 7328.86it/s] 72%|███████▏  | 287670/400000 [00:38<00:15, 7182.65it/s] 72%|███████▏  | 288436/400000 [00:38<00:15, 7319.24it/s] 72%|███████▏  | 289171/400000 [00:38<00:15, 7204.91it/s] 72%|███████▏  | 289922/400000 [00:39<00:15, 7293.29it/s] 73%|███████▎  | 290729/400000 [00:39<00:14, 7508.22it/s] 73%|███████▎  | 291507/400000 [00:39<00:14, 7587.41it/s] 73%|███████▎  | 292308/400000 [00:39<00:13, 7707.88it/s] 73%|███████▎  | 293081/400000 [00:39<00:13, 7674.30it/s] 73%|███████▎  | 293850/400000 [00:39<00:13, 7589.08it/s] 74%|███████▎  | 294618/400000 [00:39<00:13, 7614.70it/s] 74%|███████▍  | 295381/400000 [00:39<00:13, 7567.73it/s] 74%|███████▍  | 296139/400000 [00:39<00:13, 7505.61it/s] 74%|███████▍  | 296891/400000 [00:39<00:13, 7474.85it/s] 74%|███████▍  | 297639/400000 [00:40<00:14, 7274.33it/s] 75%|███████▍  | 298368/400000 [00:40<00:13, 7276.49it/s] 75%|███████▍  | 299140/400000 [00:40<00:13, 7403.96it/s] 75%|███████▍  | 299882/400000 [00:40<00:13, 7389.78it/s] 75%|███████▌  | 300622/400000 [00:40<00:13, 7346.84it/s] 75%|███████▌  | 301366/400000 [00:40<00:13, 7372.50it/s] 76%|███████▌  | 302120/400000 [00:40<00:13, 7421.06it/s] 76%|███████▌  | 302863/400000 [00:40<00:13, 7403.01it/s] 76%|███████▌  | 303636/400000 [00:40<00:12, 7496.53it/s] 76%|███████▌  | 304387/400000 [00:41<00:12, 7481.33it/s] 76%|███████▋  | 305136/400000 [00:41<00:12, 7432.53it/s] 76%|███████▋  | 305920/400000 [00:41<00:12, 7549.92it/s] 77%|███████▋  | 306676/400000 [00:41<00:12, 7406.91it/s] 77%|███████▋  | 307419/400000 [00:41<00:12, 7411.27it/s] 77%|███████▋  | 308161/400000 [00:41<00:12, 7342.86it/s] 77%|███████▋  | 308896/400000 [00:41<00:12, 7312.35it/s] 77%|███████▋  | 309653/400000 [00:41<00:12, 7386.84it/s] 78%|███████▊  | 310429/400000 [00:41<00:11, 7492.75it/s] 78%|███████▊  | 311207/400000 [00:41<00:11, 7574.88it/s] 78%|███████▊  | 312011/400000 [00:42<00:11, 7707.58it/s] 78%|███████▊  | 312783/400000 [00:42<00:11, 7621.70it/s] 78%|███████▊  | 313574/400000 [00:42<00:11, 7704.47it/s] 79%|███████▊  | 314356/400000 [00:42<00:11, 7736.45it/s] 79%|███████▉  | 315131/400000 [00:42<00:11, 7665.34it/s] 79%|███████▉  | 315899/400000 [00:42<00:11, 7576.49it/s] 79%|███████▉  | 316658/400000 [00:42<00:11, 7471.09it/s] 79%|███████▉  | 317406/400000 [00:42<00:11, 7344.51it/s] 80%|███████▉  | 318142/400000 [00:42<00:11, 7097.64it/s] 80%|███████▉  | 318858/400000 [00:42<00:11, 7114.05it/s] 80%|███████▉  | 319590/400000 [00:43<00:11, 7173.41it/s] 80%|████████  | 320341/400000 [00:43<00:10, 7269.91it/s] 80%|████████  | 321099/400000 [00:43<00:10, 7360.23it/s] 80%|████████  | 321865/400000 [00:43<00:10, 7446.03it/s] 81%|████████  | 322629/400000 [00:43<00:10, 7501.34it/s] 81%|████████  | 323419/400000 [00:43<00:10, 7613.53it/s] 81%|████████  | 324182/400000 [00:43<00:10, 7554.29it/s] 81%|████████  | 324939/400000 [00:43<00:10, 7476.15it/s] 81%|████████▏ | 325688/400000 [00:43<00:10, 7386.92it/s] 82%|████████▏ | 326443/400000 [00:43<00:09, 7432.23it/s] 82%|████████▏ | 327187/400000 [00:44<00:09, 7392.06it/s] 82%|████████▏ | 327927/400000 [00:44<00:09, 7275.16it/s] 82%|████████▏ | 328656/400000 [00:44<00:09, 7271.82it/s] 82%|████████▏ | 329390/400000 [00:44<00:09, 7290.36it/s] 83%|████████▎ | 330120/400000 [00:44<00:09, 7276.49it/s] 83%|████████▎ | 330859/400000 [00:44<00:09, 7308.42it/s] 83%|████████▎ | 331591/400000 [00:44<00:09, 7261.64it/s] 83%|████████▎ | 332330/400000 [00:44<00:09, 7298.90it/s] 83%|████████▎ | 333110/400000 [00:44<00:08, 7441.27it/s] 83%|████████▎ | 333855/400000 [00:44<00:08, 7415.55it/s] 84%|████████▎ | 334639/400000 [00:45<00:08, 7537.65it/s] 84%|████████▍ | 335394/400000 [00:45<00:08, 7449.13it/s] 84%|████████▍ | 336227/400000 [00:45<00:08, 7692.08it/s] 84%|████████▍ | 337051/400000 [00:45<00:08, 7845.54it/s] 84%|████████▍ | 337839/400000 [00:45<00:08, 7727.42it/s] 85%|████████▍ | 338614/400000 [00:45<00:08, 7579.62it/s] 85%|████████▍ | 339375/400000 [00:45<00:08, 7340.19it/s] 85%|████████▌ | 340113/400000 [00:45<00:08, 7331.50it/s] 85%|████████▌ | 340861/400000 [00:45<00:08, 7373.19it/s] 85%|████████▌ | 341600/400000 [00:46<00:07, 7357.54it/s] 86%|████████▌ | 342436/400000 [00:46<00:07, 7629.94it/s] 86%|████████▌ | 343217/400000 [00:46<00:07, 7682.49it/s] 86%|████████▌ | 344022/400000 [00:46<00:07, 7787.89it/s] 86%|████████▌ | 344803/400000 [00:46<00:07, 7714.28it/s] 86%|████████▋ | 345577/400000 [00:46<00:07, 7688.19it/s] 87%|████████▋ | 346403/400000 [00:46<00:06, 7851.20it/s] 87%|████████▋ | 347190/400000 [00:46<00:06, 7656.52it/s] 87%|████████▋ | 347970/400000 [00:46<00:06, 7696.80it/s] 87%|████████▋ | 348767/400000 [00:46<00:06, 7772.13it/s] 87%|████████▋ | 349546/400000 [00:47<00:06, 7563.85it/s] 88%|████████▊ | 350305/400000 [00:47<00:06, 7434.04it/s] 88%|████████▊ | 351051/400000 [00:47<00:06, 7381.40it/s] 88%|████████▊ | 351797/400000 [00:47<00:06, 7403.54it/s] 88%|████████▊ | 352539/400000 [00:47<00:06, 7396.68it/s] 88%|████████▊ | 353309/400000 [00:47<00:06, 7483.71it/s] 89%|████████▊ | 354102/400000 [00:47<00:06, 7611.43it/s] 89%|████████▊ | 354865/400000 [00:47<00:06, 7506.73it/s] 89%|████████▉ | 355617/400000 [00:47<00:05, 7481.80it/s] 89%|████████▉ | 356366/400000 [00:47<00:05, 7465.70it/s] 89%|████████▉ | 357135/400000 [00:48<00:05, 7529.83it/s] 89%|████████▉ | 357936/400000 [00:48<00:05, 7664.93it/s] 90%|████████▉ | 358704/400000 [00:48<00:05, 7491.83it/s] 90%|████████▉ | 359472/400000 [00:48<00:05, 7545.24it/s] 90%|█████████ | 360240/400000 [00:48<00:05, 7583.92it/s] 90%|█████████ | 361000/400000 [00:48<00:05, 7485.69it/s] 90%|█████████ | 361750/400000 [00:48<00:05, 7408.89it/s] 91%|█████████ | 362492/400000 [00:48<00:05, 7304.99it/s] 91%|█████████ | 363224/400000 [00:48<00:05, 7265.73it/s] 91%|█████████ | 363991/400000 [00:48<00:04, 7381.15it/s] 91%|█████████ | 364740/400000 [00:49<00:04, 7408.37it/s] 91%|█████████▏| 365522/400000 [00:49<00:04, 7525.16it/s] 92%|█████████▏| 366277/400000 [00:49<00:04, 7530.53it/s] 92%|█████████▏| 367074/400000 [00:49<00:04, 7654.53it/s] 92%|█████████▏| 367841/400000 [00:49<00:04, 7591.03it/s] 92%|█████████▏| 368622/400000 [00:49<00:04, 7653.07it/s] 92%|█████████▏| 369432/400000 [00:49<00:03, 7780.39it/s] 93%|█████████▎| 370212/400000 [00:49<00:03, 7638.58it/s] 93%|█████████▎| 371024/400000 [00:49<00:03, 7776.25it/s] 93%|█████████▎| 371804/400000 [00:49<00:03, 7520.76it/s] 93%|█████████▎| 372559/400000 [00:50<00:03, 7260.29it/s] 93%|█████████▎| 373299/400000 [00:50<00:03, 7299.05it/s] 94%|█████████▎| 374032/400000 [00:50<00:03, 7252.05it/s] 94%|█████████▎| 374796/400000 [00:50<00:03, 7363.41it/s] 94%|█████████▍| 375584/400000 [00:50<00:03, 7509.47it/s] 94%|█████████▍| 376347/400000 [00:50<00:03, 7543.19it/s] 94%|█████████▍| 377103/400000 [00:50<00:03, 7483.74it/s] 94%|█████████▍| 377853/400000 [00:50<00:03, 7347.01it/s] 95%|█████████▍| 378590/400000 [00:50<00:02, 7334.20it/s] 95%|█████████▍| 379376/400000 [00:51<00:02, 7484.03it/s] 95%|█████████▌| 380126/400000 [00:51<00:02, 7472.89it/s] 95%|█████████▌| 380916/400000 [00:51<00:02, 7595.78it/s] 95%|█████████▌| 381677/400000 [00:51<00:02, 7485.89it/s] 96%|█████████▌| 382427/400000 [00:51<00:02, 7459.08it/s] 96%|█████████▌| 383174/400000 [00:51<00:02, 7429.48it/s] 96%|█████████▌| 383922/400000 [00:51<00:02, 7442.69it/s] 96%|█████████▌| 384667/400000 [00:51<00:02, 7431.27it/s] 96%|█████████▋| 385411/400000 [00:51<00:01, 7348.99it/s] 97%|█████████▋| 386176/400000 [00:51<00:01, 7436.70it/s] 97%|█████████▋| 386921/400000 [00:52<00:01, 7384.66it/s] 97%|█████████▋| 387667/400000 [00:52<00:01, 7406.25it/s] 97%|█████████▋| 388452/400000 [00:52<00:01, 7531.59it/s] 97%|█████████▋| 389255/400000 [00:52<00:01, 7672.95it/s] 98%|█████████▊| 390025/400000 [00:52<00:01, 7679.84it/s] 98%|█████████▊| 390795/400000 [00:52<00:01, 7685.30it/s] 98%|█████████▊| 391613/400000 [00:52<00:01, 7825.37it/s] 98%|█████████▊| 392397/400000 [00:52<00:00, 7743.58it/s] 98%|█████████▊| 393173/400000 [00:52<00:00, 7521.04it/s] 98%|█████████▊| 393928/400000 [00:52<00:00, 7467.69it/s] 99%|█████████▊| 394677/400000 [00:53<00:00, 7356.36it/s] 99%|█████████▉| 395415/400000 [00:53<00:00, 7287.46it/s] 99%|█████████▉| 396145/400000 [00:53<00:00, 7173.02it/s] 99%|█████████▉| 396919/400000 [00:53<00:00, 7331.70it/s] 99%|█████████▉| 397679/400000 [00:53<00:00, 7407.61it/s]100%|█████████▉| 398424/400000 [00:53<00:00, 7420.08it/s]100%|█████████▉| 399244/400000 [00:53<00:00, 7637.59it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7438.91it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fda24896828> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011117601428761843 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011009352462347535 	 Accuracy: 67

  model saves at 67% accuracy 

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

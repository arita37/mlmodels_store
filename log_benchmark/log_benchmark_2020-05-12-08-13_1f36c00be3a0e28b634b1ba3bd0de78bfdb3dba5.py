
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f472b5a0fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 08:13:51.030839
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 08:13:51.034629
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 08:13:51.037812
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 08:13:51.040945
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f47375b8470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355541.8438
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 265867.0312
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 158014.3281
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 84980.1797
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 44795.5078
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 24552.5703
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 14616.0664
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 9377.4541
Epoch 9/10

1/1 [==============================] - 0s 114ms/step - loss: 6507.0259
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 4789.2188

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.06228316e-01 -6.13813639e-01  1.04127121e+00 -2.75890142e-01
   4.31796759e-01 -5.49636006e-01 -4.00051355e-01 -2.16029137e-01
   8.01284015e-02  5.39730251e-01  2.55251706e-01 -3.39382291e-01
  -2.86149859e-01  1.06981921e+00  1.54448688e+00 -9.74731624e-01
   5.32922089e-01 -4.39656600e-02 -4.48180854e-01  2.65868783e-01
  -1.04793668e-01 -9.14466381e-01  1.52373219e+00  1.31467402e+00
   5.55544496e-02 -9.01641250e-01  4.09456134e-01  4.72332358e-01
  -1.39246428e+00 -1.72187686e-02  8.80628347e-01  1.91813040e+00
   4.29813832e-01 -9.66600537e-01  1.39376819e-01 -1.34398532e+00
  -4.57199752e-01 -1.68689847e+00  3.91622901e-01 -6.14553809e-01
   7.24077106e-01  1.83738041e+00 -5.00004530e-01 -4.91762221e-01
  -6.77073002e-02 -2.57250011e-01 -5.85308522e-02  2.23566818e+00
  -1.82918584e+00 -3.05265188e-04  1.20989478e+00  1.08021033e+00
   4.76229012e-01  1.09004128e+00  4.64441180e-01  5.97446859e-01
  -2.98008919e-02 -2.42763937e-01  1.32088673e+00 -6.00503862e-01
   2.34696314e-01  9.51519489e+00  8.38973236e+00  1.05696411e+01
   1.19542418e+01  8.32792091e+00  1.14988174e+01  1.10390272e+01
   9.65321064e+00  1.02350492e+01  9.48010921e+00  8.33014011e+00
   9.32709312e+00  9.97370243e+00  1.03296680e+01  9.75110626e+00
   9.24242210e+00  1.03970003e+01  1.03589869e+01  1.15212173e+01
   1.04501505e+01  1.11855888e+01  1.00522499e+01  1.05761642e+01
   9.44625473e+00  1.03251390e+01  9.45048809e+00  1.01257792e+01
   1.02261095e+01  9.69686127e+00  1.05773926e+01  1.06642542e+01
   1.11244392e+01  9.60976410e+00  8.28337955e+00  1.06009064e+01
   9.21128845e+00  1.01312294e+01  9.09997177e+00  1.03699064e+01
   1.09735155e+01  8.52990246e+00  1.01733904e+01  1.07290421e+01
   1.11560068e+01  9.00898933e+00  1.01938562e+01  9.57447338e+00
   1.15175266e+01  9.57240677e+00  9.55240250e+00  9.82288837e+00
   9.85949802e+00  1.07596674e+01  9.51444244e+00  1.06542492e+01
   1.09025402e+01  9.82208824e+00  1.06518707e+01  9.51306534e+00
   1.98651576e+00 -1.77906811e-01  2.67140150e-01  1.07942879e+00
  -9.56738591e-02 -2.14427662e+00 -1.13753664e+00 -1.14462972e+00
   4.62753177e-01 -5.78723669e-01  3.52598429e-02  1.20296216e+00
  -6.13345504e-01 -1.58450270e+00  2.27710199e+00 -8.88562202e-02
  -3.21003348e-01  7.35783935e-01 -5.47045529e-01 -2.34575844e+00
  -1.42511129e-01 -8.81537497e-02 -1.06740665e+00  6.54722452e-01
  -1.29132175e+00  2.04606012e-01  1.67186886e-01  7.35708177e-02
   1.21012174e-01 -1.77028847e+00 -3.79531443e-01  1.63684392e+00
   2.73948520e-01  3.93395126e-01 -8.91479373e-01  3.24924886e-02
   6.26765132e-01  1.27200276e-01  6.92947865e-01  2.58819342e-01
  -7.26649523e-01  8.96455288e-01 -1.24740720e+00  8.34029436e-01
   7.53078997e-01 -2.22943425e+00  1.19606614e+00 -6.06462359e-02
   1.11403275e+00 -5.17149508e-01 -1.86308742e+00 -5.48338115e-01
   2.23920494e-02  6.66440070e-01 -7.75566638e-01 -5.37117243e-01
  -9.82497215e-01 -7.59318352e-01 -1.97930789e+00 -6.08693242e-01
   2.28668153e-01  1.89690924e+00  3.55168104e-01  1.08189583e+00
   1.13267112e+00  1.22688961e+00  2.12771034e+00  2.34424424e+00
   1.19465733e+00  6.97277188e-02  1.73974848e+00  1.70412874e+00
   1.48594272e+00  1.93650842e-01  1.53196895e+00  3.97272885e-01
   1.20173681e+00  1.17538595e+00  1.54742289e+00  4.07635748e-01
   1.08918583e+00  5.18091917e-01  2.58047581e-01  1.86230731e+00
   4.58797097e-01  2.02416468e+00  1.56559062e+00  6.08683467e-01
   7.19430447e-01  2.32422042e+00  8.06651950e-01  9.70396161e-01
   3.48978877e-01  1.53776956e+00  1.75813341e+00  1.57448435e+00
   1.07148993e+00  1.95186543e+00  1.77826643e-01  1.12354946e+00
   3.94020259e-01  2.51698613e-01  5.40837169e-01  8.19061756e-01
   8.27257037e-01  6.58410490e-01  1.31412089e+00  2.45814443e-01
   1.93673885e+00  1.94160700e-01  5.66435575e-01  2.54711151e-01
   2.88881242e-01  1.31260753e-01  6.59407139e-01  4.50373173e-01
   3.72233033e-01  1.55129755e+00  4.30753291e-01  7.99484491e-01
   2.32456446e-01  9.79388618e+00  9.45538616e+00  9.13893414e+00
   9.53018665e+00  9.29228592e+00  1.11772919e+01  9.82202530e+00
   9.38983822e+00  1.04656487e+01  1.02868080e+01  1.12971811e+01
   9.38319016e+00  9.49204826e+00  9.95312119e+00  8.20835209e+00
   1.16665134e+01  9.14319515e+00  9.68004036e+00  9.79567051e+00
   9.98129654e+00  8.38757706e+00  9.93082714e+00  1.02602262e+01
   9.90423489e+00  1.02782564e+01  1.02842207e+01  9.75938320e+00
   9.76627636e+00  1.06883726e+01  1.03448925e+01  9.20856285e+00
   1.17664785e+01  1.10091448e+01  1.10061903e+01  1.06591492e+01
   1.08812485e+01  1.08142881e+01  1.16785440e+01  1.00955343e+01
   9.30110645e+00  1.11655064e+01  1.06055269e+01  1.09679813e+01
   1.07731638e+01  1.08934536e+01  1.03302879e+01  1.06768646e+01
   1.00042963e+01  8.79927444e+00  1.02447481e+01  9.00690079e+00
   8.92207241e+00  1.05273056e+01  1.09983912e+01  1.05089121e+01
   1.00337315e+01  9.93885040e+00  1.05901022e+01  9.31447887e+00
   4.00927365e-01  3.40903401e-01  8.81948829e-01  1.53702939e+00
   3.75504076e-01  2.59597421e-01  1.28156185e+00  1.70662510e+00
   8.39376569e-01  1.10810661e+00  6.34274483e-01  4.82606173e-01
   9.48925197e-01  9.12761331e-01  4.45993781e-01  1.01814747e+00
   1.31372833e+00  1.62143171e-01  1.22785950e+00  4.94891405e-01
   4.24965262e-01  1.97044289e+00  1.21369362e+00  5.65062582e-01
   3.54308367e-01  1.72067022e+00  9.90056634e-01  1.33524728e+00
   1.13371372e+00  1.69267106e+00  1.40134466e+00  6.29897058e-01
   6.89926028e-01  1.32673907e+00  8.11112523e-01  2.95495689e-01
   4.88526285e-01  3.45535696e-01  8.42784405e-01  7.70131648e-01
   1.60208404e-01  1.15584743e+00  4.00405109e-01  7.95836806e-01
   5.56286097e-01  4.68233109e-01  1.40945816e+00  2.20505381e+00
   5.66718042e-01  7.84211934e-01  1.56283796e+00  1.29811549e+00
   1.03221786e+00  1.28097415e+00  1.30134463e+00  5.24895728e-01
   8.49673867e-01  9.72099304e-01  1.20741701e+00  1.64621449e+00
  -1.20932951e+01  8.69071770e+00 -1.30562248e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 08:14:01.728728
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.3171
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 08:14:01.732505
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8549.27
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 08:14:01.735671
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.3494
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 08:14:01.738758
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -764.648
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139943290114736
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139942080205040
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139942080205544
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139942080206048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139942080206552
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139942080207056

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f47171cdfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.668666
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.635906
grad_step = 000002, loss = 0.614087
grad_step = 000003, loss = 0.591867
grad_step = 000004, loss = 0.567146
grad_step = 000005, loss = 0.542166
grad_step = 000006, loss = 0.523033
grad_step = 000007, loss = 0.508973
grad_step = 000008, loss = 0.493072
grad_step = 000009, loss = 0.471698
grad_step = 000010, loss = 0.451374
grad_step = 000011, loss = 0.434298
grad_step = 000012, loss = 0.419240
grad_step = 000013, loss = 0.404213
grad_step = 000014, loss = 0.388909
grad_step = 000015, loss = 0.373686
grad_step = 000016, loss = 0.360038
grad_step = 000017, loss = 0.347708
grad_step = 000018, loss = 0.335494
grad_step = 000019, loss = 0.323071
grad_step = 000020, loss = 0.310687
grad_step = 000021, loss = 0.298708
grad_step = 000022, loss = 0.287156
grad_step = 000023, loss = 0.275520
grad_step = 000024, loss = 0.263889
grad_step = 000025, loss = 0.252647
grad_step = 000026, loss = 0.241823
grad_step = 000027, loss = 0.231408
grad_step = 000028, loss = 0.220999
grad_step = 000029, loss = 0.210657
grad_step = 000030, loss = 0.200852
grad_step = 000031, loss = 0.191251
grad_step = 000032, loss = 0.181718
grad_step = 000033, loss = 0.172634
grad_step = 000034, loss = 0.163964
grad_step = 000035, loss = 0.155524
grad_step = 000036, loss = 0.147276
grad_step = 000037, loss = 0.139387
grad_step = 000038, loss = 0.131831
grad_step = 000039, loss = 0.124365
grad_step = 000040, loss = 0.117136
grad_step = 000041, loss = 0.110379
grad_step = 000042, loss = 0.104018
grad_step = 000043, loss = 0.097840
grad_step = 000044, loss = 0.091943
grad_step = 000045, loss = 0.086330
grad_step = 000046, loss = 0.080952
grad_step = 000047, loss = 0.075870
grad_step = 000048, loss = 0.071040
grad_step = 000049, loss = 0.066423
grad_step = 000050, loss = 0.062077
grad_step = 000051, loss = 0.058045
grad_step = 000052, loss = 0.054165
grad_step = 000053, loss = 0.050531
grad_step = 000054, loss = 0.047144
grad_step = 000055, loss = 0.043935
grad_step = 000056, loss = 0.040897
grad_step = 000057, loss = 0.038084
grad_step = 000058, loss = 0.035427
grad_step = 000059, loss = 0.032966
grad_step = 000060, loss = 0.030657
grad_step = 000061, loss = 0.028499
grad_step = 000062, loss = 0.026510
grad_step = 000063, loss = 0.024645
grad_step = 000064, loss = 0.022892
grad_step = 000065, loss = 0.021278
grad_step = 000066, loss = 0.019767
grad_step = 000067, loss = 0.018364
grad_step = 000068, loss = 0.017079
grad_step = 000069, loss = 0.015880
grad_step = 000070, loss = 0.014770
grad_step = 000071, loss = 0.013733
grad_step = 000072, loss = 0.012772
grad_step = 000073, loss = 0.011888
grad_step = 000074, loss = 0.011061
grad_step = 000075, loss = 0.010305
grad_step = 000076, loss = 0.009603
grad_step = 000077, loss = 0.008950
grad_step = 000078, loss = 0.008348
grad_step = 000079, loss = 0.007788
grad_step = 000080, loss = 0.007272
grad_step = 000081, loss = 0.006793
grad_step = 000082, loss = 0.006353
grad_step = 000083, loss = 0.005944
grad_step = 000084, loss = 0.005564
grad_step = 000085, loss = 0.005216
grad_step = 000086, loss = 0.004893
grad_step = 000087, loss = 0.004596
grad_step = 000088, loss = 0.004322
grad_step = 000089, loss = 0.004071
grad_step = 000090, loss = 0.003842
grad_step = 000091, loss = 0.003636
grad_step = 000092, loss = 0.003455
grad_step = 000093, loss = 0.003298
grad_step = 000094, loss = 0.003163
grad_step = 000095, loss = 0.003019
grad_step = 000096, loss = 0.002851
grad_step = 000097, loss = 0.002682
grad_step = 000098, loss = 0.002559
grad_step = 000099, loss = 0.002473
grad_step = 000100, loss = 0.002399
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002321
grad_step = 000102, loss = 0.002238
grad_step = 000103, loss = 0.002161
grad_step = 000104, loss = 0.002082
grad_step = 000105, loss = 0.002012
grad_step = 000106, loss = 0.001958
grad_step = 000107, loss = 0.001925
grad_step = 000108, loss = 0.001903
grad_step = 000109, loss = 0.001875
grad_step = 000110, loss = 0.001838
grad_step = 000111, loss = 0.001789
grad_step = 000112, loss = 0.001743
grad_step = 000113, loss = 0.001710
grad_step = 000114, loss = 0.001691
grad_step = 000115, loss = 0.001683
grad_step = 000116, loss = 0.001680
grad_step = 000117, loss = 0.001679
grad_step = 000118, loss = 0.001674
grad_step = 000119, loss = 0.001665
grad_step = 000120, loss = 0.001647
grad_step = 000121, loss = 0.001624
grad_step = 000122, loss = 0.001597
grad_step = 000123, loss = 0.001573
grad_step = 000124, loss = 0.001557
grad_step = 000125, loss = 0.001547
grad_step = 000126, loss = 0.001544
grad_step = 000127, loss = 0.001547
grad_step = 000128, loss = 0.001561
grad_step = 000129, loss = 0.001589
grad_step = 000130, loss = 0.001641
grad_step = 000131, loss = 0.001677
grad_step = 000132, loss = 0.001712
grad_step = 000133, loss = 0.001813
grad_step = 000134, loss = 0.001957
grad_step = 000135, loss = 0.002020
grad_step = 000136, loss = 0.001704
grad_step = 000137, loss = 0.001515
grad_step = 000138, loss = 0.001710
grad_step = 000139, loss = 0.001788
grad_step = 000140, loss = 0.001569
grad_step = 000141, loss = 0.001496
grad_step = 000142, loss = 0.001661
grad_step = 000143, loss = 0.001627
grad_step = 000144, loss = 0.001461
grad_step = 000145, loss = 0.001517
grad_step = 000146, loss = 0.001598
grad_step = 000147, loss = 0.001481
grad_step = 000148, loss = 0.001440
grad_step = 000149, loss = 0.001519
grad_step = 000150, loss = 0.001487
grad_step = 000151, loss = 0.001415
grad_step = 000152, loss = 0.001445
grad_step = 000153, loss = 0.001464
grad_step = 000154, loss = 0.001412
grad_step = 000155, loss = 0.001396
grad_step = 000156, loss = 0.001423
grad_step = 000157, loss = 0.001410
grad_step = 000158, loss = 0.001375
grad_step = 000159, loss = 0.001379
grad_step = 000160, loss = 0.001390
grad_step = 000161, loss = 0.001372
grad_step = 000162, loss = 0.001352
grad_step = 000163, loss = 0.001354
grad_step = 000164, loss = 0.001356
grad_step = 000165, loss = 0.001344
grad_step = 000166, loss = 0.001331
grad_step = 000167, loss = 0.001326
grad_step = 000168, loss = 0.001325
grad_step = 000169, loss = 0.001320
grad_step = 000170, loss = 0.001313
grad_step = 000171, loss = 0.001304
grad_step = 000172, loss = 0.001296
grad_step = 000173, loss = 0.001291
grad_step = 000174, loss = 0.001291
grad_step = 000175, loss = 0.001289
grad_step = 000176, loss = 0.001280
grad_step = 000177, loss = 0.001269
grad_step = 000178, loss = 0.001259
grad_step = 000179, loss = 0.001255
grad_step = 000180, loss = 0.001253
grad_step = 000181, loss = 0.001252
grad_step = 000182, loss = 0.001251
grad_step = 000183, loss = 0.001251
grad_step = 000184, loss = 0.001259
grad_step = 000185, loss = 0.001280
grad_step = 000186, loss = 0.001331
grad_step = 000187, loss = 0.001433
grad_step = 000188, loss = 0.001618
grad_step = 000189, loss = 0.001810
grad_step = 000190, loss = 0.001851
grad_step = 000191, loss = 0.001557
grad_step = 000192, loss = 0.001249
grad_step = 000193, loss = 0.001277
grad_step = 000194, loss = 0.001502
grad_step = 000195, loss = 0.001541
grad_step = 000196, loss = 0.001309
grad_step = 000197, loss = 0.001207
grad_step = 000198, loss = 0.001346
grad_step = 000199, loss = 0.001433
grad_step = 000200, loss = 0.001308
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001190
grad_step = 000202, loss = 0.001266
grad_step = 000203, loss = 0.001353
grad_step = 000204, loss = 0.001284
grad_step = 000205, loss = 0.001182
grad_step = 000206, loss = 0.001218
grad_step = 000207, loss = 0.001287
grad_step = 000208, loss = 0.001264
grad_step = 000209, loss = 0.001181
grad_step = 000210, loss = 0.001180
grad_step = 000211, loss = 0.001233
grad_step = 000212, loss = 0.001241
grad_step = 000213, loss = 0.001187
grad_step = 000214, loss = 0.001157
grad_step = 000215, loss = 0.001183
grad_step = 000216, loss = 0.001211
grad_step = 000217, loss = 0.001191
grad_step = 000218, loss = 0.001154
grad_step = 000219, loss = 0.001147
grad_step = 000220, loss = 0.001168
grad_step = 000221, loss = 0.001182
grad_step = 000222, loss = 0.001166
grad_step = 000223, loss = 0.001143
grad_step = 000224, loss = 0.001133
grad_step = 000225, loss = 0.001142
grad_step = 000226, loss = 0.001153
grad_step = 000227, loss = 0.001153
grad_step = 000228, loss = 0.001141
grad_step = 000229, loss = 0.001128
grad_step = 000230, loss = 0.001122
grad_step = 000231, loss = 0.001124
grad_step = 000232, loss = 0.001129
grad_step = 000233, loss = 0.001133
grad_step = 000234, loss = 0.001134
grad_step = 000235, loss = 0.001131
grad_step = 000236, loss = 0.001125
grad_step = 000237, loss = 0.001119
grad_step = 000238, loss = 0.001113
grad_step = 000239, loss = 0.001108
grad_step = 000240, loss = 0.001105
grad_step = 000241, loss = 0.001103
grad_step = 000242, loss = 0.001102
grad_step = 000243, loss = 0.001101
grad_step = 000244, loss = 0.001102
grad_step = 000245, loss = 0.001103
grad_step = 000246, loss = 0.001108
grad_step = 000247, loss = 0.001118
grad_step = 000248, loss = 0.001139
grad_step = 000249, loss = 0.001185
grad_step = 000250, loss = 0.001279
grad_step = 000251, loss = 0.001441
grad_step = 000252, loss = 0.001682
grad_step = 000253, loss = 0.001826
grad_step = 000254, loss = 0.001691
grad_step = 000255, loss = 0.001273
grad_step = 000256, loss = 0.001096
grad_step = 000257, loss = 0.001306
grad_step = 000258, loss = 0.001459
grad_step = 000259, loss = 0.001275
grad_step = 000260, loss = 0.001095
grad_step = 000261, loss = 0.001210
grad_step = 000262, loss = 0.001335
grad_step = 000263, loss = 0.001213
grad_step = 000264, loss = 0.001091
grad_step = 000265, loss = 0.001174
grad_step = 000266, loss = 0.001242
grad_step = 000267, loss = 0.001155
grad_step = 000268, loss = 0.001081
grad_step = 000269, loss = 0.001151
grad_step = 000270, loss = 0.001193
grad_step = 000271, loss = 0.001121
grad_step = 000272, loss = 0.001071
grad_step = 000273, loss = 0.001125
grad_step = 000274, loss = 0.001155
grad_step = 000275, loss = 0.001109
grad_step = 000276, loss = 0.001063
grad_step = 000277, loss = 0.001092
grad_step = 000278, loss = 0.001123
grad_step = 000279, loss = 0.001100
grad_step = 000280, loss = 0.001060
grad_step = 000281, loss = 0.001070
grad_step = 000282, loss = 0.001093
grad_step = 000283, loss = 0.001089
grad_step = 000284, loss = 0.001059
grad_step = 000285, loss = 0.001055
grad_step = 000286, loss = 0.001071
grad_step = 000287, loss = 0.001077
grad_step = 000288, loss = 0.001060
grad_step = 000289, loss = 0.001047
grad_step = 000290, loss = 0.001051
grad_step = 000291, loss = 0.001061
grad_step = 000292, loss = 0.001057
grad_step = 000293, loss = 0.001046
grad_step = 000294, loss = 0.001040
grad_step = 000295, loss = 0.001044
grad_step = 000296, loss = 0.001049
grad_step = 000297, loss = 0.001046
grad_step = 000298, loss = 0.001039
grad_step = 000299, loss = 0.001034
grad_step = 000300, loss = 0.001035
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001038
grad_step = 000302, loss = 0.001038
grad_step = 000303, loss = 0.001034
grad_step = 000304, loss = 0.001030
grad_step = 000305, loss = 0.001028
grad_step = 000306, loss = 0.001028
grad_step = 000307, loss = 0.001029
grad_step = 000308, loss = 0.001029
grad_step = 000309, loss = 0.001026
grad_step = 000310, loss = 0.001023
grad_step = 000311, loss = 0.001021
grad_step = 000312, loss = 0.001021
grad_step = 000313, loss = 0.001021
grad_step = 000314, loss = 0.001021
grad_step = 000315, loss = 0.001020
grad_step = 000316, loss = 0.001018
grad_step = 000317, loss = 0.001016
grad_step = 000318, loss = 0.001015
grad_step = 000319, loss = 0.001014
grad_step = 000320, loss = 0.001014
grad_step = 000321, loss = 0.001013
grad_step = 000322, loss = 0.001013
grad_step = 000323, loss = 0.001013
grad_step = 000324, loss = 0.001014
grad_step = 000325, loss = 0.001017
grad_step = 000326, loss = 0.001023
grad_step = 000327, loss = 0.001037
grad_step = 000328, loss = 0.001069
grad_step = 000329, loss = 0.001124
grad_step = 000330, loss = 0.001237
grad_step = 000331, loss = 0.001305
grad_step = 000332, loss = 0.001315
grad_step = 000333, loss = 0.001131
grad_step = 000334, loss = 0.001008
grad_step = 000335, loss = 0.001060
grad_step = 000336, loss = 0.001155
grad_step = 000337, loss = 0.001141
grad_step = 000338, loss = 0.001026
grad_step = 000339, loss = 0.001014
grad_step = 000340, loss = 0.001090
grad_step = 000341, loss = 0.001094
grad_step = 000342, loss = 0.001027
grad_step = 000343, loss = 0.000998
grad_step = 000344, loss = 0.001041
grad_step = 000345, loss = 0.001069
grad_step = 000346, loss = 0.001024
grad_step = 000347, loss = 0.000992
grad_step = 000348, loss = 0.001014
grad_step = 000349, loss = 0.001037
grad_step = 000350, loss = 0.001021
grad_step = 000351, loss = 0.000991
grad_step = 000352, loss = 0.000996
grad_step = 000353, loss = 0.001016
grad_step = 000354, loss = 0.001012
grad_step = 000355, loss = 0.000992
grad_step = 000356, loss = 0.000987
grad_step = 000357, loss = 0.001000
grad_step = 000358, loss = 0.001006
grad_step = 000359, loss = 0.000996
grad_step = 000360, loss = 0.000990
grad_step = 000361, loss = 0.001002
grad_step = 000362, loss = 0.001025
grad_step = 000363, loss = 0.001047
grad_step = 000364, loss = 0.001087
grad_step = 000365, loss = 0.001166
grad_step = 000366, loss = 0.001296
grad_step = 000367, loss = 0.001444
grad_step = 000368, loss = 0.001539
grad_step = 000369, loss = 0.001452
grad_step = 000370, loss = 0.001214
grad_step = 000371, loss = 0.001012
grad_step = 000372, loss = 0.001020
grad_step = 000373, loss = 0.001161
grad_step = 000374, loss = 0.001221
grad_step = 000375, loss = 0.001118
grad_step = 000376, loss = 0.000997
grad_step = 000377, loss = 0.001008
grad_step = 000378, loss = 0.001096
grad_step = 000379, loss = 0.001111
grad_step = 000380, loss = 0.001034
grad_step = 000381, loss = 0.000975
grad_step = 000382, loss = 0.001002
grad_step = 000383, loss = 0.001053
grad_step = 000384, loss = 0.001043
grad_step = 000385, loss = 0.000991
grad_step = 000386, loss = 0.000966
grad_step = 000387, loss = 0.000991
grad_step = 000388, loss = 0.001019
grad_step = 000389, loss = 0.001008
grad_step = 000390, loss = 0.000975
grad_step = 000391, loss = 0.000959
grad_step = 000392, loss = 0.000972
grad_step = 000393, loss = 0.000991
grad_step = 000394, loss = 0.000991
grad_step = 000395, loss = 0.000973
grad_step = 000396, loss = 0.000955
grad_step = 000397, loss = 0.000955
grad_step = 000398, loss = 0.000966
grad_step = 000399, loss = 0.000974
grad_step = 000400, loss = 0.000969
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000957
grad_step = 000402, loss = 0.000949
grad_step = 000403, loss = 0.000949
grad_step = 000404, loss = 0.000954
grad_step = 000405, loss = 0.000958
grad_step = 000406, loss = 0.000956
grad_step = 000407, loss = 0.000950
grad_step = 000408, loss = 0.000945
grad_step = 000409, loss = 0.000943
grad_step = 000410, loss = 0.000944
grad_step = 000411, loss = 0.000945
grad_step = 000412, loss = 0.000945
grad_step = 000413, loss = 0.000944
grad_step = 000414, loss = 0.000943
grad_step = 000415, loss = 0.000940
grad_step = 000416, loss = 0.000938
grad_step = 000417, loss = 0.000936
grad_step = 000418, loss = 0.000935
grad_step = 000419, loss = 0.000934
grad_step = 000420, loss = 0.000934
grad_step = 000421, loss = 0.000934
grad_step = 000422, loss = 0.000934
grad_step = 000423, loss = 0.000934
grad_step = 000424, loss = 0.000933
grad_step = 000425, loss = 0.000931
grad_step = 000426, loss = 0.000929
grad_step = 000427, loss = 0.000928
grad_step = 000428, loss = 0.000926
grad_step = 000429, loss = 0.000925
grad_step = 000430, loss = 0.000924
grad_step = 000431, loss = 0.000923
grad_step = 000432, loss = 0.000922
grad_step = 000433, loss = 0.000921
grad_step = 000434, loss = 0.000920
grad_step = 000435, loss = 0.000919
grad_step = 000436, loss = 0.000918
grad_step = 000437, loss = 0.000917
grad_step = 000438, loss = 0.000917
grad_step = 000439, loss = 0.000916
grad_step = 000440, loss = 0.000916
grad_step = 000441, loss = 0.000915
grad_step = 000442, loss = 0.000915
grad_step = 000443, loss = 0.000916
grad_step = 000444, loss = 0.000917
grad_step = 000445, loss = 0.000921
grad_step = 000446, loss = 0.000928
grad_step = 000447, loss = 0.000942
grad_step = 000448, loss = 0.000967
grad_step = 000449, loss = 0.001012
grad_step = 000450, loss = 0.001092
grad_step = 000451, loss = 0.001212
grad_step = 000452, loss = 0.001373
grad_step = 000453, loss = 0.001490
grad_step = 000454, loss = 0.001481
grad_step = 000455, loss = 0.001264
grad_step = 000456, loss = 0.000997
grad_step = 000457, loss = 0.000911
grad_step = 000458, loss = 0.001032
grad_step = 000459, loss = 0.001161
grad_step = 000460, loss = 0.001120
grad_step = 000461, loss = 0.000969
grad_step = 000462, loss = 0.000906
grad_step = 000463, loss = 0.000981
grad_step = 000464, loss = 0.001056
grad_step = 000465, loss = 0.001013
grad_step = 000466, loss = 0.000922
grad_step = 000467, loss = 0.000902
grad_step = 000468, loss = 0.000958
grad_step = 000469, loss = 0.000997
grad_step = 000470, loss = 0.000969
grad_step = 000471, loss = 0.000914
grad_step = 000472, loss = 0.000893
grad_step = 000473, loss = 0.000917
grad_step = 000474, loss = 0.000949
grad_step = 000475, loss = 0.000949
grad_step = 000476, loss = 0.000921
grad_step = 000477, loss = 0.000893
grad_step = 000478, loss = 0.000890
grad_step = 000479, loss = 0.000906
grad_step = 000480, loss = 0.000922
grad_step = 000481, loss = 0.000921
grad_step = 000482, loss = 0.000904
grad_step = 000483, loss = 0.000887
grad_step = 000484, loss = 0.000882
grad_step = 000485, loss = 0.000889
grad_step = 000486, loss = 0.000898
grad_step = 000487, loss = 0.000901
grad_step = 000488, loss = 0.000895
grad_step = 000489, loss = 0.000885
grad_step = 000490, loss = 0.000878
grad_step = 000491, loss = 0.000877
grad_step = 000492, loss = 0.000881
grad_step = 000493, loss = 0.000885
grad_step = 000494, loss = 0.000885
grad_step = 000495, loss = 0.000882
grad_step = 000496, loss = 0.000877
grad_step = 000497, loss = 0.000873
grad_step = 000498, loss = 0.000871
grad_step = 000499, loss = 0.000871
grad_step = 000500, loss = 0.000873
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000874
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

  date_run                              2020-05-12 08:14:20.835842
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.29668
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 08:14:20.841284
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.217772
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 08:14:20.849194
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.159256
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 08:14:20.854379
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.30912
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
0   2020-05-12 08:13:51.030839  ...    mean_absolute_error
1   2020-05-12 08:13:51.034629  ...     mean_squared_error
2   2020-05-12 08:13:51.037812  ...  median_absolute_error
3   2020-05-12 08:13:51.040945  ...               r2_score
4   2020-05-12 08:14:01.728728  ...    mean_absolute_error
5   2020-05-12 08:14:01.732505  ...     mean_squared_error
6   2020-05-12 08:14:01.735671  ...  median_absolute_error
7   2020-05-12 08:14:01.738758  ...               r2_score
8   2020-05-12 08:14:20.835842  ...    mean_absolute_error
9   2020-05-12 08:14:20.841284  ...     mean_squared_error
10  2020-05-12 08:14:20.849194  ...  median_absolute_error
11  2020-05-12 08:14:20.854379  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:22, 445875.21it/s] 95%|█████████▍| 9396224/9912422 [00:00<00:00, 635641.92it/s]9920512it [00:00, 45716241.73it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1062067.38it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1017051.01it/s]1654784it [00:00, 12548257.61it/s]                           
0it [00:00, ?it/s]8192it [00:00, 252063.17it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558cae8b38> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f552a23de48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558caace80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5529d15080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558cae8b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f553f4a4e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558caace80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f552a23a0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558cae8b38> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f553f4a4e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f558caace80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3aa7f55208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=1fdf7244475a9e9b96a76173083e80140b600b94e62fb16d492efa77a8aaa96b
  Stored in directory: /tmp/pip-ephem-wheel-cache-j4drm_8b/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3a9e0c3080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 3s
 1900544/17464789 [==>...........................] - ETA: 0s
 6897664/17464789 [==========>...................] - ETA: 0s
16203776/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 08:15:47.665897: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 08:15:47.670934: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 08:15:47.671155: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5586f80ed900 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 08:15:47.671172: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6130 - accuracy: 0.5035 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5708 - accuracy: 0.5063
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6316 - accuracy: 0.5023
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6570 - accuracy: 0.5006
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
11000/25000 [============>.................] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6289 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6491 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6738 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6922 - accuracy: 0.4983
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6781 - accuracy: 0.4992
25000/25000 [==============================] - 7s 278us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 08:16:01.236917
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 08:16:01.236917  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 08:16:07.390330: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 08:16:07.395955: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 08:16:07.396149: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5615a5c3b650 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 08:16:07.396179: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9abdff5be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3612 - crf_viterbi_accuracy: 0.0400 - val_loss: 1.3029 - val_crf_viterbi_accuracy: 0.6000

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9a99af4e80> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0806 - accuracy: 0.4730
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8813 - accuracy: 0.4860 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7740 - accuracy: 0.4930
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 3s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6984 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.6779 - accuracy: 0.4993
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6752 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6480 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9a54cc4a90> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:30:32, 5.04kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:29:24, 7.15kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:29:40, 10.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:26:51, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:28:48, 20.8kB/s].vector_cache/glove.6B.zip:   1%|          | 7.37M/862M [00:02<8:00:11, 29.7kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:02<5:34:51, 42.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:02<3:53:15, 60.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.7M/862M [00:02<2:42:33, 86.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:02<1:53:19, 123kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 28.8M/862M [00:02<1:18:57, 176kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:02<55:07, 251kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<38:26, 357kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:03<26:54, 508kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.4M/862M [00:03<18:50, 722kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:03<13:13, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:04<10:12, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<09:01, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:06<10:44, 1.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:06<09:07, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:06<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<05:01, 2.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<09:40, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:08<08:10, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:08<06:14, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<04:30, 2.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<52:14, 254kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:10<39:18, 338kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:10<28:05, 472kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:10<19:58, 663kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<17:12, 768kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:12<13:25, 983kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:12<09:42, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:59, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<20:57, 627kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:14<16:10, 813kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:14<11:38, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<08:23, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<13:22, 978kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:16<10:47, 1.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:16<07:54, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<08:24, 1.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<07:19, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:18<05:29, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<06:39, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:19<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:20<04:35, 2.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:22<04:29, 2.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<03:18, 3.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<33:46, 379kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<25:06, 510kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:24<17:51, 715kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<12:43, 1.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:25<16:14, 784kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<12:53, 987kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:26<09:23, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:42, 1.89MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<2:34:05, 82.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<1:49:20, 116kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<1:16:47, 165kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<53:44, 234kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<2:00:07, 105kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<1:25:29, 147kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<1:00:02, 209kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<42:15, 297kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<35:01, 358kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<26:02, 481kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<18:32, 674kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<13:12, 944kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<14:44, 845kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<11:48, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<08:35, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:12, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<13:20, 928kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<13:52, 892kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<11:10, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<09:10, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<06:45, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:24, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:35, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:58, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<03:37, 3.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<1:22:19, 148kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<59:00, 206kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<41:31, 293kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<29:17, 414kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<25:53, 468kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<19:55, 608kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<14:19, 844kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<10:12, 1.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<12:30, 962kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<10:00, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:14, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:21, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<10:06, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<08:36, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:23, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:51, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:16, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:44, 2.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<03:28, 3.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<20:17, 583kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<15:49, 747kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<11:26, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<08:09, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<13:48, 851kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<11:16, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<08:17, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:55, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<34:02, 344kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<25:22, 461kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<18:07, 644kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<12:46, 910kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<11:33:11, 16.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<8:06:34, 23.9kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<5:40:22, 34.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<3:57:41, 48.6kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<2:58:38, 64.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<2:06:29, 91.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:28:45, 130kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<1:02:03, 185kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<1:25:44, 134kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<1:01:27, 187kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<43:19, 265kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<30:22, 376kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<49:11, 232kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<35:51, 318kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<25:22, 449kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<17:52, 635kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<22:45, 498kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<17:22, 653kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<12:30, 906kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:07<10:58, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<09:05, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:42, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<06:13, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<04:41, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<05:33, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<05:14, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<03:56, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<05:05, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<06:02, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<04:45, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:31, 3.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<05:47, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<05:26, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:08, 2.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<04:50, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<03:40, 2.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<04:48, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<04:35, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<03:27, 3.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<04:48, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<04:32, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<03:26, 3.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<04:45, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<05:32, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<04:25, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:12, 3.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<13:31, 788kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<10:34, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:39, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<07:48, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<07:39, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:48, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:12, 2.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<07:59, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<06:42, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:57, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<05:52, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<05:11, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<03:50, 2.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<05:09, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:44, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<04:32, 2.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:17, 3.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<1:16:37, 135kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<54:39, 189kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<38:26, 267kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<29:12, 351kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:37<22:32, 454kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<16:11, 631kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<11:29, 888kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<11:25, 891kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<09:04, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:58, 1.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<05:58, 1.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:25, 2.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:27, 1.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:38, 2.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:53, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:27, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<03:21, 2.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<04:41, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:17, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<03:15, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<04:35, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:13, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:12, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<04:32, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:12, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:08, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<02:58, 3.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<1:54:52, 84.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<1:21:22, 119kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<57:02, 169kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<42:01, 229kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<30:23, 317kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<21:25, 448kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<17:13, 555kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<13:56, 685kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:11, 937kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:13, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<11:11, 848kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<08:49, 1.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:21, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<06:39, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:29, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:11, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:01, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<39:40, 236kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<28:43, 325kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<20:17, 459kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<16:20, 568kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<12:23, 748kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<08:50, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<06:17, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<1:53:49, 81.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<1:20:34, 114kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<56:28, 163kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<41:33, 220kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<29:59, 305kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<21:09, 431kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<16:54, 537kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<12:46, 710kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<09:09, 988kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<08:30, 1.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<07:49, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:56, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:15, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<07:42, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<06:19, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:39, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<05:15, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<05:27, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:12, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:05, 2.85MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<05:10, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<04:32, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:24, 2.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<04:25, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<04:54, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<03:52, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:48, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<8:14:42, 17.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<5:46:47, 25.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<4:02:34, 35.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<2:49:03, 50.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<2:32:21, 56.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<1:47:27, 79.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<1:15:10, 114kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<54:24, 157kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<38:56, 219kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<27:23, 310kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<21:05, 401kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<15:37, 541kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<11:07, 758kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<09:44, 861kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<07:39, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:33, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:50, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<04:56, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:39, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:30, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<04:00, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:00, 2.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:01, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:39, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<02:45, 2.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<03:50, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:31, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<02:40, 3.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<03:45, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:27, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<02:37, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:42, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:25, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<02:35, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<03:40, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<03:15, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<02:27, 3.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<01:49, 4.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<32:47, 239kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<23:35, 332kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<16:39, 469kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<11:42, 664kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<37:35, 207kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<28:01, 277kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<19:58, 388kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<15:08, 509kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<11:15, 684kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:02, 955kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<05:41, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<35:43, 214kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<25:37, 298kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<18:03, 421kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<12:41, 596kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<1:11:04, 106kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<50:29, 150kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<35:25, 213kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<26:25, 284kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<19:17, 389kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<13:38, 547kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<11:15, 660kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<08:38, 858kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<06:13, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<06:04, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:00, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:40, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:39, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<02:52, 2.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:04, 3.49MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<22:09, 326kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<17:02, 424kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:13, 590kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<08:38, 831kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<08:36, 832kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<06:45, 1.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:52, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<05:03, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<04:15, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:10, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:17, 3.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<47:48, 147kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<34:09, 205kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<23:57, 292kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<16:47, 414kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<1:03:23, 110kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<45:03, 154kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<31:34, 219kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<23:37, 291kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<17:14, 399kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<12:10, 563kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<08:35, 794kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<22:44, 300kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<16:41, 408kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<11:50, 573kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<09:40, 697kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<07:27, 903kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:22, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<05:20, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:16, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<03:48, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<03:25, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:33, 2.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<01:52, 3.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<08:56, 732kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<06:26, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:36, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<05:13, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<04:19, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:09, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:18, 2.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<06:25, 994kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<06:07, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:37, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:21, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:56, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:25, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:33, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<03:14, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<02:55, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:12, 2.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<02:59, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<02:44, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:03, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<02:52, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:35<02:38, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:00, 3.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:36, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<01:55, 3.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<01:26, 4.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<45:14, 132kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<32:15, 185kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<22:37, 263kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<17:08, 345kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<12:35, 469kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<08:56, 658kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<07:34, 771kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<05:56, 982kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<04:17, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:15, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:13, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<03:13, 1.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:24, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:45, 3.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<27:28, 208kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<19:57, 286kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<14:06, 402kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<10:55, 516kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<08:18, 677kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<05:57, 941kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<05:19, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:20, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:24, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:36, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:03, 2.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:00, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:48, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:05, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<01:33, 3.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:42, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:58, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:05, 2.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:32, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:50, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:13, 2.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<01:39, 3.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:34, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:21, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:46, 2.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:23, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:43, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:07, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<01:35, 3.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:31, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:47, 2.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:17, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:37, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:02, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:32, 3.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:19, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:07, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:35, 3.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:03, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:32, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:01, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:30, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<01:59, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:30, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:58, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:28, 3.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:56, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:27, 3.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:04, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:54, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:26, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:02, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<01:52, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:25, 3.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:00, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<01:51, 2.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:24, 3.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:22, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:56, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:47, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:21, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:37, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:36, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:05, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:32, 2.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<01:59, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<01:56, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:33, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:09, 3.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:49, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:29, 2.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:06, 3.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:07, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:53, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:24, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:01, 3.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<3:39:10, 17.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<2:33:34, 25.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<1:47:03, 35.7kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<1:14:11, 50.9kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<1:05:52, 57.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<46:26, 81.2kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<32:24, 116kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<22:30, 165kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<30:23, 122kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<21:37, 171kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<15:08, 243kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<10:31, 346kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<31:17, 116kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<22:14, 163kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<15:33, 232kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<11:37, 307kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<08:29, 420kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<06:00, 590kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<04:11, 834kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<26:00, 135kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<18:32, 189kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<12:58, 268kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<09:01, 380kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<19:01, 181kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<13:38, 251kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<09:32, 356kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<06:39, 505kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<29:36, 114kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<21:02, 160kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<14:41, 227kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<10:57, 301kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<08:02, 409kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:40, 577kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<03:58, 814kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<07:36, 424kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:41, 566kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<04:02, 790kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:28, 909kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:47, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:02, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:03, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:16, 2.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:36, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:28, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:05, 2.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:23, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:33, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:14, 2.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<00:54, 3.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:48, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:33, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:09, 2.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:27, 1.93MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:19, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:00, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:17, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:12, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<00:54, 3.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:12, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:26, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:08, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:50, 3.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:24, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:16, 2.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:57, 2.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:12, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:25, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:06, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:50, 2.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:03, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:48, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:05, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:16, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:00, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:44, 3.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:15, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:07, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:50, 2.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:06, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:00, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:45, 2.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:02, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:56, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:42, 3.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:59, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:54, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:40, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:56, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:50, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:38, 3.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:54, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:50, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:37, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:48, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:36, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:51, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:35, 3.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:45, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:34, 3.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:32, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:52, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:41, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:43, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:39, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:29, 3.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:38, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:28, 3.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:39, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:36, 2.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:27, 3.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:38, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:34, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:26, 3.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:36, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:33, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:24, 3.06MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:31, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:23, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:32, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:29, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:21, 3.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:20, 3.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:26, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:19, 3.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:18, 3.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:22, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:22, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:20, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:23, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:12, 3.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:28, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:11, 3.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:40, 366kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:13, 496kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:50, 695kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:40, 804kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:31, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:20, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:10, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:06, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:02, 3.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:02, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.14MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 844/400000 [00:00<00:47, 8430.48it/s]  0%|          | 1712/400000 [00:00<00:46, 8502.20it/s]  1%|          | 2581/400000 [00:00<00:46, 8557.15it/s]  1%|          | 3460/400000 [00:00<00:45, 8624.37it/s]  1%|          | 4337/400000 [00:00<00:45, 8666.08it/s]  1%|▏         | 5171/400000 [00:00<00:46, 8565.38it/s]  2%|▏         | 6030/400000 [00:00<00:45, 8570.85it/s]  2%|▏         | 6913/400000 [00:00<00:45, 8645.34it/s]  2%|▏         | 7734/400000 [00:00<00:46, 8507.64it/s]  2%|▏         | 8613/400000 [00:01<00:45, 8589.47it/s]  2%|▏         | 9454/400000 [00:01<00:45, 8532.80it/s]  3%|▎         | 10309/400000 [00:01<00:45, 8537.64it/s]  3%|▎         | 11163/400000 [00:01<00:45, 8533.89it/s]  3%|▎         | 12022/400000 [00:01<00:45, 8550.12it/s]  3%|▎         | 12901/400000 [00:01<00:44, 8617.97it/s]  3%|▎         | 13770/400000 [00:01<00:44, 8638.08it/s]  4%|▎         | 14632/400000 [00:01<00:45, 8539.73it/s]  4%|▍         | 15511/400000 [00:01<00:44, 8612.08it/s]  4%|▍         | 16372/400000 [00:01<00:45, 8481.82it/s]  4%|▍         | 17249/400000 [00:02<00:44, 8563.64it/s]  5%|▍         | 18120/400000 [00:02<00:44, 8605.40it/s]  5%|▍         | 18981/400000 [00:02<00:44, 8522.93it/s]  5%|▍         | 19852/400000 [00:02<00:44, 8575.50it/s]  5%|▌         | 20733/400000 [00:02<00:43, 8641.68it/s]  5%|▌         | 21608/400000 [00:02<00:43, 8671.90it/s]  6%|▌         | 22476/400000 [00:02<00:44, 8482.97it/s]  6%|▌         | 23326/400000 [00:02<00:44, 8465.74it/s]  6%|▌         | 24193/400000 [00:02<00:44, 8525.54it/s]  6%|▋         | 25071/400000 [00:02<00:43, 8598.89it/s]  6%|▋         | 25955/400000 [00:03<00:43, 8668.84it/s]  7%|▋         | 26823/400000 [00:03<00:43, 8636.94it/s]  7%|▋         | 27693/400000 [00:03<00:43, 8652.97it/s]  7%|▋         | 28559/400000 [00:03<00:43, 8587.38it/s]  7%|▋         | 29422/400000 [00:03<00:43, 8599.25it/s]  8%|▊         | 30290/400000 [00:03<00:42, 8621.76it/s]  8%|▊         | 31167/400000 [00:03<00:42, 8663.66it/s]  8%|▊         | 32034/400000 [00:03<00:42, 8655.56it/s]  8%|▊         | 32901/400000 [00:03<00:42, 8658.82it/s]  8%|▊         | 33784/400000 [00:03<00:42, 8707.39it/s]  9%|▊         | 34665/400000 [00:04<00:41, 8737.87it/s]  9%|▉         | 35539/400000 [00:04<00:41, 8730.44it/s]  9%|▉         | 36413/400000 [00:04<00:41, 8701.28it/s]  9%|▉         | 37287/400000 [00:04<00:41, 8712.55it/s] 10%|▉         | 38166/400000 [00:04<00:41, 8733.83it/s] 10%|▉         | 39040/400000 [00:04<00:41, 8643.52it/s] 10%|▉         | 39912/400000 [00:04<00:41, 8664.41it/s] 10%|█         | 40782/400000 [00:04<00:41, 8672.58it/s] 10%|█         | 41663/400000 [00:04<00:41, 8712.71it/s] 11%|█         | 42535/400000 [00:04<00:42, 8497.27it/s] 11%|█         | 43409/400000 [00:05<00:41, 8568.16it/s] 11%|█         | 44281/400000 [00:05<00:41, 8612.23it/s] 11%|█▏        | 45147/400000 [00:05<00:41, 8625.65it/s] 12%|█▏        | 46011/400000 [00:05<00:41, 8624.36it/s] 12%|█▏        | 46887/400000 [00:05<00:40, 8662.81it/s] 12%|█▏        | 47754/400000 [00:05<00:41, 8528.40it/s] 12%|█▏        | 48632/400000 [00:05<00:40, 8599.77it/s] 12%|█▏        | 49499/400000 [00:05<00:40, 8620.02it/s] 13%|█▎        | 50377/400000 [00:05<00:40, 8664.93it/s] 13%|█▎        | 51257/400000 [00:05<00:40, 8703.11it/s] 13%|█▎        | 52128/400000 [00:06<00:40, 8671.83it/s] 13%|█▎        | 53002/400000 [00:06<00:39, 8690.44it/s] 13%|█▎        | 53872/400000 [00:06<00:40, 8625.39it/s] 14%|█▎        | 54735/400000 [00:06<00:40, 8572.65it/s] 14%|█▍        | 55593/400000 [00:06<00:40, 8515.90it/s] 14%|█▍        | 56445/400000 [00:06<00:40, 8499.87it/s] 14%|█▍        | 57300/400000 [00:06<00:40, 8512.83it/s] 15%|█▍        | 58152/400000 [00:06<00:40, 8501.94it/s] 15%|█▍        | 59003/400000 [00:06<00:40, 8503.46it/s] 15%|█▍        | 59885/400000 [00:06<00:39, 8594.68it/s] 15%|█▌        | 60755/400000 [00:07<00:39, 8625.92it/s] 15%|█▌        | 61618/400000 [00:07<00:39, 8529.34it/s] 16%|█▌        | 62472/400000 [00:07<00:40, 8343.72it/s] 16%|█▌        | 63332/400000 [00:07<00:39, 8417.10it/s] 16%|█▌        | 64201/400000 [00:07<00:39, 8496.22it/s] 16%|█▋        | 65071/400000 [00:07<00:39, 8556.33it/s] 16%|█▋        | 65928/400000 [00:07<00:39, 8405.05it/s] 17%|█▋        | 66770/400000 [00:07<00:40, 8214.90it/s] 17%|█▋        | 67650/400000 [00:07<00:39, 8379.76it/s] 17%|█▋        | 68523/400000 [00:07<00:39, 8481.72it/s] 17%|█▋        | 69395/400000 [00:08<00:38, 8550.54it/s] 18%|█▊        | 70274/400000 [00:08<00:38, 8620.78it/s] 18%|█▊        | 71138/400000 [00:08<00:38, 8484.91it/s] 18%|█▊        | 72020/400000 [00:08<00:38, 8582.06it/s] 18%|█▊        | 72896/400000 [00:08<00:37, 8633.55it/s] 18%|█▊        | 73770/400000 [00:08<00:37, 8662.47it/s] 19%|█▊        | 74644/400000 [00:08<00:37, 8683.84it/s] 19%|█▉        | 75513/400000 [00:08<00:38, 8397.97it/s] 19%|█▉        | 76361/400000 [00:08<00:38, 8421.71it/s] 19%|█▉        | 77235/400000 [00:08<00:37, 8514.51it/s] 20%|█▉        | 78107/400000 [00:09<00:37, 8573.93it/s] 20%|█▉        | 78984/400000 [00:09<00:37, 8629.22it/s] 20%|█▉        | 79848/400000 [00:09<00:37, 8597.50it/s] 20%|██        | 80716/400000 [00:09<00:37, 8620.20it/s] 20%|██        | 81593/400000 [00:09<00:36, 8662.60it/s] 21%|██        | 82474/400000 [00:09<00:36, 8704.62it/s] 21%|██        | 83352/400000 [00:09<00:36, 8726.43it/s] 21%|██        | 84225/400000 [00:09<00:36, 8687.03it/s] 21%|██▏       | 85108/400000 [00:09<00:36, 8726.66it/s] 21%|██▏       | 85989/400000 [00:10<00:35, 8749.59it/s] 22%|██▏       | 86865/400000 [00:10<00:35, 8750.89it/s] 22%|██▏       | 87743/400000 [00:10<00:35, 8757.10it/s] 22%|██▏       | 88619/400000 [00:10<00:35, 8703.67it/s] 22%|██▏       | 89498/400000 [00:10<00:35, 8726.96it/s] 23%|██▎       | 90384/400000 [00:10<00:35, 8765.29it/s] 23%|██▎       | 91263/400000 [00:10<00:35, 8771.31it/s] 23%|██▎       | 92148/400000 [00:10<00:35, 8793.45it/s] 23%|██▎       | 93028/400000 [00:10<00:35, 8743.41it/s] 23%|██▎       | 93914/400000 [00:10<00:34, 8775.23it/s] 24%|██▎       | 94792/400000 [00:11<00:34, 8757.12it/s] 24%|██▍       | 95668/400000 [00:11<00:34, 8753.68it/s] 24%|██▍       | 96553/400000 [00:11<00:34, 8780.59it/s] 24%|██▍       | 97433/400000 [00:11<00:34, 8784.88it/s] 25%|██▍       | 98312/400000 [00:11<00:34, 8722.87it/s] 25%|██▍       | 99185/400000 [00:11<00:34, 8679.46it/s] 25%|██▌       | 100054/400000 [00:11<00:34, 8665.13it/s] 25%|██▌       | 100921/400000 [00:11<00:34, 8637.47it/s] 25%|██▌       | 101785/400000 [00:11<00:34, 8617.31it/s] 26%|██▌       | 102662/400000 [00:11<00:34, 8661.56it/s] 26%|██▌       | 103539/400000 [00:12<00:34, 8692.52it/s] 26%|██▌       | 104415/400000 [00:12<00:33, 8711.16it/s] 26%|██▋       | 105295/400000 [00:12<00:33, 8736.00it/s] 27%|██▋       | 106169/400000 [00:12<00:33, 8725.10it/s] 27%|██▋       | 107042/400000 [00:12<00:33, 8665.86it/s] 27%|██▋       | 107927/400000 [00:12<00:33, 8719.95it/s] 27%|██▋       | 108801/400000 [00:12<00:33, 8724.40it/s] 27%|██▋       | 109683/400000 [00:12<00:33, 8750.43it/s] 28%|██▊       | 110560/400000 [00:12<00:33, 8753.35it/s] 28%|██▊       | 111443/400000 [00:12<00:32, 8774.42it/s] 28%|██▊       | 112321/400000 [00:13<00:32, 8762.60it/s] 28%|██▊       | 113198/400000 [00:13<00:32, 8723.67it/s] 29%|██▊       | 114082/400000 [00:13<00:32, 8757.64it/s] 29%|██▊       | 114958/400000 [00:13<00:32, 8714.47it/s] 29%|██▉       | 115844/400000 [00:13<00:32, 8756.33it/s] 29%|██▉       | 116729/400000 [00:13<00:32, 8784.01it/s] 29%|██▉       | 117608/400000 [00:13<00:32, 8783.24it/s] 30%|██▉       | 118487/400000 [00:13<00:32, 8747.49it/s] 30%|██▉       | 119362/400000 [00:13<00:33, 8458.18it/s] 30%|███       | 120210/400000 [00:13<00:33, 8419.01it/s] 30%|███       | 121088/400000 [00:14<00:32, 8523.71it/s] 30%|███       | 121974/400000 [00:14<00:32, 8621.63it/s] 31%|███       | 122838/400000 [00:14<00:32, 8569.01it/s] 31%|███       | 123710/400000 [00:14<00:32, 8613.28it/s] 31%|███       | 124573/400000 [00:14<00:31, 8608.67it/s] 31%|███▏      | 125457/400000 [00:14<00:31, 8675.63it/s] 32%|███▏      | 126339/400000 [00:14<00:31, 8715.83it/s] 32%|███▏      | 127217/400000 [00:14<00:31, 8732.08it/s] 32%|███▏      | 128091/400000 [00:14<00:31, 8683.62it/s] 32%|███▏      | 128960/400000 [00:14<00:31, 8682.78it/s] 32%|███▏      | 129840/400000 [00:15<00:30, 8716.92it/s] 33%|███▎      | 130712/400000 [00:15<00:31, 8674.06it/s] 33%|███▎      | 131587/400000 [00:15<00:30, 8695.55it/s] 33%|███▎      | 132457/400000 [00:15<00:30, 8690.65it/s] 33%|███▎      | 133329/400000 [00:15<00:30, 8697.12it/s] 34%|███▎      | 134208/400000 [00:15<00:30, 8724.51it/s] 34%|███▍      | 135084/400000 [00:15<00:30, 8733.54it/s] 34%|███▍      | 135958/400000 [00:15<00:30, 8728.52it/s] 34%|███▍      | 136831/400000 [00:15<00:30, 8655.21it/s] 34%|███▍      | 137711/400000 [00:15<00:30, 8695.87it/s] 35%|███▍      | 138586/400000 [00:16<00:30, 8709.40it/s] 35%|███▍      | 139473/400000 [00:16<00:29, 8754.50it/s] 35%|███▌      | 140349/400000 [00:16<00:29, 8705.43it/s] 35%|███▌      | 141220/400000 [00:16<00:29, 8699.54it/s] 36%|███▌      | 142091/400000 [00:16<00:29, 8670.73it/s] 36%|███▌      | 142977/400000 [00:16<00:29, 8725.82it/s] 36%|███▌      | 143860/400000 [00:16<00:29, 8756.40it/s] 36%|███▌      | 144744/400000 [00:16<00:29, 8779.69it/s] 36%|███▋      | 145623/400000 [00:16<00:29, 8752.60it/s] 37%|███▋      | 146499/400000 [00:16<00:28, 8748.78it/s] 37%|███▋      | 147383/400000 [00:17<00:28, 8775.17it/s] 37%|███▋      | 148261/400000 [00:17<00:28, 8760.20it/s] 37%|███▋      | 149149/400000 [00:17<00:28, 8795.73it/s] 38%|███▊      | 150029/400000 [00:17<00:28, 8683.02it/s] 38%|███▊      | 150906/400000 [00:17<00:28, 8706.14it/s] 38%|███▊      | 151793/400000 [00:17<00:28, 8753.45it/s] 38%|███▊      | 152678/400000 [00:17<00:28, 8781.23it/s] 38%|███▊      | 153562/400000 [00:17<00:28, 8796.19it/s] 39%|███▊      | 154442/400000 [00:17<00:28, 8709.64it/s] 39%|███▉      | 155317/400000 [00:17<00:28, 8718.90it/s] 39%|███▉      | 156190/400000 [00:18<00:27, 8720.99it/s] 39%|███▉      | 157078/400000 [00:18<00:27, 8765.65it/s] 39%|███▉      | 157965/400000 [00:18<00:27, 8795.64it/s] 40%|███▉      | 158845/400000 [00:18<00:27, 8795.40it/s] 40%|███▉      | 159725/400000 [00:18<00:27, 8753.67it/s] 40%|████      | 160601/400000 [00:18<00:27, 8726.71it/s] 40%|████      | 161487/400000 [00:18<00:27, 8765.27it/s] 41%|████      | 162365/400000 [00:18<00:27, 8769.38it/s] 41%|████      | 163243/400000 [00:18<00:27, 8715.80it/s] 41%|████      | 164121/400000 [00:18<00:27, 8733.24it/s] 41%|████▏     | 165006/400000 [00:19<00:26, 8765.21it/s] 41%|████▏     | 165891/400000 [00:19<00:26, 8789.44it/s] 42%|████▏     | 166780/400000 [00:19<00:26, 8817.89it/s] 42%|████▏     | 167664/400000 [00:19<00:26, 8823.15it/s] 42%|████▏     | 168547/400000 [00:19<00:26, 8800.35it/s] 42%|████▏     | 169428/400000 [00:19<00:26, 8786.96it/s] 43%|████▎     | 170315/400000 [00:19<00:26, 8810.78it/s] 43%|████▎     | 171197/400000 [00:19<00:25, 8807.20it/s] 43%|████▎     | 172078/400000 [00:19<00:25, 8802.77it/s] 43%|████▎     | 172959/400000 [00:19<00:25, 8799.16it/s] 43%|████▎     | 173842/400000 [00:20<00:25, 8805.50it/s] 44%|████▎     | 174723/400000 [00:20<00:25, 8801.39it/s] 44%|████▍     | 175613/400000 [00:20<00:25, 8829.81it/s] 44%|████▍     | 176498/400000 [00:20<00:25, 8835.73it/s] 44%|████▍     | 177382/400000 [00:20<00:25, 8813.02it/s] 45%|████▍     | 178264/400000 [00:20<00:25, 8803.34it/s] 45%|████▍     | 179147/400000 [00:20<00:25, 8809.97it/s] 45%|████▌     | 180030/400000 [00:20<00:24, 8814.64it/s] 45%|████▌     | 180916/400000 [00:20<00:24, 8826.56it/s] 45%|████▌     | 181799/400000 [00:20<00:24, 8803.77it/s] 46%|████▌     | 182681/400000 [00:21<00:24, 8805.84it/s] 46%|████▌     | 183568/400000 [00:21<00:24, 8822.72it/s] 46%|████▌     | 184451/400000 [00:21<00:24, 8790.61it/s] 46%|████▋     | 185331/400000 [00:21<00:24, 8733.92it/s] 47%|████▋     | 186205/400000 [00:21<00:24, 8712.18it/s] 47%|████▋     | 187079/400000 [00:21<00:24, 8719.79it/s] 47%|████▋     | 187958/400000 [00:21<00:24, 8740.66it/s] 47%|████▋     | 188841/400000 [00:21<00:24, 8765.52it/s] 47%|████▋     | 189718/400000 [00:21<00:24, 8759.67it/s] 48%|████▊     | 190595/400000 [00:21<00:23, 8744.47it/s] 48%|████▊     | 191472/400000 [00:22<00:23, 8749.62it/s] 48%|████▊     | 192355/400000 [00:22<00:23, 8770.74it/s] 48%|████▊     | 193233/400000 [00:22<00:23, 8767.81it/s] 49%|████▊     | 194110/400000 [00:22<00:23, 8703.34it/s] 49%|████▊     | 194981/400000 [00:22<00:23, 8682.64it/s] 49%|████▉     | 195850/400000 [00:22<00:23, 8630.87it/s] 49%|████▉     | 196714/400000 [00:22<00:23, 8574.35it/s] 49%|████▉     | 197595/400000 [00:22<00:23, 8642.09it/s] 50%|████▉     | 198473/400000 [00:22<00:23, 8682.80it/s] 50%|████▉     | 199349/400000 [00:22<00:23, 8704.13it/s] 50%|█████     | 200237/400000 [00:23<00:22, 8754.42it/s] 50%|█████     | 201121/400000 [00:23<00:22, 8779.65it/s] 51%|█████     | 202003/400000 [00:23<00:22, 8791.32it/s] 51%|█████     | 202888/400000 [00:23<00:22, 8808.30it/s] 51%|█████     | 203769/400000 [00:23<00:22, 8798.88it/s] 51%|█████     | 204649/400000 [00:23<00:22, 8784.14it/s] 51%|█████▏    | 205536/400000 [00:23<00:22, 8807.00it/s] 52%|█████▏    | 206417/400000 [00:23<00:22, 8666.36it/s] 52%|█████▏    | 207298/400000 [00:23<00:22, 8707.59it/s] 52%|█████▏    | 208170/400000 [00:23<00:22, 8657.21it/s] 52%|█████▏    | 209049/400000 [00:24<00:21, 8695.09it/s] 52%|█████▏    | 209919/400000 [00:24<00:22, 8639.32it/s] 53%|█████▎    | 210789/400000 [00:24<00:21, 8655.85it/s] 53%|█████▎    | 211675/400000 [00:24<00:21, 8713.19it/s] 53%|█████▎    | 212547/400000 [00:24<00:21, 8598.30it/s] 53%|█████▎    | 213423/400000 [00:24<00:21, 8644.90it/s] 54%|█████▎    | 214306/400000 [00:24<00:21, 8696.74it/s] 54%|█████▍    | 215177/400000 [00:24<00:21, 8558.74it/s] 54%|█████▍    | 216061/400000 [00:24<00:21, 8639.35it/s] 54%|█████▍    | 216937/400000 [00:25<00:21, 8674.08it/s] 54%|█████▍    | 217805/400000 [00:25<00:21, 8642.98it/s] 55%|█████▍    | 218676/400000 [00:25<00:20, 8662.10it/s] 55%|█████▍    | 219543/400000 [00:25<00:20, 8652.55it/s] 55%|█████▌    | 220428/400000 [00:25<00:20, 8708.03it/s] 55%|█████▌    | 221305/400000 [00:25<00:20, 8725.09it/s] 56%|█████▌    | 222184/400000 [00:25<00:20, 8743.75it/s] 56%|█████▌    | 223062/400000 [00:25<00:20, 8752.35it/s] 56%|█████▌    | 223938/400000 [00:25<00:20, 8733.34it/s] 56%|█████▌    | 224815/400000 [00:25<00:20, 8742.08it/s] 56%|█████▋    | 225690/400000 [00:26<00:20, 8685.63it/s] 57%|█████▋    | 226559/400000 [00:26<00:20, 8514.84it/s] 57%|█████▋    | 227446/400000 [00:26<00:20, 8617.77it/s] 57%|█████▋    | 228331/400000 [00:26<00:19, 8685.91it/s] 57%|█████▋    | 229201/400000 [00:26<00:19, 8658.95it/s] 58%|█████▊    | 230078/400000 [00:26<00:19, 8690.07it/s] 58%|█████▊    | 230958/400000 [00:26<00:19, 8720.78it/s] 58%|█████▊    | 231843/400000 [00:26<00:19, 8758.73it/s] 58%|█████▊    | 232727/400000 [00:26<00:19, 8781.66it/s] 58%|█████▊    | 233606/400000 [00:26<00:18, 8770.46it/s] 59%|█████▊    | 234484/400000 [00:27<00:18, 8771.67it/s] 59%|█████▉    | 235362/400000 [00:27<00:18, 8705.77it/s] 59%|█████▉    | 236244/400000 [00:27<00:18, 8737.31it/s] 59%|█████▉    | 237130/400000 [00:27<00:18, 8773.05it/s] 60%|█████▉    | 238008/400000 [00:27<00:18, 8708.61it/s] 60%|█████▉    | 238888/400000 [00:27<00:18, 8734.83it/s] 60%|█████▉    | 239765/400000 [00:27<00:18, 8745.18it/s] 60%|██████    | 240650/400000 [00:27<00:18, 8773.70it/s] 60%|██████    | 241534/400000 [00:27<00:18, 8790.65it/s] 61%|██████    | 242414/400000 [00:27<00:18, 8754.47it/s] 61%|██████    | 243290/400000 [00:28<00:17, 8742.03it/s] 61%|██████    | 244165/400000 [00:28<00:17, 8719.04it/s] 61%|██████▏   | 245049/400000 [00:28<00:17, 8753.32it/s] 61%|██████▏   | 245930/400000 [00:28<00:17, 8769.14it/s] 62%|██████▏   | 246807/400000 [00:28<00:17, 8593.24it/s] 62%|██████▏   | 247669/400000 [00:28<00:17, 8598.70it/s] 62%|██████▏   | 248540/400000 [00:28<00:17, 8628.82it/s] 62%|██████▏   | 249422/400000 [00:28<00:17, 8683.36it/s] 63%|██████▎   | 250305/400000 [00:28<00:17, 8725.33it/s] 63%|██████▎   | 251178/400000 [00:28<00:17, 8617.04it/s] 63%|██████▎   | 252045/400000 [00:29<00:17, 8630.49it/s] 63%|██████▎   | 252926/400000 [00:29<00:16, 8683.36it/s] 63%|██████▎   | 253809/400000 [00:29<00:16, 8724.63it/s] 64%|██████▎   | 254694/400000 [00:29<00:16, 8761.86it/s] 64%|██████▍   | 255579/400000 [00:29<00:16, 8785.61it/s] 64%|██████▍   | 256458/400000 [00:29<00:16, 8774.04it/s] 64%|██████▍   | 257336/400000 [00:29<00:16, 8692.47it/s] 65%|██████▍   | 258216/400000 [00:29<00:16, 8722.34it/s] 65%|██████▍   | 259097/400000 [00:29<00:16, 8747.72it/s] 65%|██████▍   | 259972/400000 [00:29<00:16, 8572.40it/s] 65%|██████▌   | 260846/400000 [00:30<00:16, 8621.14it/s] 65%|██████▌   | 261722/400000 [00:30<00:15, 8661.54it/s] 66%|██████▌   | 262605/400000 [00:30<00:15, 8710.76it/s] 66%|██████▌   | 263477/400000 [00:30<00:15, 8709.62it/s] 66%|██████▌   | 264349/400000 [00:30<00:15, 8677.63it/s] 66%|██████▋   | 265218/400000 [00:30<00:15, 8638.88it/s] 67%|██████▋   | 266094/400000 [00:30<00:15, 8673.92it/s] 67%|██████▋   | 266962/400000 [00:30<00:15, 8673.81it/s] 67%|██████▋   | 267845/400000 [00:30<00:15, 8717.71it/s] 67%|██████▋   | 268730/400000 [00:30<00:14, 8756.46it/s] 67%|██████▋   | 269610/400000 [00:31<00:14, 8768.11it/s] 68%|██████▊   | 270487/400000 [00:31<00:15, 8548.21it/s] 68%|██████▊   | 271370/400000 [00:31<00:14, 8628.51it/s] 68%|██████▊   | 272256/400000 [00:31<00:14, 8694.33it/s] 68%|██████▊   | 273144/400000 [00:31<00:14, 8748.15it/s] 69%|██████▊   | 274024/400000 [00:31<00:14, 8760.15it/s] 69%|██████▊   | 274901/400000 [00:31<00:14, 8753.42it/s] 69%|██████▉   | 275777/400000 [00:31<00:14, 8654.52it/s] 69%|██████▉   | 276647/400000 [00:31<00:14, 8666.75it/s] 69%|██████▉   | 277515/400000 [00:31<00:14, 8666.61it/s] 70%|██████▉   | 278394/400000 [00:32<00:13, 8702.36it/s] 70%|██████▉   | 279265/400000 [00:32<00:13, 8670.99it/s] 70%|███████   | 280143/400000 [00:32<00:13, 8701.99it/s] 70%|███████   | 281028/400000 [00:32<00:13, 8743.68it/s] 70%|███████   | 281913/400000 [00:32<00:13, 8773.81it/s] 71%|███████   | 282791/400000 [00:32<00:13, 8734.55it/s] 71%|███████   | 283665/400000 [00:32<00:13, 8683.43it/s] 71%|███████   | 284542/400000 [00:32<00:13, 8707.78it/s] 71%|███████▏  | 285425/400000 [00:32<00:13, 8742.77it/s] 72%|███████▏  | 286300/400000 [00:32<00:13, 8508.05it/s] 72%|███████▏  | 287184/400000 [00:33<00:13, 8604.38it/s] 72%|███████▏  | 288046/400000 [00:33<00:13, 8570.32it/s] 72%|███████▏  | 288928/400000 [00:33<00:12, 8643.34it/s] 72%|███████▏  | 289805/400000 [00:33<00:12, 8680.47it/s] 73%|███████▎  | 290688/400000 [00:33<00:12, 8723.02it/s] 73%|███████▎  | 291574/400000 [00:33<00:12, 8762.65it/s] 73%|███████▎  | 292451/400000 [00:33<00:12, 8664.50it/s] 73%|███████▎  | 293337/400000 [00:33<00:12, 8719.65it/s] 74%|███████▎  | 294210/400000 [00:33<00:12, 8637.66it/s] 74%|███████▍  | 295082/400000 [00:33<00:12, 8660.68it/s] 74%|███████▍  | 295967/400000 [00:34<00:11, 8714.02it/s] 74%|███████▍  | 296839/400000 [00:34<00:11, 8683.63it/s] 74%|███████▍  | 297725/400000 [00:34<00:11, 8733.73it/s] 75%|███████▍  | 298613/400000 [00:34<00:11, 8776.17it/s] 75%|███████▍  | 299491/400000 [00:34<00:11, 8765.62it/s] 75%|███████▌  | 300379/400000 [00:34<00:11, 8796.77it/s] 75%|███████▌  | 301259/400000 [00:34<00:11, 8743.06it/s] 76%|███████▌  | 302146/400000 [00:34<00:11, 8779.04it/s] 76%|███████▌  | 303032/400000 [00:34<00:11, 8801.92it/s] 76%|███████▌  | 303916/400000 [00:34<00:10, 8810.52it/s] 76%|███████▌  | 304804/400000 [00:35<00:10, 8829.97it/s] 76%|███████▋  | 305688/400000 [00:35<00:10, 8774.59it/s] 77%|███████▋  | 306572/400000 [00:35<00:10, 8792.56it/s] 77%|███████▋  | 307452/400000 [00:35<00:10, 8730.34it/s] 77%|███████▋  | 308332/400000 [00:35<00:10, 8750.43it/s] 77%|███████▋  | 309216/400000 [00:35<00:10, 8775.71it/s] 78%|███████▊  | 310094/400000 [00:35<00:10, 8760.20it/s] 78%|███████▊  | 310981/400000 [00:35<00:10, 8790.26it/s] 78%|███████▊  | 311861/400000 [00:35<00:10, 8737.74it/s] 78%|███████▊  | 312738/400000 [00:36<00:09, 8744.61it/s] 78%|███████▊  | 313628/400000 [00:36<00:09, 8787.45it/s] 79%|███████▊  | 314509/400000 [00:36<00:09, 8792.98it/s] 79%|███████▉  | 315391/400000 [00:36<00:09, 8800.06it/s] 79%|███████▉  | 316272/400000 [00:36<00:09, 8786.17it/s] 79%|███████▉  | 317151/400000 [00:36<00:09, 8772.33it/s] 80%|███████▉  | 318029/400000 [00:36<00:09, 8738.45it/s] 80%|███████▉  | 318903/400000 [00:36<00:09, 8726.55it/s] 80%|███████▉  | 319786/400000 [00:36<00:09, 8756.86it/s] 80%|████████  | 320662/400000 [00:36<00:09, 8594.82it/s] 80%|████████  | 321543/400000 [00:37<00:09, 8658.19it/s] 81%|████████  | 322429/400000 [00:37<00:08, 8717.46it/s] 81%|████████  | 323305/400000 [00:37<00:08, 8729.28it/s] 81%|████████  | 324191/400000 [00:37<00:08, 8767.70it/s] 81%|████████▏ | 325074/400000 [00:37<00:08, 8783.51it/s] 81%|████████▏ | 325955/400000 [00:37<00:08, 8790.25it/s] 82%|████████▏ | 326840/400000 [00:37<00:08, 8806.46it/s] 82%|████████▏ | 327721/400000 [00:37<00:08, 8710.24it/s] 82%|████████▏ | 328599/400000 [00:37<00:08, 8728.31it/s] 82%|████████▏ | 329473/400000 [00:37<00:08, 8728.87it/s] 83%|████████▎ | 330357/400000 [00:38<00:07, 8759.56it/s] 83%|████████▎ | 331235/400000 [00:38<00:07, 8764.06it/s] 83%|████████▎ | 332112/400000 [00:38<00:07, 8736.94it/s] 83%|████████▎ | 332993/400000 [00:38<00:07, 8757.42it/s] 83%|████████▎ | 333876/400000 [00:38<00:07, 8779.03it/s] 84%|████████▎ | 334754/400000 [00:38<00:07, 8742.92it/s] 84%|████████▍ | 335629/400000 [00:38<00:07, 8555.88it/s] 84%|████████▍ | 336490/400000 [00:38<00:07, 8570.51it/s] 84%|████████▍ | 337371/400000 [00:38<00:07, 8638.32it/s] 85%|████████▍ | 338236/400000 [00:38<00:07, 8510.72it/s] 85%|████████▍ | 339119/400000 [00:39<00:07, 8601.54it/s] 85%|████████▌ | 340002/400000 [00:39<00:06, 8668.73it/s] 85%|████████▌ | 340870/400000 [00:39<00:06, 8577.52it/s] 85%|████████▌ | 341748/400000 [00:39<00:06, 8636.30it/s] 86%|████████▌ | 342616/400000 [00:39<00:06, 8649.22it/s] 86%|████████▌ | 343482/400000 [00:39<00:06, 8643.49it/s] 86%|████████▌ | 344367/400000 [00:39<00:06, 8704.02it/s] 86%|████████▋ | 345239/400000 [00:39<00:06, 8707.86it/s] 87%|████████▋ | 346127/400000 [00:39<00:06, 8757.14it/s] 87%|████████▋ | 347010/400000 [00:39<00:06, 8776.13it/s] 87%|████████▋ | 347889/400000 [00:40<00:05, 8780.14it/s] 87%|████████▋ | 348768/400000 [00:40<00:05, 8701.37it/s] 87%|████████▋ | 349639/400000 [00:40<00:05, 8624.92it/s] 88%|████████▊ | 350521/400000 [00:40<00:05, 8681.29it/s] 88%|████████▊ | 351404/400000 [00:40<00:05, 8722.95it/s] 88%|████████▊ | 352287/400000 [00:40<00:05, 8752.38it/s] 88%|████████▊ | 353172/400000 [00:40<00:05, 8780.74it/s] 89%|████████▊ | 354051/400000 [00:40<00:05, 8569.42it/s] 89%|████████▊ | 354915/400000 [00:40<00:05, 8588.62it/s] 89%|████████▉ | 355785/400000 [00:40<00:05, 8620.99it/s] 89%|████████▉ | 356665/400000 [00:41<00:04, 8673.39it/s] 89%|████████▉ | 357533/400000 [00:41<00:05, 8440.60it/s] 90%|████████▉ | 358399/400000 [00:41<00:04, 8410.24it/s] 90%|████████▉ | 359262/400000 [00:41<00:04, 8472.46it/s] 90%|█████████ | 360140/400000 [00:41<00:04, 8559.79it/s] 90%|█████████ | 360997/400000 [00:41<00:04, 8560.06it/s] 90%|█████████ | 361868/400000 [00:41<00:04, 8602.70it/s] 91%|█████████ | 362739/400000 [00:41<00:04, 8633.73it/s] 91%|█████████ | 363603/400000 [00:41<00:04, 8577.98it/s] 91%|█████████ | 364464/400000 [00:41<00:04, 8585.31it/s] 91%|█████████▏| 365350/400000 [00:42<00:03, 8664.54it/s] 92%|█████████▏| 366232/400000 [00:42<00:03, 8709.99it/s] 92%|█████████▏| 367108/400000 [00:42<00:03, 8722.68it/s] 92%|█████████▏| 367981/400000 [00:42<00:03, 8476.47it/s] 92%|█████████▏| 368831/400000 [00:42<00:03, 8432.84it/s] 92%|█████████▏| 369714/400000 [00:42<00:03, 8546.92it/s] 93%|█████████▎| 370585/400000 [00:42<00:03, 8592.94it/s] 93%|█████████▎| 371459/400000 [00:42<00:03, 8636.09it/s] 93%|█████████▎| 372338/400000 [00:42<00:03, 8679.70it/s] 93%|█████████▎| 373224/400000 [00:42<00:03, 8732.16it/s] 94%|█████████▎| 374103/400000 [00:43<00:02, 8747.31it/s] 94%|█████████▎| 374990/400000 [00:43<00:02, 8782.33it/s] 94%|█████████▍| 375873/400000 [00:43<00:02, 8795.27it/s] 94%|█████████▍| 376753/400000 [00:43<00:02, 8735.15it/s] 94%|█████████▍| 377633/400000 [00:43<00:02, 8753.49it/s] 95%|█████████▍| 378509/400000 [00:43<00:02, 8726.59it/s] 95%|█████████▍| 379394/400000 [00:43<00:02, 8760.76it/s] 95%|█████████▌| 380283/400000 [00:43<00:02, 8796.97it/s] 95%|█████████▌| 381163/400000 [00:43<00:02, 8685.00it/s] 96%|█████████▌| 382048/400000 [00:43<00:02, 8732.40it/s] 96%|█████████▌| 382931/400000 [00:44<00:01, 8757.66it/s] 96%|█████████▌| 383813/400000 [00:44<00:01, 8774.36it/s] 96%|█████████▌| 384691/400000 [00:44<00:01, 8732.07it/s] 96%|█████████▋| 385566/400000 [00:44<00:01, 8734.56it/s] 97%|█████████▋| 386448/400000 [00:44<00:01, 8758.79it/s] 97%|█████████▋| 387325/400000 [00:44<00:01, 8761.61it/s] 97%|█████████▋| 388207/400000 [00:44<00:01, 8778.84it/s] 97%|█████████▋| 389091/400000 [00:44<00:01, 8795.41it/s] 97%|█████████▋| 389971/400000 [00:44<00:01, 8744.80it/s] 98%|█████████▊| 390846/400000 [00:44<00:01, 8607.09it/s] 98%|█████████▊| 391729/400000 [00:45<00:00, 8670.20it/s] 98%|█████████▊| 392597/400000 [00:45<00:00, 8616.10it/s] 98%|█████████▊| 393484/400000 [00:45<00:00, 8688.74it/s] 99%|█████████▊| 394354/400000 [00:45<00:00, 8233.53it/s] 99%|█████████▉| 395222/400000 [00:45<00:00, 8361.77it/s] 99%|█████████▉| 396109/400000 [00:45<00:00, 8505.84it/s] 99%|█████████▉| 396996/400000 [00:45<00:00, 8609.59it/s] 99%|█████████▉| 397860/400000 [00:45<00:00, 8605.55it/s]100%|█████████▉| 398723/400000 [00:45<00:00, 8612.24it/s]100%|█████████▉| 399586/400000 [00:46<00:00, 8617.27it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8682.28it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9a993f5be0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011127893164093062 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011129169559797715 	 Accuracy: 51

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

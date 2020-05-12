
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1dfa95cfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 09:12:42.126194
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 09:12:42.130503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 09:12:42.134141
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 09:12:42.137624
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1e06974470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355456.8438
Epoch 2/10

1/1 [==============================] - 0s 88ms/step - loss: 279480.4062
Epoch 3/10

1/1 [==============================] - 0s 89ms/step - loss: 174965.1406
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 100050.0000
Epoch 5/10

1/1 [==============================] - 0s 85ms/step - loss: 56736.2344
Epoch 6/10

1/1 [==============================] - 0s 87ms/step - loss: 34117.0977
Epoch 7/10

1/1 [==============================] - 0s 89ms/step - loss: 22001.2266
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 15181.7793
Epoch 9/10

1/1 [==============================] - 0s 87ms/step - loss: 11090.8691
Epoch 10/10

1/1 [==============================] - 0s 84ms/step - loss: 8519.7461

  #### Inference Need return ypred, ytrue ######################### 
[[ -0.61071414   0.96710646  -0.63852197  -1.0329019   -0.2391038
    0.32724297  -0.9322377   -0.45481342   0.52405375  -0.92657447
   -0.8221993   -0.70734406  -0.69065076   1.0548586   -0.5234905
    0.74812704   0.4338206   -2.3300843    0.68814015  -0.0358775
    0.6418687    0.8680066    0.11926341  -0.2572652    1.6995758
   -0.31410044   0.1360544   -1.2920294    0.03201078  -0.20624824
    0.40530473  -1.0773435   -1.3189516   -0.5210413   -1.6560355
    1.2732939   -1.1351976    0.5559008    0.23338614  -0.13113943
   -0.39172137  -1.0462333    0.8743222    1.0223664    0.27674124
   -0.11749482   0.67680305   0.03215203  -0.43778792  -0.7938709
   -0.42155173  -0.12266791   1.0671077   -0.97318375   1.3520799
    0.61384034  -1.0182297    0.38399875  -0.24933141   0.27908203
   -0.0374338    5.767424     6.985853     6.41919      8.142726
    6.5192914    7.002329     6.9781723    7.9012446    5.4220843
    7.434514     5.5745573    6.788436     6.651314     6.2713184
    8.193825     7.2831445    5.5819063    5.8553934    6.9504514
    6.5090294    8.5463295    6.985938     6.2245126    8.656888
    5.923866     7.41364      7.5494695    5.9813194    5.479608
    6.3764586    5.891132     8.272818     7.442151     7.1770573
    5.265813     7.1849003    7.497557     8.504842     5.2285585
    6.7905884    6.144422     5.742168     5.7228465    6.0585876
    7.9439645    8.043557     6.6551857    6.204081     8.435391
    7.561956     5.3179274    7.1845083    7.596787     6.4781375
    6.012605     6.004037     8.078361     7.245418     5.7386336
   -0.73851985  -1.5194986   -0.31215182  -0.1809324   -0.3707075
   -2.5370545    0.41118276   0.10624117   1.2209178    0.65842956
    2.2888014   -0.2459878   -0.25381866  -0.81680423   0.44530976
    0.7550587   -2.223225    -1.7475389    1.1704845    1.6514864
    1.2883784    1.4503224   -1.9243083    0.44765466   1.062298
   -0.66039896  -1.0219424   -1.9583687   -0.1726107    0.2846165
    1.0937914   -1.5867624   -0.29869917   0.18244353   0.5074614
   -1.7543412   -0.1472987    1.0034924   -0.55442095  -1.127794
   -0.55330044  -1.3496248    0.4490593    0.30703616   1.7957861
    0.24191296  -1.0389973   -0.5034418   -1.2061749    0.76983523
    0.3328517    0.93807554  -0.39408004  -0.18851675  -0.5800344
   -0.5662066    0.7491405   -0.5331432    0.03048491  -0.37829614
    1.3809186    1.797301     0.43127143   1.5802276    0.59035665
    0.56579113   1.4134908    1.7554997    0.6104057    0.7466688
    0.3700428    0.17608309   2.1703897    0.64255834   1.2202852
    0.5609326    1.2419887    0.85265803   1.1859403    1.9857395
    1.7890668    0.39259553   1.6046245    1.5854868    1.5962241
    0.90800655   0.26582354   0.34849447   0.5762442    0.11161852
    0.24685049   0.5480848    1.0323646    2.1082296    0.6920765
    0.5256008    0.3934996    0.64191914   0.7149515    1.2458037
    1.7049993    2.999332     0.82946247   0.3818587    1.0506725
    0.42245692   1.4816319    0.20724201   0.96734184   0.70536757
    1.8755374    0.8547063    1.6995707    0.36850816   0.28578877
    0.46532875   1.7716601    0.30215597   1.5302733    1.4833765
    0.07853287   6.795483     8.63648      7.3971586    8.279613
    6.3229156    8.297678     7.1398177    7.522612     8.405933
    8.082985     7.6209774    7.4290137    6.156893     7.8747573
    7.5075936    6.49547      8.103868     7.1009464    7.3891993
    7.868702     6.1450872    7.401282     8.9054165    8.095923
    8.731796     6.997743     6.71721      8.877424     6.8862343
    7.306972     7.15017      7.08597      6.7123327    6.9181905
    7.50745      8.183361     7.219039     6.773513     7.6422453
    7.1026807    6.3124523    7.6839323    7.127184     7.2106442
    7.9732223    6.344983     6.3109508    6.853148     8.371577
    6.962493     6.184043     7.7403383    6.3507786    8.218155
    6.048943     6.6988454    7.1600184    7.0881023    7.3718586
    1.0299072    1.2795962    2.2315092    0.27771854   1.8824863
    0.7114122    0.9474914    0.65672195   1.8045431    0.77318907
    1.2172225    0.25971645   0.32387912   0.2713493    1.0100746
    0.7876569    1.1187251    2.924828     2.7667148    1.6517324
    1.3580581    1.9258902    1.5717514    0.24044734   0.9168243
    0.14342898   0.67201436   0.8794787    0.45463884   0.9735587
    1.6415071    0.92643905   1.7303486    0.38447654   0.4803064
    1.3814318    1.3401904    1.4985172    1.1818053    0.77458787
    0.5184751    0.24536967   1.0184691    1.0062208    2.011902
    0.19153672   1.3528447    0.3038506    1.0467862    1.3856704
    1.3459055    0.6754805    0.43015277   1.2745951    1.2155284
    1.3104901    0.89775985   2.384707     1.2192345    0.10095304
   -5.251969     1.7801864  -10.315622  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 09:12:52.736379
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.2108
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 09:12:52.740369
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9085.91
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 09:12:52.743833
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.9173
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 09:12:52.747046
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -812.707
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139766378209408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139765436649712
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139765436650216
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139765436650720
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139765436651224
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139765436651728

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1dfa8764a8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.574793
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.534627
grad_step = 000002, loss = 0.500179
grad_step = 000003, loss = 0.463403
grad_step = 000004, loss = 0.422280
grad_step = 000005, loss = 0.380330
grad_step = 000006, loss = 0.348730
grad_step = 000007, loss = 0.342993
grad_step = 000008, loss = 0.335205
grad_step = 000009, loss = 0.313958
grad_step = 000010, loss = 0.293604
grad_step = 000011, loss = 0.280301
grad_step = 000012, loss = 0.271336
grad_step = 000013, loss = 0.262192
grad_step = 000014, loss = 0.250762
grad_step = 000015, loss = 0.237398
grad_step = 000016, loss = 0.223989
grad_step = 000017, loss = 0.212210
grad_step = 000018, loss = 0.201838
grad_step = 000019, loss = 0.191654
grad_step = 000020, loss = 0.181423
grad_step = 000021, loss = 0.171050
grad_step = 000022, loss = 0.160659
grad_step = 000023, loss = 0.150920
grad_step = 000024, loss = 0.141896
grad_step = 000025, loss = 0.133014
grad_step = 000026, loss = 0.123994
grad_step = 000027, loss = 0.115186
grad_step = 000028, loss = 0.106957
grad_step = 000029, loss = 0.099130
grad_step = 000030, loss = 0.091584
grad_step = 000031, loss = 0.084510
grad_step = 000032, loss = 0.077864
grad_step = 000033, loss = 0.071560
grad_step = 000034, loss = 0.065596
grad_step = 000035, loss = 0.060054
grad_step = 000036, loss = 0.054751
grad_step = 000037, loss = 0.049543
grad_step = 000038, loss = 0.044615
grad_step = 000039, loss = 0.040377
grad_step = 000040, loss = 0.036820
grad_step = 000041, loss = 0.033395
grad_step = 000042, loss = 0.029910
grad_step = 000043, loss = 0.026793
grad_step = 000044, loss = 0.024267
grad_step = 000045, loss = 0.021983
grad_step = 000046, loss = 0.019631
grad_step = 000047, loss = 0.017415
grad_step = 000048, loss = 0.015632
grad_step = 000049, loss = 0.014101
grad_step = 000050, loss = 0.012551
grad_step = 000051, loss = 0.011081
grad_step = 000052, loss = 0.009897
grad_step = 000053, loss = 0.008921
grad_step = 000054, loss = 0.007979
grad_step = 000055, loss = 0.007097
grad_step = 000056, loss = 0.006390
grad_step = 000057, loss = 0.005813
grad_step = 000058, loss = 0.005245
grad_step = 000059, loss = 0.004734
grad_step = 000060, loss = 0.004351
grad_step = 000061, loss = 0.004032
grad_step = 000062, loss = 0.003716
grad_step = 000063, loss = 0.003452
grad_step = 000064, loss = 0.003263
grad_step = 000065, loss = 0.003105
grad_step = 000066, loss = 0.002954
grad_step = 000067, loss = 0.002830
grad_step = 000068, loss = 0.002738
grad_step = 000069, loss = 0.002667
grad_step = 000070, loss = 0.002598
grad_step = 000071, loss = 0.002533
grad_step = 000072, loss = 0.002493
grad_step = 000073, loss = 0.002471
grad_step = 000074, loss = 0.002441
grad_step = 000075, loss = 0.002411
grad_step = 000076, loss = 0.002396
grad_step = 000077, loss = 0.002384
grad_step = 000078, loss = 0.002369
grad_step = 000079, loss = 0.002359
grad_step = 000080, loss = 0.002355
grad_step = 000081, loss = 0.002349
grad_step = 000082, loss = 0.002340
grad_step = 000083, loss = 0.002332
grad_step = 000084, loss = 0.002326
grad_step = 000085, loss = 0.002316
grad_step = 000086, loss = 0.002302
grad_step = 000087, loss = 0.002292
grad_step = 000088, loss = 0.002284
grad_step = 000089, loss = 0.002269
grad_step = 000090, loss = 0.002256
grad_step = 000091, loss = 0.002245
grad_step = 000092, loss = 0.002231
grad_step = 000093, loss = 0.002215
grad_step = 000094, loss = 0.002203
grad_step = 000095, loss = 0.002190
grad_step = 000096, loss = 0.002176
grad_step = 000097, loss = 0.002162
grad_step = 000098, loss = 0.002150
grad_step = 000099, loss = 0.002138
grad_step = 000100, loss = 0.002125
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002115
grad_step = 000102, loss = 0.002105
grad_step = 000103, loss = 0.002094
grad_step = 000104, loss = 0.002085
grad_step = 000105, loss = 0.002077
grad_step = 000106, loss = 0.002068
grad_step = 000107, loss = 0.002061
grad_step = 000108, loss = 0.002054
grad_step = 000109, loss = 0.002047
grad_step = 000110, loss = 0.002041
grad_step = 000111, loss = 0.002035
grad_step = 000112, loss = 0.002029
grad_step = 000113, loss = 0.002024
grad_step = 000114, loss = 0.002018
grad_step = 000115, loss = 0.002013
grad_step = 000116, loss = 0.002007
grad_step = 000117, loss = 0.002002
grad_step = 000118, loss = 0.001997
grad_step = 000119, loss = 0.001992
grad_step = 000120, loss = 0.001987
grad_step = 000121, loss = 0.001982
grad_step = 000122, loss = 0.001976
grad_step = 000123, loss = 0.001971
grad_step = 000124, loss = 0.001966
grad_step = 000125, loss = 0.001961
grad_step = 000126, loss = 0.001955
grad_step = 000127, loss = 0.001950
grad_step = 000128, loss = 0.001945
grad_step = 000129, loss = 0.001939
grad_step = 000130, loss = 0.001934
grad_step = 000131, loss = 0.001928
grad_step = 000132, loss = 0.001922
grad_step = 000133, loss = 0.001917
grad_step = 000134, loss = 0.001911
grad_step = 000135, loss = 0.001905
grad_step = 000136, loss = 0.001899
grad_step = 000137, loss = 0.001893
grad_step = 000138, loss = 0.001888
grad_step = 000139, loss = 0.001882
grad_step = 000140, loss = 0.001875
grad_step = 000141, loss = 0.001869
grad_step = 000142, loss = 0.001863
grad_step = 000143, loss = 0.001857
grad_step = 000144, loss = 0.001851
grad_step = 000145, loss = 0.001845
grad_step = 000146, loss = 0.001839
grad_step = 000147, loss = 0.001832
grad_step = 000148, loss = 0.001826
grad_step = 000149, loss = 0.001819
grad_step = 000150, loss = 0.001813
grad_step = 000151, loss = 0.001806
grad_step = 000152, loss = 0.001799
grad_step = 000153, loss = 0.001792
grad_step = 000154, loss = 0.001785
grad_step = 000155, loss = 0.001778
grad_step = 000156, loss = 0.001770
grad_step = 000157, loss = 0.001763
grad_step = 000158, loss = 0.001755
grad_step = 000159, loss = 0.001748
grad_step = 000160, loss = 0.001741
grad_step = 000161, loss = 0.001739
grad_step = 000162, loss = 0.001749
grad_step = 000163, loss = 0.001797
grad_step = 000164, loss = 0.001901
grad_step = 000165, loss = 0.002069
grad_step = 000166, loss = 0.002091
grad_step = 000167, loss = 0.001921
grad_step = 000168, loss = 0.001707
grad_step = 000169, loss = 0.001754
grad_step = 000170, loss = 0.001917
grad_step = 000171, loss = 0.001873
grad_step = 000172, loss = 0.001705
grad_step = 000173, loss = 0.001694
grad_step = 000174, loss = 0.001809
grad_step = 000175, loss = 0.001807
grad_step = 000176, loss = 0.001683
grad_step = 000177, loss = 0.001668
grad_step = 000178, loss = 0.001752
grad_step = 000179, loss = 0.001745
grad_step = 000180, loss = 0.001658
grad_step = 000181, loss = 0.001646
grad_step = 000182, loss = 0.001705
grad_step = 000183, loss = 0.001701
grad_step = 000184, loss = 0.001639
grad_step = 000185, loss = 0.001624
grad_step = 000186, loss = 0.001666
grad_step = 000187, loss = 0.001669
grad_step = 000188, loss = 0.001626
grad_step = 000189, loss = 0.001603
grad_step = 000190, loss = 0.001627
grad_step = 000191, loss = 0.001640
grad_step = 000192, loss = 0.001618
grad_step = 000193, loss = 0.001588
grad_step = 000194, loss = 0.001592
grad_step = 000195, loss = 0.001608
grad_step = 000196, loss = 0.001607
grad_step = 000197, loss = 0.001584
grad_step = 000198, loss = 0.001570
grad_step = 000199, loss = 0.001571
grad_step = 000200, loss = 0.001581
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001580
grad_step = 000202, loss = 0.001568
grad_step = 000203, loss = 0.001554
grad_step = 000204, loss = 0.001549
grad_step = 000205, loss = 0.001551
grad_step = 000206, loss = 0.001555
grad_step = 000207, loss = 0.001554
grad_step = 000208, loss = 0.001548
grad_step = 000209, loss = 0.001539
grad_step = 000210, loss = 0.001531
grad_step = 000211, loss = 0.001526
grad_step = 000212, loss = 0.001524
grad_step = 000213, loss = 0.001524
grad_step = 000214, loss = 0.001525
grad_step = 000215, loss = 0.001525
grad_step = 000216, loss = 0.001525
grad_step = 000217, loss = 0.001525
grad_step = 000218, loss = 0.001525
grad_step = 000219, loss = 0.001525
grad_step = 000220, loss = 0.001527
grad_step = 000221, loss = 0.001530
grad_step = 000222, loss = 0.001536
grad_step = 000223, loss = 0.001546
grad_step = 000224, loss = 0.001561
grad_step = 000225, loss = 0.001579
grad_step = 000226, loss = 0.001603
grad_step = 000227, loss = 0.001616
grad_step = 000228, loss = 0.001620
grad_step = 000229, loss = 0.001594
grad_step = 000230, loss = 0.001551
grad_step = 000231, loss = 0.001502
grad_step = 000232, loss = 0.001473
grad_step = 000233, loss = 0.001473
grad_step = 000234, loss = 0.001494
grad_step = 000235, loss = 0.001519
grad_step = 000236, loss = 0.001535
grad_step = 000237, loss = 0.001542
grad_step = 000238, loss = 0.001531
grad_step = 000239, loss = 0.001509
grad_step = 000240, loss = 0.001485
grad_step = 000241, loss = 0.001465
grad_step = 000242, loss = 0.001450
grad_step = 000243, loss = 0.001444
grad_step = 000244, loss = 0.001443
grad_step = 000245, loss = 0.001448
grad_step = 000246, loss = 0.001454
grad_step = 000247, loss = 0.001462
grad_step = 000248, loss = 0.001471
grad_step = 000249, loss = 0.001480
grad_step = 000250, loss = 0.001488
grad_step = 000251, loss = 0.001495
grad_step = 000252, loss = 0.001500
grad_step = 000253, loss = 0.001500
grad_step = 000254, loss = 0.001496
grad_step = 000255, loss = 0.001484
grad_step = 000256, loss = 0.001468
grad_step = 000257, loss = 0.001448
grad_step = 000258, loss = 0.001429
grad_step = 000259, loss = 0.001415
grad_step = 000260, loss = 0.001407
grad_step = 000261, loss = 0.001404
grad_step = 000262, loss = 0.001404
grad_step = 000263, loss = 0.001407
grad_step = 000264, loss = 0.001413
grad_step = 000265, loss = 0.001421
grad_step = 000266, loss = 0.001435
grad_step = 000267, loss = 0.001454
grad_step = 000268, loss = 0.001480
grad_step = 000269, loss = 0.001516
grad_step = 000270, loss = 0.001551
grad_step = 000271, loss = 0.001586
grad_step = 000272, loss = 0.001597
grad_step = 000273, loss = 0.001572
grad_step = 000274, loss = 0.001503
grad_step = 000275, loss = 0.001427
grad_step = 000276, loss = 0.001380
grad_step = 000277, loss = 0.001378
grad_step = 000278, loss = 0.001409
grad_step = 000279, loss = 0.001447
grad_step = 000280, loss = 0.001471
grad_step = 000281, loss = 0.001464
grad_step = 000282, loss = 0.001435
grad_step = 000283, loss = 0.001396
grad_step = 000284, loss = 0.001366
grad_step = 000285, loss = 0.001356
grad_step = 000286, loss = 0.001364
grad_step = 000287, loss = 0.001381
grad_step = 000288, loss = 0.001397
grad_step = 000289, loss = 0.001405
grad_step = 000290, loss = 0.001400
grad_step = 000291, loss = 0.001387
grad_step = 000292, loss = 0.001369
grad_step = 000293, loss = 0.001353
grad_step = 000294, loss = 0.001341
grad_step = 000295, loss = 0.001336
grad_step = 000296, loss = 0.001336
grad_step = 000297, loss = 0.001340
grad_step = 000298, loss = 0.001345
grad_step = 000299, loss = 0.001351
grad_step = 000300, loss = 0.001356
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001360
grad_step = 000302, loss = 0.001363
grad_step = 000303, loss = 0.001365
grad_step = 000304, loss = 0.001367
grad_step = 000305, loss = 0.001367
grad_step = 000306, loss = 0.001367
grad_step = 000307, loss = 0.001367
grad_step = 000308, loss = 0.001368
grad_step = 000309, loss = 0.001368
grad_step = 000310, loss = 0.001369
grad_step = 000311, loss = 0.001370
grad_step = 000312, loss = 0.001371
grad_step = 000313, loss = 0.001370
grad_step = 000314, loss = 0.001370
grad_step = 000315, loss = 0.001368
grad_step = 000316, loss = 0.001366
grad_step = 000317, loss = 0.001361
grad_step = 000318, loss = 0.001356
grad_step = 000319, loss = 0.001350
grad_step = 000320, loss = 0.001344
grad_step = 000321, loss = 0.001336
grad_step = 000322, loss = 0.001328
grad_step = 000323, loss = 0.001319
grad_step = 000324, loss = 0.001310
grad_step = 000325, loss = 0.001301
grad_step = 000326, loss = 0.001293
grad_step = 000327, loss = 0.001287
grad_step = 000328, loss = 0.001282
grad_step = 000329, loss = 0.001278
grad_step = 000330, loss = 0.001274
grad_step = 000331, loss = 0.001272
grad_step = 000332, loss = 0.001270
grad_step = 000333, loss = 0.001269
grad_step = 000334, loss = 0.001269
grad_step = 000335, loss = 0.001270
grad_step = 000336, loss = 0.001273
grad_step = 000337, loss = 0.001281
grad_step = 000338, loss = 0.001296
grad_step = 000339, loss = 0.001322
grad_step = 000340, loss = 0.001368
grad_step = 000341, loss = 0.001440
grad_step = 000342, loss = 0.001548
grad_step = 000343, loss = 0.001661
grad_step = 000344, loss = 0.001747
grad_step = 000345, loss = 0.001698
grad_step = 000346, loss = 0.001529
grad_step = 000347, loss = 0.001322
grad_step = 000348, loss = 0.001240
grad_step = 000349, loss = 0.001310
grad_step = 000350, loss = 0.001422
grad_step = 000351, loss = 0.001454
grad_step = 000352, loss = 0.001363
grad_step = 000353, loss = 0.001258
grad_step = 000354, loss = 0.001232
grad_step = 000355, loss = 0.001289
grad_step = 000356, loss = 0.001351
grad_step = 000357, loss = 0.001346
grad_step = 000358, loss = 0.001287
grad_step = 000359, loss = 0.001229
grad_step = 000360, loss = 0.001221
grad_step = 000361, loss = 0.001255
grad_step = 000362, loss = 0.001285
grad_step = 000363, loss = 0.001282
grad_step = 000364, loss = 0.001247
grad_step = 000365, loss = 0.001214
grad_step = 000366, loss = 0.001207
grad_step = 000367, loss = 0.001222
grad_step = 000368, loss = 0.001241
grad_step = 000369, loss = 0.001243
grad_step = 000370, loss = 0.001230
grad_step = 000371, loss = 0.001209
grad_step = 000372, loss = 0.001196
grad_step = 000373, loss = 0.001194
grad_step = 000374, loss = 0.001202
grad_step = 000375, loss = 0.001210
grad_step = 000376, loss = 0.001212
grad_step = 000377, loss = 0.001207
grad_step = 000378, loss = 0.001197
grad_step = 000379, loss = 0.001187
grad_step = 000380, loss = 0.001180
grad_step = 000381, loss = 0.001179
grad_step = 000382, loss = 0.001181
grad_step = 000383, loss = 0.001184
grad_step = 000384, loss = 0.001185
grad_step = 000385, loss = 0.001184
grad_step = 000386, loss = 0.001181
grad_step = 000387, loss = 0.001176
grad_step = 000388, loss = 0.001171
grad_step = 000389, loss = 0.001167
grad_step = 000390, loss = 0.001163
grad_step = 000391, loss = 0.001161
grad_step = 000392, loss = 0.001159
grad_step = 000393, loss = 0.001158
grad_step = 000394, loss = 0.001157
grad_step = 000395, loss = 0.001157
grad_step = 000396, loss = 0.001157
grad_step = 000397, loss = 0.001158
grad_step = 000398, loss = 0.001159
grad_step = 000399, loss = 0.001162
grad_step = 000400, loss = 0.001167
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001174
grad_step = 000402, loss = 0.001185
grad_step = 000403, loss = 0.001202
grad_step = 000404, loss = 0.001223
grad_step = 000405, loss = 0.001256
grad_step = 000406, loss = 0.001292
grad_step = 000407, loss = 0.001338
grad_step = 000408, loss = 0.001371
grad_step = 000409, loss = 0.001395
grad_step = 000410, loss = 0.001368
grad_step = 000411, loss = 0.001310
grad_step = 000412, loss = 0.001223
grad_step = 000413, loss = 0.001152
grad_step = 000414, loss = 0.001123
grad_step = 000415, loss = 0.001139
grad_step = 000416, loss = 0.001180
grad_step = 000417, loss = 0.001215
grad_step = 000418, loss = 0.001230
grad_step = 000419, loss = 0.001211
grad_step = 000420, loss = 0.001177
grad_step = 000421, loss = 0.001139
grad_step = 000422, loss = 0.001114
grad_step = 000423, loss = 0.001106
grad_step = 000424, loss = 0.001115
grad_step = 000425, loss = 0.001131
grad_step = 000426, loss = 0.001148
grad_step = 000427, loss = 0.001160
grad_step = 000428, loss = 0.001163
grad_step = 000429, loss = 0.001165
grad_step = 000430, loss = 0.001156
grad_step = 000431, loss = 0.001141
grad_step = 000432, loss = 0.001123
grad_step = 000433, loss = 0.001105
grad_step = 000434, loss = 0.001092
grad_step = 000435, loss = 0.001086
grad_step = 000436, loss = 0.001085
grad_step = 000437, loss = 0.001088
grad_step = 000438, loss = 0.001094
grad_step = 000439, loss = 0.001100
grad_step = 000440, loss = 0.001108
grad_step = 000441, loss = 0.001114
grad_step = 000442, loss = 0.001121
grad_step = 000443, loss = 0.001125
grad_step = 000444, loss = 0.001130
grad_step = 000445, loss = 0.001132
grad_step = 000446, loss = 0.001133
grad_step = 000447, loss = 0.001131
grad_step = 000448, loss = 0.001129
grad_step = 000449, loss = 0.001122
grad_step = 000450, loss = 0.001116
grad_step = 000451, loss = 0.001107
grad_step = 000452, loss = 0.001098
grad_step = 000453, loss = 0.001088
grad_step = 000454, loss = 0.001080
grad_step = 000455, loss = 0.001072
grad_step = 000456, loss = 0.001066
grad_step = 000457, loss = 0.001060
grad_step = 000458, loss = 0.001055
grad_step = 000459, loss = 0.001052
grad_step = 000460, loss = 0.001049
grad_step = 000461, loss = 0.001046
grad_step = 000462, loss = 0.001045
grad_step = 000463, loss = 0.001044
grad_step = 000464, loss = 0.001044
grad_step = 000465, loss = 0.001046
grad_step = 000466, loss = 0.001050
grad_step = 000467, loss = 0.001058
grad_step = 000468, loss = 0.001075
grad_step = 000469, loss = 0.001103
grad_step = 000470, loss = 0.001153
grad_step = 000471, loss = 0.001224
grad_step = 000472, loss = 0.001339
grad_step = 000473, loss = 0.001451
grad_step = 000474, loss = 0.001573
grad_step = 000475, loss = 0.001544
grad_step = 000476, loss = 0.001400
grad_step = 000477, loss = 0.001166
grad_step = 000478, loss = 0.001027
grad_step = 000479, loss = 0.001059
grad_step = 000480, loss = 0.001185
grad_step = 000481, loss = 0.001265
grad_step = 000482, loss = 0.001207
grad_step = 000483, loss = 0.001090
grad_step = 000484, loss = 0.001011
grad_step = 000485, loss = 0.001030
grad_step = 000486, loss = 0.001102
grad_step = 000487, loss = 0.001138
grad_step = 000488, loss = 0.001109
grad_step = 000489, loss = 0.001042
grad_step = 000490, loss = 0.000998
grad_step = 000491, loss = 0.001004
grad_step = 000492, loss = 0.001040
grad_step = 000493, loss = 0.001070
grad_step = 000494, loss = 0.001066
grad_step = 000495, loss = 0.001037
grad_step = 000496, loss = 0.001001
grad_step = 000497, loss = 0.000983
grad_step = 000498, loss = 0.000989
grad_step = 000499, loss = 0.001007
grad_step = 000500, loss = 0.001020
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

  date_run                              2020-05-12 09:13:15.150638
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.30469
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 09:13:15.156243
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.270977
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 09:13:15.162229
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149487
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 09:13:15.167442
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -3.11759
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
0   2020-05-12 09:12:42.126194  ...    mean_absolute_error
1   2020-05-12 09:12:42.130503  ...     mean_squared_error
2   2020-05-12 09:12:42.134141  ...  median_absolute_error
3   2020-05-12 09:12:42.137624  ...               r2_score
4   2020-05-12 09:12:52.736379  ...    mean_absolute_error
5   2020-05-12 09:12:52.740369  ...     mean_squared_error
6   2020-05-12 09:12:52.743833  ...  median_absolute_error
7   2020-05-12 09:12:52.747046  ...               r2_score
8   2020-05-12 09:13:15.150638  ...    mean_absolute_error
9   2020-05-12 09:13:15.156243  ...     mean_squared_error
10  2020-05-12 09:13:15.162229  ...  median_absolute_error
11  2020-05-12 09:13:15.167442  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 152163.85it/s]  1%|          | 98304/9912422 [00:00<00:51, 190945.78it/s]  4%|▍         | 417792/9912422 [00:01<00:36, 261546.09it/s] 11%|█         | 1105920/9912422 [00:01<00:23, 367613.56it/s] 19%|█▊        | 1843200/9912422 [00:01<00:15, 508389.38it/s] 29%|██▊       | 2842624/9912422 [00:01<00:09, 710771.06it/s] 35%|███▌      | 3514368/9912422 [00:01<00:06, 944191.82it/s] 48%|████▊     | 4784128/9912422 [00:01<00:04, 1279893.43it/s] 62%|██████▏   | 6184960/9912422 [00:01<00:02, 1714933.80it/s] 70%|██████▉   | 6938624/9912422 [00:02<00:01, 2223927.63it/s] 86%|████████▋ | 8552448/9912422 [00:02<00:00, 2876316.39it/s] 99%|█████████▉| 9854976/9912422 [00:02<00:00, 3672806.14it/s]9920512it [00:02, 4173786.61it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 150859.90it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 305970.64it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 398830.25it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 552191.51it/s]1654784it [00:00, 2799875.41it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53296.20it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f024f93c0f0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9fbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f024e84bbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f024e80feb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0201217da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9fbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9fe10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9c048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9fe10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9c0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f01ebf9fe10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1b1bcfd1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=32bbb12bd112f22c2aece9dc035cfd4d25a1c8f6136865d2eb020476d9f91b26
  Stored in directory: /tmp/pip-ephem-wheel-cache-ysl87901/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1ab38e51d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 41s
   90112/17464789 [..............................] - ETA: 39s
  196608/17464789 [..............................] - ETA: 23s
  385024/17464789 [..............................] - ETA: 15s
  770048/17464789 [>.............................] - ETA: 8s 
 1556480/17464789 [=>............................] - ETA: 4s
 3096576/17464789 [====>.........................] - ETA: 2s
 6012928/17464789 [=========>....................] - ETA: 1s
 8536064/17464789 [=============>................] - ETA: 0s
11501568/17464789 [==================>...........] - ETA: 0s
14417920/17464789 [=======================>......] - ETA: 0s
17235968/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 09:14:46.982441: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 09:14:46.986546: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-12 09:14:46.986861: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d62ed99d20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 09:14:46.986888: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5440 - accuracy: 0.5080 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5838 - accuracy: 0.5054
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6272 - accuracy: 0.5026
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6245 - accuracy: 0.5027
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 4s - loss: 7.6987 - accuracy: 0.4979
12000/25000 [=============>................] - ETA: 3s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6925 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6470 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6798 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 8s 339us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 09:15:02.414168
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 09:15:02.414168  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 09:15:08.694613: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 09:15:08.699489: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-12 09:15:08.699651: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a9c2d9df40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 09:15:08.699667: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fa58bdcdbe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7706 - crf_viterbi_accuracy: 0.2267 - val_loss: 1.6576 - val_crf_viterbi_accuracy: 0.2267

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa58bdcdcf8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6513 - accuracy: 0.5010 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5685 - accuracy: 0.5064
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5107 - accuracy: 0.5102
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5177 - accuracy: 0.5097
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5152 - accuracy: 0.5099
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5048 - accuracy: 0.5106
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
11000/25000 [============>.................] - ETA: 4s - loss: 7.5147 - accuracy: 0.5099
12000/25000 [=============>................] - ETA: 3s - loss: 7.5529 - accuracy: 0.5074
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5734 - accuracy: 0.5061
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5943 - accuracy: 0.5047
15000/25000 [=================>............] - ETA: 2s - loss: 7.6032 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6235 - accuracy: 0.5028
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6843 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 342us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fa5643aba58> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:55:48, 20.1kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:00<8:23:08, 28.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.74M/862M [00:00<5:50:52, 40.8kB/s].vector_cache/glove.6B.zip:   1%|          | 10.5M/862M [00:00<4:03:45, 58.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:00<2:48:35, 83.2kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.2M/862M [00:00<1:56:36, 119kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.3M/862M [00:01<1:20:46, 170kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:01<55:50, 242kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:01<39:30, 341kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:02<6:08:19, 36.6kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<4:17:45, 52.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<8:41:07, 25.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<6:04:28, 36.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<9:46:33, 22.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:50:06, 32.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<10:28:36, 21.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<7:19:28, 30.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:44:20, 20.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:30:29, 29.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:43:26, 20.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:29:48, 29.5kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:53:59, 20.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:37:16, 28.9kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:34:02, 20.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:23:13, 29.8kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:47:08, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:32:23, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:49:36, 20.3kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:34:08, 28.9kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:39:55, 20.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:27:19, 29.3kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:48:25, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:33:21, 28.8kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:24:33, 20.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:16:35, 29.8kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:37:32, 20.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:25:41, 29.2kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:34:25, 20.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:23:28, 29.2kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:41:49, 20.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:28:42, 28.8kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:23:17, 20.7kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:15:41, 29.6kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:32:37, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:22:15, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:22:38, 20.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:15:14, 29.4kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:28:48, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:19:31, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:32:04, 20.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:21:48, 28.9kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:30:19, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:20:35, 28.9kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:29:29, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:19:59, 28.8kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:29:20, 20.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:19:55, 28.7kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:19:35, 20.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:13:08, 29.1kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:05:14, 20.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:03:02, 29.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:17:22, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:11:31, 29.1kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:19:29, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<7:13:01, 28.9kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:10:48, 20.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<7:06:55, 29.2kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:16:41, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:11:01, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:14:34, 20.2kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:09:39, 28.9kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<9:48:36, 21.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<6:51:25, 30.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:00:23, 20.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<6:59:38, 29.4kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:05:41, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<7:03:21, 29.0kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:00:21, 20.5kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<6:59:40, 29.2kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<9:48:35, 20.8kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<6:51:21, 29.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:00:56, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<7:00:00, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:02:46, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<7:01:22, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<9:36:08, 21.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:42:40, 30.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:50:36, 20.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:52:54, 29.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:20:55, 21.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:32:01, 30.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:43:14, 20.7kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:47:42, 29.5kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:30:04, 21.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:38:25, 30.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:42:31, 20.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<6:46:54, 29.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<4:48:50, 41.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<8:01:51, 24.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<5:36:50, 35.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:00:29, 22.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:17:33, 31.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<4:28:14, 44.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<7:43:57, 25.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<5:24:20, 36.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<8:45:29, 22.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:07:19, 32.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:04:07, 21.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:20:16, 30.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:28:20, 20.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:37:13, 29.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<4:39:32, 42.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<7:38:27, 25.6kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<5:20:29, 36.5kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<8:39:37, 22.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:02:55, 32.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<4:19:31, 44.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<7:34:45, 25.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<5:17:53, 36.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<8:37:30, 22.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:01:49, 32.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<8:32:05, 22.6kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<5:57:53, 32.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:06:39, 21.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:22:02, 30.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:09:31, 21.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:24:05, 29.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:02:46, 21.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:19:18, 30.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:12:45, 20.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:26:15, 29.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:21:31, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:32:23, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:19:27, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:30:54, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:24:32, 20.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:34:28, 28.7kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:18:43, 20.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:30:28, 28.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<8:58:18, 21.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:16:13, 29.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<8:48:59, 21.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:09:38, 30.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:05:38, 20.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:21:16, 29.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<9:08:24, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:23:10, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:10:32, 20.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:24:40, 28.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:08:28, 20.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:23:17, 28.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<8:52:35, 20.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:12:07, 29.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:02:26, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:19:00, 29.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<9:02:08, 20.3kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:18:51, 28.9kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<8:43:40, 20.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:05:53, 29.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<8:51:33, 20.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:11:27, 29.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:39:25, 21.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:02:54, 29.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:54:01, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:13:06, 29.0kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:55:53, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:14:25, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:47:42, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<6:08:41, 29.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:48:33, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:09:19, 29.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:36:05, 20.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<6:00:32, 29.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:46:19, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:07:42, 29.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:49:11, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:09:43, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:39:47, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<6:03:06, 29.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:45:25, 20.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<6:07:04, 28.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:38:12, 20.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<6:02:00, 29.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:41:38, 20.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<6:04:26, 28.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:32:54, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<5:58:17, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:54:35, 19.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<6:13:31, 27.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:01:22, 21.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:36:17, 30.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:20:43, 20.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:49:47, 29.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:24:16, 20.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:52:20, 29.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:08:39, 21.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:41:20, 30.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:21:19, 20.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:50:11, 29.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:24:52, 20.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:52:42, 28.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:09:46, 20.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:42:07, 29.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:19:51, 20.3kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:49:10, 29.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:16:20, 20.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:46:41, 29.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:20:51, 20.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:49:51, 28.7kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:15:05, 20.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:45:48, 29.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:16:50, 20.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:47:08, 28.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<7:46:46, 21.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:26:02, 30.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:02:26, 20.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:37:02, 29.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<7:45:57, 21.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:25:27, 30.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<8:01:24, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:36:19, 29.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<7:41:13, 21.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:22:08, 30.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<7:57:02, 20.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:33:15, 29.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<7:39:07, 21.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:20:39, 30.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<7:54:05, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:31:12, 29.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:33:12, 21.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:16:36, 30.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:31:15, 21.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:15:10, 30.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:44:14, 20.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:24:18, 29.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:30:37, 21.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:14:42, 30.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:43:35, 20.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:23:45, 29.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:47:20, 20.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:26:21, 29.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:47:47, 20.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:26:40, 29.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:47:16, 20.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:26:19, 28.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:41:38, 20.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:22:22, 29.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:42:02, 20.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:22:39, 29.0kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:40:12, 20.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:21:23, 29.0kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:35:38, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:18:09, 29.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:39:06, 20.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:20:40, 28.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:14:04, 21.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:03:07, 30.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:25:39, 20.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:11:12, 29.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:23:41, 20.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<5:09:48, 29.5kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:28:24, 20.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<5:13:11, 29.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:06:42, 21.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<4:57:57, 30.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:16:58, 20.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<5:05:06, 29.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:22:29, 20.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<5:08:59, 29.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:14:55, 20.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:03:40, 29.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:23:44, 20.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<5:09:50, 28.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:14:18, 20.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<5:03:14, 29.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<7:16:24, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<5:04:42, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<7:09:00, 20.7kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<4:59:31, 29.5kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<7:14:41, 20.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<5:03:29, 29.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<7:07:46, 20.6kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<4:58:39, 29.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<7:06:29, 20.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<4:57:44, 29.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<7:07:34, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<4:58:30, 29.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<7:05:45, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<4:57:15, 29.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<6:56:15, 20.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<4:50:35, 29.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<7:02:44, 20.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<4:55:09, 29.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<6:53:16, 20.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:48:30, 29.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<6:58:05, 20.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<4:51:53, 29.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<6:50:14, 20.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<4:46:26, 29.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<6:38:59, 21.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<4:38:32, 30.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:44:49, 20.9kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<4:42:36, 29.8kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<6:45:02, 20.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<4:42:44, 29.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:44:16, 20.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<4:42:12, 29.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<6:43:42, 20.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<4:41:48, 29.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<6:39:21, 20.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<4:38:45, 29.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<6:44:48, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<4:42:34, 29.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<6:35:16, 20.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<4:35:53, 29.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<6:41:50, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<4:40:29, 29.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<6:32:46, 20.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<4:34:11, 29.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<6:22:32, 21.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<4:27:00, 30.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:27:38, 20.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<4:30:32, 29.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<6:32:04, 20.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<4:33:37, 29.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<6:33:54, 20.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<4:34:56, 29.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<6:25:43, 20.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<4:29:11, 29.6kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<6:29:36, 20.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<4:31:45, 29.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<3:13:49, 40.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<5:19:37, 24.8kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<3:43:06, 35.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<5:54:28, 22.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<4:07:24, 31.7kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<6:07:32, 21.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<4:16:29, 30.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<6:15:05, 20.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<4:21:46, 29.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<6:10:01, 21.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<4:18:12, 30.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<6:15:42, 20.6kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<4:22:17, 29.4kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<3:04:46, 41.8kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<4:47:32, 26.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<3:20:42, 38.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<5:34:03, 23.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<3:53:18, 32.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<2:44:03, 46.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<4:40:51, 27.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<3:16:03, 38.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<5:26:05, 23.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<3:47:27, 33.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<2:41:39, 46.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<4:48:49, 26.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<3:21:38, 37.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<5:22:16, 23.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<3:44:45, 33.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<2:41:08, 46.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<4:32:49, 27.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<3:10:25, 39.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<5:22:38, 23.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<3:45:16, 33.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<2:38:23, 46.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<4:43:44, 26.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<3:18:00, 37.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<5:26:52, 22.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<3:48:02, 32.3kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<2:41:09, 45.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<4:46:11, 25.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<3:19:42, 36.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<5:27:17, 22.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<3:48:11, 32.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<2:44:31, 44.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<4:46:47, 25.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<3:20:06, 36.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<5:25:40, 22.3kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<3:47:17, 31.8kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:18:05, 22.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<3:41:54, 32.4kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<5:38:27, 21.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<3:56:03, 30.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<2:47:03, 42.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<4:42:27, 25.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<3:17:04, 36.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<5:20:09, 22.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<3:43:29, 31.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<2:37:07, 45.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<4:30:48, 26.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<3:08:57, 37.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:11:13, 22.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<3:37:18, 32.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<2:32:39, 46.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<4:28:01, 26.2kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<3:06:59, 37.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:08:09, 22.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<3:34:58, 32.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:16:08, 22.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<3:40:29, 31.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:31:06, 20.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<3:50:58, 29.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:17:43, 21.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:41:34, 30.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<5:29:12, 20.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<3:49:36, 29.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<5:25:36, 20.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<3:47:03, 29.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<5:31:12, 20.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<3:51:01, 29.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<5:14:35, 21.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<3:39:24, 30.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:11:35, 21.5kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<3:37:19, 30.7kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<5:10:46, 21.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<3:36:41, 30.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<5:26:11, 20.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<3:47:27, 29.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<5:19:23, 20.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<3:42:41, 29.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<5:22:15, 20.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<3:44:41, 29.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<5:18:13, 20.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<3:41:51, 29.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<5:20:07, 20.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<3:43:11, 28.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<5:13:42, 20.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<3:38:41, 29.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<5:16:14, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<3:40:28, 29.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<5:10:16, 20.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<3:36:17, 29.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<5:12:51, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<3:38:06, 29.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<5:07:50, 20.5kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<3:34:34, 29.3kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<5:09:54, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<3:36:01, 28.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<5:04:46, 20.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<3:32:25, 29.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<5:06:07, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<3:33:22, 28.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<5:01:29, 20.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<3:30:07, 29.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<5:02:48, 20.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<3:31:04, 28.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<4:53:41, 20.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<3:24:40, 29.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<4:57:58, 20.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<3:27:43, 29.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<4:42:08, 21.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<3:16:37, 30.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<4:50:42, 20.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<3:22:35, 29.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<4:48:11, 20.7kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<3:20:48, 29.5kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<5:00:00, 19.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<3:29:06, 28.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<4:39:17, 21.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<3:14:36, 30.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<4:44:24, 20.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<3:18:10, 29.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<4:42:44, 20.6kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<3:16:59, 29.4kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<4:44:23, 20.4kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<3:18:08, 29.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<4:39:25, 20.6kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<3:14:40, 29.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<4:41:49, 20.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<3:16:23, 29.0kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<4:25:59, 21.4kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<3:05:20, 30.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<4:23:24, 21.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<3:03:30, 30.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:29:34, 20.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<3:07:47, 29.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<4:28:03, 20.8kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<3:06:43, 29.7kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<4:30:55, 20.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<3:08:45, 29.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<4:18:10, 21.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<2:59:51, 30.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<4:15:54, 21.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<2:58:14, 30.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<4:22:07, 20.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<3:02:33, 29.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<4:20:37, 20.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<3:01:29, 29.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<4:22:26, 20.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<3:02:46, 29.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<4:18:56, 20.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<3:00:19, 29.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<4:19:30, 20.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<3:00:41, 29.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<4:20:37, 20.2kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<3:01:28, 28.8kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<4:14:51, 20.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<2:57:27, 29.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<4:13:35, 20.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<2:56:32, 29.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<4:14:14, 20.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<2:57:06, 29.0kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<2:04:46, 41.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<3:24:42, 25.0kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<2:22:32, 35.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<3:48:06, 22.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<2:38:51, 31.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<3:46:03, 22.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<2:37:25, 31.9kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<3:46:35, 22.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<2:37:44, 31.6kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<3:55:56, 21.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<2:44:14, 30.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<3:55:16, 21.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<2:43:44, 30.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<3:59:33, 20.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<2:46:44, 29.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<3:55:22, 20.7kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:08<2:43:48, 29.6kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<3:57:33, 20.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:09<2:45:19, 29.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<3:52:59, 20.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<2:42:07, 29.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<3:53:38, 20.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<2:42:36, 29.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<3:46:11, 21.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<2:37:23, 29.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<3:47:27, 20.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<2:38:14, 29.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<3:49:13, 20.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<2:39:30, 29.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<3:35:48, 21.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<2:30:07, 30.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<3:42:01, 20.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<2:34:28, 29.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<3:36:52, 21.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<2:30:51, 30.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<3:41:18, 20.5kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<2:34:01, 29.2kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<1:48:37, 41.4kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<3:00:16, 25.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<2:05:25, 35.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<3:20:10, 22.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<2:19:15, 31.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<1:38:30, 45.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<2:51:29, 25.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<1:59:18, 36.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<3:14:15, 22.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<2:15:06, 32.3kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<3:18:09, 22.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<2:17:48, 31.4kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<3:25:30, 21.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<2:22:56, 30.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<3:15:11, 22.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<2:15:42, 31.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<3:22:16, 21.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<2:20:38, 30.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<3:20:02, 21.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<2:19:03, 30.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<3:23:38, 20.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<2:21:33, 29.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<3:20:07, 20.7kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<2:19:05, 29.6kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<3:21:50, 20.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<2:20:17, 29.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<3:16:00, 20.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<2:16:12, 29.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<3:18:18, 20.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<2:17:51, 29.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<3:05:43, 21.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<2:09:03, 30.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<3:10:07, 20.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<2:12:05, 29.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<3:10:13, 20.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<2:12:09, 29.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<3:11:27, 20.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<2:13:00, 29.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<3:07:49, 20.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<2:10:44, 29.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<1:31:47, 41.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<2:29:23, 25.7kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<1:43:48, 36.6kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<2:47:05, 22.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<1:56:03, 32.4kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<2:55:21, 21.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<2:01:47, 30.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<2:59:40, 20.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<2:05:01, 29.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<1:27:43, 42.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<2:27:18, 25.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<1:42:18, 35.8kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<2:43:40, 22.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<1:53:59, 31.9kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<1:19:52, 45.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<2:17:33, 26.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<1:35:31, 37.6kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<2:37:51, 22.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<1:49:54, 32.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<1:17:03, 46.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<2:11:59, 26.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<1:31:38, 38.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<2:31:20, 23.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:45:01, 33.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:15:36, 46.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<2:06:58, 27.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<1:28:08, 39.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<2:28:40, 23.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<1:43:33, 33.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<1:12:31, 47.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<2:06:35, 27.0kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<1:27:53, 38.5kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<2:20:33, 24.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<1:37:45, 34.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<1:08:38, 48.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<2:03:49, 27.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<1:25:57, 38.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<2:16:05, 24.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<1:34:27, 34.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<1:06:51, 49.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<2:02:54, 26.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<1:25:16, 38.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<2:19:31, 23.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<1:36:46, 33.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<2:24:08, 22.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<1:39:58, 31.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<2:22:48, 22.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<1:39:31, 31.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<1:09:32, 45.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<1:56:40, 26.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<1:20:51, 38.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<59:26, 52.2kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<1:49:37, 28.3kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<1:15:56, 40.4kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<57:14, 53.6kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<1:51:50, 27.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<1:17:33, 39.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<2:04:20, 24.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:26:26, 34.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:00:37, 49.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:50:23, 27.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<1:16:31, 38.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<2:00:55, 24.5kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<1:24:00, 35.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<58:58, 49.6kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<1:47:42, 27.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<1:14:38, 38.7kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<2:03:16, 23.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<1:25:23, 33.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<2:07:59, 22.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<1:28:37, 31.8kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<2:11:29, 21.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:31:01, 30.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:05:09, 42.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:51:21, 25.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<1:17:06, 35.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<2:02:28, 22.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<1:24:47, 32.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<2:01:05, 22.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<1:23:47, 32.0kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<2:05:34, 21.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<1:26:53, 30.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<2:06:01, 21.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<1:27:11, 29.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<2:04:17, 21.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<1:25:58, 30.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<2:05:53, 20.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<1:27:08, 29.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<1:01:32, 41.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<2:00:15, 21.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<1:23:05, 30.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<1:49:45, 22.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<1:15:52, 32.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<1:55:14, 21.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<1:19:38, 30.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<1:55:54, 21.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<1:20:06, 30.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:52:15, 21.4kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<1:17:33, 30.5kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<1:52:45, 21.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<1:17:52, 29.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<1:52:07, 20.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<1:17:25, 29.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<1:50:59, 20.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<1:16:36, 29.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:50:47, 20.4kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<1:16:28, 29.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<1:47:49, 20.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:24<1:14:24, 29.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<1:46:31, 20.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<1:13:28, 29.4kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<1:45:31, 20.4kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<1:12:50, 29.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<51:34, 41.1kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<1:26:21, 24.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<59:32, 35.0kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<1:33:29, 22.3kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<1:04:26, 31.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<1:36:07, 21.3kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<1:06:14, 30.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<1:37:24, 20.7kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<1:07:06, 29.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<1:35:07, 20.8kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<1:05:30, 29.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<1:34:20, 20.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<1:04:56, 29.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<1:33:34, 20.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<1:04:23, 29.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<1:31:48, 20.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<1:03:09, 29.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<1:30:34, 20.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<1:02:17, 29.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<1:28:17, 20.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<1:00:41, 29.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<1:27:23, 20.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<1:00:03, 28.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<1:24:13, 20.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<57:51, 29.4kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<1:22:40, 20.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<56:45, 29.4kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<1:22:11, 20.3kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<56:24, 28.9kB/s]  .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<1:19:18, 20.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<54:24, 29.4kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<1:17:34, 20.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<53:11, 29.4kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<1:14:09, 21.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<50:49, 30.1kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<1:13:29, 20.8kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<50:20, 29.7kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<1:12:55, 20.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<49:55, 29.2kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<1:10:29, 20.7kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<48:14, 29.5kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<1:09:12, 20.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<47:19, 29.3kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<1:08:23, 20.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<46:44, 28.9kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<1:05:23, 20.7kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<44:39, 29.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<1:03:56, 20.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<43:38, 29.4kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<1:03:08, 20.3kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<43:03, 29.0kB/s]  .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<1:00:29, 20.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<41:12, 29.4kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<58:34, 20.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<39:52, 29.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<56:59, 20.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<38:46, 29.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<55:03, 20.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<37:25, 29.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<53:23, 20.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<36:14, 29.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<51:40, 20.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<35:02, 29.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<49:55, 20.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<33:49, 29.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<48:21, 20.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<32:43, 29.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<47:04, 20.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:00<31:49, 29.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<44:57, 20.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<30:20, 29.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<43:17, 20.7kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<29:10, 29.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<39:37, 21.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<26:39, 31.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<39:04, 21.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<26:14, 30.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<37:47, 21.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<25:19, 29.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<35:05, 21.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<23:28, 30.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<32:45, 22.1kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<21:51, 31.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<32:10, 21.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<21:24, 30.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<30:48, 21.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<20:27, 30.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<28:59, 21.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<19:10, 30.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<27:39, 21.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<18:14, 30.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<25:48, 21.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<16:56, 30.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<24:23, 21.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<15:56, 30.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<22:42, 21.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<14:45, 30.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<21:08, 21.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<13:39, 29.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<19:38, 20.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<12:35, 29.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<17:55, 20.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<11:23, 29.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<16:09, 21.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<10:09, 29.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<14:30, 21.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<09:00, 29.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<12:53, 20.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<07:52, 29.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<11:14, 20.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<06:44, 29.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<04:55, 40.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<08:07, 24.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<04:42, 35.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<07:19, 22.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<04:08, 32.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<02:53, 44.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<24:41, 5.24kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<16:49, 7.48kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<10:15, 10.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<05:52, 15.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<03:08, 21.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<02:00, 30.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<01:20, 42.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:41, 61.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:18, 86.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:05, 124kB/s] .vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 742/400000 [00:00<00:53, 7410.24it/s]  0%|          | 1545/400000 [00:00<00:52, 7583.88it/s]  1%|          | 2346/400000 [00:00<00:51, 7704.69it/s]  1%|          | 3153/400000 [00:00<00:50, 7810.28it/s]  1%|          | 3939/400000 [00:00<00:50, 7825.12it/s]  1%|          | 4697/400000 [00:00<00:51, 7748.31it/s]  1%|▏         | 5511/400000 [00:00<00:50, 7861.21it/s]  2%|▏         | 6304/400000 [00:00<00:49, 7880.50it/s]  2%|▏         | 7085/400000 [00:00<00:50, 7857.16it/s]  2%|▏         | 7893/400000 [00:01<00:49, 7922.57it/s]  2%|▏         | 8671/400000 [00:01<00:49, 7876.53it/s]  2%|▏         | 9445/400000 [00:01<00:49, 7821.45it/s]  3%|▎         | 10225/400000 [00:01<00:49, 7813.58it/s]  3%|▎         | 11000/400000 [00:01<00:50, 7709.83it/s]  3%|▎         | 11784/400000 [00:01<00:50, 7748.03it/s]  3%|▎         | 12556/400000 [00:01<00:50, 7710.07it/s]  3%|▎         | 13340/400000 [00:01<00:49, 7747.25it/s]  4%|▎         | 14133/400000 [00:01<00:49, 7800.45it/s]  4%|▎         | 14922/400000 [00:01<00:49, 7826.33it/s]  4%|▍         | 15723/400000 [00:02<00:48, 7879.83it/s]  4%|▍         | 16515/400000 [00:02<00:48, 7891.32it/s]  4%|▍         | 17304/400000 [00:02<00:48, 7837.12it/s]  5%|▍         | 18099/400000 [00:02<00:48, 7870.21it/s]  5%|▍         | 18886/400000 [00:02<00:48, 7827.96it/s]  5%|▍         | 19669/400000 [00:02<00:49, 7737.10it/s]  5%|▌         | 20443/400000 [00:02<00:49, 7691.87it/s]  5%|▌         | 21213/400000 [00:02<00:49, 7680.08it/s]  5%|▌         | 21982/400000 [00:02<00:49, 7656.03it/s]  6%|▌         | 22770/400000 [00:02<00:48, 7721.40it/s]  6%|▌         | 23569/400000 [00:03<00:48, 7799.86it/s]  6%|▌         | 24350/400000 [00:03<00:49, 7654.86it/s]  6%|▋         | 25117/400000 [00:03<00:49, 7536.67it/s]  6%|▋         | 25888/400000 [00:03<00:49, 7584.80it/s]  7%|▋         | 26687/400000 [00:03<00:48, 7700.89it/s]  7%|▋         | 27459/400000 [00:03<00:48, 7690.46it/s]  7%|▋         | 28229/400000 [00:03<00:49, 7528.15it/s]  7%|▋         | 28984/400000 [00:03<00:49, 7504.21it/s]  7%|▋         | 29775/400000 [00:03<00:48, 7619.11it/s]  8%|▊         | 30554/400000 [00:03<00:48, 7668.08it/s]  8%|▊         | 31322/400000 [00:04<00:48, 7634.85it/s]  8%|▊         | 32087/400000 [00:04<00:48, 7536.37it/s]  8%|▊         | 32867/400000 [00:04<00:48, 7612.29it/s]  8%|▊         | 33629/400000 [00:04<00:48, 7553.55it/s]  9%|▊         | 34398/400000 [00:04<00:48, 7593.23it/s]  9%|▉         | 35169/400000 [00:04<00:47, 7625.61it/s]  9%|▉         | 35935/400000 [00:04<00:47, 7634.41it/s]  9%|▉         | 36705/400000 [00:04<00:47, 7651.61it/s]  9%|▉         | 37471/400000 [00:04<00:48, 7497.77it/s] 10%|▉         | 38222/400000 [00:04<00:48, 7401.35it/s] 10%|▉         | 39021/400000 [00:05<00:47, 7568.12it/s] 10%|▉         | 39822/400000 [00:05<00:46, 7694.47it/s] 10%|█         | 40608/400000 [00:05<00:46, 7741.43it/s] 10%|█         | 41401/400000 [00:05<00:46, 7794.65it/s] 11%|█         | 42201/400000 [00:05<00:45, 7854.96it/s] 11%|█         | 42988/400000 [00:05<00:46, 7729.14it/s] 11%|█         | 43762/400000 [00:05<00:46, 7695.16it/s] 11%|█         | 44533/400000 [00:05<00:46, 7615.45it/s] 11%|█▏        | 45296/400000 [00:05<00:48, 7344.38it/s] 12%|█▏        | 46089/400000 [00:05<00:47, 7510.41it/s] 12%|█▏        | 46891/400000 [00:06<00:46, 7655.52it/s] 12%|█▏        | 47660/400000 [00:06<00:46, 7653.66it/s] 12%|█▏        | 48461/400000 [00:06<00:45, 7755.88it/s] 12%|█▏        | 49265/400000 [00:06<00:44, 7838.11it/s] 13%|█▎        | 50073/400000 [00:06<00:44, 7908.09it/s] 13%|█▎        | 50874/400000 [00:06<00:43, 7937.22it/s] 13%|█▎        | 51669/400000 [00:06<00:44, 7856.03it/s] 13%|█▎        | 52456/400000 [00:06<00:44, 7791.56it/s] 13%|█▎        | 53255/400000 [00:06<00:44, 7847.78it/s] 14%|█▎        | 54060/400000 [00:06<00:43, 7905.95it/s] 14%|█▎        | 54852/400000 [00:07<00:43, 7885.68it/s] 14%|█▍        | 55641/400000 [00:07<00:44, 7787.42it/s] 14%|█▍        | 56421/400000 [00:07<00:44, 7777.68it/s] 14%|█▍        | 57200/400000 [00:07<00:44, 7744.00it/s] 14%|█▍        | 57980/400000 [00:07<00:44, 7758.41it/s] 15%|█▍        | 58757/400000 [00:07<00:44, 7713.46it/s] 15%|█▍        | 59529/400000 [00:07<00:45, 7515.84it/s] 15%|█▌        | 60282/400000 [00:07<00:45, 7388.74it/s] 15%|█▌        | 61067/400000 [00:07<00:45, 7519.22it/s] 15%|█▌        | 61872/400000 [00:08<00:44, 7669.18it/s] 16%|█▌        | 62685/400000 [00:08<00:43, 7799.00it/s] 16%|█▌        | 63467/400000 [00:08<00:43, 7688.80it/s] 16%|█▌        | 64253/400000 [00:08<00:43, 7737.35it/s] 16%|█▋        | 65051/400000 [00:08<00:42, 7807.86it/s] 16%|█▋        | 65841/400000 [00:08<00:42, 7832.73it/s] 17%|█▋        | 66625/400000 [00:08<00:45, 7358.87it/s] 17%|█▋        | 67369/400000 [00:08<00:45, 7381.08it/s] 17%|█▋        | 68132/400000 [00:08<00:44, 7452.51it/s] 17%|█▋        | 68940/400000 [00:08<00:43, 7628.07it/s] 17%|█▋        | 69707/400000 [00:09<00:43, 7638.86it/s] 18%|█▊        | 70505/400000 [00:09<00:42, 7737.93it/s] 18%|█▊        | 71281/400000 [00:09<00:43, 7596.43it/s] 18%|█▊        | 72096/400000 [00:09<00:42, 7753.98it/s] 18%|█▊        | 72889/400000 [00:09<00:41, 7803.66it/s] 18%|█▊        | 73671/400000 [00:09<00:41, 7790.41it/s] 19%|█▊        | 74468/400000 [00:09<00:41, 7842.80it/s] 19%|█▉        | 75254/400000 [00:09<00:42, 7711.37it/s] 19%|█▉        | 76027/400000 [00:09<00:42, 7631.38it/s] 19%|█▉        | 76792/400000 [00:09<00:42, 7594.86it/s] 19%|█▉        | 77553/400000 [00:10<00:44, 7249.61it/s] 20%|█▉        | 78345/400000 [00:10<00:43, 7436.47it/s] 20%|█▉        | 79127/400000 [00:10<00:42, 7546.12it/s] 20%|█▉        | 79916/400000 [00:10<00:41, 7644.96it/s] 20%|██        | 80683/400000 [00:10<00:41, 7637.69it/s] 20%|██        | 81449/400000 [00:10<00:41, 7599.56it/s] 21%|██        | 82211/400000 [00:10<00:41, 7590.91it/s] 21%|██        | 82971/400000 [00:10<00:41, 7554.11it/s] 21%|██        | 83756/400000 [00:10<00:41, 7638.32it/s] 21%|██        | 84539/400000 [00:10<00:41, 7693.62it/s] 21%|██▏       | 85309/400000 [00:11<00:40, 7692.74it/s] 22%|██▏       | 86081/400000 [00:11<00:40, 7698.86it/s] 22%|██▏       | 86852/400000 [00:11<00:41, 7608.56it/s] 22%|██▏       | 87614/400000 [00:11<00:41, 7539.77it/s] 22%|██▏       | 88369/400000 [00:11<00:41, 7506.44it/s] 22%|██▏       | 89142/400000 [00:11<00:41, 7571.51it/s] 22%|██▏       | 89911/400000 [00:11<00:40, 7603.61it/s] 23%|██▎       | 90672/400000 [00:11<00:41, 7508.42it/s] 23%|██▎       | 91424/400000 [00:11<00:41, 7502.37it/s] 23%|██▎       | 92175/400000 [00:12<00:41, 7460.65it/s] 23%|██▎       | 92928/400000 [00:12<00:41, 7480.38it/s] 23%|██▎       | 93677/400000 [00:12<00:40, 7481.76it/s] 24%|██▎       | 94426/400000 [00:12<00:41, 7334.75it/s] 24%|██▍       | 95175/400000 [00:12<00:41, 7379.19it/s] 24%|██▍       | 95982/400000 [00:12<00:40, 7571.25it/s] 24%|██▍       | 96782/400000 [00:12<00:39, 7693.81it/s] 24%|██▍       | 97582/400000 [00:12<00:38, 7780.22it/s] 25%|██▍       | 98362/400000 [00:12<00:39, 7542.54it/s] 25%|██▍       | 99144/400000 [00:12<00:39, 7621.45it/s] 25%|██▍       | 99941/400000 [00:13<00:38, 7722.21it/s] 25%|██▌       | 100754/400000 [00:13<00:38, 7837.74it/s] 25%|██▌       | 101561/400000 [00:13<00:37, 7905.82it/s] 26%|██▌       | 102353/400000 [00:13<00:37, 7857.39it/s] 26%|██▌       | 103140/400000 [00:13<00:37, 7843.31it/s] 26%|██▌       | 103951/400000 [00:13<00:37, 7918.70it/s] 26%|██▌       | 104744/400000 [00:13<00:37, 7871.64it/s] 26%|██▋       | 105532/400000 [00:13<00:37, 7791.04it/s] 27%|██▋       | 106312/400000 [00:13<00:38, 7601.67it/s] 27%|██▋       | 107090/400000 [00:13<00:38, 7653.58it/s] 27%|██▋       | 107861/400000 [00:14<00:38, 7667.68it/s] 27%|██▋       | 108631/400000 [00:14<00:37, 7675.76it/s] 27%|██▋       | 109444/400000 [00:14<00:37, 7803.87it/s] 28%|██▊       | 110226/400000 [00:14<00:37, 7707.90it/s] 28%|██▊       | 111012/400000 [00:14<00:37, 7752.46it/s] 28%|██▊       | 111788/400000 [00:14<00:37, 7587.09it/s] 28%|██▊       | 112548/400000 [00:14<00:38, 7556.05it/s] 28%|██▊       | 113305/400000 [00:14<00:38, 7525.76it/s] 29%|██▊       | 114063/400000 [00:14<00:37, 7540.26it/s] 29%|██▊       | 114818/400000 [00:14<00:38, 7481.19it/s] 29%|██▉       | 115633/400000 [00:15<00:37, 7669.51it/s] 29%|██▉       | 116409/400000 [00:15<00:36, 7695.45it/s] 29%|██▉       | 117180/400000 [00:15<00:37, 7635.31it/s] 29%|██▉       | 117984/400000 [00:15<00:36, 7752.03it/s] 30%|██▉       | 118761/400000 [00:15<00:36, 7731.71it/s] 30%|██▉       | 119578/400000 [00:15<00:35, 7855.28it/s] 30%|███       | 120372/400000 [00:15<00:35, 7879.21it/s] 30%|███       | 121172/400000 [00:15<00:35, 7912.97it/s] 30%|███       | 121964/400000 [00:15<00:35, 7742.19it/s] 31%|███       | 122755/400000 [00:15<00:35, 7789.97it/s] 31%|███       | 123535/400000 [00:16<00:35, 7740.48it/s] 31%|███       | 124345/400000 [00:16<00:35, 7841.40it/s] 31%|███▏      | 125157/400000 [00:16<00:34, 7919.92it/s] 31%|███▏      | 125950/400000 [00:16<00:34, 7907.42it/s] 32%|███▏      | 126749/400000 [00:16<00:34, 7929.69it/s] 32%|███▏      | 127570/400000 [00:16<00:34, 8009.25it/s] 32%|███▏      | 128387/400000 [00:16<00:33, 8056.47it/s] 32%|███▏      | 129194/400000 [00:16<00:34, 7855.57it/s] 32%|███▏      | 129981/400000 [00:16<00:35, 7561.47it/s] 33%|███▎      | 130741/400000 [00:16<00:35, 7510.20it/s] 33%|███▎      | 131526/400000 [00:17<00:35, 7607.74it/s] 33%|███▎      | 132317/400000 [00:17<00:34, 7695.69it/s] 33%|███▎      | 133120/400000 [00:17<00:34, 7792.55it/s] 33%|███▎      | 133901/400000 [00:17<00:34, 7723.66it/s] 34%|███▎      | 134702/400000 [00:17<00:33, 7805.60it/s] 34%|███▍      | 135484/400000 [00:17<00:33, 7805.29it/s] 34%|███▍      | 136301/400000 [00:17<00:33, 7911.18it/s] 34%|███▍      | 137096/400000 [00:17<00:33, 7921.16it/s] 34%|███▍      | 137889/400000 [00:17<00:34, 7660.96it/s] 35%|███▍      | 138670/400000 [00:18<00:33, 7704.65it/s] 35%|███▍      | 139467/400000 [00:18<00:33, 7782.01it/s] 35%|███▌      | 140285/400000 [00:18<00:32, 7895.17it/s] 35%|███▌      | 141076/400000 [00:18<00:33, 7804.97it/s] 35%|███▌      | 141858/400000 [00:18<00:33, 7785.85it/s] 36%|███▌      | 142638/400000 [00:18<00:33, 7772.74it/s] 36%|███▌      | 143455/400000 [00:18<00:32, 7887.74it/s] 36%|███▌      | 144276/400000 [00:18<00:32, 7980.94it/s] 36%|███▋      | 145075/400000 [00:18<00:32, 7946.61it/s] 36%|███▋      | 145871/400000 [00:18<00:32, 7900.55it/s] 37%|███▋      | 146662/400000 [00:19<00:32, 7828.18it/s] 37%|███▋      | 147465/400000 [00:19<00:32, 7884.99it/s] 37%|███▋      | 148278/400000 [00:19<00:31, 7956.73it/s] 37%|███▋      | 149075/400000 [00:19<00:31, 7924.64it/s] 37%|███▋      | 149869/400000 [00:19<00:31, 7927.05it/s] 38%|███▊      | 150662/400000 [00:19<00:32, 7748.93it/s] 38%|███▊      | 151471/400000 [00:19<00:31, 7846.19it/s] 38%|███▊      | 152281/400000 [00:19<00:31, 7920.37it/s] 38%|███▊      | 153074/400000 [00:19<00:31, 7748.46it/s] 38%|███▊      | 153851/400000 [00:19<00:31, 7703.38it/s] 39%|███▊      | 154623/400000 [00:20<00:32, 7574.27it/s] 39%|███▉      | 155432/400000 [00:20<00:31, 7719.77it/s] 39%|███▉      | 156231/400000 [00:20<00:31, 7797.57it/s] 39%|███▉      | 157013/400000 [00:20<00:31, 7771.75it/s] 39%|███▉      | 157792/400000 [00:20<00:31, 7740.48it/s] 40%|███▉      | 158567/400000 [00:20<00:31, 7696.94it/s] 40%|███▉      | 159399/400000 [00:20<00:30, 7873.79it/s] 40%|████      | 160211/400000 [00:20<00:30, 7944.47it/s] 40%|████      | 161007/400000 [00:20<00:30, 7929.51it/s] 40%|████      | 161801/400000 [00:20<00:30, 7869.12it/s] 41%|████      | 162589/400000 [00:21<00:30, 7822.55it/s] 41%|████      | 163372/400000 [00:21<00:30, 7710.46it/s] 41%|████      | 164177/400000 [00:21<00:30, 7809.04it/s] 41%|████      | 164965/400000 [00:21<00:30, 7829.99it/s] 41%|████▏     | 165749/400000 [00:21<00:31, 7434.01it/s] 42%|████▏     | 166498/400000 [00:21<00:31, 7450.62it/s] 42%|████▏     | 167303/400000 [00:21<00:30, 7620.55it/s] 42%|████▏     | 168069/400000 [00:21<00:30, 7608.70it/s] 42%|████▏     | 168855/400000 [00:21<00:30, 7681.95it/s] 42%|████▏     | 169660/400000 [00:21<00:29, 7786.35it/s] 43%|████▎     | 170441/400000 [00:22<00:29, 7750.59it/s] 43%|████▎     | 171218/400000 [00:22<00:29, 7720.25it/s] 43%|████▎     | 172034/400000 [00:22<00:29, 7847.07it/s] 43%|████▎     | 172840/400000 [00:22<00:28, 7906.87it/s] 43%|████▎     | 173658/400000 [00:22<00:28, 7984.50it/s] 44%|████▎     | 174458/400000 [00:22<00:28, 7854.69it/s] 44%|████▍     | 175260/400000 [00:22<00:28, 7902.67it/s] 44%|████▍     | 176052/400000 [00:22<00:28, 7808.03it/s] 44%|████▍     | 176834/400000 [00:22<00:28, 7747.75it/s] 44%|████▍     | 177644/400000 [00:22<00:28, 7848.00it/s] 45%|████▍     | 178430/400000 [00:23<00:28, 7640.81it/s] 45%|████▍     | 179218/400000 [00:23<00:28, 7708.94it/s] 45%|████▌     | 180033/400000 [00:23<00:28, 7833.90it/s] 45%|████▌     | 180827/400000 [00:23<00:27, 7863.28it/s] 45%|████▌     | 181643/400000 [00:23<00:27, 7948.84it/s] 46%|████▌     | 182439/400000 [00:23<00:27, 7783.93it/s] 46%|████▌     | 183219/400000 [00:23<00:28, 7645.15it/s] 46%|████▌     | 183986/400000 [00:23<00:28, 7581.74it/s] 46%|████▌     | 184748/400000 [00:23<00:28, 7590.77it/s] 46%|████▋     | 185539/400000 [00:24<00:27, 7682.74it/s] 47%|████▋     | 186309/400000 [00:24<00:27, 7680.12it/s] 47%|████▋     | 187116/400000 [00:24<00:27, 7792.91it/s] 47%|████▋     | 187917/400000 [00:24<00:26, 7856.70it/s] 47%|████▋     | 188710/400000 [00:24<00:26, 7878.13it/s] 47%|████▋     | 189518/400000 [00:24<00:26, 7936.08it/s] 48%|████▊     | 190313/400000 [00:24<00:26, 7770.89it/s] 48%|████▊     | 191092/400000 [00:24<00:27, 7710.11it/s] 48%|████▊     | 191864/400000 [00:24<00:27, 7519.77it/s] 48%|████▊     | 192618/400000 [00:24<00:28, 7212.79it/s] 48%|████▊     | 193362/400000 [00:25<00:28, 7276.17it/s] 49%|████▊     | 194130/400000 [00:25<00:27, 7390.52it/s] 49%|████▊     | 194872/400000 [00:25<00:27, 7396.15it/s] 49%|████▉     | 195651/400000 [00:25<00:27, 7507.96it/s] 49%|████▉     | 196405/400000 [00:25<00:27, 7516.49it/s] 49%|████▉     | 197158/400000 [00:25<00:27, 7374.69it/s] 49%|████▉     | 197897/400000 [00:25<00:27, 7254.76it/s] 50%|████▉     | 198624/400000 [00:25<00:28, 7159.44it/s] 50%|████▉     | 199342/400000 [00:25<00:28, 7125.56it/s] 50%|█████     | 200075/400000 [00:25<00:27, 7184.45it/s] 50%|█████     | 200804/400000 [00:26<00:27, 7213.92it/s] 50%|█████     | 201526/400000 [00:26<00:27, 7187.29it/s] 51%|█████     | 202246/400000 [00:26<00:27, 7190.22it/s] 51%|█████     | 202983/400000 [00:26<00:27, 7240.57it/s] 51%|█████     | 203710/400000 [00:26<00:27, 7247.84it/s] 51%|█████     | 204442/400000 [00:26<00:26, 7268.15it/s] 51%|█████▏    | 205191/400000 [00:26<00:26, 7333.28it/s] 51%|█████▏    | 205925/400000 [00:26<00:26, 7305.94it/s] 52%|█████▏    | 206656/400000 [00:26<00:27, 7159.80it/s] 52%|█████▏    | 207402/400000 [00:26<00:26, 7245.48it/s] 52%|█████▏    | 208193/400000 [00:27<00:25, 7431.60it/s] 52%|█████▏    | 208957/400000 [00:27<00:25, 7492.60it/s] 52%|█████▏    | 209740/400000 [00:27<00:25, 7588.50it/s] 53%|█████▎    | 210516/400000 [00:27<00:24, 7638.94it/s] 53%|█████▎    | 211281/400000 [00:27<00:25, 7500.24it/s] 53%|█████▎    | 212039/400000 [00:27<00:24, 7518.71it/s] 53%|█████▎    | 212792/400000 [00:27<00:25, 7462.67it/s] 53%|█████▎    | 213571/400000 [00:27<00:24, 7556.18it/s] 54%|█████▎    | 214328/400000 [00:27<00:24, 7463.21it/s] 54%|█████▍    | 215108/400000 [00:28<00:24, 7560.25it/s] 54%|█████▍    | 215865/400000 [00:28<00:24, 7536.09it/s] 54%|█████▍    | 216635/400000 [00:28<00:24, 7582.54it/s] 54%|█████▍    | 217447/400000 [00:28<00:23, 7734.39it/s] 55%|█████▍    | 218249/400000 [00:28<00:23, 7816.80it/s] 55%|█████▍    | 219061/400000 [00:28<00:22, 7902.83it/s] 55%|█████▍    | 219865/400000 [00:28<00:22, 7943.13it/s] 55%|█████▌    | 220660/400000 [00:28<00:22, 7931.42it/s] 55%|█████▌    | 221454/400000 [00:28<00:22, 7815.63it/s] 56%|█████▌    | 222241/400000 [00:28<00:22, 7828.56it/s] 56%|█████▌    | 223025/400000 [00:29<00:22, 7785.03it/s] 56%|█████▌    | 223804/400000 [00:29<00:22, 7766.61it/s] 56%|█████▌    | 224581/400000 [00:29<00:22, 7746.50it/s] 56%|█████▋    | 225401/400000 [00:29<00:22, 7875.28it/s] 57%|█████▋    | 226212/400000 [00:29<00:21, 7941.51it/s] 57%|█████▋    | 227019/400000 [00:29<00:21, 7979.03it/s] 57%|█████▋    | 227818/400000 [00:29<00:21, 7955.49it/s] 57%|█████▋    | 228614/400000 [00:29<00:21, 7815.39it/s] 57%|█████▋    | 229397/400000 [00:29<00:21, 7796.00it/s] 58%|█████▊    | 230191/400000 [00:29<00:21, 7838.34it/s] 58%|█████▊    | 230999/400000 [00:30<00:21, 7908.79it/s] 58%|█████▊    | 231791/400000 [00:30<00:21, 7785.06it/s] 58%|█████▊    | 232585/400000 [00:30<00:21, 7828.02it/s] 58%|█████▊    | 233400/400000 [00:30<00:21, 7920.51it/s] 59%|█████▊    | 234224/400000 [00:30<00:20, 8012.33it/s] 59%|█████▉    | 235041/400000 [00:30<00:20, 8058.07it/s] 59%|█████▉    | 235848/400000 [00:30<00:20, 7972.75it/s] 59%|█████▉    | 236646/400000 [00:30<00:21, 7718.01it/s] 59%|█████▉    | 237441/400000 [00:30<00:20, 7784.96it/s] 60%|█████▉    | 238255/400000 [00:30<00:20, 7887.03it/s] 60%|█████▉    | 239046/400000 [00:31<00:20, 7855.47it/s] 60%|█████▉    | 239847/400000 [00:31<00:20, 7900.95it/s] 60%|██████    | 240638/400000 [00:31<00:20, 7854.62it/s] 60%|██████    | 241436/400000 [00:31<00:20, 7889.99it/s] 61%|██████    | 242234/400000 [00:31<00:19, 7916.27it/s] 61%|██████    | 243034/400000 [00:31<00:19, 7937.08it/s] 61%|██████    | 243836/400000 [00:31<00:19, 7961.51it/s] 61%|██████    | 244633/400000 [00:31<00:19, 7879.54it/s] 61%|██████▏   | 245442/400000 [00:31<00:19, 7939.05it/s] 62%|██████▏   | 246257/400000 [00:31<00:19, 7998.53it/s] 62%|██████▏   | 247058/400000 [00:32<00:19, 7956.46it/s] 62%|██████▏   | 247875/400000 [00:32<00:18, 8018.18it/s] 62%|██████▏   | 248678/400000 [00:32<00:19, 7817.84it/s] 62%|██████▏   | 249462/400000 [00:32<00:19, 7680.93it/s] 63%|██████▎   | 250232/400000 [00:32<00:19, 7660.74it/s] 63%|██████▎   | 251000/400000 [00:32<00:19, 7622.76it/s] 63%|██████▎   | 251768/400000 [00:32<00:19, 7637.04it/s] 63%|██████▎   | 252533/400000 [00:32<00:19, 7504.68it/s] 63%|██████▎   | 253293/400000 [00:32<00:19, 7532.98it/s] 64%|██████▎   | 254101/400000 [00:32<00:18, 7687.97it/s] 64%|██████▎   | 254916/400000 [00:33<00:18, 7818.26it/s] 64%|██████▍   | 255722/400000 [00:33<00:18, 7885.79it/s] 64%|██████▍   | 256512/400000 [00:33<00:18, 7869.24it/s] 64%|██████▍   | 257330/400000 [00:33<00:17, 7957.24it/s] 65%|██████▍   | 258143/400000 [00:33<00:17, 8002.53it/s] 65%|██████▍   | 258944/400000 [00:33<00:17, 7907.67it/s] 65%|██████▍   | 259744/400000 [00:33<00:17, 7933.42it/s] 65%|██████▌   | 260538/400000 [00:33<00:17, 7755.58it/s] 65%|██████▌   | 261343/400000 [00:33<00:17, 7839.76it/s] 66%|██████▌   | 262157/400000 [00:33<00:17, 7926.08it/s] 66%|██████▌   | 262968/400000 [00:34<00:17, 7977.69it/s] 66%|██████▌   | 263767/400000 [00:34<00:17, 7848.54it/s] 66%|██████▌   | 264553/400000 [00:34<00:17, 7796.76it/s] 66%|██████▋   | 265334/400000 [00:34<00:17, 7759.89it/s] 67%|██████▋   | 266158/400000 [00:34<00:16, 7895.41it/s] 67%|██████▋   | 266964/400000 [00:34<00:16, 7943.74it/s] 67%|██████▋   | 267760/400000 [00:34<00:16, 7917.43it/s] 67%|██████▋   | 268553/400000 [00:34<00:16, 7783.97it/s] 67%|██████▋   | 269349/400000 [00:34<00:16, 7834.05it/s] 68%|██████▊   | 270152/400000 [00:35<00:16, 7889.37it/s] 68%|██████▊   | 270942/400000 [00:35<00:16, 7756.47it/s] 68%|██████▊   | 271740/400000 [00:35<00:16, 7819.96it/s] 68%|██████▊   | 272523/400000 [00:35<00:16, 7797.26it/s] 68%|██████▊   | 273335/400000 [00:35<00:16, 7890.49it/s] 69%|██████▊   | 274128/400000 [00:35<00:15, 7899.82it/s] 69%|██████▊   | 274940/400000 [00:35<00:15, 7962.21it/s] 69%|██████▉   | 275737/400000 [00:35<00:15, 7771.26it/s] 69%|██████▉   | 276516/400000 [00:35<00:16, 7435.97it/s] 69%|██████▉   | 277264/400000 [00:35<00:16, 7411.23it/s] 70%|██████▉   | 278081/400000 [00:36<00:15, 7621.28it/s] 70%|██████▉   | 278894/400000 [00:36<00:15, 7764.02it/s] 70%|██████▉   | 279674/400000 [00:36<00:15, 7715.55it/s] 70%|███████   | 280448/400000 [00:36<00:15, 7667.51it/s] 70%|███████   | 281256/400000 [00:36<00:15, 7786.43it/s] 71%|███████   | 282059/400000 [00:36<00:15, 7856.80it/s] 71%|███████   | 282869/400000 [00:36<00:14, 7927.99it/s] 71%|███████   | 283678/400000 [00:36<00:14, 7974.12it/s] 71%|███████   | 284477/400000 [00:36<00:14, 7834.10it/s] 71%|███████▏  | 285262/400000 [00:36<00:14, 7718.95it/s] 72%|███████▏  | 286036/400000 [00:37<00:14, 7704.81it/s] 72%|███████▏  | 286808/400000 [00:37<00:14, 7697.67it/s] 72%|███████▏  | 287579/400000 [00:37<00:14, 7638.41it/s] 72%|███████▏  | 288344/400000 [00:37<00:14, 7544.75it/s] 72%|███████▏  | 289100/400000 [00:37<00:14, 7485.71it/s] 72%|███████▏  | 289900/400000 [00:37<00:14, 7631.99it/s] 73%|███████▎  | 290690/400000 [00:37<00:14, 7708.77it/s] 73%|███████▎  | 291462/400000 [00:37<00:14, 7625.54it/s] 73%|███████▎  | 292247/400000 [00:37<00:14, 7691.31it/s] 73%|███████▎  | 293046/400000 [00:37<00:13, 7778.37it/s] 73%|███████▎  | 293825/400000 [00:38<00:13, 7769.88it/s] 74%|███████▎  | 294603/400000 [00:38<00:13, 7746.11it/s] 74%|███████▍  | 295387/400000 [00:38<00:13, 7771.06it/s] 74%|███████▍  | 296186/400000 [00:38<00:13, 7832.79it/s] 74%|███████▍  | 296987/400000 [00:38<00:13, 7882.43it/s] 74%|███████▍  | 297776/400000 [00:38<00:12, 7864.11it/s] 75%|███████▍  | 298581/400000 [00:38<00:12, 7917.30it/s] 75%|███████▍  | 299373/400000 [00:38<00:12, 7825.62it/s] 75%|███████▌  | 300158/400000 [00:38<00:12, 7832.65it/s] 75%|███████▌  | 300942/400000 [00:38<00:12, 7789.35it/s] 75%|███████▌  | 301740/400000 [00:39<00:12, 7843.41it/s] 76%|███████▌  | 302554/400000 [00:39<00:12, 7928.15it/s] 76%|███████▌  | 303348/400000 [00:39<00:12, 7681.48it/s] 76%|███████▌  | 304155/400000 [00:39<00:12, 7793.66it/s] 76%|███████▌  | 304974/400000 [00:39<00:12, 7907.60it/s] 76%|███████▋  | 305792/400000 [00:39<00:11, 7986.33it/s] 77%|███████▋  | 306621/400000 [00:39<00:11, 8073.23it/s] 77%|███████▋  | 307430/400000 [00:39<00:11, 7990.49it/s] 77%|███████▋  | 308233/400000 [00:39<00:11, 8001.19it/s] 77%|███████▋  | 309034/400000 [00:39<00:11, 7934.82it/s] 77%|███████▋  | 309829/400000 [00:40<00:11, 7862.83it/s] 78%|███████▊  | 310636/400000 [00:40<00:11, 7921.50it/s] 78%|███████▊  | 311429/400000 [00:40<00:11, 7868.79it/s] 78%|███████▊  | 312219/400000 [00:40<00:11, 7876.93it/s] 78%|███████▊  | 313008/400000 [00:40<00:11, 7668.74it/s] 78%|███████▊  | 313815/400000 [00:40<00:11, 7783.09it/s] 79%|███████▊  | 314633/400000 [00:40<00:10, 7895.71it/s] 79%|███████▉  | 315424/400000 [00:40<00:10, 7729.10it/s] 79%|███████▉  | 316221/400000 [00:40<00:10, 7798.10it/s] 79%|███████▉  | 317003/400000 [00:41<00:10, 7763.84it/s] 79%|███████▉  | 317781/400000 [00:41<00:10, 7658.54it/s] 80%|███████▉  | 318581/400000 [00:41<00:10, 7757.36it/s] 80%|███████▉  | 319365/400000 [00:41<00:10, 7780.22it/s] 80%|████████  | 320173/400000 [00:41<00:10, 7867.52it/s] 80%|████████  | 320990/400000 [00:41<00:09, 7953.35it/s] 80%|████████  | 321797/400000 [00:41<00:09, 7986.65it/s] 81%|████████  | 322608/400000 [00:41<00:09, 8022.18it/s] 81%|████████  | 323411/400000 [00:41<00:09, 7825.08it/s] 81%|████████  | 324195/400000 [00:41<00:09, 7776.86it/s] 81%|████████  | 324974/400000 [00:42<00:09, 7613.17it/s] 81%|████████▏ | 325737/400000 [00:42<00:09, 7536.57it/s] 82%|████████▏ | 326509/400000 [00:42<00:09, 7589.22it/s] 82%|████████▏ | 327269/400000 [00:42<00:09, 7586.17it/s] 82%|████████▏ | 328092/400000 [00:42<00:09, 7767.12it/s] 82%|████████▏ | 328900/400000 [00:42<00:09, 7855.94it/s] 82%|████████▏ | 329701/400000 [00:42<00:08, 7899.34it/s] 83%|████████▎ | 330504/400000 [00:42<00:08, 7937.03it/s] 83%|████████▎ | 331299/400000 [00:42<00:08, 7822.19it/s] 83%|████████▎ | 332083/400000 [00:42<00:08, 7741.41it/s] 83%|████████▎ | 332858/400000 [00:43<00:08, 7649.21it/s] 83%|████████▎ | 333644/400000 [00:43<00:08, 7709.02it/s] 84%|████████▎ | 334458/400000 [00:43<00:08, 7831.58it/s] 84%|████████▍ | 335243/400000 [00:43<00:08, 7770.95it/s] 84%|████████▍ | 336040/400000 [00:43<00:08, 7827.12it/s] 84%|████████▍ | 336839/400000 [00:43<00:08, 7873.04it/s] 84%|████████▍ | 337640/400000 [00:43<00:07, 7912.62it/s] 85%|████████▍ | 338432/400000 [00:43<00:07, 7796.28it/s] 85%|████████▍ | 339213/400000 [00:43<00:07, 7635.72it/s] 85%|████████▌ | 340022/400000 [00:43<00:07, 7765.94it/s] 85%|████████▌ | 340849/400000 [00:44<00:07, 7908.88it/s] 85%|████████▌ | 341652/400000 [00:44<00:07, 7943.90it/s] 86%|████████▌ | 342448/400000 [00:44<00:07, 7851.16it/s] 86%|████████▌ | 343235/400000 [00:44<00:07, 7601.50it/s] 86%|████████▌ | 344021/400000 [00:44<00:07, 7676.28it/s] 86%|████████▌ | 344826/400000 [00:44<00:07, 7782.71it/s] 86%|████████▋ | 345641/400000 [00:44<00:06, 7888.59it/s] 87%|████████▋ | 346432/400000 [00:44<00:06, 7892.75it/s] 87%|████████▋ | 347223/400000 [00:44<00:06, 7680.56it/s] 87%|████████▋ | 348004/400000 [00:45<00:06, 7718.25it/s] 87%|████████▋ | 348798/400000 [00:45<00:06, 7781.48it/s] 87%|████████▋ | 349583/400000 [00:45<00:06, 7799.43it/s] 88%|████████▊ | 350364/400000 [00:45<00:06, 7749.75it/s] 88%|████████▊ | 351140/400000 [00:45<00:06, 7726.75it/s] 88%|████████▊ | 351950/400000 [00:45<00:06, 7832.33it/s] 88%|████████▊ | 352768/400000 [00:45<00:05, 7932.80it/s] 88%|████████▊ | 353600/400000 [00:45<00:05, 8044.11it/s] 89%|████████▊ | 354406/400000 [00:45<00:05, 7917.24it/s] 89%|████████▉ | 355199/400000 [00:45<00:05, 7869.51it/s] 89%|████████▉ | 355987/400000 [00:46<00:05, 7788.79it/s] 89%|████████▉ | 356767/400000 [00:46<00:05, 7700.80it/s] 89%|████████▉ | 357538/400000 [00:46<00:05, 7498.98it/s] 90%|████████▉ | 358295/400000 [00:46<00:05, 7517.72it/s] 90%|████████▉ | 359048/400000 [00:46<00:05, 7490.94it/s] 90%|████████▉ | 359870/400000 [00:46<00:05, 7694.34it/s] 90%|█████████ | 360658/400000 [00:46<00:05, 7748.37it/s] 90%|█████████ | 361444/400000 [00:46<00:04, 7778.45it/s] 91%|█████████ | 362223/400000 [00:46<00:05, 7536.05it/s] 91%|█████████ | 362979/400000 [00:46<00:04, 7531.47it/s] 91%|█████████ | 363795/400000 [00:47<00:04, 7709.50it/s] 91%|█████████ | 364606/400000 [00:47<00:04, 7824.26it/s] 91%|█████████▏| 365392/400000 [00:47<00:04, 7830.81it/s] 92%|█████████▏| 366179/400000 [00:47<00:04, 7840.74it/s] 92%|█████████▏| 366965/400000 [00:47<00:04, 7822.98it/s] 92%|█████████▏| 367766/400000 [00:47<00:04, 7876.70it/s] 92%|█████████▏| 368580/400000 [00:47<00:03, 7951.99it/s] 92%|█████████▏| 369390/400000 [00:47<00:03, 7993.90it/s] 93%|█████████▎| 370190/400000 [00:47<00:03, 7976.55it/s] 93%|█████████▎| 370988/400000 [00:47<00:03, 7918.30it/s] 93%|█████████▎| 371781/400000 [00:48<00:03, 7883.77it/s] 93%|█████████▎| 372570/400000 [00:48<00:03, 7855.79it/s] 93%|█████████▎| 373365/400000 [00:48<00:03, 7882.55it/s] 94%|█████████▎| 374154/400000 [00:48<00:03, 7831.24it/s] 94%|█████████▎| 374938/400000 [00:48<00:03, 7523.40it/s] 94%|█████████▍| 375694/400000 [00:48<00:03, 7418.52it/s] 94%|█████████▍| 376498/400000 [00:48<00:03, 7592.79it/s] 94%|█████████▍| 377260/400000 [00:48<00:03, 7559.35it/s] 95%|█████████▍| 378046/400000 [00:48<00:02, 7645.20it/s] 95%|█████████▍| 378821/400000 [00:48<00:02, 7673.14it/s] 95%|█████████▍| 379614/400000 [00:49<00:02, 7748.36it/s] 95%|█████████▌| 380399/400000 [00:49<00:02, 7777.38it/s] 95%|█████████▌| 381178/400000 [00:49<00:02, 7698.53it/s] 95%|█████████▌| 381964/400000 [00:49<00:02, 7745.57it/s] 96%|█████████▌| 382740/400000 [00:49<00:02, 7730.97it/s] 96%|█████████▌| 383547/400000 [00:49<00:02, 7827.13it/s] 96%|█████████▌| 384365/400000 [00:49<00:01, 7927.40it/s] 96%|█████████▋| 385159/400000 [00:49<00:01, 7573.63it/s] 96%|█████████▋| 385921/400000 [00:49<00:01, 7376.97it/s] 97%|█████████▋| 386663/400000 [00:50<00:01, 7308.00it/s] 97%|█████████▋| 387397/400000 [00:50<00:01, 7232.50it/s] 97%|█████████▋| 388153/400000 [00:50<00:01, 7327.36it/s] 97%|█████████▋| 388888/400000 [00:50<00:01, 7332.47it/s] 97%|█████████▋| 389704/400000 [00:50<00:01, 7562.18it/s] 98%|█████████▊| 390463/400000 [00:50<00:01, 7456.35it/s] 98%|█████████▊| 391229/400000 [00:50<00:01, 7514.38it/s] 98%|█████████▊| 392021/400000 [00:50<00:01, 7630.70it/s] 98%|█████████▊| 392786/400000 [00:50<00:00, 7546.97it/s] 98%|█████████▊| 393601/400000 [00:50<00:00, 7718.00it/s] 99%|█████████▊| 394375/400000 [00:51<00:00, 7691.33it/s] 99%|█████████▉| 395189/400000 [00:51<00:00, 7818.49it/s] 99%|█████████▉| 396014/400000 [00:51<00:00, 7941.94it/s] 99%|█████████▉| 396810/400000 [00:51<00:00, 7713.35it/s] 99%|█████████▉| 397584/400000 [00:51<00:00, 7529.81it/s]100%|█████████▉| 398340/400000 [00:51<00:00, 7520.92it/s]100%|█████████▉| 399129/400000 [00:51<00:00, 7626.10it/s]100%|█████████▉| 399902/400000 [00:51<00:00, 7655.19it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7728.40it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa53c825f28> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011168648074454333 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011011741432457863 	 Accuracy: 52

  model saves at 52% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15987 out of table with 15964 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15987 out of table with 15964 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

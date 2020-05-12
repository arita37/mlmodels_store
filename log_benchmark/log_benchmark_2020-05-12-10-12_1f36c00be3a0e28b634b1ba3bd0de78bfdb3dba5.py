
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1a4f810fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 10:12:49.048272
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 10:12:49.054360
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 10:12:49.057887
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 10:12:49.061662
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1a5b828400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 360741.0938
Epoch 2/10

1/1 [==============================] - 0s 108ms/step - loss: 261555.5000
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 160836.4375
Epoch 4/10

1/1 [==============================] - 0s 109ms/step - loss: 88120.3984
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 48288.6523
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 28230.1777
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 17825.0039
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 12107.0215
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 8727.5615
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 6661.6318

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.4672697e+00 -2.4594191e-01  3.3769691e-01 -1.3962423e+00
  -9.7992110e-01 -7.7069759e-02 -1.1321741e-01 -4.2197126e-01
   6.9003469e-01 -1.1429989e+00 -6.3159639e-01 -5.5647099e-01
  -9.3813014e-01 -7.1477693e-01  7.7811801e-01 -1.4040350e+00
   9.7347647e-01  1.5165831e+00 -6.6814518e-01 -8.7560725e-01
   7.3078024e-01 -4.3616945e-01  1.1319066e+00 -9.1258478e-01
   2.5274855e-01 -1.7742143e+00 -9.9551058e-01 -2.6448631e-01
   5.0219351e-01  4.5143580e-01  8.1708205e-01 -2.4812622e-01
  -3.3530343e-01 -2.1136075e-01  7.4813753e-02  5.2026290e-01
   8.7378192e-01  7.0862043e-01  5.2915573e-01  3.9704862e-01
  -9.5415354e-02 -8.4302789e-01 -8.0682886e-01 -1.2381622e+00
  -1.3493993e+00 -6.4139819e-01 -6.1914045e-01 -9.2396104e-01
   5.2622020e-01  1.9552680e+00  6.4317420e-02  3.6078736e-01
   5.4006147e-01  2.0050263e+00 -2.7188498e-01  3.2003346e-01
   2.8489247e-01  5.5117249e-01 -6.6923606e-01 -3.9856666e-01
  -1.0634837e+00 -1.4841992e+00 -2.1803889e-01  3.2543993e-01
   1.2907846e-01  4.3987519e-01 -6.5718412e-01  1.5964258e+00
  -6.1887646e-01 -8.9341134e-01 -9.7926694e-01  1.1217245e+00
   2.8808159e-01  7.6603323e-01  1.2245591e+00 -1.1841565e+00
  -3.4005386e-01 -8.0627486e-02 -2.7189514e-01  6.5710914e-01
   9.4041330e-01 -6.2375849e-01 -1.5204800e+00 -1.0799518e-01
  -6.8679607e-01 -3.0444229e-01  9.8946691e-04  6.0068011e-01
  -6.3871688e-01  2.4521984e-01 -9.5442736e-01  1.2857472e+00
   1.3584449e+00  3.1117749e-01  3.3597285e-01 -9.5332539e-01
   1.3759480e-01 -5.9909821e-01 -7.7207005e-01 -4.7415900e-01
  -3.1661159e-01  2.4162394e-01  2.2583224e-01 -4.5151949e-01
   8.0135506e-01  1.2160355e-01  8.6477488e-01 -4.1735509e-01
  -3.1534761e-02 -1.1911392e-01 -6.0883474e-01 -6.9611752e-01
  -5.4979980e-02 -8.6702782e-01  2.4683711e-01  6.3775599e-01
   4.8760894e-01 -5.0683767e-01 -2.5450036e-02 -7.0405817e-01
  -2.5932190e-01  7.6678529e+00  8.2058649e+00  7.2054067e+00
   6.9060307e+00  9.1275301e+00  9.0648651e+00  8.5237722e+00
   6.2216396e+00  6.4219584e+00  7.4014935e+00  8.5138054e+00
   8.2784672e+00  8.2530441e+00  9.0509377e+00  9.0981150e+00
   7.8661485e+00  7.0023823e+00  8.2178202e+00  7.8533797e+00
   8.0775824e+00  7.5909796e+00  7.1331329e+00  8.4752235e+00
   8.5597734e+00  8.3734808e+00  7.3435001e+00  7.2059069e+00
   7.8537583e+00  7.9684658e+00  8.5628901e+00  7.7667999e+00
   7.1695561e+00  8.1349783e+00  8.1294603e+00  8.3578358e+00
   7.2241063e+00  8.9456902e+00  7.5832429e+00  8.5578909e+00
   7.6985540e+00  8.3548431e+00  8.7521296e+00  8.3648453e+00
   8.6008177e+00  8.0779943e+00  9.5163250e+00  7.0090494e+00
   7.6105947e+00  8.6535892e+00  8.4831429e+00  8.2181787e+00
   7.3966885e+00  7.7004485e+00  7.9513354e+00  9.0293798e+00
   8.5671759e+00  8.7912102e+00  7.5152206e+00  7.6794457e+00
   1.8552387e+00  6.2701941e-01  1.2054278e+00  1.0657004e+00
   1.2343678e+00  9.9163234e-01  1.4556330e+00  2.8914893e-01
   8.1646812e-01  4.3979251e-01  9.5430785e-01  2.0236506e+00
   3.3097112e-01  1.6451594e+00  4.2702973e-01  5.8461851e-01
   6.6233027e-01  2.0025712e-01  3.3572811e-01  1.2745401e+00
   2.2282801e+00  2.1040423e+00  7.8320229e-01  8.3443028e-01
   2.0356297e+00  6.0421550e-01  5.8909798e-01  4.8291886e-01
   1.4807320e+00  1.6907419e+00  1.1413083e+00  8.0012614e-01
   2.2715726e+00  1.0072438e+00  3.0678815e-01  1.2767452e+00
   1.4065051e+00  5.5843854e-01  2.0149603e+00  1.0188899e+00
   4.2453271e-01  1.9516521e+00  1.7055807e+00  8.1826794e-01
   1.2383388e+00  3.4550178e-01  3.2172096e-01  4.7662151e-01
   9.3766004e-01  6.4886010e-01  8.0768543e-01  1.0743852e+00
   7.9407883e-01  4.9584872e-01  3.2255167e-01  1.3993796e+00
   7.4174958e-01  7.1734113e-01  7.2638559e-01  6.1727691e-01
   1.8361883e+00  9.2807716e-01  5.7630843e-01  5.8773834e-01
   1.0732732e+00  3.7839472e-01  5.2206033e-01  4.5267379e-01
   1.7781163e+00  9.6252066e-01  4.2113173e-01  1.3998748e+00
   5.8264512e-01  7.0152390e-01  1.1906768e+00  5.3068942e-01
   2.0313540e+00  3.5352910e-01  8.9179844e-01  7.3935020e-01
   2.1945407e+00  1.6491392e+00  1.4919422e+00  4.7701371e-01
   1.3611441e+00  1.9982743e+00  1.4458677e+00  4.2768735e-01
   9.2207396e-01  1.8725135e+00  3.6428255e-01  1.2621272e+00
   5.0215811e-01  5.3426546e-01  1.6146264e+00  1.6022080e+00
   1.2879401e+00  2.1121883e+00  9.8901737e-01  1.6026113e+00
   1.1750779e+00  8.9786345e-01  1.3603907e+00  1.1739563e+00
   3.4005320e-01  1.1376615e+00  1.0461411e+00  6.4270133e-01
   3.2503896e+00  4.3473661e-01  9.6227401e-01  2.9019983e+00
   1.5890003e+00  1.5968919e-01  9.8513263e-01  1.1858032e+00
   2.3299592e+00  3.1220078e-01  1.6434386e+00  1.0267431e+00
   8.6168349e-02  7.8765707e+00  8.3290730e+00  7.2253871e+00
   8.3715572e+00  8.5301323e+00  9.5600014e+00  8.0972166e+00
   9.4049215e+00  8.3893089e+00  8.8546152e+00  8.1106224e+00
   9.1009827e+00  8.0797653e+00  9.8625097e+00  7.6118836e+00
   7.8286629e+00  7.4446435e+00  7.8088665e+00  8.5861702e+00
   8.6710835e+00  7.9769139e+00  8.9927378e+00  8.4408197e+00
   9.0118942e+00  9.1279631e+00  7.9111753e+00  8.2424231e+00
   7.7197113e+00  7.6032701e+00  7.7649703e+00  9.1091633e+00
   7.6712604e+00  8.2097416e+00  8.3767509e+00  8.6100368e+00
   8.4706936e+00  8.3878441e+00  8.3734951e+00  9.0850906e+00
   8.3155527e+00  8.1532631e+00  8.5495386e+00  9.0021524e+00
   8.1333389e+00  8.5109978e+00  8.7635794e+00  8.3090982e+00
   8.8565483e+00  8.8507214e+00  7.5342956e+00  9.1385698e+00
   9.0049505e+00  9.0050478e+00  7.8476748e+00  8.1623497e+00
   8.4625521e+00  7.1079659e+00  8.3895807e+00  7.0834222e+00
  -1.0422710e+01 -5.8219833e+00  4.2167158e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 10:12:58.314938
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.8683
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 10:12:58.319617
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8832.92
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 10:12:58.324190
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2079
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 10:12:58.328352
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -790.05
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139750623023792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139749413277936
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139749413278440
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139749413278944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139749413279448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139749413279952

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1a3b43dfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.635840
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.600883
grad_step = 000002, loss = 0.572073
grad_step = 000003, loss = 0.539774
grad_step = 000004, loss = 0.503816
grad_step = 000005, loss = 0.467906
grad_step = 000006, loss = 0.438738
grad_step = 000007, loss = 0.431647
grad_step = 000008, loss = 0.428364
grad_step = 000009, loss = 0.406790
grad_step = 000010, loss = 0.385513
grad_step = 000011, loss = 0.372798
grad_step = 000012, loss = 0.364856
grad_step = 000013, loss = 0.356963
grad_step = 000014, loss = 0.346694
grad_step = 000015, loss = 0.333946
grad_step = 000016, loss = 0.320164
grad_step = 000017, loss = 0.307543
grad_step = 000018, loss = 0.297444
grad_step = 000019, loss = 0.288549
grad_step = 000020, loss = 0.278492
grad_step = 000021, loss = 0.266919
grad_step = 000022, loss = 0.255516
grad_step = 000023, loss = 0.245724
grad_step = 000024, loss = 0.237331
grad_step = 000025, loss = 0.229164
grad_step = 000026, loss = 0.220351
grad_step = 000027, loss = 0.210939
grad_step = 000028, loss = 0.201652
grad_step = 000029, loss = 0.193130
grad_step = 000030, loss = 0.185212
grad_step = 000031, loss = 0.177233
grad_step = 000032, loss = 0.169013
grad_step = 000033, loss = 0.161017
grad_step = 000034, loss = 0.153608
grad_step = 000035, loss = 0.146669
grad_step = 000036, loss = 0.139821
grad_step = 000037, loss = 0.132878
grad_step = 000038, loss = 0.126062
grad_step = 000039, loss = 0.119679
grad_step = 000040, loss = 0.113657
grad_step = 000041, loss = 0.107657
grad_step = 000042, loss = 0.101671
grad_step = 000043, loss = 0.096024
grad_step = 000044, loss = 0.090799
grad_step = 000045, loss = 0.085749
grad_step = 000046, loss = 0.080729
grad_step = 000047, loss = 0.075906
grad_step = 000048, loss = 0.071452
grad_step = 000049, loss = 0.067215
grad_step = 000050, loss = 0.062996
grad_step = 000051, loss = 0.058929
grad_step = 000052, loss = 0.055179
grad_step = 000053, loss = 0.051648
grad_step = 000054, loss = 0.048202
grad_step = 000055, loss = 0.044917
grad_step = 000056, loss = 0.041809
grad_step = 000057, loss = 0.038834
grad_step = 000058, loss = 0.035923
grad_step = 000059, loss = 0.033179
grad_step = 000060, loss = 0.030665
grad_step = 000061, loss = 0.028300
grad_step = 000062, loss = 0.026035
grad_step = 000063, loss = 0.023934
grad_step = 000064, loss = 0.021984
grad_step = 000065, loss = 0.020124
grad_step = 000066, loss = 0.018379
grad_step = 000067, loss = 0.016760
grad_step = 000068, loss = 0.015240
grad_step = 000069, loss = 0.013842
grad_step = 000070, loss = 0.012577
grad_step = 000071, loss = 0.011396
grad_step = 000072, loss = 0.010296
grad_step = 000073, loss = 0.009303
grad_step = 000074, loss = 0.008392
grad_step = 000075, loss = 0.007553
grad_step = 000076, loss = 0.006809
grad_step = 000077, loss = 0.006136
grad_step = 000078, loss = 0.005517
grad_step = 000079, loss = 0.004981
grad_step = 000080, loss = 0.004514
grad_step = 000081, loss = 0.004094
grad_step = 000082, loss = 0.003738
grad_step = 000083, loss = 0.003429
grad_step = 000084, loss = 0.003160
grad_step = 000085, loss = 0.002943
grad_step = 000086, loss = 0.002764
grad_step = 000087, loss = 0.002619
grad_step = 000088, loss = 0.002510
grad_step = 000089, loss = 0.002425
grad_step = 000090, loss = 0.002364
grad_step = 000091, loss = 0.002324
grad_step = 000092, loss = 0.002293
grad_step = 000093, loss = 0.002276
grad_step = 000094, loss = 0.002267
grad_step = 000095, loss = 0.002262
grad_step = 000096, loss = 0.002262
grad_step = 000097, loss = 0.002264
grad_step = 000098, loss = 0.002267
grad_step = 000099, loss = 0.002270
grad_step = 000100, loss = 0.002272
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002272
grad_step = 000102, loss = 0.002270
grad_step = 000103, loss = 0.002268
grad_step = 000104, loss = 0.002263
grad_step = 000105, loss = 0.002257
grad_step = 000106, loss = 0.002248
grad_step = 000107, loss = 0.002238
grad_step = 000108, loss = 0.002227
grad_step = 000109, loss = 0.002215
grad_step = 000110, loss = 0.002202
grad_step = 000111, loss = 0.002189
grad_step = 000112, loss = 0.002175
grad_step = 000113, loss = 0.002162
grad_step = 000114, loss = 0.002149
grad_step = 000115, loss = 0.002136
grad_step = 000116, loss = 0.002124
grad_step = 000117, loss = 0.002112
grad_step = 000118, loss = 0.002100
grad_step = 000119, loss = 0.002090
grad_step = 000120, loss = 0.002079
grad_step = 000121, loss = 0.002070
grad_step = 000122, loss = 0.002062
grad_step = 000123, loss = 0.002055
grad_step = 000124, loss = 0.002047
grad_step = 000125, loss = 0.002038
grad_step = 000126, loss = 0.002030
grad_step = 000127, loss = 0.002025
grad_step = 000128, loss = 0.002020
grad_step = 000129, loss = 0.002014
grad_step = 000130, loss = 0.002007
grad_step = 000131, loss = 0.002001
grad_step = 000132, loss = 0.001996
grad_step = 000133, loss = 0.001992
grad_step = 000134, loss = 0.001988
grad_step = 000135, loss = 0.001984
grad_step = 000136, loss = 0.001978
grad_step = 000137, loss = 0.001973
grad_step = 000138, loss = 0.001968
grad_step = 000139, loss = 0.001963
grad_step = 000140, loss = 0.001959
grad_step = 000141, loss = 0.001955
grad_step = 000142, loss = 0.001952
grad_step = 000143, loss = 0.001951
grad_step = 000144, loss = 0.001952
grad_step = 000145, loss = 0.001957
grad_step = 000146, loss = 0.001958
grad_step = 000147, loss = 0.001950
grad_step = 000148, loss = 0.001933
grad_step = 000149, loss = 0.001918
grad_step = 000150, loss = 0.001913
grad_step = 000151, loss = 0.001916
grad_step = 000152, loss = 0.001921
grad_step = 000153, loss = 0.001921
grad_step = 000154, loss = 0.001917
grad_step = 000155, loss = 0.001906
grad_step = 000156, loss = 0.001892
grad_step = 000157, loss = 0.001881
grad_step = 000158, loss = 0.001875
grad_step = 000159, loss = 0.001872
grad_step = 000160, loss = 0.001872
grad_step = 000161, loss = 0.001876
grad_step = 000162, loss = 0.001885
grad_step = 000163, loss = 0.001906
grad_step = 000164, loss = 0.001934
grad_step = 000165, loss = 0.001967
grad_step = 000166, loss = 0.001958
grad_step = 000167, loss = 0.001908
grad_step = 000168, loss = 0.001846
grad_step = 000169, loss = 0.001830
grad_step = 000170, loss = 0.001861
grad_step = 000171, loss = 0.001896
grad_step = 000172, loss = 0.001903
grad_step = 000173, loss = 0.001865
grad_step = 000174, loss = 0.001822
grad_step = 000175, loss = 0.001805
grad_step = 000176, loss = 0.001820
grad_step = 000177, loss = 0.001844
grad_step = 000178, loss = 0.001852
grad_step = 000179, loss = 0.001837
grad_step = 000180, loss = 0.001807
grad_step = 000181, loss = 0.001785
grad_step = 000182, loss = 0.001780
grad_step = 000183, loss = 0.001790
grad_step = 000184, loss = 0.001803
grad_step = 000185, loss = 0.001810
grad_step = 000186, loss = 0.001809
grad_step = 000187, loss = 0.001798
grad_step = 000188, loss = 0.001782
grad_step = 000189, loss = 0.001766
grad_step = 000190, loss = 0.001753
grad_step = 000191, loss = 0.001745
grad_step = 000192, loss = 0.001741
grad_step = 000193, loss = 0.001740
grad_step = 000194, loss = 0.001741
grad_step = 000195, loss = 0.001747
grad_step = 000196, loss = 0.001761
grad_step = 000197, loss = 0.001793
grad_step = 000198, loss = 0.001856
grad_step = 000199, loss = 0.001977
grad_step = 000200, loss = 0.002063
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002102
grad_step = 000202, loss = 0.001855
grad_step = 000203, loss = 0.001708
grad_step = 000204, loss = 0.001765
grad_step = 000205, loss = 0.001861
grad_step = 000206, loss = 0.001861
grad_step = 000207, loss = 0.001719
grad_step = 000208, loss = 0.001744
grad_step = 000209, loss = 0.001844
grad_step = 000210, loss = 0.001794
grad_step = 000211, loss = 0.001708
grad_step = 000212, loss = 0.001686
grad_step = 000213, loss = 0.001735
grad_step = 000214, loss = 0.001770
grad_step = 000215, loss = 0.001743
grad_step = 000216, loss = 0.001685
grad_step = 000217, loss = 0.001668
grad_step = 000218, loss = 0.001698
grad_step = 000219, loss = 0.001721
grad_step = 000220, loss = 0.001694
grad_step = 000221, loss = 0.001664
grad_step = 000222, loss = 0.001652
grad_step = 000223, loss = 0.001667
grad_step = 000224, loss = 0.001680
grad_step = 000225, loss = 0.001671
grad_step = 000226, loss = 0.001646
grad_step = 000227, loss = 0.001635
grad_step = 000228, loss = 0.001637
grad_step = 000229, loss = 0.001646
grad_step = 000230, loss = 0.001649
grad_step = 000231, loss = 0.001643
grad_step = 000232, loss = 0.001628
grad_step = 000233, loss = 0.001617
grad_step = 000234, loss = 0.001613
grad_step = 000235, loss = 0.001614
grad_step = 000236, loss = 0.001616
grad_step = 000237, loss = 0.001619
grad_step = 000238, loss = 0.001619
grad_step = 000239, loss = 0.001615
grad_step = 000240, loss = 0.001610
grad_step = 000241, loss = 0.001604
grad_step = 000242, loss = 0.001598
grad_step = 000243, loss = 0.001592
grad_step = 000244, loss = 0.001588
grad_step = 000245, loss = 0.001585
grad_step = 000246, loss = 0.001582
grad_step = 000247, loss = 0.001580
grad_step = 000248, loss = 0.001581
grad_step = 000249, loss = 0.001585
grad_step = 000250, loss = 0.001597
grad_step = 000251, loss = 0.001624
grad_step = 000252, loss = 0.001694
grad_step = 000253, loss = 0.001815
grad_step = 000254, loss = 0.002044
grad_step = 000255, loss = 0.002066
grad_step = 000256, loss = 0.001986
grad_step = 000257, loss = 0.001639
grad_step = 000258, loss = 0.001577
grad_step = 000259, loss = 0.001745
grad_step = 000260, loss = 0.001747
grad_step = 000261, loss = 0.001599
grad_step = 000262, loss = 0.001573
grad_step = 000263, loss = 0.001683
grad_step = 000264, loss = 0.001710
grad_step = 000265, loss = 0.001580
grad_step = 000266, loss = 0.001546
grad_step = 000267, loss = 0.001586
grad_step = 000268, loss = 0.001615
grad_step = 000269, loss = 0.001642
grad_step = 000270, loss = 0.001585
grad_step = 000271, loss = 0.001527
grad_step = 000272, loss = 0.001540
grad_step = 000273, loss = 0.001561
grad_step = 000274, loss = 0.001572
grad_step = 000275, loss = 0.001562
grad_step = 000276, loss = 0.001527
grad_step = 000277, loss = 0.001508
grad_step = 000278, loss = 0.001527
grad_step = 000279, loss = 0.001537
grad_step = 000280, loss = 0.001535
grad_step = 000281, loss = 0.001527
grad_step = 000282, loss = 0.001503
grad_step = 000283, loss = 0.001496
grad_step = 000284, loss = 0.001506
grad_step = 000285, loss = 0.001509
grad_step = 000286, loss = 0.001508
grad_step = 000287, loss = 0.001504
grad_step = 000288, loss = 0.001491
grad_step = 000289, loss = 0.001482
grad_step = 000290, loss = 0.001483
grad_step = 000291, loss = 0.001484
grad_step = 000292, loss = 0.001486
grad_step = 000293, loss = 0.001488
grad_step = 000294, loss = 0.001486
grad_step = 000295, loss = 0.001481
grad_step = 000296, loss = 0.001476
grad_step = 000297, loss = 0.001470
grad_step = 000298, loss = 0.001465
grad_step = 000299, loss = 0.001462
grad_step = 000300, loss = 0.001460
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001458
grad_step = 000302, loss = 0.001457
grad_step = 000303, loss = 0.001457
grad_step = 000304, loss = 0.001455
grad_step = 000305, loss = 0.001456
grad_step = 000306, loss = 0.001458
grad_step = 000307, loss = 0.001461
grad_step = 000308, loss = 0.001465
grad_step = 000309, loss = 0.001473
grad_step = 000310, loss = 0.001479
grad_step = 000311, loss = 0.001488
grad_step = 000312, loss = 0.001485
grad_step = 000313, loss = 0.001480
grad_step = 000314, loss = 0.001469
grad_step = 000315, loss = 0.001473
grad_step = 000316, loss = 0.001521
grad_step = 000317, loss = 0.001657
grad_step = 000318, loss = 0.001997
grad_step = 000319, loss = 0.002098
grad_step = 000320, loss = 0.002404
grad_step = 000321, loss = 0.002372
grad_step = 000322, loss = 0.001821
grad_step = 000323, loss = 0.001486
grad_step = 000324, loss = 0.001669
grad_step = 000325, loss = 0.001976
grad_step = 000326, loss = 0.001706
grad_step = 000327, loss = 0.001457
grad_step = 000328, loss = 0.001587
grad_step = 000329, loss = 0.001723
grad_step = 000330, loss = 0.001566
grad_step = 000331, loss = 0.001435
grad_step = 000332, loss = 0.001560
grad_step = 000333, loss = 0.001600
grad_step = 000334, loss = 0.001486
grad_step = 000335, loss = 0.001433
grad_step = 000336, loss = 0.001516
grad_step = 000337, loss = 0.001519
grad_step = 000338, loss = 0.001440
grad_step = 000339, loss = 0.001433
grad_step = 000340, loss = 0.001485
grad_step = 000341, loss = 0.001475
grad_step = 000342, loss = 0.001418
grad_step = 000343, loss = 0.001428
grad_step = 000344, loss = 0.001457
grad_step = 000345, loss = 0.001441
grad_step = 000346, loss = 0.001408
grad_step = 000347, loss = 0.001417
grad_step = 000348, loss = 0.001436
grad_step = 000349, loss = 0.001421
grad_step = 000350, loss = 0.001399
grad_step = 000351, loss = 0.001405
grad_step = 000352, loss = 0.001418
grad_step = 000353, loss = 0.001408
grad_step = 000354, loss = 0.001392
grad_step = 000355, loss = 0.001392
grad_step = 000356, loss = 0.001400
grad_step = 000357, loss = 0.001400
grad_step = 000358, loss = 0.001388
grad_step = 000359, loss = 0.001381
grad_step = 000360, loss = 0.001383
grad_step = 000361, loss = 0.001387
grad_step = 000362, loss = 0.001382
grad_step = 000363, loss = 0.001375
grad_step = 000364, loss = 0.001371
grad_step = 000365, loss = 0.001372
grad_step = 000366, loss = 0.001374
grad_step = 000367, loss = 0.001371
grad_step = 000368, loss = 0.001366
grad_step = 000369, loss = 0.001361
grad_step = 000370, loss = 0.001359
grad_step = 000371, loss = 0.001359
grad_step = 000372, loss = 0.001360
grad_step = 000373, loss = 0.001360
grad_step = 000374, loss = 0.001359
grad_step = 000375, loss = 0.001355
grad_step = 000376, loss = 0.001350
grad_step = 000377, loss = 0.001346
grad_step = 000378, loss = 0.001343
grad_step = 000379, loss = 0.001340
grad_step = 000380, loss = 0.001339
grad_step = 000381, loss = 0.001339
grad_step = 000382, loss = 0.001338
grad_step = 000383, loss = 0.001338
grad_step = 000384, loss = 0.001338
grad_step = 000385, loss = 0.001337
grad_step = 000386, loss = 0.001337
grad_step = 000387, loss = 0.001338
grad_step = 000388, loss = 0.001339
grad_step = 000389, loss = 0.001342
grad_step = 000390, loss = 0.001346
grad_step = 000391, loss = 0.001353
grad_step = 000392, loss = 0.001365
grad_step = 000393, loss = 0.001381
grad_step = 000394, loss = 0.001403
grad_step = 000395, loss = 0.001431
grad_step = 000396, loss = 0.001459
grad_step = 000397, loss = 0.001486
grad_step = 000398, loss = 0.001494
grad_step = 000399, loss = 0.001481
grad_step = 000400, loss = 0.001438
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001381
grad_step = 000402, loss = 0.001326
grad_step = 000403, loss = 0.001295
grad_step = 000404, loss = 0.001292
grad_step = 000405, loss = 0.001310
grad_step = 000406, loss = 0.001337
grad_step = 000407, loss = 0.001362
grad_step = 000408, loss = 0.001373
grad_step = 000409, loss = 0.001369
grad_step = 000410, loss = 0.001348
grad_step = 000411, loss = 0.001319
grad_step = 000412, loss = 0.001293
grad_step = 000413, loss = 0.001275
grad_step = 000414, loss = 0.001268
grad_step = 000415, loss = 0.001271
grad_step = 000416, loss = 0.001280
grad_step = 000417, loss = 0.001291
grad_step = 000418, loss = 0.001299
grad_step = 000419, loss = 0.001303
grad_step = 000420, loss = 0.001303
grad_step = 000421, loss = 0.001298
grad_step = 000422, loss = 0.001288
grad_step = 000423, loss = 0.001275
grad_step = 000424, loss = 0.001262
grad_step = 000425, loss = 0.001251
grad_step = 000426, loss = 0.001244
grad_step = 000427, loss = 0.001240
grad_step = 000428, loss = 0.001239
grad_step = 000429, loss = 0.001240
grad_step = 000430, loss = 0.001244
grad_step = 000431, loss = 0.001248
grad_step = 000432, loss = 0.001254
grad_step = 000433, loss = 0.001264
grad_step = 000434, loss = 0.001280
grad_step = 000435, loss = 0.001301
grad_step = 000436, loss = 0.001327
grad_step = 000437, loss = 0.001356
grad_step = 000438, loss = 0.001381
grad_step = 000439, loss = 0.001403
grad_step = 000440, loss = 0.001417
grad_step = 000441, loss = 0.001407
grad_step = 000442, loss = 0.001372
grad_step = 000443, loss = 0.001313
grad_step = 000444, loss = 0.001256
grad_step = 000445, loss = 0.001220
grad_step = 000446, loss = 0.001208
grad_step = 000447, loss = 0.001214
grad_step = 000448, loss = 0.001231
grad_step = 000449, loss = 0.001253
grad_step = 000450, loss = 0.001271
grad_step = 000451, loss = 0.001275
grad_step = 000452, loss = 0.001261
grad_step = 000453, loss = 0.001236
grad_step = 000454, loss = 0.001213
grad_step = 000455, loss = 0.001198
grad_step = 000456, loss = 0.001189
grad_step = 000457, loss = 0.001188
grad_step = 000458, loss = 0.001193
grad_step = 000459, loss = 0.001202
grad_step = 000460, loss = 0.001211
grad_step = 000461, loss = 0.001214
grad_step = 000462, loss = 0.001212
grad_step = 000463, loss = 0.001206
grad_step = 000464, loss = 0.001201
grad_step = 000465, loss = 0.001196
grad_step = 000466, loss = 0.001190
grad_step = 000467, loss = 0.001183
grad_step = 000468, loss = 0.001175
grad_step = 000469, loss = 0.001169
grad_step = 000470, loss = 0.001165
grad_step = 000471, loss = 0.001163
grad_step = 000472, loss = 0.001161
grad_step = 000473, loss = 0.001159
grad_step = 000474, loss = 0.001156
grad_step = 000475, loss = 0.001153
grad_step = 000476, loss = 0.001152
grad_step = 000477, loss = 0.001151
grad_step = 000478, loss = 0.001150
grad_step = 000479, loss = 0.001150
grad_step = 000480, loss = 0.001151
grad_step = 000481, loss = 0.001154
grad_step = 000482, loss = 0.001162
grad_step = 000483, loss = 0.001181
grad_step = 000484, loss = 0.001226
grad_step = 000485, loss = 0.001324
grad_step = 000486, loss = 0.001497
grad_step = 000487, loss = 0.001743
grad_step = 000488, loss = 0.002008
grad_step = 000489, loss = 0.002188
grad_step = 000490, loss = 0.002067
grad_step = 000491, loss = 0.001713
grad_step = 000492, loss = 0.001214
grad_step = 000493, loss = 0.001242
grad_step = 000494, loss = 0.001612
grad_step = 000495, loss = 0.001546
grad_step = 000496, loss = 0.001220
grad_step = 000497, loss = 0.001201
grad_step = 000498, loss = 0.001360
grad_step = 000499, loss = 0.001311
grad_step = 000500, loss = 0.001214
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001243
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

  date_run                              2020-05-12 10:13:23.047906
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.257581
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 10:13:23.054312
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.168023
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 10:13:23.062545
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13936
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 10:13:23.069038
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.55317
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
0   2020-05-12 10:12:49.048272  ...    mean_absolute_error
1   2020-05-12 10:12:49.054360  ...     mean_squared_error
2   2020-05-12 10:12:49.057887  ...  median_absolute_error
3   2020-05-12 10:12:49.061662  ...               r2_score
4   2020-05-12 10:12:58.314938  ...    mean_absolute_error
5   2020-05-12 10:12:58.319617  ...     mean_squared_error
6   2020-05-12 10:12:58.324190  ...  median_absolute_error
7   2020-05-12 10:12:58.328352  ...               r2_score
8   2020-05-12 10:13:23.047906  ...    mean_absolute_error
9   2020-05-12 10:13:23.054312  ...     mean_squared_error
10  2020-05-12 10:13:23.062545  ...  median_absolute_error
11  2020-05-12 10:13:23.069038  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140279.62it/s] 49%|████▉     | 4898816/9912422 [00:00<00:25, 200152.53it/s]9920512it [00:00, 36006399.32it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1105472.33it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475150.80it/s]1654784it [00:00, 10615465.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210151.31it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f57132defd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56b09faef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f571326aef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56b04d20f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f57132defd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56c5c62e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f571326aef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56b9d13748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f57132a6ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56b9d13748> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56b04d2438> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff20e063208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=1c658d6294c1e90548bd6fb29c75d2e532110060cb7b7e5f7a12e9b47baacdf2
  Stored in directory: /tmp/pip-ephem-wheel-cache-cmopf9x6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff2041d1080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3334144/17464789 [====>.........................] - ETA: 0s
11280384/17464789 [==================>...........] - ETA: 0s
16252928/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 10:14:50.747380: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 10:14:50.751802: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-12 10:14:50.751949: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566f040d310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 10:14:50.751966: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7126 - accuracy: 0.4970
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7075 - accuracy: 0.4973 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7030 - accuracy: 0.4976
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 4s - loss: 7.6736 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 4s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6654 - accuracy: 0.5001
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 3s - loss: 7.6452 - accuracy: 0.5014
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6982 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7118 - accuracy: 0.4971
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 10s 386us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 10:15:07.615166
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 10:15:07.615166  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 10:15:14.186843: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 10:15:14.192368: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-12 10:15:14.192542: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5629cdf7c340 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 10:15:14.192557: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6a1800cbe0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4819 - crf_viterbi_accuracy: 0.2000 - val_loss: 1.3851 - val_crf_viterbi_accuracy: 0.8133

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6a18045c18> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7663 - accuracy: 0.4935
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7331 - accuracy: 0.4957 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7801 - accuracy: 0.4926
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7637 - accuracy: 0.4937
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7345 - accuracy: 0.4956
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7318 - accuracy: 0.4958
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7416 - accuracy: 0.4951
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
11000/25000 [============>.................] - ETA: 4s - loss: 7.7015 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 4s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7044 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7028 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 3s - loss: 7.6758 - accuracy: 0.4994
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7000 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6990 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6996 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7075 - accuracy: 0.4973
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7168 - accuracy: 0.4967
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7193 - accuracy: 0.4966
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6871 - accuracy: 0.4987
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f69f5930eb8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<100:38:43, 2.38kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<70:41:02, 3.39kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<49:31:31, 4.83kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<34:39:17, 6.90kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<24:11:04, 9.86kB/s].vector_cache/glove.6B.zip:   1%|          | 8.13M/862M [00:04<16:50:32, 14.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:04<11:43:42, 20.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:04<8:10:33, 28.7kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:04<5:41:41, 41.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.0M/862M [00:04<3:58:05, 58.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:04<2:45:55, 83.7kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:04<1:55:57, 119kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:04<1:20:49, 170kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:04<56:23, 243kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:05<39:21, 346kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:05<27:30, 493kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:05<19:51, 680kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:07<15:44, 853kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:07<15:08, 887kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:07<11:24, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:07<08:14, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:08<06:00, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:09<19:52, 672kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:09<15:53, 841kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:09<11:32, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<08:12, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:11<47:24, 280kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:11<35:02, 379kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:11<24:57, 532kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:13<19:52, 665kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:13<16:54, 781kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:13<12:34, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:14<08:57, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:15<22:09, 593kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:15<16:51, 780kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:15<12:06, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:17<11:32, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:17<10:44, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:17<08:10, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:17<05:53, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:19<10:16, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:19<08:48, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:19<06:29, 2.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<04:43, 2.74MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:21<17:42, 731kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:21<13:55, 929kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:21<10:06, 1.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:23<09:44, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:23<09:33, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:23<07:24, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<05:20, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:25<41:59, 305kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:25<30:43, 417kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:25<21:44, 587kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:27<18:10, 701kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:27<15:21, 829kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:27<11:23, 1.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<08:04, 1.57MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<22:26, 565kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<17:01, 744kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<12:13, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<11:27, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<09:18, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:31<06:49, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<07:43, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<06:41, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<04:59, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:26, 1.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:35<04:21, 2.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<05:59, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<05:26, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<06:51, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<06:24, 1.92MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<05:45, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:39<04:18, 2.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<05:53, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<06:38, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<05:39, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<05:11, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<03:56, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<06:17, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<04:55, 2.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<03:37, 3.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<07:25, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<06:26, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<04:45, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<06:08, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<05:21, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<04:02, 2.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<02:58, 4.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<50:04, 238kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<37:28, 318kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<26:43, 445kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<18:47, 631kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<20:54, 566kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<15:53, 745kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<11:23, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<08:06, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<49:17, 239kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<35:42, 330kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<25:14, 465kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<20:22, 574kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<15:36, 750kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<11:29, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<08:10, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<11:57, 973kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<11:14, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<08:56, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<06:36, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<04:44, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<1:16:30, 151kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<54:46, 211kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<38:36, 299kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<27:04, 425kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<59:06, 195kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:02<42:34, 270kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<30:02, 382kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<21:05, 542kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<1:39:54, 114kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:04<1:11:05, 161kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<49:59, 228kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<34:59, 325kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<1:39:09, 115kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<1:11:47, 158kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<50:46, 224kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<35:46, 317kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<27:33, 410kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<20:28, 551kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<14:38, 770kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<10:20, 1.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<48:51, 230kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<36:29, 308kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<26:13, 428kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<18:36, 602kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<15:27, 722kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<11:56, 933kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<08:34, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:08, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<48:12, 230kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<34:50, 318kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<24:37, 449kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<19:46, 557kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<14:47, 745kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<10:37, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<07:42, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<09:06, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<07:42, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<05:40, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<04:07, 2.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<11:44, 927kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<09:30, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<06:56, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:00, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<11:29, 941kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<09:23, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<06:53, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<06:58, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<06:09, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<04:35, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<03:21, 3.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<16:01, 666kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<12:29, 854kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<09:00, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<06:26, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<14:25, 735kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<11:19, 936kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<08:11, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<15:21, 686kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<11:45, 896kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<08:33, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<06:06, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<12:30, 837kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<09:57, 1.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<07:12, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:11, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<13:22, 777kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<10:32, 986kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<07:37, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:28, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<19:19, 535kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<14:30, 711kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<10:27, 986kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:26, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<14:09, 724kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<10:51, 944kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:38<07:57, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:41, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<10:03, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<09:26, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<07:07, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<06:01, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<04:02, 2.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<03:01, 3.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<07:25, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<06:33, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<04:53, 2.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<03:35, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<07:27, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<06:34, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<04:56, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<05:20, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<04:51, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<03:37, 2.72MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<02:39, 3.70MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<1:24:56, 116kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<1:00:27, 163kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<42:26, 231kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<31:51, 307kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<23:16, 420kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<16:29, 591kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<13:48, 703kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<10:39, 911kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<07:41, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<07:38, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<04:33, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<03:18, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<38:29, 249kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<27:54, 343kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<19:41, 484kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<15:59, 594kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<13:07, 723kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<09:35, 988kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<06:51, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<08:08, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<06:40, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<04:51, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<05:36, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<06:07, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<04:44, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<03:27, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<05:37, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<04:45, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:06<03:34, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<02:36, 3.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<19:13, 480kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<14:23, 641kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:08<10:16, 894kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<09:19, 983kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<07:27, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:10<05:26, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<05:56, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<05:05, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<03:46, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<04:46, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<05:10, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<04:04, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<04:17, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<03:47, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<02:53, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<04:04, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<04:41, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<03:39, 2.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<02:41, 3.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<05:03, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<04:27, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<03:20, 2.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<04:22, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<02:55, 2.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<04:06, 2.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<04:40, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<03:43, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:25<03:58, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<03:39, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<03:00, 2.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<02:13, 3.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<05:02, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<05:29, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<04:38, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<03:29, 2.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<04:03, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<03:57, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<03:09, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<02:20, 3.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<04:38, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<05:55, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<05:44, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<04:22, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<03:12, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:41, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:03, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<03:13, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<02:22, 3.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<05:46, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<05:46, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<04:23, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<03:17, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<04:12, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<03:49, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<02:52, 2.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<02:06, 3.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<31:38, 257kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<23:51, 341kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<17:01, 476kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<11:59, 674kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<11:37, 693kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<08:57, 899kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<06:32, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:38, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<32:54, 243kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<23:50, 335kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<16:50, 473kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<11:49, 670kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<1:07:12, 118kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<48:39, 163kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<34:27, 230kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<24:15, 325kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<18:43, 420kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<13:55, 564kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<09:55, 789kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<08:44, 891kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<06:48, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<04:55, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<03:43, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<04:54, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<04:23, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:51<03:16, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<02:25, 3.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<04:42, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<05:58, 1.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<04:45, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<03:27, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<04:21, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<03:57, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<02:57, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<03:37, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<04:04, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<03:13, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<03:10, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<02:24, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:01<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<03:54, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<03:05, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<02:14, 3.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<20:51, 349kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<16:01, 454kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<11:34, 629kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<09:13, 783kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<07:12, 1.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<05:12, 1.38MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<05:30, 1.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<05:19, 1.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<04:01, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<02:55, 2.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<04:42, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<03:55, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<02:53, 2.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<02:06, 3.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<18:23, 381kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<14:19, 490kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:11<10:22, 675kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<07:17, 954kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<6:44:07, 17.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<4:43:20, 24.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<3:17:46, 35.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<2:19:20, 49.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<1:38:08, 70.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<1:08:35, 99.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<49:23, 138kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<35:56, 189kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<25:27, 267kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<17:45, 379kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<6:40:01, 16.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<4:40:26, 24.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<3:15:43, 34.3kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<2:17:50, 48.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<1:37:52, 68.1kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<1:08:43, 96.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<47:49, 138kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<57:17, 115kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<40:38, 162kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<28:31, 230kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<21:21, 306kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<15:37, 418kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<11:02, 589kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<09:12, 702kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<07:05, 910kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<05:06, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<05:04, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<04:11, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<03:05, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<03:38, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<03:52, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<02:58, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<02:11, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<03:27, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<03:04, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<02:18, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<03:02, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<03:23, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<02:38, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<01:54, 3.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<06:05, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<04:54, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<03:53, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<03:58, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<03:02, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<02:12, 2.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:54, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<02:38, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<01:54, 3.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<04:53, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<04:41, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<03:51, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:50, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<02:02, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<56:54, 103kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<40:24, 145kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<28:18, 206kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<21:02, 275kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<15:56, 363kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<11:23, 506kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<07:59, 716kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<08:33, 667kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<06:34, 868kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<04:43, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<04:36, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<04:21, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<03:17, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<02:23, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<03:36, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<03:07, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:20, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<01:41, 3.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<14:17, 385kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<10:28, 525kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<07:32, 728kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:17, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<20:06, 270kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<14:37, 372kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<10:19, 524kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<07:14, 741kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<44:54, 120kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<31:57, 168kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<22:23, 238kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<16:49, 315kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<12:18, 430kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<08:41, 607kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<07:17, 718kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<06:09, 849kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<04:31, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<03:16, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:42, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<03:07, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<02:17, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<02:47, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<02:23, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<01:46, 2.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:06<01:18, 3.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<21:07, 238kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<15:16, 329kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<10:45, 464kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:10<08:39, 573kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<06:33, 755kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<04:40, 1.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<04:24, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<04:05, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<03:06, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:12, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<05:30, 876kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<04:20, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<03:08, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:18, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:16, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:29, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<01:48, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<03:02, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<02:36, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<01:56, 2.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:25, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<02:05, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:34, 2.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<02:10, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:28, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<02:05, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:23, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<01:52, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:20, 3.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<09:35, 460kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<07:09, 615kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<05:05, 859kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<04:33, 953kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<03:38, 1.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:37, 1.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:50, 1.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:52, 1.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<02:11, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<01:35, 2.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:40, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:18, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:42, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<02:09, 1.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<02:21, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<01:49, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:19, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<03:38, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<02:09, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<02:25, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:06, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<01:33, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:00, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<01:44, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<01:17, 3.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<00:57, 4.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<15:16, 253kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<11:03, 348kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<07:47, 491kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<06:17, 602kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<05:10, 732kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<03:48, 993kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<03:13, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<02:38, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<01:55, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:16, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<01:44, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<01:15, 2.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<04:04, 879kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<03:13, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<02:19, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<02:26, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<02:03, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<01:31, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<01:52, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<01:39, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:13, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<01:25, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:04, 3.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<00:47, 4.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<15:59, 207kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<11:30, 287kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<08:04, 406kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<06:21, 510kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<04:45, 679kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<03:22, 950kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<03:05, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:48, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:07, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:57, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:38, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<01:12, 2.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<00:52, 3.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<13:44, 221kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<10:13, 297kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<07:16, 415kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<05:29, 541kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<04:08, 716kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<02:56, 997kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<02:43, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<02:29, 1.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:51, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:19, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<02:41, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<02:10, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<01:34, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<01:44, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:30, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:06, 2.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<01:24, 1.92MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:15, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<00:56, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:16, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<00:52, 2.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:12, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:28, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:16, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<00:56, 2.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<01:09, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<01:04, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<00:48, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<00:35, 4.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<09:23, 258kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<06:48, 354kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<04:46, 500kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<03:47, 620kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<03:09, 746kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:24<02:21, 994kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:41, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:41, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:25, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:03, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:45, 2.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<04:45, 465kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<03:32, 622kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<02:33, 858kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<01:46, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<15:23, 139kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<11:12, 191kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<07:57, 268kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<05:33, 380kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:32<04:17, 484kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<03:10, 652kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<02:18, 891kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:37, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<02:12, 911kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<01:57, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:26, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:02, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:36<01:14, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<01:03, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<00:46, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<00:34, 3.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<01:34, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<01:18, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<00:56, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<01:03, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<01:06, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<00:52, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<00:37, 2.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<01:02, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<00:40, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<00:48, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<01:11, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:58, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:45<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:53, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:48, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:36, 2.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<00:40, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:29, 3.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<00:30, 2.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:22, 3.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:57, 1.45MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:57, 1.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:46, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:34, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:39, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:35, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:26, 2.88MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:35, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:32, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:23, 3.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:37, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:29, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:20, 3.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<1:03:08, 17.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<44:03, 24.9kB/s]  .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<30:08, 35.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<20:37, 50.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<14:36, 70.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<10:08, 100kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<06:56, 143kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<05:00, 193kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<03:33, 269kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<02:31, 376kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<01:42, 534kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:06<01:45, 512kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<01:18, 680kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:54, 950kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:37, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<09:05, 91.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<06:30, 127kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<04:31, 180kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<03:03, 256kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<02:23, 318kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<01:44, 434kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<01:11, 610kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:57, 723kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:48, 853kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:35, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:24, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:25, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:21, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:15, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:10, 3.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<02:18, 240kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<01:43, 320kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<01:12, 447kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:16<00:48, 631kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:43, 670kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:32, 872kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:22, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:20, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:16, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:11, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:08, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:12, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:08, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:05, 3.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:11, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:09, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:06, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:13, 930kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:10, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:07, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:09, 932kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:07, 1.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:04, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 2.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:05, 848kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:03, 1.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:01, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 768kB/s] .vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 753/400000 [00:00<00:53, 7526.08it/s]  0%|          | 1518/400000 [00:00<00:52, 7562.73it/s]  1%|          | 2385/400000 [00:00<00:50, 7856.62it/s]  1%|          | 3266/400000 [00:00<00:48, 8118.52it/s]  1%|          | 4137/400000 [00:00<00:47, 8284.86it/s]  1%|▏         | 5022/400000 [00:00<00:46, 8444.27it/s]  1%|▏         | 5806/400000 [00:00<00:47, 8251.39it/s]  2%|▏         | 6630/400000 [00:00<00:47, 8245.13it/s]  2%|▏         | 7473/400000 [00:00<00:47, 8299.53it/s]  2%|▏         | 8334/400000 [00:01<00:46, 8387.68it/s]  2%|▏         | 9185/400000 [00:01<00:46, 8421.52it/s]  3%|▎         | 10014/400000 [00:01<00:48, 8077.35it/s]  3%|▎         | 10895/400000 [00:01<00:46, 8283.74it/s]  3%|▎         | 11780/400000 [00:01<00:45, 8442.11it/s]  3%|▎         | 12623/400000 [00:01<00:46, 8381.15it/s]  3%|▎         | 13488/400000 [00:01<00:45, 8459.94it/s]  4%|▎         | 14334/400000 [00:01<00:47, 8197.04it/s]  4%|▍         | 15156/400000 [00:01<00:51, 7505.35it/s]  4%|▍         | 15919/400000 [00:01<00:52, 7349.14it/s]  4%|▍         | 16664/400000 [00:02<00:53, 7208.62it/s]  4%|▍         | 17393/400000 [00:02<00:53, 7184.00it/s]  5%|▍         | 18144/400000 [00:02<00:52, 7278.67it/s]  5%|▍         | 18935/400000 [00:02<00:51, 7456.41it/s]  5%|▍         | 19721/400000 [00:02<00:50, 7572.64it/s]  5%|▌         | 20525/400000 [00:02<00:49, 7704.73it/s]  5%|▌         | 21323/400000 [00:02<00:48, 7782.66it/s]  6%|▌         | 22104/400000 [00:02<00:49, 7652.36it/s]  6%|▌         | 22891/400000 [00:02<00:48, 7715.03it/s]  6%|▌         | 23727/400000 [00:02<00:47, 7896.23it/s]  6%|▌         | 24519/400000 [00:03<00:47, 7877.48it/s]  6%|▋         | 25309/400000 [00:03<00:47, 7807.28it/s]  7%|▋         | 26120/400000 [00:03<00:47, 7894.13it/s]  7%|▋         | 26911/400000 [00:03<00:47, 7885.04it/s]  7%|▋         | 27701/400000 [00:03<00:47, 7849.81it/s]  7%|▋         | 28531/400000 [00:03<00:46, 7978.78it/s]  7%|▋         | 29337/400000 [00:03<00:46, 8001.29it/s]  8%|▊         | 30178/400000 [00:03<00:45, 8117.42it/s]  8%|▊         | 31023/400000 [00:03<00:44, 8213.39it/s]  8%|▊         | 31846/400000 [00:03<00:45, 8094.77it/s]  8%|▊         | 32661/400000 [00:04<00:45, 8110.72it/s]  8%|▊         | 33473/400000 [00:04<00:46, 7964.96it/s]  9%|▊         | 34271/400000 [00:04<00:46, 7808.76it/s]  9%|▉         | 35054/400000 [00:04<00:47, 7714.66it/s]  9%|▉         | 35827/400000 [00:04<00:47, 7685.01it/s]  9%|▉         | 36600/400000 [00:04<00:47, 7695.64it/s]  9%|▉         | 37371/400000 [00:04<00:47, 7607.37it/s] 10%|▉         | 38133/400000 [00:04<00:48, 7492.78it/s] 10%|▉         | 38884/400000 [00:04<00:49, 7243.50it/s] 10%|▉         | 39611/400000 [00:05<00:50, 7107.01it/s] 10%|█         | 40325/400000 [00:05<00:50, 7115.69it/s] 10%|█         | 41102/400000 [00:05<00:49, 7298.82it/s] 10%|█         | 41975/400000 [00:05<00:46, 7673.90it/s] 11%|█         | 42821/400000 [00:05<00:45, 7892.94it/s] 11%|█         | 43617/400000 [00:05<00:45, 7901.68it/s] 11%|█         | 44412/400000 [00:05<00:45, 7800.23it/s] 11%|█▏        | 45197/400000 [00:05<00:45, 7814.23it/s] 12%|█▏        | 46025/400000 [00:05<00:44, 7947.79it/s] 12%|█▏        | 46822/400000 [00:05<00:44, 7872.23it/s] 12%|█▏        | 47611/400000 [00:06<00:45, 7699.00it/s] 12%|█▏        | 48425/400000 [00:06<00:44, 7825.59it/s] 12%|█▏        | 49210/400000 [00:06<00:44, 7825.03it/s] 13%|█▎        | 50014/400000 [00:06<00:44, 7886.68it/s] 13%|█▎        | 50847/400000 [00:06<00:43, 8013.67it/s] 13%|█▎        | 51671/400000 [00:06<00:43, 8079.86it/s] 13%|█▎        | 52480/400000 [00:06<00:42, 8082.05it/s] 13%|█▎        | 53289/400000 [00:06<00:44, 7828.31it/s] 14%|█▎        | 54104/400000 [00:06<00:43, 7920.41it/s] 14%|█▎        | 54939/400000 [00:06<00:42, 8043.38it/s] 14%|█▍        | 55770/400000 [00:07<00:42, 8121.47it/s] 14%|█▍        | 56584/400000 [00:07<00:42, 8094.45it/s] 14%|█▍        | 57395/400000 [00:07<00:42, 7986.88it/s] 15%|█▍        | 58195/400000 [00:07<00:43, 7914.23it/s] 15%|█▍        | 59007/400000 [00:07<00:42, 7974.11it/s] 15%|█▍        | 59806/400000 [00:07<00:42, 7977.67it/s] 15%|█▌        | 60605/400000 [00:07<00:43, 7860.72it/s] 15%|█▌        | 61392/400000 [00:07<00:44, 7690.08it/s] 16%|█▌        | 62163/400000 [00:07<00:45, 7478.18it/s] 16%|█▌        | 62929/400000 [00:08<00:44, 7530.38it/s] 16%|█▌        | 63684/400000 [00:08<00:45, 7457.57it/s] 16%|█▌        | 64450/400000 [00:08<00:44, 7515.15it/s] 16%|█▋        | 65203/400000 [00:08<00:44, 7500.86it/s] 17%|█▋        | 66016/400000 [00:08<00:43, 7677.03it/s] 17%|█▋        | 66794/400000 [00:08<00:43, 7706.71it/s] 17%|█▋        | 67606/400000 [00:08<00:42, 7824.93it/s] 17%|█▋        | 68390/400000 [00:08<00:42, 7766.86it/s] 17%|█▋        | 69168/400000 [00:08<00:43, 7677.95it/s] 17%|█▋        | 69965/400000 [00:08<00:42, 7762.84it/s] 18%|█▊        | 70743/400000 [00:09<00:42, 7758.12it/s] 18%|█▊        | 71562/400000 [00:09<00:41, 7879.35it/s] 18%|█▊        | 72378/400000 [00:09<00:41, 7960.06it/s] 18%|█▊        | 73175/400000 [00:09<00:41, 7875.67it/s] 18%|█▊        | 73964/400000 [00:09<00:43, 7491.01it/s] 19%|█▊        | 74718/400000 [00:09<00:44, 7270.81it/s] 19%|█▉        | 75450/400000 [00:09<00:44, 7231.01it/s] 19%|█▉        | 76177/400000 [00:09<00:44, 7208.03it/s] 19%|█▉        | 76901/400000 [00:09<00:45, 7160.92it/s] 19%|█▉        | 77619/400000 [00:09<00:45, 7075.46it/s] 20%|█▉        | 78328/400000 [00:10<00:46, 6974.15it/s] 20%|█▉        | 79027/400000 [00:10<00:46, 6944.78it/s] 20%|█▉        | 79757/400000 [00:10<00:45, 7046.97it/s] 20%|██        | 80496/400000 [00:10<00:44, 7146.39it/s] 20%|██        | 81283/400000 [00:10<00:43, 7347.54it/s] 21%|██        | 82020/400000 [00:10<00:43, 7259.60it/s] 21%|██        | 82748/400000 [00:10<00:43, 7250.38it/s] 21%|██        | 83551/400000 [00:10<00:42, 7465.76it/s] 21%|██        | 84314/400000 [00:10<00:42, 7511.61it/s] 21%|██▏       | 85079/400000 [00:10<00:41, 7549.87it/s] 21%|██▏       | 85851/400000 [00:11<00:41, 7598.28it/s] 22%|██▏       | 86612/400000 [00:11<00:41, 7595.44it/s] 22%|██▏       | 87373/400000 [00:11<00:41, 7547.06it/s] 22%|██▏       | 88129/400000 [00:11<00:42, 7364.31it/s] 22%|██▏       | 88867/400000 [00:11<00:42, 7285.59it/s] 22%|██▏       | 89667/400000 [00:11<00:41, 7484.29it/s] 23%|██▎       | 90427/400000 [00:11<00:41, 7517.09it/s] 23%|██▎       | 91181/400000 [00:11<00:41, 7409.60it/s] 23%|██▎       | 91924/400000 [00:11<00:42, 7280.67it/s] 23%|██▎       | 92710/400000 [00:11<00:41, 7444.98it/s] 23%|██▎       | 93502/400000 [00:12<00:40, 7578.63it/s] 24%|██▎       | 94285/400000 [00:12<00:39, 7650.42it/s] 24%|██▍       | 95146/400000 [00:12<00:38, 7914.47it/s] 24%|██▍       | 95965/400000 [00:12<00:38, 7994.27it/s] 24%|██▍       | 96815/400000 [00:12<00:37, 8138.91it/s] 24%|██▍       | 97709/400000 [00:12<00:36, 8362.19it/s] 25%|██▍       | 98586/400000 [00:12<00:35, 8478.82it/s] 25%|██▍       | 99453/400000 [00:12<00:35, 8533.83it/s] 25%|██▌       | 100309/400000 [00:12<00:35, 8391.80it/s] 25%|██▌       | 101158/400000 [00:13<00:35, 8418.80it/s] 26%|██▌       | 102010/400000 [00:13<00:35, 8448.31it/s] 26%|██▌       | 102856/400000 [00:13<00:35, 8421.71it/s] 26%|██▌       | 103705/400000 [00:13<00:35, 8441.54it/s] 26%|██▌       | 104550/400000 [00:13<00:35, 8267.83it/s] 26%|██▋       | 105406/400000 [00:13<00:35, 8352.10it/s] 27%|██▋       | 106304/400000 [00:13<00:34, 8528.46it/s] 27%|██▋       | 107159/400000 [00:13<00:34, 8523.17it/s] 27%|██▋       | 108049/400000 [00:13<00:33, 8630.30it/s] 27%|██▋       | 108914/400000 [00:13<00:34, 8482.67it/s] 27%|██▋       | 109776/400000 [00:14<00:34, 8522.91it/s] 28%|██▊       | 110630/400000 [00:14<00:33, 8520.72it/s] 28%|██▊       | 111525/400000 [00:14<00:33, 8643.83it/s] 28%|██▊       | 112397/400000 [00:14<00:33, 8664.34it/s] 28%|██▊       | 113265/400000 [00:14<00:34, 8401.35it/s] 29%|██▊       | 114113/400000 [00:14<00:33, 8424.21it/s] 29%|██▉       | 115006/400000 [00:14<00:33, 8568.18it/s] 29%|██▉       | 115881/400000 [00:14<00:32, 8619.82it/s] 29%|██▉       | 116769/400000 [00:14<00:32, 8694.20it/s] 29%|██▉       | 117640/400000 [00:14<00:33, 8530.75it/s] 30%|██▉       | 118495/400000 [00:15<00:33, 8392.12it/s] 30%|██▉       | 119336/400000 [00:15<00:34, 8039.91it/s] 30%|███       | 120172/400000 [00:15<00:34, 8133.01it/s] 30%|███       | 121018/400000 [00:15<00:33, 8228.31it/s] 30%|███       | 121844/400000 [00:15<00:34, 8108.46it/s] 31%|███       | 122657/400000 [00:15<00:34, 8111.30it/s] 31%|███       | 123478/400000 [00:15<00:33, 8138.76it/s] 31%|███       | 124297/400000 [00:15<00:33, 8151.64it/s] 31%|███▏      | 125130/400000 [00:15<00:33, 8203.82it/s] 31%|███▏      | 125951/400000 [00:15<00:33, 8121.85it/s] 32%|███▏      | 126764/400000 [00:16<00:35, 7754.53it/s] 32%|███▏      | 127544/400000 [00:16<00:37, 7339.87it/s] 32%|███▏      | 128325/400000 [00:16<00:36, 7473.34it/s] 32%|███▏      | 129079/400000 [00:16<00:36, 7487.14it/s] 32%|███▏      | 129857/400000 [00:16<00:35, 7571.96it/s] 33%|███▎      | 130618/400000 [00:16<00:35, 7547.42it/s] 33%|███▎      | 131436/400000 [00:16<00:34, 7724.90it/s] 33%|███▎      | 132212/400000 [00:16<00:35, 7598.82it/s] 33%|███▎      | 132975/400000 [00:16<00:35, 7555.02it/s] 33%|███▎      | 133752/400000 [00:17<00:34, 7615.91it/s] 34%|███▎      | 134526/400000 [00:17<00:34, 7652.54it/s] 34%|███▍      | 135296/400000 [00:17<00:34, 7664.77it/s] 34%|███▍      | 136067/400000 [00:17<00:34, 7676.13it/s] 34%|███▍      | 136849/400000 [00:17<00:34, 7718.16it/s] 34%|███▍      | 137647/400000 [00:17<00:33, 7791.81it/s] 35%|███▍      | 138430/400000 [00:17<00:33, 7802.37it/s] 35%|███▍      | 139226/400000 [00:17<00:33, 7847.37it/s] 35%|███▌      | 140012/400000 [00:17<00:33, 7798.73it/s] 35%|███▌      | 140797/400000 [00:17<00:33, 7813.12it/s] 35%|███▌      | 141631/400000 [00:18<00:32, 7962.95it/s] 36%|███▌      | 142443/400000 [00:18<00:32, 8008.80it/s] 36%|███▌      | 143249/400000 [00:18<00:32, 8021.97it/s] 36%|███▌      | 144067/400000 [00:18<00:31, 8066.29it/s] 36%|███▌      | 144875/400000 [00:18<00:32, 7774.22it/s] 36%|███▋      | 145668/400000 [00:18<00:32, 7819.73it/s] 37%|███▋      | 146530/400000 [00:18<00:31, 8042.47it/s] 37%|███▋      | 147338/400000 [00:18<00:31, 7939.98it/s] 37%|███▋      | 148135/400000 [00:18<00:31, 7928.04it/s] 37%|███▋      | 148930/400000 [00:18<00:31, 7890.93it/s] 37%|███▋      | 149721/400000 [00:19<00:32, 7680.43it/s] 38%|███▊      | 150503/400000 [00:19<00:32, 7717.54it/s] 38%|███▊      | 151308/400000 [00:19<00:31, 7814.35it/s] 38%|███▊      | 152122/400000 [00:19<00:31, 7908.56it/s] 38%|███▊      | 152916/400000 [00:19<00:31, 7917.19it/s] 38%|███▊      | 153724/400000 [00:19<00:30, 7964.74it/s] 39%|███▊      | 154522/400000 [00:19<00:31, 7917.28it/s] 39%|███▉      | 155351/400000 [00:19<00:30, 8022.15it/s] 39%|███▉      | 156175/400000 [00:19<00:30, 8083.56it/s] 39%|███▉      | 156984/400000 [00:19<00:30, 8052.39it/s] 39%|███▉      | 157790/400000 [00:20<00:31, 7783.61it/s] 40%|███▉      | 158571/400000 [00:20<00:31, 7772.19it/s] 40%|███▉      | 159445/400000 [00:20<00:29, 8039.22it/s] 40%|████      | 160316/400000 [00:20<00:29, 8228.42it/s] 40%|████      | 161143/400000 [00:20<00:29, 8206.99it/s] 40%|████      | 161978/400000 [00:20<00:28, 8249.01it/s] 41%|████      | 162855/400000 [00:20<00:28, 8396.17it/s] 41%|████      | 163726/400000 [00:20<00:27, 8487.12it/s] 41%|████      | 164614/400000 [00:20<00:27, 8598.60it/s] 41%|████▏     | 165476/400000 [00:20<00:27, 8469.91it/s] 42%|████▏     | 166325/400000 [00:21<00:27, 8449.32it/s] 42%|████▏     | 167197/400000 [00:21<00:27, 8526.97it/s] 42%|████▏     | 168051/400000 [00:21<00:27, 8482.02it/s] 42%|████▏     | 168900/400000 [00:21<00:27, 8336.89it/s] 42%|████▏     | 169735/400000 [00:21<00:28, 8154.05it/s] 43%|████▎     | 170575/400000 [00:21<00:27, 8225.26it/s] 43%|████▎     | 171420/400000 [00:21<00:27, 8291.04it/s] 43%|████▎     | 172251/400000 [00:21<00:28, 8133.44it/s] 43%|████▎     | 173066/400000 [00:21<00:28, 8063.26it/s] 43%|████▎     | 173874/400000 [00:22<00:28, 7881.63it/s] 44%|████▎     | 174664/400000 [00:22<00:29, 7756.15it/s] 44%|████▍     | 175461/400000 [00:22<00:28, 7818.02it/s] 44%|████▍     | 176265/400000 [00:22<00:28, 7880.90it/s] 44%|████▍     | 177055/400000 [00:22<00:28, 7865.75it/s] 44%|████▍     | 177843/400000 [00:22<00:28, 7867.47it/s] 45%|████▍     | 178694/400000 [00:22<00:27, 8047.71it/s] 45%|████▍     | 179574/400000 [00:22<00:26, 8259.14it/s] 45%|████▌     | 180452/400000 [00:22<00:26, 8407.34it/s] 45%|████▌     | 181347/400000 [00:22<00:25, 8561.83it/s] 46%|████▌     | 182206/400000 [00:23<00:25, 8481.46it/s] 46%|████▌     | 183104/400000 [00:23<00:25, 8624.03it/s] 46%|████▌     | 184002/400000 [00:23<00:24, 8725.50it/s] 46%|████▌     | 184877/400000 [00:23<00:24, 8651.74it/s] 46%|████▋     | 185744/400000 [00:23<00:25, 8537.51it/s] 47%|████▋     | 186599/400000 [00:23<00:26, 8171.79it/s] 47%|████▋     | 187429/400000 [00:23<00:25, 8209.45it/s] 47%|████▋     | 188253/400000 [00:23<00:26, 8140.65it/s] 47%|████▋     | 189070/400000 [00:23<00:26, 7921.92it/s] 47%|████▋     | 189866/400000 [00:23<00:27, 7724.69it/s] 48%|████▊     | 190642/400000 [00:24<00:27, 7600.94it/s] 48%|████▊     | 191405/400000 [00:24<00:27, 7544.50it/s] 48%|████▊     | 192185/400000 [00:24<00:27, 7617.43it/s] 48%|████▊     | 192950/400000 [00:24<00:27, 7626.08it/s] 48%|████▊     | 193722/400000 [00:24<00:26, 7652.85it/s] 49%|████▊     | 194491/400000 [00:24<00:26, 7663.94it/s] 49%|████▉     | 195299/400000 [00:24<00:26, 7783.26it/s] 49%|████▉     | 196086/400000 [00:24<00:26, 7808.42it/s] 49%|████▉     | 196868/400000 [00:24<00:26, 7786.96it/s] 49%|████▉     | 197670/400000 [00:24<00:25, 7854.38it/s] 50%|████▉     | 198456/400000 [00:25<00:26, 7626.29it/s] 50%|████▉     | 199221/400000 [00:25<00:26, 7609.63it/s] 50%|█████     | 200044/400000 [00:25<00:25, 7783.79it/s] 50%|█████     | 200870/400000 [00:25<00:25, 7918.52it/s] 50%|█████     | 201707/400000 [00:25<00:24, 8046.73it/s] 51%|█████     | 202514/400000 [00:25<00:25, 7812.98it/s] 51%|█████     | 203299/400000 [00:25<00:25, 7766.00it/s] 51%|█████     | 204083/400000 [00:25<00:25, 7785.19it/s] 51%|█████     | 204891/400000 [00:25<00:24, 7869.63it/s] 51%|█████▏    | 205707/400000 [00:25<00:24, 7947.62it/s] 52%|█████▏    | 206503/400000 [00:26<00:24, 7921.10it/s] 52%|█████▏    | 207339/400000 [00:26<00:23, 8047.69it/s] 52%|█████▏    | 208145/400000 [00:26<00:24, 7971.05it/s] 52%|█████▏    | 208943/400000 [00:26<00:24, 7910.10it/s] 52%|█████▏    | 209735/400000 [00:26<00:24, 7817.91it/s] 53%|█████▎    | 210518/400000 [00:26<00:24, 7808.57it/s] 53%|█████▎    | 211300/400000 [00:26<00:24, 7717.48it/s] 53%|█████▎    | 212073/400000 [00:26<00:24, 7521.97it/s] 53%|█████▎    | 212905/400000 [00:26<00:24, 7740.87it/s] 53%|█████▎    | 213708/400000 [00:27<00:23, 7821.72it/s] 54%|█████▎    | 214506/400000 [00:27<00:23, 7865.84it/s] 54%|█████▍    | 215295/400000 [00:27<00:23, 7836.72it/s] 54%|█████▍    | 216083/400000 [00:27<00:23, 7847.12it/s] 54%|█████▍    | 216894/400000 [00:27<00:23, 7923.43it/s] 54%|█████▍    | 217688/400000 [00:27<00:23, 7833.61it/s] 55%|█████▍    | 218473/400000 [00:27<00:23, 7768.98it/s] 55%|█████▍    | 219251/400000 [00:27<00:23, 7718.33it/s] 55%|█████▌    | 220029/400000 [00:27<00:23, 7736.38it/s] 55%|█████▌    | 220804/400000 [00:27<00:23, 7665.06it/s] 55%|█████▌    | 221571/400000 [00:28<00:23, 7615.60it/s] 56%|█████▌    | 222333/400000 [00:28<00:23, 7408.41it/s] 56%|█████▌    | 223098/400000 [00:28<00:23, 7479.09it/s] 56%|█████▌    | 223864/400000 [00:28<00:23, 7529.55it/s] 56%|█████▌    | 224618/400000 [00:28<00:23, 7488.99it/s] 56%|█████▋    | 225368/400000 [00:28<00:23, 7395.36it/s] 57%|█████▋    | 226160/400000 [00:28<00:23, 7543.09it/s] 57%|█████▋    | 226926/400000 [00:28<00:22, 7577.68it/s] 57%|█████▋    | 227711/400000 [00:28<00:22, 7656.26it/s] 57%|█████▋    | 228511/400000 [00:28<00:22, 7756.04it/s] 57%|█████▋    | 229313/400000 [00:29<00:21, 7830.31it/s] 58%|█████▊    | 230146/400000 [00:29<00:21, 7970.71it/s] 58%|█████▊    | 230948/400000 [00:29<00:21, 7983.64it/s] 58%|█████▊    | 231748/400000 [00:29<00:21, 7953.38it/s] 58%|█████▊    | 232584/400000 [00:29<00:20, 8069.85it/s] 58%|█████▊    | 233392/400000 [00:29<00:20, 8003.08it/s] 59%|█████▊    | 234233/400000 [00:29<00:20, 8118.90it/s] 59%|█████▉    | 235055/400000 [00:29<00:20, 8147.04it/s] 59%|█████▉    | 235871/400000 [00:29<00:20, 8071.75it/s] 59%|█████▉    | 236688/400000 [00:29<00:20, 8099.61it/s] 59%|█████▉    | 237499/400000 [00:30<00:20, 7974.14it/s] 60%|█████▉    | 238298/400000 [00:30<00:20, 7936.55it/s] 60%|█████▉    | 239131/400000 [00:30<00:19, 8049.76it/s] 60%|█████▉    | 239995/400000 [00:30<00:19, 8217.80it/s] 60%|██████    | 240855/400000 [00:30<00:19, 8327.72it/s] 60%|██████    | 241690/400000 [00:30<00:19, 8282.04it/s] 61%|██████    | 242558/400000 [00:30<00:18, 8396.04it/s] 61%|██████    | 243413/400000 [00:30<00:18, 8441.11it/s] 61%|██████    | 244273/400000 [00:30<00:18, 8487.93it/s] 61%|██████▏   | 245123/400000 [00:30<00:18, 8324.67it/s] 61%|██████▏   | 245957/400000 [00:31<00:18, 8140.54it/s] 62%|██████▏   | 246816/400000 [00:31<00:18, 8270.31it/s] 62%|██████▏   | 247673/400000 [00:31<00:18, 8356.95it/s] 62%|██████▏   | 248511/400000 [00:31<00:18, 8320.50it/s] 62%|██████▏   | 249345/400000 [00:31<00:18, 8193.27it/s] 63%|██████▎   | 250166/400000 [00:31<00:18, 8115.15it/s] 63%|██████▎   | 251045/400000 [00:31<00:17, 8304.58it/s] 63%|██████▎   | 251878/400000 [00:31<00:17, 8233.82it/s] 63%|██████▎   | 252703/400000 [00:31<00:18, 8127.10it/s] 63%|██████▎   | 253566/400000 [00:32<00:17, 8271.70it/s] 64%|██████▎   | 254395/400000 [00:32<00:17, 8206.17it/s] 64%|██████▍   | 255263/400000 [00:32<00:17, 8342.53it/s] 64%|██████▍   | 256125/400000 [00:32<00:17, 8423.11it/s] 64%|██████▍   | 256969/400000 [00:32<00:17, 8405.13it/s] 64%|██████▍   | 257818/400000 [00:32<00:16, 8427.25it/s] 65%|██████▍   | 258662/400000 [00:32<00:17, 8284.72it/s] 65%|██████▍   | 259492/400000 [00:32<00:17, 8053.16it/s] 65%|██████▌   | 260300/400000 [00:32<00:17, 7828.87it/s] 65%|██████▌   | 261092/400000 [00:32<00:17, 7855.04it/s] 65%|██████▌   | 261880/400000 [00:33<00:17, 7813.49it/s] 66%|██████▌   | 262663/400000 [00:33<00:17, 7746.92it/s] 66%|██████▌   | 263470/400000 [00:33<00:17, 7841.02it/s] 66%|██████▌   | 264257/400000 [00:33<00:17, 7849.14it/s] 66%|██████▋   | 265043/400000 [00:33<00:17, 7734.12it/s] 66%|██████▋   | 265818/400000 [00:33<00:17, 7627.68it/s] 67%|██████▋   | 266582/400000 [00:33<00:17, 7565.82it/s] 67%|██████▋   | 267373/400000 [00:33<00:17, 7664.35it/s] 67%|██████▋   | 268176/400000 [00:33<00:16, 7769.44it/s] 67%|██████▋   | 268959/400000 [00:33<00:16, 7785.64it/s] 67%|██████▋   | 269853/400000 [00:34<00:16, 8098.74it/s] 68%|██████▊   | 270667/400000 [00:34<00:16, 8080.75it/s] 68%|██████▊   | 271478/400000 [00:34<00:16, 7892.77it/s] 68%|██████▊   | 272271/400000 [00:34<00:16, 7804.83it/s] 68%|██████▊   | 273065/400000 [00:34<00:16, 7844.29it/s] 68%|██████▊   | 273852/400000 [00:34<00:16, 7546.08it/s] 69%|██████▊   | 274611/400000 [00:34<00:16, 7465.48it/s] 69%|██████▉   | 275376/400000 [00:34<00:16, 7519.05it/s] 69%|██████▉   | 276149/400000 [00:34<00:16, 7579.72it/s] 69%|██████▉   | 276929/400000 [00:34<00:16, 7643.38it/s] 69%|██████▉   | 277695/400000 [00:35<00:16, 7517.67it/s] 70%|██████▉   | 278461/400000 [00:35<00:16, 7559.54it/s] 70%|██████▉   | 279249/400000 [00:35<00:15, 7652.90it/s] 70%|███████   | 280018/400000 [00:35<00:15, 7663.29it/s] 70%|███████   | 280785/400000 [00:35<00:15, 7650.35it/s] 70%|███████   | 281551/400000 [00:35<00:15, 7626.32it/s] 71%|███████   | 282342/400000 [00:35<00:15, 7707.11it/s] 71%|███████   | 283223/400000 [00:35<00:14, 8006.66it/s] 71%|███████   | 284072/400000 [00:35<00:14, 8145.19it/s] 71%|███████   | 284890/400000 [00:35<00:14, 8008.48it/s] 71%|███████▏  | 285694/400000 [00:36<00:14, 7841.32it/s] 72%|███████▏  | 286485/400000 [00:36<00:14, 7861.70it/s] 72%|███████▏  | 287330/400000 [00:36<00:14, 8026.89it/s] 72%|███████▏  | 288191/400000 [00:36<00:13, 8192.38it/s] 72%|███████▏  | 289040/400000 [00:36<00:13, 8278.38it/s] 72%|███████▏  | 289891/400000 [00:36<00:13, 8345.16it/s] 73%|███████▎  | 290727/400000 [00:36<00:13, 8340.84it/s] 73%|███████▎  | 291563/400000 [00:36<00:13, 8329.11it/s] 73%|███████▎  | 292414/400000 [00:36<00:12, 8381.61it/s] 73%|███████▎  | 293311/400000 [00:37<00:12, 8549.04it/s] 74%|███████▎  | 294192/400000 [00:37<00:12, 8625.47it/s] 74%|███████▍  | 295056/400000 [00:37<00:12, 8491.13it/s] 74%|███████▍  | 295914/400000 [00:37<00:12, 8516.19it/s] 74%|███████▍  | 296767/400000 [00:37<00:12, 8468.96it/s] 74%|███████▍  | 297615/400000 [00:37<00:12, 8448.21it/s] 75%|███████▍  | 298475/400000 [00:37<00:11, 8490.79it/s] 75%|███████▍  | 299325/400000 [00:37<00:12, 8353.52it/s] 75%|███████▌  | 300203/400000 [00:37<00:11, 8476.14it/s] 75%|███████▌  | 301087/400000 [00:37<00:11, 8581.22it/s] 75%|███████▌  | 301969/400000 [00:38<00:11, 8651.24it/s] 76%|███████▌  | 302835/400000 [00:38<00:11, 8617.75it/s] 76%|███████▌  | 303698/400000 [00:38<00:11, 8521.88it/s] 76%|███████▌  | 304553/400000 [00:38<00:11, 8527.16it/s] 76%|███████▋  | 305407/400000 [00:38<00:11, 8503.79it/s] 77%|███████▋  | 306258/400000 [00:38<00:11, 8326.21it/s] 77%|███████▋  | 307117/400000 [00:38<00:11, 8401.30it/s] 77%|███████▋  | 307959/400000 [00:38<00:11, 8322.05it/s] 77%|███████▋  | 308835/400000 [00:38<00:10, 8447.34it/s] 77%|███████▋  | 309721/400000 [00:38<00:10, 8565.92it/s] 78%|███████▊  | 310579/400000 [00:39<00:10, 8522.05it/s] 78%|███████▊  | 311462/400000 [00:39<00:10, 8611.15it/s] 78%|███████▊  | 312324/400000 [00:39<00:10, 8471.45it/s] 78%|███████▊  | 313185/400000 [00:39<00:10, 8509.86it/s] 79%|███████▊  | 314037/400000 [00:39<00:10, 8399.53it/s] 79%|███████▊  | 314878/400000 [00:39<00:10, 8299.02it/s] 79%|███████▉  | 315709/400000 [00:39<00:10, 8137.68it/s] 79%|███████▉  | 316531/400000 [00:39<00:10, 8162.11it/s] 79%|███████▉  | 317409/400000 [00:39<00:09, 8336.26it/s] 80%|███████▉  | 318295/400000 [00:39<00:09, 8485.64it/s] 80%|███████▉  | 319146/400000 [00:40<00:09, 8209.14it/s] 80%|███████▉  | 319971/400000 [00:40<00:10, 7878.02it/s] 80%|████████  | 320793/400000 [00:40<00:09, 7974.24it/s] 80%|████████  | 321654/400000 [00:40<00:09, 8152.80it/s] 81%|████████  | 322510/400000 [00:40<00:09, 8268.99it/s] 81%|████████  | 323392/400000 [00:40<00:09, 8425.83it/s] 81%|████████  | 324238/400000 [00:40<00:09, 8262.69it/s] 81%|████████▏ | 325070/400000 [00:40<00:09, 8276.50it/s] 81%|████████▏ | 325904/400000 [00:40<00:08, 8292.91it/s] 82%|████████▏ | 326782/400000 [00:40<00:08, 8433.22it/s] 82%|████████▏ | 327648/400000 [00:41<00:08, 8499.97it/s] 82%|████████▏ | 328500/400000 [00:41<00:08, 8290.21it/s] 82%|████████▏ | 329331/400000 [00:41<00:08, 8212.08it/s] 83%|████████▎ | 330173/400000 [00:41<00:08, 8271.69it/s] 83%|████████▎ | 331048/400000 [00:41<00:08, 8408.44it/s] 83%|████████▎ | 331891/400000 [00:41<00:08, 8411.50it/s] 83%|████████▎ | 332734/400000 [00:41<00:08, 8011.87it/s] 83%|████████▎ | 333540/400000 [00:41<00:08, 7854.31it/s] 84%|████████▎ | 334352/400000 [00:41<00:08, 7930.36it/s] 84%|████████▍ | 335194/400000 [00:42<00:08, 8069.34it/s] 84%|████████▍ | 336040/400000 [00:42<00:07, 8181.00it/s] 84%|████████▍ | 336861/400000 [00:42<00:07, 8071.02it/s] 84%|████████▍ | 337718/400000 [00:42<00:07, 8211.56it/s] 85%|████████▍ | 338542/400000 [00:42<00:07, 8202.52it/s] 85%|████████▍ | 339364/400000 [00:42<00:07, 8183.90it/s] 85%|████████▌ | 340184/400000 [00:42<00:07, 8112.63it/s] 85%|████████▌ | 340997/400000 [00:42<00:07, 8070.38it/s] 85%|████████▌ | 341862/400000 [00:42<00:07, 8234.02it/s] 86%|████████▌ | 342719/400000 [00:42<00:06, 8331.18it/s] 86%|████████▌ | 343572/400000 [00:43<00:06, 8389.62it/s] 86%|████████▌ | 344412/400000 [00:43<00:06, 8183.56it/s] 86%|████████▋ | 345250/400000 [00:43<00:06, 8241.13it/s] 87%|████████▋ | 346105/400000 [00:43<00:06, 8329.31it/s] 87%|████████▋ | 346940/400000 [00:43<00:06, 8273.03it/s] 87%|████████▋ | 347769/400000 [00:43<00:06, 8212.11it/s] 87%|████████▋ | 348591/400000 [00:43<00:06, 8175.25it/s] 87%|████████▋ | 349410/400000 [00:43<00:06, 7980.19it/s] 88%|████████▊ | 350271/400000 [00:43<00:06, 8157.81it/s] 88%|████████▊ | 351148/400000 [00:43<00:05, 8330.73it/s] 88%|████████▊ | 352038/400000 [00:44<00:05, 8492.80it/s] 88%|████████▊ | 352890/400000 [00:44<00:05, 8458.16it/s] 88%|████████▊ | 353738/400000 [00:44<00:05, 8176.37it/s] 89%|████████▊ | 354559/400000 [00:44<00:05, 8016.74it/s] 89%|████████▉ | 355415/400000 [00:44<00:05, 8172.02it/s] 89%|████████▉ | 356235/400000 [00:44<00:05, 8049.09it/s] 89%|████████▉ | 357051/400000 [00:44<00:05, 8078.11it/s] 89%|████████▉ | 357894/400000 [00:44<00:05, 8179.26it/s] 90%|████████▉ | 358747/400000 [00:44<00:04, 8278.82it/s] 90%|████████▉ | 359577/400000 [00:44<00:04, 8110.21it/s] 90%|█████████ | 360390/400000 [00:45<00:04, 8059.41it/s] 90%|█████████ | 361198/400000 [00:45<00:04, 7872.87it/s] 91%|█████████ | 362059/400000 [00:45<00:04, 8079.87it/s] 91%|█████████ | 362870/400000 [00:45<00:04, 7947.57it/s] 91%|█████████ | 363718/400000 [00:45<00:04, 8099.60it/s] 91%|█████████ | 364591/400000 [00:45<00:04, 8277.27it/s] 91%|█████████▏| 365422/400000 [00:45<00:04, 8100.23it/s] 92%|█████████▏| 366256/400000 [00:45<00:04, 8170.25it/s] 92%|█████████▏| 367076/400000 [00:45<00:04, 8122.46it/s] 92%|█████████▏| 367900/400000 [00:46<00:03, 8156.85it/s] 92%|█████████▏| 368787/400000 [00:46<00:03, 8357.36it/s] 92%|█████████▏| 369625/400000 [00:46<00:03, 8262.18it/s] 93%|█████████▎| 370475/400000 [00:46<00:03, 8330.19it/s] 93%|█████████▎| 371328/400000 [00:46<00:03, 8389.09it/s] 93%|█████████▎| 372172/400000 [00:46<00:03, 8404.30it/s] 93%|█████████▎| 373032/400000 [00:46<00:03, 8461.09it/s] 93%|█████████▎| 373879/400000 [00:46<00:03, 8353.18it/s] 94%|█████████▎| 374716/400000 [00:46<00:03, 8201.65it/s] 94%|█████████▍| 375610/400000 [00:46<00:02, 8409.15it/s] 94%|█████████▍| 376503/400000 [00:47<00:02, 8558.85it/s] 94%|█████████▍| 377361/400000 [00:47<00:02, 8558.89it/s] 95%|█████████▍| 378219/400000 [00:47<00:02, 8442.36it/s] 95%|█████████▍| 379089/400000 [00:47<00:02, 8517.83it/s] 95%|█████████▍| 379975/400000 [00:47<00:02, 8617.02it/s] 95%|█████████▌| 380841/400000 [00:47<00:02, 8628.47it/s] 95%|█████████▌| 381705/400000 [00:47<00:02, 8571.11it/s] 96%|█████████▌| 382563/400000 [00:47<00:02, 8383.49it/s] 96%|█████████▌| 383427/400000 [00:47<00:01, 8458.51it/s] 96%|█████████▌| 384306/400000 [00:47<00:01, 8553.32it/s] 96%|█████████▋| 385163/400000 [00:48<00:01, 8542.35it/s] 97%|█████████▋| 386038/400000 [00:48<00:01, 8602.76it/s] 97%|█████████▋| 386899/400000 [00:48<00:01, 8447.58it/s] 97%|█████████▋| 387745/400000 [00:48<00:01, 8395.88it/s] 97%|█████████▋| 388586/400000 [00:48<00:01, 8331.72it/s] 97%|█████████▋| 389463/400000 [00:48<00:01, 8457.48it/s] 98%|█████████▊| 390340/400000 [00:48<00:01, 8548.12it/s] 98%|█████████▊| 391196/400000 [00:48<00:01, 8385.53it/s] 98%|█████████▊| 392077/400000 [00:48<00:00, 8507.18it/s] 98%|█████████▊| 392938/400000 [00:48<00:00, 8536.00it/s] 98%|█████████▊| 393826/400000 [00:49<00:00, 8634.98it/s] 99%|█████████▊| 394691/400000 [00:49<00:00, 8622.12it/s] 99%|█████████▉| 395554/400000 [00:49<00:00, 8512.76it/s] 99%|█████████▉| 396418/400000 [00:49<00:00, 8548.30it/s] 99%|█████████▉| 397290/400000 [00:49<00:00, 8598.75it/s]100%|█████████▉| 398156/400000 [00:49<00:00, 8616.23it/s]100%|█████████▉| 399025/400000 [00:49<00:00, 8636.11it/s]100%|█████████▉| 399889/400000 [00:49<00:00, 8471.13it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8032.34it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f69aecb74e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011651240022738433 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011511792307314665 	 Accuracy: 48

  model saves at 48% accuracy 

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

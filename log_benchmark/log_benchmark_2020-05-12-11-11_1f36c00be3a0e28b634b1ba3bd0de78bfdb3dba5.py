
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f8721b1cfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 11:11:51.745773
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 11:11:51.749577
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 11:11:51.752692
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 11:11:51.755880
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f872dbf5470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352443.7188
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 227529.3750
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 129581.3047
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 66157.0625
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 35014.5273
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 20756.8125
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 13613.6211
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 9532.5713
Epoch 9/10

1/1 [==============================] - 0s 92ms/step - loss: 7228.0181
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 5657.5903

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.09788406e-02 -3.32013577e-01  3.43504846e-01  4.57881689e-01
  -4.71514195e-01  2.26144600e+00 -8.44808340e-01 -2.58454037e+00
  -1.28618526e+00  8.95671248e-01  2.06767178e+00 -1.27308369e+00
   1.94390225e+00  3.68913680e-01 -1.88382006e+00  5.98611593e-01
  -1.77987432e+00  3.00174505e-01 -8.81877124e-01 -1.85440853e-02
  -1.48729157e+00  3.28856647e-01 -1.45604730e+00 -9.27001238e-01
   4.12695736e-01  5.94422579e-01  4.29349214e-01  1.83456802e+00
  -1.94385618e-01 -1.09393764e+00 -2.35025120e+00  1.43123460e+00
  -6.13826752e-01  8.35384846e-01  1.92383695e+00  6.47293806e-01
  -4.38251078e-01 -2.83002675e-01 -4.47259039e-01 -4.60126340e-01
   1.17953134e+00 -5.68288803e-01  2.50406682e-01  8.90024364e-01
   9.39356148e-01 -2.12254912e-01 -1.93651631e-01 -1.00523043e+00
   1.05177581e-01 -1.28649962e+00  2.04540539e+00  3.32801640e-01
  -1.51529932e+00  7.94352293e-01  7.92762578e-01  1.26237988e+00
  -3.15057039e-01 -2.13275456e+00 -1.77980351e+00  1.80875421e-01
  -1.69993192e-03  9.91588116e+00  8.64513779e+00  8.70642662e+00
   9.47381115e+00  9.31053257e+00  8.58665371e+00  1.06877861e+01
   8.75184059e+00  9.81650543e+00  7.81292439e+00  6.92761946e+00
   7.63986731e+00  9.50587273e+00  8.85794640e+00  1.02475128e+01
   8.46507168e+00  8.18844032e+00  9.27097607e+00  9.82645798e+00
   9.64923096e+00  7.16180849e+00  1.10354872e+01  8.64984703e+00
   9.70274639e+00  8.30513763e+00  8.30057812e+00  8.18482399e+00
   8.65114975e+00  9.93648624e+00  6.55683517e+00  8.27615643e+00
   9.71204185e+00  7.40237045e+00  1.01993351e+01  8.13543320e+00
   8.34020996e+00  7.79130459e+00  8.64031315e+00  6.98855495e+00
   8.06273365e+00  7.88618231e+00  1.01945801e+01  8.43438339e+00
   8.79418564e+00  8.09755611e+00  1.03438587e+01  9.90917778e+00
   9.04012489e+00  7.73637199e+00  8.76737022e+00  8.84417725e+00
   9.51464558e+00  8.54659653e+00  9.13933945e+00  6.57533264e+00
   7.97166061e+00  8.43178272e+00  8.72216034e+00  8.85083199e+00
   1.32503188e+00 -3.15485001e-02 -2.40421295e-01  3.97939205e-01
   1.56407833e+00  4.57409620e-01  1.24038085e-01 -6.69409335e-01
   8.25666964e-01  1.24566209e+00  1.15155661e+00  2.03733826e+00
  -5.55316746e-01  1.57004833e+00 -1.43671960e-01 -3.14051241e-01
   1.54596829e+00  3.99421871e-01 -1.11470270e+00  6.04297638e-01
   5.67670047e-01 -1.27107501e+00  3.17663074e-01  8.79211277e-02
  -3.19579691e-01  1.60143185e+00 -8.75690639e-01  1.84484565e+00
  -3.47716212e-01 -1.81231737e+00 -2.55998993e+00  1.43997538e+00
   2.06581950e-01  1.94069660e+00 -1.02958608e+00  1.85248899e+00
   1.74131989e-01 -5.61963916e-02  4.38960612e-01 -4.31624919e-01
  -1.27243030e+00  1.40197682e+00  8.37672293e-01 -1.35062826e+00
   1.47165644e+00 -1.68636072e+00  4.53245491e-01 -1.67829108e+00
  -3.03128481e-01 -1.53721035e+00 -9.08694685e-01 -5.07483006e-01
  -5.00056744e-01  1.68994498e+00  9.99564230e-01 -6.58479035e-01
  -5.92236757e-01 -1.11410141e+00  1.20990980e+00 -1.92834616e+00
   1.23529387e+00  2.75647759e-01  1.16273069e+00  4.09447432e-01
   1.06171465e+00  1.46473825e-01  1.64147258e+00  1.47031736e+00
   3.04339981e+00  1.26646161e+00  1.29373777e+00  9.41814840e-01
   5.50024688e-01  4.51208055e-01  3.23823690e-01  9.58784223e-02
   8.68924856e-01  2.74010003e-01  5.20699263e-01  1.23731923e+00
   3.57987165e-01  2.72242641e+00  2.82067299e-01  2.52133727e-01
   4.69510138e-01  1.64019763e-01  1.06150770e+00  2.08701324e+00
   8.63242090e-01  8.44867051e-01  1.72080851e+00  1.49650323e+00
   3.30809057e-01  7.68176794e-01  1.92601180e+00  1.21630692e+00
   7.32672453e-01  8.87154996e-01  5.75894117e-01  1.33476257e-01
   2.00295162e+00  6.41815543e-01  6.63003445e-01  1.65547037e+00
   9.03831720e-01  3.40804994e-01  1.30463088e+00  2.63266802e+00
   6.63297176e-01  4.58660841e-01  2.47975349e+00  1.24902844e-01
   3.90623152e-01  4.71308351e-01  2.70319843e+00  1.19212723e+00
   2.01816320e-01  3.80719900e-01  9.09660757e-01  1.29999042e-01
   6.32641912e-02  1.00832310e+01  9.40609837e+00  7.48172426e+00
   9.90825176e+00  9.29317474e+00  9.46869087e+00  1.13474693e+01
   8.49969864e+00  8.99746609e+00  7.88578892e+00  7.14375401e+00
   8.55660629e+00  9.61205482e+00  9.79849720e+00  8.80327034e+00
   1.00956316e+01  7.37650728e+00  8.04940414e+00  8.77458668e+00
   8.35395145e+00  1.07372313e+01  1.01176186e+01  8.33792877e+00
   9.22942066e+00  8.13687992e+00  9.60667896e+00  9.17262173e+00
   7.56663227e+00  9.42126846e+00  8.89052010e+00  7.89166546e+00
   7.70974493e+00  8.98942852e+00  9.23630238e+00  1.06042786e+01
   9.86490440e+00  1.05115976e+01  8.32959938e+00  8.16914463e+00
   9.04608440e+00  8.93891907e+00  9.59712124e+00  9.32430744e+00
   1.08377266e+01  1.09549866e+01  7.90397930e+00  9.92668533e+00
   7.65538692e+00  9.40385437e+00  8.69322777e+00  8.72390461e+00
   1.04957905e+01  9.57868195e+00  9.66919994e+00  7.35668516e+00
   9.29835606e+00  8.94006062e+00  8.69064331e+00  9.85892391e+00
   5.36362290e-01  2.18476820e+00  1.27154303e+00  1.54813457e+00
   1.52216196e+00  3.23998451e-01  1.02236867e-01  2.42898226e-01
   2.65386295e+00  2.30802488e+00  3.06696749e+00  1.16103351e+00
   2.12439346e+00  1.12812638e-01  1.28611875e+00  1.86738551e+00
   8.15120220e-01  1.59372211e-01  4.55648422e-01  1.51273751e+00
   2.36266375e+00  7.19412684e-01  1.45206308e+00  1.66526890e+00
   1.73608279e+00  4.66908574e-01  2.50361502e-01  6.22146130e-01
   5.24531007e-01  8.57934177e-01  4.30766344e-01  8.53207529e-01
   1.13164282e+00  2.86346912e-01  3.97735357e-01  9.34370518e-01
   1.86411679e+00  1.98718548e+00  7.46616900e-01  1.59599948e+00
   3.80140543e-01  2.52853775e+00  7.08585978e-02  5.57443857e-01
   1.97711229e+00  2.83505201e-01  7.69381404e-01  3.42286706e-01
   3.68592024e-01  2.12642503e+00  7.82529116e-01  1.70801425e+00
   1.93744755e+00  6.89016223e-01  1.80162561e+00  2.50898147e+00
   2.47994959e-01  1.93888259e+00  3.46278310e-01  1.60562944e+00
  -4.36029339e+00  6.00088406e+00 -8.86427402e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 11:12:02.490467
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6136
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 11:12:02.500588
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8600.56
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 11:12:02.503973
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0409
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 11:12:02.507539
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.241
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140218006659592
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140217065185520
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140217065186024
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140217065186528
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140217065187032
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140217065187536

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f870d80afd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.587729
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.555475
grad_step = 000002, loss = 0.525828
grad_step = 000003, loss = 0.494573
grad_step = 000004, loss = 0.462038
grad_step = 000005, loss = 0.430978
grad_step = 000006, loss = 0.412185
grad_step = 000007, loss = 0.391943
grad_step = 000008, loss = 0.365660
grad_step = 000009, loss = 0.349666
grad_step = 000010, loss = 0.338477
grad_step = 000011, loss = 0.323558
grad_step = 000012, loss = 0.308953
grad_step = 000013, loss = 0.298142
grad_step = 000014, loss = 0.289739
grad_step = 000015, loss = 0.281702
grad_step = 000016, loss = 0.272715
grad_step = 000017, loss = 0.262877
grad_step = 000018, loss = 0.252725
grad_step = 000019, loss = 0.242706
grad_step = 000020, loss = 0.232561
grad_step = 000021, loss = 0.222120
grad_step = 000022, loss = 0.211875
grad_step = 000023, loss = 0.202569
grad_step = 000024, loss = 0.194419
grad_step = 000025, loss = 0.187104
grad_step = 000026, loss = 0.180165
grad_step = 000027, loss = 0.173262
grad_step = 000028, loss = 0.166240
grad_step = 000029, loss = 0.159060
grad_step = 000030, loss = 0.151725
grad_step = 000031, loss = 0.144369
grad_step = 000032, loss = 0.137267
grad_step = 000033, loss = 0.130641
grad_step = 000034, loss = 0.124523
grad_step = 000035, loss = 0.118816
grad_step = 000036, loss = 0.113402
grad_step = 000037, loss = 0.108165
grad_step = 000038, loss = 0.102986
grad_step = 000039, loss = 0.097806
grad_step = 000040, loss = 0.092677
grad_step = 000041, loss = 0.087713
grad_step = 000042, loss = 0.082977
grad_step = 000043, loss = 0.078495
grad_step = 000044, loss = 0.074273
grad_step = 000045, loss = 0.070282
grad_step = 000046, loss = 0.066459
grad_step = 000047, loss = 0.062760
grad_step = 000048, loss = 0.059193
grad_step = 000049, loss = 0.055763
grad_step = 000050, loss = 0.052459
grad_step = 000051, loss = 0.049293
grad_step = 000052, loss = 0.046290
grad_step = 000053, loss = 0.043447
grad_step = 000054, loss = 0.040772
grad_step = 000055, loss = 0.038261
grad_step = 000056, loss = 0.035882
grad_step = 000057, loss = 0.033608
grad_step = 000058, loss = 0.031428
grad_step = 000059, loss = 0.029350
grad_step = 000060, loss = 0.027379
grad_step = 000061, loss = 0.025524
grad_step = 000062, loss = 0.023795
grad_step = 000063, loss = 0.022179
grad_step = 000064, loss = 0.020658
grad_step = 000065, loss = 0.019223
grad_step = 000066, loss = 0.017863
grad_step = 000067, loss = 0.016575
grad_step = 000068, loss = 0.015367
grad_step = 000069, loss = 0.014241
grad_step = 000070, loss = 0.013195
grad_step = 000071, loss = 0.012225
grad_step = 000072, loss = 0.011326
grad_step = 000073, loss = 0.010487
grad_step = 000074, loss = 0.009706
grad_step = 000075, loss = 0.008980
grad_step = 000076, loss = 0.008306
grad_step = 000077, loss = 0.007687
grad_step = 000078, loss = 0.007121
grad_step = 000079, loss = 0.006605
grad_step = 000080, loss = 0.006134
grad_step = 000081, loss = 0.005701
grad_step = 000082, loss = 0.005305
grad_step = 000083, loss = 0.004943
grad_step = 000084, loss = 0.004616
grad_step = 000085, loss = 0.004322
grad_step = 000086, loss = 0.004057
grad_step = 000087, loss = 0.003818
grad_step = 000088, loss = 0.003603
grad_step = 000089, loss = 0.003410
grad_step = 000090, loss = 0.003239
grad_step = 000091, loss = 0.003086
grad_step = 000092, loss = 0.002949
grad_step = 000093, loss = 0.002828
grad_step = 000094, loss = 0.002721
grad_step = 000095, loss = 0.002628
grad_step = 000096, loss = 0.002547
grad_step = 000097, loss = 0.002476
grad_step = 000098, loss = 0.002413
grad_step = 000099, loss = 0.002359
grad_step = 000100, loss = 0.002312
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002272
grad_step = 000102, loss = 0.002238
grad_step = 000103, loss = 0.002209
grad_step = 000104, loss = 0.002183
grad_step = 000105, loss = 0.002162
grad_step = 000106, loss = 0.002143
grad_step = 000107, loss = 0.002128
grad_step = 000108, loss = 0.002114
grad_step = 000109, loss = 0.002103
grad_step = 000110, loss = 0.002093
grad_step = 000111, loss = 0.002085
grad_step = 000112, loss = 0.002077
grad_step = 000113, loss = 0.002070
grad_step = 000114, loss = 0.002064
grad_step = 000115, loss = 0.002059
grad_step = 000116, loss = 0.002053
grad_step = 000117, loss = 0.002048
grad_step = 000118, loss = 0.002043
grad_step = 000119, loss = 0.002039
grad_step = 000120, loss = 0.002034
grad_step = 000121, loss = 0.002029
grad_step = 000122, loss = 0.002025
grad_step = 000123, loss = 0.002020
grad_step = 000124, loss = 0.002015
grad_step = 000125, loss = 0.002010
grad_step = 000126, loss = 0.002005
grad_step = 000127, loss = 0.002000
grad_step = 000128, loss = 0.001995
grad_step = 000129, loss = 0.001990
grad_step = 000130, loss = 0.001985
grad_step = 000131, loss = 0.001980
grad_step = 000132, loss = 0.001976
grad_step = 000133, loss = 0.001970
grad_step = 000134, loss = 0.001964
grad_step = 000135, loss = 0.001960
grad_step = 000136, loss = 0.001955
grad_step = 000137, loss = 0.001949
grad_step = 000138, loss = 0.001943
grad_step = 000139, loss = 0.001939
grad_step = 000140, loss = 0.001934
grad_step = 000141, loss = 0.001929
grad_step = 000142, loss = 0.001923
grad_step = 000143, loss = 0.001918
grad_step = 000144, loss = 0.001914
grad_step = 000145, loss = 0.001909
grad_step = 000146, loss = 0.001904
grad_step = 000147, loss = 0.001899
grad_step = 000148, loss = 0.001893
grad_step = 000149, loss = 0.001887
grad_step = 000150, loss = 0.001882
grad_step = 000151, loss = 0.001878
grad_step = 000152, loss = 0.001874
grad_step = 000153, loss = 0.001872
grad_step = 000154, loss = 0.001872
grad_step = 000155, loss = 0.001869
grad_step = 000156, loss = 0.001859
grad_step = 000157, loss = 0.001848
grad_step = 000158, loss = 0.001846
grad_step = 000159, loss = 0.001847
grad_step = 000160, loss = 0.001841
grad_step = 000161, loss = 0.001830
grad_step = 000162, loss = 0.001824
grad_step = 000163, loss = 0.001823
grad_step = 000164, loss = 0.001821
grad_step = 000165, loss = 0.001814
grad_step = 000166, loss = 0.001805
grad_step = 000167, loss = 0.001799
grad_step = 000168, loss = 0.001797
grad_step = 000169, loss = 0.001796
grad_step = 000170, loss = 0.001791
grad_step = 000171, loss = 0.001784
grad_step = 000172, loss = 0.001776
grad_step = 000173, loss = 0.001770
grad_step = 000174, loss = 0.001765
grad_step = 000175, loss = 0.001762
grad_step = 000176, loss = 0.001760
grad_step = 000177, loss = 0.001759
grad_step = 000178, loss = 0.001760
grad_step = 000179, loss = 0.001760
grad_step = 000180, loss = 0.001759
grad_step = 000181, loss = 0.001746
grad_step = 000182, loss = 0.001731
grad_step = 000183, loss = 0.001721
grad_step = 000184, loss = 0.001719
grad_step = 000185, loss = 0.001721
grad_step = 000186, loss = 0.001716
grad_step = 000187, loss = 0.001706
grad_step = 000188, loss = 0.001696
grad_step = 000189, loss = 0.001691
grad_step = 000190, loss = 0.001690
grad_step = 000191, loss = 0.001687
grad_step = 000192, loss = 0.001682
grad_step = 000193, loss = 0.001672
grad_step = 000194, loss = 0.001664
grad_step = 000195, loss = 0.001657
grad_step = 000196, loss = 0.001653
grad_step = 000197, loss = 0.001650
grad_step = 000198, loss = 0.001645
grad_step = 000199, loss = 0.001639
grad_step = 000200, loss = 0.001631
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001623
grad_step = 000202, loss = 0.001616
grad_step = 000203, loss = 0.001610
grad_step = 000204, loss = 0.001610
grad_step = 000205, loss = 0.001624
grad_step = 000206, loss = 0.001647
grad_step = 000207, loss = 0.001629
grad_step = 000208, loss = 0.001589
grad_step = 000209, loss = 0.001594
grad_step = 000210, loss = 0.001620
grad_step = 000211, loss = 0.001611
grad_step = 000212, loss = 0.001594
grad_step = 000213, loss = 0.001617
grad_step = 000214, loss = 0.001644
grad_step = 000215, loss = 0.001634
grad_step = 000216, loss = 0.001577
grad_step = 000217, loss = 0.001551
grad_step = 000218, loss = 0.001568
grad_step = 000219, loss = 0.001591
grad_step = 000220, loss = 0.001589
grad_step = 000221, loss = 0.001556
grad_step = 000222, loss = 0.001531
grad_step = 000223, loss = 0.001533
grad_step = 000224, loss = 0.001550
grad_step = 000225, loss = 0.001557
grad_step = 000226, loss = 0.001545
grad_step = 000227, loss = 0.001534
grad_step = 000228, loss = 0.001524
grad_step = 000229, loss = 0.001511
grad_step = 000230, loss = 0.001504
grad_step = 000231, loss = 0.001507
grad_step = 000232, loss = 0.001519
grad_step = 000233, loss = 0.001535
grad_step = 000234, loss = 0.001548
grad_step = 000235, loss = 0.001572
grad_step = 000236, loss = 0.001579
grad_step = 000237, loss = 0.001559
grad_step = 000238, loss = 0.001518
grad_step = 000239, loss = 0.001486
grad_step = 000240, loss = 0.001478
grad_step = 000241, loss = 0.001487
grad_step = 000242, loss = 0.001508
grad_step = 000243, loss = 0.001548
grad_step = 000244, loss = 0.001608
grad_step = 000245, loss = 0.001593
grad_step = 000246, loss = 0.001545
grad_step = 000247, loss = 0.001480
grad_step = 000248, loss = 0.001483
grad_step = 000249, loss = 0.001485
grad_step = 000250, loss = 0.001536
grad_step = 000251, loss = 0.001573
grad_step = 000252, loss = 0.001533
grad_step = 000253, loss = 0.001503
grad_step = 000254, loss = 0.001459
grad_step = 000255, loss = 0.001530
grad_step = 000256, loss = 0.001523
grad_step = 000257, loss = 0.001530
grad_step = 000258, loss = 0.001483
grad_step = 000259, loss = 0.001464
grad_step = 000260, loss = 0.001471
grad_step = 000261, loss = 0.001485
grad_step = 000262, loss = 0.001509
grad_step = 000263, loss = 0.001466
grad_step = 000264, loss = 0.001457
grad_step = 000265, loss = 0.001434
grad_step = 000266, loss = 0.001451
grad_step = 000267, loss = 0.001459
grad_step = 000268, loss = 0.001453
grad_step = 000269, loss = 0.001448
grad_step = 000270, loss = 0.001424
grad_step = 000271, loss = 0.001428
grad_step = 000272, loss = 0.001424
grad_step = 000273, loss = 0.001432
grad_step = 000274, loss = 0.001436
grad_step = 000275, loss = 0.001428
grad_step = 000276, loss = 0.001426
grad_step = 000277, loss = 0.001413
grad_step = 000278, loss = 0.001408
grad_step = 000279, loss = 0.001407
grad_step = 000280, loss = 0.001401
grad_step = 000281, loss = 0.001403
grad_step = 000282, loss = 0.001406
grad_step = 000283, loss = 0.001408
grad_step = 000284, loss = 0.001421
grad_step = 000285, loss = 0.001448
grad_step = 000286, loss = 0.001495
grad_step = 000287, loss = 0.001510
grad_step = 000288, loss = 0.001520
grad_step = 000289, loss = 0.001476
grad_step = 000290, loss = 0.001407
grad_step = 000291, loss = 0.001375
grad_step = 000292, loss = 0.001404
grad_step = 000293, loss = 0.001430
grad_step = 000294, loss = 0.001478
grad_step = 000295, loss = 0.001554
grad_step = 000296, loss = 0.001438
grad_step = 000297, loss = 0.001451
grad_step = 000298, loss = 0.001426
grad_step = 000299, loss = 0.001517
grad_step = 000300, loss = 0.001529
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001386
grad_step = 000302, loss = 0.001413
grad_step = 000303, loss = 0.001393
grad_step = 000304, loss = 0.001571
grad_step = 000305, loss = 0.001530
grad_step = 000306, loss = 0.001509
grad_step = 000307, loss = 0.001349
grad_step = 000308, loss = 0.001529
grad_step = 000309, loss = 0.001410
grad_step = 000310, loss = 0.001424
grad_step = 000311, loss = 0.001397
grad_step = 000312, loss = 0.001426
grad_step = 000313, loss = 0.001392
grad_step = 000314, loss = 0.001372
grad_step = 000315, loss = 0.001338
grad_step = 000316, loss = 0.001388
grad_step = 000317, loss = 0.001340
grad_step = 000318, loss = 0.001344
grad_step = 000319, loss = 0.001306
grad_step = 000320, loss = 0.001329
grad_step = 000321, loss = 0.001325
grad_step = 000322, loss = 0.001299
grad_step = 000323, loss = 0.001299
grad_step = 000324, loss = 0.001279
grad_step = 000325, loss = 0.001301
grad_step = 000326, loss = 0.001272
grad_step = 000327, loss = 0.001275
grad_step = 000328, loss = 0.001257
grad_step = 000329, loss = 0.001251
grad_step = 000330, loss = 0.001259
grad_step = 000331, loss = 0.001247
grad_step = 000332, loss = 0.001247
grad_step = 000333, loss = 0.001228
grad_step = 000334, loss = 0.001222
grad_step = 000335, loss = 0.001207
grad_step = 000336, loss = 0.001208
grad_step = 000337, loss = 0.001192
grad_step = 000338, loss = 0.001189
grad_step = 000339, loss = 0.001181
grad_step = 000340, loss = 0.001187
grad_step = 000341, loss = 0.001226
grad_step = 000342, loss = 0.001439
grad_step = 000343, loss = 0.001916
grad_step = 000344, loss = 0.001853
grad_step = 000345, loss = 0.001680
grad_step = 000346, loss = 0.001371
grad_step = 000347, loss = 0.002093
grad_step = 000348, loss = 0.001456
grad_step = 000349, loss = 0.001402
grad_step = 000350, loss = 0.001436
grad_step = 000351, loss = 0.001411
grad_step = 000352, loss = 0.001319
grad_step = 000353, loss = 0.001364
grad_step = 000354, loss = 0.001273
grad_step = 000355, loss = 0.001245
grad_step = 000356, loss = 0.001283
grad_step = 000357, loss = 0.001209
grad_step = 000358, loss = 0.001202
grad_step = 000359, loss = 0.001183
grad_step = 000360, loss = 0.001148
grad_step = 000361, loss = 0.001166
grad_step = 000362, loss = 0.001108
grad_step = 000363, loss = 0.001169
grad_step = 000364, loss = 0.001086
grad_step = 000365, loss = 0.001063
grad_step = 000366, loss = 0.001115
grad_step = 000367, loss = 0.001047
grad_step = 000368, loss = 0.001026
grad_step = 000369, loss = 0.001034
grad_step = 000370, loss = 0.001023
grad_step = 000371, loss = 0.001002
grad_step = 000372, loss = 0.000982
grad_step = 000373, loss = 0.000959
grad_step = 000374, loss = 0.000946
grad_step = 000375, loss = 0.000965
grad_step = 000376, loss = 0.000948
grad_step = 000377, loss = 0.000929
grad_step = 000378, loss = 0.000939
grad_step = 000379, loss = 0.000891
grad_step = 000380, loss = 0.000880
grad_step = 000381, loss = 0.000882
grad_step = 000382, loss = 0.000853
grad_step = 000383, loss = 0.000848
grad_step = 000384, loss = 0.000887
grad_step = 000385, loss = 0.000901
grad_step = 000386, loss = 0.001036
grad_step = 000387, loss = 0.001209
grad_step = 000388, loss = 0.001408
grad_step = 000389, loss = 0.000967
grad_step = 000390, loss = 0.000786
grad_step = 000391, loss = 0.001022
grad_step = 000392, loss = 0.001257
grad_step = 000393, loss = 0.001086
grad_step = 000394, loss = 0.000746
grad_step = 000395, loss = 0.001056
grad_step = 000396, loss = 0.001221
grad_step = 000397, loss = 0.000818
grad_step = 000398, loss = 0.000984
grad_step = 000399, loss = 0.001231
grad_step = 000400, loss = 0.000855
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000820
grad_step = 000402, loss = 0.001119
grad_step = 000403, loss = 0.000795
grad_step = 000404, loss = 0.000765
grad_step = 000405, loss = 0.000876
grad_step = 000406, loss = 0.000782
grad_step = 000407, loss = 0.000653
grad_step = 000408, loss = 0.000773
grad_step = 000409, loss = 0.000682
grad_step = 000410, loss = 0.000646
grad_step = 000411, loss = 0.000671
grad_step = 000412, loss = 0.000669
grad_step = 000413, loss = 0.000619
grad_step = 000414, loss = 0.000609
grad_step = 000415, loss = 0.000650
grad_step = 000416, loss = 0.000599
grad_step = 000417, loss = 0.000577
grad_step = 000418, loss = 0.000604
grad_step = 000419, loss = 0.000594
grad_step = 000420, loss = 0.000562
grad_step = 000421, loss = 0.000561
grad_step = 000422, loss = 0.000554
grad_step = 000423, loss = 0.000568
grad_step = 000424, loss = 0.000533
grad_step = 000425, loss = 0.000517
grad_step = 000426, loss = 0.000534
grad_step = 000427, loss = 0.000524
grad_step = 000428, loss = 0.000511
grad_step = 000429, loss = 0.000497
grad_step = 000430, loss = 0.000498
grad_step = 000431, loss = 0.000502
grad_step = 000432, loss = 0.000491
grad_step = 000433, loss = 0.000474
grad_step = 000434, loss = 0.000475
grad_step = 000435, loss = 0.000475
grad_step = 000436, loss = 0.000476
grad_step = 000437, loss = 0.000468
grad_step = 000438, loss = 0.000451
grad_step = 000439, loss = 0.000451
grad_step = 000440, loss = 0.000450
grad_step = 000441, loss = 0.000449
grad_step = 000442, loss = 0.000447
grad_step = 000443, loss = 0.000434
grad_step = 000444, loss = 0.000428
grad_step = 000445, loss = 0.000428
grad_step = 000446, loss = 0.000426
grad_step = 000447, loss = 0.000425
grad_step = 000448, loss = 0.000422
grad_step = 000449, loss = 0.000415
grad_step = 000450, loss = 0.000410
grad_step = 000451, loss = 0.000406
grad_step = 000452, loss = 0.000402
grad_step = 000453, loss = 0.000400
grad_step = 000454, loss = 0.000399
grad_step = 000455, loss = 0.000396
grad_step = 000456, loss = 0.000394
grad_step = 000457, loss = 0.000392
grad_step = 000458, loss = 0.000389
grad_step = 000459, loss = 0.000386
grad_step = 000460, loss = 0.000383
grad_step = 000461, loss = 0.000380
grad_step = 000462, loss = 0.000378
grad_step = 000463, loss = 0.000375
grad_step = 000464, loss = 0.000373
grad_step = 000465, loss = 0.000371
grad_step = 000466, loss = 0.000369
grad_step = 000467, loss = 0.000367
grad_step = 000468, loss = 0.000366
grad_step = 000469, loss = 0.000364
grad_step = 000470, loss = 0.000364
grad_step = 000471, loss = 0.000365
grad_step = 000472, loss = 0.000370
grad_step = 000473, loss = 0.000384
grad_step = 000474, loss = 0.000410
grad_step = 000475, loss = 0.000465
grad_step = 000476, loss = 0.000532
grad_step = 000477, loss = 0.000624
grad_step = 000478, loss = 0.000611
grad_step = 000479, loss = 0.000536
grad_step = 000480, loss = 0.000403
grad_step = 000481, loss = 0.000351
grad_step = 000482, loss = 0.000397
grad_step = 000483, loss = 0.000453
grad_step = 000484, loss = 0.000450
grad_step = 000485, loss = 0.000375
grad_step = 000486, loss = 0.000345
grad_step = 000487, loss = 0.000375
grad_step = 000488, loss = 0.000405
grad_step = 000489, loss = 0.000403
grad_step = 000490, loss = 0.000360
grad_step = 000491, loss = 0.000339
grad_step = 000492, loss = 0.000354
grad_step = 000493, loss = 0.000377
grad_step = 000494, loss = 0.000383
grad_step = 000495, loss = 0.000356
grad_step = 000496, loss = 0.000336
grad_step = 000497, loss = 0.000337
grad_step = 000498, loss = 0.000350
grad_step = 000499, loss = 0.000361
grad_step = 000500, loss = 0.000350
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000336
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

  date_run                              2020-05-12 11:12:21.289104
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.260568
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 11:12:21.295577
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.159299
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 11:12:21.303246
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149499
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 11:12:21.308497
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.42061
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
0   2020-05-12 11:11:51.745773  ...    mean_absolute_error
1   2020-05-12 11:11:51.749577  ...     mean_squared_error
2   2020-05-12 11:11:51.752692  ...  median_absolute_error
3   2020-05-12 11:11:51.755880  ...               r2_score
4   2020-05-12 11:12:02.490467  ...    mean_absolute_error
5   2020-05-12 11:12:02.500588  ...     mean_squared_error
6   2020-05-12 11:12:02.503973  ...  median_absolute_error
7   2020-05-12 11:12:02.507539  ...               r2_score
8   2020-05-12 11:12:21.289104  ...    mean_absolute_error
9   2020-05-12 11:12:21.295577  ...     mean_squared_error
10  2020-05-12 11:12:21.303246  ...  median_absolute_error
11  2020-05-12 11:12:21.308497  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2924544/9912422 [00:00<00:00, 28162944.03it/s]9920512it [00:00, 33680373.80it/s]                             
0it [00:00, ?it/s]32768it [00:00, 604980.89it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463701.48it/s]1654784it [00:00, 10391729.26it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208331.75it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4a2952b38> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4400a7e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4a2916e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa438e95080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4a2952b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa45530ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4a2952b38> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa45530ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4a295efd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa45530ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4400a4080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7efc96c5b208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=713901a0e0e52ef1d8ca69ac7cd96550a8b87ad8c2bf35b22bf143354839efe5
  Stored in directory: /tmp/pip-ephem-wheel-cache-l7f2yuwd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7efc2e843198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  245760/17464789 [..............................] - ETA: 3s
  598016/17464789 [>.............................] - ETA: 3s
  958464/17464789 [>.............................] - ETA: 2s
 1376256/17464789 [=>............................] - ETA: 2s
 1810432/17464789 [==>...........................] - ETA: 2s
 2252800/17464789 [==>...........................] - ETA: 2s
 2719744/17464789 [===>..........................] - ETA: 2s
 3276800/17464789 [====>.........................] - ETA: 1s
 3850240/17464789 [=====>........................] - ETA: 1s
 4513792/17464789 [======>.......................] - ETA: 1s
 5193728/17464789 [=======>......................] - ETA: 1s
 5971968/17464789 [=========>....................] - ETA: 1s
 6742016/17464789 [==========>...................] - ETA: 1s
 7593984/17464789 [============>.................] - ETA: 0s
 8495104/17464789 [=============>................] - ETA: 0s
 9404416/17464789 [===============>..............] - ETA: 0s
10379264/17464789 [================>.............] - ETA: 0s
11354112/17464789 [==================>...........] - ETA: 0s
12312576/17464789 [====================>.........] - ETA: 0s
13205504/17464789 [=====================>........] - ETA: 0s
14155776/17464789 [=======================>......] - ETA: 0s
15040512/17464789 [========================>.....] - ETA: 0s
16039936/17464789 [==========================>...] - ETA: 0s
16990208/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 11:13:49.074771: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 11:13:49.079731: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 11:13:49.079881: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d23d7a1850 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 11:13:49.079894: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8148 - accuracy: 0.4903
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7893 - accuracy: 0.4920
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7617 - accuracy: 0.4938
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7535 - accuracy: 0.4943
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7805 - accuracy: 0.4926
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7299 - accuracy: 0.4959
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7194 - accuracy: 0.4966
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6605 - accuracy: 0.5004
11000/25000 [============>.................] - ETA: 3s - loss: 7.6806 - accuracy: 0.4991
12000/25000 [=============>................] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7044 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6984 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6648 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6581 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6644 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 11:14:02.760518
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 11:14:02.760518  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 11:14:08.982280: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 11:14:08.988972: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 11:14:08.989336: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612f2414b80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 11:14:08.989353: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ffbd4d44be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8973 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.8608 - val_crf_viterbi_accuracy: 0.2400

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ffbb0b45b38> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7024 - accuracy: 0.4977
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7258 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7632 - accuracy: 0.4937
11000/25000 [============>.................] - ETA: 3s - loss: 7.7544 - accuracy: 0.4943
12000/25000 [=============>................] - ETA: 3s - loss: 7.7663 - accuracy: 0.4935
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7362 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7269 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7050 - accuracy: 0.4975
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7108 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ffbac320a90> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:32:55, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:13:01, 18.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:18:21, 25.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:31:23, 36.7kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:33:17, 52.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.05M/862M [00:01<3:10:24, 74.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:12:42, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:32:37, 152kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:04:33, 217kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<44:58, 310kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:01<31:30, 440kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.2M/862M [00:01<21:58, 627kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:01<15:27, 888kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.8M/862M [00:02<10:51, 1.26MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.4M/862M [00:02<07:38, 1.78MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<05:25, 2.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:05, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:09, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<05:11, 2.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<05:55, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:46, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:26, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<06:46, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:21, 2.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<03:52, 3.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<21:42, 609kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<16:32, 799kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:10<11:53, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<11:22, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<09:06, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:12<06:41, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<07:46, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<06:46, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:14<05:01, 2.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:37, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<07:11, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<05:41, 2.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<05:35, 2.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<04:11, 3.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:57, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:48, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<05:18, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<03:53, 3.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<08:10, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<07:01, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<05:14, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<06:39, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:15, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<05:36, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:06, 3.09MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:27, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:13, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:22, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:41, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:15, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:38, 2.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:03, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<20:28, 612kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<15:38, 801kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<11:12, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:44, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:05, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:41, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:22, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:25, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:46, 2.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:13, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:36, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:14, 2.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:51, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:35, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:10, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:45, 3.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:02, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:16, 1.47MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:04, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:27, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:14, 2.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:42, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:56, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:28, 2.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:54, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:34, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:08, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:41, 3.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:20:39, 148kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<57:40, 207kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<40:33, 293kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<31:05, 381kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<24:09, 490kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<17:24, 680kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<12:17, 959kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<15:34, 757kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:05, 974kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:44, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:51, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:42, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:43, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:50, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<21:41, 537kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<16:26, 708kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<11:45, 987kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<10:46, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<10:02, 1.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:35, 1.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<05:28, 2.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:16, 1.39MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:02, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:11, 2.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:09, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:52, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:20, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:51, 2.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<13:52, 819kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<10:57, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:54, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:00, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<08:01, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:09, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:24, 2.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<13:17, 844kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<10:31, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:39, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:48, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:41, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:55, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:52, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:19, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<03:59, 2.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:06, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:47, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:29, 3.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:50, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:40, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:52, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:25, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:40, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<18:45, 576kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<14:06, 766kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<10:06, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<07:13, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<15:31, 692kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<13:15, 810kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<09:46, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:57, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<11:05, 962kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:54, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:28, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:52, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<07:09, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:58, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:13, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:16, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<06:01, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:32, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:56, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:20, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:52, 2.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:47, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:53, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:23, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:21, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:58, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:25, 3.00MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<17:49, 575kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<13:34, 754kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<09:45, 1.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:04, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:38, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:44, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<15:48, 640kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<12:09, 832kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<08:42, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<08:19, 1.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:54, 1.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:03, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:45, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:15, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:51, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:32, 2.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:09, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:21, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:58, 2.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:59, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:40, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:26, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:13, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:38, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:42, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:13, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:05, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:37, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:24, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:14, 2.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:32, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:56, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:40, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:41, 2.04MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:19, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:14, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:55<03:05, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<16:40, 569kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<12:41, 747kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<09:07, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:27, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [01:59<06:56, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:06, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:38, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:57, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:41, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:37, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:12, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:03, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<02:56, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:06, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:30, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:47, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:27, 2.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<12:36, 725kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<10:46, 849kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<07:57, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<05:38, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<13:10, 689kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<10:12, 889kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<07:19, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<05:23, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:08, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<09:02, 995kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:17, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<05:19, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:39, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:53, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:34, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:18, 2.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<1:05:31, 135kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<47:45, 186kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<33:49, 262kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<23:39, 372kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<1:19:20, 111kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<57:15, 154kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<40:25, 217kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<29:34, 295kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<22:33, 387kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<16:09, 539kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<11:21, 763kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<14:40, 590kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:09, 712kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<08:56, 966kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<06:19, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<53:51, 160kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<39:33, 217kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<28:01, 306kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<19:39, 434kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<17:37, 483kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<14:10, 601kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<10:20, 822kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<07:17, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<35:54, 235kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<26:55, 314kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<19:13, 439kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<13:29, 623kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<15:46, 532kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<12:49, 654kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<09:20, 897kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<06:38, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:51, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:16, 1.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:26, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:54, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<06:44, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:29, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:57, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:33, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<1:00:48, 135kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<43:53, 186kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<31:00, 263kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<22:59, 353kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<17:46, 456kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<12:47, 633kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<09:00, 894kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<11:33, 696kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<09:47, 820kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:13, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<05:08, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<08:02, 992kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<06:56, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:08, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<03:40, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<19:14, 411kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<15:06, 523kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<10:55, 722kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<07:43, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:53, 881kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:50, 999kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:51, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<04:10, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<30:48, 252kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<22:49, 340kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<16:15, 476kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<12:39, 608kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<10:32, 730kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<07:43, 995kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:36, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:38, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:13, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:56, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:54, 2.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:27, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:24, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:22, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:40, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:37, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:42, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:05, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:22, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<03:27, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:56<02:29, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<1:01:37, 119kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<44:37, 165kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<31:34, 232kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<22:02, 331kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:14:47, 97.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<53:49, 135kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<37:56, 192kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<26:35, 272kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<20:52, 346kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<16:04, 449kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<11:33, 623kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<08:09, 878kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<08:22, 853kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<07:19, 975kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:26, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<03:53, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:23, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:58, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:32, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<52:16, 134kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<37:43, 186kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<26:35, 263kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<18:37, 373kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<16:53, 411kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<12:56, 536kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<09:17, 745kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<06:39, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:26, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:54, 1.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:28, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<03:10, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<28:42, 237kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<21:28, 317kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<15:20, 443kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<10:45, 627kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<11:43, 575kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<09:34, 703kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:59, 961kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<04:58, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:31, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:56, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<03:10, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:14, 1.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:24, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:02, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:57, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:09, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:13, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:18, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<07:17, 886kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:09, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:33, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:17, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:24, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:32, 2.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:12, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:15, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:31, 2.50MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:51, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:16, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:37, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<01:53, 3.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<51:44, 120kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<37:27, 165kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<26:29, 233kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<18:27, 331kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<1:02:52, 97.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<45:14, 135kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<31:53, 191kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<22:15, 273kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<19:11, 315kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<14:39, 413kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<10:31, 574kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<07:21, 813kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<15:14, 393kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<11:35, 516kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<08:19, 716kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:50, 865kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<05:59, 987kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:29, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:11, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<43:57, 133kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<31:38, 185kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<22:18, 261kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<16:31, 349kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<12:36, 458kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<09:03, 635kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<07:15, 787kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<06:15, 912kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<04:37, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<03:18, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:28, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:00, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:05, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:18, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:36, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:41, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:00, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:22, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:53, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:17, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:27, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:48, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:14, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:24, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:06, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:33, 3.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:52, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:04, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<02:25, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<01:44, 2.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<07:13, 714kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<06:06, 843kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<04:32, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<03:11, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<1:40:25, 50.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<1:11:18, 71.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<50:00, 101kB/s]   .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<34:45, 145kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<28:51, 174kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<21:13, 237kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<15:01, 333kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<10:33, 472kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<08:59, 551kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:18, 678kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<05:21, 922kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<03:45, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<11:48, 413kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<09:16, 527kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<06:43, 724kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<04:42, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<1:36:02, 50.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<1:08:11, 70.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<47:50, 100kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<33:11, 143kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<1:54:02, 41.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<1:20:45, 58.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<56:36, 83.6kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<39:56, 117kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<28:54, 162kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<20:23, 229kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<14:11, 325kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<15:02, 307kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<11:27, 402kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<08:14, 557kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<05:45, 789kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<1:31:18, 49.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<1:04:47, 70.0kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<45:27, 99.5kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<31:30, 142kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:48:05, 41.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<1:16:31, 58.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<53:38, 83.1kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<37:09, 119kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<1:50:31, 39.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<1:18:12, 56.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<54:47, 80.1kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<38:00, 114kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<31:02, 140kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<22:35, 192kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<15:59, 270kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<11:45, 363kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<09:05, 469kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:32, 649kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<04:35, 917kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:26, 651kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<05:04, 826kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:39, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<02:35, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<06:44, 613kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<05:27, 755kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:00, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:26, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:09, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:23, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:19, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:20, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:48, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:17, 3.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:31, 709kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:34, 856kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:22, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:58, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:48, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:08, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:06, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:11, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:41, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:47, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:31, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:40, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:51, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:27, 2.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:46, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:24, 2.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:33, 2.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:43, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:21, 2.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:31, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:31, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:08, 2.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:50<00:49, 4.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<51:05, 66.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<37:00, 91.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<26:08, 129kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<18:13, 183kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<13:30, 245kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<10:01, 329kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<07:07, 461kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<05:36, 576kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:11, 755kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:23, 930kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:29, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:16, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:08, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:37, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:38, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:41, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:18, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:30, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:10, 2.50MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:06, 2.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<00:47, 3.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<13:31, 209kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<10:32, 268kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<07:36, 370kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<05:19, 523kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<04:23, 626kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:33, 774kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:35, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:14, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:02, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:32, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:30, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:30, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:49, 3.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:31, 723kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:55, 870kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:08, 1.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:53, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:45, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:19, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:19, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:21, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:02, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:07, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:11, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:55, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:39, 3.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<34:04, 66.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<24:13, 93.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<16:56, 133kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<11:39, 190kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<10:38, 207kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<07:49, 281kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<05:30, 396kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<03:49, 562kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<04:10, 512kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:43, 573kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:47, 761kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:58, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:53, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:41, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:12, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:32, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:13, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:53, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:07, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:07, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:37, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:13, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:55, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:56, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:44, 2.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:40, 2.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:28, 3.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:56, 853kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:38, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:12, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:50, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<04:20, 365kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<03:18, 478kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<02:21, 666kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<01:37, 941kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:36, 582kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:05, 725kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:30, 998kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<01:02, 1.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:30, 956kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:18, 1.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:57, 1.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<00:40, 2.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:31, 902kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:19, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:57, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:40, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:44, 753kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:26, 906kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:03, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:55, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:06, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:52, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:37, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:44, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:43, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:33, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:35, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:30, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:23, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<00:16, 3.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:35, 608kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:16, 751kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:55, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:45, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:41, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:30, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:20, 2.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:13, 677kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:59, 826kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:43, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:36, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:33, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:17, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:11, 3.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:55, 676kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:44, 825kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:31, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:21, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:31, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:27, 1.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:06, 3.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<02:20, 148kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:41, 203kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:08, 287kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:43, 383kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:36, 457kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:26, 617kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:16, 865kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:13, 936kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.44MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 840/400000 [00:00<00:47, 8390.75it/s]  0%|          | 1655/400000 [00:00<00:47, 8301.71it/s]  1%|          | 2479/400000 [00:00<00:48, 8280.47it/s]  1%|          | 3284/400000 [00:00<00:48, 8207.74it/s]  1%|          | 4132/400000 [00:00<00:47, 8287.41it/s]  1%|▏         | 5009/400000 [00:00<00:46, 8424.51it/s]  1%|▏         | 5848/400000 [00:00<00:46, 8414.00it/s]  2%|▏         | 6654/400000 [00:00<00:47, 8303.50it/s]  2%|▏         | 7500/400000 [00:00<00:47, 8347.87it/s]  2%|▏         | 8339/400000 [00:01<00:46, 8359.51it/s]  2%|▏         | 9210/400000 [00:01<00:46, 8459.73it/s]  3%|▎         | 10089/400000 [00:01<00:45, 8554.69it/s]  3%|▎         | 10937/400000 [00:01<00:45, 8531.78it/s]  3%|▎         | 11807/400000 [00:01<00:45, 8579.05it/s]  3%|▎         | 12680/400000 [00:01<00:44, 8621.45it/s]  3%|▎         | 13543/400000 [00:01<00:44, 8621.72it/s]  4%|▎         | 14403/400000 [00:01<00:44, 8590.58it/s]  4%|▍         | 15261/400000 [00:01<00:44, 8581.97it/s]  4%|▍         | 16126/400000 [00:01<00:44, 8598.72it/s]  4%|▍         | 16986/400000 [00:02<00:44, 8532.83it/s]  4%|▍         | 17840/400000 [00:02<00:44, 8534.39it/s]  5%|▍         | 18696/400000 [00:02<00:44, 8539.49it/s]  5%|▍         | 19550/400000 [00:02<00:44, 8506.34it/s]  5%|▌         | 20414/400000 [00:02<00:44, 8543.82it/s]  5%|▌         | 21271/400000 [00:02<00:44, 8548.73it/s]  6%|▌         | 22126/400000 [00:02<00:44, 8546.79it/s]  6%|▌         | 22981/400000 [00:02<00:44, 8536.12it/s]  6%|▌         | 23835/400000 [00:02<00:44, 8495.41it/s]  6%|▌         | 24685/400000 [00:02<00:44, 8439.52it/s]  6%|▋         | 25530/400000 [00:03<00:44, 8396.46it/s]  7%|▋         | 26388/400000 [00:03<00:44, 8448.87it/s]  7%|▋         | 27249/400000 [00:03<00:43, 8494.12it/s]  7%|▋         | 28099/400000 [00:03<00:43, 8494.53it/s]  7%|▋         | 28949/400000 [00:03<00:44, 8407.82it/s]  7%|▋         | 29801/400000 [00:03<00:43, 8440.21it/s]  8%|▊         | 30667/400000 [00:03<00:43, 8504.76it/s]  8%|▊         | 31518/400000 [00:03<00:43, 8483.97it/s]  8%|▊         | 32367/400000 [00:03<00:43, 8460.12it/s]  8%|▊         | 33214/400000 [00:03<00:45, 8111.13it/s]  9%|▊         | 34058/400000 [00:04<00:44, 8205.88it/s]  9%|▊         | 34908/400000 [00:04<00:44, 8289.25it/s]  9%|▉         | 35760/400000 [00:04<00:43, 8354.54it/s]  9%|▉         | 36597/400000 [00:04<00:44, 8102.60it/s]  9%|▉         | 37457/400000 [00:04<00:43, 8243.12it/s] 10%|▉         | 38303/400000 [00:04<00:43, 8306.26it/s] 10%|▉         | 39161/400000 [00:04<00:43, 8384.78it/s] 10%|█         | 40017/400000 [00:04<00:42, 8433.82it/s] 10%|█         | 40862/400000 [00:04<00:42, 8401.02it/s] 10%|█         | 41703/400000 [00:04<00:43, 8269.51it/s] 11%|█         | 42531/400000 [00:05<00:44, 8028.42it/s] 11%|█         | 43337/400000 [00:05<00:44, 8033.64it/s] 11%|█         | 44193/400000 [00:05<00:43, 8182.90it/s] 11%|█▏        | 45014/400000 [00:05<00:43, 8129.30it/s] 11%|█▏        | 45829/400000 [00:05<00:43, 8106.90it/s] 12%|█▏        | 46686/400000 [00:05<00:42, 8237.92it/s] 12%|█▏        | 47539/400000 [00:05<00:42, 8322.67it/s] 12%|█▏        | 48401/400000 [00:05<00:41, 8406.97it/s] 12%|█▏        | 49252/400000 [00:05<00:41, 8435.32it/s] 13%|█▎        | 50116/400000 [00:05<00:41, 8494.38it/s] 13%|█▎        | 50967/400000 [00:06<00:41, 8488.90it/s] 13%|█▎        | 51817/400000 [00:06<00:41, 8458.85it/s] 13%|█▎        | 52668/400000 [00:06<00:40, 8473.02it/s] 13%|█▎        | 53516/400000 [00:06<00:41, 8377.65it/s] 14%|█▎        | 54359/400000 [00:06<00:41, 8392.50it/s] 14%|█▍        | 55202/400000 [00:06<00:41, 8402.19it/s] 14%|█▍        | 56061/400000 [00:06<00:40, 8456.85it/s] 14%|█▍        | 56907/400000 [00:06<00:41, 8351.82it/s] 14%|█▍        | 57743/400000 [00:06<00:41, 8317.78it/s] 15%|█▍        | 58598/400000 [00:06<00:40, 8383.90it/s] 15%|█▍        | 59448/400000 [00:07<00:40, 8415.86it/s] 15%|█▌        | 60311/400000 [00:07<00:40, 8476.28it/s] 15%|█▌        | 61169/400000 [00:07<00:39, 8506.80it/s] 16%|█▌        | 62020/400000 [00:07<00:39, 8462.08it/s] 16%|█▌        | 62875/400000 [00:07<00:39, 8486.60it/s] 16%|█▌        | 63728/400000 [00:07<00:39, 8497.82it/s] 16%|█▌        | 64582/400000 [00:07<00:39, 8507.81it/s] 16%|█▋        | 65438/400000 [00:07<00:39, 8521.05it/s] 17%|█▋        | 66291/400000 [00:07<00:40, 8225.95it/s] 17%|█▋        | 67131/400000 [00:07<00:40, 8275.08it/s] 17%|█▋        | 67988/400000 [00:08<00:39, 8359.85it/s] 17%|█▋        | 68846/400000 [00:08<00:39, 8424.08it/s] 17%|█▋        | 69703/400000 [00:08<00:39, 8465.68it/s] 18%|█▊        | 70551/400000 [00:08<00:39, 8444.38it/s] 18%|█▊        | 71415/400000 [00:08<00:38, 8500.06it/s] 18%|█▊        | 72281/400000 [00:08<00:38, 8547.19it/s] 18%|█▊        | 73143/400000 [00:08<00:38, 8566.57it/s] 19%|█▊        | 74008/400000 [00:08<00:37, 8588.59it/s] 19%|█▊        | 74868/400000 [00:08<00:38, 8518.17it/s] 19%|█▉        | 75733/400000 [00:08<00:37, 8554.32it/s] 19%|█▉        | 76589/400000 [00:09<00:38, 8372.16it/s] 19%|█▉        | 77428/400000 [00:09<00:39, 8262.12it/s] 20%|█▉        | 78256/400000 [00:09<00:39, 8188.24it/s] 20%|█▉        | 79076/400000 [00:09<00:39, 8126.91it/s] 20%|█▉        | 79908/400000 [00:09<00:39, 8182.55it/s] 20%|██        | 80772/400000 [00:09<00:38, 8313.19it/s] 20%|██        | 81626/400000 [00:09<00:37, 8378.75it/s] 21%|██        | 82498/400000 [00:09<00:37, 8476.65it/s] 21%|██        | 83347/400000 [00:09<00:37, 8440.77it/s] 21%|██        | 84192/400000 [00:10<00:37, 8353.82it/s] 21%|██▏       | 85032/400000 [00:10<00:37, 8364.84it/s] 21%|██▏       | 85883/400000 [00:10<00:37, 8407.30it/s] 22%|██▏       | 86747/400000 [00:10<00:36, 8475.27it/s] 22%|██▏       | 87595/400000 [00:10<00:37, 8275.35it/s] 22%|██▏       | 88453/400000 [00:10<00:37, 8364.29it/s] 22%|██▏       | 89312/400000 [00:10<00:36, 8429.09it/s] 23%|██▎       | 90166/400000 [00:10<00:36, 8459.64it/s] 23%|██▎       | 91029/400000 [00:10<00:36, 8509.81it/s] 23%|██▎       | 91881/400000 [00:10<00:36, 8492.67it/s] 23%|██▎       | 92745/400000 [00:11<00:36, 8533.40it/s] 23%|██▎       | 93601/400000 [00:11<00:35, 8540.79it/s] 24%|██▎       | 94476/400000 [00:11<00:35, 8600.15it/s] 24%|██▍       | 95351/400000 [00:11<00:35, 8643.54it/s] 24%|██▍       | 96221/400000 [00:11<00:35, 8658.16it/s] 24%|██▍       | 97087/400000 [00:11<00:35, 8651.79it/s] 24%|██▍       | 97953/400000 [00:11<00:35, 8616.92it/s] 25%|██▍       | 98815/400000 [00:11<00:35, 8407.52it/s] 25%|██▍       | 99686/400000 [00:11<00:35, 8494.03it/s] 25%|██▌       | 100537/400000 [00:11<00:35, 8487.71it/s] 25%|██▌       | 101411/400000 [00:12<00:34, 8559.80it/s] 26%|██▌       | 102268/400000 [00:12<00:34, 8531.82it/s] 26%|██▌       | 103123/400000 [00:12<00:34, 8535.06it/s] 26%|██▌       | 103999/400000 [00:12<00:34, 8600.42it/s] 26%|██▌       | 104860/400000 [00:12<00:34, 8593.36it/s] 26%|██▋       | 105731/400000 [00:12<00:34, 8627.53it/s] 27%|██▋       | 106594/400000 [00:12<00:34, 8599.35it/s] 27%|██▋       | 107455/400000 [00:12<00:34, 8552.34it/s] 27%|██▋       | 108324/400000 [00:12<00:33, 8590.75it/s] 27%|██▋       | 109186/400000 [00:12<00:33, 8597.58it/s] 28%|██▊       | 110057/400000 [00:13<00:33, 8629.20it/s] 28%|██▊       | 110921/400000 [00:13<00:33, 8607.65it/s] 28%|██▊       | 111782/400000 [00:13<00:33, 8498.12it/s] 28%|██▊       | 112633/400000 [00:13<00:33, 8478.21it/s] 28%|██▊       | 113483/400000 [00:13<00:33, 8482.49it/s] 29%|██▊       | 114347/400000 [00:13<00:33, 8526.82it/s] 29%|██▉       | 115200/400000 [00:13<00:33, 8501.76it/s] 29%|██▉       | 116058/400000 [00:13<00:33, 8524.88it/s] 29%|██▉       | 116927/400000 [00:13<00:33, 8571.65it/s] 29%|██▉       | 117785/400000 [00:13<00:33, 8482.08it/s] 30%|██▉       | 118640/400000 [00:14<00:33, 8500.31it/s] 30%|██▉       | 119491/400000 [00:14<00:33, 8449.79it/s] 30%|███       | 120371/400000 [00:14<00:32, 8549.98it/s] 30%|███       | 121227/400000 [00:14<00:32, 8515.95it/s] 31%|███       | 122079/400000 [00:14<00:32, 8509.58it/s] 31%|███       | 122939/400000 [00:14<00:32, 8536.24it/s] 31%|███       | 123807/400000 [00:14<00:32, 8577.90it/s] 31%|███       | 124667/400000 [00:14<00:32, 8582.24it/s] 31%|███▏      | 125536/400000 [00:14<00:31, 8613.81it/s] 32%|███▏      | 126398/400000 [00:14<00:32, 8512.59it/s] 32%|███▏      | 127250/400000 [00:15<00:32, 8442.08it/s] 32%|███▏      | 128095/400000 [00:15<00:33, 8215.85it/s] 32%|███▏      | 128943/400000 [00:15<00:32, 8291.34it/s] 32%|███▏      | 129798/400000 [00:15<00:32, 8366.61it/s] 33%|███▎      | 130641/400000 [00:15<00:32, 8384.68it/s] 33%|███▎      | 131496/400000 [00:15<00:31, 8432.61it/s] 33%|███▎      | 132364/400000 [00:15<00:31, 8502.82it/s] 33%|███▎      | 133217/400000 [00:15<00:31, 8509.51it/s] 34%|███▎      | 134081/400000 [00:15<00:31, 8546.73it/s] 34%|███▎      | 134936/400000 [00:15<00:31, 8431.76it/s] 34%|███▍      | 135793/400000 [00:16<00:31, 8471.16it/s] 34%|███▍      | 136641/400000 [00:16<00:31, 8448.12it/s] 34%|███▍      | 137493/400000 [00:16<00:31, 8467.15it/s] 35%|███▍      | 138340/400000 [00:16<00:31, 8421.29it/s] 35%|███▍      | 139188/400000 [00:16<00:30, 8436.89it/s] 35%|███▌      | 140054/400000 [00:16<00:30, 8501.34it/s] 35%|███▌      | 140905/400000 [00:16<00:30, 8495.17it/s] 35%|███▌      | 141755/400000 [00:16<00:30, 8397.28it/s] 36%|███▌      | 142605/400000 [00:16<00:30, 8426.78it/s] 36%|███▌      | 143454/400000 [00:16<00:30, 8444.51it/s] 36%|███▌      | 144303/400000 [00:17<00:30, 8456.63it/s] 36%|███▋      | 145153/400000 [00:17<00:30, 8468.65it/s] 37%|███▋      | 146020/400000 [00:17<00:29, 8525.97it/s] 37%|███▋      | 146873/400000 [00:17<00:29, 8518.04it/s] 37%|███▋      | 147728/400000 [00:17<00:29, 8526.80it/s] 37%|███▋      | 148582/400000 [00:17<00:29, 8529.80it/s] 37%|███▋      | 149447/400000 [00:17<00:29, 8564.33it/s] 38%|███▊      | 150307/400000 [00:17<00:29, 8572.38it/s] 38%|███▊      | 151170/400000 [00:17<00:28, 8587.23it/s] 38%|███▊      | 152029/400000 [00:17<00:28, 8567.81it/s] 38%|███▊      | 152886/400000 [00:18<00:28, 8521.27it/s] 38%|███▊      | 153739/400000 [00:18<00:29, 8441.97it/s] 39%|███▊      | 154584/400000 [00:18<00:29, 8411.14it/s] 39%|███▉      | 155443/400000 [00:18<00:28, 8463.22it/s] 39%|███▉      | 156292/400000 [00:18<00:28, 8470.21it/s] 39%|███▉      | 157153/400000 [00:18<00:28, 8509.56it/s] 40%|███▉      | 158020/400000 [00:18<00:28, 8555.49it/s] 40%|███▉      | 158883/400000 [00:18<00:28, 8575.52it/s] 40%|███▉      | 159750/400000 [00:18<00:27, 8602.31it/s] 40%|████      | 160611/400000 [00:19<00:28, 8469.60it/s] 40%|████      | 161465/400000 [00:19<00:28, 8490.02it/s] 41%|████      | 162332/400000 [00:19<00:27, 8540.41it/s] 41%|████      | 163187/400000 [00:19<00:27, 8486.26it/s] 41%|████      | 164044/400000 [00:19<00:27, 8509.42it/s] 41%|████      | 164896/400000 [00:19<00:27, 8508.60it/s] 41%|████▏     | 165762/400000 [00:19<00:27, 8551.96it/s] 42%|████▏     | 166618/400000 [00:19<00:27, 8553.76it/s] 42%|████▏     | 167477/400000 [00:19<00:27, 8559.89it/s] 42%|████▏     | 168334/400000 [00:19<00:27, 8424.27it/s] 42%|████▏     | 169197/400000 [00:20<00:27, 8484.70it/s] 43%|████▎     | 170046/400000 [00:20<00:27, 8484.11it/s] 43%|████▎     | 170903/400000 [00:20<00:26, 8508.27it/s] 43%|████▎     | 171755/400000 [00:20<00:27, 8441.05it/s] 43%|████▎     | 172604/400000 [00:20<00:26, 8455.26it/s] 43%|████▎     | 173450/400000 [00:20<00:26, 8444.95it/s] 44%|████▎     | 174303/400000 [00:20<00:26, 8467.57it/s] 44%|████▍     | 175160/400000 [00:20<00:26, 8496.74it/s] 44%|████▍     | 176025/400000 [00:20<00:26, 8541.91it/s] 44%|████▍     | 176880/400000 [00:20<00:26, 8401.24it/s] 44%|████▍     | 177721/400000 [00:21<00:26, 8378.67it/s] 45%|████▍     | 178560/400000 [00:21<00:26, 8271.03it/s] 45%|████▍     | 179388/400000 [00:21<00:26, 8244.32it/s] 45%|████▌     | 180235/400000 [00:21<00:26, 8308.97it/s] 45%|████▌     | 181102/400000 [00:21<00:26, 8412.32it/s] 45%|████▌     | 181983/400000 [00:21<00:25, 8525.79it/s] 46%|████▌     | 182837/400000 [00:21<00:25, 8492.93it/s] 46%|████▌     | 183701/400000 [00:21<00:25, 8534.83it/s] 46%|████▌     | 184565/400000 [00:21<00:25, 8564.24it/s] 46%|████▋     | 185422/400000 [00:21<00:25, 8475.49it/s] 47%|████▋     | 186284/400000 [00:22<00:25, 8516.17it/s] 47%|████▋     | 187138/400000 [00:22<00:24, 8520.61it/s] 47%|████▋     | 187993/400000 [00:22<00:24, 8528.90it/s] 47%|████▋     | 188847/400000 [00:22<00:24, 8505.61it/s] 47%|████▋     | 189734/400000 [00:22<00:24, 8610.83it/s] 48%|████▊     | 190621/400000 [00:22<00:24, 8685.22it/s] 48%|████▊     | 191490/400000 [00:22<00:24, 8644.32it/s] 48%|████▊     | 192365/400000 [00:22<00:23, 8673.82it/s] 48%|████▊     | 193236/400000 [00:22<00:23, 8682.65it/s] 49%|████▊     | 194105/400000 [00:22<00:23, 8670.81it/s] 49%|████▊     | 194982/400000 [00:23<00:23, 8696.32it/s] 49%|████▉     | 195852/400000 [00:23<00:23, 8659.74it/s] 49%|████▉     | 196737/400000 [00:23<00:23, 8713.81it/s] 49%|████▉     | 197609/400000 [00:23<00:23, 8691.60it/s] 50%|████▉     | 198486/400000 [00:23<00:23, 8714.11it/s] 50%|████▉     | 199384/400000 [00:23<00:22, 8791.37it/s] 50%|█████     | 200264/400000 [00:23<00:23, 8680.97it/s] 50%|█████     | 201133/400000 [00:23<00:22, 8670.56it/s] 51%|█████     | 202001/400000 [00:23<00:23, 8540.02it/s] 51%|█████     | 202878/400000 [00:23<00:22, 8605.47it/s] 51%|█████     | 203753/400000 [00:24<00:22, 8646.49it/s] 51%|█████     | 204619/400000 [00:24<00:22, 8641.61it/s] 51%|█████▏    | 205484/400000 [00:24<00:22, 8475.45it/s] 52%|█████▏    | 206363/400000 [00:24<00:22, 8565.87it/s] 52%|█████▏    | 207237/400000 [00:24<00:22, 8615.64it/s] 52%|█████▏    | 208130/400000 [00:24<00:22, 8706.15it/s] 52%|█████▏    | 209002/400000 [00:24<00:22, 8620.89it/s] 52%|█████▏    | 209871/400000 [00:24<00:22, 8641.00it/s] 53%|█████▎    | 210736/400000 [00:24<00:22, 8515.16it/s] 53%|█████▎    | 211600/400000 [00:24<00:22, 8550.39it/s] 53%|█████▎    | 212482/400000 [00:25<00:21, 8628.51it/s] 53%|█████▎    | 213355/400000 [00:25<00:21, 8658.59it/s] 54%|█████▎    | 214232/400000 [00:25<00:21, 8691.68it/s] 54%|█████▍    | 215102/400000 [00:25<00:21, 8673.21it/s] 54%|█████▍    | 215970/400000 [00:25<00:21, 8626.76it/s] 54%|█████▍    | 216859/400000 [00:25<00:21, 8703.37it/s] 54%|█████▍    | 217730/400000 [00:25<00:21, 8662.21it/s] 55%|█████▍    | 218597/400000 [00:25<00:20, 8648.08it/s] 55%|█████▍    | 219463/400000 [00:25<00:21, 8525.82it/s] 55%|█████▌    | 220332/400000 [00:25<00:20, 8573.37it/s] 55%|█████▌    | 221190/400000 [00:26<00:20, 8533.12it/s] 56%|█████▌    | 222044/400000 [00:26<00:20, 8496.48it/s] 56%|█████▌    | 222911/400000 [00:26<00:20, 8544.38it/s] 56%|█████▌    | 223778/400000 [00:26<00:20, 8578.82it/s] 56%|█████▌    | 224637/400000 [00:26<00:20, 8580.97it/s] 56%|█████▋    | 225510/400000 [00:26<00:20, 8624.20it/s] 57%|█████▋    | 226373/400000 [00:26<00:20, 8444.44it/s] 57%|█████▋    | 227219/400000 [00:26<00:20, 8398.65it/s] 57%|█████▋    | 228094/400000 [00:26<00:20, 8495.18it/s] 57%|█████▋    | 228959/400000 [00:26<00:20, 8539.56it/s] 57%|█████▋    | 229831/400000 [00:27<00:19, 8592.34it/s] 58%|█████▊    | 230691/400000 [00:27<00:19, 8551.86it/s] 58%|█████▊    | 231571/400000 [00:27<00:19, 8623.13it/s] 58%|█████▊    | 232437/400000 [00:27<00:19, 8632.97it/s] 58%|█████▊    | 233312/400000 [00:27<00:19, 8667.00it/s] 59%|█████▊    | 234179/400000 [00:27<00:19, 8641.80it/s] 59%|█████▉    | 235061/400000 [00:27<00:18, 8691.91it/s] 59%|█████▉    | 235949/400000 [00:27<00:18, 8747.12it/s] 59%|█████▉    | 236844/400000 [00:27<00:18, 8806.15it/s] 59%|█████▉    | 237725/400000 [00:27<00:18, 8737.85it/s] 60%|█████▉    | 238610/400000 [00:28<00:18, 8769.74it/s] 60%|█████▉    | 239488/400000 [00:28<00:18, 8658.55it/s] 60%|██████    | 240355/400000 [00:28<00:18, 8629.38it/s] 60%|██████    | 241224/400000 [00:28<00:18, 8644.76it/s] 61%|██████    | 242092/400000 [00:28<00:18, 8653.53it/s] 61%|██████    | 242973/400000 [00:28<00:18, 8699.23it/s] 61%|██████    | 243844/400000 [00:28<00:18, 8644.63it/s] 61%|██████    | 244709/400000 [00:28<00:18, 8596.95it/s] 61%|██████▏   | 245572/400000 [00:28<00:17, 8604.07it/s] 62%|██████▏   | 246433/400000 [00:29<00:17, 8579.56it/s] 62%|██████▏   | 247303/400000 [00:29<00:17, 8615.03it/s] 62%|██████▏   | 248170/400000 [00:29<00:17, 8628.99it/s] 62%|██████▏   | 249038/400000 [00:29<00:17, 8642.37it/s] 62%|██████▏   | 249909/400000 [00:29<00:17, 8659.85it/s] 63%|██████▎   | 250779/400000 [00:29<00:17, 8671.52it/s] 63%|██████▎   | 251647/400000 [00:29<00:17, 8663.60it/s] 63%|██████▎   | 252514/400000 [00:29<00:17, 8557.95it/s] 63%|██████▎   | 253376/400000 [00:29<00:17, 8568.78it/s] 64%|██████▎   | 254239/400000 [00:29<00:16, 8585.68it/s] 64%|██████▍   | 255098/400000 [00:30<00:16, 8566.83it/s] 64%|██████▍   | 255962/400000 [00:30<00:16, 8587.14it/s] 64%|██████▍   | 256831/400000 [00:30<00:16, 8615.24it/s] 64%|██████▍   | 257695/400000 [00:30<00:16, 8619.42it/s] 65%|██████▍   | 258568/400000 [00:30<00:16, 8649.90it/s] 65%|██████▍   | 259434/400000 [00:30<00:16, 8597.81it/s] 65%|██████▌   | 260294/400000 [00:30<00:16, 8512.18it/s] 65%|██████▌   | 261158/400000 [00:30<00:16, 8549.70it/s] 66%|██████▌   | 262024/400000 [00:30<00:16, 8581.20it/s] 66%|██████▌   | 262893/400000 [00:30<00:15, 8612.62it/s] 66%|██████▌   | 263774/400000 [00:31<00:15, 8668.65it/s] 66%|██████▌   | 264642/400000 [00:31<00:15, 8542.88it/s] 66%|██████▋   | 265525/400000 [00:31<00:15, 8626.45it/s] 67%|██████▋   | 266389/400000 [00:31<00:15, 8591.09it/s] 67%|██████▋   | 267263/400000 [00:31<00:15, 8634.90it/s] 67%|██████▋   | 268139/400000 [00:31<00:15, 8670.04it/s] 67%|██████▋   | 269007/400000 [00:31<00:15, 8659.44it/s] 67%|██████▋   | 269882/400000 [00:31<00:14, 8684.85it/s] 68%|██████▊   | 270751/400000 [00:31<00:15, 8599.64it/s] 68%|██████▊   | 271627/400000 [00:31<00:14, 8646.39it/s] 68%|██████▊   | 272494/400000 [00:32<00:14, 8651.36it/s] 68%|██████▊   | 273364/400000 [00:32<00:14, 8664.56it/s] 69%|██████▊   | 274233/400000 [00:32<00:14, 8669.34it/s] 69%|██████▉   | 275101/400000 [00:32<00:14, 8651.02it/s] 69%|██████▉   | 275967/400000 [00:32<00:14, 8638.66it/s] 69%|██████▉   | 276844/400000 [00:32<00:14, 8674.84it/s] 69%|██████▉   | 277733/400000 [00:32<00:13, 8737.87it/s] 70%|██████▉   | 278625/400000 [00:32<00:13, 8790.66it/s] 70%|██████▉   | 279505/400000 [00:32<00:14, 8567.09it/s] 70%|███████   | 280369/400000 [00:32<00:13, 8587.26it/s] 70%|███████   | 281243/400000 [00:33<00:13, 8630.79it/s] 71%|███████   | 282119/400000 [00:33<00:13, 8666.48it/s] 71%|███████   | 283003/400000 [00:33<00:13, 8717.04it/s] 71%|███████   | 283876/400000 [00:33<00:13, 8618.31it/s] 71%|███████   | 284744/400000 [00:33<00:13, 8636.04it/s] 71%|███████▏  | 285625/400000 [00:33<00:13, 8685.80it/s] 72%|███████▏  | 286495/400000 [00:33<00:13, 8687.73it/s] 72%|███████▏  | 287365/400000 [00:33<00:12, 8690.02it/s] 72%|███████▏  | 288235/400000 [00:33<00:12, 8650.75it/s] 72%|███████▏  | 289101/400000 [00:33<00:13, 8518.18it/s] 72%|███████▏  | 289988/400000 [00:34<00:12, 8618.19it/s] 73%|███████▎  | 290851/400000 [00:34<00:12, 8450.60it/s] 73%|███████▎  | 291698/400000 [00:34<00:12, 8331.30it/s] 73%|███████▎  | 292533/400000 [00:34<00:13, 8209.11it/s] 73%|███████▎  | 293402/400000 [00:34<00:12, 8347.11it/s] 74%|███████▎  | 294285/400000 [00:34<00:12, 8483.99it/s] 74%|███████▍  | 295150/400000 [00:34<00:12, 8531.94it/s] 74%|███████▍  | 296027/400000 [00:34<00:12, 8600.98it/s] 74%|███████▍  | 296899/400000 [00:34<00:11, 8636.13it/s] 74%|███████▍  | 297764/400000 [00:34<00:11, 8607.02it/s] 75%|███████▍  | 298665/400000 [00:35<00:11, 8723.31it/s] 75%|███████▍  | 299551/400000 [00:35<00:11, 8761.68it/s] 75%|███████▌  | 300428/400000 [00:35<00:11, 8745.65it/s] 75%|███████▌  | 301303/400000 [00:35<00:11, 8705.57it/s] 76%|███████▌  | 302174/400000 [00:35<00:11, 8519.36it/s] 76%|███████▌  | 303070/400000 [00:35<00:11, 8646.57it/s] 76%|███████▌  | 303962/400000 [00:35<00:11, 8726.52it/s] 76%|███████▌  | 304836/400000 [00:35<00:10, 8708.59it/s] 76%|███████▋  | 305708/400000 [00:35<00:10, 8632.95it/s] 77%|███████▋  | 306572/400000 [00:35<00:10, 8569.26it/s] 77%|███████▋  | 307446/400000 [00:36<00:10, 8617.77it/s] 77%|███████▋  | 308320/400000 [00:36<00:10, 8652.24it/s] 77%|███████▋  | 309186/400000 [00:36<00:10, 8651.36it/s] 78%|███████▊  | 310052/400000 [00:36<00:10, 8604.14it/s] 78%|███████▊  | 310913/400000 [00:36<00:10, 8519.50it/s] 78%|███████▊  | 311778/400000 [00:36<00:10, 8556.18it/s] 78%|███████▊  | 312634/400000 [00:36<00:10, 8544.49it/s] 78%|███████▊  | 313517/400000 [00:36<00:10, 8626.51it/s] 79%|███████▊  | 314380/400000 [00:36<00:09, 8565.57it/s] 79%|███████▉  | 315239/400000 [00:36<00:09, 8572.45it/s] 79%|███████▉  | 316117/400000 [00:37<00:09, 8630.91it/s] 79%|███████▉  | 316986/400000 [00:37<00:09, 8646.81it/s] 79%|███████▉  | 317869/400000 [00:37<00:09, 8699.24it/s] 80%|███████▉  | 318740/400000 [00:37<00:09, 8532.65it/s] 80%|███████▉  | 319615/400000 [00:37<00:09, 8595.33it/s] 80%|████████  | 320476/400000 [00:37<00:09, 8551.71it/s] 80%|████████  | 321332/400000 [00:37<00:09, 8506.21it/s] 81%|████████  | 322218/400000 [00:37<00:09, 8608.61it/s] 81%|████████  | 323083/400000 [00:37<00:08, 8615.32it/s] 81%|████████  | 323945/400000 [00:38<00:08, 8594.77it/s] 81%|████████  | 324815/400000 [00:38<00:08, 8625.83it/s] 81%|████████▏ | 325683/400000 [00:38<00:08, 8641.18it/s] 82%|████████▏ | 326554/400000 [00:38<00:08, 8661.60it/s] 82%|████████▏ | 327421/400000 [00:38<00:08, 8651.61it/s] 82%|████████▏ | 328287/400000 [00:38<00:08, 8624.28it/s] 82%|████████▏ | 329159/400000 [00:38<00:08, 8651.26it/s] 83%|████████▎ | 330025/400000 [00:38<00:08, 8579.82it/s] 83%|████████▎ | 330899/400000 [00:38<00:08, 8627.10it/s] 83%|████████▎ | 331762/400000 [00:38<00:07, 8619.68it/s] 83%|████████▎ | 332625/400000 [00:39<00:07, 8542.34it/s] 83%|████████▎ | 333496/400000 [00:39<00:07, 8589.02it/s] 84%|████████▎ | 334356/400000 [00:39<00:07, 8459.95it/s] 84%|████████▍ | 335207/400000 [00:39<00:07, 8472.12it/s] 84%|████████▍ | 336055/400000 [00:39<00:07, 8465.02it/s] 84%|████████▍ | 336902/400000 [00:39<00:07, 8417.27it/s] 84%|████████▍ | 337744/400000 [00:39<00:07, 8392.08it/s] 85%|████████▍ | 338594/400000 [00:39<00:07, 8423.59it/s] 85%|████████▍ | 339440/400000 [00:39<00:07, 8433.74it/s] 85%|████████▌ | 340294/400000 [00:39<00:07, 8464.23it/s] 85%|████████▌ | 341158/400000 [00:40<00:06, 8514.40it/s] 86%|████████▌ | 342029/400000 [00:40<00:06, 8571.13it/s] 86%|████████▌ | 342894/400000 [00:40<00:06, 8594.36it/s] 86%|████████▌ | 343754/400000 [00:40<00:06, 8559.76it/s] 86%|████████▌ | 344620/400000 [00:40<00:06, 8587.87it/s] 86%|████████▋ | 345484/400000 [00:40<00:06, 8601.95it/s] 87%|████████▋ | 346356/400000 [00:40<00:06, 8623.62it/s] 87%|████████▋ | 347219/400000 [00:40<00:06, 8619.97it/s] 87%|████████▋ | 348087/400000 [00:40<00:06, 8635.35it/s] 87%|████████▋ | 348962/400000 [00:40<00:05, 8669.42it/s] 87%|████████▋ | 349830/400000 [00:41<00:05, 8621.04it/s] 88%|████████▊ | 350693/400000 [00:41<00:05, 8551.45it/s] 88%|████████▊ | 351549/400000 [00:41<00:05, 8522.72it/s] 88%|████████▊ | 352412/400000 [00:41<00:05, 8553.21it/s] 88%|████████▊ | 353285/400000 [00:41<00:05, 8604.05it/s] 89%|████████▊ | 354158/400000 [00:41<00:05, 8639.96it/s] 89%|████████▉ | 355037/400000 [00:41<00:05, 8682.00it/s] 89%|████████▉ | 355906/400000 [00:41<00:05, 8679.41it/s] 89%|████████▉ | 356775/400000 [00:41<00:04, 8654.66it/s] 89%|████████▉ | 357641/400000 [00:41<00:04, 8650.83it/s] 90%|████████▉ | 358517/400000 [00:42<00:04, 8681.04it/s] 90%|████████▉ | 359386/400000 [00:42<00:04, 8671.10it/s] 90%|█████████ | 360273/400000 [00:42<00:04, 8727.48it/s] 90%|█████████ | 361157/400000 [00:42<00:04, 8758.18it/s] 91%|█████████ | 362049/400000 [00:42<00:04, 8803.80it/s] 91%|█████████ | 362930/400000 [00:42<00:04, 8802.28it/s] 91%|█████████ | 363811/400000 [00:42<00:04, 8778.62it/s] 91%|█████████ | 364689/400000 [00:42<00:04, 8773.28it/s] 91%|█████████▏| 365567/400000 [00:42<00:03, 8757.77it/s] 92%|█████████▏| 366443/400000 [00:42<00:03, 8707.55it/s] 92%|█████████▏| 367314/400000 [00:43<00:03, 8683.41it/s] 92%|█████████▏| 368192/400000 [00:43<00:03, 8709.22it/s] 92%|█████████▏| 369063/400000 [00:43<00:03, 8704.69it/s] 92%|█████████▏| 369934/400000 [00:43<00:03, 8682.46it/s] 93%|█████████▎| 370803/400000 [00:43<00:03, 8580.36it/s] 93%|█████████▎| 371662/400000 [00:43<00:03, 8567.86it/s] 93%|█████████▎| 372535/400000 [00:43<00:03, 8613.79it/s] 93%|█████████▎| 373408/400000 [00:43<00:03, 8648.30it/s] 94%|█████████▎| 374275/400000 [00:43<00:02, 8653.26it/s] 94%|█████████▍| 375141/400000 [00:43<00:02, 8642.44it/s] 94%|█████████▍| 376006/400000 [00:44<00:02, 8640.90it/s] 94%|█████████▍| 376871/400000 [00:44<00:02, 8635.14it/s] 94%|█████████▍| 377735/400000 [00:44<00:02, 8492.05it/s] 95%|█████████▍| 378585/400000 [00:44<00:02, 8488.55it/s] 95%|█████████▍| 379439/400000 [00:44<00:02, 8503.57it/s] 95%|█████████▌| 380290/400000 [00:44<00:02, 8498.87it/s] 95%|█████████▌| 381147/400000 [00:44<00:02, 8519.64it/s] 96%|█████████▌| 382002/400000 [00:44<00:02, 8527.17it/s] 96%|█████████▌| 382855/400000 [00:44<00:02, 8515.51it/s] 96%|█████████▌| 383707/400000 [00:44<00:01, 8504.34it/s] 96%|█████████▌| 384560/400000 [00:45<00:01, 8509.64it/s] 96%|█████████▋| 385414/400000 [00:45<00:01, 8516.67it/s] 97%|█████████▋| 386274/400000 [00:45<00:01, 8539.71it/s] 97%|█████████▋| 387137/400000 [00:45<00:01, 8564.04it/s] 97%|█████████▋| 387994/400000 [00:45<00:01, 8526.84it/s] 97%|█████████▋| 388847/400000 [00:45<00:01, 8492.59it/s] 97%|█████████▋| 389700/400000 [00:45<00:01, 8503.31it/s] 98%|█████████▊| 390551/400000 [00:45<00:01, 8495.61it/s] 98%|█████████▊| 391420/400000 [00:45<00:01, 8552.16it/s] 98%|█████████▊| 392276/400000 [00:45<00:00, 8548.24it/s] 98%|█████████▊| 393131/400000 [00:46<00:00, 8432.26it/s] 98%|█████████▊| 393975/400000 [00:46<00:00, 8416.85it/s] 99%|█████████▊| 394841/400000 [00:46<00:00, 8485.80it/s] 99%|█████████▉| 395690/400000 [00:46<00:00, 8422.70it/s] 99%|█████████▉| 396558/400000 [00:46<00:00, 8496.99it/s] 99%|█████████▉| 397411/400000 [00:46<00:00, 8506.81it/s]100%|█████████▉| 398276/400000 [00:46<00:00, 8548.93it/s]100%|█████████▉| 399138/400000 [00:46<00:00, 8569.57it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8535.30it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ffbb0144f60> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011514992928811921 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011129575230206136 	 Accuracy: 59

  model saves at 59% accuracy 

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

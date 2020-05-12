
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f717174df98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 12:13:52.224393
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 12:13:52.229180
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 12:13:52.233225
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 12:13:52.237433
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f717d511438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 359782.4375
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 343792.4062
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 282767.5938
Epoch 4/10

1/1 [==============================] - 0s 106ms/step - loss: 229586.8750
Epoch 5/10

1/1 [==============================] - 0s 107ms/step - loss: 177474.1875
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 131259.2188
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 94047.3906
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 64547.0938
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 44065.5742
Epoch 10/10

1/1 [==============================] - 0s 110ms/step - loss: 30788.4023

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.73988676  0.4159964  -0.21391833 -0.23274352 -0.43975502 -0.52214193
  -0.6332595   0.3766001   0.2802047  -0.43654266 -0.9374842  -0.7246464
  -0.06294554  0.15291643 -0.6631775  -0.24841747 -0.64947325  0.18973947
  -0.08980837  0.10811448 -0.78161156  0.22799252  0.2821081  -0.10440341
  -0.11920571  0.40296602  0.5293845  -0.2278038  -0.44230667 -0.02333571
   0.37248445  0.05967113  0.89020663 -0.78558934 -0.50388193 -0.15240753
   0.5895641   0.88188696  0.05021138  0.50362587  0.60827863  0.30225068
   0.02267358  0.05068438  0.18107644 -0.2691342   0.02918708 -0.41441882
  -0.02040957  0.36150253  0.3311232  -0.01440336 -0.19076672  0.060976
  -0.38015682  0.16028687  0.77908015  0.24526384  0.35689625  0.12479696
  -0.15047087  3.2406864   2.7848537   3.4017448   2.8999047   2.6560981
   2.3053894   2.79523     2.1053646   3.4959834   3.0813708   2.4865608
   3.039839    3.1445749   2.7928705   3.1478176   3.5287526   2.0351322
   3.2467177   3.0405436   2.746162    3.7159684   3.1703174   3.9496033
   3.2449763   3.9575388   3.8973718   3.0849905   2.9763613   2.8940167
   3.2951682   3.8634722   3.0211997   2.3107207   2.4709032   3.125981
   3.7602336   2.8312526   2.8471975   3.25605     3.5473223   2.3764265
   2.9402602   2.3698215   3.8728206   3.1450365   3.756016    2.374175
   2.6325297   3.102037    2.6866372   3.2288947   2.7080908   4.0412803
   3.2115574   3.6001422   3.2216759   2.8338816   2.8009021   3.6831698
   0.37789655 -0.14756735 -0.12095132 -0.10479365  0.45740232 -0.66808087
   0.43702617  0.31199405 -0.19399732  0.16470239 -0.4889973  -0.26866323
   0.22585712 -0.06319612 -0.33849972 -0.7225381  -0.03693956  0.22411545
   0.41029298  0.06376627 -0.79605824 -0.5916443  -0.6797402  -0.04708794
   0.2701841   0.40274715 -0.24521908  0.09593603 -0.41114214 -0.4870864
   0.5396819  -0.2500417   0.1627753   0.16478871  0.01065152  0.45443156
  -0.6308219   0.7965054  -0.42562258 -0.87630874  0.29787278 -0.18508884
  -0.3834524  -0.1516809   0.27318332  0.21941999  0.60824484 -0.6719046
   0.5139067   0.68197143  0.7575375   0.06765443  0.3164301   0.28494668
  -0.51689386  0.0856811   0.36720613  0.04678518  0.03986908  0.52863896
   1.818895    1.5637859   0.8343171   0.816831    1.475555    1.0353565
   0.622493    0.76774853  1.3760186   1.8543531   0.65231836  0.93281496
   0.42033732  1.0104607   1.1292857   0.5788986   0.5499383   1.880347
   1.1796577   0.5659267   1.725075    0.7650689   0.9641006   0.9747869
   0.56517875  0.35267556  1.7938848   1.2065252   0.787481    0.8691901
   1.1713494   1.2005528   1.5335901   0.3545813   1.3060367   1.7761111
   0.57961476  0.9424043   0.8339362   0.7823075   1.377995    1.7122376
   0.7253746   1.0689579   0.80107737  0.72080004  0.9764032   0.9652048
   0.66718286  0.5111963   1.5001867   0.5825654   1.5910604   1.3590455
   1.0618035   1.6956009   1.5315641   0.9623471   0.63611436  0.7792613
   0.02015787  4.845887    4.0629554   3.8575487   3.4678245   3.8435826
   4.1491766   3.6519232   3.421424    4.381808    3.7766056   4.483182
   4.512846    3.90419     3.7484422   4.3734035   4.406733    4.713069
   4.4777308   3.3261294   4.0183167   3.8694334   4.3416014   3.4856935
   3.5671077   4.543867    3.6248264   4.710055    3.3587728   3.1123853
   3.0826054   4.1067915   4.2777486   4.7709293   4.3963237   4.322975
   3.628138    3.5554075   4.2595816   4.6606784   3.9343538   4.1302586
   4.014353    4.1275845   3.2574768   3.9096622   4.6474776   3.7790923
   4.641473    4.331343    3.4221535   3.6413116   3.8903384   3.9647293
   4.0135922   4.100654    3.8245292   4.4763513   3.628591    4.3482537
   1.7186649   0.7782937   1.3279039   1.6949378   0.84360486  0.7698473
   0.9403304   1.116689    0.39227134  0.58324456  0.9194122   0.8553783
   1.3144348   0.7151514   1.1937885   0.89728767  0.44076127  1.6387614
   0.66276705  1.6395521   0.99409324  0.49022543  1.7201016   1.0548837
   0.56271684  1.1290543   1.9970503   1.7925421   1.3275249   1.161625
   0.6119067   1.088191    0.77552533  1.5971217   1.4706949   0.81729424
   1.0217695   0.9946628   0.40461516  0.4162408   0.8856843   2.0615442
   1.1620059   0.9961513   0.4567144   1.6259344   0.56317073  0.80123764
   1.5605516   1.0128791   1.1175182   0.830843    1.4898343   0.6960239
   1.152899    1.3107215   0.70513874  0.5725119   1.547622    1.0135415
  -6.0712595   3.0229447  -3.2546666 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 12:14:03.040444
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.5523
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 12:14:03.045211
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9724.39
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 12:14:03.049204
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.7444
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 12:14:03.053204
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -869.888
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140124852363448
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140122339598576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140122339599080
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140122339599584
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140122339600088
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140122339600592

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7179392f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.709948
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.679744
grad_step = 000002, loss = 0.655275
grad_step = 000003, loss = 0.626376
grad_step = 000004, loss = 0.594827
grad_step = 000005, loss = 0.560377
grad_step = 000006, loss = 0.529834
grad_step = 000007, loss = 0.506195
grad_step = 000008, loss = 0.498119
grad_step = 000009, loss = 0.483428
grad_step = 000010, loss = 0.461206
grad_step = 000011, loss = 0.442400
grad_step = 000012, loss = 0.428285
grad_step = 000013, loss = 0.415511
grad_step = 000014, loss = 0.401801
grad_step = 000015, loss = 0.387014
grad_step = 000016, loss = 0.372003
grad_step = 000017, loss = 0.357349
grad_step = 000018, loss = 0.343447
grad_step = 000019, loss = 0.330096
grad_step = 000020, loss = 0.317117
grad_step = 000021, loss = 0.304343
grad_step = 000022, loss = 0.291126
grad_step = 000023, loss = 0.277756
grad_step = 000024, loss = 0.265346
grad_step = 000025, loss = 0.253899
grad_step = 000026, loss = 0.242337
grad_step = 000027, loss = 0.230561
grad_step = 000028, loss = 0.219146
grad_step = 000029, loss = 0.208361
grad_step = 000030, loss = 0.198155
grad_step = 000031, loss = 0.188112
grad_step = 000032, loss = 0.177965
grad_step = 000033, loss = 0.167883
grad_step = 000034, loss = 0.158342
grad_step = 000035, loss = 0.149418
grad_step = 000036, loss = 0.140712
grad_step = 000037, loss = 0.132300
grad_step = 000038, loss = 0.124207
grad_step = 000039, loss = 0.116226
grad_step = 000040, loss = 0.108671
grad_step = 000041, loss = 0.101526
grad_step = 000042, loss = 0.094621
grad_step = 000043, loss = 0.088000
grad_step = 000044, loss = 0.081656
grad_step = 000045, loss = 0.075624
grad_step = 000046, loss = 0.069967
grad_step = 000047, loss = 0.064578
grad_step = 000048, loss = 0.059527
grad_step = 000049, loss = 0.054749
grad_step = 000050, loss = 0.050210
grad_step = 000051, loss = 0.046037
grad_step = 000052, loss = 0.042130
grad_step = 000053, loss = 0.038434
grad_step = 000054, loss = 0.034966
grad_step = 000055, loss = 0.031771
grad_step = 000056, loss = 0.028899
grad_step = 000057, loss = 0.026227
grad_step = 000058, loss = 0.023735
grad_step = 000059, loss = 0.021431
grad_step = 000060, loss = 0.019367
grad_step = 000061, loss = 0.017484
grad_step = 000062, loss = 0.015745
grad_step = 000063, loss = 0.014193
grad_step = 000064, loss = 0.012795
grad_step = 000065, loss = 0.011542
grad_step = 000066, loss = 0.010400
grad_step = 000067, loss = 0.009386
grad_step = 000068, loss = 0.008483
grad_step = 000069, loss = 0.007682
grad_step = 000070, loss = 0.006969
grad_step = 000071, loss = 0.006337
grad_step = 000072, loss = 0.005771
grad_step = 000073, loss = 0.005284
grad_step = 000074, loss = 0.004858
grad_step = 000075, loss = 0.004480
grad_step = 000076, loss = 0.004146
grad_step = 000077, loss = 0.003860
grad_step = 000078, loss = 0.003608
grad_step = 000079, loss = 0.003385
grad_step = 000080, loss = 0.003196
grad_step = 000081, loss = 0.003036
grad_step = 000082, loss = 0.002897
grad_step = 000083, loss = 0.002776
grad_step = 000084, loss = 0.002672
grad_step = 000085, loss = 0.002587
grad_step = 000086, loss = 0.002515
grad_step = 000087, loss = 0.002453
grad_step = 000088, loss = 0.002399
grad_step = 000089, loss = 0.002358
grad_step = 000090, loss = 0.002322
grad_step = 000091, loss = 0.002292
grad_step = 000092, loss = 0.002267
grad_step = 000093, loss = 0.002247
grad_step = 000094, loss = 0.002227
grad_step = 000095, loss = 0.002211
grad_step = 000096, loss = 0.002198
grad_step = 000097, loss = 0.002185
grad_step = 000098, loss = 0.002174
grad_step = 000099, loss = 0.002163
grad_step = 000100, loss = 0.002153
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002143
grad_step = 000102, loss = 0.002133
grad_step = 000103, loss = 0.002124
grad_step = 000104, loss = 0.002115
grad_step = 000105, loss = 0.002106
grad_step = 000106, loss = 0.002096
grad_step = 000107, loss = 0.002088
grad_step = 000108, loss = 0.002079
grad_step = 000109, loss = 0.002070
grad_step = 000110, loss = 0.002061
grad_step = 000111, loss = 0.002052
grad_step = 000112, loss = 0.002044
grad_step = 000113, loss = 0.002036
grad_step = 000114, loss = 0.002028
grad_step = 000115, loss = 0.002020
grad_step = 000116, loss = 0.002013
grad_step = 000117, loss = 0.002007
grad_step = 000118, loss = 0.002003
grad_step = 000119, loss = 0.002003
grad_step = 000120, loss = 0.002012
grad_step = 000121, loss = 0.002043
grad_step = 000122, loss = 0.002095
grad_step = 000123, loss = 0.002166
grad_step = 000124, loss = 0.002159
grad_step = 000125, loss = 0.002070
grad_step = 000126, loss = 0.001966
grad_step = 000127, loss = 0.001964
grad_step = 000128, loss = 0.002034
grad_step = 000129, loss = 0.002056
grad_step = 000130, loss = 0.001998
grad_step = 000131, loss = 0.001935
grad_step = 000132, loss = 0.001948
grad_step = 000133, loss = 0.001995
grad_step = 000134, loss = 0.001991
grad_step = 000135, loss = 0.001943
grad_step = 000136, loss = 0.001915
grad_step = 000137, loss = 0.001934
grad_step = 000138, loss = 0.001958
grad_step = 000139, loss = 0.001946
grad_step = 000140, loss = 0.001914
grad_step = 000141, loss = 0.001899
grad_step = 000142, loss = 0.001910
grad_step = 000143, loss = 0.001925
grad_step = 000144, loss = 0.001918
grad_step = 000145, loss = 0.001898
grad_step = 000146, loss = 0.001884
grad_step = 000147, loss = 0.001885
grad_step = 000148, loss = 0.001894
grad_step = 000149, loss = 0.001897
grad_step = 000150, loss = 0.001890
grad_step = 000151, loss = 0.001877
grad_step = 000152, loss = 0.001867
grad_step = 000153, loss = 0.001864
grad_step = 000154, loss = 0.001867
grad_step = 000155, loss = 0.001870
grad_step = 000156, loss = 0.001871
grad_step = 000157, loss = 0.001869
grad_step = 000158, loss = 0.001863
grad_step = 000159, loss = 0.001857
grad_step = 000160, loss = 0.001851
grad_step = 000161, loss = 0.001845
grad_step = 000162, loss = 0.001840
grad_step = 000163, loss = 0.001837
grad_step = 000164, loss = 0.001834
grad_step = 000165, loss = 0.001831
grad_step = 000166, loss = 0.001829
grad_step = 000167, loss = 0.001828
grad_step = 000168, loss = 0.001829
grad_step = 000169, loss = 0.001834
grad_step = 000170, loss = 0.001852
grad_step = 000171, loss = 0.001895
grad_step = 000172, loss = 0.001999
grad_step = 000173, loss = 0.002168
grad_step = 000174, loss = 0.002375
grad_step = 000175, loss = 0.002308
grad_step = 000176, loss = 0.001995
grad_step = 000177, loss = 0.001814
grad_step = 000178, loss = 0.002003
grad_step = 000179, loss = 0.002130
grad_step = 000180, loss = 0.001922
grad_step = 000181, loss = 0.001813
grad_step = 000182, loss = 0.001971
grad_step = 000183, loss = 0.002001
grad_step = 000184, loss = 0.001840
grad_step = 000185, loss = 0.001823
grad_step = 000186, loss = 0.001936
grad_step = 000187, loss = 0.001920
grad_step = 000188, loss = 0.001799
grad_step = 000189, loss = 0.001833
grad_step = 000190, loss = 0.001905
grad_step = 000191, loss = 0.001847
grad_step = 000192, loss = 0.001785
grad_step = 000193, loss = 0.001831
grad_step = 000194, loss = 0.001862
grad_step = 000195, loss = 0.001804
grad_step = 000196, loss = 0.001782
grad_step = 000197, loss = 0.001819
grad_step = 000198, loss = 0.001828
grad_step = 000199, loss = 0.001783
grad_step = 000200, loss = 0.001776
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001804
grad_step = 000202, loss = 0.001799
grad_step = 000203, loss = 0.001772
grad_step = 000204, loss = 0.001767
grad_step = 000205, loss = 0.001785
grad_step = 000206, loss = 0.001783
grad_step = 000207, loss = 0.001763
grad_step = 000208, loss = 0.001758
grad_step = 000209, loss = 0.001768
grad_step = 000210, loss = 0.001770
grad_step = 000211, loss = 0.001758
grad_step = 000212, loss = 0.001749
grad_step = 000213, loss = 0.001753
grad_step = 000214, loss = 0.001757
grad_step = 000215, loss = 0.001753
grad_step = 000216, loss = 0.001744
grad_step = 000217, loss = 0.001741
grad_step = 000218, loss = 0.001744
grad_step = 000219, loss = 0.001745
grad_step = 000220, loss = 0.001741
grad_step = 000221, loss = 0.001735
grad_step = 000222, loss = 0.001732
grad_step = 000223, loss = 0.001732
grad_step = 000224, loss = 0.001733
grad_step = 000225, loss = 0.001732
grad_step = 000226, loss = 0.001728
grad_step = 000227, loss = 0.001724
grad_step = 000228, loss = 0.001721
grad_step = 000229, loss = 0.001720
grad_step = 000230, loss = 0.001720
grad_step = 000231, loss = 0.001719
grad_step = 000232, loss = 0.001718
grad_step = 000233, loss = 0.001715
grad_step = 000234, loss = 0.001712
grad_step = 000235, loss = 0.001709
grad_step = 000236, loss = 0.001707
grad_step = 000237, loss = 0.001706
grad_step = 000238, loss = 0.001705
grad_step = 000239, loss = 0.001703
grad_step = 000240, loss = 0.001703
grad_step = 000241, loss = 0.001703
grad_step = 000242, loss = 0.001703
grad_step = 000243, loss = 0.001705
grad_step = 000244, loss = 0.001710
grad_step = 000245, loss = 0.001718
grad_step = 000246, loss = 0.001732
grad_step = 000247, loss = 0.001754
grad_step = 000248, loss = 0.001793
grad_step = 000249, loss = 0.001840
grad_step = 000250, loss = 0.001909
grad_step = 000251, loss = 0.001940
grad_step = 000252, loss = 0.001941
grad_step = 000253, loss = 0.001846
grad_step = 000254, loss = 0.001738
grad_step = 000255, loss = 0.001680
grad_step = 000256, loss = 0.001704
grad_step = 000257, loss = 0.001767
grad_step = 000258, loss = 0.001796
grad_step = 000259, loss = 0.001768
grad_step = 000260, loss = 0.001707
grad_step = 000261, loss = 0.001670
grad_step = 000262, loss = 0.001680
grad_step = 000263, loss = 0.001714
grad_step = 000264, loss = 0.001739
grad_step = 000265, loss = 0.001730
grad_step = 000266, loss = 0.001700
grad_step = 000267, loss = 0.001669
grad_step = 000268, loss = 0.001658
grad_step = 000269, loss = 0.001667
grad_step = 000270, loss = 0.001685
grad_step = 000271, loss = 0.001696
grad_step = 000272, loss = 0.001693
grad_step = 000273, loss = 0.001680
grad_step = 000274, loss = 0.001662
grad_step = 000275, loss = 0.001650
grad_step = 000276, loss = 0.001647
grad_step = 000277, loss = 0.001651
grad_step = 000278, loss = 0.001658
grad_step = 000279, loss = 0.001664
grad_step = 000280, loss = 0.001666
grad_step = 000281, loss = 0.001663
grad_step = 000282, loss = 0.001657
grad_step = 000283, loss = 0.001649
grad_step = 000284, loss = 0.001642
grad_step = 000285, loss = 0.001636
grad_step = 000286, loss = 0.001633
grad_step = 000287, loss = 0.001631
grad_step = 000288, loss = 0.001629
grad_step = 000289, loss = 0.001628
grad_step = 000290, loss = 0.001627
grad_step = 000291, loss = 0.001627
grad_step = 000292, loss = 0.001627
grad_step = 000293, loss = 0.001630
grad_step = 000294, loss = 0.001637
grad_step = 000295, loss = 0.001653
grad_step = 000296, loss = 0.001684
grad_step = 000297, loss = 0.001743
grad_step = 000298, loss = 0.001829
grad_step = 000299, loss = 0.001965
grad_step = 000300, loss = 0.002057
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002088
grad_step = 000302, loss = 0.001915
grad_step = 000303, loss = 0.001702
grad_step = 000304, loss = 0.001612
grad_step = 000305, loss = 0.001698
grad_step = 000306, loss = 0.001816
grad_step = 000307, loss = 0.001795
grad_step = 000308, loss = 0.001676
grad_step = 000309, loss = 0.001607
grad_step = 000310, loss = 0.001658
grad_step = 000311, loss = 0.001729
grad_step = 000312, loss = 0.001706
grad_step = 000313, loss = 0.001630
grad_step = 000314, loss = 0.001602
grad_step = 000315, loss = 0.001644
grad_step = 000316, loss = 0.001678
grad_step = 000317, loss = 0.001652
grad_step = 000318, loss = 0.001607
grad_step = 000319, loss = 0.001598
grad_step = 000320, loss = 0.001625
grad_step = 000321, loss = 0.001644
grad_step = 000322, loss = 0.001627
grad_step = 000323, loss = 0.001598
grad_step = 000324, loss = 0.001590
grad_step = 000325, loss = 0.001603
grad_step = 000326, loss = 0.001617
grad_step = 000327, loss = 0.001612
grad_step = 000328, loss = 0.001595
grad_step = 000329, loss = 0.001584
grad_step = 000330, loss = 0.001587
grad_step = 000331, loss = 0.001596
grad_step = 000332, loss = 0.001598
grad_step = 000333, loss = 0.001591
grad_step = 000334, loss = 0.001581
grad_step = 000335, loss = 0.001577
grad_step = 000336, loss = 0.001579
grad_step = 000337, loss = 0.001583
grad_step = 000338, loss = 0.001584
grad_step = 000339, loss = 0.001579
grad_step = 000340, loss = 0.001573
grad_step = 000341, loss = 0.001569
grad_step = 000342, loss = 0.001569
grad_step = 000343, loss = 0.001571
grad_step = 000344, loss = 0.001572
grad_step = 000345, loss = 0.001571
grad_step = 000346, loss = 0.001568
grad_step = 000347, loss = 0.001568
grad_step = 000348, loss = 0.001569
grad_step = 000349, loss = 0.001566
grad_step = 000350, loss = 0.001560
grad_step = 000351, loss = 0.001563
grad_step = 000352, loss = 0.001566
grad_step = 000353, loss = 0.001562
grad_step = 000354, loss = 0.001558
grad_step = 000355, loss = 0.001559
grad_step = 000356, loss = 0.001558
grad_step = 000357, loss = 0.001554
grad_step = 000358, loss = 0.001553
grad_step = 000359, loss = 0.001554
grad_step = 000360, loss = 0.001553
grad_step = 000361, loss = 0.001551
grad_step = 000362, loss = 0.001550
grad_step = 000363, loss = 0.001551
grad_step = 000364, loss = 0.001550
grad_step = 000365, loss = 0.001548
grad_step = 000366, loss = 0.001547
grad_step = 000367, loss = 0.001547
grad_step = 000368, loss = 0.001546
grad_step = 000369, loss = 0.001545
grad_step = 000370, loss = 0.001544
grad_step = 000371, loss = 0.001544
grad_step = 000372, loss = 0.001545
grad_step = 000373, loss = 0.001544
grad_step = 000374, loss = 0.001544
grad_step = 000375, loss = 0.001545
grad_step = 000376, loss = 0.001548
grad_step = 000377, loss = 0.001552
grad_step = 000378, loss = 0.001558
grad_step = 000379, loss = 0.001569
grad_step = 000380, loss = 0.001584
grad_step = 000381, loss = 0.001612
grad_step = 000382, loss = 0.001647
grad_step = 000383, loss = 0.001706
grad_step = 000384, loss = 0.001758
grad_step = 000385, loss = 0.001827
grad_step = 000386, loss = 0.001827
grad_step = 000387, loss = 0.001792
grad_step = 000388, loss = 0.001680
grad_step = 000389, loss = 0.001577
grad_step = 000390, loss = 0.001530
grad_step = 000391, loss = 0.001555
grad_step = 000392, loss = 0.001614
grad_step = 000393, loss = 0.001649
grad_step = 000394, loss = 0.001639
grad_step = 000395, loss = 0.001586
grad_step = 000396, loss = 0.001538
grad_step = 000397, loss = 0.001524
grad_step = 000398, loss = 0.001545
grad_step = 000399, loss = 0.001576
grad_step = 000400, loss = 0.001584
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001569
grad_step = 000402, loss = 0.001540
grad_step = 000403, loss = 0.001521
grad_step = 000404, loss = 0.001522
grad_step = 000405, loss = 0.001536
grad_step = 000406, loss = 0.001548
grad_step = 000407, loss = 0.001548
grad_step = 000408, loss = 0.001538
grad_step = 000409, loss = 0.001523
grad_step = 000410, loss = 0.001515
grad_step = 000411, loss = 0.001516
grad_step = 000412, loss = 0.001523
grad_step = 000413, loss = 0.001529
grad_step = 000414, loss = 0.001529
grad_step = 000415, loss = 0.001524
grad_step = 000416, loss = 0.001516
grad_step = 000417, loss = 0.001511
grad_step = 000418, loss = 0.001509
grad_step = 000419, loss = 0.001511
grad_step = 000420, loss = 0.001514
grad_step = 000421, loss = 0.001516
grad_step = 000422, loss = 0.001515
grad_step = 000423, loss = 0.001513
grad_step = 000424, loss = 0.001510
grad_step = 000425, loss = 0.001507
grad_step = 000426, loss = 0.001504
grad_step = 000427, loss = 0.001503
grad_step = 000428, loss = 0.001503
grad_step = 000429, loss = 0.001503
grad_step = 000430, loss = 0.001504
grad_step = 000431, loss = 0.001504
grad_step = 000432, loss = 0.001504
grad_step = 000433, loss = 0.001504
grad_step = 000434, loss = 0.001503
grad_step = 000435, loss = 0.001501
grad_step = 000436, loss = 0.001500
grad_step = 000437, loss = 0.001499
grad_step = 000438, loss = 0.001497
grad_step = 000439, loss = 0.001496
grad_step = 000440, loss = 0.001495
grad_step = 000441, loss = 0.001494
grad_step = 000442, loss = 0.001493
grad_step = 000443, loss = 0.001493
grad_step = 000444, loss = 0.001492
grad_step = 000445, loss = 0.001491
grad_step = 000446, loss = 0.001491
grad_step = 000447, loss = 0.001490
grad_step = 000448, loss = 0.001489
grad_step = 000449, loss = 0.001489
grad_step = 000450, loss = 0.001488
grad_step = 000451, loss = 0.001488
grad_step = 000452, loss = 0.001487
grad_step = 000453, loss = 0.001487
grad_step = 000454, loss = 0.001487
grad_step = 000455, loss = 0.001488
grad_step = 000456, loss = 0.001489
grad_step = 000457, loss = 0.001493
grad_step = 000458, loss = 0.001499
grad_step = 000459, loss = 0.001511
grad_step = 000460, loss = 0.001531
grad_step = 000461, loss = 0.001565
grad_step = 000462, loss = 0.001626
grad_step = 000463, loss = 0.001717
grad_step = 000464, loss = 0.001886
grad_step = 000465, loss = 0.002054
grad_step = 000466, loss = 0.002251
grad_step = 000467, loss = 0.002131
grad_step = 000468, loss = 0.001837
grad_step = 000469, loss = 0.001539
grad_step = 000470, loss = 0.001519
grad_step = 000471, loss = 0.001714
grad_step = 000472, loss = 0.001817
grad_step = 000473, loss = 0.001685
grad_step = 000474, loss = 0.001518
grad_step = 000475, loss = 0.001534
grad_step = 000476, loss = 0.001645
grad_step = 000477, loss = 0.001644
grad_step = 000478, loss = 0.001543
grad_step = 000479, loss = 0.001495
grad_step = 000480, loss = 0.001545
grad_step = 000481, loss = 0.001585
grad_step = 000482, loss = 0.001546
grad_step = 000483, loss = 0.001499
grad_step = 000484, loss = 0.001500
grad_step = 000485, loss = 0.001522
grad_step = 000486, loss = 0.001524
grad_step = 000487, loss = 0.001504
grad_step = 000488, loss = 0.001494
grad_step = 000489, loss = 0.001493
grad_step = 000490, loss = 0.001489
grad_step = 000491, loss = 0.001489
grad_step = 000492, loss = 0.001493
grad_step = 000493, loss = 0.001489
grad_step = 000494, loss = 0.001473
grad_step = 000495, loss = 0.001467
grad_step = 000496, loss = 0.001479
grad_step = 000497, loss = 0.001486
grad_step = 000498, loss = 0.001474
grad_step = 000499, loss = 0.001460
grad_step = 000500, loss = 0.001462
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001472
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

  date_run                              2020-05-12 12:14:27.307503
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.298379
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 12:14:27.313798
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.22528
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 12:14:27.322214
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.172032
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 12:14:27.328296
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.42321
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
0   2020-05-12 12:13:52.224393  ...    mean_absolute_error
1   2020-05-12 12:13:52.229180  ...     mean_squared_error
2   2020-05-12 12:13:52.233225  ...  median_absolute_error
3   2020-05-12 12:13:52.237433  ...               r2_score
4   2020-05-12 12:14:03.040444  ...    mean_absolute_error
5   2020-05-12 12:14:03.045211  ...     mean_squared_error
6   2020-05-12 12:14:03.049204  ...  median_absolute_error
7   2020-05-12 12:14:03.053204  ...               r2_score
8   2020-05-12 12:14:27.307503  ...    mean_absolute_error
9   2020-05-12 12:14:27.313798  ...     mean_squared_error
10  2020-05-12 12:14:27.322214  ...  median_absolute_error
11  2020-05-12 12:14:27.328296  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 145704.79it/s] 77%|███████▋  | 7634944/9912422 [00:00<00:10, 207978.38it/s]9920512it [00:00, 39538948.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 541469.17it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 459095.99it/s]1654784it [00:00, 10506754.03it/s]                         
0it [00:00, ?it/s]8192it [00:00, 201347.43it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f830b1dc9b0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82bdb90e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f830b1e0710> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f830b198e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82bdba0ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82a8925048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82bdb90e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82b1c42e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f830b1dc9b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f830b198e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f82bdba0dd8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fab0f1b9208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=21750fcac75afa18629dc185b80a6e4062e9e3f84a0500ff0a510cbf42eeadf1
  Stored in directory: /tmp/pip-ephem-wheel-cache-v5182n61/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fab05328048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2777088/17464789 [===>..........................] - ETA: 0s
 9994240/17464789 [================>.............] - ETA: 0s
15818752/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 12:15:56.213575: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 12:15:56.218154: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 12:15:56.218314: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a70897e470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 12:15:56.218331: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6225 - accuracy: 0.5029
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
11000/25000 [============>.................] - ETA: 4s - loss: 7.6415 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 4s - loss: 7.6117 - accuracy: 0.5036
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6480 - accuracy: 0.5012
15000/25000 [=================>............] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6800 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6760 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 10s 403us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 12:16:14.284818
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 12:16:14.284818  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 12:16:21.389302: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 12:16:21.395793: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 12:16:21.396034: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a2b7a98e80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 12:16:21.396068: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f471343b898> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1398 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.0727 - val_crf_viterbi_accuracy: 0.6533

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4718f27128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7740 - accuracy: 0.4930 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7356 - accuracy: 0.4955
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6682 - accuracy: 0.4999
11000/25000 [============>.................] - ETA: 4s - loss: 7.6889 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6926 - accuracy: 0.4983
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7061 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 3s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6594 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6454 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6533 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f46a8488c88> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:26:12, 9.80kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:20:13, 13.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:11:20, 19.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:32:22, 28.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:57:44, 40.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.90M/862M [00:01<4:08:40, 57.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:01<2:52:59, 81.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<2:00:49, 116kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<1:24:07, 166kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<58:47, 237kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:02<40:59, 337kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:02<28:40, 480kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<20:01, 683kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:02<14:05, 968kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<09:52, 1.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<07:21, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<07:02, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<06:47, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:04<05:09, 2.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<06:09, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<07:29, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:52, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<04:24, 3.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:24, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<05:52, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<04:26, 2.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<06:12, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<07:02, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:31, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.6M/862M [00:10<04:00, 3.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<13:10, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<10:35, 1.24MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<07:41, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<08:26, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<08:35, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:40, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:46, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<06:04, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<04:32, 2.86MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<06:12, 2.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:59, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<04:02, 3.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<07:22, 1.75MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<06:30, 1.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<04:52, 2.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<06:24, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<07:07, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<05:38, 2.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<05:59, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<05:29, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<04:09, 3.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:41, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:13, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<03:49, 3.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<09:46, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:09, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:02, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<07:07, 1.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:24, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:46, 2.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:02, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:59, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:35, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<04:02, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<18:08, 684kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<12:51, 962kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:50, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:38, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:02, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:35, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:03, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:12, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:32, 2.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:17, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:28, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:51, 2.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:56, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:31, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:01, 3.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<17:26, 692kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:32, 891kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<09:44, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:25, 1.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:55, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:51, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:42, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:20, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:48, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:11, 2.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<14:12, 834kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<11:13, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:10, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:17, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:24, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:26, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:39, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:03, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:56, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:09, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:53, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:27, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:58, 2.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<16:49, 688kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<13:04, 885kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:27, 1.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<09:05, 1.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:38, 1.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:37, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:44, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:19, 2.64MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:30, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:06, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:53, 2.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:53, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:41, 3.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:00, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:00, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:49, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:30, 3.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<15:59, 698kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<12:25, 898kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<08:56, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:38, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:31, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:30, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:41, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:58, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:50, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:02, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:53, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:33, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:46, 2.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<15:49, 687kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<12:17, 885kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<08:50, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:30, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:21, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:27, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:38, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<16:15, 660kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:34, 854kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<09:02, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:36, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:24, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:22, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:36, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:40, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<06:32, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:49, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:40, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:18, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:00, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:36, 2.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:29, 838kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<09:53, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:12, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:15, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:16, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:35, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:00, 2.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<10:39, 969kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:35, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:35, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<05:19, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:51, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:33, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:24, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:42, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:29, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:06, 1.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:45, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:40<03:26, 2.93MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:02, 1.11MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:26, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:26, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:58, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:14, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:53, 2.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:56, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:38, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:25, 2.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:12, 3.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:48, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:15, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<05:18, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:49, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:09, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:49, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:29, 2.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<33:13, 292kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<24:19, 398kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<17:13, 561kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<14:05, 684kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<11:54, 809kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:45, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<06:13, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<10:29, 912kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<08:24, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<06:06, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:18, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:30, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:05, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:39, 2.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<12:38, 745kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:50, 957kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<07:05, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:54, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:15, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:47, 2.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:54, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:51, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:00, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:25, 2.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:22, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:59, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:54, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<02:50, 3.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:41, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:23, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:42, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:14, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:39, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:27, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:12, 2.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<12:44, 701kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<09:52, 905kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<07:06, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:57, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:50, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:16, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<03:46, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<11:00, 799kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<08:40, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<06:16, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:16, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:19, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:54, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:32, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:58, 668kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<09:59, 866kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<07:10, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:56, 1.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:45, 1.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:12, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:44, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<12:56, 658kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:57, 855kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<07:10, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:54, 1.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:37, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:04, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<03:37, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<20:37, 406kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<15:19, 547kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<10:52, 768kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<09:27, 878kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:27, 982kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:41, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:54, 1.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:46, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:57, 2.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<16:50, 489kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<15:26, 534kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:40, 705kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<08:26, 973kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<06:12, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<04:38, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:31, 859kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:58, 819kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:46, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:44, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<03:15, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<14:07, 574kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<12:42, 637kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<09:30, 851kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<06:56, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<05:09, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<03:48, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<2:13:08, 60.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<1:35:58, 83.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<1:07:41, 119kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<47:36, 168kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<33:29, 239kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<23:36, 338kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<34:37, 230kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<26:59, 295kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:28, 409kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<13:52, 572kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<09:55, 798kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<10:06, 782kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<11:53, 664kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<09:28, 833kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:59, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<05:04, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:53, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:47, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<06:35, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:13, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:54, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:57, 2.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:37, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:20, 830kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:50, 989kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:46, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<04:15, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:12, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<12:54, 596kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<11:22, 676kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<08:33, 897kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:12, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:33, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:30, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<09:16, 821kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:30, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:37, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:09, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<03:07, 2.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<17:49, 424kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<14:47, 510kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<10:51, 695kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<07:48, 965kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<05:42, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<04:10, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<11:04, 676kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:19, 607kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<09:48, 763kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:08, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:13, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:53, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<12:32, 591kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<11:03, 670kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:18, 890kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:03, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<04:26, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:15, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<08:51, 829kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:21, 998kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:23, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:59, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:59, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<19:46, 368kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<16:12, 449kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<11:47, 616kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<08:27, 857kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:09, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<04:29, 1.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<09:04, 795kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<10:43, 672kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<08:38, 833kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:18, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:33, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:29, 2.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:52, 1.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:16, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:52, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:38, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:46, 2.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:17, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<08:01, 881kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<06:43, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:00, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:39, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:39, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:05, 988kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<05:59, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:29, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:19, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:23, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:03, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:01, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:59, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:16, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:21, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<08:11, 838kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<06:39, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:53, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:32, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:44, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<10:26, 651kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<10:37, 640kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<08:17, 818kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:03, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<04:21, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:46, 1.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:18, 919kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:57, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:24, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:13, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:23, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<07:01, 947kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:41, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:10, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:02, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:59, 943kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<07:48, 844kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:11, 1.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:31, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<03:17, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:02, 811kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:17, 786kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:39, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:23, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<10:40, 604kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<09:53, 652kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<07:26, 865kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:23, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:51, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:33, 843kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:41, 830kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:52, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:16, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:21, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:14, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:13, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<03:05, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:43, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:31, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:40, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:42, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:27, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:11, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:19, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<01:47, 3.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<22:44, 268kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<17:39, 345kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<12:43, 479kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<08:59, 674kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<08:06, 744kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<07:18, 826kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:30, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<03:56, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:48, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<04:58, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:52, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:47, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:59, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:18, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:23, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:27, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:49, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:06, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:11, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:19, 2.49MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:57, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:20, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:22, 1.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:26, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:34, 1.59MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:02, 1.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<03:09, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:17, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:11, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:51, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:02, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:12, 2.53MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:17, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:35, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:47, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:02, 2.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:53, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:12, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:16, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:23, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:35, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:58, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:07, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:16, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:38, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:55, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:05, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:14, 2.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:37, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:53, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:11, 2.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:05, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:29, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:44, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:00, 2.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:24, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:13, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:25, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:45, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:14, 1.19MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:02, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:03, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:11, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:48, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:21, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:40, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:42, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:03, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:14, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:46, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:42, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:06, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:31, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:10, 926kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:28, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:19, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<02:21, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<14:19, 330kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<10:52, 434kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<07:46, 605kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<05:27, 855kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<07:00, 663kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<05:44, 809kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:13, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:41, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:24, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:33, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<01:50, 2.46MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:59, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:38, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:43, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:38, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:40, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:10, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:20, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:47, 2.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:17, 3.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:28, 788kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:38, 929kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:25, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:05, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:45, 1.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:09, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:31, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:31, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:55, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:23, 2.96MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:26, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:09, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:23, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:19, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:21, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:48, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:18, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:34, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:13, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:24, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:42, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<04:16, 911kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:43, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:45, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:56, 971kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:28, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:35, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:25, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:24, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:50, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:53, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:00, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:45, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:20, 2.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<00:59, 3.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:58, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:44, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:04, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:01, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:04, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:36, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:41, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:49, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:24, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:01, 3.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:20, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<02:54, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:21, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:42, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:01, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:01, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:37, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:45, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:22, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:38, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:17, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:25, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:34, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:14, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<00:53, 3.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<07:31, 408kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<05:50, 525kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<04:11, 729kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<02:55, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<05:44, 522kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:33, 657kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:17, 904kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<02:18, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<06:18, 465kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:56, 593kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<03:34, 815kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:56, 974kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:34, 1.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:54, 1.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:22, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:57, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:52, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:25, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:26, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:30, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:09, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:50, 3.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:44, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:42, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:18, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:20, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:24, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:05, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:10, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:17, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:00, 2.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:06, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:14, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:58, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:03, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:40, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:21, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:02, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:06, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:11, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:55, 2.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:00, 2.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:06, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:52, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:57, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:04, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:50, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:55, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:02, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:48, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:53, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:59, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:47, 2.58MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:52, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:57, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:45, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:50, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:43, 2.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:48, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:42, 2.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:41, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:45, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:39, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:43, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:48, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:38, 2.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:46, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:36, 2.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:25, 3.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:52, 797kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:34, 944kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:09, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:01, 1.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:58, 1.46MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:43, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:24, 3.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:52, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:50, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<00:26, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<02:15, 540kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:47, 677kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:17, 931kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:52, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:49, 627kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:28, 771kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:04, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:44, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:23, 772kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:10, 920kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:51, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:44, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:41, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:31, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:30, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:31, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:24, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:24, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:18, 2.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:12, 3.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:04, 685kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:53, 825kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:38, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:25, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:41, 953kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:36, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:23, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:22, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<02:32, 208kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:51, 282kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:17, 396kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:53, 517kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:41, 651kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:29, 897kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:19, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:21, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:18, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:09, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:03, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 3.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:47, 148kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:33, 203kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:20, 287kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 383kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:05, 496kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 687kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 726/400000 [00:00<00:55, 7257.19it/s]  0%|          | 1444/400000 [00:00<00:55, 7232.46it/s]  1%|          | 2184/400000 [00:00<00:54, 7280.69it/s]  1%|          | 2925/400000 [00:00<00:54, 7318.95it/s]  1%|          | 3676/400000 [00:00<00:53, 7374.16it/s]  1%|          | 4383/400000 [00:00<00:54, 7279.40it/s]  1%|▏         | 5108/400000 [00:00<00:54, 7268.77it/s]  1%|▏         | 5863/400000 [00:00<00:53, 7344.85it/s]  2%|▏         | 6586/400000 [00:00<00:53, 7309.27it/s]  2%|▏         | 7312/400000 [00:01<00:53, 7294.07it/s]  2%|▏         | 8022/400000 [00:01<00:54, 7202.01it/s]  2%|▏         | 8791/400000 [00:01<00:53, 7341.64it/s]  2%|▏         | 9548/400000 [00:01<00:52, 7408.28it/s]  3%|▎         | 10283/400000 [00:01<00:52, 7380.16it/s]  3%|▎         | 11033/400000 [00:01<00:52, 7414.76it/s]  3%|▎         | 11772/400000 [00:01<00:52, 7330.81it/s]  3%|▎         | 12504/400000 [00:01<00:53, 7302.16it/s]  3%|▎         | 13241/400000 [00:01<00:52, 7319.87it/s]  3%|▎         | 13973/400000 [00:01<00:52, 7306.19it/s]  4%|▎         | 14704/400000 [00:02<00:52, 7299.73it/s]  4%|▍         | 15434/400000 [00:02<00:53, 7233.07it/s]  4%|▍         | 16160/400000 [00:02<00:53, 7238.67it/s]  4%|▍         | 16897/400000 [00:02<00:52, 7276.00it/s]  4%|▍         | 17625/400000 [00:02<00:52, 7254.40it/s]  5%|▍         | 18351/400000 [00:02<00:52, 7208.38it/s]  5%|▍         | 19074/400000 [00:02<00:52, 7211.89it/s]  5%|▍         | 19803/400000 [00:02<00:52, 7233.80it/s]  5%|▌         | 20538/400000 [00:02<00:52, 7266.88it/s]  5%|▌         | 21272/400000 [00:02<00:51, 7286.23it/s]  6%|▌         | 22011/400000 [00:03<00:51, 7316.76it/s]  6%|▌         | 22743/400000 [00:03<00:52, 7230.99it/s]  6%|▌         | 23467/400000 [00:03<00:52, 7167.66it/s]  6%|▌         | 24185/400000 [00:03<00:52, 7163.63it/s]  6%|▌         | 24902/400000 [00:03<00:52, 7160.52it/s]  6%|▋         | 25625/400000 [00:03<00:52, 7180.36it/s]  7%|▋         | 26344/400000 [00:03<00:52, 7182.63it/s]  7%|▋         | 27084/400000 [00:03<00:51, 7244.75it/s]  7%|▋         | 27821/400000 [00:03<00:51, 7280.11it/s]  7%|▋         | 28563/400000 [00:03<00:50, 7319.76it/s]  7%|▋         | 29308/400000 [00:04<00:50, 7358.27it/s]  8%|▊         | 30044/400000 [00:04<00:50, 7319.36it/s]  8%|▊         | 30777/400000 [00:04<00:50, 7304.84it/s]  8%|▊         | 31517/400000 [00:04<00:50, 7331.36it/s]  8%|▊         | 32251/400000 [00:04<00:51, 7193.41it/s]  8%|▊         | 32972/400000 [00:04<00:51, 7196.35it/s]  8%|▊         | 33700/400000 [00:04<00:50, 7219.49it/s]  9%|▊         | 34424/400000 [00:04<00:50, 7223.60it/s]  9%|▉         | 35147/400000 [00:04<00:51, 7073.75it/s]  9%|▉         | 35856/400000 [00:04<00:51, 7076.69it/s]  9%|▉         | 36565/400000 [00:05<00:53, 6853.82it/s]  9%|▉         | 37274/400000 [00:05<00:52, 6921.75it/s]  9%|▉         | 37992/400000 [00:05<00:51, 6994.68it/s] 10%|▉         | 38699/400000 [00:05<00:51, 7015.28it/s] 10%|▉         | 39433/400000 [00:05<00:50, 7106.99it/s] 10%|█         | 40156/400000 [00:05<00:50, 7143.11it/s] 10%|█         | 40871/400000 [00:05<00:50, 7085.21it/s] 10%|█         | 41582/400000 [00:05<00:50, 7090.33it/s] 11%|█         | 42320/400000 [00:05<00:49, 7172.91it/s] 11%|█         | 43066/400000 [00:05<00:49, 7256.25it/s] 11%|█         | 43825/400000 [00:06<00:48, 7352.17it/s] 11%|█         | 44561/400000 [00:06<00:48, 7306.36it/s] 11%|█▏        | 45326/400000 [00:06<00:47, 7405.58it/s] 12%|█▏        | 46089/400000 [00:06<00:47, 7468.07it/s] 12%|█▏        | 46841/400000 [00:06<00:47, 7483.49it/s] 12%|█▏        | 47590/400000 [00:06<00:47, 7437.89it/s] 12%|█▏        | 48335/400000 [00:06<00:48, 7254.50it/s] 12%|█▏        | 49084/400000 [00:06<00:47, 7321.15it/s] 12%|█▏        | 49831/400000 [00:06<00:47, 7362.45it/s] 13%|█▎        | 50580/400000 [00:06<00:47, 7397.63it/s] 13%|█▎        | 51347/400000 [00:07<00:46, 7475.92it/s] 13%|█▎        | 52096/400000 [00:07<00:46, 7426.22it/s] 13%|█▎        | 52854/400000 [00:07<00:46, 7470.95it/s] 13%|█▎        | 53617/400000 [00:07<00:46, 7516.82it/s] 14%|█▎        | 54370/400000 [00:07<00:45, 7517.86it/s] 14%|█▍        | 55123/400000 [00:07<00:46, 7489.80it/s] 14%|█▍        | 55873/400000 [00:07<00:46, 7376.62it/s] 14%|█▍        | 56624/400000 [00:07<00:46, 7409.37it/s] 14%|█▍        | 57369/400000 [00:07<00:46, 7420.12it/s] 15%|█▍        | 58114/400000 [00:07<00:46, 7428.77it/s] 15%|█▍        | 58886/400000 [00:08<00:45, 7511.19it/s] 15%|█▍        | 59638/400000 [00:08<00:45, 7467.50it/s] 15%|█▌        | 60386/400000 [00:08<00:45, 7423.06it/s] 15%|█▌        | 61143/400000 [00:08<00:45, 7464.30it/s] 15%|█▌        | 61914/400000 [00:08<00:44, 7536.23it/s] 16%|█▌        | 62694/400000 [00:08<00:44, 7613.26it/s] 16%|█▌        | 63456/400000 [00:08<00:44, 7592.49it/s] 16%|█▌        | 64221/400000 [00:08<00:44, 7607.46it/s] 16%|█▌        | 64982/400000 [00:08<00:44, 7543.17it/s] 16%|█▋        | 65737/400000 [00:08<00:44, 7477.74it/s] 17%|█▋        | 66486/400000 [00:09<00:44, 7454.48it/s] 17%|█▋        | 67232/400000 [00:09<00:45, 7386.07it/s] 17%|█▋        | 67985/400000 [00:09<00:44, 7426.45it/s] 17%|█▋        | 68728/400000 [00:09<00:44, 7404.73it/s] 17%|█▋        | 69469/400000 [00:09<00:45, 7344.48it/s] 18%|█▊        | 70221/400000 [00:09<00:44, 7394.00it/s] 18%|█▊        | 70980/400000 [00:09<00:44, 7449.14it/s] 18%|█▊        | 71749/400000 [00:09<00:43, 7519.39it/s] 18%|█▊        | 72503/400000 [00:09<00:43, 7523.46it/s] 18%|█▊        | 73256/400000 [00:10<00:45, 7172.35it/s] 18%|█▊        | 73998/400000 [00:10<00:45, 7243.32it/s] 19%|█▊        | 74726/400000 [00:10<00:45, 7168.50it/s] 19%|█▉        | 75479/400000 [00:10<00:44, 7272.32it/s] 19%|█▉        | 76235/400000 [00:10<00:44, 7355.66it/s] 19%|█▉        | 76986/400000 [00:10<00:43, 7399.89it/s] 19%|█▉        | 77731/400000 [00:10<00:43, 7414.44it/s] 20%|█▉        | 78474/400000 [00:10<00:43, 7390.29it/s] 20%|█▉        | 79214/400000 [00:10<00:43, 7372.77it/s] 20%|█▉        | 79958/400000 [00:10<00:43, 7392.54it/s] 20%|██        | 80717/400000 [00:11<00:42, 7450.23it/s] 20%|██        | 81467/400000 [00:11<00:42, 7463.98it/s] 21%|██        | 82214/400000 [00:11<00:43, 7370.61it/s] 21%|██        | 82952/400000 [00:11<00:43, 7264.25it/s] 21%|██        | 83697/400000 [00:11<00:43, 7317.56it/s] 21%|██        | 84430/400000 [00:11<00:43, 7286.21it/s] 21%|██▏       | 85160/400000 [00:11<00:43, 7188.73it/s] 21%|██▏       | 85889/400000 [00:11<00:43, 7218.13it/s] 22%|██▏       | 86632/400000 [00:11<00:43, 7278.32it/s] 22%|██▏       | 87361/400000 [00:11<00:43, 7260.93it/s] 22%|██▏       | 88101/400000 [00:12<00:42, 7301.57it/s] 22%|██▏       | 88867/400000 [00:12<00:42, 7403.63it/s] 22%|██▏       | 89608/400000 [00:12<00:42, 7373.18it/s] 23%|██▎       | 90349/400000 [00:12<00:41, 7383.14it/s] 23%|██▎       | 91088/400000 [00:12<00:42, 7329.45it/s] 23%|██▎       | 91822/400000 [00:12<00:42, 7315.38it/s] 23%|██▎       | 92568/400000 [00:12<00:41, 7356.30it/s] 23%|██▎       | 93306/400000 [00:12<00:41, 7361.50it/s] 24%|██▎       | 94069/400000 [00:12<00:41, 7436.67it/s] 24%|██▎       | 94813/400000 [00:12<00:41, 7345.05it/s] 24%|██▍       | 95548/400000 [00:13<00:41, 7315.27it/s] 24%|██▍       | 96280/400000 [00:13<00:41, 7313.20it/s] 24%|██▍       | 97012/400000 [00:13<00:41, 7285.22it/s] 24%|██▍       | 97741/400000 [00:13<00:41, 7284.20it/s] 25%|██▍       | 98486/400000 [00:13<00:41, 7332.87it/s] 25%|██▍       | 99226/400000 [00:13<00:40, 7351.19it/s] 25%|██▍       | 99962/400000 [00:13<00:41, 7287.63it/s] 25%|██▌       | 100691/400000 [00:13<00:41, 7217.62it/s] 25%|██▌       | 101426/400000 [00:13<00:41, 7255.08it/s] 26%|██▌       | 102152/400000 [00:13<00:41, 7200.95it/s] 26%|██▌       | 102873/400000 [00:14<00:41, 7177.81it/s] 26%|██▌       | 103591/400000 [00:14<00:41, 7170.32it/s] 26%|██▌       | 104309/400000 [00:14<00:41, 7125.61it/s] 26%|██▋       | 105044/400000 [00:14<00:41, 7180.38it/s] 26%|██▋       | 105763/400000 [00:14<00:41, 7152.98it/s] 27%|██▋       | 106479/400000 [00:14<00:41, 7104.84it/s] 27%|██▋       | 107227/400000 [00:14<00:40, 7211.11it/s] 27%|██▋       | 107961/400000 [00:14<00:40, 7246.54it/s] 27%|██▋       | 108687/400000 [00:14<00:40, 7237.10it/s] 27%|██▋       | 109413/400000 [00:14<00:40, 7242.19it/s] 28%|██▊       | 110138/400000 [00:15<00:40, 7167.81it/s] 28%|██▊       | 110873/400000 [00:15<00:40, 7221.25it/s] 28%|██▊       | 111596/400000 [00:15<00:40, 7179.80it/s] 28%|██▊       | 112316/400000 [00:15<00:40, 7183.72it/s] 28%|██▊       | 113038/400000 [00:15<00:39, 7194.22it/s] 28%|██▊       | 113786/400000 [00:15<00:39, 7277.63it/s] 29%|██▊       | 114536/400000 [00:15<00:38, 7342.63it/s] 29%|██▉       | 115277/400000 [00:15<00:38, 7360.44it/s] 29%|██▉       | 116014/400000 [00:15<00:39, 7281.18it/s] 29%|██▉       | 116743/400000 [00:15<00:38, 7270.35it/s] 29%|██▉       | 117472/400000 [00:16<00:38, 7274.60it/s] 30%|██▉       | 118222/400000 [00:16<00:38, 7339.08it/s] 30%|██▉       | 118958/400000 [00:16<00:38, 7343.82it/s] 30%|██▉       | 119693/400000 [00:16<00:38, 7232.84it/s] 30%|███       | 120417/400000 [00:16<00:38, 7213.24it/s] 30%|███       | 121140/400000 [00:16<00:38, 7217.51it/s] 30%|███       | 121863/400000 [00:16<00:38, 7146.10it/s] 31%|███       | 122624/400000 [00:16<00:38, 7277.09it/s] 31%|███       | 123353/400000 [00:16<00:38, 7230.97it/s] 31%|███       | 124093/400000 [00:16<00:37, 7278.33it/s] 31%|███       | 124845/400000 [00:17<00:37, 7347.30it/s] 31%|███▏      | 125599/400000 [00:17<00:37, 7403.00it/s] 32%|███▏      | 126355/400000 [00:17<00:36, 7449.12it/s] 32%|███▏      | 127101/400000 [00:17<00:36, 7400.43it/s] 32%|███▏      | 127842/400000 [00:17<00:36, 7358.68it/s] 32%|███▏      | 128579/400000 [00:17<00:36, 7338.80it/s] 32%|███▏      | 129345/400000 [00:17<00:36, 7428.80it/s] 33%|███▎      | 130100/400000 [00:17<00:36, 7463.98it/s] 33%|███▎      | 130847/400000 [00:17<00:36, 7414.28it/s] 33%|███▎      | 131618/400000 [00:17<00:35, 7498.70it/s] 33%|███▎      | 132369/400000 [00:18<00:35, 7469.75it/s] 33%|███▎      | 133117/400000 [00:18<00:36, 7368.85it/s] 33%|███▎      | 133855/400000 [00:18<00:36, 7312.07it/s] 34%|███▎      | 134587/400000 [00:18<00:36, 7242.63it/s] 34%|███▍      | 135316/400000 [00:18<00:36, 7254.06it/s] 34%|███▍      | 136056/400000 [00:18<00:36, 7295.38it/s] 34%|███▍      | 136797/400000 [00:18<00:35, 7324.55it/s] 34%|███▍      | 137572/400000 [00:18<00:35, 7444.74it/s] 35%|███▍      | 138318/400000 [00:18<00:35, 7361.41it/s] 35%|███▍      | 139055/400000 [00:19<00:35, 7346.03it/s] 35%|███▍      | 139791/400000 [00:19<00:35, 7341.72it/s] 35%|███▌      | 140533/400000 [00:19<00:35, 7363.87it/s] 35%|███▌      | 141285/400000 [00:19<00:34, 7408.78it/s] 36%|███▌      | 142027/400000 [00:19<00:34, 7411.53it/s] 36%|███▌      | 142776/400000 [00:19<00:34, 7433.50it/s] 36%|███▌      | 143538/400000 [00:19<00:34, 7485.86it/s] 36%|███▌      | 144289/400000 [00:19<00:34, 7486.88it/s] 36%|███▋      | 145039/400000 [00:19<00:34, 7490.32it/s] 36%|███▋      | 145791/400000 [00:19<00:33, 7498.47it/s] 37%|███▋      | 146541/400000 [00:20<00:33, 7496.37it/s] 37%|███▋      | 147294/400000 [00:20<00:33, 7505.07it/s] 37%|███▋      | 148045/400000 [00:20<00:33, 7484.95it/s] 37%|███▋      | 148794/400000 [00:20<00:33, 7458.95it/s] 37%|███▋      | 149540/400000 [00:20<00:34, 7308.09it/s] 38%|███▊      | 150273/400000 [00:20<00:34, 7314.16it/s] 38%|███▊      | 151005/400000 [00:20<00:34, 7286.36it/s] 38%|███▊      | 151735/400000 [00:20<00:34, 7290.34it/s] 38%|███▊      | 152465/400000 [00:20<00:34, 7248.63it/s] 38%|███▊      | 153191/400000 [00:20<00:34, 7156.69it/s] 38%|███▊      | 153928/400000 [00:21<00:34, 7217.77it/s] 39%|███▊      | 154675/400000 [00:21<00:33, 7290.20it/s] 39%|███▉      | 155405/400000 [00:21<00:33, 7250.98it/s] 39%|███▉      | 156131/400000 [00:21<00:34, 7169.76it/s] 39%|███▉      | 156849/400000 [00:21<00:34, 7105.21it/s] 39%|███▉      | 157560/400000 [00:21<00:34, 7048.05it/s] 40%|███▉      | 158266/400000 [00:21<00:34, 7008.78it/s] 40%|███▉      | 158977/400000 [00:21<00:34, 7037.40it/s] 40%|███▉      | 159723/400000 [00:21<00:33, 7158.20it/s] 40%|████      | 160440/400000 [00:21<00:33, 7160.03it/s] 40%|████      | 161187/400000 [00:22<00:32, 7249.28it/s] 40%|████      | 161943/400000 [00:22<00:32, 7338.37it/s] 41%|████      | 162703/400000 [00:22<00:32, 7411.66it/s] 41%|████      | 163451/400000 [00:22<00:31, 7429.41it/s] 41%|████      | 164195/400000 [00:22<00:32, 7322.44it/s] 41%|████      | 164937/400000 [00:22<00:31, 7349.26it/s] 41%|████▏     | 165673/400000 [00:22<00:31, 7352.30it/s] 42%|████▏     | 166417/400000 [00:22<00:31, 7376.78it/s] 42%|████▏     | 167155/400000 [00:22<00:31, 7345.16it/s] 42%|████▏     | 167890/400000 [00:22<00:31, 7291.92it/s] 42%|████▏     | 168644/400000 [00:23<00:31, 7364.31it/s] 42%|████▏     | 169400/400000 [00:23<00:31, 7420.61it/s] 43%|████▎     | 170143/400000 [00:23<00:31, 7383.02it/s] 43%|████▎     | 170889/400000 [00:23<00:30, 7404.41it/s] 43%|████▎     | 171630/400000 [00:23<00:31, 7321.07it/s] 43%|████▎     | 172391/400000 [00:23<00:30, 7404.76it/s] 43%|████▎     | 173132/400000 [00:23<00:30, 7394.16it/s] 43%|████▎     | 173872/400000 [00:23<00:30, 7352.59it/s] 44%|████▎     | 174608/400000 [00:23<00:30, 7322.18it/s] 44%|████▍     | 175341/400000 [00:23<00:30, 7301.86it/s] 44%|████▍     | 176095/400000 [00:24<00:30, 7368.91it/s] 44%|████▍     | 176849/400000 [00:24<00:30, 7418.53it/s] 44%|████▍     | 177601/400000 [00:24<00:29, 7444.20it/s] 45%|████▍     | 178350/400000 [00:24<00:29, 7456.85it/s] 45%|████▍     | 179096/400000 [00:24<00:30, 7347.34it/s] 45%|████▍     | 179832/400000 [00:24<00:30, 7293.64it/s] 45%|████▌     | 180569/400000 [00:24<00:29, 7315.98it/s] 45%|████▌     | 181325/400000 [00:24<00:29, 7387.15it/s] 46%|████▌     | 182065/400000 [00:24<00:30, 7241.99it/s] 46%|████▌     | 182791/400000 [00:24<00:30, 7103.60it/s] 46%|████▌     | 183514/400000 [00:25<00:30, 7140.54it/s] 46%|████▌     | 184239/400000 [00:25<00:30, 7171.36it/s] 46%|████▌     | 184957/400000 [00:25<00:29, 7171.67it/s] 46%|████▋     | 185675/400000 [00:25<00:30, 7109.53it/s] 47%|████▋     | 186387/400000 [00:25<00:30, 7067.84it/s] 47%|████▋     | 187095/400000 [00:25<00:30, 7065.83it/s] 47%|████▋     | 187817/400000 [00:25<00:29, 7109.08it/s] 47%|████▋     | 188565/400000 [00:25<00:29, 7215.10it/s] 47%|████▋     | 189316/400000 [00:25<00:28, 7295.93it/s] 48%|████▊     | 190047/400000 [00:25<00:28, 7283.05it/s] 48%|████▊     | 190790/400000 [00:26<00:28, 7323.92it/s] 48%|████▊     | 191531/400000 [00:26<00:28, 7348.98it/s] 48%|████▊     | 192284/400000 [00:26<00:28, 7402.29it/s] 48%|████▊     | 193044/400000 [00:26<00:27, 7458.12it/s] 48%|████▊     | 193791/400000 [00:26<00:27, 7437.35it/s] 49%|████▊     | 194535/400000 [00:26<00:27, 7373.57it/s] 49%|████▉     | 195273/400000 [00:26<00:28, 7236.46it/s] 49%|████▉     | 196002/400000 [00:26<00:28, 7252.16it/s] 49%|████▉     | 196728/400000 [00:26<00:28, 7252.47it/s] 49%|████▉     | 197454/400000 [00:27<00:28, 7059.98it/s] 50%|████▉     | 198177/400000 [00:27<00:28, 7108.87it/s] 50%|████▉     | 198920/400000 [00:27<00:27, 7200.79it/s] 50%|████▉     | 199681/400000 [00:27<00:27, 7317.15it/s] 50%|█████     | 200426/400000 [00:27<00:27, 7355.85it/s] 50%|█████     | 201163/400000 [00:27<00:27, 7313.47it/s] 50%|█████     | 201896/400000 [00:27<00:27, 7315.60it/s] 51%|█████     | 202647/400000 [00:27<00:26, 7370.43it/s] 51%|█████     | 203385/400000 [00:27<00:26, 7322.60it/s] 51%|█████     | 204118/400000 [00:27<00:26, 7304.25it/s] 51%|█████     | 204849/400000 [00:28<00:26, 7231.85it/s] 51%|█████▏    | 205583/400000 [00:28<00:26, 7263.01it/s] 52%|█████▏    | 206312/400000 [00:28<00:26, 7268.10it/s] 52%|█████▏    | 207040/400000 [00:28<00:26, 7264.65it/s] 52%|█████▏    | 207794/400000 [00:28<00:26, 7341.14it/s] 52%|█████▏    | 208529/400000 [00:28<00:26, 7233.37it/s] 52%|█████▏    | 209274/400000 [00:28<00:26, 7295.18it/s] 53%|█████▎    | 210036/400000 [00:28<00:25, 7385.94it/s] 53%|█████▎    | 210776/400000 [00:28<00:25, 7389.52it/s] 53%|█████▎    | 211516/400000 [00:28<00:25, 7364.16it/s] 53%|█████▎    | 212253/400000 [00:29<00:25, 7330.38it/s] 53%|█████▎    | 212987/400000 [00:29<00:25, 7315.32it/s] 53%|█████▎    | 213732/400000 [00:29<00:25, 7354.84it/s] 54%|█████▎    | 214468/400000 [00:29<00:25, 7349.29it/s] 54%|█████▍    | 215209/400000 [00:29<00:25, 7366.80it/s] 54%|█████▍    | 215946/400000 [00:29<00:25, 7314.80it/s] 54%|█████▍    | 216678/400000 [00:29<00:25, 7278.41it/s] 54%|█████▍    | 217427/400000 [00:29<00:24, 7340.43it/s] 55%|█████▍    | 218188/400000 [00:29<00:24, 7418.85it/s] 55%|█████▍    | 218931/400000 [00:29<00:24, 7403.05it/s] 55%|█████▍    | 219672/400000 [00:30<00:25, 7153.87it/s] 55%|█████▌    | 220420/400000 [00:30<00:24, 7246.94it/s] 55%|█████▌    | 221156/400000 [00:30<00:24, 7279.33it/s] 55%|█████▌    | 221900/400000 [00:30<00:24, 7326.07it/s] 56%|█████▌    | 222634/400000 [00:30<00:24, 7294.10it/s] 56%|█████▌    | 223365/400000 [00:30<00:24, 7203.54it/s] 56%|█████▌    | 224092/400000 [00:30<00:24, 7223.24it/s] 56%|█████▌    | 224827/400000 [00:30<00:24, 7258.74it/s] 56%|█████▋    | 225576/400000 [00:30<00:23, 7324.78it/s] 57%|█████▋    | 226309/400000 [00:30<00:24, 7135.44it/s] 57%|█████▋    | 227031/400000 [00:31<00:24, 7158.28it/s] 57%|█████▋    | 227766/400000 [00:31<00:23, 7214.15it/s] 57%|█████▋    | 228508/400000 [00:31<00:23, 7272.49it/s] 57%|█████▋    | 229267/400000 [00:31<00:23, 7363.87it/s] 58%|█████▊    | 230024/400000 [00:31<00:22, 7424.15it/s] 58%|█████▊    | 230768/400000 [00:31<00:23, 7354.46it/s] 58%|█████▊    | 231505/400000 [00:31<00:22, 7352.79it/s] 58%|█████▊    | 232241/400000 [00:31<00:23, 7175.66it/s] 58%|█████▊    | 233005/400000 [00:31<00:22, 7306.34it/s] 58%|█████▊    | 233745/400000 [00:31<00:22, 7334.11it/s] 59%|█████▊    | 234480/400000 [00:32<00:22, 7333.96it/s] 59%|█████▉    | 235234/400000 [00:32<00:22, 7390.89it/s] 59%|█████▉    | 235997/400000 [00:32<00:21, 7460.16it/s] 59%|█████▉    | 236770/400000 [00:32<00:21, 7539.00it/s] 59%|█████▉    | 237525/400000 [00:32<00:21, 7523.87it/s] 60%|█████▉    | 238278/400000 [00:32<00:21, 7457.83it/s] 60%|█████▉    | 239025/400000 [00:32<00:21, 7360.44it/s] 60%|█████▉    | 239762/400000 [00:32<00:21, 7354.09it/s] 60%|██████    | 240523/400000 [00:32<00:21, 7427.33it/s] 60%|██████    | 241281/400000 [00:32<00:21, 7471.46it/s] 61%|██████    | 242029/400000 [00:33<00:21, 7455.70it/s] 61%|██████    | 242775/400000 [00:33<00:21, 7359.07it/s] 61%|██████    | 243512/400000 [00:33<00:21, 7287.80it/s] 61%|██████    | 244242/400000 [00:33<00:21, 7276.77it/s] 61%|██████    | 244989/400000 [00:33<00:21, 7332.76it/s] 61%|██████▏   | 245723/400000 [00:33<00:21, 7207.78it/s] 62%|██████▏   | 246452/400000 [00:33<00:21, 7231.15it/s] 62%|██████▏   | 247211/400000 [00:33<00:20, 7334.88it/s] 62%|██████▏   | 247957/400000 [00:33<00:20, 7370.54it/s] 62%|██████▏   | 248700/400000 [00:34<00:20, 7385.95it/s] 62%|██████▏   | 249439/400000 [00:34<00:20, 7376.30it/s] 63%|██████▎   | 250177/400000 [00:34<00:20, 7343.77it/s] 63%|██████▎   | 250922/400000 [00:34<00:20, 7373.51it/s] 63%|██████▎   | 251660/400000 [00:34<00:20, 7331.55it/s] 63%|██████▎   | 252394/400000 [00:34<00:20, 7331.83it/s] 63%|██████▎   | 253128/400000 [00:34<00:20, 7140.74it/s] 63%|██████▎   | 253844/400000 [00:34<00:20, 7097.99it/s] 64%|██████▎   | 254571/400000 [00:34<00:20, 7148.04it/s] 64%|██████▍   | 255287/400000 [00:34<00:20, 7085.69it/s] 64%|██████▍   | 255997/400000 [00:35<00:20, 7004.35it/s] 64%|██████▍   | 256723/400000 [00:35<00:20, 7077.97it/s] 64%|██████▍   | 257432/400000 [00:35<00:20, 7069.33it/s] 65%|██████▍   | 258170/400000 [00:35<00:19, 7158.55it/s] 65%|██████▍   | 258888/400000 [00:35<00:19, 7160.18it/s] 65%|██████▍   | 259639/400000 [00:35<00:19, 7259.44it/s] 65%|██████▌   | 260366/400000 [00:35<00:19, 7016.33it/s] 65%|██████▌   | 261088/400000 [00:35<00:19, 7075.04it/s] 65%|██████▌   | 261808/400000 [00:35<00:19, 7109.95it/s] 66%|██████▌   | 262521/400000 [00:35<00:19, 7110.99it/s] 66%|██████▌   | 263267/400000 [00:36<00:18, 7210.54it/s] 66%|██████▌   | 263989/400000 [00:36<00:18, 7206.89it/s] 66%|██████▌   | 264733/400000 [00:36<00:18, 7275.07it/s] 66%|██████▋   | 265496/400000 [00:36<00:18, 7375.61it/s] 67%|██████▋   | 266257/400000 [00:36<00:17, 7442.17it/s] 67%|██████▋   | 267002/400000 [00:36<00:17, 7421.65it/s] 67%|██████▋   | 267745/400000 [00:36<00:18, 7338.33it/s] 67%|██████▋   | 268486/400000 [00:36<00:17, 7359.55it/s] 67%|██████▋   | 269235/400000 [00:36<00:17, 7396.07it/s] 67%|██████▋   | 269991/400000 [00:36<00:17, 7442.85it/s] 68%|██████▊   | 270736/400000 [00:37<00:17, 7417.06it/s] 68%|██████▊   | 271478/400000 [00:37<00:17, 7271.84it/s] 68%|██████▊   | 272206/400000 [00:37<00:17, 7171.12it/s] 68%|██████▊   | 272938/400000 [00:37<00:17, 7214.81it/s] 68%|██████▊   | 273662/400000 [00:37<00:17, 7219.65it/s] 69%|██████▊   | 274412/400000 [00:37<00:17, 7298.31it/s] 69%|██████▉   | 275143/400000 [00:37<00:17, 7237.13it/s] 69%|██████▉   | 275868/400000 [00:37<00:17, 7222.97it/s] 69%|██████▉   | 276598/400000 [00:37<00:17, 7244.96it/s] 69%|██████▉   | 277331/400000 [00:37<00:16, 7268.41it/s] 70%|██████▉   | 278073/400000 [00:38<00:16, 7310.97it/s] 70%|██████▉   | 278811/400000 [00:38<00:16, 7330.05it/s] 70%|██████▉   | 279545/400000 [00:38<00:16, 7252.10it/s] 70%|███████   | 280304/400000 [00:38<00:16, 7349.11it/s] 70%|███████   | 281040/400000 [00:38<00:16, 7330.63it/s] 70%|███████   | 281787/400000 [00:38<00:16, 7369.01it/s] 71%|███████   | 282525/400000 [00:38<00:16, 7291.55it/s] 71%|███████   | 283255/400000 [00:38<00:16, 7254.43it/s] 71%|███████   | 283986/400000 [00:38<00:15, 7268.25it/s] 71%|███████   | 284714/400000 [00:38<00:15, 7241.66it/s] 71%|███████▏  | 285450/400000 [00:39<00:15, 7276.03it/s] 72%|███████▏  | 286191/400000 [00:39<00:15, 7314.26it/s] 72%|███████▏  | 286923/400000 [00:39<00:15, 7264.16it/s] 72%|███████▏  | 287667/400000 [00:39<00:15, 7315.06it/s] 72%|███████▏  | 288419/400000 [00:39<00:15, 7375.18it/s] 72%|███████▏  | 289174/400000 [00:39<00:14, 7425.04it/s] 72%|███████▏  | 289917/400000 [00:39<00:14, 7381.44it/s] 73%|███████▎  | 290656/400000 [00:39<00:15, 7217.09it/s] 73%|███████▎  | 291379/400000 [00:39<00:15, 7132.92it/s] 73%|███████▎  | 292094/400000 [00:40<00:15, 6976.61it/s] 73%|███████▎  | 292820/400000 [00:40<00:15, 7058.37it/s] 73%|███████▎  | 293537/400000 [00:40<00:15, 7088.72it/s] 74%|███████▎  | 294259/400000 [00:40<00:14, 7125.51it/s] 74%|███████▍  | 295022/400000 [00:40<00:14, 7268.19it/s] 74%|███████▍  | 295782/400000 [00:40<00:14, 7363.02it/s] 74%|███████▍  | 296530/400000 [00:40<00:13, 7396.28it/s] 74%|███████▍  | 297271/400000 [00:40<00:14, 7337.35it/s] 75%|███████▍  | 298006/400000 [00:40<00:13, 7313.40it/s] 75%|███████▍  | 298758/400000 [00:40<00:13, 7373.23it/s] 75%|███████▍  | 299496/400000 [00:41<00:13, 7367.38it/s] 75%|███████▌  | 300246/400000 [00:41<00:13, 7406.04it/s] 75%|███████▌  | 300987/400000 [00:41<00:13, 7312.63it/s] 75%|███████▌  | 301719/400000 [00:41<00:13, 7294.62it/s] 76%|███████▌  | 302475/400000 [00:41<00:13, 7370.04it/s] 76%|███████▌  | 303226/400000 [00:41<00:13, 7409.33it/s] 76%|███████▌  | 303968/400000 [00:41<00:13, 7316.85it/s] 76%|███████▌  | 304701/400000 [00:41<00:13, 7267.66it/s] 76%|███████▋  | 305429/400000 [00:41<00:13, 7134.31it/s] 77%|███████▋  | 306153/400000 [00:41<00:13, 7165.63it/s] 77%|███████▋  | 306885/400000 [00:42<00:12, 7210.55it/s] 77%|███████▋  | 307607/400000 [00:42<00:12, 7177.15it/s] 77%|███████▋  | 308339/400000 [00:42<00:12, 7218.20it/s] 77%|███████▋  | 309062/400000 [00:42<00:12, 7185.24it/s] 77%|███████▋  | 309798/400000 [00:42<00:12, 7235.34it/s] 78%|███████▊  | 310530/400000 [00:42<00:12, 7259.33it/s] 78%|███████▊  | 311257/400000 [00:42<00:12, 7250.90it/s] 78%|███████▊  | 311983/400000 [00:42<00:12, 7240.05it/s] 78%|███████▊  | 312708/400000 [00:42<00:12, 7166.64it/s] 78%|███████▊  | 313430/400000 [00:42<00:12, 7181.05it/s] 79%|███████▊  | 314173/400000 [00:43<00:11, 7250.65it/s] 79%|███████▊  | 314913/400000 [00:43<00:11, 7291.21it/s] 79%|███████▉  | 315643/400000 [00:43<00:11, 7292.78it/s] 79%|███████▉  | 316373/400000 [00:43<00:11, 7260.46it/s] 79%|███████▉  | 317138/400000 [00:43<00:11, 7371.56it/s] 79%|███████▉  | 317907/400000 [00:43<00:10, 7463.72it/s] 80%|███████▉  | 318655/400000 [00:43<00:11, 7359.55it/s] 80%|███████▉  | 319392/400000 [00:43<00:10, 7328.51it/s] 80%|████████  | 320126/400000 [00:43<00:11, 7253.85it/s] 80%|████████  | 320852/400000 [00:43<00:10, 7209.43it/s] 80%|████████  | 321576/400000 [00:44<00:10, 7217.32it/s] 81%|████████  | 322330/400000 [00:44<00:10, 7309.91it/s] 81%|████████  | 323062/400000 [00:44<00:10, 7297.76it/s] 81%|████████  | 323795/400000 [00:44<00:10, 7305.43it/s] 81%|████████  | 324551/400000 [00:44<00:10, 7378.65it/s] 81%|████████▏ | 325297/400000 [00:44<00:10, 7400.68it/s] 82%|████████▏ | 326048/400000 [00:44<00:09, 7431.70it/s] 82%|████████▏ | 326796/400000 [00:44<00:09, 7445.86it/s] 82%|████████▏ | 327541/400000 [00:44<00:09, 7383.67it/s] 82%|████████▏ | 328280/400000 [00:44<00:09, 7296.98it/s] 82%|████████▏ | 329011/400000 [00:45<00:09, 7225.37it/s] 82%|████████▏ | 329758/400000 [00:45<00:09, 7287.09it/s] 83%|████████▎ | 330488/400000 [00:45<00:09, 7256.95it/s] 83%|████████▎ | 331215/400000 [00:45<00:09, 7212.58it/s] 83%|████████▎ | 331938/400000 [00:45<00:09, 7215.88it/s] 83%|████████▎ | 332714/400000 [00:45<00:09, 7369.68it/s] 83%|████████▎ | 333477/400000 [00:45<00:08, 7443.89it/s] 84%|████████▎ | 334223/400000 [00:45<00:08, 7443.38it/s] 84%|████████▎ | 334968/400000 [00:45<00:09, 7191.01it/s] 84%|████████▍ | 335690/400000 [00:45<00:08, 7151.80it/s] 84%|████████▍ | 336409/400000 [00:46<00:08, 7162.21it/s] 84%|████████▍ | 337127/400000 [00:46<00:08, 7156.25it/s] 84%|████████▍ | 337844/400000 [00:46<00:08, 7098.20it/s] 85%|████████▍ | 338558/400000 [00:46<00:08, 7109.29it/s] 85%|████████▍ | 339320/400000 [00:46<00:08, 7252.97it/s] 85%|████████▌ | 340062/400000 [00:46<00:08, 7300.98it/s] 85%|████████▌ | 340793/400000 [00:46<00:08, 7255.34it/s] 85%|████████▌ | 341535/400000 [00:46<00:08, 7300.50it/s] 86%|████████▌ | 342266/400000 [00:46<00:07, 7279.68it/s] 86%|████████▌ | 343002/400000 [00:46<00:07, 7302.11it/s] 86%|████████▌ | 343744/400000 [00:47<00:07, 7334.96it/s] 86%|████████▌ | 344478/400000 [00:47<00:07, 7306.52it/s] 86%|████████▋ | 345229/400000 [00:47<00:07, 7364.32it/s] 86%|████████▋ | 345966/400000 [00:47<00:07, 7344.18it/s] 87%|████████▋ | 346705/400000 [00:47<00:07, 7357.73it/s] 87%|████████▋ | 347463/400000 [00:47<00:07, 7422.92it/s] 87%|████████▋ | 348207/400000 [00:47<00:06, 7426.60it/s] 87%|████████▋ | 348966/400000 [00:47<00:06, 7472.52it/s] 87%|████████▋ | 349714/400000 [00:47<00:06, 7409.94it/s] 88%|████████▊ | 350463/400000 [00:47<00:06, 7432.48it/s] 88%|████████▊ | 351207/400000 [00:48<00:06, 7327.43it/s] 88%|████████▊ | 351941/400000 [00:48<00:06, 7203.43it/s] 88%|████████▊ | 352668/400000 [00:48<00:06, 7222.43it/s] 88%|████████▊ | 353391/400000 [00:48<00:06, 7129.59it/s] 89%|████████▊ | 354135/400000 [00:48<00:06, 7216.44it/s] 89%|████████▊ | 354889/400000 [00:48<00:06, 7310.44it/s] 89%|████████▉ | 355641/400000 [00:48<00:06, 7371.14it/s] 89%|████████▉ | 356388/400000 [00:48<00:05, 7400.26it/s] 89%|████████▉ | 357129/400000 [00:48<00:05, 7343.90it/s] 89%|████████▉ | 357865/400000 [00:49<00:05, 7347.41it/s] 90%|████████▉ | 358636/400000 [00:49<00:05, 7450.74it/s] 90%|████████▉ | 359382/400000 [00:49<00:05, 7432.13it/s] 90%|█████████ | 360126/400000 [00:49<00:05, 7381.35it/s] 90%|█████████ | 360865/400000 [00:49<00:05, 7294.73it/s] 90%|█████████ | 361595/400000 [00:49<00:05, 7132.67it/s] 91%|█████████ | 362326/400000 [00:49<00:05, 7182.69it/s] 91%|█████████ | 363069/400000 [00:49<00:05, 7252.66it/s] 91%|█████████ | 363828/400000 [00:49<00:04, 7349.32it/s] 91%|█████████ | 364564/400000 [00:49<00:04, 7323.50it/s] 91%|█████████▏| 365297/400000 [00:50<00:04, 7155.10it/s] 92%|█████████▏| 366058/400000 [00:50<00:04, 7284.16it/s] 92%|█████████▏| 366789/400000 [00:50<00:04, 7290.55it/s] 92%|█████████▏| 367527/400000 [00:50<00:04, 7316.30it/s] 92%|█████████▏| 368260/400000 [00:50<00:04, 7171.38it/s] 92%|█████████▏| 369008/400000 [00:50<00:04, 7260.17it/s] 92%|█████████▏| 369765/400000 [00:50<00:04, 7349.62it/s] 93%|█████████▎| 370501/400000 [00:50<00:04, 7326.37it/s] 93%|█████████▎| 371260/400000 [00:50<00:03, 7403.37it/s] 93%|█████████▎| 372002/400000 [00:50<00:03, 7110.26it/s] 93%|█████████▎| 372742/400000 [00:51<00:03, 7194.59it/s] 93%|█████████▎| 373509/400000 [00:51<00:03, 7329.92it/s] 94%|█████████▎| 374250/400000 [00:51<00:03, 7353.06it/s] 94%|█████████▍| 375019/400000 [00:51<00:03, 7449.35it/s] 94%|█████████▍| 375766/400000 [00:51<00:03, 7343.19it/s] 94%|█████████▍| 376522/400000 [00:51<00:03, 7405.52it/s] 94%|█████████▍| 377267/400000 [00:51<00:03, 7416.20it/s] 95%|█████████▍| 378018/400000 [00:51<00:02, 7443.72it/s] 95%|█████████▍| 378767/400000 [00:51<00:02, 7456.70it/s] 95%|█████████▍| 379514/400000 [00:51<00:02, 7223.55it/s] 95%|█████████▌| 380258/400000 [00:52<00:02, 7286.45it/s] 95%|█████████▌| 381020/400000 [00:52<00:02, 7382.80it/s] 95%|█████████▌| 381760/400000 [00:52<00:02, 7385.99it/s] 96%|█████████▌| 382500/400000 [00:52<00:02, 7390.11it/s] 96%|█████████▌| 383240/400000 [00:52<00:02, 7310.61it/s] 96%|█████████▌| 383976/400000 [00:52<00:02, 7323.33it/s] 96%|█████████▌| 384709/400000 [00:52<00:02, 7264.08it/s] 96%|█████████▋| 385436/400000 [00:52<00:02, 7260.02it/s] 97%|█████████▋| 386163/400000 [00:52<00:01, 7230.73it/s] 97%|█████████▋| 386887/400000 [00:52<00:01, 7168.19it/s] 97%|█████████▋| 387641/400000 [00:53<00:01, 7273.97it/s] 97%|█████████▋| 388383/400000 [00:53<00:01, 7316.56it/s] 97%|█████████▋| 389116/400000 [00:53<00:01, 7254.54it/s] 97%|█████████▋| 389852/400000 [00:53<00:01, 7285.39it/s] 98%|█████████▊| 390582/400000 [00:53<00:01, 7289.34it/s] 98%|█████████▊| 391313/400000 [00:53<00:01, 7295.05it/s] 98%|█████████▊| 392050/400000 [00:53<00:01, 7315.87it/s] 98%|█████████▊| 392782/400000 [00:53<00:00, 7239.07it/s] 98%|█████████▊| 393544/400000 [00:53<00:00, 7345.89it/s] 99%|█████████▊| 394280/400000 [00:53<00:00, 7290.85it/s] 99%|█████████▉| 395024/400000 [00:54<00:00, 7333.15it/s] 99%|█████████▉| 395782/400000 [00:54<00:00, 7403.30it/s] 99%|█████████▉| 396559/400000 [00:54<00:00, 7508.07it/s] 99%|█████████▉| 397315/400000 [00:54<00:00, 7522.88it/s]100%|█████████▉| 398068/400000 [00:54<00:00, 7392.44it/s]100%|█████████▉| 398809/400000 [00:54<00:00, 7393.30it/s]100%|█████████▉| 399578/400000 [00:54<00:00, 7478.68it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7305.07it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f46ecbc5588> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011340077854533052 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.0111503146563884 	 Accuracy: 52

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

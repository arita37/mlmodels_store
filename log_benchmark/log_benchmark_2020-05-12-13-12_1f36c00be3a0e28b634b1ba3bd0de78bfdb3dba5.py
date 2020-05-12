
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f9ed291efd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 13:12:36.400547
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 13:12:36.404064
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 13:12:36.407028
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 13:12:36.410107
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f9ede936400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355660.0312
Epoch 2/10

1/1 [==============================] - 0s 120ms/step - loss: 266171.5625
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 184506.6875
Epoch 4/10

1/1 [==============================] - 0s 95ms/step - loss: 109479.2188
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 59825.2812
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 33747.8086
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 20370.1426
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 13083.5098
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 8971.5596
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 6626.0166

  #### Inference Need return ypred, ytrue ######################### 
[[-0.11374754  1.9164581  -1.6244726   0.82700604 -0.07917467 -0.28302062
   0.28061107  1.3117858  -1.78302     0.5939134   0.5711891  -0.19468582
  -0.46566218 -1.6650299  -1.0575507  -0.9924946   2.4702108  -0.03198412
   1.6661977  -2.2763958   0.62079185  1.8130767  -1.0660923  -0.5394348
  -0.19544615  0.19088039 -0.14401749  0.6228577   1.3424168   0.894396
   0.863066    0.4699359  -0.5016678   1.3656833   0.18772417  1.445525
   1.3722944   0.71529526 -0.5873972  -0.36230612  1.3637353  -0.277369
   0.5454478   0.55886316  0.02128035  0.26908955 -0.3879063  -0.53862333
   1.998624    0.68561786 -0.73951054  0.11477826 -0.45658377  2.243002
  -0.13819048 -0.7575425   1.1100496   0.9966636   1.1151509  -0.9214026
   0.04245088  9.111057    8.678626    6.6554046   7.8318624   8.609141
   8.448514    9.302136    8.287822    8.449511    7.613891    7.770522
   8.4239025   8.199217    9.503828    8.904345    8.194534    8.121483
   7.6461873   9.196785    7.648486    7.765791    9.020149    8.436006
   7.8308883   9.396455    8.674033    8.587778    9.601762    9.353144
   9.598153    7.523126    9.889245    7.600312    9.201715    8.507929
   9.393646    6.9806767   8.477368    7.587555    8.077452    8.518838
   8.535354    6.8520193  10.208998    6.8040605   7.361597    8.375921
   7.605285    9.011854    7.9435377   8.467538    9.017412    8.505027
   6.6638217   7.976422    8.963028    8.263742    7.467921    8.154227
   0.31600073  0.6444777  -0.7387005   1.7631096   1.1745601   1.1529578
  -1.0150317   0.6630021  -0.33287713  1.0825883   0.76361454  0.26258984
   0.82481396 -0.39003262  1.2218273   0.727226    1.1452379   0.36708862
  -1.4535023   0.96903974  0.25358236  0.8188545   1.5922198   1.290466
  -0.32191396 -0.7861823   0.45529714 -1.6501124  -0.97973996 -0.37539914
  -1.8460271  -1.7950494   0.10323559  0.49494117  0.9993015  -0.47905785
   0.22912772  0.35293108  0.65650344  0.6843714  -0.08876836  1.2205112
  -2.208514    0.01597625  1.3242517  -1.5413291  -1.1216881   0.585148
  -0.7400569  -0.30050933  0.20315385 -0.40507698  0.8087137   0.32866335
   1.5321398   0.54355806  0.40549144 -0.5824121  -1.8216672   1.0394816
   0.84315604  2.0103302   1.4408789   0.22704113  2.335977    0.65446997
   0.664351    0.1366263   0.35063535  0.8878444   0.95705986  1.9479944
   0.90571856  0.52352715  0.69395334  2.8644874   0.8483254   1.1484358
   1.3330257   2.1953719   0.40019405  0.3134458   0.95941675  0.7159383
   1.1962539   0.264426    0.42329025  1.4358319   3.0629034   1.6670197
   0.8430344   1.8304563   1.1191733   1.3519655   0.3086114   0.47719556
   0.90198034  1.6161473   0.7912882   0.37409514  0.14406341  2.5942788
   0.33681536  0.62409866  1.270257    3.1400275   2.2715964   0.45818728
   0.75010085  0.4607544   0.42842567  1.5060906   0.28111887  0.38154066
   0.9525576   0.47446048  0.65826595  0.347798    0.6979391   1.7921017
   0.08253485  8.701519    9.4079275   7.5711856   8.283772    9.826893
  10.295395    7.6938176   9.14691     8.573421    7.3827906   8.045131
   7.5476093   9.890559    9.718511    7.4291124   8.237091    8.401817
   9.435943    8.526276    7.7648263   7.2631083   6.8717194   7.822482
   8.385197    8.479078    7.841143    8.875754    8.752834    8.520817
   9.238584    8.57383     7.7098613   8.985185    8.133994    9.879804
   7.4719973   8.274654    7.4247475   9.44137     8.470338    8.701616
   7.690975    7.9188204   8.597642    8.590746    9.676419    9.351716
   8.827958    8.672072    9.592709    8.354854    8.679102    6.771443
   7.958464    8.83392     8.390681    7.6441207   9.1456      8.187738
   1.7285721   0.40359747  1.4851937   0.88751215  0.3869177   0.5610462
   0.36274433  1.2180336   3.2824254   0.8406425   0.85958445  0.710608
   3.0883327   0.91552734  1.4949894   1.4880972   0.49831724  1.2847431
   0.22370905  2.4740114   0.82227194  1.1007524   1.2488954   0.16711605
   1.4932925   0.24027765  1.1909881   0.32764107  1.5301371   0.29771942
   0.28203166  2.0925817   0.4063176   0.52822095  0.2051192   0.33622324
   0.793175    0.9741848   0.8523639   1.3408897   0.17842281  1.1078426
   0.8722031   0.92705     0.2534392   0.77819276  0.8135642   0.7312967
   0.894987    0.98619807  0.7410463   0.57546496  0.99724513  0.23955196
   1.7870665   1.2961193   1.5105507   0.56979334  0.22177112  0.3185755
  -7.921446    6.179842   -3.9727595 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 13:12:47.854464
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.8744
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 13:12:47.858839
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8833.64
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 13:12:47.862399
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4685
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 13:12:47.866440
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -790.115
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140319757648056
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140318816071976
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140318816072480
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140318816072984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140318816073488
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140318816073992

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f9eda7b9ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.560137
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.525838
grad_step = 000002, loss = 0.498098
grad_step = 000003, loss = 0.467944
grad_step = 000004, loss = 0.433994
grad_step = 000005, loss = 0.398177
grad_step = 000006, loss = 0.374466
grad_step = 000007, loss = 0.379240
grad_step = 000008, loss = 0.367501
grad_step = 000009, loss = 0.343651
grad_step = 000010, loss = 0.327248
grad_step = 000011, loss = 0.317843
grad_step = 000012, loss = 0.310183
grad_step = 000013, loss = 0.301268
grad_step = 000014, loss = 0.290806
grad_step = 000015, loss = 0.279531
grad_step = 000016, loss = 0.268885
grad_step = 000017, loss = 0.260128
grad_step = 000018, loss = 0.252468
grad_step = 000019, loss = 0.243736
grad_step = 000020, loss = 0.233881
grad_step = 000021, loss = 0.224488
grad_step = 000022, loss = 0.216068
grad_step = 000023, loss = 0.207946
grad_step = 000024, loss = 0.199697
grad_step = 000025, loss = 0.191505
grad_step = 000026, loss = 0.183697
grad_step = 000027, loss = 0.176286
grad_step = 000028, loss = 0.168936
grad_step = 000029, loss = 0.161573
grad_step = 000030, loss = 0.154634
grad_step = 000031, loss = 0.148291
grad_step = 000032, loss = 0.141966
grad_step = 000033, loss = 0.135279
grad_step = 000034, loss = 0.128703
grad_step = 000035, loss = 0.122787
grad_step = 000036, loss = 0.117431
grad_step = 000037, loss = 0.112063
grad_step = 000038, loss = 0.106460
grad_step = 000039, loss = 0.101066
grad_step = 000040, loss = 0.096248
grad_step = 000041, loss = 0.091749
grad_step = 000042, loss = 0.087190
grad_step = 000043, loss = 0.082675
grad_step = 000044, loss = 0.078461
grad_step = 000045, loss = 0.074476
grad_step = 000046, loss = 0.070573
grad_step = 000047, loss = 0.066838
grad_step = 000048, loss = 0.063328
grad_step = 000049, loss = 0.059928
grad_step = 000050, loss = 0.056628
grad_step = 000051, loss = 0.053541
grad_step = 000052, loss = 0.050627
grad_step = 000053, loss = 0.047758
grad_step = 000054, loss = 0.045008
grad_step = 000055, loss = 0.042466
grad_step = 000056, loss = 0.040038
grad_step = 000057, loss = 0.037652
grad_step = 000058, loss = 0.035399
grad_step = 000059, loss = 0.033314
grad_step = 000060, loss = 0.031321
grad_step = 000061, loss = 0.029388
grad_step = 000062, loss = 0.027558
grad_step = 000063, loss = 0.025849
grad_step = 000064, loss = 0.024233
grad_step = 000065, loss = 0.022706
grad_step = 000066, loss = 0.021258
grad_step = 000067, loss = 0.019878
grad_step = 000068, loss = 0.018590
grad_step = 000069, loss = 0.017398
grad_step = 000070, loss = 0.016255
grad_step = 000071, loss = 0.015172
grad_step = 000072, loss = 0.014182
grad_step = 000073, loss = 0.013255
grad_step = 000074, loss = 0.012375
grad_step = 000075, loss = 0.011562
grad_step = 000076, loss = 0.010805
grad_step = 000077, loss = 0.010092
grad_step = 000078, loss = 0.009436
grad_step = 000079, loss = 0.008823
grad_step = 000080, loss = 0.008254
grad_step = 000081, loss = 0.007733
grad_step = 000082, loss = 0.007245
grad_step = 000083, loss = 0.006795
grad_step = 000084, loss = 0.006387
grad_step = 000085, loss = 0.006003
grad_step = 000086, loss = 0.005649
grad_step = 000087, loss = 0.005329
grad_step = 000088, loss = 0.005034
grad_step = 000089, loss = 0.004762
grad_step = 000090, loss = 0.004514
grad_step = 000091, loss = 0.004286
grad_step = 000092, loss = 0.004079
grad_step = 000093, loss = 0.003889
grad_step = 000094, loss = 0.003714
grad_step = 000095, loss = 0.003557
grad_step = 000096, loss = 0.003412
grad_step = 000097, loss = 0.003280
grad_step = 000098, loss = 0.003161
grad_step = 000099, loss = 0.003051
grad_step = 000100, loss = 0.002952
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002862
grad_step = 000102, loss = 0.002780
grad_step = 000103, loss = 0.002706
grad_step = 000104, loss = 0.002638
grad_step = 000105, loss = 0.002577
grad_step = 000106, loss = 0.002521
grad_step = 000107, loss = 0.002471
grad_step = 000108, loss = 0.002426
grad_step = 000109, loss = 0.002385
grad_step = 000110, loss = 0.002348
grad_step = 000111, loss = 0.002314
grad_step = 000112, loss = 0.002284
grad_step = 000113, loss = 0.002257
grad_step = 000114, loss = 0.002233
grad_step = 000115, loss = 0.002211
grad_step = 000116, loss = 0.002191
grad_step = 000117, loss = 0.002174
grad_step = 000118, loss = 0.002158
grad_step = 000119, loss = 0.002143
grad_step = 000120, loss = 0.002130
grad_step = 000121, loss = 0.002119
grad_step = 000122, loss = 0.002108
grad_step = 000123, loss = 0.002099
grad_step = 000124, loss = 0.002091
grad_step = 000125, loss = 0.002085
grad_step = 000126, loss = 0.002082
grad_step = 000127, loss = 0.002085
grad_step = 000128, loss = 0.002099
grad_step = 000129, loss = 0.002142
grad_step = 000130, loss = 0.002221
grad_step = 000131, loss = 0.002325
grad_step = 000132, loss = 0.002299
grad_step = 000133, loss = 0.002145
grad_step = 000134, loss = 0.002038
grad_step = 000135, loss = 0.002117
grad_step = 000136, loss = 0.002206
grad_step = 000137, loss = 0.002123
grad_step = 000138, loss = 0.002029
grad_step = 000139, loss = 0.002079
grad_step = 000140, loss = 0.002132
grad_step = 000141, loss = 0.002070
grad_step = 000142, loss = 0.002017
grad_step = 000143, loss = 0.002063
grad_step = 000144, loss = 0.002083
grad_step = 000145, loss = 0.002028
grad_step = 000146, loss = 0.002013
grad_step = 000147, loss = 0.002051
grad_step = 000148, loss = 0.002044
grad_step = 000149, loss = 0.002006
grad_step = 000150, loss = 0.002009
grad_step = 000151, loss = 0.002032
grad_step = 000152, loss = 0.002017
grad_step = 000153, loss = 0.001993
grad_step = 000154, loss = 0.002002
grad_step = 000155, loss = 0.002013
grad_step = 000156, loss = 0.001997
grad_step = 000157, loss = 0.001984
grad_step = 000158, loss = 0.001992
grad_step = 000159, loss = 0.001996
grad_step = 000160, loss = 0.001984
grad_step = 000161, loss = 0.001975
grad_step = 000162, loss = 0.001980
grad_step = 000163, loss = 0.001983
grad_step = 000164, loss = 0.001974
grad_step = 000165, loss = 0.001966
grad_step = 000166, loss = 0.001968
grad_step = 000167, loss = 0.001970
grad_step = 000168, loss = 0.001964
grad_step = 000169, loss = 0.001957
grad_step = 000170, loss = 0.001956
grad_step = 000171, loss = 0.001958
grad_step = 000172, loss = 0.001955
grad_step = 000173, loss = 0.001949
grad_step = 000174, loss = 0.001946
grad_step = 000175, loss = 0.001946
grad_step = 000176, loss = 0.001945
grad_step = 000177, loss = 0.001941
grad_step = 000178, loss = 0.001937
grad_step = 000179, loss = 0.001935
grad_step = 000180, loss = 0.001934
grad_step = 000181, loss = 0.001932
grad_step = 000182, loss = 0.001929
grad_step = 000183, loss = 0.001926
grad_step = 000184, loss = 0.001923
grad_step = 000185, loss = 0.001922
grad_step = 000186, loss = 0.001920
grad_step = 000187, loss = 0.001918
grad_step = 000188, loss = 0.001915
grad_step = 000189, loss = 0.001912
grad_step = 000190, loss = 0.001910
grad_step = 000191, loss = 0.001908
grad_step = 000192, loss = 0.001906
grad_step = 000193, loss = 0.001904
grad_step = 000194, loss = 0.001902
grad_step = 000195, loss = 0.001899
grad_step = 000196, loss = 0.001897
grad_step = 000197, loss = 0.001894
grad_step = 000198, loss = 0.001892
grad_step = 000199, loss = 0.001890
grad_step = 000200, loss = 0.001888
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001886
grad_step = 000202, loss = 0.001884
grad_step = 000203, loss = 0.001882
grad_step = 000204, loss = 0.001879
grad_step = 000205, loss = 0.001877
grad_step = 000206, loss = 0.001875
grad_step = 000207, loss = 0.001873
grad_step = 000208, loss = 0.001870
grad_step = 000209, loss = 0.001868
grad_step = 000210, loss = 0.001866
grad_step = 000211, loss = 0.001864
grad_step = 000212, loss = 0.001862
grad_step = 000213, loss = 0.001860
grad_step = 000214, loss = 0.001858
grad_step = 000215, loss = 0.001857
grad_step = 000216, loss = 0.001855
grad_step = 000217, loss = 0.001855
grad_step = 000218, loss = 0.001855
grad_step = 000219, loss = 0.001858
grad_step = 000220, loss = 0.001865
grad_step = 000221, loss = 0.001881
grad_step = 000222, loss = 0.001914
grad_step = 000223, loss = 0.001979
grad_step = 000224, loss = 0.002080
grad_step = 000225, loss = 0.002205
grad_step = 000226, loss = 0.002237
grad_step = 000227, loss = 0.002106
grad_step = 000228, loss = 0.001900
grad_step = 000229, loss = 0.001851
grad_step = 000230, loss = 0.001972
grad_step = 000231, loss = 0.002056
grad_step = 000232, loss = 0.001978
grad_step = 000233, loss = 0.001842
grad_step = 000234, loss = 0.001851
grad_step = 000235, loss = 0.001955
grad_step = 000236, loss = 0.001964
grad_step = 000237, loss = 0.001867
grad_step = 000238, loss = 0.001817
grad_step = 000239, loss = 0.001873
grad_step = 000240, loss = 0.001913
grad_step = 000241, loss = 0.001864
grad_step = 000242, loss = 0.001813
grad_step = 000243, loss = 0.001834
grad_step = 000244, loss = 0.001870
grad_step = 000245, loss = 0.001849
grad_step = 000246, loss = 0.001808
grad_step = 000247, loss = 0.001809
grad_step = 000248, loss = 0.001838
grad_step = 000249, loss = 0.001838
grad_step = 000250, loss = 0.001810
grad_step = 000251, loss = 0.001798
grad_step = 000252, loss = 0.001815
grad_step = 000253, loss = 0.001828
grad_step = 000254, loss = 0.001818
grad_step = 000255, loss = 0.001809
grad_step = 000256, loss = 0.001822
grad_step = 000257, loss = 0.001848
grad_step = 000258, loss = 0.001868
grad_step = 000259, loss = 0.001871
grad_step = 000260, loss = 0.001867
grad_step = 000261, loss = 0.001840
grad_step = 000262, loss = 0.001806
grad_step = 000263, loss = 0.001782
grad_step = 000264, loss = 0.001786
grad_step = 000265, loss = 0.001809
grad_step = 000266, loss = 0.001820
grad_step = 000267, loss = 0.001809
grad_step = 000268, loss = 0.001783
grad_step = 000269, loss = 0.001771
grad_step = 000270, loss = 0.001778
grad_step = 000271, loss = 0.001789
grad_step = 000272, loss = 0.001791
grad_step = 000273, loss = 0.001781
grad_step = 000274, loss = 0.001771
grad_step = 000275, loss = 0.001768
grad_step = 000276, loss = 0.001769
grad_step = 000277, loss = 0.001769
grad_step = 000278, loss = 0.001767
grad_step = 000279, loss = 0.001765
grad_step = 000280, loss = 0.001764
grad_step = 000281, loss = 0.001764
grad_step = 000282, loss = 0.001763
grad_step = 000283, loss = 0.001760
grad_step = 000284, loss = 0.001754
grad_step = 000285, loss = 0.001750
grad_step = 000286, loss = 0.001747
grad_step = 000287, loss = 0.001747
grad_step = 000288, loss = 0.001748
grad_step = 000289, loss = 0.001750
grad_step = 000290, loss = 0.001750
grad_step = 000291, loss = 0.001750
grad_step = 000292, loss = 0.001749
grad_step = 000293, loss = 0.001750
grad_step = 000294, loss = 0.001752
grad_step = 000295, loss = 0.001758
grad_step = 000296, loss = 0.001770
grad_step = 000297, loss = 0.001789
grad_step = 000298, loss = 0.001819
grad_step = 000299, loss = 0.001856
grad_step = 000300, loss = 0.001905
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001938
grad_step = 000302, loss = 0.001963
grad_step = 000303, loss = 0.001972
grad_step = 000304, loss = 0.001965
grad_step = 000305, loss = 0.001933
grad_step = 000306, loss = 0.001862
grad_step = 000307, loss = 0.001778
grad_step = 000308, loss = 0.001734
grad_step = 000309, loss = 0.001753
grad_step = 000310, loss = 0.001810
grad_step = 000311, loss = 0.001850
grad_step = 000312, loss = 0.001835
grad_step = 000313, loss = 0.001776
grad_step = 000314, loss = 0.001723
grad_step = 000315, loss = 0.001715
grad_step = 000316, loss = 0.001744
grad_step = 000317, loss = 0.001776
grad_step = 000318, loss = 0.001780
grad_step = 000319, loss = 0.001754
grad_step = 000320, loss = 0.001723
grad_step = 000321, loss = 0.001710
grad_step = 000322, loss = 0.001718
grad_step = 000323, loss = 0.001734
grad_step = 000324, loss = 0.001740
grad_step = 000325, loss = 0.001731
grad_step = 000326, loss = 0.001715
grad_step = 000327, loss = 0.001703
grad_step = 000328, loss = 0.001702
grad_step = 000329, loss = 0.001710
grad_step = 000330, loss = 0.001718
grad_step = 000331, loss = 0.001720
grad_step = 000332, loss = 0.001716
grad_step = 000333, loss = 0.001705
grad_step = 000334, loss = 0.001695
grad_step = 000335, loss = 0.001689
grad_step = 000336, loss = 0.001688
grad_step = 000337, loss = 0.001691
grad_step = 000338, loss = 0.001695
grad_step = 000339, loss = 0.001696
grad_step = 000340, loss = 0.001694
grad_step = 000341, loss = 0.001690
grad_step = 000342, loss = 0.001686
grad_step = 000343, loss = 0.001684
grad_step = 000344, loss = 0.001683
grad_step = 000345, loss = 0.001684
grad_step = 000346, loss = 0.001686
grad_step = 000347, loss = 0.001687
grad_step = 000348, loss = 0.001689
grad_step = 000349, loss = 0.001691
grad_step = 000350, loss = 0.001695
grad_step = 000351, loss = 0.001704
grad_step = 000352, loss = 0.001719
grad_step = 000353, loss = 0.001746
grad_step = 000354, loss = 0.001795
grad_step = 000355, loss = 0.001871
grad_step = 000356, loss = 0.001982
grad_step = 000357, loss = 0.002085
grad_step = 000358, loss = 0.002135
grad_step = 000359, loss = 0.002049
grad_step = 000360, loss = 0.001883
grad_step = 000361, loss = 0.001746
grad_step = 000362, loss = 0.001717
grad_step = 000363, loss = 0.001777
grad_step = 000364, loss = 0.001824
grad_step = 000365, loss = 0.001797
grad_step = 000366, loss = 0.001732
grad_step = 000367, loss = 0.001705
grad_step = 000368, loss = 0.001731
grad_step = 000369, loss = 0.001752
grad_step = 000370, loss = 0.001736
grad_step = 000371, loss = 0.001697
grad_step = 000372, loss = 0.001678
grad_step = 000373, loss = 0.001695
grad_step = 000374, loss = 0.001717
grad_step = 000375, loss = 0.001714
grad_step = 000376, loss = 0.001683
grad_step = 000377, loss = 0.001653
grad_step = 000378, loss = 0.001651
grad_step = 000379, loss = 0.001671
grad_step = 000380, loss = 0.001690
grad_step = 000381, loss = 0.001685
grad_step = 000382, loss = 0.001663
grad_step = 000383, loss = 0.001641
grad_step = 000384, loss = 0.001636
grad_step = 000385, loss = 0.001647
grad_step = 000386, loss = 0.001659
grad_step = 000387, loss = 0.001662
grad_step = 000388, loss = 0.001653
grad_step = 000389, loss = 0.001640
grad_step = 000390, loss = 0.001631
grad_step = 000391, loss = 0.001630
grad_step = 000392, loss = 0.001634
grad_step = 000393, loss = 0.001639
grad_step = 000394, loss = 0.001640
grad_step = 000395, loss = 0.001636
grad_step = 000396, loss = 0.001630
grad_step = 000397, loss = 0.001625
grad_step = 000398, loss = 0.001622
grad_step = 000399, loss = 0.001621
grad_step = 000400, loss = 0.001622
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001624
grad_step = 000402, loss = 0.001625
grad_step = 000403, loss = 0.001624
grad_step = 000404, loss = 0.001623
grad_step = 000405, loss = 0.001620
grad_step = 000406, loss = 0.001618
grad_step = 000407, loss = 0.001615
grad_step = 000408, loss = 0.001612
grad_step = 000409, loss = 0.001610
grad_step = 000410, loss = 0.001608
grad_step = 000411, loss = 0.001606
grad_step = 000412, loss = 0.001605
grad_step = 000413, loss = 0.001604
grad_step = 000414, loss = 0.001603
grad_step = 000415, loss = 0.001604
grad_step = 000416, loss = 0.001606
grad_step = 000417, loss = 0.001610
grad_step = 000418, loss = 0.001622
grad_step = 000419, loss = 0.001645
grad_step = 000420, loss = 0.001697
grad_step = 000421, loss = 0.001793
grad_step = 000422, loss = 0.001989
grad_step = 000423, loss = 0.002242
grad_step = 000424, loss = 0.002589
grad_step = 000425, loss = 0.002708
grad_step = 000426, loss = 0.002495
grad_step = 000427, loss = 0.002048
grad_step = 000428, loss = 0.001769
grad_step = 000429, loss = 0.001905
grad_step = 000430, loss = 0.002083
grad_step = 000431, loss = 0.002014
grad_step = 000432, loss = 0.001781
grad_step = 000433, loss = 0.001733
grad_step = 000434, loss = 0.001871
grad_step = 000435, loss = 0.001876
grad_step = 000436, loss = 0.001725
grad_step = 000437, loss = 0.001689
grad_step = 000438, loss = 0.001790
grad_step = 000439, loss = 0.001770
grad_step = 000440, loss = 0.001668
grad_step = 000441, loss = 0.001664
grad_step = 000442, loss = 0.001724
grad_step = 000443, loss = 0.001688
grad_step = 000444, loss = 0.001622
grad_step = 000445, loss = 0.001652
grad_step = 000446, loss = 0.001673
grad_step = 000447, loss = 0.001624
grad_step = 000448, loss = 0.001606
grad_step = 000449, loss = 0.001639
grad_step = 000450, loss = 0.001634
grad_step = 000451, loss = 0.001591
grad_step = 000452, loss = 0.001595
grad_step = 000453, loss = 0.001624
grad_step = 000454, loss = 0.001607
grad_step = 000455, loss = 0.001577
grad_step = 000456, loss = 0.001587
grad_step = 000457, loss = 0.001607
grad_step = 000458, loss = 0.001589
grad_step = 000459, loss = 0.001567
grad_step = 000460, loss = 0.001578
grad_step = 000461, loss = 0.001591
grad_step = 000462, loss = 0.001577
grad_step = 000463, loss = 0.001561
grad_step = 000464, loss = 0.001568
grad_step = 000465, loss = 0.001577
grad_step = 000466, loss = 0.001569
grad_step = 000467, loss = 0.001558
grad_step = 000468, loss = 0.001560
grad_step = 000469, loss = 0.001566
grad_step = 000470, loss = 0.001562
grad_step = 000471, loss = 0.001554
grad_step = 000472, loss = 0.001554
grad_step = 000473, loss = 0.001556
grad_step = 000474, loss = 0.001555
grad_step = 000475, loss = 0.001551
grad_step = 000476, loss = 0.001549
grad_step = 000477, loss = 0.001549
grad_step = 000478, loss = 0.001549
grad_step = 000479, loss = 0.001546
grad_step = 000480, loss = 0.001544
grad_step = 000481, loss = 0.001544
grad_step = 000482, loss = 0.001543
grad_step = 000483, loss = 0.001541
grad_step = 000484, loss = 0.001539
grad_step = 000485, loss = 0.001538
grad_step = 000486, loss = 0.001538
grad_step = 000487, loss = 0.001537
grad_step = 000488, loss = 0.001535
grad_step = 000489, loss = 0.001533
grad_step = 000490, loss = 0.001532
grad_step = 000491, loss = 0.001531
grad_step = 000492, loss = 0.001531
grad_step = 000493, loss = 0.001529
grad_step = 000494, loss = 0.001528
grad_step = 000495, loss = 0.001526
grad_step = 000496, loss = 0.001525
grad_step = 000497, loss = 0.001525
grad_step = 000498, loss = 0.001523
grad_step = 000499, loss = 0.001522
grad_step = 000500, loss = 0.001521
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001520
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

  date_run                              2020-05-12 13:13:11.106627
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.259057
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 13:13:11.111834
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.169279
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 13:13:11.118069
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.143196
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 13:13:11.122678
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.57226
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
0   2020-05-12 13:12:36.400547  ...    mean_absolute_error
1   2020-05-12 13:12:36.404064  ...     mean_squared_error
2   2020-05-12 13:12:36.407028  ...  median_absolute_error
3   2020-05-12 13:12:36.410107  ...               r2_score
4   2020-05-12 13:12:47.854464  ...    mean_absolute_error
5   2020-05-12 13:12:47.858839  ...     mean_squared_error
6   2020-05-12 13:12:47.862399  ...  median_absolute_error
7   2020-05-12 13:12:47.866440  ...               r2_score
8   2020-05-12 13:13:11.106627  ...    mean_absolute_error
9   2020-05-12 13:13:11.111834  ...     mean_squared_error
10  2020-05-12 13:13:11.118069  ...  median_absolute_error
11  2020-05-12 13:13:11.122678  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4055040/9912422 [00:00<00:00, 40461302.84it/s]9920512it [00:00, 35702947.53it/s]                             
0it [00:00, ?it/s]32768it [00:00, 600159.62it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479188.59it/s]1654784it [00:00, 11967926.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 177835.31it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8f5fe7fd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc893703f28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8f5f72ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8904fe048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8937000b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8a896de80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc893703e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8a897bf28> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8937000b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8a896de80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc8f5fe7fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f76f840f208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8e147d701253e08cdefeed718b6b03ea7eec66d96f83e68efd30166292a26cbb
  Stored in directory: /tmp/pip-ephem-wheel-cache-ar0s25i3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f76ee57e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3055616/17464789 [====>.........................] - ETA: 0s
11419648/17464789 [==================>...........] - ETA: 0s
16408576/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 13:14:35.894288: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 13:14:35.899051: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 13:14:35.899292: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56188a8c5c40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 13:14:35.899309: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5900 - accuracy: 0.5050
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6535 - accuracy: 0.5009
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
11000/25000 [============>.................] - ETA: 4s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 4s - loss: 7.6245 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6041 - accuracy: 0.5041
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
15000/25000 [=================>............] - ETA: 3s - loss: 7.6257 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6378 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 9s 375us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 13:14:52.184616
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 13:14:52.184616  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-12 13:14:58.659462: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 13:14:58.664855: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 13:14:58.665484: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5595bc3e97f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 13:14:58.665505: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f7f06316be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2930 - crf_viterbi_accuracy: 0.2800 - val_loss: 1.1726 - val_crf_viterbi_accuracy: 0.3333

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7f0da6c128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6462 - accuracy: 0.5013
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.4788 - accuracy: 0.5123
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4397 - accuracy: 0.5148
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.4852 - accuracy: 0.5118
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.4980 - accuracy: 0.5110
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5363 - accuracy: 0.5085
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5542 - accuracy: 0.5073
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5470 - accuracy: 0.5078
11000/25000 [============>.................] - ETA: 4s - loss: 7.6025 - accuracy: 0.5042
12000/25000 [=============>................] - ETA: 3s - loss: 7.6066 - accuracy: 0.5039
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6312 - accuracy: 0.5023
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 2s - loss: 7.6492 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6260 - accuracy: 0.5026
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6326 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6490 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 9s 362us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7e9cfc7c88> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:49:12, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:48:22, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:25:04, 23.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:18:02, 32.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:05:51, 46.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.70M/862M [00:01<3:32:41, 66.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<2:27:59, 95.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<1:43:05, 136kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<1:11:57, 194kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<50:07, 277kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<35:05, 394kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:02<24:29, 561kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<17:11, 796kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:02<12:03, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:02<08:30, 1.59MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:37, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<08:29, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<05:00, 2.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<09:14, 1.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<07:52, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<07:05, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<07:33, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:18, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<04:19, 3.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<06:06, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:56, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<05:25, 2.42MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<03:58, 3.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<08:42, 1.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<07:17, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<05:49, 2.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:15<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<07:29, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<07:53, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<06:07, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<04:28, 2.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<08:25, 1.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<07:13, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<05:22, 2.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:46, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<06:02, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<04:32, 2.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<06:12, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:56, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<05:24, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<03:56, 3.23MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<09:12, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<07:45, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<05:42, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:57, 1.82MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:25, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:50, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:07, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:33, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:12, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:52, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:39, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:12, 2.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:46, 3.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<06:16, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<9:21:12, 22.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<6:33:05, 31.7kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<4:34:11, 45.2kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<3:35:54, 57.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:33:36, 80.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:48:01, 115kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:17:19, 159kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<55:22, 223kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<38:59, 315kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<30:07, 407kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<23:34, 520kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<17:00, 720kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<12:01, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<16:42, 730kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:59, 938kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<09:23, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:23, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:02, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:52, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<04:55, 2.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<31:53, 378kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<23:31, 512kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<16:41, 720kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<14:28, 828kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:34, 953kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:19, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<06:40, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:28, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:18, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:10, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:26, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:48, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:57, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:22, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:00, 2.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:33, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:18, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:59, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:37, 3.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<11:08:13, 17.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<7:48:42, 24.8kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<5:27:39, 35.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<3:51:22, 50.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<2:44:16, 70.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<1:55:27, 100kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:22:19, 140kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<58:47, 196kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<41:21, 277kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<31:31, 363kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<24:24, 469kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<17:38, 647kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<12:24, 916kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<45:32, 250kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<32:52, 346kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<23:18, 487kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<16:24, 689kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<26:16, 430kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<20:46, 544kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<15:07, 746kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<10:41, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<15:00, 749kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<11:41, 961kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<08:27, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<08:25, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:16, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:17, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:32, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:01, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:29, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:28, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:19, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:47, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:14, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:49, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:32, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:26, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:45, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:50, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:24, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:40, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<34:56, 310kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<25:36, 422kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<18:10, 593kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<15:06, 712kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<12:51, 836kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:29, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<06:45, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<10:57, 975kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:48, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<06:26, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:52, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:30, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:58, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<27:49, 379kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<20:35, 512kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<14:37, 719kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:34, 833kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:02, 948kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<08:16, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<05:54, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<35:21, 294kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<25:40, 405kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<18:12, 569kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<15:00, 688kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:41, 814kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:20, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<06:39, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<10:06, 1.02MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:10, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:26, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:41, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:09, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:43, 2.73MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:33, 1.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:47, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:42, 1.77MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:14, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:30, 1.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:00, 2.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:37, 2.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:48, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:32, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:48, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:35, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:01, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:44, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:26, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<33:33, 294kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<24:30, 402kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<17:20, 566kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<14:18, 684kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<12:06, 807kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:59, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<06:22, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<27:47, 349kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<20:29, 474kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<14:31, 667kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<12:19, 783kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:40, 902kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:54, 1.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<05:38, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:28, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:56, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:04, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:40, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:00, 1.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:39, 2.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:20, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<11:26, 824kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:01, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:31, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:39, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:40, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:09, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:43, 2.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<30:55, 301kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<22:29, 413kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<15:55, 582kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<11:14, 822kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<18:58, 486kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<15:15, 604kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<11:09, 825kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<07:53, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:07<26:33, 345kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<19:33, 468kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<13:53, 657kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<11:44, 774kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<10:09, 894kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:35, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<05:24, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<25:16, 357kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<18:38, 484kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<13:12, 681kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<09:20, 959kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<20:26, 438kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<16:07, 555kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<11:40, 766kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<08:19, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:27, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:50, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:30, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:37, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:26, 2.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<02:34, 3.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:02, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:10, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:44, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:24, 2.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:17, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:01, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:24, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:01, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:20, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:11, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:01, 2.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<09:07, 936kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:17, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:16, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:37, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:45, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:24, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:11, 2.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:42, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:36, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:23, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:49, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:54, 2.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:52, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:28, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:30, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:34, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:35, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:03, 2.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:57, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:30, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:34, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:33, 3.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<21:23, 377kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<15:49, 509kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<11:13, 715kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:38, 829kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:22, 954kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<06:16, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<04:26, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<44:44, 177kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<32:08, 246kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<22:35, 349kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<17:31, 448kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<13:56, 563kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<10:08, 772kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<07:08, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<42:26, 183kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<30:22, 256kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<21:22, 363kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<15:00, 514kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<20:34, 375kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<15:10, 507kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<10:45, 714kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<09:17, 822kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<07:19, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<05:16, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:23, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:34, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:23, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:03, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:40, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:37, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:07, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:12, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:19, 3.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<06:40, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:20, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:49, 2.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<11:28, 636kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<09:32, 764kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:00, 1.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<04:58, 1.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:56, 1.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:36, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:04, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:29, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:40, 1.53MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<02:37, 2.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<23:07, 307kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<16:56, 418kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<11:58, 590kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:55, 708kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:40, 914kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:31, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:29, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:34, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:21, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:13, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:16, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:22, 2.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:35, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:38, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:25, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:54, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:07, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:14, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<02:19, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<49:14, 136kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<35:08, 190kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<24:39, 269kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<18:39, 354kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<14:29, 456kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<10:27, 630kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<07:19, 892kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<19:49, 330kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<14:33, 449kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<10:18, 632kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<07:16, 891kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<17:06, 378kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<12:32, 515kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:56, 721kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<06:18, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<11:33, 554kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<09:26, 677kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:56, 920kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<04:54, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<22:27, 282kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<16:23, 386kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<11:34, 544kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<09:26, 663kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<08:03, 777kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:58, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<04:14, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:17, 984kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:03, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:40, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:38, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<6:09:49, 16.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<4:20:02, 23.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<3:01:58, 33.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<2:06:33, 47.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<1:33:21, 64.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<1:05:55, 91.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<46:05, 131kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<33:27, 179kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<24:38, 243kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<17:28, 342kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<12:15, 485kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<10:57, 540kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<08:11, 722kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:51, 1.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:25, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:03, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:51, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:44, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<16:18, 355kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<12:00, 481kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<08:31, 675kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<07:14, 789kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:28, 883kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<05:22, 1.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:59, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:21, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:40, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:42, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:07, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:22, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:36, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<01:57, 2.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:40, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:27, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:50, 2.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:30, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:54, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:19, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<01:40, 3.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:56, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:01, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:56, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:14, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:23, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:36, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:53, 2.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:22, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:54, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:10, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:41, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:58, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:21, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:01<01:41, 3.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<10:10, 500kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<07:39, 664kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<05:28, 926kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<03:52, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<10:19, 487kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<08:17, 606kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<06:01, 832kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<04:17, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:26, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:38, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:39, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:56, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:05, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:25, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:44, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<13:10, 365kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<09:43, 495kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<06:54, 693kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:51, 810kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:06, 928kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:47, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:02, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:37, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:55, 2.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:26, 3.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:49, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:05, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:42, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:48, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:26, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:49, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:16, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:32, 2.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:03, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:22, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:51, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:23, 3.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:09, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:29, 2.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:18, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:50, 2.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:19, 3.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<10:35, 396kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<07:50, 534kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<05:34, 747kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:47, 860kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:11, 982kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:07, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<02:12, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:31, 895kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:35, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:35, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:42, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:44, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:07, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:31, 2.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<12:51, 305kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<09:23, 416kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<06:37, 587kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<05:28, 703kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:14, 908kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:02, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:58, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:52, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:12, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:52, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:24, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:46, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:02, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:14, 2.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:15, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:26, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:47, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:00, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:34, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:07, 3.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:22, 787kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:25, 1.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:27, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:27, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:26, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:52, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:19, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<06:55, 476kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<05:11, 635kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:40, 889kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<03:16, 984kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:57, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:13, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<17:32, 180kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<12:35, 251kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<08:49, 355kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<06:47, 455kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:24, 571kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<03:56, 781kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:45, 1.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<09:06, 332kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<06:40, 451kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<04:42, 635kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:55, 752kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:22, 873kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:30, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:46, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<09:53, 291kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<07:13, 399kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<05:05, 561kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:09, 677kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:12, 877kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:17, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:12, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:07, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:36, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:09, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:46, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:31, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:07, 2.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:21, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:54, 2.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:12, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:21, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:03, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:45, 3.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:01, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:40, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:12, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:23, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:13, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:54, 2.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:09, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:18, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:00, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:43, 3.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:25, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:14, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:54, 2.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:07, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:15, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:42, 3.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:52, 309kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<05:01, 422kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<03:31, 593kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:53, 711kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:27, 836kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:48, 1.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:15, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:34, 434kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<03:23, 583kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<02:24, 815kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:05, 917kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:52, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:58, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:26, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:12, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:52, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:00, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:05, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:35, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:10, 785kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:42, 1.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:12, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:12, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:59, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:43, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:51, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:56, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:43, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:30, 2.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:38, 923kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:17, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:55, 1.59MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:57, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:58, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:31, 2.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<04:29, 305kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<03:16, 416kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:16, 587kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:50, 704kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:34, 828kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:08, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:47, 1.57MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:07, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:55, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:39, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:42, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:44, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:34, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:24, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:43, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:37, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:32, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:35, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 3.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:41, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:25, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:16, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:39, 308kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:55, 421kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<01:19, 593kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:03, 708kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:53, 832kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:39, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:26, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<05:10, 132kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<03:39, 185kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:28, 262kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:46, 345kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<01:22, 445kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:58, 618kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:39, 871kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:38, 855kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:29, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:20, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:19, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:11, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:08, 2.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:05, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:20, 200kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:57, 277kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:36, 392kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:24, 496kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:19, 615kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:13, 841kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:06, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:42, 185kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:29, 258kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:16, 365kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:08, 466kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 624kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 874kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 836/400000 [00:00<00:47, 8356.26it/s]  0%|          | 1597/400000 [00:00<00:49, 8116.77it/s]  1%|          | 2401/400000 [00:00<00:49, 8091.46it/s]  1%|          | 3293/400000 [00:00<00:47, 8321.07it/s]  1%|          | 4197/400000 [00:00<00:46, 8522.52it/s]  1%|▏         | 5023/400000 [00:00<00:46, 8439.49it/s]  1%|▏         | 5871/400000 [00:00<00:46, 8451.17it/s]  2%|▏         | 6796/400000 [00:00<00:45, 8675.71it/s]  2%|▏         | 7669/400000 [00:00<00:45, 8690.76it/s]  2%|▏         | 8629/400000 [00:01<00:43, 8943.69it/s]  2%|▏         | 9505/400000 [00:01<00:43, 8886.91it/s]  3%|▎         | 10383/400000 [00:01<00:44, 8853.23it/s]  3%|▎         | 11264/400000 [00:01<00:43, 8840.06it/s]  3%|▎         | 12142/400000 [00:01<00:44, 8681.46it/s]  3%|▎         | 13006/400000 [00:01<00:46, 8311.06it/s]  3%|▎         | 13838/400000 [00:01<00:47, 8152.36it/s]  4%|▎         | 14655/400000 [00:01<00:47, 8038.68it/s]  4%|▍         | 15460/400000 [00:01<00:48, 7926.65it/s]  4%|▍         | 16267/400000 [00:01<00:48, 7968.58it/s]  4%|▍         | 17090/400000 [00:02<00:47, 8042.98it/s]  4%|▍         | 17896/400000 [00:02<00:48, 7885.36it/s]  5%|▍         | 18686/400000 [00:02<00:49, 7650.58it/s]  5%|▍         | 19479/400000 [00:02<00:49, 7728.60it/s]  5%|▌         | 20254/400000 [00:02<00:49, 7632.48it/s]  5%|▌         | 21019/400000 [00:02<00:49, 7596.53it/s]  5%|▌         | 21780/400000 [00:02<00:50, 7436.94it/s]  6%|▌         | 22526/400000 [00:02<00:50, 7403.95it/s]  6%|▌         | 23268/400000 [00:02<00:51, 7365.46it/s]  6%|▌         | 24062/400000 [00:02<00:49, 7526.67it/s]  6%|▌         | 24881/400000 [00:03<00:48, 7712.08it/s]  6%|▋         | 25668/400000 [00:03<00:48, 7757.76it/s]  7%|▋         | 26446/400000 [00:03<00:48, 7663.98it/s]  7%|▋         | 27239/400000 [00:03<00:48, 7738.61it/s]  7%|▋         | 28044/400000 [00:03<00:47, 7829.24it/s]  7%|▋         | 28828/400000 [00:03<00:47, 7808.93it/s]  7%|▋         | 29628/400000 [00:03<00:47, 7862.04it/s]  8%|▊         | 30427/400000 [00:03<00:46, 7897.52it/s]  8%|▊         | 31263/400000 [00:03<00:45, 8030.51it/s]  8%|▊         | 32100/400000 [00:03<00:45, 8128.94it/s]  8%|▊         | 32914/400000 [00:04<00:45, 8111.12it/s]  8%|▊         | 33726/400000 [00:04<00:45, 8000.01it/s]  9%|▊         | 34527/400000 [00:04<00:47, 7639.98it/s]  9%|▉         | 35295/400000 [00:04<00:48, 7499.49it/s]  9%|▉         | 36106/400000 [00:04<00:47, 7670.70it/s]  9%|▉         | 36899/400000 [00:04<00:46, 7745.97it/s]  9%|▉         | 37745/400000 [00:04<00:45, 7944.33it/s] 10%|▉         | 38549/400000 [00:04<00:45, 7972.78it/s] 10%|▉         | 39383/400000 [00:04<00:44, 8076.91it/s] 10%|█         | 40244/400000 [00:05<00:43, 8228.63it/s] 10%|█         | 41085/400000 [00:05<00:43, 8279.30it/s] 10%|█         | 41915/400000 [00:05<00:44, 7986.04it/s] 11%|█         | 42717/400000 [00:05<00:46, 7744.22it/s] 11%|█         | 43496/400000 [00:05<00:47, 7493.78it/s] 11%|█         | 44283/400000 [00:05<00:46, 7600.87it/s] 11%|█▏        | 45047/400000 [00:05<00:47, 7507.81it/s] 11%|█▏        | 45852/400000 [00:05<00:46, 7662.53it/s] 12%|█▏        | 46621/400000 [00:05<00:46, 7562.58it/s] 12%|█▏        | 47380/400000 [00:05<00:46, 7525.72it/s] 12%|█▏        | 48184/400000 [00:06<00:45, 7671.08it/s] 12%|█▏        | 48974/400000 [00:06<00:45, 7738.28it/s] 12%|█▏        | 49751/400000 [00:06<00:45, 7746.13it/s] 13%|█▎        | 50545/400000 [00:06<00:44, 7803.12it/s] 13%|█▎        | 51406/400000 [00:06<00:43, 8026.92it/s] 13%|█▎        | 52211/400000 [00:06<00:43, 8033.62it/s] 13%|█▎        | 53024/400000 [00:06<00:43, 8061.03it/s] 13%|█▎        | 53832/400000 [00:06<00:43, 8044.74it/s] 14%|█▎        | 54638/400000 [00:06<00:43, 7982.84it/s] 14%|█▍        | 55437/400000 [00:06<00:43, 7970.57it/s] 14%|█▍        | 56259/400000 [00:07<00:42, 8039.96it/s] 14%|█▍        | 57121/400000 [00:07<00:41, 8204.56it/s] 14%|█▍        | 57994/400000 [00:07<00:40, 8354.76it/s] 15%|█▍        | 58831/400000 [00:07<00:42, 8095.71it/s] 15%|█▍        | 59644/400000 [00:07<00:42, 7920.34it/s] 15%|█▌        | 60499/400000 [00:07<00:41, 8099.22it/s] 15%|█▌        | 61368/400000 [00:07<00:40, 8267.51it/s] 16%|█▌        | 62245/400000 [00:07<00:40, 8410.64it/s] 16%|█▌        | 63089/400000 [00:07<00:40, 8329.51it/s] 16%|█▌        | 63933/400000 [00:07<00:40, 8361.27it/s] 16%|█▌        | 64788/400000 [00:08<00:39, 8414.60it/s] 16%|█▋        | 65631/400000 [00:08<00:40, 8352.82it/s] 17%|█▋        | 66542/400000 [00:08<00:38, 8564.64it/s] 17%|█▋        | 67401/400000 [00:08<00:39, 8524.55it/s] 17%|█▋        | 68255/400000 [00:08<00:39, 8408.95it/s] 17%|█▋        | 69098/400000 [00:08<00:39, 8357.38it/s] 17%|█▋        | 69935/400000 [00:08<00:39, 8335.09it/s] 18%|█▊        | 70770/400000 [00:08<00:40, 8151.57it/s] 18%|█▊        | 71587/400000 [00:08<00:41, 7974.19it/s] 18%|█▊        | 72389/400000 [00:09<00:41, 7985.91it/s] 18%|█▊        | 73189/400000 [00:09<00:41, 7840.55it/s] 18%|█▊        | 73978/400000 [00:09<00:41, 7853.58it/s] 19%|█▊        | 74765/400000 [00:09<00:41, 7847.16it/s] 19%|█▉        | 75551/400000 [00:09<00:41, 7765.65it/s] 19%|█▉        | 76379/400000 [00:09<00:40, 7912.40it/s] 19%|█▉        | 77218/400000 [00:09<00:40, 8049.62it/s] 20%|█▉        | 78059/400000 [00:09<00:39, 8152.29it/s] 20%|█▉        | 78908/400000 [00:09<00:38, 8248.96it/s] 20%|█▉        | 79735/400000 [00:09<00:39, 8112.99it/s] 20%|██        | 80548/400000 [00:10<00:40, 7843.69it/s] 20%|██        | 81363/400000 [00:10<00:40, 7929.13it/s] 21%|██        | 82159/400000 [00:10<00:40, 7828.59it/s] 21%|██        | 82944/400000 [00:10<00:41, 7600.98it/s] 21%|██        | 83735/400000 [00:10<00:41, 7690.76it/s] 21%|██        | 84597/400000 [00:10<00:39, 7947.76it/s] 21%|██▏       | 85499/400000 [00:10<00:38, 8240.53it/s] 22%|██▏       | 86359/400000 [00:10<00:37, 8344.94it/s] 22%|██▏       | 87240/400000 [00:10<00:36, 8478.77it/s] 22%|██▏       | 88107/400000 [00:10<00:36, 8533.34it/s] 22%|██▏       | 89009/400000 [00:11<00:35, 8672.43it/s] 22%|██▏       | 89885/400000 [00:11<00:35, 8698.08it/s] 23%|██▎       | 90757/400000 [00:11<00:35, 8704.10it/s] 23%|██▎       | 91629/400000 [00:11<00:36, 8427.86it/s] 23%|██▎       | 92475/400000 [00:11<00:36, 8344.85it/s] 23%|██▎       | 93322/400000 [00:11<00:36, 8380.14it/s] 24%|██▎       | 94162/400000 [00:11<00:36, 8344.58it/s] 24%|██▍       | 95006/400000 [00:11<00:36, 8371.53it/s] 24%|██▍       | 95871/400000 [00:11<00:35, 8451.31it/s] 24%|██▍       | 96717/400000 [00:11<00:36, 8287.90it/s] 24%|██▍       | 97563/400000 [00:12<00:36, 8337.39it/s] 25%|██▍       | 98408/400000 [00:12<00:36, 8368.40it/s] 25%|██▍       | 99246/400000 [00:12<00:36, 8197.44it/s] 25%|██▌       | 100067/400000 [00:12<00:37, 8025.88it/s] 25%|██▌       | 100872/400000 [00:12<00:37, 7980.78it/s] 25%|██▌       | 101672/400000 [00:12<00:38, 7661.55it/s] 26%|██▌       | 102442/400000 [00:12<00:39, 7503.39it/s] 26%|██▌       | 103196/400000 [00:12<00:39, 7470.31it/s] 26%|██▌       | 103946/400000 [00:12<00:39, 7444.53it/s] 26%|██▌       | 104693/400000 [00:13<00:39, 7412.09it/s] 26%|██▋       | 105527/400000 [00:13<00:38, 7665.32it/s] 27%|██▋       | 106413/400000 [00:13<00:36, 7987.44it/s] 27%|██▋       | 107296/400000 [00:13<00:35, 8222.04it/s] 27%|██▋       | 108134/400000 [00:13<00:35, 8267.67it/s] 27%|██▋       | 108965/400000 [00:13<00:35, 8186.35it/s] 27%|██▋       | 109787/400000 [00:13<00:35, 8073.61it/s] 28%|██▊       | 110601/400000 [00:13<00:35, 8091.14it/s] 28%|██▊       | 111418/400000 [00:13<00:35, 8113.84it/s] 28%|██▊       | 112314/400000 [00:13<00:34, 8350.36it/s] 28%|██▊       | 113175/400000 [00:14<00:34, 8424.73it/s] 29%|██▊       | 114106/400000 [00:14<00:32, 8670.50it/s] 29%|██▉       | 115116/400000 [00:14<00:31, 9053.40it/s] 29%|██▉       | 116095/400000 [00:14<00:30, 9259.94it/s] 29%|██▉       | 117059/400000 [00:14<00:30, 9368.93it/s] 30%|██▉       | 118001/400000 [00:14<00:30, 9283.14it/s] 30%|██▉       | 118934/400000 [00:14<00:30, 9295.82it/s] 30%|██▉       | 119910/400000 [00:14<00:29, 9427.98it/s] 30%|███       | 120895/400000 [00:14<00:29, 9549.63it/s] 30%|███       | 121852/400000 [00:14<00:29, 9434.18it/s] 31%|███       | 122797/400000 [00:15<00:30, 9219.87it/s] 31%|███       | 123796/400000 [00:15<00:29, 9436.59it/s] 31%|███       | 124743/400000 [00:15<00:30, 9119.38it/s] 31%|███▏      | 125660/400000 [00:15<00:30, 8952.44it/s] 32%|███▏      | 126559/400000 [00:15<00:31, 8700.65it/s] 32%|███▏      | 127434/400000 [00:15<00:31, 8608.24it/s] 32%|███▏      | 128323/400000 [00:15<00:31, 8688.68it/s] 32%|███▏      | 129215/400000 [00:15<00:30, 8756.66it/s] 33%|███▎      | 130126/400000 [00:15<00:30, 8859.59it/s] 33%|███▎      | 131014/400000 [00:15<00:30, 8865.34it/s] 33%|███▎      | 131902/400000 [00:16<00:33, 8075.77it/s] 33%|███▎      | 132724/400000 [00:16<00:33, 7929.44it/s] 33%|███▎      | 133567/400000 [00:16<00:33, 8072.53it/s] 34%|███▎      | 134408/400000 [00:16<00:32, 8169.64it/s] 34%|███▍      | 135280/400000 [00:16<00:31, 8326.40it/s] 34%|███▍      | 136181/400000 [00:16<00:30, 8519.52it/s] 34%|███▍      | 137046/400000 [00:16<00:30, 8556.32it/s] 34%|███▍      | 137905/400000 [00:16<00:31, 8386.43it/s] 35%|███▍      | 138747/400000 [00:16<00:32, 7977.18it/s] 35%|███▍      | 139551/400000 [00:17<00:32, 7975.43it/s] 35%|███▌      | 140455/400000 [00:17<00:31, 8267.07it/s] 35%|███▌      | 141354/400000 [00:17<00:30, 8470.36it/s] 36%|███▌      | 142279/400000 [00:17<00:29, 8689.64it/s] 36%|███▌      | 143154/400000 [00:17<00:30, 8543.52it/s] 36%|███▌      | 144013/400000 [00:17<00:29, 8534.81it/s] 36%|███▌      | 144905/400000 [00:17<00:29, 8646.50it/s] 36%|███▋      | 145773/400000 [00:17<00:29, 8653.57it/s] 37%|███▋      | 146722/400000 [00:17<00:28, 8886.65it/s] 37%|███▋      | 147614/400000 [00:17<00:28, 8827.29it/s] 37%|███▋      | 148499/400000 [00:18<00:28, 8729.25it/s] 37%|███▋      | 149439/400000 [00:18<00:28, 8917.99it/s] 38%|███▊      | 150396/400000 [00:18<00:27, 9101.97it/s] 38%|███▊      | 151347/400000 [00:18<00:26, 9220.36it/s] 38%|███▊      | 152272/400000 [00:18<00:27, 9098.19it/s] 38%|███▊      | 153184/400000 [00:18<00:28, 8697.10it/s] 39%|███▊      | 154059/400000 [00:18<00:28, 8612.14it/s] 39%|███▊      | 154972/400000 [00:18<00:27, 8760.82it/s] 39%|███▉      | 155852/400000 [00:18<00:27, 8754.73it/s] 39%|███▉      | 156734/400000 [00:18<00:27, 8772.68it/s] 39%|███▉      | 157613/400000 [00:19<00:27, 8726.25it/s] 40%|███▉      | 158487/400000 [00:19<00:27, 8679.23it/s] 40%|███▉      | 159430/400000 [00:19<00:27, 8890.95it/s] 40%|████      | 160321/400000 [00:19<00:26, 8888.83it/s] 40%|████      | 161212/400000 [00:19<00:27, 8831.05it/s] 41%|████      | 162097/400000 [00:19<00:27, 8645.58it/s] 41%|████      | 163086/400000 [00:19<00:26, 8982.87it/s] 41%|████      | 164067/400000 [00:19<00:25, 9214.58it/s] 41%|████▏     | 165034/400000 [00:19<00:25, 9346.53it/s] 41%|████▏     | 165997/400000 [00:19<00:24, 9428.22it/s] 42%|████▏     | 166949/400000 [00:20<00:24, 9454.19it/s] 42%|████▏     | 167897/400000 [00:20<00:24, 9453.50it/s] 42%|████▏     | 168844/400000 [00:20<00:25, 9028.83it/s] 42%|████▏     | 169752/400000 [00:20<00:26, 8835.00it/s] 43%|████▎     | 170685/400000 [00:20<00:25, 8975.80it/s] 43%|████▎     | 171587/400000 [00:20<00:26, 8728.17it/s] 43%|████▎     | 172464/400000 [00:20<00:26, 8475.22it/s] 43%|████▎     | 173389/400000 [00:20<00:26, 8693.21it/s] 44%|████▎     | 174263/400000 [00:20<00:26, 8565.07it/s] 44%|████▍     | 175193/400000 [00:21<00:25, 8771.81it/s] 44%|████▍     | 176084/400000 [00:21<00:25, 8810.45it/s] 44%|████▍     | 177037/400000 [00:21<00:24, 9013.57it/s] 44%|████▍     | 177942/400000 [00:21<00:24, 8939.72it/s] 45%|████▍     | 178888/400000 [00:21<00:24, 9088.36it/s] 45%|████▍     | 179840/400000 [00:21<00:23, 9211.83it/s] 45%|████▌     | 180764/400000 [00:21<00:24, 9072.54it/s] 45%|████▌     | 181750/400000 [00:21<00:23, 9294.37it/s] 46%|████▌     | 182712/400000 [00:21<00:23, 9387.77it/s] 46%|████▌     | 183653/400000 [00:21<00:23, 9302.93it/s] 46%|████▌     | 184586/400000 [00:22<00:23, 9308.66it/s] 46%|████▋     | 185518/400000 [00:22<00:23, 9284.51it/s] 47%|████▋     | 186448/400000 [00:22<00:23, 9093.62it/s] 47%|████▋     | 187359/400000 [00:22<00:24, 8726.65it/s] 47%|████▋     | 188295/400000 [00:22<00:23, 8907.45it/s] 47%|████▋     | 189190/400000 [00:22<00:24, 8526.89it/s] 48%|████▊     | 190049/400000 [00:22<00:26, 7915.04it/s] 48%|████▊     | 190854/400000 [00:22<00:27, 7630.13it/s] 48%|████▊     | 191629/400000 [00:22<00:28, 7211.18it/s] 48%|████▊     | 192363/400000 [00:23<00:29, 7127.75it/s] 48%|████▊     | 193085/400000 [00:23<00:29, 7023.32it/s] 48%|████▊     | 193795/400000 [00:23<00:29, 6940.13it/s] 49%|████▊     | 194508/400000 [00:23<00:29, 6995.68it/s] 49%|████▉     | 195279/400000 [00:23<00:28, 7193.94it/s] 49%|████▉     | 196069/400000 [00:23<00:27, 7390.49it/s] 49%|████▉     | 196813/400000 [00:23<00:27, 7382.05it/s] 49%|████▉     | 197666/400000 [00:23<00:26, 7691.16it/s] 50%|████▉     | 198484/400000 [00:23<00:25, 7828.75it/s] 50%|████▉     | 199317/400000 [00:23<00:25, 7960.28it/s] 50%|█████     | 200131/400000 [00:24<00:24, 8011.45it/s] 50%|█████     | 200953/400000 [00:24<00:24, 8071.44it/s] 50%|█████     | 201835/400000 [00:24<00:23, 8280.97it/s] 51%|█████     | 202683/400000 [00:24<00:23, 8338.17it/s] 51%|█████     | 203554/400000 [00:24<00:23, 8445.84it/s] 51%|█████     | 204401/400000 [00:24<00:23, 8352.14it/s] 51%|█████▏    | 205238/400000 [00:24<00:23, 8117.41it/s] 52%|█████▏    | 206081/400000 [00:24<00:23, 8201.60it/s] 52%|█████▏    | 206904/400000 [00:24<00:24, 8033.05it/s] 52%|█████▏    | 207710/400000 [00:25<00:24, 7907.95it/s] 52%|█████▏    | 208559/400000 [00:25<00:23, 8073.19it/s] 52%|█████▏    | 209369/400000 [00:25<00:23, 8042.60it/s] 53%|█████▎    | 210175/400000 [00:25<00:23, 8022.94it/s] 53%|█████▎    | 211039/400000 [00:25<00:23, 8197.30it/s] 53%|█████▎    | 211861/400000 [00:25<00:22, 8203.44it/s] 53%|█████▎    | 212756/400000 [00:25<00:22, 8413.72it/s] 53%|█████▎    | 213645/400000 [00:25<00:21, 8550.95it/s] 54%|█████▎    | 214564/400000 [00:25<00:21, 8732.52it/s] 54%|█████▍    | 215440/400000 [00:25<00:21, 8497.16it/s] 54%|█████▍    | 216293/400000 [00:26<00:21, 8356.98it/s] 54%|█████▍    | 217182/400000 [00:26<00:21, 8508.68it/s] 55%|█████▍    | 218036/400000 [00:26<00:21, 8423.00it/s] 55%|█████▍    | 218881/400000 [00:26<00:21, 8401.02it/s] 55%|█████▍    | 219723/400000 [00:26<00:21, 8392.66it/s] 55%|█████▌    | 220645/400000 [00:26<00:20, 8624.41it/s] 55%|█████▌    | 221540/400000 [00:26<00:20, 8718.06it/s] 56%|█████▌    | 222414/400000 [00:26<00:20, 8612.52it/s] 56%|█████▌    | 223304/400000 [00:26<00:20, 8693.93it/s] 56%|█████▌    | 224175/400000 [00:26<00:20, 8451.26it/s] 56%|█████▋    | 225023/400000 [00:27<00:20, 8387.21it/s] 56%|█████▋    | 225892/400000 [00:27<00:20, 8474.31it/s] 57%|█████▋    | 226741/400000 [00:27<00:21, 8249.29it/s] 57%|█████▋    | 227588/400000 [00:27<00:20, 8313.25it/s] 57%|█████▋    | 228442/400000 [00:27<00:20, 8377.43it/s] 57%|█████▋    | 229329/400000 [00:27<00:20, 8517.47it/s] 58%|█████▊    | 230270/400000 [00:27<00:19, 8766.89it/s] 58%|█████▊    | 231150/400000 [00:27<00:19, 8608.99it/s] 58%|█████▊    | 232071/400000 [00:27<00:19, 8778.98it/s] 58%|█████▊    | 232984/400000 [00:27<00:18, 8880.23it/s] 58%|█████▊    | 233937/400000 [00:28<00:18, 9065.49it/s] 59%|█████▊    | 234909/400000 [00:28<00:17, 9250.59it/s] 59%|█████▉    | 235837/400000 [00:28<00:18, 9104.57it/s] 59%|█████▉    | 236750/400000 [00:28<00:18, 8977.19it/s] 59%|█████▉    | 237650/400000 [00:28<00:18, 8919.23it/s] 60%|█████▉    | 238573/400000 [00:28<00:17, 9008.14it/s] 60%|█████▉    | 239497/400000 [00:28<00:17, 9072.69it/s] 60%|██████    | 240414/400000 [00:28<00:17, 9099.07it/s] 60%|██████    | 241378/400000 [00:28<00:17, 9254.51it/s] 61%|██████    | 242324/400000 [00:28<00:16, 9314.88it/s] 61%|██████    | 243282/400000 [00:29<00:16, 9391.50it/s] 61%|██████    | 244232/400000 [00:29<00:16, 9421.90it/s] 61%|██████▏   | 245175/400000 [00:29<00:16, 9266.20it/s] 62%|██████▏   | 246103/400000 [00:29<00:16, 9226.84it/s] 62%|██████▏   | 247060/400000 [00:29<00:16, 9326.68it/s] 62%|██████▏   | 248037/400000 [00:29<00:16, 9454.82it/s] 62%|██████▏   | 248984/400000 [00:29<00:16, 9383.07it/s] 62%|██████▏   | 249924/400000 [00:29<00:16, 9325.16it/s] 63%|██████▎   | 250858/400000 [00:29<00:16, 9311.13it/s] 63%|██████▎   | 251828/400000 [00:29<00:15, 9422.81it/s] 63%|██████▎   | 252771/400000 [00:30<00:15, 9386.71it/s] 63%|██████▎   | 253712/400000 [00:30<00:15, 9390.95it/s] 64%|██████▎   | 254652/400000 [00:30<00:16, 9025.28it/s] 64%|██████▍   | 255558/400000 [00:30<00:16, 8930.28it/s] 64%|██████▍   | 256454/400000 [00:30<00:16, 8913.16it/s] 64%|██████▍   | 257404/400000 [00:30<00:15, 9080.16it/s] 65%|██████▍   | 258373/400000 [00:30<00:15, 9254.28it/s] 65%|██████▍   | 259301/400000 [00:30<00:15, 9196.49it/s] 65%|██████▌   | 260223/400000 [00:30<00:15, 9188.48it/s] 65%|██████▌   | 261143/400000 [00:31<00:15, 9120.56it/s] 66%|██████▌   | 262071/400000 [00:31<00:15, 9166.16it/s] 66%|██████▌   | 263039/400000 [00:31<00:14, 9312.04it/s] 66%|██████▌   | 263972/400000 [00:31<00:14, 9191.22it/s] 66%|██████▌   | 264893/400000 [00:31<00:14, 9055.66it/s] 66%|██████▋   | 265827/400000 [00:31<00:14, 9138.23it/s] 67%|██████▋   | 266800/400000 [00:31<00:14, 9306.44it/s] 67%|██████▋   | 267747/400000 [00:31<00:14, 9352.59it/s] 67%|██████▋   | 268684/400000 [00:31<00:14, 9321.15it/s] 67%|██████▋   | 269617/400000 [00:31<00:14, 9310.02it/s] 68%|██████▊   | 270549/400000 [00:32<00:14, 9231.96it/s] 68%|██████▊   | 271473/400000 [00:32<00:13, 9212.88it/s] 68%|██████▊   | 272456/400000 [00:32<00:13, 9389.32it/s] 68%|██████▊   | 273397/400000 [00:32<00:14, 8987.92it/s] 69%|██████▊   | 274301/400000 [00:32<00:14, 8902.12it/s] 69%|██████▉   | 275195/400000 [00:32<00:14, 8535.92it/s] 69%|██████▉   | 276055/400000 [00:32<00:15, 8103.58it/s] 69%|██████▉   | 276874/400000 [00:32<00:15, 7856.76it/s] 69%|██████▉   | 277668/400000 [00:32<00:15, 7675.97it/s] 70%|██████▉   | 278442/400000 [00:33<00:16, 7567.90it/s] 70%|██████▉   | 279252/400000 [00:33<00:15, 7718.29it/s] 70%|███████   | 280132/400000 [00:33<00:14, 8013.08it/s] 70%|███████   | 280939/400000 [00:33<00:14, 8006.52it/s] 70%|███████   | 281780/400000 [00:33<00:14, 8122.49it/s] 71%|███████   | 282596/400000 [00:33<00:14, 7856.13it/s] 71%|███████   | 283430/400000 [00:33<00:14, 7994.42it/s] 71%|███████   | 284370/400000 [00:33<00:13, 8369.79it/s] 71%|███████▏  | 285217/400000 [00:33<00:13, 8397.87it/s] 72%|███████▏  | 286131/400000 [00:33<00:13, 8607.39it/s] 72%|███████▏  | 286997/400000 [00:34<00:13, 8434.55it/s] 72%|███████▏  | 287845/400000 [00:34<00:13, 8353.19it/s] 72%|███████▏  | 288752/400000 [00:34<00:13, 8553.69it/s] 72%|███████▏  | 289647/400000 [00:34<00:12, 8667.64it/s] 73%|███████▎  | 290517/400000 [00:34<00:12, 8670.44it/s] 73%|███████▎  | 291396/400000 [00:34<00:12, 8704.49it/s] 73%|███████▎  | 292312/400000 [00:34<00:12, 8834.30it/s] 73%|███████▎  | 293295/400000 [00:34<00:11, 9108.05it/s] 74%|███████▎  | 294225/400000 [00:34<00:11, 9161.58it/s] 74%|███████▍  | 295144/400000 [00:34<00:11, 8967.45it/s] 74%|███████▍  | 296079/400000 [00:35<00:11, 9078.66it/s] 74%|███████▍  | 296989/400000 [00:35<00:11, 9015.17it/s] 74%|███████▍  | 297893/400000 [00:35<00:11, 8970.90it/s] 75%|███████▍  | 298792/400000 [00:35<00:11, 8835.06it/s] 75%|███████▍  | 299689/400000 [00:35<00:11, 8872.86it/s] 75%|███████▌  | 300621/400000 [00:35<00:11, 9001.99it/s] 75%|███████▌  | 301569/400000 [00:35<00:10, 9138.14it/s] 76%|███████▌  | 302536/400000 [00:35<00:10, 9289.60it/s] 76%|███████▌  | 303497/400000 [00:35<00:10, 9379.83it/s] 76%|███████▌  | 304437/400000 [00:35<00:10, 9098.95it/s] 76%|███████▋  | 305355/400000 [00:36<00:10, 9121.87it/s] 77%|███████▋  | 306303/400000 [00:36<00:10, 9224.91it/s] 77%|███████▋  | 307228/400000 [00:36<00:10, 9220.55it/s] 77%|███████▋  | 308152/400000 [00:36<00:10, 9002.68it/s] 77%|███████▋  | 309087/400000 [00:36<00:09, 9102.92it/s] 78%|███████▊  | 310036/400000 [00:36<00:09, 9213.53it/s] 78%|███████▊  | 310980/400000 [00:36<00:09, 9279.27it/s] 78%|███████▊  | 311967/400000 [00:36<00:09, 9446.40it/s] 78%|███████▊  | 312914/400000 [00:36<00:09, 9313.41it/s] 78%|███████▊  | 313847/400000 [00:36<00:09, 9067.04it/s] 79%|███████▊  | 314783/400000 [00:37<00:09, 9150.20it/s] 79%|███████▉  | 315758/400000 [00:37<00:09, 9322.00it/s] 79%|███████▉  | 316714/400000 [00:37<00:08, 9391.33it/s] 79%|███████▉  | 317655/400000 [00:37<00:08, 9362.19it/s] 80%|███████▉  | 318593/400000 [00:37<00:08, 9263.66it/s] 80%|███████▉  | 319554/400000 [00:37<00:08, 9364.42it/s] 80%|████████  | 320504/400000 [00:37<00:08, 9403.08it/s] 80%|████████  | 321446/400000 [00:37<00:08, 9380.11it/s] 81%|████████  | 322390/400000 [00:37<00:08, 9396.22it/s] 81%|████████  | 323330/400000 [00:38<00:08, 9252.44it/s] 81%|████████  | 324323/400000 [00:38<00:08, 9444.53it/s] 81%|████████▏ | 325280/400000 [00:38<00:07, 9480.33it/s] 82%|████████▏ | 326240/400000 [00:38<00:07, 9513.53it/s] 82%|████████▏ | 327193/400000 [00:38<00:07, 9111.70it/s] 82%|████████▏ | 328109/400000 [00:38<00:07, 9048.74it/s] 82%|████████▏ | 329050/400000 [00:38<00:07, 9153.91it/s] 83%|████████▎ | 330017/400000 [00:38<00:07, 9302.51it/s] 83%|████████▎ | 330953/400000 [00:38<00:07, 9319.55it/s] 83%|████████▎ | 331899/400000 [00:38<00:07, 9357.36it/s] 83%|████████▎ | 332836/400000 [00:39<00:07, 9260.72it/s] 83%|████████▎ | 333812/400000 [00:39<00:07, 9404.97it/s] 84%|████████▎ | 334774/400000 [00:39<00:06, 9466.62it/s] 84%|████████▍ | 335722/400000 [00:39<00:06, 9183.58it/s] 84%|████████▍ | 336643/400000 [00:39<00:06, 9080.15it/s] 84%|████████▍ | 337557/400000 [00:39<00:06, 9096.25it/s] 85%|████████▍ | 338508/400000 [00:39<00:06, 9215.75it/s] 85%|████████▍ | 339490/400000 [00:39<00:06, 9386.61it/s] 85%|████████▌ | 340431/400000 [00:39<00:06, 9293.30it/s] 85%|████████▌ | 341381/400000 [00:39<00:06, 9351.88it/s] 86%|████████▌ | 342318/400000 [00:40<00:06, 8992.29it/s] 86%|████████▌ | 343225/400000 [00:40<00:06, 9015.11it/s] 86%|████████▌ | 344144/400000 [00:40<00:06, 9066.36it/s] 86%|████████▋ | 345053/400000 [00:40<00:06, 8803.37it/s] 86%|████████▋ | 345937/400000 [00:40<00:06, 8682.05it/s] 87%|████████▋ | 346862/400000 [00:40<00:06, 8842.89it/s] 87%|████████▋ | 347783/400000 [00:40<00:05, 8949.11it/s] 87%|████████▋ | 348680/400000 [00:40<00:05, 8777.81it/s] 87%|████████▋ | 349592/400000 [00:40<00:05, 8876.89it/s] 88%|████████▊ | 350584/400000 [00:40<00:05, 9163.85it/s] 88%|████████▊ | 351504/400000 [00:41<00:05, 9044.10it/s] 88%|████████▊ | 352449/400000 [00:41<00:05, 9160.01it/s] 88%|████████▊ | 353385/400000 [00:41<00:05, 9213.79it/s] 89%|████████▊ | 354309/400000 [00:41<00:05, 9043.68it/s] 89%|████████▉ | 355231/400000 [00:41<00:04, 9094.76it/s] 89%|████████▉ | 356142/400000 [00:41<00:04, 8930.55it/s] 89%|████████▉ | 357037/400000 [00:41<00:04, 8918.85it/s] 89%|████████▉ | 357969/400000 [00:41<00:04, 9034.91it/s] 90%|████████▉ | 358874/400000 [00:41<00:04, 8929.46it/s] 90%|████████▉ | 359769/400000 [00:42<00:04, 8872.38it/s] 90%|█████████ | 360658/400000 [00:42<00:04, 8507.77it/s] 90%|█████████ | 361513/400000 [00:42<00:04, 7987.16it/s] 91%|█████████ | 362321/400000 [00:42<00:05, 7487.04it/s] 91%|█████████ | 363083/400000 [00:42<00:05, 7260.66it/s] 91%|█████████ | 363820/400000 [00:42<00:05, 7213.98it/s] 91%|█████████ | 364549/400000 [00:42<00:04, 7176.14it/s] 91%|█████████▏| 365276/400000 [00:42<00:04, 7202.80it/s] 92%|█████████▏| 366000/400000 [00:42<00:04, 7213.68it/s] 92%|█████████▏| 366724/400000 [00:42<00:04, 7194.97it/s] 92%|█████████▏| 367446/400000 [00:43<00:04, 7128.11it/s] 92%|█████████▏| 368161/400000 [00:43<00:04, 7058.16it/s] 92%|█████████▏| 368899/400000 [00:43<00:04, 7149.85it/s] 92%|█████████▏| 369625/400000 [00:43<00:04, 7181.02it/s] 93%|█████████▎| 370344/400000 [00:43<00:04, 7078.88it/s] 93%|█████████▎| 371053/400000 [00:43<00:04, 6869.74it/s] 93%|█████████▎| 371764/400000 [00:43<00:04, 6938.87it/s] 93%|█████████▎| 372505/400000 [00:43<00:03, 7070.97it/s] 93%|█████████▎| 373244/400000 [00:43<00:03, 7162.91it/s] 93%|█████████▎| 373986/400000 [00:44<00:03, 7236.26it/s] 94%|█████████▎| 374722/400000 [00:44<00:03, 7271.03it/s] 94%|█████████▍| 375568/400000 [00:44<00:03, 7589.41it/s] 94%|█████████▍| 376402/400000 [00:44<00:03, 7799.60it/s] 94%|█████████▍| 377280/400000 [00:44<00:02, 8069.31it/s] 95%|█████████▍| 378230/400000 [00:44<00:02, 8449.36it/s] 95%|█████████▍| 379110/400000 [00:44<00:02, 8550.76it/s] 95%|█████████▍| 379972/400000 [00:44<00:02, 8560.41it/s] 95%|█████████▌| 380834/400000 [00:44<00:02, 8576.48it/s] 95%|█████████▌| 381713/400000 [00:44<00:02, 8637.03it/s] 96%|█████████▌| 382579/400000 [00:45<00:02, 8262.57it/s] 96%|█████████▌| 383411/400000 [00:45<00:02, 8082.97it/s] 96%|█████████▌| 384224/400000 [00:45<00:02, 7752.83it/s] 96%|█████████▋| 385018/400000 [00:45<00:01, 7805.59it/s] 96%|█████████▋| 385803/400000 [00:45<00:01, 7805.21it/s] 97%|█████████▋| 386587/400000 [00:45<00:01, 7740.69it/s] 97%|█████████▋| 387364/400000 [00:45<00:01, 7526.58it/s] 97%|█████████▋| 388120/400000 [00:45<00:01, 7445.74it/s] 97%|█████████▋| 388933/400000 [00:45<00:01, 7638.37it/s] 97%|█████████▋| 389719/400000 [00:45<00:01, 7703.51it/s] 98%|█████████▊| 390533/400000 [00:46<00:01, 7827.75it/s] 98%|█████████▊| 391318/400000 [00:46<00:01, 7806.42it/s] 98%|█████████▊| 392110/400000 [00:46<00:01, 7837.92it/s] 98%|█████████▊| 392905/400000 [00:46<00:00, 7868.95it/s] 98%|█████████▊| 393695/400000 [00:46<00:00, 7875.63it/s] 99%|█████████▊| 394491/400000 [00:46<00:00, 7900.25it/s] 99%|█████████▉| 395282/400000 [00:46<00:00, 7816.86it/s] 99%|█████████▉| 396065/400000 [00:46<00:00, 7790.18it/s] 99%|█████████▉| 396845/400000 [00:46<00:00, 7599.18it/s] 99%|█████████▉| 397607/400000 [00:46<00:00, 7556.56it/s]100%|█████████▉| 398390/400000 [00:47<00:00, 7636.55it/s]100%|█████████▉| 399162/400000 [00:47<00:00, 7659.09it/s]100%|█████████▉| 399938/400000 [00:47<00:00, 7687.56it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8458.60it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7e9cfb74e0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011071920906525314 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.01122571652549565 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15671 out of table with 15659 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15671 out of table with 15659 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1ed9b0afd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 16:14:13.035681
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 16:14:13.039967
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 16:14:13.043249
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 16:14:13.046793
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1ee58d4470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354681.9688
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 281362.1875
Epoch 3/10

1/1 [==============================] - 0s 100ms/step - loss: 179096.5625
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 98741.2031
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 54053.7891
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 31177.5000
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 19106.8711
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 12496.2559
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 8785.1055
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 6527.9780

  #### Inference Need return ypred, ytrue ######################### 
[[-0.29008913 -1.2845507   1.2880906   1.0754886  -0.27636406 -0.90616465
  -0.6248224   0.41949028  0.8328401   0.22885975  1.7858906  -0.530181
  -0.76345253  0.95369387  1.2601075   1.7566582  -0.06340843 -1.1193855
   0.39381462 -0.28064567  1.8294568   0.60696816  0.2574377   0.6825949
   0.37033996 -0.50616276 -0.21915519  0.74667907  0.36961848 -0.14129254
  -0.3921122  -1.8533025   0.55089283  0.33835602  0.98346114 -1.0640656
   0.08070397  0.02903032  0.1689378  -2.1674626   0.47791708 -0.07794429
   0.39508936  2.6179702  -0.63559103 -0.5812924   0.5651349   0.79983366
   1.7238979  -1.9341445  -0.02131999  0.58003485  0.03594385 -1.3460836
  -2.6424215  -0.4594283  -1.0797799  -1.3732777   0.4663543   1.810699
   0.02329513  9.711886    8.453199    7.7310925   8.843986    8.703402
   8.4224615   7.6331115   6.4158115   8.322213    8.092962    8.446593
   8.338667    8.383111    7.6623507   8.607601    8.53569     8.875359
   7.6395087   8.678321    7.041216   10.097664    8.47894     8.309138
   8.088454    8.593317    7.604466    7.906875    7.5148163   8.940256
   8.311972    9.363911   10.050582    7.8717256   9.43638     9.159862
   6.8859158   9.132161    8.934656    7.6891985   7.9642386   8.806885
   8.9547825   7.1935196   8.1560755   6.428904    8.291024    8.028764
   6.91546     7.525817    8.168055    7.01395     7.233289    8.870511
   7.8864264   6.9678383   7.3818126   7.708914    8.92444     6.9361
   0.2802358  -0.98377     0.30830348  1.0290259   1.8737025   0.29207352
   1.5269983  -1.0458813   0.4981404   0.57603073  1.9467397  -0.69350684
   0.8820977  -1.1482923   0.06030791 -0.05383161  1.2630665  -0.7549207
   0.3109901   0.47442508 -1.6697817   0.5732217   1.7195013   1.3209877
  -0.70200634  0.8811897  -0.34452063  1.363219   -0.29776323 -0.23472068
  -1.6965702  -0.89353496 -0.23073502 -0.28195426 -2.6080952   1.2174087
   0.5322427   0.04857935  1.6303265  -0.7627625   0.18851979 -1.0653489
  -1.1631231   2.50197    -0.40347022 -0.2948885   0.94032687  0.40458506
  -0.15030101 -0.18577537  1.1401886   0.18743634 -1.3702531  -1.3764172
  -0.8985082  -0.19621927  0.81105965  0.49085957  0.8702474  -0.42528838
   0.52969694  0.64198875  1.5084846   1.1698657   0.3622862   1.5866585
   0.23526025  0.85136503  1.0705807   1.6694535   0.5548679   1.3782465
   0.9692701   1.2372031   2.5025516   1.8328416   1.181963    1.1227914
   0.7114933   0.93185794  0.9883997   1.3127806   0.5276659   0.23956716
   1.6231639   1.0051068   0.9869207   1.382706    0.63068724  0.7512624
   0.56030256  0.94239724  0.6558549   1.400814    0.6784576   0.05307835
   1.3686092   0.9837181   1.6075578   1.5202101   0.5663758   1.2216605
   1.4154894   0.84752595  1.5708297   1.6341743   2.053375    0.63150877
   0.73072827  0.41765153  0.90018296  0.6723069   1.600607    1.5360516
   3.1622958   1.1605356   0.36022455  1.5594245   2.1556945   0.55576015
   0.13155329  9.529943    9.415652    8.701548    6.7562723   9.479192
   8.595165    7.770865    7.5458      8.210371    7.2409735   8.734084
   8.50262     8.923421    8.7353525   7.5042458   9.246124    8.104679
   9.895349    8.104763    9.463658    8.172174    9.194119    8.261297
   9.692936    8.500028    7.3969927   7.6435046   8.476341    9.003729
   9.225834    9.604991   10.09515     8.61863     9.097872    7.787164
   9.149301    7.839815    7.0128675   8.529486    8.535593    8.534757
   9.401818    8.283584    9.067756    7.096691    9.530747    8.194442
   8.7134285   9.035594    9.123462    7.870884    9.750837    8.589639
   9.256661    7.922346    7.0916      8.075314    7.7678843   7.7692227
   0.49049425  0.5205913   0.35626435  1.3501414   0.33762425  0.25325358
   0.46902514  2.2350743   1.9121406   0.70108426  0.55993587  1.447802
   0.83273447  0.5188985   0.55267763  3.0414906   2.2430573   2.2110596
   3.693077    0.9530464   0.44956577  0.46309537  2.0224748   0.5306641
   0.30435526  1.5883281   1.3584545   1.3327595   0.6593931   0.6601364
   2.047886    1.498982    0.7184113   1.4754734   1.0947477   3.2050467
   0.93370354  0.25496125  0.43681693  1.7606528   0.84172446  2.2104053
   1.1068932   0.34434474  1.040355    2.710277    0.25356948  0.14850324
   1.7006179   0.6775075   0.26567155  0.53350365  0.30848914  1.8840334
   0.21965414  1.3878828   1.6267816   2.1746845   1.2185445   2.2251496
  -5.7428846   5.0289497  -7.3903203 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 16:14:22.798448
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.059
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 16:14:22.802848
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8870.93
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 16:14:22.807140
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.3539
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 16:14:22.811131
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -793.455
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139770118914176
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139769177432808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139769177433312
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769177433816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769177434320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769177434824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1ed9893cf8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.634121
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.589703
grad_step = 000002, loss = 0.547913
grad_step = 000003, loss = 0.503324
grad_step = 000004, loss = 0.456941
grad_step = 000005, loss = 0.420915
grad_step = 000006, loss = 0.400638
grad_step = 000007, loss = 0.391677
grad_step = 000008, loss = 0.369134
grad_step = 000009, loss = 0.344065
grad_step = 000010, loss = 0.327227
grad_step = 000011, loss = 0.317086
grad_step = 000012, loss = 0.308011
grad_step = 000013, loss = 0.296580
grad_step = 000014, loss = 0.282319
grad_step = 000015, loss = 0.266677
grad_step = 000016, loss = 0.252014
grad_step = 000017, loss = 0.240351
grad_step = 000018, loss = 0.230449
grad_step = 000019, loss = 0.218407
grad_step = 000020, loss = 0.204373
grad_step = 000021, loss = 0.191662
grad_step = 000022, loss = 0.181462
grad_step = 000023, loss = 0.172385
grad_step = 000024, loss = 0.162866
grad_step = 000025, loss = 0.152548
grad_step = 000026, loss = 0.142215
grad_step = 000027, loss = 0.132966
grad_step = 000028, loss = 0.124807
grad_step = 000029, loss = 0.114651
grad_step = 000030, loss = 0.103045
grad_step = 000031, loss = 0.091553
grad_step = 000032, loss = 0.081077
grad_step = 000033, loss = 0.071770
grad_step = 000034, loss = 0.063530
grad_step = 000035, loss = 0.056424
grad_step = 000036, loss = 0.050292
grad_step = 000037, loss = 0.045063
grad_step = 000038, loss = 0.040698
grad_step = 000039, loss = 0.036750
grad_step = 000040, loss = 0.032742
grad_step = 000041, loss = 0.028764
grad_step = 000042, loss = 0.025245
grad_step = 000043, loss = 0.022239
grad_step = 000044, loss = 0.019487
grad_step = 000045, loss = 0.016802
grad_step = 000046, loss = 0.014544
grad_step = 000047, loss = 0.012784
grad_step = 000048, loss = 0.011114
grad_step = 000049, loss = 0.009533
grad_step = 000050, loss = 0.008382
grad_step = 000051, loss = 0.007565
grad_step = 000052, loss = 0.006774
grad_step = 000053, loss = 0.006053
grad_step = 000054, loss = 0.005541
grad_step = 000055, loss = 0.005108
grad_step = 000056, loss = 0.004621
grad_step = 000057, loss = 0.004188
grad_step = 000058, loss = 0.003878
grad_step = 000059, loss = 0.003606
grad_step = 000060, loss = 0.003349
grad_step = 000061, loss = 0.003183
grad_step = 000062, loss = 0.003088
grad_step = 000063, loss = 0.003007
grad_step = 000064, loss = 0.002945
grad_step = 000065, loss = 0.002903
grad_step = 000066, loss = 0.002855
grad_step = 000067, loss = 0.002805
grad_step = 000068, loss = 0.002758
grad_step = 000069, loss = 0.002705
grad_step = 000070, loss = 0.002666
grad_step = 000071, loss = 0.002639
grad_step = 000072, loss = 0.002605
grad_step = 000073, loss = 0.002575
grad_step = 000074, loss = 0.002560
grad_step = 000075, loss = 0.002535
grad_step = 000076, loss = 0.002499
grad_step = 000077, loss = 0.002476
grad_step = 000078, loss = 0.002450
grad_step = 000079, loss = 0.002411
grad_step = 000080, loss = 0.002380
grad_step = 000081, loss = 0.002352
grad_step = 000082, loss = 0.002316
grad_step = 000083, loss = 0.002281
grad_step = 000084, loss = 0.002252
grad_step = 000085, loss = 0.002222
grad_step = 000086, loss = 0.002194
grad_step = 000087, loss = 0.002172
grad_step = 000088, loss = 0.002149
grad_step = 000089, loss = 0.002127
grad_step = 000090, loss = 0.002106
grad_step = 000091, loss = 0.002084
grad_step = 000092, loss = 0.002063
grad_step = 000093, loss = 0.002042
grad_step = 000094, loss = 0.002024
grad_step = 000095, loss = 0.002008
grad_step = 000096, loss = 0.001996
grad_step = 000097, loss = 0.001985
grad_step = 000098, loss = 0.001975
grad_step = 000099, loss = 0.001967
grad_step = 000100, loss = 0.001957
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001947
grad_step = 000102, loss = 0.001938
grad_step = 000103, loss = 0.001929
grad_step = 000104, loss = 0.001920
grad_step = 000105, loss = 0.001912
grad_step = 000106, loss = 0.001904
grad_step = 000107, loss = 0.001896
grad_step = 000108, loss = 0.001889
grad_step = 000109, loss = 0.001880
grad_step = 000110, loss = 0.001872
grad_step = 000111, loss = 0.001863
grad_step = 000112, loss = 0.001853
grad_step = 000113, loss = 0.001844
grad_step = 000114, loss = 0.001835
grad_step = 000115, loss = 0.001827
grad_step = 000116, loss = 0.001826
grad_step = 000117, loss = 0.001827
grad_step = 000118, loss = 0.001825
grad_step = 000119, loss = 0.001810
grad_step = 000120, loss = 0.001785
grad_step = 000121, loss = 0.001770
grad_step = 000122, loss = 0.001767
grad_step = 000123, loss = 0.001767
grad_step = 000124, loss = 0.001768
grad_step = 000125, loss = 0.001762
grad_step = 000126, loss = 0.001742
grad_step = 000127, loss = 0.001724
grad_step = 000128, loss = 0.001711
grad_step = 000129, loss = 0.001703
grad_step = 000130, loss = 0.001705
grad_step = 000131, loss = 0.001710
grad_step = 000132, loss = 0.001720
grad_step = 000133, loss = 0.001743
grad_step = 000134, loss = 0.001757
grad_step = 000135, loss = 0.001722
grad_step = 000136, loss = 0.001691
grad_step = 000137, loss = 0.001656
grad_step = 000138, loss = 0.001640
grad_step = 000139, loss = 0.001664
grad_step = 000140, loss = 0.001677
grad_step = 000141, loss = 0.001672
grad_step = 000142, loss = 0.001675
grad_step = 000143, loss = 0.001663
grad_step = 000144, loss = 0.001620
grad_step = 000145, loss = 0.001611
grad_step = 000146, loss = 0.001612
grad_step = 000147, loss = 0.001601
grad_step = 000148, loss = 0.001612
grad_step = 000149, loss = 0.001623
grad_step = 000150, loss = 0.001612
grad_step = 000151, loss = 0.001604
grad_step = 000152, loss = 0.001611
grad_step = 000153, loss = 0.001595
grad_step = 000154, loss = 0.001576
grad_step = 000155, loss = 0.001572
grad_step = 000156, loss = 0.001567
grad_step = 000157, loss = 0.001552
grad_step = 000158, loss = 0.001544
grad_step = 000159, loss = 0.001545
grad_step = 000160, loss = 0.001541
grad_step = 000161, loss = 0.001531
grad_step = 000162, loss = 0.001527
grad_step = 000163, loss = 0.001530
grad_step = 000164, loss = 0.001530
grad_step = 000165, loss = 0.001531
grad_step = 000166, loss = 0.001546
grad_step = 000167, loss = 0.001592
grad_step = 000168, loss = 0.001690
grad_step = 000169, loss = 0.001854
grad_step = 000170, loss = 0.001982
grad_step = 000171, loss = 0.001920
grad_step = 000172, loss = 0.001632
grad_step = 000173, loss = 0.001500
grad_step = 000174, loss = 0.001631
grad_step = 000175, loss = 0.001758
grad_step = 000176, loss = 0.001655
grad_step = 000177, loss = 0.001483
grad_step = 000178, loss = 0.001541
grad_step = 000179, loss = 0.001655
grad_step = 000180, loss = 0.001596
grad_step = 000181, loss = 0.001489
grad_step = 000182, loss = 0.001486
grad_step = 000183, loss = 0.001558
grad_step = 000184, loss = 0.001564
grad_step = 000185, loss = 0.001478
grad_step = 000186, loss = 0.001448
grad_step = 000187, loss = 0.001498
grad_step = 000188, loss = 0.001512
grad_step = 000189, loss = 0.001460
grad_step = 000190, loss = 0.001433
grad_step = 000191, loss = 0.001452
grad_step = 000192, loss = 0.001467
grad_step = 000193, loss = 0.001452
grad_step = 000194, loss = 0.001419
grad_step = 000195, loss = 0.001417
grad_step = 000196, loss = 0.001438
grad_step = 000197, loss = 0.001434
grad_step = 000198, loss = 0.001411
grad_step = 000199, loss = 0.001399
grad_step = 000200, loss = 0.001405
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001410
grad_step = 000202, loss = 0.001405
grad_step = 000203, loss = 0.001394
grad_step = 000204, loss = 0.001385
grad_step = 000205, loss = 0.001381
grad_step = 000206, loss = 0.001383
grad_step = 000207, loss = 0.001386
grad_step = 000208, loss = 0.001380
grad_step = 000209, loss = 0.001371
grad_step = 000210, loss = 0.001363
grad_step = 000211, loss = 0.001360
grad_step = 000212, loss = 0.001361
grad_step = 000213, loss = 0.001361
grad_step = 000214, loss = 0.001359
grad_step = 000215, loss = 0.001355
grad_step = 000216, loss = 0.001351
grad_step = 000217, loss = 0.001346
grad_step = 000218, loss = 0.001341
grad_step = 000219, loss = 0.001336
grad_step = 000220, loss = 0.001332
grad_step = 000221, loss = 0.001330
grad_step = 000222, loss = 0.001334
grad_step = 000223, loss = 0.001341
grad_step = 000224, loss = 0.001353
grad_step = 000225, loss = 0.001366
grad_step = 000226, loss = 0.001381
grad_step = 000227, loss = 0.001398
grad_step = 000228, loss = 0.001415
grad_step = 000229, loss = 0.001432
grad_step = 000230, loss = 0.001435
grad_step = 000231, loss = 0.001427
grad_step = 000232, loss = 0.001396
grad_step = 000233, loss = 0.001350
grad_step = 000234, loss = 0.001314
grad_step = 000235, loss = 0.001303
grad_step = 000236, loss = 0.001304
grad_step = 000237, loss = 0.001302
grad_step = 000238, loss = 0.001305
grad_step = 000239, loss = 0.001322
grad_step = 000240, loss = 0.001340
grad_step = 000241, loss = 0.001344
grad_step = 000242, loss = 0.001343
grad_step = 000243, loss = 0.001345
grad_step = 000244, loss = 0.001345
grad_step = 000245, loss = 0.001335
grad_step = 000246, loss = 0.001320
grad_step = 000247, loss = 0.001307
grad_step = 000248, loss = 0.001297
grad_step = 000249, loss = 0.001283
grad_step = 000250, loss = 0.001267
grad_step = 000251, loss = 0.001253
grad_step = 000252, loss = 0.001247
grad_step = 000253, loss = 0.001244
grad_step = 000254, loss = 0.001240
grad_step = 000255, loss = 0.001235
grad_step = 000256, loss = 0.001230
grad_step = 000257, loss = 0.001227
grad_step = 000258, loss = 0.001224
grad_step = 000259, loss = 0.001221
grad_step = 000260, loss = 0.001219
grad_step = 000261, loss = 0.001221
grad_step = 000262, loss = 0.001228
grad_step = 000263, loss = 0.001246
grad_step = 000264, loss = 0.001288
grad_step = 000265, loss = 0.001375
grad_step = 000266, loss = 0.001545
grad_step = 000267, loss = 0.001813
grad_step = 000268, loss = 0.002057
grad_step = 000269, loss = 0.001992
grad_step = 000270, loss = 0.001520
grad_step = 000271, loss = 0.001207
grad_step = 000272, loss = 0.001414
grad_step = 000273, loss = 0.001668
grad_step = 000274, loss = 0.001483
grad_step = 000275, loss = 0.001195
grad_step = 000276, loss = 0.001279
grad_step = 000277, loss = 0.001468
grad_step = 000278, loss = 0.001374
grad_step = 000279, loss = 0.001183
grad_step = 000280, loss = 0.001232
grad_step = 000281, loss = 0.001360
grad_step = 000282, loss = 0.001277
grad_step = 000283, loss = 0.001165
grad_step = 000284, loss = 0.001208
grad_step = 000285, loss = 0.001267
grad_step = 000286, loss = 0.001205
grad_step = 000287, loss = 0.001149
grad_step = 000288, loss = 0.001184
grad_step = 000289, loss = 0.001200
grad_step = 000290, loss = 0.001160
grad_step = 000291, loss = 0.001139
grad_step = 000292, loss = 0.001157
grad_step = 000293, loss = 0.001154
grad_step = 000294, loss = 0.001139
grad_step = 000295, loss = 0.001129
grad_step = 000296, loss = 0.001130
grad_step = 000297, loss = 0.001129
grad_step = 000298, loss = 0.001126
grad_step = 000299, loss = 0.001114
grad_step = 000300, loss = 0.001109
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001115
grad_step = 000302, loss = 0.001115
grad_step = 000303, loss = 0.001101
grad_step = 000304, loss = 0.001097
grad_step = 000305, loss = 0.001106
grad_step = 000306, loss = 0.001105
grad_step = 000307, loss = 0.001092
grad_step = 000308, loss = 0.001089
grad_step = 000309, loss = 0.001096
grad_step = 000310, loss = 0.001095
grad_step = 000311, loss = 0.001086
grad_step = 000312, loss = 0.001084
grad_step = 000313, loss = 0.001087
grad_step = 000314, loss = 0.001086
grad_step = 000315, loss = 0.001081
grad_step = 000316, loss = 0.001080
grad_step = 000317, loss = 0.001079
grad_step = 000318, loss = 0.001078
grad_step = 000319, loss = 0.001077
grad_step = 000320, loss = 0.001076
grad_step = 000321, loss = 0.001073
grad_step = 000322, loss = 0.001071
grad_step = 000323, loss = 0.001071
grad_step = 000324, loss = 0.001072
grad_step = 000325, loss = 0.001069
grad_step = 000326, loss = 0.001067
grad_step = 000327, loss = 0.001067
grad_step = 000328, loss = 0.001069
grad_step = 000329, loss = 0.001071
grad_step = 000330, loss = 0.001076
grad_step = 000331, loss = 0.001086
grad_step = 000332, loss = 0.001100
grad_step = 000333, loss = 0.001117
grad_step = 000334, loss = 0.001124
grad_step = 000335, loss = 0.001111
grad_step = 000336, loss = 0.001083
grad_step = 000337, loss = 0.001060
grad_step = 000338, loss = 0.001058
grad_step = 000339, loss = 0.001070
grad_step = 000340, loss = 0.001078
grad_step = 000341, loss = 0.001070
grad_step = 000342, loss = 0.001058
grad_step = 000343, loss = 0.001055
grad_step = 000344, loss = 0.001060
grad_step = 000345, loss = 0.001065
grad_step = 000346, loss = 0.001061
grad_step = 000347, loss = 0.001053
grad_step = 000348, loss = 0.001046
grad_step = 000349, loss = 0.001045
grad_step = 000350, loss = 0.001048
grad_step = 000351, loss = 0.001052
grad_step = 000352, loss = 0.001053
grad_step = 000353, loss = 0.001049
grad_step = 000354, loss = 0.001044
grad_step = 000355, loss = 0.001040
grad_step = 000356, loss = 0.001039
grad_step = 000357, loss = 0.001039
grad_step = 000358, loss = 0.001039
grad_step = 000359, loss = 0.001039
grad_step = 000360, loss = 0.001037
grad_step = 000361, loss = 0.001034
grad_step = 000362, loss = 0.001032
grad_step = 000363, loss = 0.001030
grad_step = 000364, loss = 0.001030
grad_step = 000365, loss = 0.001030
grad_step = 000366, loss = 0.001030
grad_step = 000367, loss = 0.001030
grad_step = 000368, loss = 0.001030
grad_step = 000369, loss = 0.001030
grad_step = 000370, loss = 0.001031
grad_step = 000371, loss = 0.001033
grad_step = 000372, loss = 0.001039
grad_step = 000373, loss = 0.001051
grad_step = 000374, loss = 0.001077
grad_step = 000375, loss = 0.001127
grad_step = 000376, loss = 0.001223
grad_step = 000377, loss = 0.001380
grad_step = 000378, loss = 0.001611
grad_step = 000379, loss = 0.001809
grad_step = 000380, loss = 0.001817
grad_step = 000381, loss = 0.001500
grad_step = 000382, loss = 0.001116
grad_step = 000383, loss = 0.001043
grad_step = 000384, loss = 0.001265
grad_step = 000385, loss = 0.001415
grad_step = 000386, loss = 0.001272
grad_step = 000387, loss = 0.001053
grad_step = 000388, loss = 0.001059
grad_step = 000389, loss = 0.001215
grad_step = 000390, loss = 0.001235
grad_step = 000391, loss = 0.001094
grad_step = 000392, loss = 0.001015
grad_step = 000393, loss = 0.001090
grad_step = 000394, loss = 0.001160
grad_step = 000395, loss = 0.001098
grad_step = 000396, loss = 0.001016
grad_step = 000397, loss = 0.001032
grad_step = 000398, loss = 0.001093
grad_step = 000399, loss = 0.001084
grad_step = 000400, loss = 0.001022
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001007
grad_step = 000402, loss = 0.001046
grad_step = 000403, loss = 0.001061
grad_step = 000404, loss = 0.001026
grad_step = 000405, loss = 0.000999
grad_step = 000406, loss = 0.001014
grad_step = 000407, loss = 0.001037
grad_step = 000408, loss = 0.001026
grad_step = 000409, loss = 0.001001
grad_step = 000410, loss = 0.000997
grad_step = 000411, loss = 0.001011
grad_step = 000412, loss = 0.001017
grad_step = 000413, loss = 0.001005
grad_step = 000414, loss = 0.000992
grad_step = 000415, loss = 0.000993
grad_step = 000416, loss = 0.001001
grad_step = 000417, loss = 0.001002
grad_step = 000418, loss = 0.000996
grad_step = 000419, loss = 0.000988
grad_step = 000420, loss = 0.000986
grad_step = 000421, loss = 0.000991
grad_step = 000422, loss = 0.000993
grad_step = 000423, loss = 0.000990
grad_step = 000424, loss = 0.000984
grad_step = 000425, loss = 0.000981
grad_step = 000426, loss = 0.000982
grad_step = 000427, loss = 0.000985
grad_step = 000428, loss = 0.000985
grad_step = 000429, loss = 0.000982
grad_step = 000430, loss = 0.000979
grad_step = 000431, loss = 0.000978
grad_step = 000432, loss = 0.000981
grad_step = 000433, loss = 0.000986
grad_step = 000434, loss = 0.000995
grad_step = 000435, loss = 0.001006
grad_step = 000436, loss = 0.001025
grad_step = 000437, loss = 0.001047
grad_step = 000438, loss = 0.001076
grad_step = 000439, loss = 0.001080
grad_step = 000440, loss = 0.001060
grad_step = 000441, loss = 0.001013
grad_step = 000442, loss = 0.000977
grad_step = 000443, loss = 0.000974
grad_step = 000444, loss = 0.000994
grad_step = 000445, loss = 0.001008
grad_step = 000446, loss = 0.000998
grad_step = 000447, loss = 0.000977
grad_step = 000448, loss = 0.000966
grad_step = 000449, loss = 0.000972
grad_step = 000450, loss = 0.000983
grad_step = 000451, loss = 0.000984
grad_step = 000452, loss = 0.000973
grad_step = 000453, loss = 0.000959
grad_step = 000454, loss = 0.000955
grad_step = 000455, loss = 0.000960
grad_step = 000456, loss = 0.000965
grad_step = 000457, loss = 0.000965
grad_step = 000458, loss = 0.000959
grad_step = 000459, loss = 0.000952
grad_step = 000460, loss = 0.000951
grad_step = 000461, loss = 0.000955
grad_step = 000462, loss = 0.000961
grad_step = 000463, loss = 0.000967
grad_step = 000464, loss = 0.000973
grad_step = 000465, loss = 0.000982
grad_step = 000466, loss = 0.000997
grad_step = 000467, loss = 0.001020
grad_step = 000468, loss = 0.001053
grad_step = 000469, loss = 0.001088
grad_step = 000470, loss = 0.001117
grad_step = 000471, loss = 0.001125
grad_step = 000472, loss = 0.001107
grad_step = 000473, loss = 0.001055
grad_step = 000474, loss = 0.000992
grad_step = 000475, loss = 0.000947
grad_step = 000476, loss = 0.000938
grad_step = 000477, loss = 0.000959
grad_step = 000478, loss = 0.000989
grad_step = 000479, loss = 0.001009
grad_step = 000480, loss = 0.001012
grad_step = 000481, loss = 0.001002
grad_step = 000482, loss = 0.000982
grad_step = 000483, loss = 0.000960
grad_step = 000484, loss = 0.000941
grad_step = 000485, loss = 0.000930
grad_step = 000486, loss = 0.000927
grad_step = 000487, loss = 0.000931
grad_step = 000488, loss = 0.000939
grad_step = 000489, loss = 0.000946
grad_step = 000490, loss = 0.000950
grad_step = 000491, loss = 0.000950
grad_step = 000492, loss = 0.000946
grad_step = 000493, loss = 0.000938
grad_step = 000494, loss = 0.000930
grad_step = 000495, loss = 0.000922
grad_step = 000496, loss = 0.000916
grad_step = 000497, loss = 0.000912
grad_step = 000498, loss = 0.000910
grad_step = 000499, loss = 0.000910
grad_step = 000500, loss = 0.000911
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000913
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

  date_run                              2020-05-12 16:14:46.486737
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.222671
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 16:14:46.492917
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.128235
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 16:14:46.499543
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.129366
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 16:14:46.505530
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.948577
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
0   2020-05-12 16:14:13.035681  ...    mean_absolute_error
1   2020-05-12 16:14:13.039967  ...     mean_squared_error
2   2020-05-12 16:14:13.043249  ...  median_absolute_error
3   2020-05-12 16:14:13.046793  ...               r2_score
4   2020-05-12 16:14:22.798448  ...    mean_absolute_error
5   2020-05-12 16:14:22.802848  ...     mean_squared_error
6   2020-05-12 16:14:22.807140  ...  median_absolute_error
7   2020-05-12 16:14:22.811131  ...               r2_score
8   2020-05-12 16:14:46.486737  ...    mean_absolute_error
9   2020-05-12 16:14:46.492917  ...     mean_squared_error
10  2020-05-12 16:14:46.499543  ...  median_absolute_error
11  2020-05-12 16:14:46.505530  ...               r2_score

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

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2924544/9912422 [00:00<00:00, 29148244.53it/s]9920512it [00:00, 34403118.68it/s]                             
0it [00:00, ?it/s]32768it [00:00, 664453.83it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1023133.60it/s]1654784it [00:00, 12624812.01it/s]                           
0it [00:00, ?it/s]8192it [00:00, 264747.60it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fafe3e07c88> <class 'mlmodels.model_tch.torchhub.Model'>
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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf967c0e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf95df00b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf967c0e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf95d48048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf93572470> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf9356cba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf967c0e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf95d05668> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7faf93572470> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fafe3dcaeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fbf99536208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=536108792a5e3cdd17bfa81d307daf684af0233e1a8a3f20766c6f6758caf279
  Stored in directory: /tmp/pip-ephem-wheel-cache-wp9d0hpw/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fbf31331710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4268032/17464789 [======>.......................] - ETA: 0s
11223040/17464789 [==================>...........] - ETA: 0s
16211968/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 16:16:14.373797: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 16:16:14.378052: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 16:16:14.378207: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557515ccdf10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 16:16:14.378223: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5491 - accuracy: 0.5077
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6785 - accuracy: 0.4992
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6222 - accuracy: 0.5029
11000/25000 [============>.................] - ETA: 4s - loss: 7.6471 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6392 - accuracy: 0.5018
15000/25000 [=================>............] - ETA: 3s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6648 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 10s 392us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 16:16:31.805506
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 16:16:31.805506  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}] 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<27:55:53, 8.57kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<19:47:22, 12.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<13:54:22, 17.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<9:44:26, 24.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:48:01, 35.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.60M/862M [00:01<4:43:44, 50.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:01<3:17:22, 71.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<2:17:19, 102kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:02<1:35:35, 146kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:02<1:06:33, 208kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<46:21, 296kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<32:19, 422kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<22:40, 599kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<15:51, 852kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<14:06, 957kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<11:44, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<10:06, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<07:28, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<07:46, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<07:00, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<05:13, 2.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:28, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<04:28, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<06:13, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<07:03, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<05:37, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:11<04:05, 3.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<1:50:01, 120kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<1:18:20, 168kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<55:01, 238kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<41:28, 315kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<30:20, 431kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<21:28, 608kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<18:05, 719kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<15:19, 849kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<11:17, 1.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:17<08:01, 1.61MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<18:33, 697kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<14:18, 904kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<10:17, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<10:13, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<09:47, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<07:25, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<05:19, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<17:28, 733kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<13:32, 946kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<09:44, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<09:47, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<08:09, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<06:01, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:12, 1.76MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:40, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:01, 2.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:14, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:40, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:15, 2.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:54, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:39, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:15, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<03:47, 3.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<20:55, 596kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<15:57, 781kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<11:27, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:54, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:55, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:32, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:29, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:30, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:51, 2.53MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:17, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:29, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:09, 2.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:46, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:32, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:07, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<03:45, 3.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:41, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:37, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:56, 2.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:16, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:36, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:13, 2.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:46, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:31, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:06, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:44, 3.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:10, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:56, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:06, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:21, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:43, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:16, 2.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:37, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:27, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:03, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:40, 3.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:37, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:59, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:53, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:44, 1.73MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:57, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:25, 2.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:41, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:27, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:03, 2.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:39, 3.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<14:39, 784kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:29, 1.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:17, 1.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<05:56, 1.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<1:06:44, 171kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<49:08, 233kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<34:53, 327kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<24:34, 464kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<20:42, 549kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<15:42, 723kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<11:13, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<10:23, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<09:41, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:22, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:17, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:34, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:36, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:15, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:53, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:12, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:38, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:02, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<18:35, 596kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<14:13, 779kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<10:10, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<09:35, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<09:03, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:52, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:55, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<13:29, 811kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<10:25, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:31, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<18:01, 604kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<14:56, 728kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<10:56, 993kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:47, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:33, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:32, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:14, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:47, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<07:03, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:30, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:58, 2.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<27:21, 390kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<20:17, 526kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<14:24, 739kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<12:26, 852kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<10:59, 965kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:09, 1.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<05:50, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:59, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:25, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:27, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:09, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:26, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:04, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:11, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:51, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:21, 3.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:12, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:16, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:53, 2.64MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<02:52, 3.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<15:23, 666kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<12:59, 790kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<09:37, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<06:50, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<35:22, 288kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<25:51, 394kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<18:18, 555kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<15:00, 674kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<12:40, 799kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<09:19, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:39, 1.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:48, 1.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:13, 1.39MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:19, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:57, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:18, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:51, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:31, 2.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:54, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:52, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:21, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:18, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<05:51, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:32, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:17, 2.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:00, 1.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:38, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:54, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:36, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:02, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:39, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:22, 2.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:12, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:45, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:58, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:40, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:01, 1.59MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:43, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:24, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<30:53, 308kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<22:38, 419kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<16:01, 591kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<13:17, 709kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<11:20, 831kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:25, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<05:58, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<56:47, 165kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<41:46, 224kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<29:37, 316kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<20:52, 447kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<17:08, 543kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<12:58, 716kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<09:15, 1.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:34, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:59, 1.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:07, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:37, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:55, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:20, 2.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<29:33, 307kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<21:39, 419kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<15:19, 591kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<12:44, 708kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:53, 912kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<07:08, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:58, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:45, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:11, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:42, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<49:04, 181kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<35:15, 252kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<24:50, 356kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<19:20, 456kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<14:27, 609kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<10:18, 852kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<09:13, 949kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:20, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:13, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:26, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:30, 1.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:52, 1.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:01, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:26, 1.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:43, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:29, 2.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:33, 3.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<33:56, 252kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<25:34, 334kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<18:16, 466kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<12:53, 659kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<12:00, 705kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<09:18, 909kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:41, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<06:33, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<06:18, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:48, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:29, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<05:27, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:41, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:29, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:20, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:53, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<02:55, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:57, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:38, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:43, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<03:44, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<04:21, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:24, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:30, 3.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:58, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:20, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:13, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<04:02, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:38, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<02:43, 2.92MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<03:45, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<03:29, 2.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<02:36, 3.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<03:35, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:12, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:16, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:25, 3.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:23, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:54, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:54, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:48, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:29, 2.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<02:38, 2.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:33, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<03:18, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:28, 3.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:26, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:01, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:12, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:20, 3.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<24:09, 311kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<17:41, 424kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<12:31, 597kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<10:24, 715kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<08:48, 843kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<06:32, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:38, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<55:48, 132kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<39:41, 185kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<27:53, 263kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<21:03, 346kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<15:23, 474kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<10:55, 665kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<09:14, 782kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:00, 902kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:59, 1.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<20:04, 357kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:47, 484kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<10:30, 678kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:56, 792kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:42, 919kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:42, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:05, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:29, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:35, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:13, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:16, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:22, 2.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:29, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:45, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:46, 2.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:01, 3.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<05:01, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<06:32, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:18, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:51, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:09, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:37, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:42, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:19, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:08, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<02:54, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:10, 3.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:59, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:29, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:43, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:00, 3.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:57, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:27, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:33, 2.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:13, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:37, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:48, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:05, 3.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:23, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:03, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:16, 2.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:00, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<01:56, 3.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:39, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:06, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:19, 2.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:01, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:45, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:04, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:50, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:17, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:36, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<01:53, 3.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<15:30, 385kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<11:29, 519kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<08:08, 730kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<07:00, 842kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:09, 957kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:37, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:16, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<19:05, 306kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<13:58, 417kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<09:52, 588kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<08:09, 706kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:56, 830kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:09, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<03:38, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<16:13, 351kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<11:57, 476kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<08:28, 669kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<07:09, 786kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:12, 906kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:37, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<03:16, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<11:59, 464kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<08:57, 620kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<06:23, 865kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:42, 962kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:09, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:54, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:46, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<07:52, 688kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<06:05, 889kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<04:21, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:14, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:04, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:07, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:14, 2.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<39:47, 133kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<28:22, 186kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<19:54, 264kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<15:02, 347kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<11:38, 448kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:24, 619kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<05:53, 873kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<16:20, 315kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<11:58, 429kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<08:28, 605kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<05:57, 853kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<11:50, 429kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<09:22, 541kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:48, 743kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<04:47, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<39:02, 128kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<27:50, 180kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<19:31, 255kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<14:42, 336kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<11:20, 436kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<08:10, 602kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<05:44, 850kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<19:24, 251kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<14:07, 345kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<09:57, 486kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<08:01, 599kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<06:01, 797kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:19, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:03, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<11:23, 416kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<08:22, 565kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:57, 790kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<05:13, 894kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<04:38, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:29, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<05:07, 897kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:59, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:55, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:05, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<06:31, 694kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:32, 818kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:04, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:52, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<08:15, 540kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<06:14, 713kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:27, 995kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:05, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<03:48, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:51, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:02, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:48, 1.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<03:07, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:32, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:41, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:06, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:30, 2.78MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<10:42, 391kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<07:56, 527kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:37, 739kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:49, 853kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:16, 963kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<03:12, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:16, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<11:11, 362kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:11, 494kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<05:47, 694kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:04, 977kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<07:22, 539kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<06:00, 661kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<04:22, 907kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:05, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:42, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:59, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:10, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:23, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:29, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:56, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:23, 2.71MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<12:18, 306kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<09:00, 418kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<06:21, 589kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<05:13, 708kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:03, 910kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:54, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:50, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:45, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:07, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:30, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<09:31, 374kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<07:02, 505kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:58, 710kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<04:15, 822kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:20, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:24, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:27, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:26, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:50, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:19, 2.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:30, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:06, 1.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:49, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:37, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:12, 2.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:46, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:23, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:00, 3.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:14, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:54, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:39, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:49, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:24, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:00, 2.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:11, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:51, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:21, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:36, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:45, 1.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:21, 2.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:59, 2.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:38, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:27, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:04, 2.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:22, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:15, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:56, 2.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:17, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:29, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:10, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:49, 3.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<04:56, 540kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:44, 711kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:39, 992kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:14, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:40, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:11, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:55, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:34, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:09, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:20, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:28, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:48, 2.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:53, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:34, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:08, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:19, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:10, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:51, 2.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:06, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:15, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:58, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:42, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:10, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:03, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:46, 2.77MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:01, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:09, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:54, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:38, 3.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<03:31, 580kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:38, 774kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:52, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:19, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<04:36, 430kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:23, 582kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:23, 813kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:04, 920kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:39, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:11, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:13, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:58, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:40, 2.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<04:34, 388kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<03:22, 522kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:22, 734kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<02:00, 850kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:45, 964kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:18, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:54, 1.81MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:42, 959kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:21, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:58, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:01, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:03, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:47, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:34, 2.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:59, 1.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:50, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:36, 2.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:49, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:38, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:27, 2.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<03:41, 368kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:42, 498kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:53, 700kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:34, 815kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:22, 930kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:00, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:42, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:02, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:51, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:40, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:26, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:36, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:28, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:20, 3.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<03:16, 310kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:23, 422kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:39, 595kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:19, 714kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:00, 924kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:42, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:41, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:34, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:24, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:22, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:50, 882kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:39, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:27, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:27, 1.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:27, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:20, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:14, 2.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:27, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:22, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:19, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:29, 954kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:22, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:15, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:19, 378kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:13, 511kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:07, 717kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:03, 832kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 948kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.27MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 788/400000 [00:00<00:50, 7870.59it/s]  0%|          | 1545/400000 [00:00<00:51, 7777.36it/s]  1%|          | 2297/400000 [00:00<00:51, 7697.02it/s]  1%|          | 3090/400000 [00:00<00:51, 7764.02it/s]  1%|          | 3861/400000 [00:00<00:51, 7747.21it/s]  1%|          | 4656/400000 [00:00<00:50, 7805.09it/s]  1%|▏         | 5428/400000 [00:00<00:50, 7779.05it/s]  2%|▏         | 6200/400000 [00:00<00:50, 7759.87it/s]  2%|▏         | 6996/400000 [00:00<00:50, 7817.41it/s]  2%|▏         | 7802/400000 [00:01<00:49, 7886.66it/s]  2%|▏         | 8595/400000 [00:01<00:49, 7897.01it/s]  2%|▏         | 9379/400000 [00:01<00:49, 7875.28it/s]  3%|▎         | 10156/400000 [00:01<00:49, 7834.17it/s]  3%|▎         | 10940/400000 [00:01<00:49, 7833.36it/s]  3%|▎         | 11719/400000 [00:01<00:50, 7682.98it/s]  3%|▎         | 12485/400000 [00:01<00:50, 7668.31it/s]  3%|▎         | 13250/400000 [00:01<00:50, 7651.29it/s]  4%|▎         | 14014/400000 [00:01<00:50, 7634.51it/s]  4%|▎         | 14777/400000 [00:01<00:50, 7582.12it/s]  4%|▍         | 15535/400000 [00:02<00:51, 7433.86it/s]  4%|▍         | 16279/400000 [00:02<00:54, 7026.66it/s]  4%|▍         | 17012/400000 [00:02<00:53, 7113.12it/s]  4%|▍         | 17733/400000 [00:02<00:53, 7139.99it/s]  5%|▍         | 18514/400000 [00:02<00:52, 7328.29it/s]  5%|▍         | 19300/400000 [00:02<00:50, 7479.61it/s]  5%|▌         | 20070/400000 [00:02<00:50, 7543.39it/s]  5%|▌         | 20827/400000 [00:02<00:50, 7493.82it/s]  5%|▌         | 21578/400000 [00:02<00:50, 7479.80it/s]  6%|▌         | 22332/400000 [00:02<00:50, 7496.78it/s]  6%|▌         | 23122/400000 [00:03<00:49, 7611.45it/s]  6%|▌         | 23899/400000 [00:03<00:49, 7658.21it/s]  6%|▌         | 24666/400000 [00:03<00:48, 7660.51it/s]  6%|▋         | 25447/400000 [00:03<00:48, 7703.12it/s]  7%|▋         | 26218/400000 [00:03<00:49, 7613.94it/s]  7%|▋         | 26985/400000 [00:03<00:48, 7628.73it/s]  7%|▋         | 27749/400000 [00:03<00:48, 7629.39it/s]  7%|▋         | 28513/400000 [00:03<00:48, 7593.64it/s]  7%|▋         | 29289/400000 [00:03<00:48, 7641.92it/s]  8%|▊         | 30057/400000 [00:03<00:48, 7652.44it/s]  8%|▊         | 30849/400000 [00:04<00:47, 7728.11it/s]  8%|▊         | 31637/400000 [00:04<00:47, 7772.40it/s]  8%|▊         | 32415/400000 [00:04<00:48, 7523.10it/s]  8%|▊         | 33170/400000 [00:04<00:48, 7527.42it/s]  8%|▊         | 33925/400000 [00:04<00:48, 7475.13it/s]  9%|▊         | 34696/400000 [00:04<00:48, 7543.84it/s]  9%|▉         | 35473/400000 [00:04<00:47, 7607.80it/s]  9%|▉         | 36235/400000 [00:04<00:48, 7564.48it/s]  9%|▉         | 36993/400000 [00:04<00:49, 7357.71it/s]  9%|▉         | 37761/400000 [00:04<00:48, 7450.06it/s] 10%|▉         | 38508/400000 [00:05<00:48, 7386.03it/s] 10%|▉         | 39249/400000 [00:05<00:48, 7391.22it/s] 10%|▉         | 39989/400000 [00:05<00:48, 7373.34it/s] 10%|█         | 40736/400000 [00:05<00:48, 7400.46it/s] 10%|█         | 41484/400000 [00:05<00:48, 7421.41it/s] 11%|█         | 42227/400000 [00:05<00:48, 7419.98it/s] 11%|█         | 43017/400000 [00:05<00:47, 7554.73it/s] 11%|█         | 43793/400000 [00:05<00:46, 7613.78it/s] 11%|█         | 44563/400000 [00:05<00:46, 7638.10it/s] 11%|█▏        | 45328/400000 [00:05<00:46, 7600.00it/s] 12%|█▏        | 46130/400000 [00:06<00:45, 7719.97it/s] 12%|█▏        | 46922/400000 [00:06<00:45, 7778.54it/s] 12%|█▏        | 47701/400000 [00:06<00:46, 7653.36it/s] 12%|█▏        | 48477/400000 [00:06<00:45, 7683.92it/s] 12%|█▏        | 49262/400000 [00:06<00:45, 7731.33it/s] 13%|█▎        | 50036/400000 [00:06<00:45, 7699.66it/s] 13%|█▎        | 50807/400000 [00:06<00:45, 7680.82it/s] 13%|█▎        | 51576/400000 [00:06<00:45, 7667.49it/s] 13%|█▎        | 52348/400000 [00:06<00:45, 7680.56it/s] 13%|█▎        | 53134/400000 [00:06<00:44, 7732.06it/s] 13%|█▎        | 53923/400000 [00:07<00:44, 7776.53it/s] 14%|█▎        | 54712/400000 [00:07<00:44, 7809.11it/s] 14%|█▍        | 55494/400000 [00:07<00:45, 7603.94it/s] 14%|█▍        | 56256/400000 [00:07<00:45, 7556.77it/s] 14%|█▍        | 57033/400000 [00:07<00:45, 7617.20it/s] 14%|█▍        | 57819/400000 [00:07<00:44, 7688.09it/s] 15%|█▍        | 58605/400000 [00:07<00:44, 7738.41it/s] 15%|█▍        | 59400/400000 [00:07<00:43, 7800.35it/s] 15%|█▌        | 60196/400000 [00:07<00:43, 7845.98it/s] 15%|█▌        | 60988/400000 [00:08<00:43, 7867.64it/s] 15%|█▌        | 61785/400000 [00:08<00:42, 7897.09it/s] 16%|█▌        | 62575/400000 [00:08<00:43, 7737.21it/s] 16%|█▌        | 63360/400000 [00:08<00:43, 7769.98it/s] 16%|█▌        | 64175/400000 [00:08<00:42, 7878.55it/s] 16%|█▌        | 64964/400000 [00:08<00:43, 7765.55it/s] 16%|█▋        | 65775/400000 [00:08<00:42, 7863.82it/s] 17%|█▋        | 66563/400000 [00:08<00:42, 7855.59it/s] 17%|█▋        | 67365/400000 [00:08<00:42, 7902.87it/s] 17%|█▋        | 68157/400000 [00:08<00:41, 7907.97it/s] 17%|█▋        | 68949/400000 [00:09<00:41, 7903.37it/s] 17%|█▋        | 69740/400000 [00:09<00:41, 7904.18it/s] 18%|█▊        | 70531/400000 [00:09<00:41, 7886.70it/s] 18%|█▊        | 71320/400000 [00:09<00:41, 7826.39it/s] 18%|█▊        | 72124/400000 [00:09<00:41, 7888.39it/s] 18%|█▊        | 72939/400000 [00:09<00:41, 7961.48it/s] 18%|█▊        | 73753/400000 [00:09<00:40, 8011.61it/s] 19%|█▊        | 74578/400000 [00:09<00:40, 8079.51it/s] 19%|█▉        | 75387/400000 [00:09<00:41, 7841.30it/s] 19%|█▉        | 76194/400000 [00:09<00:40, 7905.49it/s] 19%|█▉        | 76986/400000 [00:10<00:40, 7886.88it/s] 19%|█▉        | 77776/400000 [00:10<00:40, 7888.71it/s] 20%|█▉        | 78566/400000 [00:10<00:40, 7846.90it/s] 20%|█▉        | 79352/400000 [00:10<00:41, 7811.94it/s] 20%|██        | 80150/400000 [00:10<00:40, 7861.41it/s] 20%|██        | 80937/400000 [00:10<00:40, 7805.21it/s] 20%|██        | 81718/400000 [00:10<00:41, 7725.70it/s] 21%|██        | 82502/400000 [00:10<00:40, 7759.54it/s] 21%|██        | 83279/400000 [00:10<00:41, 7648.59it/s] 21%|██        | 84045/400000 [00:10<00:42, 7509.79it/s] 21%|██        | 84801/400000 [00:11<00:41, 7522.03it/s] 21%|██▏       | 85578/400000 [00:11<00:41, 7594.47it/s] 22%|██▏       | 86361/400000 [00:11<00:40, 7661.63it/s] 22%|██▏       | 87128/400000 [00:11<00:41, 7555.28it/s] 22%|██▏       | 87885/400000 [00:11<00:42, 7408.65it/s] 22%|██▏       | 88631/400000 [00:11<00:41, 7421.58it/s] 22%|██▏       | 89404/400000 [00:11<00:41, 7509.52it/s] 23%|██▎       | 90193/400000 [00:11<00:40, 7618.31it/s] 23%|██▎       | 90956/400000 [00:11<00:40, 7612.10it/s] 23%|██▎       | 91737/400000 [00:11<00:40, 7667.29it/s] 23%|██▎       | 92507/400000 [00:12<00:40, 7676.75it/s] 23%|██▎       | 93276/400000 [00:12<00:40, 7666.51it/s] 24%|██▎       | 94060/400000 [00:12<00:39, 7716.68it/s] 24%|██▎       | 94857/400000 [00:12<00:39, 7789.83it/s] 24%|██▍       | 95637/400000 [00:12<00:39, 7679.07it/s] 24%|██▍       | 96406/400000 [00:12<00:39, 7671.91it/s] 24%|██▍       | 97181/400000 [00:12<00:39, 7691.82it/s] 24%|██▍       | 97951/400000 [00:12<00:39, 7659.94it/s] 25%|██▍       | 98718/400000 [00:12<00:39, 7615.22it/s] 25%|██▍       | 99480/400000 [00:12<00:39, 7594.14it/s] 25%|██▌       | 100240/400000 [00:13<00:39, 7508.30it/s] 25%|██▌       | 101016/400000 [00:13<00:39, 7580.11it/s] 25%|██▌       | 101804/400000 [00:13<00:38, 7665.64it/s] 26%|██▌       | 102610/400000 [00:13<00:38, 7778.63it/s] 26%|██▌       | 103389/400000 [00:13<00:38, 7770.14it/s] 26%|██▌       | 104167/400000 [00:13<00:38, 7735.93it/s] 26%|██▌       | 104959/400000 [00:13<00:37, 7789.75it/s] 26%|██▋       | 105741/400000 [00:13<00:37, 7798.43it/s] 27%|██▋       | 106522/400000 [00:13<00:37, 7743.14it/s] 27%|██▋       | 107297/400000 [00:13<00:38, 7684.93it/s] 27%|██▋       | 108068/400000 [00:14<00:37, 7690.48it/s] 27%|██▋       | 108838/400000 [00:14<00:38, 7651.48it/s] 27%|██▋       | 109604/400000 [00:14<00:38, 7567.87it/s] 28%|██▊       | 110362/400000 [00:14<00:38, 7512.57it/s] 28%|██▊       | 111137/400000 [00:14<00:38, 7579.95it/s] 28%|██▊       | 111896/400000 [00:14<00:38, 7534.45it/s] 28%|██▊       | 112650/400000 [00:14<00:38, 7462.86it/s] 28%|██▊       | 113451/400000 [00:14<00:37, 7617.77it/s] 29%|██▊       | 114240/400000 [00:14<00:37, 7695.94it/s] 29%|██▉       | 115022/400000 [00:14<00:36, 7732.39it/s] 29%|██▉       | 115817/400000 [00:15<00:36, 7795.60it/s] 29%|██▉       | 116598/400000 [00:15<00:36, 7744.91it/s] 29%|██▉       | 117380/400000 [00:15<00:36, 7766.50it/s] 30%|██▉       | 118167/400000 [00:15<00:36, 7795.06it/s] 30%|██▉       | 118947/400000 [00:15<00:36, 7708.54it/s] 30%|██▉       | 119730/400000 [00:15<00:36, 7744.24it/s] 30%|███       | 120506/400000 [00:15<00:36, 7747.53it/s] 30%|███       | 121281/400000 [00:15<00:36, 7628.31it/s] 31%|███       | 122045/400000 [00:15<00:36, 7566.45it/s] 31%|███       | 122803/400000 [00:16<00:36, 7504.23it/s] 31%|███       | 123556/400000 [00:16<00:36, 7511.56it/s] 31%|███       | 124308/400000 [00:16<00:37, 7306.51it/s] 31%|███▏      | 125085/400000 [00:16<00:36, 7439.49it/s] 31%|███▏      | 125868/400000 [00:16<00:36, 7550.64it/s] 32%|███▏      | 126625/400000 [00:16<00:36, 7468.37it/s] 32%|███▏      | 127374/400000 [00:16<00:36, 7388.83it/s] 32%|███▏      | 128134/400000 [00:16<00:36, 7448.62it/s] 32%|███▏      | 128893/400000 [00:16<00:36, 7490.17it/s] 32%|███▏      | 129647/400000 [00:16<00:36, 7504.90it/s] 33%|███▎      | 130398/400000 [00:17<00:36, 7405.77it/s] 33%|███▎      | 131174/400000 [00:17<00:35, 7505.38it/s] 33%|███▎      | 131937/400000 [00:17<00:35, 7539.73it/s] 33%|███▎      | 132711/400000 [00:17<00:35, 7596.69it/s] 33%|███▎      | 133483/400000 [00:17<00:34, 7631.33it/s] 34%|███▎      | 134247/400000 [00:17<00:35, 7556.81it/s] 34%|███▍      | 135004/400000 [00:17<00:35, 7552.86it/s] 34%|███▍      | 135765/400000 [00:17<00:34, 7566.72it/s] 34%|███▍      | 136522/400000 [00:17<00:36, 7303.44it/s] 34%|███▍      | 137282/400000 [00:17<00:35, 7386.90it/s] 35%|███▍      | 138033/400000 [00:18<00:35, 7420.23it/s] 35%|███▍      | 138803/400000 [00:18<00:34, 7499.30it/s] 35%|███▍      | 139560/400000 [00:18<00:34, 7519.26it/s] 35%|███▌      | 140313/400000 [00:18<00:34, 7504.74it/s] 35%|███▌      | 141075/400000 [00:18<00:34, 7537.64it/s] 35%|███▌      | 141830/400000 [00:18<00:34, 7451.47it/s] 36%|███▌      | 142605/400000 [00:18<00:34, 7536.49it/s] 36%|███▌      | 143379/400000 [00:18<00:33, 7594.21it/s] 36%|███▌      | 144139/400000 [00:18<00:33, 7552.33it/s] 36%|███▌      | 144895/400000 [00:18<00:33, 7543.34it/s] 36%|███▋      | 145650/400000 [00:19<00:33, 7499.22it/s] 37%|███▋      | 146401/400000 [00:19<00:33, 7485.37it/s] 37%|███▋      | 147159/400000 [00:19<00:33, 7512.63it/s] 37%|███▋      | 147911/400000 [00:19<00:33, 7504.68it/s] 37%|███▋      | 148672/400000 [00:19<00:33, 7535.79it/s] 37%|███▋      | 149434/400000 [00:19<00:33, 7559.33it/s] 38%|███▊      | 150209/400000 [00:19<00:32, 7613.74it/s] 38%|███▊      | 150971/400000 [00:19<00:34, 7314.09it/s] 38%|███▊      | 151708/400000 [00:19<00:33, 7328.09it/s] 38%|███▊      | 152469/400000 [00:19<00:33, 7407.98it/s] 38%|███▊      | 153219/400000 [00:20<00:33, 7434.89it/s] 38%|███▊      | 153990/400000 [00:20<00:32, 7511.68it/s] 39%|███▊      | 154770/400000 [00:20<00:32, 7594.32it/s] 39%|███▉      | 155546/400000 [00:20<00:31, 7642.93it/s] 39%|███▉      | 156332/400000 [00:20<00:31, 7702.82it/s] 39%|███▉      | 157103/400000 [00:20<00:31, 7695.40it/s] 39%|███▉      | 157901/400000 [00:20<00:31, 7776.95it/s] 40%|███▉      | 158691/400000 [00:20<00:30, 7813.26it/s] 40%|███▉      | 159473/400000 [00:20<00:31, 7637.78it/s] 40%|████      | 160238/400000 [00:20<00:31, 7509.54it/s] 40%|████      | 160991/400000 [00:21<00:31, 7479.80it/s] 40%|████      | 161765/400000 [00:21<00:31, 7552.45it/s] 41%|████      | 162535/400000 [00:21<00:31, 7594.06it/s] 41%|████      | 163306/400000 [00:21<00:31, 7627.72it/s] 41%|████      | 164074/400000 [00:21<00:30, 7642.98it/s] 41%|████      | 164839/400000 [00:21<00:31, 7582.43it/s] 41%|████▏     | 165609/400000 [00:21<00:30, 7616.31it/s] 42%|████▏     | 166381/400000 [00:21<00:30, 7645.94it/s] 42%|████▏     | 167152/400000 [00:21<00:30, 7664.98it/s] 42%|████▏     | 167919/400000 [00:22<00:30, 7655.23it/s] 42%|████▏     | 168685/400000 [00:22<00:30, 7576.59it/s] 42%|████▏     | 169448/400000 [00:22<00:30, 7590.22it/s] 43%|████▎     | 170216/400000 [00:22<00:30, 7613.63it/s] 43%|████▎     | 170978/400000 [00:22<00:30, 7591.48it/s] 43%|████▎     | 171738/400000 [00:22<00:30, 7535.20it/s] 43%|████▎     | 172492/400000 [00:22<00:30, 7479.35it/s] 43%|████▎     | 173264/400000 [00:22<00:30, 7549.78it/s] 44%|████▎     | 174023/400000 [00:22<00:29, 7560.03it/s] 44%|████▎     | 174780/400000 [00:22<00:30, 7490.67it/s] 44%|████▍     | 175530/400000 [00:23<00:31, 7113.34it/s] 44%|████▍     | 176246/400000 [00:23<00:31, 7034.92it/s] 44%|████▍     | 177007/400000 [00:23<00:30, 7198.02it/s] 44%|████▍     | 177777/400000 [00:23<00:30, 7339.56it/s] 45%|████▍     | 178538/400000 [00:23<00:29, 7417.30it/s] 45%|████▍     | 179317/400000 [00:23<00:29, 7523.96it/s] 45%|████▌     | 180076/400000 [00:23<00:29, 7541.66it/s] 45%|████▌     | 180843/400000 [00:23<00:28, 7578.09it/s] 45%|████▌     | 181602/400000 [00:23<00:29, 7492.33it/s] 46%|████▌     | 182353/400000 [00:23<00:29, 7476.54it/s] 46%|████▌     | 183102/400000 [00:24<00:29, 7454.94it/s] 46%|████▌     | 183848/400000 [00:24<00:29, 7396.19it/s] 46%|████▌     | 184589/400000 [00:24<00:29, 7307.18it/s] 46%|████▋     | 185349/400000 [00:24<00:29, 7390.75it/s] 47%|████▋     | 186130/400000 [00:24<00:28, 7509.86it/s] 47%|████▋     | 186918/400000 [00:24<00:27, 7615.67it/s] 47%|████▋     | 187681/400000 [00:24<00:27, 7604.20it/s] 47%|████▋     | 188472/400000 [00:24<00:27, 7692.96it/s] 47%|████▋     | 189263/400000 [00:24<00:27, 7755.23it/s] 48%|████▊     | 190040/400000 [00:24<00:27, 7741.75it/s] 48%|████▊     | 190826/400000 [00:25<00:26, 7775.85it/s] 48%|████▊     | 191604/400000 [00:25<00:26, 7758.84it/s] 48%|████▊     | 192383/400000 [00:25<00:26, 7766.22it/s] 48%|████▊     | 193168/400000 [00:25<00:26, 7788.30it/s] 48%|████▊     | 193947/400000 [00:25<00:26, 7779.40it/s] 49%|████▊     | 194726/400000 [00:25<00:26, 7771.45it/s] 49%|████▉     | 195504/400000 [00:25<00:26, 7665.65it/s] 49%|████▉     | 196271/400000 [00:25<00:26, 7600.23it/s] 49%|████▉     | 197032/400000 [00:25<00:26, 7520.80it/s] 49%|████▉     | 197789/400000 [00:25<00:26, 7535.11it/s] 50%|████▉     | 198556/400000 [00:26<00:26, 7574.48it/s] 50%|████▉     | 199314/400000 [00:26<00:26, 7558.12it/s] 50%|█████     | 200071/400000 [00:26<00:26, 7528.88it/s] 50%|█████     | 200825/400000 [00:26<00:26, 7518.58it/s] 50%|█████     | 201577/400000 [00:26<00:26, 7499.21it/s] 51%|█████     | 202355/400000 [00:26<00:26, 7579.59it/s] 51%|█████     | 203130/400000 [00:26<00:25, 7629.21it/s] 51%|█████     | 203931/400000 [00:26<00:25, 7737.82it/s] 51%|█████     | 204708/400000 [00:26<00:25, 7745.73it/s] 51%|█████▏    | 205502/400000 [00:26<00:24, 7800.53it/s] 52%|█████▏    | 206283/400000 [00:27<00:25, 7723.95it/s] 52%|█████▏    | 207056/400000 [00:27<00:25, 7644.54it/s] 52%|█████▏    | 207821/400000 [00:27<00:25, 7565.49it/s] 52%|█████▏    | 208579/400000 [00:27<00:25, 7546.74it/s] 52%|█████▏    | 209336/400000 [00:27<00:25, 7551.78it/s] 53%|█████▎    | 210092/400000 [00:27<00:25, 7411.66it/s] 53%|█████▎    | 210839/400000 [00:27<00:25, 7427.26it/s] 53%|█████▎    | 211583/400000 [00:27<00:25, 7323.41it/s] 53%|█████▎    | 212318/400000 [00:27<00:25, 7328.81it/s] 53%|█████▎    | 213060/400000 [00:27<00:25, 7354.39it/s] 53%|█████▎    | 213815/400000 [00:28<00:25, 7411.00it/s] 54%|█████▎    | 214557/400000 [00:28<00:25, 7320.74it/s] 54%|█████▍    | 215290/400000 [00:28<00:25, 7249.20it/s] 54%|█████▍    | 216034/400000 [00:28<00:25, 7303.03it/s] 54%|█████▍    | 216800/400000 [00:28<00:24, 7405.04it/s] 54%|█████▍    | 217554/400000 [00:28<00:24, 7442.94it/s] 55%|█████▍    | 218299/400000 [00:28<00:24, 7429.31it/s] 55%|█████▍    | 219068/400000 [00:28<00:24, 7505.12it/s] 55%|█████▍    | 219819/400000 [00:28<00:24, 7396.47it/s] 55%|█████▌    | 220560/400000 [00:29<00:24, 7342.60it/s] 55%|█████▌    | 221295/400000 [00:29<00:24, 7322.97it/s] 56%|█████▌    | 222028/400000 [00:29<00:24, 7255.32it/s] 56%|█████▌    | 222764/400000 [00:29<00:24, 7284.74it/s] 56%|█████▌    | 223499/400000 [00:29<00:24, 7303.71it/s] 56%|█████▌    | 224237/400000 [00:29<00:23, 7325.72it/s] 56%|█████▌    | 224970/400000 [00:29<00:23, 7299.71it/s] 56%|█████▋    | 225701/400000 [00:29<00:24, 7148.95it/s] 57%|█████▋    | 226443/400000 [00:29<00:24, 7226.00it/s] 57%|█████▋    | 227180/400000 [00:29<00:23, 7267.85it/s] 57%|█████▋    | 227920/400000 [00:30<00:23, 7305.19it/s] 57%|█████▋    | 228651/400000 [00:30<00:23, 7291.85it/s] 57%|█████▋    | 229381/400000 [00:30<00:23, 7266.47it/s] 58%|█████▊    | 230118/400000 [00:30<00:23, 7296.78it/s] 58%|█████▊    | 230862/400000 [00:30<00:23, 7337.62it/s] 58%|█████▊    | 231600/400000 [00:30<00:22, 7349.13it/s] 58%|█████▊    | 232336/400000 [00:30<00:23, 7207.60it/s] 58%|█████▊    | 233090/400000 [00:30<00:22, 7304.14it/s] 58%|█████▊    | 233837/400000 [00:30<00:22, 7350.86it/s] 59%|█████▊    | 234578/400000 [00:30<00:22, 7368.38it/s] 59%|█████▉    | 235316/400000 [00:31<00:22, 7368.27it/s] 59%|█████▉    | 236069/400000 [00:31<00:22, 7414.53it/s] 59%|█████▉    | 236811/400000 [00:31<00:22, 7408.46it/s] 59%|█████▉    | 237553/400000 [00:31<00:22, 7342.87it/s] 60%|█████▉    | 238306/400000 [00:31<00:21, 7396.56it/s] 60%|█████▉    | 239078/400000 [00:31<00:21, 7489.78it/s] 60%|█████▉    | 239829/400000 [00:31<00:21, 7492.60it/s] 60%|██████    | 240579/400000 [00:31<00:21, 7437.83it/s] 60%|██████    | 241324/400000 [00:31<00:21, 7415.73it/s] 61%|██████    | 242066/400000 [00:31<00:21, 7387.40it/s] 61%|██████    | 242805/400000 [00:32<00:21, 7346.90it/s] 61%|██████    | 243554/400000 [00:32<00:21, 7387.95it/s] 61%|██████    | 244293/400000 [00:32<00:21, 7373.49it/s] 61%|██████▏   | 245031/400000 [00:32<00:21, 7364.22it/s] 61%|██████▏   | 245769/400000 [00:32<00:20, 7368.61it/s] 62%|██████▏   | 246516/400000 [00:32<00:20, 7396.43it/s] 62%|██████▏   | 247256/400000 [00:32<00:21, 7198.00it/s] 62%|██████▏   | 248006/400000 [00:32<00:20, 7285.14it/s] 62%|██████▏   | 248758/400000 [00:32<00:20, 7352.31it/s] 62%|██████▏   | 249521/400000 [00:32<00:20, 7432.14it/s] 63%|██████▎   | 250284/400000 [00:33<00:19, 7489.38it/s] 63%|██████▎   | 251054/400000 [00:33<00:19, 7548.55it/s] 63%|██████▎   | 251810/400000 [00:33<00:19, 7512.80it/s] 63%|██████▎   | 252574/400000 [00:33<00:19, 7548.79it/s] 63%|██████▎   | 253336/400000 [00:33<00:19, 7567.39it/s] 64%|██████▎   | 254094/400000 [00:33<00:19, 7570.70it/s] 64%|██████▎   | 254852/400000 [00:33<00:19, 7488.95it/s] 64%|██████▍   | 255602/400000 [00:33<00:19, 7462.64it/s] 64%|██████▍   | 256378/400000 [00:33<00:19, 7547.04it/s] 64%|██████▍   | 257144/400000 [00:33<00:18, 7578.31it/s] 64%|██████▍   | 257918/400000 [00:34<00:18, 7625.42it/s] 65%|██████▍   | 258717/400000 [00:34<00:18, 7730.02it/s] 65%|██████▍   | 259491/400000 [00:34<00:18, 7703.52it/s] 65%|██████▌   | 260262/400000 [00:34<00:18, 7587.51it/s] 65%|██████▌   | 261047/400000 [00:34<00:18, 7662.12it/s] 65%|██████▌   | 261836/400000 [00:34<00:17, 7726.44it/s] 66%|██████▌   | 262643/400000 [00:34<00:17, 7823.53it/s] 66%|██████▌   | 263442/400000 [00:34<00:17, 7871.95it/s] 66%|██████▌   | 264230/400000 [00:34<00:17, 7838.71it/s] 66%|██████▋   | 265015/400000 [00:34<00:17, 7839.52it/s] 66%|██████▋   | 265800/400000 [00:35<00:17, 7670.34it/s] 67%|██████▋   | 266584/400000 [00:35<00:17, 7717.74it/s] 67%|██████▋   | 267357/400000 [00:35<00:17, 7668.42it/s] 67%|██████▋   | 268125/400000 [00:35<00:17, 7621.49it/s] 67%|██████▋   | 268888/400000 [00:35<00:17, 7570.30it/s] 67%|██████▋   | 269676/400000 [00:35<00:17, 7659.63it/s] 68%|██████▊   | 270467/400000 [00:35<00:16, 7730.57it/s] 68%|██████▊   | 271241/400000 [00:35<00:16, 7720.45it/s] 68%|██████▊   | 272018/400000 [00:35<00:16, 7734.57it/s] 68%|██████▊   | 272816/400000 [00:35<00:16, 7805.06it/s] 68%|██████▊   | 273632/400000 [00:36<00:15, 7907.90it/s] 69%|██████▊   | 274435/400000 [00:36<00:15, 7941.71it/s] 69%|██████▉   | 275240/400000 [00:36<00:15, 7971.41it/s] 69%|██████▉   | 276054/400000 [00:36<00:15, 8020.62it/s] 69%|██████▉   | 276857/400000 [00:36<00:15, 8000.11it/s] 69%|██████▉   | 277658/400000 [00:36<00:15, 7960.09it/s] 70%|██████▉   | 278466/400000 [00:36<00:15, 7993.50it/s] 70%|██████▉   | 279272/400000 [00:36<00:15, 8012.08it/s] 70%|███████   | 280074/400000 [00:36<00:15, 7989.23it/s] 70%|███████   | 280874/400000 [00:36<00:14, 7968.76it/s] 70%|███████   | 281686/400000 [00:37<00:14, 8011.20it/s] 71%|███████   | 282488/400000 [00:37<00:14, 7968.54it/s] 71%|███████   | 283296/400000 [00:37<00:14, 7998.03it/s] 71%|███████   | 284096/400000 [00:37<00:14, 7980.84it/s] 71%|███████   | 284895/400000 [00:37<00:14, 7977.54it/s] 71%|███████▏  | 285697/400000 [00:37<00:14, 7989.56it/s] 72%|███████▏  | 286511/400000 [00:37<00:14, 8033.79it/s] 72%|███████▏  | 287315/400000 [00:37<00:14, 8026.04it/s] 72%|███████▏  | 288118/400000 [00:37<00:13, 8010.17it/s] 72%|███████▏  | 288921/400000 [00:37<00:13, 8013.48it/s] 72%|███████▏  | 289723/400000 [00:38<00:13, 8004.32it/s] 73%|███████▎  | 290524/400000 [00:38<00:13, 7973.56it/s] 73%|███████▎  | 291322/400000 [00:38<00:14, 7714.79it/s] 73%|███████▎  | 292096/400000 [00:38<00:14, 7542.94it/s] 73%|███████▎  | 292885/400000 [00:38<00:14, 7642.90it/s] 73%|███████▎  | 293688/400000 [00:38<00:13, 7752.65it/s] 74%|███████▎  | 294465/400000 [00:38<00:13, 7747.71it/s] 74%|███████▍  | 295279/400000 [00:38<00:13, 7858.62it/s] 74%|███████▍  | 296077/400000 [00:38<00:13, 7894.38it/s] 74%|███████▍  | 296892/400000 [00:39<00:12, 7967.41it/s] 74%|███████▍  | 297690/400000 [00:39<00:12, 7970.78it/s] 75%|███████▍  | 298488/400000 [00:39<00:12, 7903.36it/s] 75%|███████▍  | 299279/400000 [00:39<00:12, 7884.40it/s] 75%|███████▌  | 300068/400000 [00:39<00:12, 7874.89it/s] 75%|███████▌  | 300877/400000 [00:39<00:12, 7936.63it/s] 75%|███████▌  | 301684/400000 [00:39<00:12, 7974.48it/s] 76%|███████▌  | 302484/400000 [00:39<00:12, 7980.59it/s] 76%|███████▌  | 303298/400000 [00:39<00:12, 8027.39it/s] 76%|███████▌  | 304101/400000 [00:39<00:12, 7906.28it/s] 76%|███████▌  | 304893/400000 [00:40<00:12, 7787.11it/s] 76%|███████▋  | 305683/400000 [00:40<00:12, 7820.51it/s] 77%|███████▋  | 306487/400000 [00:40<00:11, 7885.01it/s] 77%|███████▋  | 307299/400000 [00:40<00:11, 7950.45it/s] 77%|███████▋  | 308109/400000 [00:40<00:11, 7992.62it/s] 77%|███████▋  | 308917/400000 [00:40<00:11, 8018.00it/s] 77%|███████▋  | 309720/400000 [00:40<00:11, 7951.08it/s] 78%|███████▊  | 310534/400000 [00:40<00:11, 8005.34it/s] 78%|███████▊  | 311335/400000 [00:40<00:11, 7742.44it/s] 78%|███████▊  | 312112/400000 [00:40<00:11, 7739.20it/s] 78%|███████▊  | 312921/400000 [00:41<00:11, 7838.60it/s] 78%|███████▊  | 313722/400000 [00:41<00:10, 7886.57it/s] 79%|███████▊  | 314514/400000 [00:41<00:10, 7895.27it/s] 79%|███████▉  | 315326/400000 [00:41<00:10, 7959.47it/s] 79%|███████▉  | 316144/400000 [00:41<00:10, 8023.10it/s] 79%|███████▉  | 316947/400000 [00:41<00:10, 7951.23it/s] 79%|███████▉  | 317743/400000 [00:41<00:10, 7935.03it/s] 80%|███████▉  | 318542/400000 [00:41<00:10, 7947.77it/s] 80%|███████▉  | 319338/400000 [00:41<00:10, 7946.02it/s] 80%|████████  | 320133/400000 [00:41<00:10, 7919.96it/s] 80%|████████  | 320930/400000 [00:42<00:09, 7934.55it/s] 80%|████████  | 321724/400000 [00:42<00:09, 7906.67it/s] 81%|████████  | 322515/400000 [00:42<00:09, 7828.86it/s] 81%|████████  | 323299/400000 [00:42<00:09, 7828.66it/s] 81%|████████  | 324083/400000 [00:42<00:09, 7808.49it/s] 81%|████████  | 324864/400000 [00:42<00:09, 7793.56it/s] 81%|████████▏ | 325671/400000 [00:42<00:09, 7871.16it/s] 82%|████████▏ | 326486/400000 [00:42<00:09, 7950.92it/s] 82%|████████▏ | 327284/400000 [00:42<00:09, 7958.84it/s] 82%|████████▏ | 328081/400000 [00:42<00:09, 7912.01it/s] 82%|████████▏ | 328876/400000 [00:43<00:08, 7921.34it/s] 82%|████████▏ | 329675/400000 [00:43<00:08, 7940.16it/s] 83%|████████▎ | 330470/400000 [00:43<00:08, 7864.78it/s] 83%|████████▎ | 331257/400000 [00:43<00:08, 7830.28it/s] 83%|████████▎ | 332041/400000 [00:43<00:08, 7816.36it/s] 83%|████████▎ | 332823/400000 [00:43<00:08, 7790.07it/s] 83%|████████▎ | 333623/400000 [00:43<00:08, 7849.31it/s] 84%|████████▎ | 334416/400000 [00:43<00:08, 7871.76it/s] 84%|████████▍ | 335204/400000 [00:43<00:08, 7806.68it/s] 84%|████████▍ | 335985/400000 [00:43<00:08, 7755.13it/s] 84%|████████▍ | 336761/400000 [00:44<00:08, 7743.06it/s] 84%|████████▍ | 337553/400000 [00:44<00:08, 7794.41it/s] 85%|████████▍ | 338335/400000 [00:44<00:07, 7799.84it/s] 85%|████████▍ | 339155/400000 [00:44<00:07, 7915.72it/s] 85%|████████▍ | 339954/400000 [00:44<00:07, 7936.18it/s] 85%|████████▌ | 340767/400000 [00:44<00:07, 7990.74it/s] 85%|████████▌ | 341584/400000 [00:44<00:07, 8041.73it/s] 86%|████████▌ | 342390/400000 [00:44<00:07, 8044.70it/s] 86%|████████▌ | 343195/400000 [00:44<00:07, 7993.13it/s] 86%|████████▌ | 343995/400000 [00:44<00:07, 7866.60it/s] 86%|████████▌ | 344786/400000 [00:45<00:07, 7877.59it/s] 86%|████████▋ | 345575/400000 [00:45<00:06, 7867.72it/s] 87%|████████▋ | 346375/400000 [00:45<00:06, 7906.52it/s] 87%|████████▋ | 347166/400000 [00:45<00:06, 7832.80it/s] 87%|████████▋ | 347950/400000 [00:45<00:06, 7799.91it/s] 87%|████████▋ | 348731/400000 [00:45<00:06, 7742.26it/s] 87%|████████▋ | 349506/400000 [00:45<00:06, 7595.49it/s] 88%|████████▊ | 350303/400000 [00:45<00:06, 7702.67it/s] 88%|████████▊ | 351075/400000 [00:45<00:06, 7661.10it/s] 88%|████████▊ | 351842/400000 [00:45<00:06, 7608.23it/s] 88%|████████▊ | 352672/400000 [00:46<00:06, 7801.67it/s] 88%|████████▊ | 353481/400000 [00:46<00:05, 7883.60it/s] 89%|████████▊ | 354291/400000 [00:46<00:05, 7939.26it/s] 89%|████████▉ | 355086/400000 [00:46<00:05, 7801.99it/s] 89%|████████▉ | 355876/400000 [00:46<00:05, 7826.94it/s] 89%|████████▉ | 356669/400000 [00:46<00:05, 7855.05it/s] 89%|████████▉ | 357456/400000 [00:46<00:05, 7738.58it/s] 90%|████████▉ | 358259/400000 [00:46<00:05, 7821.53it/s] 90%|████████▉ | 359063/400000 [00:46<00:05, 7884.70it/s] 90%|████████▉ | 359853/400000 [00:47<00:05, 7835.91it/s] 90%|█████████ | 360638/400000 [00:47<00:05, 7800.32it/s] 90%|█████████ | 361419/400000 [00:47<00:04, 7724.50it/s] 91%|█████████ | 362192/400000 [00:47<00:04, 7633.32it/s] 91%|█████████ | 362986/400000 [00:47<00:04, 7722.76it/s] 91%|█████████ | 363764/400000 [00:47<00:04, 7737.77it/s] 91%|█████████ | 364556/400000 [00:47<00:04, 7790.89it/s] 91%|█████████▏| 365354/400000 [00:47<00:04, 7845.11it/s] 92%|█████████▏| 366139/400000 [00:47<00:04, 7630.51it/s] 92%|█████████▏| 366920/400000 [00:47<00:04, 7682.14it/s] 92%|█████████▏| 367696/400000 [00:48<00:04, 7703.38it/s] 92%|█████████▏| 368504/400000 [00:48<00:04, 7811.72it/s] 92%|█████████▏| 369287/400000 [00:48<00:04, 7546.83it/s] 93%|█████████▎| 370070/400000 [00:48<00:03, 7628.46it/s] 93%|█████████▎| 370875/400000 [00:48<00:03, 7747.22it/s] 93%|█████████▎| 371652/400000 [00:48<00:03, 7727.98it/s] 93%|█████████▎| 372427/400000 [00:48<00:03, 7662.99it/s] 93%|█████████▎| 373233/400000 [00:48<00:03, 7776.17it/s] 94%|█████████▎| 374012/400000 [00:48<00:03, 7778.24it/s] 94%|█████████▎| 374791/400000 [00:48<00:03, 7766.10it/s] 94%|█████████▍| 375569/400000 [00:49<00:03, 7715.40it/s] 94%|█████████▍| 376342/400000 [00:49<00:03, 7640.99it/s] 94%|█████████▍| 377107/400000 [00:49<00:03, 7601.45it/s] 94%|█████████▍| 377872/400000 [00:49<00:02, 7615.64it/s] 95%|█████████▍| 378646/400000 [00:49<00:02, 7651.79it/s] 95%|█████████▍| 379423/400000 [00:49<00:02, 7685.51it/s] 95%|█████████▌| 380233/400000 [00:49<00:02, 7804.77it/s] 95%|█████████▌| 381053/400000 [00:49<00:02, 7918.31it/s] 95%|█████████▌| 381873/400000 [00:49<00:02, 7999.56it/s] 96%|█████████▌| 382676/400000 [00:49<00:02, 8006.67it/s] 96%|█████████▌| 383478/400000 [00:50<00:02, 7999.38it/s] 96%|█████████▌| 384279/400000 [00:50<00:01, 7967.64it/s] 96%|█████████▋| 385077/400000 [00:50<00:01, 7931.41it/s] 96%|█████████▋| 385878/400000 [00:50<00:01, 7953.58it/s] 97%|█████████▋| 386684/400000 [00:50<00:01, 7985.10it/s] 97%|█████████▋| 387498/400000 [00:50<00:01, 8029.36it/s] 97%|█████████▋| 388318/400000 [00:50<00:01, 8078.45it/s] 97%|█████████▋| 389142/400000 [00:50<00:01, 8125.26it/s] 97%|█████████▋| 389955/400000 [00:50<00:01, 8106.83it/s] 98%|█████████▊| 390766/400000 [00:50<00:01, 7990.01it/s] 98%|█████████▊| 391566/400000 [00:51<00:01, 7930.12it/s] 98%|█████████▊| 392360/400000 [00:51<00:00, 7745.82it/s] 98%|█████████▊| 393136/400000 [00:51<00:00, 7647.50it/s] 98%|█████████▊| 393927/400000 [00:51<00:00, 7724.15it/s] 99%|█████████▊| 394731/400000 [00:51<00:00, 7814.65it/s] 99%|█████████▉| 395546/400000 [00:51<00:00, 7910.03it/s] 99%|█████████▉| 396338/400000 [00:51<00:00, 7801.66it/s] 99%|█████████▉| 397120/400000 [00:51<00:00, 7761.31it/s] 99%|█████████▉| 397897/400000 [00:51<00:00, 7674.13it/s]100%|█████████▉| 398666/400000 [00:52<00:00, 7504.15it/s]100%|█████████▉| 399418/400000 [00:52<00:00, 7259.26it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7664.14it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7dd0d23d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011024336872864861 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.0106842438114128 	 Accuracy: 70

  model saves at 70% accuracy 

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

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 

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
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 16:25:44.648272: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 16:25:44.653254: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-12 16:25:44.653388: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55716c2658b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 16:25:44.653405: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7d7a28b7f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9426 - accuracy: 0.4820
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8455 - accuracy: 0.4883 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6181 - accuracy: 0.5032
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6428 - accuracy: 0.5016
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 4s - loss: 7.6443 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6187 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6233 - accuracy: 0.5028
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6392 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6549 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7d4d893f98> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 75, 40)            1720      
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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f7d44152e48> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2191 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.1821 - val_crf_viterbi_accuracy: 0.6533

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

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
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
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
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

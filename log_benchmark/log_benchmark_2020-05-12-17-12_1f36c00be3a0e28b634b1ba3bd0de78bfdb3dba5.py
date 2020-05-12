
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fdf1b5bbfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 17:12:28.665997
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 17:12:28.671231
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 17:12:28.674302
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 17:12:28.677697
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fdf27385438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355234.4375
Epoch 2/10

1/1 [==============================] - 0s 111ms/step - loss: 298193.6250
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 207067.0312
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 131101.5781
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 81155.3125
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 51218.5391
Epoch 7/10

1/1 [==============================] - 0s 94ms/step - loss: 33569.4766
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 23065.7227
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 16648.9043
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 12558.1475

  #### Inference Need return ypred, ytrue ######################### 
[[-1.32469535e-01  5.56668162e-01 -2.70431578e-01 -3.53603482e-01
   2.92941928e-01 -1.17346621e+00 -2.06286386e-01  3.88659835e-01
   5.94256938e-01  5.03715038e-01 -4.75650012e-01  1.19763926e-01
  -1.11985362e+00 -9.80132818e-02 -1.34165466e-01  7.72473752e-01
   4.81890976e-01 -3.50239336e-01  5.33400476e-01 -5.64351678e-03
  -7.91568220e-01  2.85159707e-01 -9.74449813e-01 -5.14181018e-01
   8.00760746e-01  1.41663551e-02  1.86320513e-01 -4.62044775e-03
   3.67341399e-01 -6.39570594e-01 -9.08015966e-01  1.03190041e+00
   4.14122820e-01 -2.85485297e-01  1.18652213e+00 -2.48572767e-01
   1.16239047e+00  1.11537576e-01 -6.99896574e-01 -1.16783366e-01
  -1.03539526e-02  1.66665852e-01  4.52949405e-01  2.36484855e-02
  -3.77204031e-01  7.96147883e-01  8.52454603e-02  8.24115127e-02
  -3.57037157e-01 -5.93653083e-01  6.80382133e-01 -2.56282091e-02
   2.27699697e-01 -7.06202924e-01 -2.53641874e-01 -8.99977505e-01
   2.82055199e-01  1.48964077e-01  5.24889231e-01 -4.23309773e-01
   2.73615062e-01  5.24861526e+00  4.81070852e+00  5.83383274e+00
   4.66084623e+00  5.70283127e+00  4.87103748e+00  4.91483212e+00
   5.44010067e+00  6.32601404e+00  5.95402527e+00  5.75391674e+00
   5.73057127e+00  5.06752014e+00  5.18781233e+00  5.55856085e+00
   5.06717014e+00  5.11533833e+00  6.55337143e+00  5.39977551e+00
   5.39961481e+00  5.62143040e+00  5.71500731e+00  5.12005186e+00
   5.25597620e+00  5.40413904e+00  6.20252848e+00  4.86270332e+00
   4.78379679e+00  6.71471548e+00  5.32617188e+00  5.08125496e+00
   5.44563341e+00  6.34536123e+00  5.73415041e+00  5.05801392e+00
   5.20988989e+00  5.15915918e+00  5.05340052e+00  6.32027483e+00
   5.16851044e+00  6.55052757e+00  6.01003456e+00  5.12619495e+00
   4.82291794e+00  4.76371574e+00  5.52258921e+00  6.02261877e+00
   5.59648132e+00  5.07431984e+00  6.00687313e+00  6.08622742e+00
   5.84298897e+00  5.51231670e+00  4.23257637e+00  5.89750051e+00
   5.15873623e+00  5.38210964e+00  5.74500179e+00  5.79771948e+00
  -5.52797616e-01  4.18279618e-02  1.52633059e+00 -1.97495863e-01
  -5.32320261e-01  8.82156849e-01  2.50861257e-01 -1.79400295e-01
  -1.24022353e+00  5.52074611e-01 -7.90207922e-01  5.75670004e-01
   1.17500329e+00 -6.48953915e-01 -6.89893961e-01 -5.32349706e-01
  -2.55562663e-01 -2.89992690e-01  6.24819338e-01 -1.74629420e-01
  -6.26094460e-01 -6.92178607e-02 -8.21531296e-01 -2.46391729e-01
   7.34820843e-01  2.15278268e-01 -2.56724417e-01  7.66190648e-01
  -2.33247206e-01 -8.42461944e-01 -5.92170179e-01 -5.40301323e-01
   1.38812721e-01  4.00559336e-01  5.05419523e-02  1.69597745e-01
   2.40887925e-01  1.22423559e-01 -7.91645288e-01 -5.72467744e-01
   1.28281260e+00  8.48467290e-01 -7.82216609e-01  1.93999231e-01
  -8.55371356e-01  2.88346827e-01 -6.73850775e-02 -1.02464509e+00
   5.86555004e-01 -5.91491699e-01 -7.57788062e-01 -2.67183632e-02
   3.20642233e-01 -1.33132994e-01  1.69232696e-01 -3.11715513e-01
  -3.10540527e-01  6.19555712e-01 -4.67702448e-02 -1.98352188e-02
   1.08173275e+00  1.12125695e+00  5.94178915e-01  1.40041387e+00
   7.85409451e-01  8.70968819e-01  5.62064409e-01  7.76471496e-01
   1.01158309e+00  2.14554644e+00  1.34877050e+00  6.98362231e-01
   1.83824050e+00  1.83971524e+00  2.23284388e+00  1.17583799e+00
   1.33311784e+00  2.06455994e+00  4.52522516e-01  1.22504210e+00
   1.21343720e+00  1.17466795e+00  1.54982257e+00  6.73545241e-01
   1.53278470e+00  1.37221253e+00  1.78341222e+00  1.36276901e+00
   1.43002796e+00  1.19306111e+00  3.18226159e-01  2.43525457e+00
   6.42168283e-01  1.55549943e+00  3.79109263e-01  5.42813003e-01
   5.66176534e-01  4.89769578e-01  5.85426569e-01  9.64742243e-01
   6.19644165e-01  4.35832024e-01  1.93221283e+00  7.56860554e-01
   4.75177407e-01  5.47835171e-01  6.08434975e-01  1.84947467e+00
   1.06778562e+00  1.10086250e+00  2.44069219e-01  1.08564210e+00
   9.99418974e-01  5.06927013e-01  5.83624363e-01  6.52769923e-01
   3.48377407e-01  1.05176127e+00  4.04663980e-01  3.42149615e-01
   5.96544147e-02  5.68649101e+00  6.07889271e+00  5.53741837e+00
   6.50342178e+00  6.37176752e+00  6.05192804e+00  6.44896603e+00
   6.24183464e+00  6.62897730e+00  5.06974840e+00  5.84664774e+00
   6.93303061e+00  6.06419277e+00  5.49745035e+00  5.86989212e+00
   5.93084097e+00  6.11919737e+00  5.29575062e+00  5.91590118e+00
   5.69031572e+00  5.29432583e+00  5.20250654e+00  6.25935984e+00
   6.72191715e+00  5.64517641e+00  5.52403116e+00  5.78498268e+00
   6.27459860e+00  5.89780760e+00  6.21096277e+00  6.65554047e+00
   6.59965658e+00  6.95195103e+00  6.60856915e+00  5.67202187e+00
   6.23974752e+00  6.42540216e+00  5.45089960e+00  6.04777861e+00
   5.21381426e+00  6.06685162e+00  5.21482706e+00  6.81253004e+00
   6.47344971e+00  5.26207685e+00  6.12101221e+00  6.10494232e+00
   6.50402594e+00  6.51543331e+00  6.21869516e+00  6.33862782e+00
   5.41947699e+00  5.78368044e+00  7.08843899e+00  5.69430780e+00
   5.81973314e+00  5.59172297e+00  6.34043646e+00  5.66936922e+00
   7.93256164e-01  7.62670994e-01  2.11389899e-01  6.92720771e-01
   1.72450423e+00  1.99305415e+00  1.10513842e+00  2.01212931e+00
   9.11263287e-01  1.15346241e+00  2.61797071e-01  1.29177248e+00
   1.37300348e+00  8.62219751e-01  1.75842988e+00  1.10872042e+00
   9.27068472e-01  1.24956203e+00  8.65455151e-01  8.42099726e-01
   5.70923805e-01  1.21180117e+00  2.14844656e+00  2.02089214e+00
   8.70079696e-01  5.12602806e-01  5.52581966e-01  8.43857527e-01
   1.50313318e+00  1.71362710e+00  7.12806165e-01  2.30158997e+00
   1.64533269e+00  1.26817060e+00  6.10837162e-01  1.40954685e+00
   1.71966851e+00  1.76563764e+00  1.05533957e+00  4.68493462e-01
   8.20553422e-01  1.65299106e+00  5.65651894e-01  3.14514518e-01
   1.37333488e+00  9.20892179e-01  1.18723774e+00  1.52963042e+00
   1.54722643e+00  5.33494949e-01  9.65891838e-01  1.82548535e+00
   9.01398301e-01  1.25646782e+00  1.29434729e+00  1.48073697e+00
   1.44258189e+00  2.12758160e+00  1.28907180e+00  1.71123266e+00
  -4.00549603e+00  7.61669922e+00 -2.42095041e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 17:12:37.408868
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.523
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 17:12:37.413354
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9330.38
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 17:12:37.417203
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.8834
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 17:12:37.421056
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -834.601
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140595854283160
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140594644521592
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140594644522096
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140594644522600
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140594644523104
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140594644523608

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fdf1d25bf98> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.408854
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.380330
grad_step = 000002, loss = 0.355525
grad_step = 000003, loss = 0.330056
grad_step = 000004, loss = 0.303630
grad_step = 000005, loss = 0.280558
grad_step = 000006, loss = 0.270965
grad_step = 000007, loss = 0.269342
grad_step = 000008, loss = 0.256019
grad_step = 000009, loss = 0.240432
grad_step = 000010, loss = 0.230091
grad_step = 000011, loss = 0.222306
grad_step = 000012, loss = 0.214596
grad_step = 000013, loss = 0.205116
grad_step = 000014, loss = 0.193594
grad_step = 000015, loss = 0.180880
grad_step = 000016, loss = 0.168489
grad_step = 000017, loss = 0.157839
grad_step = 000018, loss = 0.149129
grad_step = 000019, loss = 0.139656
grad_step = 000020, loss = 0.130132
grad_step = 000021, loss = 0.121649
grad_step = 000022, loss = 0.114352
grad_step = 000023, loss = 0.107356
grad_step = 000024, loss = 0.100149
grad_step = 000025, loss = 0.092859
grad_step = 000026, loss = 0.086069
grad_step = 000027, loss = 0.080275
grad_step = 000028, loss = 0.075295
grad_step = 000029, loss = 0.069729
grad_step = 000030, loss = 0.063656
grad_step = 000031, loss = 0.058186
grad_step = 000032, loss = 0.053417
grad_step = 000033, loss = 0.048869
grad_step = 000034, loss = 0.044421
grad_step = 000035, loss = 0.040353
grad_step = 000036, loss = 0.036788
grad_step = 000037, loss = 0.033515
grad_step = 000038, loss = 0.030401
grad_step = 000039, loss = 0.027426
grad_step = 000040, loss = 0.024664
grad_step = 000041, loss = 0.022260
grad_step = 000042, loss = 0.020216
grad_step = 000043, loss = 0.018312
grad_step = 000044, loss = 0.016499
grad_step = 000045, loss = 0.014905
grad_step = 000046, loss = 0.013564
grad_step = 000047, loss = 0.012263
grad_step = 000048, loss = 0.010964
grad_step = 000049, loss = 0.009866
grad_step = 000050, loss = 0.009001
grad_step = 000051, loss = 0.008209
grad_step = 000052, loss = 0.007437
grad_step = 000053, loss = 0.006758
grad_step = 000054, loss = 0.006208
grad_step = 000055, loss = 0.005704
grad_step = 000056, loss = 0.005217
grad_step = 000057, loss = 0.004794
grad_step = 000058, loss = 0.004458
grad_step = 000059, loss = 0.004186
grad_step = 000060, loss = 0.003949
grad_step = 000061, loss = 0.003731
grad_step = 000062, loss = 0.003539
grad_step = 000063, loss = 0.003381
grad_step = 000064, loss = 0.003234
grad_step = 000065, loss = 0.003094
grad_step = 000066, loss = 0.002989
grad_step = 000067, loss = 0.002909
grad_step = 000068, loss = 0.002824
grad_step = 000069, loss = 0.002744
grad_step = 000070, loss = 0.002681
grad_step = 000071, loss = 0.002625
grad_step = 000072, loss = 0.002564
grad_step = 000073, loss = 0.002513
grad_step = 000074, loss = 0.002478
grad_step = 000075, loss = 0.002450
grad_step = 000076, loss = 0.002441
grad_step = 000077, loss = 0.002445
grad_step = 000078, loss = 0.002402
grad_step = 000079, loss = 0.002337
grad_step = 000080, loss = 0.002359
grad_step = 000081, loss = 0.002343
grad_step = 000082, loss = 0.002288
grad_step = 000083, loss = 0.002322
grad_step = 000084, loss = 0.002299
grad_step = 000085, loss = 0.002261
grad_step = 000086, loss = 0.002295
grad_step = 000087, loss = 0.002257
grad_step = 000088, loss = 0.002238
grad_step = 000089, loss = 0.002261
grad_step = 000090, loss = 0.002223
grad_step = 000091, loss = 0.002220
grad_step = 000092, loss = 0.002227
grad_step = 000093, loss = 0.002194
grad_step = 000094, loss = 0.002194
grad_step = 000095, loss = 0.002195
grad_step = 000096, loss = 0.002168
grad_step = 000097, loss = 0.002164
grad_step = 000098, loss = 0.002167
grad_step = 000099, loss = 0.002148
grad_step = 000100, loss = 0.002133
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002137
grad_step = 000102, loss = 0.002134
grad_step = 000103, loss = 0.002114
grad_step = 000104, loss = 0.002102
grad_step = 000105, loss = 0.002102
grad_step = 000106, loss = 0.002104
grad_step = 000107, loss = 0.002101
grad_step = 000108, loss = 0.002085
grad_step = 000109, loss = 0.002071
grad_step = 000110, loss = 0.002063
grad_step = 000111, loss = 0.002060
grad_step = 000112, loss = 0.002063
grad_step = 000113, loss = 0.002068
grad_step = 000114, loss = 0.002086
grad_step = 000115, loss = 0.002082
grad_step = 000116, loss = 0.002078
grad_step = 000117, loss = 0.002037
grad_step = 000118, loss = 0.002022
grad_step = 000119, loss = 0.002037
grad_step = 000120, loss = 0.002044
grad_step = 000121, loss = 0.002042
grad_step = 000122, loss = 0.002012
grad_step = 000123, loss = 0.001993
grad_step = 000124, loss = 0.001990
grad_step = 000125, loss = 0.001997
grad_step = 000126, loss = 0.002015
grad_step = 000127, loss = 0.002010
grad_step = 000128, loss = 0.002002
grad_step = 000129, loss = 0.001969
grad_step = 000130, loss = 0.001957
grad_step = 000131, loss = 0.001965
grad_step = 000132, loss = 0.001970
grad_step = 000133, loss = 0.001971
grad_step = 000134, loss = 0.001949
grad_step = 000135, loss = 0.001933
grad_step = 000136, loss = 0.001922
grad_step = 000137, loss = 0.001916
grad_step = 000138, loss = 0.001914
grad_step = 000139, loss = 0.001919
grad_step = 000140, loss = 0.001938
grad_step = 000141, loss = 0.001956
grad_step = 000142, loss = 0.002001
grad_step = 000143, loss = 0.001927
grad_step = 000144, loss = 0.001875
grad_step = 000145, loss = 0.001880
grad_step = 000146, loss = 0.001905
grad_step = 000147, loss = 0.001916
grad_step = 000148, loss = 0.001865
grad_step = 000149, loss = 0.001833
grad_step = 000150, loss = 0.001834
grad_step = 000151, loss = 0.001855
grad_step = 000152, loss = 0.001902
grad_step = 000153, loss = 0.001882
grad_step = 000154, loss = 0.001862
grad_step = 000155, loss = 0.001799
grad_step = 000156, loss = 0.001789
grad_step = 000157, loss = 0.001820
grad_step = 000158, loss = 0.001820
grad_step = 000159, loss = 0.001805
grad_step = 000160, loss = 0.001760
grad_step = 000161, loss = 0.001742
grad_step = 000162, loss = 0.001742
grad_step = 000163, loss = 0.001767
grad_step = 000164, loss = 0.001844
grad_step = 000165, loss = 0.001874
grad_step = 000166, loss = 0.001952
grad_step = 000167, loss = 0.001773
grad_step = 000168, loss = 0.001708
grad_step = 000169, loss = 0.001799
grad_step = 000170, loss = 0.001796
grad_step = 000171, loss = 0.001737
grad_step = 000172, loss = 0.001675
grad_step = 000173, loss = 0.001739
grad_step = 000174, loss = 0.001847
grad_step = 000175, loss = 0.001736
grad_step = 000176, loss = 0.001661
grad_step = 000177, loss = 0.001696
grad_step = 000178, loss = 0.001723
grad_step = 000179, loss = 0.001687
grad_step = 000180, loss = 0.001643
grad_step = 000181, loss = 0.001667
grad_step = 000182, loss = 0.001724
grad_step = 000183, loss = 0.001687
grad_step = 000184, loss = 0.001646
grad_step = 000185, loss = 0.001621
grad_step = 000186, loss = 0.001639
grad_step = 000187, loss = 0.001671
grad_step = 000188, loss = 0.001656
grad_step = 000189, loss = 0.001635
grad_step = 000190, loss = 0.001606
grad_step = 000191, loss = 0.001603
grad_step = 000192, loss = 0.001616
grad_step = 000193, loss = 0.001635
grad_step = 000194, loss = 0.001667
grad_step = 000195, loss = 0.001657
grad_step = 000196, loss = 0.001640
grad_step = 000197, loss = 0.001599
grad_step = 000198, loss = 0.001580
grad_step = 000199, loss = 0.001587
grad_step = 000200, loss = 0.001609
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001635
grad_step = 000202, loss = 0.001636
grad_step = 000203, loss = 0.001650
grad_step = 000204, loss = 0.001612
grad_step = 000205, loss = 0.001573
grad_step = 000206, loss = 0.001556
grad_step = 000207, loss = 0.001571
grad_step = 000208, loss = 0.001590
grad_step = 000209, loss = 0.001584
grad_step = 000210, loss = 0.001569
grad_step = 000211, loss = 0.001545
grad_step = 000212, loss = 0.001534
grad_step = 000213, loss = 0.001537
grad_step = 000214, loss = 0.001546
grad_step = 000215, loss = 0.001566
grad_step = 000216, loss = 0.001577
grad_step = 000217, loss = 0.001599
grad_step = 000218, loss = 0.001581
grad_step = 000219, loss = 0.001555
grad_step = 000220, loss = 0.001519
grad_step = 000221, loss = 0.001501
grad_step = 000222, loss = 0.001508
grad_step = 000223, loss = 0.001523
grad_step = 000224, loss = 0.001537
grad_step = 000225, loss = 0.001529
grad_step = 000226, loss = 0.001516
grad_step = 000227, loss = 0.001493
grad_step = 000228, loss = 0.001473
grad_step = 000229, loss = 0.001463
grad_step = 000230, loss = 0.001463
grad_step = 000231, loss = 0.001472
grad_step = 000232, loss = 0.001493
grad_step = 000233, loss = 0.001561
grad_step = 000234, loss = 0.001644
grad_step = 000235, loss = 0.001865
grad_step = 000236, loss = 0.001687
grad_step = 000237, loss = 0.001475
grad_step = 000238, loss = 0.001460
grad_step = 000239, loss = 0.001588
grad_step = 000240, loss = 0.001583
grad_step = 000241, loss = 0.001421
grad_step = 000242, loss = 0.001512
grad_step = 000243, loss = 0.001623
grad_step = 000244, loss = 0.001429
grad_step = 000245, loss = 0.001458
grad_step = 000246, loss = 0.001578
grad_step = 000247, loss = 0.001423
grad_step = 000248, loss = 0.001431
grad_step = 000249, loss = 0.001538
grad_step = 000250, loss = 0.001414
grad_step = 000251, loss = 0.001397
grad_step = 000252, loss = 0.001483
grad_step = 000253, loss = 0.001406
grad_step = 000254, loss = 0.001371
grad_step = 000255, loss = 0.001421
grad_step = 000256, loss = 0.001395
grad_step = 000257, loss = 0.001356
grad_step = 000258, loss = 0.001372
grad_step = 000259, loss = 0.001384
grad_step = 000260, loss = 0.001365
grad_step = 000261, loss = 0.001341
grad_step = 000262, loss = 0.001353
grad_step = 000263, loss = 0.001372
grad_step = 000264, loss = 0.001354
grad_step = 000265, loss = 0.001332
grad_step = 000266, loss = 0.001326
grad_step = 000267, loss = 0.001335
grad_step = 000268, loss = 0.001345
grad_step = 000269, loss = 0.001339
grad_step = 000270, loss = 0.001326
grad_step = 000271, loss = 0.001312
grad_step = 000272, loss = 0.001305
grad_step = 000273, loss = 0.001306
grad_step = 000274, loss = 0.001311
grad_step = 000275, loss = 0.001319
grad_step = 000276, loss = 0.001323
grad_step = 000277, loss = 0.001332
grad_step = 000278, loss = 0.001332
grad_step = 000279, loss = 0.001333
grad_step = 000280, loss = 0.001322
grad_step = 000281, loss = 0.001309
grad_step = 000282, loss = 0.001292
grad_step = 000283, loss = 0.001278
grad_step = 000284, loss = 0.001270
grad_step = 000285, loss = 0.001268
grad_step = 000286, loss = 0.001270
grad_step = 000287, loss = 0.001274
grad_step = 000288, loss = 0.001279
grad_step = 000289, loss = 0.001283
grad_step = 000290, loss = 0.001291
grad_step = 000291, loss = 0.001295
grad_step = 000292, loss = 0.001312
grad_step = 000293, loss = 0.001317
grad_step = 000294, loss = 0.001336
grad_step = 000295, loss = 0.001320
grad_step = 000296, loss = 0.001301
grad_step = 000297, loss = 0.001258
grad_step = 000298, loss = 0.001231
grad_step = 000299, loss = 0.001231
grad_step = 000300, loss = 0.001250
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001268
grad_step = 000302, loss = 0.001256
grad_step = 000303, loss = 0.001236
grad_step = 000304, loss = 0.001214
grad_step = 000305, loss = 0.001207
grad_step = 000306, loss = 0.001215
grad_step = 000307, loss = 0.001225
grad_step = 000308, loss = 0.001243
grad_step = 000309, loss = 0.001249
grad_step = 000310, loss = 0.001279
grad_step = 000311, loss = 0.001282
grad_step = 000312, loss = 0.001301
grad_step = 000313, loss = 0.001267
grad_step = 000314, loss = 0.001234
grad_step = 000315, loss = 0.001190
grad_step = 000316, loss = 0.001173
grad_step = 000317, loss = 0.001185
grad_step = 000318, loss = 0.001205
grad_step = 000319, loss = 0.001234
grad_step = 000320, loss = 0.001218
grad_step = 000321, loss = 0.001197
grad_step = 000322, loss = 0.001164
grad_step = 000323, loss = 0.001150
grad_step = 000324, loss = 0.001155
grad_step = 000325, loss = 0.001171
grad_step = 000326, loss = 0.001194
grad_step = 000327, loss = 0.001201
grad_step = 000328, loss = 0.001230
grad_step = 000329, loss = 0.001224
grad_step = 000330, loss = 0.001237
grad_step = 000331, loss = 0.001199
grad_step = 000332, loss = 0.001164
grad_step = 000333, loss = 0.001124
grad_step = 000334, loss = 0.001117
grad_step = 000335, loss = 0.001137
grad_step = 000336, loss = 0.001157
grad_step = 000337, loss = 0.001182
grad_step = 000338, loss = 0.001161
grad_step = 000339, loss = 0.001138
grad_step = 000340, loss = 0.001105
grad_step = 000341, loss = 0.001090
grad_step = 000342, loss = 0.001093
grad_step = 000343, loss = 0.001106
grad_step = 000344, loss = 0.001128
grad_step = 000345, loss = 0.001140
grad_step = 000346, loss = 0.001183
grad_step = 000347, loss = 0.001203
grad_step = 000348, loss = 0.001257
grad_step = 000349, loss = 0.001207
grad_step = 000350, loss = 0.001159
grad_step = 000351, loss = 0.001077
grad_step = 000352, loss = 0.001062
grad_step = 000353, loss = 0.001108
grad_step = 000354, loss = 0.001130
grad_step = 000355, loss = 0.001137
grad_step = 000356, loss = 0.001074
grad_step = 000357, loss = 0.001040
grad_step = 000358, loss = 0.001054
grad_step = 000359, loss = 0.001078
grad_step = 000360, loss = 0.001097
grad_step = 000361, loss = 0.001066
grad_step = 000362, loss = 0.001038
grad_step = 000363, loss = 0.001018
grad_step = 000364, loss = 0.001021
grad_step = 000365, loss = 0.001040
grad_step = 000366, loss = 0.001050
grad_step = 000367, loss = 0.001064
grad_step = 000368, loss = 0.001051
grad_step = 000369, loss = 0.001042
grad_step = 000370, loss = 0.001019
grad_step = 000371, loss = 0.001001
grad_step = 000372, loss = 0.000988
grad_step = 000373, loss = 0.000982
grad_step = 000374, loss = 0.000983
grad_step = 000375, loss = 0.000986
grad_step = 000376, loss = 0.000996
grad_step = 000377, loss = 0.001007
grad_step = 000378, loss = 0.001033
grad_step = 000379, loss = 0.001057
grad_step = 000380, loss = 0.001136
grad_step = 000381, loss = 0.001160
grad_step = 000382, loss = 0.001250
grad_step = 000383, loss = 0.001146
grad_step = 000384, loss = 0.001046
grad_step = 000385, loss = 0.000950
grad_step = 000386, loss = 0.000986
grad_step = 000387, loss = 0.001100
grad_step = 000388, loss = 0.001068
grad_step = 000389, loss = 0.001020
grad_step = 000390, loss = 0.000940
grad_step = 000391, loss = 0.000964
grad_step = 000392, loss = 0.001031
grad_step = 000393, loss = 0.000991
grad_step = 000394, loss = 0.000937
grad_step = 000395, loss = 0.000920
grad_step = 000396, loss = 0.000955
grad_step = 000397, loss = 0.001016
grad_step = 000398, loss = 0.000978
grad_step = 000399, loss = 0.000932
grad_step = 000400, loss = 0.000900
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000919
grad_step = 000402, loss = 0.000959
grad_step = 000403, loss = 0.000954
grad_step = 000404, loss = 0.000941
grad_step = 000405, loss = 0.000898
grad_step = 000406, loss = 0.000877
grad_step = 000407, loss = 0.000886
grad_step = 000408, loss = 0.000905
grad_step = 000409, loss = 0.000926
grad_step = 000410, loss = 0.000916
grad_step = 000411, loss = 0.000903
grad_step = 000412, loss = 0.000876
grad_step = 000413, loss = 0.000860
grad_step = 000414, loss = 0.000851
grad_step = 000415, loss = 0.000853
grad_step = 000416, loss = 0.000861
grad_step = 000417, loss = 0.000866
grad_step = 000418, loss = 0.000878
grad_step = 000419, loss = 0.000882
grad_step = 000420, loss = 0.000897
grad_step = 000421, loss = 0.000901
grad_step = 000422, loss = 0.000917
grad_step = 000423, loss = 0.000909
grad_step = 000424, loss = 0.000918
grad_step = 000425, loss = 0.000881
grad_step = 000426, loss = 0.000848
grad_step = 000427, loss = 0.000817
grad_step = 000428, loss = 0.000813
grad_step = 000429, loss = 0.000829
grad_step = 000430, loss = 0.000838
grad_step = 000431, loss = 0.000839
grad_step = 000432, loss = 0.000819
grad_step = 000433, loss = 0.000801
grad_step = 000434, loss = 0.000792
grad_step = 000435, loss = 0.000795
grad_step = 000436, loss = 0.000804
grad_step = 000437, loss = 0.000809
grad_step = 000438, loss = 0.000819
grad_step = 000439, loss = 0.000818
grad_step = 000440, loss = 0.000824
grad_step = 000441, loss = 0.000819
grad_step = 000442, loss = 0.000826
grad_step = 000443, loss = 0.000819
grad_step = 000444, loss = 0.000825
grad_step = 000445, loss = 0.000815
grad_step = 000446, loss = 0.000816
grad_step = 000447, loss = 0.000796
grad_step = 000448, loss = 0.000779
grad_step = 000449, loss = 0.000757
grad_step = 000450, loss = 0.000747
grad_step = 000451, loss = 0.000750
grad_step = 000452, loss = 0.000759
grad_step = 000453, loss = 0.000774
grad_step = 000454, loss = 0.000777
grad_step = 000455, loss = 0.000786
grad_step = 000456, loss = 0.000779
grad_step = 000457, loss = 0.000780
grad_step = 000458, loss = 0.000765
grad_step = 000459, loss = 0.000759
grad_step = 000460, loss = 0.000744
grad_step = 000461, loss = 0.000731
grad_step = 000462, loss = 0.000720
grad_step = 000463, loss = 0.000713
grad_step = 000464, loss = 0.000710
grad_step = 000465, loss = 0.000710
grad_step = 000466, loss = 0.000713
grad_step = 000467, loss = 0.000716
grad_step = 000468, loss = 0.000729
grad_step = 000469, loss = 0.000750
grad_step = 000470, loss = 0.000817
grad_step = 000471, loss = 0.000895
grad_step = 000472, loss = 0.001164
grad_step = 000473, loss = 0.001267
grad_step = 000474, loss = 0.001433
grad_step = 000475, loss = 0.000873
grad_step = 000476, loss = 0.000718
grad_step = 000477, loss = 0.000921
grad_step = 000478, loss = 0.000913
grad_step = 000479, loss = 0.000869
grad_step = 000480, loss = 0.000718
grad_step = 000481, loss = 0.000823
grad_step = 000482, loss = 0.000927
grad_step = 000483, loss = 0.000729
grad_step = 000484, loss = 0.000740
grad_step = 000485, loss = 0.000881
grad_step = 000486, loss = 0.000769
grad_step = 000487, loss = 0.000692
grad_step = 000488, loss = 0.000713
grad_step = 000489, loss = 0.000748
grad_step = 000490, loss = 0.000744
grad_step = 000491, loss = 0.000679
grad_step = 000492, loss = 0.000680
grad_step = 000493, loss = 0.000719
grad_step = 000494, loss = 0.000714
grad_step = 000495, loss = 0.000697
grad_step = 000496, loss = 0.000658
grad_step = 000497, loss = 0.000649
grad_step = 000498, loss = 0.000679
grad_step = 000499, loss = 0.000690
grad_step = 000500, loss = 0.000680
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000652
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

  date_run                              2020-05-12 17:12:56.111744
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.159681
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 17:12:56.117309
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.055549
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 17:12:56.124503
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.108403
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 17:12:56.129520
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.155913
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
0   2020-05-12 17:12:28.665997  ...    mean_absolute_error
1   2020-05-12 17:12:28.671231  ...     mean_squared_error
2   2020-05-12 17:12:28.674302  ...  median_absolute_error
3   2020-05-12 17:12:28.677697  ...               r2_score
4   2020-05-12 17:12:37.408868  ...    mean_absolute_error
5   2020-05-12 17:12:37.413354  ...     mean_squared_error
6   2020-05-12 17:12:37.417203  ...  median_absolute_error
7   2020-05-12 17:12:37.421056  ...               r2_score
8   2020-05-12 17:12:56.111744  ...    mean_absolute_error
9   2020-05-12 17:12:56.117309  ...     mean_squared_error
10  2020-05-12 17:12:56.124503  ...  median_absolute_error
11  2020-05-12 17:12:56.129520  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 301329.41it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 389562.04it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 539143.21it/s] 35%|███▌      | 3497984/9912422 [00:00<00:08, 762926.23it/s] 66%|██████▌   | 6553600/9912422 [00:00<00:03, 1078305.38it/s] 94%|█████████▍| 9314304/9912422 [00:00<00:00, 1515051.22it/s]9920512it [00:01, 9821714.22it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 137420.68it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 303753.08it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 391944.53it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 542013.65it/s]1654784it [00:00, 2681277.77it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 48290.82it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b7064b9b0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b22ff9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b1fe460b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b22ff9e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b22582240> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b1fdab438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b1fda5b70> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b22ff9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b2253e630> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b1fdab438> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b70603e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9aea81e208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=65c327a9a3d7eaf67a5dd05a61f0b897e07286d47a77e43c6e167600930cc2cd
  Stored in directory: /tmp/pip-ephem-wheel-cache-2pshyzhs/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9a82619710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  196608/17464789 [..............................] - ETA: 22s
  385024/17464789 [..............................] - ETA: 14s
  786432/17464789 [>.............................] - ETA: 8s 
 1589248/17464789 [=>............................] - ETA: 4s
 3186688/17464789 [====>.........................] - ETA: 2s
 6152192/17464789 [=========>....................] - ETA: 1s
 9134080/17464789 [==============>...............] - ETA: 0s
12034048/17464789 [===================>..........] - ETA: 0s
15114240/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 17:14:26.221490: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 17:14:26.225868: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-12 17:14:26.226015: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c49db4a930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 17:14:26.226029: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6609 - accuracy: 0.5004
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 3s - loss: 7.6290 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6611 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 2s - loss: 7.6758 - accuracy: 0.4994
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6800 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7027 - accuracy: 0.4976
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7143 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7005 - accuracy: 0.4978
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6866 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 17:14:39.476867
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 17:14:39.476867  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<7:33:29, 31.7kB/s].vector_cache/glove.6B.zip:   0%|          | 238k/862M [00:00<5:19:15, 45.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.11M/862M [00:00<3:42:52, 64.2kB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:00<2:34:43, 91.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:00<1:47:25, 131kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:00<1:14:32, 187kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:00<51:53, 267kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 36.9M/862M [00:01<36:11, 380kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:01<25:06, 542kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:01<17:43, 762kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<12:46, 1.05MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<10:33:18, 21.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<7:22:22, 30.3kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<5:46:27, 38.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<8:56:54, 25.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:15:32, 35.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<9:39:30, 23.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<6:45:17, 32.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<9:57:52, 22.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<6:57:59, 31.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:33:38, 21.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:23:00, 30.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:38:58, 20.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:26:41, 29.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:48:02, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:33:05, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:35:44, 20.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:24:25, 29.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:44:28, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:30:34, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:36:24, 20.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:24:54, 29.4kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:39:53, 20.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:27:20, 29.2kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:36:49, 20.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:25:15, 29.3kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:20:08, 21.0kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:13:30, 30.0kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:30:50, 20.6kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:20:58, 29.4kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:36:08, 20.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:24:48, 29.1kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:10:13, 21.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:06:34, 30.2kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:25:44, 20.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:17:29, 29.4kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:08:27, 21.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:05:19, 30.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:25:36, 20.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:17:18, 29.2kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:29:28, 20.3kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:20:03, 29.0kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:17:27, 20.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:11:35, 29.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:27:29, 20.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:18:39, 28.9kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:15:52, 20.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:10:29, 29.4kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:22:49, 20.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:15:23, 29.0kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:10:53, 20.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:06:59, 29.4kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:20:48, 20.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:14:01, 28.9kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<9:52:11, 21.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<6:53:55, 30.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:08:48, 20.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<7:05:38, 29.3kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<9:44:37, 21.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<6:48:37, 30.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:05:31, 20.5kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:03:13, 29.3kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:08:28, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<7:05:16, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:16:17, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<7:10:44, 28.6kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:08:30, 20.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<7:05:17, 28.9kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:11:34, 20.1kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<7:07:26, 28.7kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<10:09:47, 20.1kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<7:06:12, 28.7kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:00:17, 20.4kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<6:59:37, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<9:41:10, 21.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<6:46:11, 29.9kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<9:55:18, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:56:03, 29.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:56:34, 20.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:56:55, 29.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<10:01:32, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<7:00:24, 28.7kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:58:33, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:58:25, 28.7kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:29:50, 21.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:38:15, 30.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:48:17, 20.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:51:08, 29.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:49:25, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:51:56, 28.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:43:54, 20.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:48:03, 29.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:48:57, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:51:40, 28.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:24:21, 21.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:34:24, 29.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:37:49, 20.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:43:48, 29.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:40:04, 20.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:45:24, 28.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:33:04, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<6:40:33, 29.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:16:44, 21.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:29:03, 30.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:30:28, 20.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:38:39, 29.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:33:17, 20.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:40:38, 28.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:27:21, 20.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:36:32, 29.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:14:05, 20.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:27:11, 29.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:24:37, 20.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:34:34, 29.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:22:19, 20.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:33:01, 29.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:05:34, 21.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:21:18, 29.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<8:59:53, 21.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:17:16, 30.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:13:34, 20.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:26:49, 29.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:17:16, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:29:25, 29.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:13:04, 20.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:26:27, 29.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:18:31, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:30:15, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<9:16:34, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:28:54, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:10:19, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:24:30, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<9:16:13, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:28:39, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:09:56, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:24:14, 28.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:12:46, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:26:17, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<8:51:18, 20.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:11:13, 29.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:01:02, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:18:03, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<8:57:00, 20.5kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:15:11, 29.2kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<9:02:36, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:19:06, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<9:00:53, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:17:56, 28.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:54:10, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:13:12, 29.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:59:05, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:16:39, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:53:48, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:12:55, 28.9kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:57:43, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<6:15:40, 28.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:55:56, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:14:26, 28.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:46:41, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<6:07:57, 29.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:49:29, 20.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:09:54, 28.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:50:40, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:10:44, 28.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:42:18, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<6:04:57, 29.0kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:26:53, 20.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<5:54:10, 29.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:20:15, 21.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<5:49:29, 30.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:32:33, 20.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<5:58:08, 29.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:14:49, 21.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<5:45:44, 30.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:10:40, 21.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<5:42:46, 30.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:26:05, 20.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:53:36, 29.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:11:58, 21.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:43:40, 30.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:23:43, 20.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:51:52, 29.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:25:26, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:53:03, 29.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:28:31, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:55:12, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:26:24, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:53:43, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:26:35, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:53:55, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:04:43, 21.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:38:35, 29.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:13:50, 20.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:45:01, 29.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<7:56:46, 21.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:33:02, 30.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:05:17, 20.7kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:38:58, 29.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:12:50, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:44:14, 29.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<8:11:05, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:43:00, 29.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:12:52, 20.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:44:19, 28.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<7:49:14, 21.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:27:45, 30.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<7:58:56, 20.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:34:31, 29.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<7:59:06, 20.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:34:37, 29.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<8:03:14, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:37:31, 29.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<7:56:37, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:32:52, 29.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<8:00:47, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:35:47, 28.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:55:56, 20.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:32:22, 29.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<8:00:01, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:35:19, 28.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:34:12, 21.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:17:12, 30.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:46:25, 20.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:25:48, 29.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:33:16, 21.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:16:33, 30.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:45:55, 20.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:25:24, 29.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:38:26, 20.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:20:08, 29.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:46:53, 20.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:26:07, 28.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:24:43, 21.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:10:33, 30.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:37:57, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:19:52, 29.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:19:12, 21.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:06:43, 30.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:29:48, 20.7kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:14:07, 29.6kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:30:04, 20.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:14:16, 29.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:34:47, 20.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:17:33, 29.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:37:29, 20.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:19:26, 28.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:35:01, 20.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<5:18:12, 28.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<3:42:59, 41.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<6:05:09, 25.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<4:15:02, 35.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<6:51:39, 22.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<4:47:31, 31.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<6:54:03, 21.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<4:49:11, 31.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<6:56:16, 21.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<4:50:39, 31.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:10:41, 20.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:00:43, 29.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:18:05, 20.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<5:05:56, 29.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:01:42, 21.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<4:54:11, 30.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<3:26:36, 42.9kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<2:24:35, 61.1kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<4:59:36, 29.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<3:29:21, 41.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<2:26:55, 59.6kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<1:43:01, 84.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<4:32:02, 32.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<3:10:10, 45.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<2:13:43, 64.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<1:33:44, 92.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<4:20:18, 33.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<3:01:51, 47.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<5:46:14, 24.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:01:49, 35.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<6:11:20, 23.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<4:19:04, 32.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<3:02:04, 46.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<2:07:26, 66.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<4:36:18, 30.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<3:13:02, 43.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<2:15:37, 61.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<1:35:03, 87.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<4:14:54, 32.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<2:58:09, 46.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<2:05:53, 65.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<1:27:58, 93.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<1:04:08, 128kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<3:22:45, 40.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<2:21:55, 57.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<1:39:57, 81.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<1:10:09, 116kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<3:49:38, 35.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<2:40:32, 50.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<1:52:58, 71.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<1:19:12, 101kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<3:56:06, 34.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<2:44:59, 48.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<1:56:08, 68.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<1:21:24, 97.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<3:54:14, 33.8kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<2:43:40, 48.0kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<1:55:09, 68.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<1:20:42, 96.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<3:52:51, 33.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<2:42:40, 47.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<1:54:27, 67.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<1:20:13, 96.2kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<3:47:23, 33.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<2:38:51, 48.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<1:52:00, 68.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<1:18:27, 97.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<3:46:11, 33.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<2:37:59, 47.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<1:51:06, 67.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<1:17:52, 96.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<3:42:12, 33.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<2:35:11, 47.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<1:49:31, 67.9kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<1:16:42, 96.5kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<3:38:55, 33.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<2:32:52, 48.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<1:47:55, 67.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<1:15:35, 96.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<3:35:57, 33.8kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<2:30:46, 47.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<1:46:09, 68.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<1:14:21, 96.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<3:35:38, 33.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:02<2:29:49, 47.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:02<1:45:24, 67.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<8:18:15, 14.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<5:48:13, 20.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<4:03:10, 29.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<2:50:27, 41.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<1:59:15, 58.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<3:58:09, 29.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<2:46:04, 41.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<1:56:36, 59.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<1:21:40, 84.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<3:32:19, 32.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<2:28:08, 46.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<1:44:10, 65.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<1:12:59, 93.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<3:18:50, 34.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:13<2:17:55, 48.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<8:29:32, 13.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<5:56:41, 18.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<4:08:26, 26.8kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<2:53:46, 38.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<2:01:42, 54.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<3:56:58, 28.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<2:45:06, 39.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<1:56:15, 56.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<1:21:18, 80.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<3:23:12, 32.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<2:21:40, 45.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<1:39:32, 64.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<1:09:41, 92.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<3:16:45, 32.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<2:17:10, 46.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<1:36:54, 65.5kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<1:07:48, 93.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<3:06:46, 33.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<2:10:13, 48.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<1:31:34, 68.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<1:04:07, 96.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<3:03:39, 33.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<2:08:00, 48.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<1:29:43, 68.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<1:02:54, 97.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<3:07:34, 32.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<2:10:41, 46.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<1:31:48, 65.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<1:04:20, 93.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<2:46:33, 36.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<1:56:06, 51.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<1:21:49, 72.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<57:16, 103kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<2:54:17, 33.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<2:01:25, 48.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<1:25:23, 68.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<59:46, 96.9kB/s]  .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<2:51:57, 33.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<1:59:45, 47.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<1:24:11, 67.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<58:56, 96.5kB/s]  .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<2:49:34, 33.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<1:58:04, 47.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<1:22:54, 67.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<58:02, 96.2kB/s]  .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<2:49:47, 32.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<1:58:10, 46.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<1:23:23, 66.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<58:18, 93.9kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<2:43:38, 33.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:53:53, 47.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:20:25, 67.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<56:13, 95.5kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<2:39:36, 33.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<1:51:03, 47.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<1:18:04, 67.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<54:40, 96.4kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<2:28:22, 35.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<1:43:15, 50.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<1:12:35, 71.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<50:48, 102kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<2:32:46, 33.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<1:46:15, 47.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<1:14:41, 68.1kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<52:14, 96.8kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<2:31:11, 33.5kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<1:45:06, 47.5kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<1:13:52, 67.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<51:40, 95.8kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<2:26:24, 33.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<1:41:46, 48.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<1:11:45, 68.0kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:08<50:09, 96.7kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<2:22:32, 34.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<1:39:03, 48.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<1:09:37, 68.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<48:41, 97.4kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<2:19:48, 33.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<1:37:07, 48.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<1:08:14, 68.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<47:43, 97.2kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<2:18:28, 33.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<1:36:09, 47.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<1:07:32, 67.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<47:13, 96.0kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<2:13:40, 33.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:32:47, 48.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<1:05:15, 68.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<45:36, 97.1kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<2:10:55, 33.8kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<1:30:50, 48.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<1:03:58, 68.1kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<44:41, 96.8kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<2:07:41, 33.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:28:34, 48.0kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:02:15, 68.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<43:31, 96.9kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<2:02:58, 34.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<1:25:16, 48.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<1:00:13, 68.9kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<42:02, 97.9kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<2:00:28, 34.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<1:23:30, 48.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<58:38, 68.9kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<40:58, 97.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<1:58:12, 33.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:21:53, 48.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<57:32, 68.4kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<40:11, 97.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<1:55:57, 33.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<1:20:17, 47.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<56:26, 67.8kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<39:24, 96.4kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<1:52:04, 33.9kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<1:17:34, 48.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<54:31, 68.3kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<38:03, 97.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<1:49:24, 33.8kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<1:15:41, 47.9kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<53:06, 68.1kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<37:04, 96.8kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<1:47:10, 33.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<1:14:05, 47.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<52:04, 67.5kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<36:19, 95.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<1:43:10, 33.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<1:11:17, 47.9kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<50:02, 68.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<34:55, 96.8kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<1:40:35, 33.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<1:09:27, 47.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<48:46, 67.7kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<34:01, 96.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<1:36:57, 33.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:06:54, 47.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<47:07, 68.0kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<32:49, 96.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<1:33:54, 33.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<1:04:45, 47.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<45:29, 68.0kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<31:42, 96.7kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<1:30:43, 33.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:02:30, 47.9kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<43:53, 68.1kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<30:35, 96.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<1:27:50, 33.7kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:03<1:00:07, 48.1kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<3:33:55, 13.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<2:28:57, 19.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<1:42:54, 27.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<1:11:53, 39.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<50:02, 55.7kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:40:09, 27.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<1:08:46, 39.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<48:25, 56.0kB/s]  .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<33:37, 79.7kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:23:49, 32.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<57:32, 45.4kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<40:22, 64.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<28:05, 91.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<1:16:45, 33.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<52:39, 47.6kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<36:57, 67.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<25:42, 96.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<1:13:27, 33.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<50:20, 47.7kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<35:09, 68.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<24:30, 96.5kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<1:12:43, 32.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<49:45, 46.2kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<34:55, 65.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<24:14, 93.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:08:06, 33.2kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<46:32, 47.1kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<32:39, 67.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<22:40, 95.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<1:00:48, 35.5kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<41:30, 50.3kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<29:08, 71.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<20:13, 101kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<1:00:24, 34.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<41:09, 48.2kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:31<28:52, 68.5kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<20:00, 97.3kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<57:49, 33.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<39:19, 47.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<27:34, 67.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<19:05, 96.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<54:50, 33.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<37:13, 47.7kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<26:04, 67.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<18:02, 96.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<51:42, 33.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<35:00, 47.7kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<24:32, 67.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<16:57, 96.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<48:29, 33.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<32:44, 47.8kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<23:00, 67.8kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<15:51, 96.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<44:59, 34.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<30:17, 48.2kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<21:14, 68.4kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<14:37, 97.2kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<42:12, 33.7kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<28:18, 47.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<19:57, 67.7kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<13:41, 96.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<39:00, 33.8kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<26:03, 47.9kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<18:13, 68.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<12:32, 96.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<34:01, 35.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<22:37, 50.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<15:50, 71.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<10:51, 102kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<32:27, 34.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<21:27, 48.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<15:03, 68.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<10:16, 97.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<29:32, 34.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<19:23, 48.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<13:35, 68.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<09:14, 97.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<26:38, 33.7kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<17:20, 47.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<12:06, 68.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<08:12, 96.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<23:55, 33.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<15:24, 47.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<10:50, 66.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<07:17, 94.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<19:04, 36.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<12:06, 51.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<08:27, 72.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<05:39, 103kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<16:54, 34.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<10:30, 49.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<07:18, 69.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<04:50, 98.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<14:04, 34.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<08:29, 48.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<05:56, 68.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<03:51, 97.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<11:03, 33.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<06:22, 48.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<04:24, 68.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<02:47, 96.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<07:26, 36.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<03:54, 51.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<02:40, 72.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<01:35, 103kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<04:48, 34.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<01:58, 48.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<01:19, 68.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:36, 97.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:45, 33.9kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 846/400000 [00:00<00:47, 8456.66it/s]  0%|          | 1707/400000 [00:00<00:46, 8501.29it/s]  1%|          | 2628/400000 [00:00<00:45, 8699.47it/s]  1%|          | 3553/400000 [00:00<00:44, 8857.30it/s]  1%|          | 4495/400000 [00:00<00:43, 9016.23it/s]  1%|▏         | 5399/400000 [00:00<00:43, 9020.11it/s]  2%|▏         | 6303/400000 [00:00<00:43, 9025.41it/s]  2%|▏         | 7235/400000 [00:00<00:43, 9110.69it/s]  2%|▏         | 8148/400000 [00:00<00:42, 9113.71it/s]  2%|▏         | 9029/400000 [00:01<00:43, 9020.37it/s]  2%|▏         | 9915/400000 [00:01<00:43, 8970.65it/s]  3%|▎         | 10796/400000 [00:01<00:43, 8851.93it/s]  3%|▎         | 11705/400000 [00:01<00:43, 8921.12it/s]  3%|▎         | 12597/400000 [00:01<00:43, 8917.98it/s]  3%|▎         | 13505/400000 [00:01<00:43, 8963.51it/s]  4%|▎         | 14398/400000 [00:01<00:43, 8860.64it/s]  4%|▍         | 15282/400000 [00:01<00:43, 8794.73it/s]  4%|▍         | 16178/400000 [00:01<00:43, 8841.84it/s]  4%|▍         | 17062/400000 [00:01<00:43, 8833.00it/s]  4%|▍         | 17961/400000 [00:02<00:43, 8878.98it/s]  5%|▍         | 18849/400000 [00:02<00:43, 8852.67it/s]  5%|▍         | 19734/400000 [00:02<00:44, 8620.00it/s]  5%|▌         | 20601/400000 [00:02<00:43, 8633.02it/s]  5%|▌         | 21475/400000 [00:02<00:43, 8662.25it/s]  6%|▌         | 22355/400000 [00:02<00:43, 8700.76it/s]  6%|▌         | 23248/400000 [00:02<00:42, 8765.74it/s]  6%|▌         | 24129/400000 [00:02<00:42, 8776.72it/s]  6%|▋         | 25042/400000 [00:02<00:42, 8877.48it/s]  6%|▋         | 25944/400000 [00:02<00:41, 8919.01it/s]  7%|▋         | 26855/400000 [00:03<00:41, 8972.93it/s]  7%|▋         | 27753/400000 [00:03<00:41, 8973.16it/s]  7%|▋         | 28651/400000 [00:03<00:41, 8948.77it/s]  7%|▋         | 29547/400000 [00:03<00:41, 8946.83it/s]  8%|▊         | 30449/400000 [00:03<00:41, 8966.44it/s]  8%|▊         | 31365/400000 [00:03<00:40, 9023.32it/s]  8%|▊         | 32286/400000 [00:03<00:40, 9077.59it/s]  8%|▊         | 33194/400000 [00:03<00:40, 8997.65it/s]  9%|▊         | 34095/400000 [00:03<00:40, 8937.04it/s]  9%|▊         | 34990/400000 [00:03<00:40, 8926.36it/s]  9%|▉         | 35883/400000 [00:04<00:41, 8865.68it/s]  9%|▉         | 36770/400000 [00:04<00:41, 8798.49it/s]  9%|▉         | 37651/400000 [00:04<00:41, 8706.36it/s] 10%|▉         | 38537/400000 [00:04<00:41, 8750.00it/s] 10%|▉         | 39413/400000 [00:04<00:41, 8741.08it/s] 10%|█         | 40315/400000 [00:04<00:40, 8820.49it/s] 10%|█         | 41198/400000 [00:04<00:40, 8816.15it/s] 11%|█         | 42091/400000 [00:04<00:40, 8847.48it/s] 11%|█         | 43023/400000 [00:04<00:39, 8984.02it/s] 11%|█         | 43923/400000 [00:04<00:39, 8986.44it/s] 11%|█         | 44823/400000 [00:05<00:39, 8925.97it/s] 11%|█▏        | 45717/400000 [00:05<00:39, 8910.98it/s] 12%|█▏        | 46609/400000 [00:05<00:39, 8907.42it/s] 12%|█▏        | 47512/400000 [00:05<00:39, 8941.49it/s] 12%|█▏        | 48432/400000 [00:05<00:38, 9016.32it/s] 12%|█▏        | 49340/400000 [00:05<00:38, 9034.02it/s] 13%|█▎        | 50265/400000 [00:05<00:38, 9096.25it/s] 13%|█▎        | 51175/400000 [00:05<00:38, 8995.53it/s] 13%|█▎        | 52075/400000 [00:05<00:38, 8954.22it/s] 13%|█▎        | 52971/400000 [00:05<00:38, 8945.26it/s] 13%|█▎        | 53866/400000 [00:06<00:39, 8824.58it/s] 14%|█▎        | 54779/400000 [00:06<00:38, 8912.39it/s] 14%|█▍        | 55674/400000 [00:06<00:38, 8921.71it/s] 14%|█▍        | 56567/400000 [00:06<00:38, 8904.73it/s] 14%|█▍        | 57475/400000 [00:06<00:38, 8956.55it/s] 15%|█▍        | 58379/400000 [00:06<00:38, 8981.09it/s] 15%|█▍        | 59278/400000 [00:06<00:38, 8897.13it/s] 15%|█▌        | 60182/400000 [00:06<00:38, 8937.86it/s] 15%|█▌        | 61077/400000 [00:06<00:38, 8699.01it/s] 15%|█▌        | 61949/400000 [00:06<00:38, 8680.82it/s] 16%|█▌        | 62854/400000 [00:07<00:38, 8787.50it/s] 16%|█▌        | 63782/400000 [00:07<00:37, 8929.54it/s] 16%|█▌        | 64692/400000 [00:07<00:37, 8980.00it/s] 16%|█▋        | 65607/400000 [00:07<00:37, 9029.85it/s] 17%|█▋        | 66511/400000 [00:07<00:37, 9000.30it/s] 17%|█▋        | 67413/400000 [00:07<00:36, 9003.67it/s] 17%|█▋        | 68314/400000 [00:07<00:37, 8946.95it/s] 17%|█▋        | 69210/400000 [00:07<00:37, 8925.48it/s] 18%|█▊        | 70104/400000 [00:07<00:36, 8929.60it/s] 18%|█▊        | 71026/400000 [00:07<00:36, 9014.31it/s] 18%|█▊        | 71951/400000 [00:08<00:36, 9081.00it/s] 18%|█▊        | 72860/400000 [00:08<00:36, 8977.45it/s] 18%|█▊        | 73759/400000 [00:08<00:36, 8954.72it/s] 19%|█▊        | 74663/400000 [00:08<00:36, 8979.81it/s] 19%|█▉        | 75562/400000 [00:08<00:36, 8971.59it/s] 19%|█▉        | 76460/400000 [00:08<00:36, 8939.39it/s] 19%|█▉        | 77384/400000 [00:08<00:35, 9027.14it/s] 20%|█▉        | 78298/400000 [00:08<00:35, 9057.64it/s] 20%|█▉        | 79205/400000 [00:08<00:35, 9011.52it/s] 20%|██        | 80107/400000 [00:08<00:35, 8983.92it/s] 20%|██        | 81031/400000 [00:09<00:35, 9057.28it/s] 20%|██        | 81937/400000 [00:09<00:35, 9019.06it/s] 21%|██        | 82841/400000 [00:09<00:35, 9025.16it/s] 21%|██        | 83751/400000 [00:09<00:34, 9046.09it/s] 21%|██        | 84656/400000 [00:09<00:34, 9040.23it/s] 21%|██▏       | 85600/400000 [00:09<00:34, 9154.21it/s] 22%|██▏       | 86521/400000 [00:09<00:34, 9169.09it/s] 22%|██▏       | 87439/400000 [00:09<00:34, 9095.03it/s] 22%|██▏       | 88349/400000 [00:09<00:35, 8881.91it/s] 22%|██▏       | 89268/400000 [00:09<00:34, 8972.06it/s] 23%|██▎       | 90167/400000 [00:10<00:34, 8968.39it/s] 23%|██▎       | 91065/400000 [00:10<00:34, 8961.73it/s] 23%|██▎       | 91962/400000 [00:10<00:34, 8930.11it/s] 23%|██▎       | 92860/400000 [00:10<00:34, 8942.71it/s] 23%|██▎       | 93776/400000 [00:10<00:34, 9005.52it/s] 24%|██▎       | 94683/400000 [00:10<00:33, 9022.19it/s] 24%|██▍       | 95614/400000 [00:10<00:33, 9106.64it/s] 24%|██▍       | 96527/400000 [00:10<00:33, 9113.26it/s] 24%|██▍       | 97439/400000 [00:10<00:34, 8864.71it/s] 25%|██▍       | 98328/400000 [00:11<00:34, 8772.18it/s] 25%|██▍       | 99207/400000 [00:11<00:34, 8709.09it/s] 25%|██▌       | 100107/400000 [00:11<00:34, 8792.34it/s] 25%|██▌       | 100988/400000 [00:11<00:34, 8687.87it/s] 25%|██▌       | 101877/400000 [00:11<00:34, 8747.10it/s] 26%|██▌       | 102777/400000 [00:11<00:33, 8819.84it/s] 26%|██▌       | 103688/400000 [00:11<00:33, 8902.52it/s] 26%|██▌       | 104591/400000 [00:11<00:33, 8937.49it/s] 26%|██▋       | 105486/400000 [00:11<00:32, 8930.33it/s] 27%|██▋       | 106380/400000 [00:11<00:33, 8708.87it/s] 27%|██▋       | 107253/400000 [00:12<00:33, 8695.07it/s] 27%|██▋       | 108129/400000 [00:12<00:33, 8711.77it/s] 27%|██▋       | 109001/400000 [00:12<00:34, 8548.17it/s] 27%|██▋       | 109858/400000 [00:12<00:33, 8550.27it/s] 28%|██▊       | 110764/400000 [00:12<00:33, 8694.65it/s] 28%|██▊       | 111679/400000 [00:12<00:32, 8823.99it/s] 28%|██▊       | 112573/400000 [00:12<00:32, 8857.56it/s] 28%|██▊       | 113468/400000 [00:12<00:32, 8882.48it/s] 29%|██▊       | 114371/400000 [00:12<00:32, 8923.84it/s] 29%|██▉       | 115270/400000 [00:12<00:31, 8942.34it/s] 29%|██▉       | 116165/400000 [00:13<00:32, 8866.88it/s] 29%|██▉       | 117068/400000 [00:13<00:31, 8912.88it/s] 29%|██▉       | 117960/400000 [00:13<00:31, 8906.40it/s] 30%|██▉       | 118851/400000 [00:13<00:31, 8880.70it/s] 30%|██▉       | 119740/400000 [00:13<00:32, 8748.46it/s] 30%|███       | 120622/400000 [00:13<00:31, 8769.26it/s] 30%|███       | 121500/400000 [00:13<00:31, 8764.38it/s] 31%|███       | 122408/400000 [00:13<00:31, 8855.67it/s] 31%|███       | 123294/400000 [00:13<00:31, 8745.12it/s] 31%|███       | 124183/400000 [00:13<00:31, 8785.55it/s] 31%|███▏      | 125063/400000 [00:14<00:31, 8755.97it/s] 31%|███▏      | 125939/400000 [00:14<00:31, 8751.21it/s] 32%|███▏      | 126842/400000 [00:14<00:30, 8831.04it/s] 32%|███▏      | 127761/400000 [00:14<00:30, 8933.01it/s] 32%|███▏      | 128656/400000 [00:14<00:30, 8937.38it/s] 32%|███▏      | 129551/400000 [00:14<00:30, 8934.97it/s] 33%|███▎      | 130445/400000 [00:14<00:30, 8901.10it/s] 33%|███▎      | 131350/400000 [00:14<00:30, 8945.16it/s] 33%|███▎      | 132249/400000 [00:14<00:29, 8956.40it/s] 33%|███▎      | 133170/400000 [00:14<00:29, 9029.32it/s] 34%|███▎      | 134074/400000 [00:15<00:29, 8918.11it/s] 34%|███▎      | 134967/400000 [00:15<00:29, 8878.42it/s] 34%|███▍      | 135870/400000 [00:15<00:29, 8921.60it/s] 34%|███▍      | 136763/400000 [00:15<00:29, 8891.96it/s] 34%|███▍      | 137653/400000 [00:15<00:30, 8687.02it/s] 35%|███▍      | 138532/400000 [00:15<00:29, 8717.65it/s] 35%|███▍      | 139405/400000 [00:15<00:30, 8586.42it/s] 35%|███▌      | 140323/400000 [00:15<00:29, 8755.26it/s] 35%|███▌      | 141217/400000 [00:15<00:29, 8808.24it/s] 36%|███▌      | 142100/400000 [00:15<00:29, 8814.22it/s] 36%|███▌      | 142988/400000 [00:16<00:29, 8832.51it/s] 36%|███▌      | 143872/400000 [00:16<00:29, 8796.06it/s] 36%|███▌      | 144805/400000 [00:16<00:28, 8948.69it/s] 36%|███▋      | 145701/400000 [00:16<00:28, 8929.47it/s] 37%|███▋      | 146601/400000 [00:16<00:28, 8948.66it/s] 37%|███▋      | 147497/400000 [00:16<00:29, 8697.76it/s] 37%|███▋      | 148369/400000 [00:16<00:29, 8658.92it/s] 37%|███▋      | 149242/400000 [00:16<00:28, 8677.94it/s] 38%|███▊      | 150111/400000 [00:16<00:29, 8459.98it/s] 38%|███▊      | 150959/400000 [00:16<00:29, 8386.80it/s] 38%|███▊      | 151805/400000 [00:17<00:29, 8406.28it/s] 38%|███▊      | 152647/400000 [00:17<00:31, 7772.90it/s] 38%|███▊      | 153501/400000 [00:17<00:30, 7986.49it/s] 39%|███▊      | 154357/400000 [00:17<00:30, 8148.67it/s] 39%|███▉      | 155250/400000 [00:17<00:29, 8366.21it/s] 39%|███▉      | 156136/400000 [00:17<00:28, 8507.20it/s] 39%|███▉      | 157043/400000 [00:17<00:28, 8666.77it/s] 39%|███▉      | 157962/400000 [00:17<00:27, 8815.18it/s] 40%|███▉      | 158861/400000 [00:17<00:27, 8866.19it/s] 40%|███▉      | 159762/400000 [00:18<00:26, 8908.72it/s] 40%|████      | 160655/400000 [00:18<00:27, 8828.93it/s] 40%|████      | 161540/400000 [00:18<00:27, 8800.25it/s] 41%|████      | 162423/400000 [00:18<00:26, 8807.02it/s] 41%|████      | 163331/400000 [00:18<00:26, 8885.95it/s] 41%|████      | 164221/400000 [00:18<00:26, 8865.93it/s] 41%|████▏     | 165118/400000 [00:18<00:26, 8895.17it/s] 42%|████▏     | 166008/400000 [00:18<00:26, 8870.87it/s] 42%|████▏     | 166909/400000 [00:18<00:26, 8910.44it/s] 42%|████▏     | 167803/400000 [00:18<00:26, 8916.10it/s] 42%|████▏     | 168722/400000 [00:19<00:25, 8995.80it/s] 42%|████▏     | 169629/400000 [00:19<00:25, 9017.05it/s] 43%|████▎     | 170540/400000 [00:19<00:25, 9042.47it/s] 43%|████▎     | 171445/400000 [00:19<00:25, 9043.86it/s] 43%|████▎     | 172350/400000 [00:19<00:25, 9025.78it/s] 43%|████▎     | 173265/400000 [00:19<00:25, 9062.45it/s] 44%|████▎     | 174177/400000 [00:19<00:24, 9077.89it/s] 44%|████▍     | 175085/400000 [00:19<00:24, 9027.55it/s] 44%|████▍     | 175988/400000 [00:19<00:25, 8846.39it/s] 44%|████▍     | 176900/400000 [00:19<00:24, 8926.18it/s] 44%|████▍     | 177794/400000 [00:20<00:25, 8869.66it/s] 45%|████▍     | 178707/400000 [00:20<00:24, 8944.91it/s] 45%|████▍     | 179623/400000 [00:20<00:24, 9007.61it/s] 45%|████▌     | 180552/400000 [00:20<00:24, 9089.38it/s] 45%|████▌     | 181462/400000 [00:20<00:24, 9063.08it/s] 46%|████▌     | 182369/400000 [00:20<00:24, 8947.06it/s] 46%|████▌     | 183317/400000 [00:20<00:23, 9097.75it/s] 46%|████▌     | 184280/400000 [00:20<00:23, 9250.55it/s] 46%|████▋     | 185214/400000 [00:20<00:23, 9276.98it/s] 47%|████▋     | 186143/400000 [00:20<00:23, 9163.37it/s] 47%|████▋     | 187061/400000 [00:21<00:23, 9133.82it/s] 47%|████▋     | 187976/400000 [00:21<00:23, 9092.72it/s] 47%|████▋     | 188906/400000 [00:21<00:23, 9153.81it/s] 47%|████▋     | 189822/400000 [00:21<00:23, 9137.80it/s] 48%|████▊     | 190737/400000 [00:21<00:22, 9104.26it/s] 48%|████▊     | 191662/400000 [00:21<00:22, 9146.49it/s] 48%|████▊     | 192629/400000 [00:21<00:22, 9295.22it/s] 48%|████▊     | 193580/400000 [00:21<00:22, 9357.72it/s] 49%|████▊     | 194517/400000 [00:21<00:22, 9328.65it/s] 49%|████▉     | 195451/400000 [00:21<00:22, 9172.71it/s] 49%|████▉     | 196370/400000 [00:22<00:22, 9103.83it/s] 49%|████▉     | 197282/400000 [00:22<00:22, 8951.35it/s] 50%|████▉     | 198179/400000 [00:22<00:22, 8870.41it/s] 50%|████▉     | 199067/400000 [00:22<00:22, 8831.02it/s] 50%|████▉     | 199951/400000 [00:22<00:22, 8756.87it/s] 50%|█████     | 200828/400000 [00:22<00:22, 8754.52it/s] 50%|█████     | 201704/400000 [00:22<00:22, 8755.62it/s] 51%|█████     | 202580/400000 [00:22<00:22, 8725.20it/s] 51%|█████     | 203461/400000 [00:22<00:22, 8748.59it/s] 51%|█████     | 204346/400000 [00:22<00:22, 8775.86it/s] 51%|█████▏    | 205224/400000 [00:23<00:22, 8632.65it/s] 52%|█████▏    | 206134/400000 [00:23<00:22, 8767.45it/s] 52%|█████▏    | 207059/400000 [00:23<00:21, 8904.89it/s] 52%|█████▏    | 207967/400000 [00:23<00:21, 8954.87it/s] 52%|█████▏    | 208874/400000 [00:23<00:21, 8986.59it/s] 52%|█████▏    | 209774/400000 [00:23<00:21, 8933.08it/s] 53%|█████▎    | 210668/400000 [00:23<00:21, 8765.34it/s] 53%|█████▎    | 211548/400000 [00:23<00:21, 8773.64it/s] 53%|█████▎    | 212427/400000 [00:23<00:21, 8767.78it/s] 53%|█████▎    | 213306/400000 [00:23<00:21, 8772.99it/s] 54%|█████▎    | 214184/400000 [00:24<00:21, 8774.44it/s] 54%|█████▍    | 215062/400000 [00:24<00:21, 8500.35it/s] 54%|█████▍    | 215966/400000 [00:24<00:21, 8653.65it/s] 54%|█████▍    | 216876/400000 [00:24<00:20, 8781.09it/s] 54%|█████▍    | 217757/400000 [00:24<00:20, 8703.13it/s] 55%|█████▍    | 218655/400000 [00:24<00:20, 8783.44it/s] 55%|█████▍    | 219538/400000 [00:24<00:20, 8797.25it/s] 55%|█████▌    | 220419/400000 [00:24<00:20, 8750.47it/s] 55%|█████▌    | 221295/400000 [00:24<00:20, 8629.97it/s] 56%|█████▌    | 222159/400000 [00:25<00:21, 8441.85it/s] 56%|█████▌    | 223033/400000 [00:25<00:20, 8528.09it/s] 56%|█████▌    | 223917/400000 [00:25<00:20, 8616.97it/s] 56%|█████▌    | 224838/400000 [00:25<00:19, 8784.42it/s] 56%|█████▋    | 225753/400000 [00:25<00:19, 8887.82it/s] 57%|█████▋    | 226644/400000 [00:25<00:19, 8867.77it/s] 57%|█████▋    | 227532/400000 [00:25<00:19, 8846.58it/s] 57%|█████▋    | 228457/400000 [00:25<00:19, 8963.35it/s] 57%|█████▋    | 229379/400000 [00:25<00:18, 9036.48it/s] 58%|█████▊    | 230284/400000 [00:25<00:18, 8993.04it/s] 58%|█████▊    | 231189/400000 [00:26<00:18, 9008.05it/s] 58%|█████▊    | 232091/400000 [00:26<00:18, 8913.93it/s] 58%|█████▊    | 232983/400000 [00:26<00:19, 8706.68it/s] 58%|█████▊    | 233891/400000 [00:26<00:18, 8815.00it/s] 59%|█████▊    | 234774/400000 [00:26<00:18, 8784.25it/s] 59%|█████▉    | 235680/400000 [00:26<00:18, 8864.37it/s] 59%|█████▉    | 236568/400000 [00:26<00:18, 8675.25it/s] 59%|█████▉    | 237456/400000 [00:26<00:18, 8735.18it/s] 60%|█████▉    | 238340/400000 [00:26<00:18, 8765.22it/s] 60%|█████▉    | 239218/400000 [00:26<00:18, 8655.03it/s] 60%|██████    | 240088/400000 [00:27<00:18, 8665.94it/s] 60%|██████    | 240985/400000 [00:27<00:18, 8753.83it/s] 60%|██████    | 241862/400000 [00:27<00:18, 8715.16it/s] 61%|██████    | 242735/400000 [00:27<00:18, 8702.54it/s] 61%|██████    | 243606/400000 [00:27<00:17, 8703.04it/s] 61%|██████    | 244491/400000 [00:27<00:17, 8745.66it/s] 61%|██████▏   | 245374/400000 [00:27<00:17, 8768.35it/s] 62%|██████▏   | 246267/400000 [00:27<00:17, 8815.54it/s] 62%|██████▏   | 247149/400000 [00:27<00:17, 8790.97it/s] 62%|██████▏   | 248029/400000 [00:27<00:17, 8692.41it/s] 62%|██████▏   | 248911/400000 [00:28<00:17, 8727.98it/s] 62%|██████▏   | 249785/400000 [00:28<00:17, 8664.45it/s] 63%|██████▎   | 250652/400000 [00:28<00:17, 8595.97it/s] 63%|██████▎   | 251533/400000 [00:28<00:17, 8658.42it/s] 63%|██████▎   | 252406/400000 [00:28<00:17, 8677.25it/s] 63%|██████▎   | 253307/400000 [00:28<00:16, 8773.79it/s] 64%|██████▎   | 254204/400000 [00:28<00:16, 8829.73it/s] 64%|██████▍   | 255117/400000 [00:28<00:16, 8915.28it/s] 64%|██████▍   | 256009/400000 [00:28<00:16, 8859.73it/s] 64%|██████▍   | 256896/400000 [00:28<00:16, 8552.48it/s] 64%|██████▍   | 257754/400000 [00:29<00:16, 8434.36it/s] 65%|██████▍   | 258655/400000 [00:29<00:16, 8597.36it/s] 65%|██████▍   | 259546/400000 [00:29<00:16, 8688.65it/s] 65%|██████▌   | 260441/400000 [00:29<00:15, 8763.37it/s] 65%|██████▌   | 261325/400000 [00:29<00:15, 8784.24it/s] 66%|██████▌   | 262206/400000 [00:29<00:15, 8791.14it/s] 66%|██████▌   | 263091/400000 [00:29<00:15, 8808.08it/s] 66%|██████▌   | 263981/400000 [00:29<00:15, 8832.77it/s] 66%|██████▌   | 264865/400000 [00:29<00:15, 8808.02it/s] 66%|██████▋   | 265747/400000 [00:29<00:15, 8785.63it/s] 67%|██████▋   | 266640/400000 [00:30<00:15, 8826.78it/s] 67%|██████▋   | 267523/400000 [00:30<00:15, 8807.79it/s] 67%|██████▋   | 268404/400000 [00:30<00:14, 8785.64it/s] 67%|██████▋   | 269301/400000 [00:30<00:14, 8837.76it/s] 68%|██████▊   | 270212/400000 [00:30<00:14, 8915.69it/s] 68%|██████▊   | 271110/400000 [00:30<00:14, 8934.08it/s] 68%|██████▊   | 272027/400000 [00:30<00:14, 9001.30it/s] 68%|██████▊   | 272954/400000 [00:30<00:13, 9079.29it/s] 68%|██████▊   | 273888/400000 [00:30<00:13, 9155.30it/s] 69%|██████▊   | 274804/400000 [00:30<00:13, 9074.15it/s] 69%|██████▉   | 275721/400000 [00:31<00:13, 9101.02it/s] 69%|██████▉   | 276632/400000 [00:31<00:13, 9046.47it/s] 69%|██████▉   | 277555/400000 [00:31<00:13, 9098.82it/s] 70%|██████▉   | 278488/400000 [00:31<00:13, 9165.22it/s] 70%|██████▉   | 279405/400000 [00:31<00:13, 9132.87it/s] 70%|███████   | 280328/400000 [00:31<00:13, 9160.49it/s] 70%|███████   | 281268/400000 [00:31<00:12, 9230.08it/s] 71%|███████   | 282204/400000 [00:31<00:12, 9268.33it/s] 71%|███████   | 283132/400000 [00:31<00:12, 9035.33it/s] 71%|███████   | 284037/400000 [00:32<00:12, 9021.45it/s] 71%|███████   | 284962/400000 [00:32<00:12, 9086.77it/s] 71%|███████▏  | 285883/400000 [00:32<00:12, 9123.30it/s] 72%|███████▏  | 286796/400000 [00:32<00:12, 9045.52it/s] 72%|███████▏  | 287702/400000 [00:32<00:12, 9035.26it/s] 72%|███████▏  | 288606/400000 [00:32<00:12, 8965.23it/s] 72%|███████▏  | 289503/400000 [00:32<00:12, 8914.73it/s] 73%|███████▎  | 290439/400000 [00:32<00:12, 9042.63it/s] 73%|███████▎  | 291344/400000 [00:32<00:12, 9041.53it/s] 73%|███████▎  | 292283/400000 [00:32<00:11, 9141.64it/s] 73%|███████▎  | 293198/400000 [00:33<00:11, 9044.56it/s] 74%|███████▎  | 294107/400000 [00:33<00:11, 9055.38it/s] 74%|███████▍  | 295015/400000 [00:33<00:11, 9061.83it/s] 74%|███████▍  | 295932/400000 [00:33<00:11, 9091.99it/s] 74%|███████▍  | 296842/400000 [00:33<00:11, 9070.81it/s] 74%|███████▍  | 297753/400000 [00:33<00:11, 9081.90it/s] 75%|███████▍  | 298662/400000 [00:33<00:11, 9044.91it/s] 75%|███████▍  | 299579/400000 [00:33<00:11, 9081.62it/s] 75%|███████▌  | 300488/400000 [00:33<00:11, 9001.28it/s] 75%|███████▌  | 301389/400000 [00:33<00:11, 8950.97it/s] 76%|███████▌  | 302285/400000 [00:34<00:10, 8920.07it/s] 76%|███████▌  | 303178/400000 [00:34<00:10, 8880.09it/s] 76%|███████▌  | 304067/400000 [00:34<00:10, 8862.72it/s] 76%|███████▌  | 304954/400000 [00:34<00:10, 8831.11it/s] 76%|███████▋  | 305839/400000 [00:34<00:10, 8835.02it/s] 77%|███████▋  | 306723/400000 [00:34<00:10, 8713.24it/s] 77%|███████▋  | 307621/400000 [00:34<00:10, 8789.97it/s] 77%|███████▋  | 308522/400000 [00:34<00:10, 8853.80it/s] 77%|███████▋  | 309425/400000 [00:34<00:10, 8902.84it/s] 78%|███████▊  | 310316/400000 [00:34<00:10, 8901.84it/s] 78%|███████▊  | 311207/400000 [00:35<00:10, 8763.12it/s] 78%|███████▊  | 312105/400000 [00:35<00:09, 8826.43it/s] 78%|███████▊  | 313012/400000 [00:35<00:09, 8896.23it/s] 78%|███████▊  | 313903/400000 [00:35<00:09, 8767.82it/s] 79%|███████▊  | 314781/400000 [00:35<00:09, 8712.96it/s] 79%|███████▉  | 315656/400000 [00:35<00:09, 8722.32it/s] 79%|███████▉  | 316529/400000 [00:35<00:09, 8640.13it/s] 79%|███████▉  | 317412/400000 [00:35<00:09, 8693.88it/s] 80%|███████▉  | 318282/400000 [00:35<00:09, 8685.83it/s] 80%|███████▉  | 319184/400000 [00:35<00:09, 8782.75it/s] 80%|████████  | 320089/400000 [00:36<00:09, 8861.10it/s] 80%|████████  | 320976/400000 [00:36<00:08, 8856.39it/s] 80%|████████  | 321862/400000 [00:36<00:08, 8812.13it/s] 81%|████████  | 322744/400000 [00:36<00:08, 8772.54it/s] 81%|████████  | 323626/400000 [00:36<00:08, 8783.98it/s] 81%|████████  | 324529/400000 [00:36<00:08, 8853.61it/s] 81%|████████▏ | 325420/400000 [00:36<00:08, 8868.84it/s] 82%|████████▏ | 326331/400000 [00:36<00:08, 8937.98it/s] 82%|████████▏ | 327248/400000 [00:36<00:08, 9004.28it/s] 82%|████████▏ | 328149/400000 [00:36<00:08, 8934.74it/s] 82%|████████▏ | 329058/400000 [00:37<00:07, 8979.63it/s] 82%|████████▏ | 329960/400000 [00:37<00:07, 8988.88it/s] 83%|████████▎ | 330860/400000 [00:37<00:07, 8930.55it/s] 83%|████████▎ | 331757/400000 [00:37<00:07, 8939.76it/s] 83%|████████▎ | 332652/400000 [00:37<00:07, 8900.12it/s] 83%|████████▎ | 333543/400000 [00:37<00:07, 8833.38it/s] 84%|████████▎ | 334427/400000 [00:37<00:07, 8813.64it/s] 84%|████████▍ | 335349/400000 [00:37<00:07, 8931.02it/s] 84%|████████▍ | 336266/400000 [00:37<00:07, 9000.58it/s] 84%|████████▍ | 337178/400000 [00:37<00:06, 9033.13it/s] 85%|████████▍ | 338082/400000 [00:38<00:06, 9026.33it/s] 85%|████████▍ | 338985/400000 [00:38<00:06, 8969.11it/s] 85%|████████▍ | 339894/400000 [00:38<00:06, 9004.69it/s] 85%|████████▌ | 340798/400000 [00:38<00:06, 9013.74it/s] 85%|████████▌ | 341716/400000 [00:38<00:06, 9061.69it/s] 86%|████████▌ | 342626/400000 [00:38<00:06, 9072.91it/s] 86%|████████▌ | 343534/400000 [00:38<00:06, 9062.47it/s] 86%|████████▌ | 344453/400000 [00:38<00:06, 9099.57it/s] 86%|████████▋ | 345371/400000 [00:38<00:05, 9120.59it/s] 87%|████████▋ | 346284/400000 [00:38<00:05, 9107.95it/s] 87%|████████▋ | 347204/400000 [00:39<00:05, 9133.51it/s] 87%|████████▋ | 348118/400000 [00:39<00:05, 8994.04it/s] 87%|████████▋ | 349018/400000 [00:39<00:05, 8975.06it/s] 87%|████████▋ | 349935/400000 [00:39<00:05, 9030.90it/s] 88%|████████▊ | 350846/400000 [00:39<00:05, 9052.68it/s] 88%|████████▊ | 351767/400000 [00:39<00:05, 9099.22it/s] 88%|████████▊ | 352678/400000 [00:39<00:05, 9043.26it/s] 88%|████████▊ | 353589/400000 [00:39<00:05, 9062.76it/s] 89%|████████▊ | 354496/400000 [00:39<00:05, 9029.05it/s] 89%|████████▉ | 355400/400000 [00:39<00:05, 8777.53it/s] 89%|████████▉ | 356280/400000 [00:40<00:05, 8553.27it/s] 89%|████████▉ | 357168/400000 [00:40<00:04, 8648.16it/s] 90%|████████▉ | 358049/400000 [00:40<00:04, 8694.67it/s] 90%|████████▉ | 358934/400000 [00:40<00:04, 8740.29it/s] 90%|████████▉ | 359828/400000 [00:40<00:04, 8797.38it/s] 90%|█████████ | 360728/400000 [00:40<00:04, 8854.48it/s] 90%|█████████ | 361615/400000 [00:40<00:04, 8673.22it/s] 91%|█████████ | 362484/400000 [00:40<00:04, 8499.06it/s] 91%|█████████ | 363354/400000 [00:40<00:04, 8556.70it/s] 91%|█████████ | 364231/400000 [00:41<00:04, 8619.37it/s] 91%|█████████▏| 365094/400000 [00:41<00:04, 8586.70it/s] 91%|█████████▏| 365954/400000 [00:41<00:03, 8543.16it/s] 92%|█████████▏| 366821/400000 [00:41<00:03, 8580.67it/s] 92%|█████████▏| 367714/400000 [00:41<00:03, 8682.18it/s] 92%|█████████▏| 368601/400000 [00:41<00:03, 8736.28it/s] 92%|█████████▏| 369508/400000 [00:41<00:03, 8832.81it/s] 93%|█████████▎| 370413/400000 [00:41<00:03, 8895.60it/s] 93%|█████████▎| 371304/400000 [00:41<00:03, 8780.82it/s] 93%|█████████▎| 372183/400000 [00:41<00:03, 8704.00it/s] 93%|█████████▎| 373070/400000 [00:42<00:03, 8751.91it/s] 93%|█████████▎| 373964/400000 [00:42<00:02, 8805.36it/s] 94%|█████████▎| 374862/400000 [00:42<00:02, 8854.19it/s] 94%|█████████▍| 375763/400000 [00:42<00:02, 8900.13it/s] 94%|█████████▍| 376654/400000 [00:42<00:02, 8827.81it/s] 94%|█████████▍| 377542/400000 [00:42<00:02, 8842.96it/s] 95%|█████████▍| 378427/400000 [00:42<00:02, 8686.48it/s] 95%|█████████▍| 379301/400000 [00:42<00:02, 8700.13it/s] 95%|█████████▌| 380179/400000 [00:42<00:02, 8721.42it/s] 95%|█████████▌| 381060/400000 [00:42<00:02, 8745.84it/s] 95%|█████████▌| 381935/400000 [00:43<00:02, 8578.45it/s] 96%|█████████▌| 382830/400000 [00:43<00:01, 8683.76it/s] 96%|█████████▌| 383711/400000 [00:43<00:01, 8720.86it/s] 96%|█████████▌| 384596/400000 [00:43<00:01, 8758.72it/s] 96%|█████████▋| 385475/400000 [00:43<00:01, 8767.19it/s] 97%|█████████▋| 386353/400000 [00:43<00:01, 8734.09it/s] 97%|█████████▋| 387227/400000 [00:43<00:01, 8668.27it/s] 97%|█████████▋| 388095/400000 [00:43<00:01, 8572.07it/s] 97%|█████████▋| 388963/400000 [00:43<00:01, 8603.08it/s] 97%|█████████▋| 389842/400000 [00:43<00:01, 8657.99it/s] 98%|█████████▊| 390726/400000 [00:44<00:01, 8711.65it/s] 98%|█████████▊| 391598/400000 [00:44<00:00, 8615.98it/s] 98%|█████████▊| 392461/400000 [00:44<00:00, 8503.80it/s] 98%|█████████▊| 393325/400000 [00:44<00:00, 8543.69it/s] 99%|█████████▊| 394180/400000 [00:44<00:00, 8425.72it/s] 99%|█████████▉| 395024/400000 [00:44<00:00, 8192.59it/s] 99%|█████████▉| 395888/400000 [00:44<00:00, 8321.85it/s] 99%|█████████▉| 396759/400000 [00:44<00:00, 8434.38it/s] 99%|█████████▉| 397648/400000 [00:44<00:00, 8563.95it/s]100%|█████████▉| 398513/400000 [00:44<00:00, 8587.47it/s]100%|█████████▉| 399373/400000 [00:45<00:00, 8556.51it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8859.28it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb32cd22d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011150961440690768 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.01100512591492771 	 Accuracy: 63

  model saves at 63% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15954 out of table with 15770 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15954 out of table with 15770 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-12 17:23:41.160527: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 17:23:41.164959: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-12 17:23:41.165080: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559a51891b50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 17:23:41.165093: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb2d9703d68> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6130 - accuracy: 0.5035 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6002 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7170 - accuracy: 0.4967
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7586 - accuracy: 0.4940
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 3s - loss: 7.6903 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 3s - loss: 7.6794 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6546 - accuracy: 0.5008
15000/25000 [=================>............] - ETA: 2s - loss: 7.6431 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6187 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6251 - accuracy: 0.5027
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6582 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6381 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
25000/25000 [==============================] - 7s 274us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb2997b8b38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb29a99b128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0361 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.9934 - val_crf_viterbi_accuracy: 0.1600

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

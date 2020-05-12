
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f3818deffd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 18:12:48.411578
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 18:12:48.415672
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 18:12:48.419028
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 18:12:48.422108
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f3824bb9438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 351143.9375
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 221329.0938
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 117099.1016
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 58795.6523
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 32230.6973
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 19538.8828
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 12912.4365
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 9148.7949
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 6876.0864
Epoch 10/10

1/1 [==============================] - 0s 116ms/step - loss: 5435.7695

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.04930401e-02 -3.38372350e-01  4.18514758e-01  2.30637741e+00
   7.38177180e-01  1.01834345e+00  9.75373685e-01 -1.70405746e-01
   8.19979489e-01  1.14888871e+00 -2.24215314e-01  6.99091017e-01
  -5.74770093e-01 -2.69843310e-01 -9.88484383e-01  1.86805379e+00
   6.52453840e-01 -6.19563818e-01 -1.21228647e+00  5.77577591e-01
  -1.02886856e+00  1.17244840e-01 -2.47253299e-01  7.30782449e-01
  -1.01173663e+00  6.60474658e-01 -1.33918095e+00  8.18135917e-01
  -9.49076235e-01  9.55662668e-01 -1.33638096e+00  2.39378959e-01
   1.13508439e+00  2.03143120e+00  1.32799983e+00 -2.13189292e+00
   3.74822855e-01 -1.43631864e+00 -9.75223780e-01  9.12104845e-01
   1.59411693e+00  1.00534558e-02  7.35597610e-02  1.04672432e+00
   1.28334153e+00 -2.51805472e+00 -8.31984520e-01  1.14221501e+00
   6.01749599e-01  5.36937833e-01  1.40653789e-01  7.10325241e-01
  -1.25789046e+00 -1.00612998e+00 -5.17803371e-01  7.23841190e-01
   2.50967681e-01 -1.64964914e+00  2.22969651e-02 -1.04714572e-01
  -4.89016265e-01  6.65505552e+00  8.13968468e+00  8.55711174e+00
   8.78915787e+00  8.42982960e+00  1.01320629e+01  7.41269016e+00
   8.39327621e+00  9.74049473e+00  9.92109776e+00  8.96434784e+00
   8.78563213e+00  9.57135677e+00  1.12157593e+01  9.59122467e+00
   7.49398422e+00  9.05543900e+00  9.48850155e+00  8.12822914e+00
   7.79942083e+00  7.88878584e+00  8.68726730e+00  9.81488419e+00
   9.58865929e+00  9.95214081e+00  7.92195225e+00  1.04824753e+01
   8.55584240e+00  8.83792973e+00  7.66242886e+00  8.87700367e+00
   9.24091339e+00  8.98066044e+00  8.13629532e+00  1.02739325e+01
   1.04280453e+01  9.33837318e+00  9.32572746e+00  7.51358938e+00
   6.79517221e+00  1.02805882e+01  8.73142052e+00  7.76661873e+00
   8.90876198e+00  8.31948853e+00  8.97352123e+00  8.70939445e+00
   7.52316093e+00  1.09033298e+01  9.91649818e+00  8.62264061e+00
   6.89666271e+00  8.77409458e+00  8.90800762e+00  9.74070263e+00
   9.58337879e+00  9.79942608e+00  7.83639240e+00  1.13471155e+01
   1.06139362e+00 -9.34548974e-01 -1.63082266e+00 -1.92505181e-01
   3.53397310e-01 -6.71676993e-01  1.54741108e+00  1.28148901e+00
   1.40566099e+00 -1.35647929e+00  8.20794225e-01  2.90474892e-01
   1.86601579e+00  6.28999650e-01  2.05614969e-01 -2.65405631e+00
  -1.08403718e+00 -9.30039942e-01  2.70010382e-01  9.59009349e-01
  -9.44444954e-01 -9.89379287e-02 -1.87892795e+00 -3.62343639e-02
  -6.26246870e-01  4.37289596e-01  5.36546230e-01  2.16618896e+00
   1.09178519e+00  6.61527395e-01 -5.46084583e-01 -1.17768693e+00
   5.35799861e-01  1.25786436e+00 -1.13119453e-01 -1.73319554e+00
  -1.42135000e+00  6.47762179e-01 -6.78866088e-01 -4.29988980e-01
  -1.25060511e+00 -2.38212943e-02 -5.42508423e-01  7.56914258e-01
  -3.97918582e-01  7.71428943e-02  1.92805767e+00  1.40898407e-01
  -8.92517567e-01  5.13624787e-01 -1.10871649e+00 -1.40489984e+00
  -7.99812734e-01  1.99552834e-01  4.71696228e-01  6.39162064e-01
   1.78641760e+00  1.84713101e+00  1.84755790e+00 -1.19035220e+00
   1.16468263e+00  6.05142117e-01  4.45489407e-01  2.80304718e+00
   1.15742803e+00  1.46801519e+00  5.67871451e-01  1.18444061e+00
   7.54406095e-01  4.24866199e-01  2.31424904e+00  6.88877106e-01
   1.83022857e-01  4.20664310e-01  2.17827976e-01  9.33355927e-01
   4.04848814e-01  1.30931461e+00  2.40790415e+00  7.72561550e-01
   9.42959428e-01  1.22804821e-01  4.79001641e-01  1.19315696e+00
   2.07868814e-01  1.32764542e+00  2.67284346e+00  6.78453624e-01
   4.18577671e-01  1.03460765e+00  1.40008366e+00  5.23771763e-01
   7.83163130e-01  6.17629588e-01  1.20238733e+00  3.49875259e+00
   1.18549275e+00  1.18374050e+00  2.06405044e+00  7.85780787e-01
   2.18449306e+00  4.23700571e-01  8.86393487e-01  1.23340130e+00
   2.69935083e+00  9.30324912e-01  1.27902913e+00  4.28856969e-01
   1.11730623e+00  2.22596586e-01  1.37060916e+00  6.51731312e-01
   2.91781664e-01  1.51711047e+00  7.53894806e-01  3.47369576e+00
   5.95158935e-02  7.62198687e-01  2.25463295e+00  1.51582885e+00
   4.12033558e-01  7.63900328e+00  1.05628414e+01  1.05753307e+01
   1.12597036e+01  9.29329014e+00  6.91611338e+00  1.19843302e+01
   7.90514469e+00  1.04848385e+01  9.96809387e+00  1.03902979e+01
   9.60275650e+00  6.51851130e+00  8.93141174e+00  1.00687971e+01
   1.11164522e+01  7.19340563e+00  1.02302380e+01  9.61829758e+00
   7.40226316e+00  9.03882313e+00  1.04839468e+01  9.66228008e+00
   1.03303127e+01  9.26692867e+00  1.12368841e+01  7.86046982e+00
   9.89556503e+00  1.04313927e+01  6.87007809e+00  9.87314129e+00
   7.68744898e+00  9.88819027e+00  1.03671846e+01  9.20713997e+00
   6.85740995e+00  1.03137007e+01  1.01010971e+01  9.28262997e+00
   9.47982502e+00  9.04334545e+00  9.71192551e+00  8.36732578e+00
   9.51365566e+00  9.42008686e+00  9.48258209e+00  9.19203281e+00
   1.06441317e+01  1.02377586e+01  7.04680347e+00  9.69153786e+00
   7.62012053e+00  8.66225910e+00  8.72842216e+00  9.30822468e+00
   1.07019644e+01  9.50267410e+00  9.13099098e+00  1.05383081e+01
   2.10189962e+00  1.28319550e+00  1.47445691e+00  1.14475870e+00
   2.07595015e+00  2.83580732e+00  1.70633566e+00  3.55188322e+00
   2.32625437e+00  2.30822802e-01  1.79570436e+00  2.57534695e+00
   7.37714767e-01  5.81610560e-01  2.63955474e-01  2.91994333e-01
   1.90477037e+00  2.66165018e+00  4.02528882e-01  6.60935640e-01
   1.36473298e+00  5.59930265e-01  8.08315814e-01  6.91668570e-01
   8.03495646e-02  1.40870500e+00  1.10205829e-01  4.62523818e-01
   3.06086826e+00  1.24619818e+00  3.43474865e-01  4.03283834e-02
   4.59253073e-01  1.27724743e+00  7.35770702e-01  5.31674027e-01
   7.99091280e-01  2.63542891e-01  3.23273611e+00  2.49605989e+00
   1.10723758e+00  4.19157267e-01  2.54167795e-01  2.08196938e-01
   3.79836440e-01  1.64267421e-01  2.41523647e+00  1.29236507e+00
   1.20444441e+00  1.73046899e+00  8.24403286e-01  1.52532077e+00
   2.27591872e-01  2.61186838e+00  2.13336420e+00  3.15886378e-01
   1.59968483e+00  1.34944391e+00  1.14423811e+00  3.58919084e-01
  -7.30820847e+00  9.61721516e+00 -3.42397642e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 18:12:56.535694
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6293
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 18:12:56.540173
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8598.18
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 18:12:56.543547
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    92.677
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 18:12:56.546758
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.028
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139878553014736
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139877343228592
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139877343229096
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139877343229600
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139877343230104
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139877343230608

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3818def240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.629465
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.603257
grad_step = 000002, loss = 0.585706
grad_step = 000003, loss = 0.568157
grad_step = 000004, loss = 0.549963
grad_step = 000005, loss = 0.530481
grad_step = 000006, loss = 0.506938
grad_step = 000007, loss = 0.481622
grad_step = 000008, loss = 0.451992
grad_step = 000009, loss = 0.424155
grad_step = 000010, loss = 0.401040
grad_step = 000011, loss = 0.382856
grad_step = 000012, loss = 0.369250
grad_step = 000013, loss = 0.355747
grad_step = 000014, loss = 0.337572
grad_step = 000015, loss = 0.320098
grad_step = 000016, loss = 0.306699
grad_step = 000017, loss = 0.296248
grad_step = 000018, loss = 0.286532
grad_step = 000019, loss = 0.277024
grad_step = 000020, loss = 0.267953
grad_step = 000021, loss = 0.258820
grad_step = 000022, loss = 0.248761
grad_step = 000023, loss = 0.238443
grad_step = 000024, loss = 0.229217
grad_step = 000025, loss = 0.220797
grad_step = 000026, loss = 0.211945
grad_step = 000027, loss = 0.202684
grad_step = 000028, loss = 0.193621
grad_step = 000029, loss = 0.185275
grad_step = 000030, loss = 0.177737
grad_step = 000031, loss = 0.170525
grad_step = 000032, loss = 0.163275
grad_step = 000033, loss = 0.156113
grad_step = 000034, loss = 0.149384
grad_step = 000035, loss = 0.143114
grad_step = 000036, loss = 0.136883
grad_step = 000037, loss = 0.130761
grad_step = 000038, loss = 0.124825
grad_step = 000039, loss = 0.118972
grad_step = 000040, loss = 0.113241
grad_step = 000041, loss = 0.107830
grad_step = 000042, loss = 0.102741
grad_step = 000043, loss = 0.097711
grad_step = 000044, loss = 0.092722
grad_step = 000045, loss = 0.088078
grad_step = 000046, loss = 0.083784
grad_step = 000047, loss = 0.079508
grad_step = 000048, loss = 0.075330
grad_step = 000049, loss = 0.071407
grad_step = 000050, loss = 0.067608
grad_step = 000051, loss = 0.063925
grad_step = 000052, loss = 0.060439
grad_step = 000053, loss = 0.057065
grad_step = 000054, loss = 0.053741
grad_step = 000055, loss = 0.050602
grad_step = 000056, loss = 0.047657
grad_step = 000057, loss = 0.044837
grad_step = 000058, loss = 0.042145
grad_step = 000059, loss = 0.039568
grad_step = 000060, loss = 0.037130
grad_step = 000061, loss = 0.034813
grad_step = 000062, loss = 0.032587
grad_step = 000063, loss = 0.030495
grad_step = 000064, loss = 0.028529
grad_step = 000065, loss = 0.026642
grad_step = 000066, loss = 0.024855
grad_step = 000067, loss = 0.023163
grad_step = 000068, loss = 0.021567
grad_step = 000069, loss = 0.020096
grad_step = 000070, loss = 0.018714
grad_step = 000071, loss = 0.017421
grad_step = 000072, loss = 0.016234
grad_step = 000073, loss = 0.015127
grad_step = 000074, loss = 0.014111
grad_step = 000075, loss = 0.013144
grad_step = 000076, loss = 0.012208
grad_step = 000077, loss = 0.011283
grad_step = 000078, loss = 0.010475
grad_step = 000079, loss = 0.009794
grad_step = 000080, loss = 0.009148
grad_step = 000081, loss = 0.008503
grad_step = 000082, loss = 0.007881
grad_step = 000083, loss = 0.007356
grad_step = 000084, loss = 0.006907
grad_step = 000085, loss = 0.006456
grad_step = 000086, loss = 0.006007
grad_step = 000087, loss = 0.005603
grad_step = 000088, loss = 0.005269
grad_step = 000089, loss = 0.004970
grad_step = 000090, loss = 0.004670
grad_step = 000091, loss = 0.004376
grad_step = 000092, loss = 0.004114
grad_step = 000093, loss = 0.003894
grad_step = 000094, loss = 0.003700
grad_step = 000095, loss = 0.003513
grad_step = 000096, loss = 0.003329
grad_step = 000097, loss = 0.003155
grad_step = 000098, loss = 0.003003
grad_step = 000099, loss = 0.002876
grad_step = 000100, loss = 0.002765
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002664
grad_step = 000102, loss = 0.002567
grad_step = 000103, loss = 0.002475
grad_step = 000104, loss = 0.002387
grad_step = 000105, loss = 0.002307
grad_step = 000106, loss = 0.002237
grad_step = 000107, loss = 0.002177
grad_step = 000108, loss = 0.002125
grad_step = 000109, loss = 0.002082
grad_step = 000110, loss = 0.002044
grad_step = 000111, loss = 0.002013
grad_step = 000112, loss = 0.001992
grad_step = 000113, loss = 0.001979
grad_step = 000114, loss = 0.001982
grad_step = 000115, loss = 0.001989
grad_step = 000116, loss = 0.002001
grad_step = 000117, loss = 0.001967
grad_step = 000118, loss = 0.001907
grad_step = 000119, loss = 0.001832
grad_step = 000120, loss = 0.001798
grad_step = 000121, loss = 0.001808
grad_step = 000122, loss = 0.001831
grad_step = 000123, loss = 0.001838
grad_step = 000124, loss = 0.001808
grad_step = 000125, loss = 0.001769
grad_step = 000126, loss = 0.001743
grad_step = 000127, loss = 0.001740
grad_step = 000128, loss = 0.001752
grad_step = 000129, loss = 0.001760
grad_step = 000130, loss = 0.001763
grad_step = 000131, loss = 0.001749
grad_step = 000132, loss = 0.001731
grad_step = 000133, loss = 0.001710
grad_step = 000134, loss = 0.001695
grad_step = 000135, loss = 0.001692
grad_step = 000136, loss = 0.001695
grad_step = 000137, loss = 0.001705
grad_step = 000138, loss = 0.001717
grad_step = 000139, loss = 0.001737
grad_step = 000140, loss = 0.001760
grad_step = 000141, loss = 0.001798
grad_step = 000142, loss = 0.001831
grad_step = 000143, loss = 0.001869
grad_step = 000144, loss = 0.001853
grad_step = 000145, loss = 0.001817
grad_step = 000146, loss = 0.001736
grad_step = 000147, loss = 0.001674
grad_step = 000148, loss = 0.001646
grad_step = 000149, loss = 0.001660
grad_step = 000150, loss = 0.001700
grad_step = 000151, loss = 0.001738
grad_step = 000152, loss = 0.001769
grad_step = 000153, loss = 0.001764
grad_step = 000154, loss = 0.001745
grad_step = 000155, loss = 0.001699
grad_step = 000156, loss = 0.001656
grad_step = 000157, loss = 0.001626
grad_step = 000158, loss = 0.001618
grad_step = 000159, loss = 0.001625
grad_step = 000160, loss = 0.001644
grad_step = 000161, loss = 0.001674
grad_step = 000162, loss = 0.001709
grad_step = 000163, loss = 0.001763
grad_step = 000164, loss = 0.001797
grad_step = 000165, loss = 0.001845
grad_step = 000166, loss = 0.001819
grad_step = 000167, loss = 0.001775
grad_step = 000168, loss = 0.001681
grad_step = 000169, loss = 0.001614
grad_step = 000170, loss = 0.001589
grad_step = 000171, loss = 0.001611
grad_step = 000172, loss = 0.001661
grad_step = 000173, loss = 0.001701
grad_step = 000174, loss = 0.001732
grad_step = 000175, loss = 0.001712
grad_step = 000176, loss = 0.001679
grad_step = 000177, loss = 0.001625
grad_step = 000178, loss = 0.001588
grad_step = 000179, loss = 0.001572
grad_step = 000180, loss = 0.001577
grad_step = 000181, loss = 0.001598
grad_step = 000182, loss = 0.001624
grad_step = 000183, loss = 0.001659
grad_step = 000184, loss = 0.001682
grad_step = 000185, loss = 0.001713
grad_step = 000186, loss = 0.001713
grad_step = 000187, loss = 0.001713
grad_step = 000188, loss = 0.001675
grad_step = 000189, loss = 0.001639
grad_step = 000190, loss = 0.001594
grad_step = 000191, loss = 0.001565
grad_step = 000192, loss = 0.001552
grad_step = 000193, loss = 0.001555
grad_step = 000194, loss = 0.001570
grad_step = 000195, loss = 0.001589
grad_step = 000196, loss = 0.001617
grad_step = 000197, loss = 0.001643
grad_step = 000198, loss = 0.001683
grad_step = 000199, loss = 0.001706
grad_step = 000200, loss = 0.001745
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001733
grad_step = 000202, loss = 0.001722
grad_step = 000203, loss = 0.001655
grad_step = 000204, loss = 0.001598
grad_step = 000205, loss = 0.001549
grad_step = 000206, loss = 0.001534
grad_step = 000207, loss = 0.001551
grad_step = 000208, loss = 0.001579
grad_step = 000209, loss = 0.001607
grad_step = 000210, loss = 0.001610
grad_step = 000211, loss = 0.001607
grad_step = 000212, loss = 0.001582
grad_step = 000213, loss = 0.001559
grad_step = 000214, loss = 0.001538
grad_step = 000215, loss = 0.001525
grad_step = 000216, loss = 0.001520
grad_step = 000217, loss = 0.001520
grad_step = 000218, loss = 0.001526
grad_step = 000219, loss = 0.001536
grad_step = 000220, loss = 0.001553
grad_step = 000221, loss = 0.001576
grad_step = 000222, loss = 0.001619
grad_step = 000223, loss = 0.001670
grad_step = 000224, loss = 0.001767
grad_step = 000225, loss = 0.001830
grad_step = 000226, loss = 0.001927
grad_step = 000227, loss = 0.001848
grad_step = 000228, loss = 0.001739
grad_step = 000229, loss = 0.001573
grad_step = 000230, loss = 0.001507
grad_step = 000231, loss = 0.001554
grad_step = 000232, loss = 0.001635
grad_step = 000233, loss = 0.001672
grad_step = 000234, loss = 0.001604
grad_step = 000235, loss = 0.001526
grad_step = 000236, loss = 0.001499
grad_step = 000237, loss = 0.001531
grad_step = 000238, loss = 0.001584
grad_step = 000239, loss = 0.001598
grad_step = 000240, loss = 0.001582
grad_step = 000241, loss = 0.001536
grad_step = 000242, loss = 0.001500
grad_step = 000243, loss = 0.001488
grad_step = 000244, loss = 0.001498
grad_step = 000245, loss = 0.001520
grad_step = 000246, loss = 0.001537
grad_step = 000247, loss = 0.001545
grad_step = 000248, loss = 0.001534
grad_step = 000249, loss = 0.001517
grad_step = 000250, loss = 0.001497
grad_step = 000251, loss = 0.001483
grad_step = 000252, loss = 0.001476
grad_step = 000253, loss = 0.001477
grad_step = 000254, loss = 0.001483
grad_step = 000255, loss = 0.001491
grad_step = 000256, loss = 0.001500
grad_step = 000257, loss = 0.001507
grad_step = 000258, loss = 0.001514
grad_step = 000259, loss = 0.001519
grad_step = 000260, loss = 0.001526
grad_step = 000261, loss = 0.001530
grad_step = 000262, loss = 0.001539
grad_step = 000263, loss = 0.001544
grad_step = 000264, loss = 0.001557
grad_step = 000265, loss = 0.001564
grad_step = 000266, loss = 0.001579
grad_step = 000267, loss = 0.001582
grad_step = 000268, loss = 0.001592
grad_step = 000269, loss = 0.001581
grad_step = 000270, loss = 0.001573
grad_step = 000271, loss = 0.001544
grad_step = 000272, loss = 0.001518
grad_step = 000273, loss = 0.001486
grad_step = 000274, loss = 0.001464
grad_step = 000275, loss = 0.001450
grad_step = 000276, loss = 0.001447
grad_step = 000277, loss = 0.001453
grad_step = 000278, loss = 0.001463
grad_step = 000279, loss = 0.001474
grad_step = 000280, loss = 0.001482
grad_step = 000281, loss = 0.001492
grad_step = 000282, loss = 0.001496
grad_step = 000283, loss = 0.001504
grad_step = 000284, loss = 0.001506
grad_step = 000285, loss = 0.001513
grad_step = 000286, loss = 0.001514
grad_step = 000287, loss = 0.001521
grad_step = 000288, loss = 0.001519
grad_step = 000289, loss = 0.001522
grad_step = 000290, loss = 0.001514
grad_step = 000291, loss = 0.001507
grad_step = 000292, loss = 0.001490
grad_step = 000293, loss = 0.001475
grad_step = 000294, loss = 0.001457
grad_step = 000295, loss = 0.001442
grad_step = 000296, loss = 0.001431
grad_step = 000297, loss = 0.001424
grad_step = 000298, loss = 0.001419
grad_step = 000299, loss = 0.001418
grad_step = 000300, loss = 0.001418
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001421
grad_step = 000302, loss = 0.001426
grad_step = 000303, loss = 0.001435
grad_step = 000304, loss = 0.001455
grad_step = 000305, loss = 0.001488
grad_step = 000306, loss = 0.001556
grad_step = 000307, loss = 0.001663
grad_step = 000308, loss = 0.001881
grad_step = 000309, loss = 0.002091
grad_step = 000310, loss = 0.002412
grad_step = 000311, loss = 0.002236
grad_step = 000312, loss = 0.001896
grad_step = 000313, loss = 0.001480
grad_step = 000314, loss = 0.001443
grad_step = 000315, loss = 0.001701
grad_step = 000316, loss = 0.001814
grad_step = 000317, loss = 0.001660
grad_step = 000318, loss = 0.001423
grad_step = 000319, loss = 0.001464
grad_step = 000320, loss = 0.001644
grad_step = 000321, loss = 0.001620
grad_step = 000322, loss = 0.001455
grad_step = 000323, loss = 0.001409
grad_step = 000324, loss = 0.001519
grad_step = 000325, loss = 0.001570
grad_step = 000326, loss = 0.001465
grad_step = 000327, loss = 0.001393
grad_step = 000328, loss = 0.001452
grad_step = 000329, loss = 0.001507
grad_step = 000330, loss = 0.001470
grad_step = 000331, loss = 0.001398
grad_step = 000332, loss = 0.001401
grad_step = 000333, loss = 0.001453
grad_step = 000334, loss = 0.001457
grad_step = 000335, loss = 0.001412
grad_step = 000336, loss = 0.001381
grad_step = 000337, loss = 0.001404
grad_step = 000338, loss = 0.001430
grad_step = 000339, loss = 0.001420
grad_step = 000340, loss = 0.001387
grad_step = 000341, loss = 0.001376
grad_step = 000342, loss = 0.001393
grad_step = 000343, loss = 0.001407
grad_step = 000344, loss = 0.001398
grad_step = 000345, loss = 0.001377
grad_step = 000346, loss = 0.001369
grad_step = 000347, loss = 0.001376
grad_step = 000348, loss = 0.001387
grad_step = 000349, loss = 0.001386
grad_step = 000350, loss = 0.001375
grad_step = 000351, loss = 0.001365
grad_step = 000352, loss = 0.001362
grad_step = 000353, loss = 0.001366
grad_step = 000354, loss = 0.001371
grad_step = 000355, loss = 0.001372
grad_step = 000356, loss = 0.001367
grad_step = 000357, loss = 0.001361
grad_step = 000358, loss = 0.001356
grad_step = 000359, loss = 0.001354
grad_step = 000360, loss = 0.001355
grad_step = 000361, loss = 0.001357
grad_step = 000362, loss = 0.001359
grad_step = 000363, loss = 0.001359
grad_step = 000364, loss = 0.001358
grad_step = 000365, loss = 0.001355
grad_step = 000366, loss = 0.001352
grad_step = 000367, loss = 0.001349
grad_step = 000368, loss = 0.001346
grad_step = 000369, loss = 0.001344
grad_step = 000370, loss = 0.001343
grad_step = 000371, loss = 0.001341
grad_step = 000372, loss = 0.001341
grad_step = 000373, loss = 0.001340
grad_step = 000374, loss = 0.001340
grad_step = 000375, loss = 0.001340
grad_step = 000376, loss = 0.001340
grad_step = 000377, loss = 0.001342
grad_step = 000378, loss = 0.001345
grad_step = 000379, loss = 0.001351
grad_step = 000380, loss = 0.001360
grad_step = 000381, loss = 0.001376
grad_step = 000382, loss = 0.001400
grad_step = 000383, loss = 0.001444
grad_step = 000384, loss = 0.001495
grad_step = 000385, loss = 0.001589
grad_step = 000386, loss = 0.001653
grad_step = 000387, loss = 0.001744
grad_step = 000388, loss = 0.001711
grad_step = 000389, loss = 0.001636
grad_step = 000390, loss = 0.001476
grad_step = 000391, loss = 0.001357
grad_step = 000392, loss = 0.001329
grad_step = 000393, loss = 0.001384
grad_step = 000394, loss = 0.001462
grad_step = 000395, loss = 0.001482
grad_step = 000396, loss = 0.001443
grad_step = 000397, loss = 0.001368
grad_step = 000398, loss = 0.001323
grad_step = 000399, loss = 0.001330
grad_step = 000400, loss = 0.001368
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001401
grad_step = 000402, loss = 0.001399
grad_step = 000403, loss = 0.001371
grad_step = 000404, loss = 0.001334
grad_step = 000405, loss = 0.001313
grad_step = 000406, loss = 0.001315
grad_step = 000407, loss = 0.001331
grad_step = 000408, loss = 0.001350
grad_step = 000409, loss = 0.001359
grad_step = 000410, loss = 0.001356
grad_step = 000411, loss = 0.001341
grad_step = 000412, loss = 0.001324
grad_step = 000413, loss = 0.001310
grad_step = 000414, loss = 0.001303
grad_step = 000415, loss = 0.001304
grad_step = 000416, loss = 0.001309
grad_step = 000417, loss = 0.001316
grad_step = 000418, loss = 0.001322
grad_step = 000419, loss = 0.001327
grad_step = 000420, loss = 0.001328
grad_step = 000421, loss = 0.001327
grad_step = 000422, loss = 0.001324
grad_step = 000423, loss = 0.001321
grad_step = 000424, loss = 0.001316
grad_step = 000425, loss = 0.001313
grad_step = 000426, loss = 0.001310
grad_step = 000427, loss = 0.001307
grad_step = 000428, loss = 0.001305
grad_step = 000429, loss = 0.001305
grad_step = 000430, loss = 0.001305
grad_step = 000431, loss = 0.001307
grad_step = 000432, loss = 0.001311
grad_step = 000433, loss = 0.001318
grad_step = 000434, loss = 0.001330
grad_step = 000435, loss = 0.001352
grad_step = 000436, loss = 0.001383
grad_step = 000437, loss = 0.001440
grad_step = 000438, loss = 0.001513
grad_step = 000439, loss = 0.001638
grad_step = 000440, loss = 0.001730
grad_step = 000441, loss = 0.001855
grad_step = 000442, loss = 0.001808
grad_step = 000443, loss = 0.001702
grad_step = 000444, loss = 0.001469
grad_step = 000445, loss = 0.001307
grad_step = 000446, loss = 0.001291
grad_step = 000447, loss = 0.001390
grad_step = 000448, loss = 0.001492
grad_step = 000449, loss = 0.001477
grad_step = 000450, loss = 0.001380
grad_step = 000451, loss = 0.001289
grad_step = 000452, loss = 0.001284
grad_step = 000453, loss = 0.001345
grad_step = 000454, loss = 0.001392
grad_step = 000455, loss = 0.001384
grad_step = 000456, loss = 0.001325
grad_step = 000457, loss = 0.001277
grad_step = 000458, loss = 0.001271
grad_step = 000459, loss = 0.001300
grad_step = 000460, loss = 0.001332
grad_step = 000461, loss = 0.001335
grad_step = 000462, loss = 0.001313
grad_step = 000463, loss = 0.001281
grad_step = 000464, loss = 0.001263
grad_step = 000465, loss = 0.001266
grad_step = 000466, loss = 0.001282
grad_step = 000467, loss = 0.001297
grad_step = 000468, loss = 0.001298
grad_step = 000469, loss = 0.001288
grad_step = 000470, loss = 0.001272
grad_step = 000471, loss = 0.001259
grad_step = 000472, loss = 0.001254
grad_step = 000473, loss = 0.001257
grad_step = 000474, loss = 0.001264
grad_step = 000475, loss = 0.001270
grad_step = 000476, loss = 0.001274
grad_step = 000477, loss = 0.001272
grad_step = 000478, loss = 0.001268
grad_step = 000479, loss = 0.001261
grad_step = 000480, loss = 0.001255
grad_step = 000481, loss = 0.001250
grad_step = 000482, loss = 0.001246
grad_step = 000483, loss = 0.001244
grad_step = 000484, loss = 0.001244
grad_step = 000485, loss = 0.001244
grad_step = 000486, loss = 0.001245
grad_step = 000487, loss = 0.001247
grad_step = 000488, loss = 0.001249
grad_step = 000489, loss = 0.001252
grad_step = 000490, loss = 0.001256
grad_step = 000491, loss = 0.001263
grad_step = 000492, loss = 0.001273
grad_step = 000493, loss = 0.001290
grad_step = 000494, loss = 0.001312
grad_step = 000495, loss = 0.001353
grad_step = 000496, loss = 0.001403
grad_step = 000497, loss = 0.001491
grad_step = 000498, loss = 0.001575
grad_step = 000499, loss = 0.001699
grad_step = 000500, loss = 0.001730
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001730
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

  date_run                              2020-05-12 18:13:18.960097
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.288567
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 18:13:18.966245
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.221301
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 18:13:18.972555
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149699
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 18:13:18.977438
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.36274
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
0   2020-05-12 18:12:48.411578  ...    mean_absolute_error
1   2020-05-12 18:12:48.415672  ...     mean_squared_error
2   2020-05-12 18:12:48.419028  ...  median_absolute_error
3   2020-05-12 18:12:48.422108  ...               r2_score
4   2020-05-12 18:12:56.535694  ...    mean_absolute_error
5   2020-05-12 18:12:56.540173  ...     mean_squared_error
6   2020-05-12 18:12:56.543547  ...  median_absolute_error
7   2020-05-12 18:12:56.546758  ...               r2_score
8   2020-05-12 18:13:18.960097  ...    mean_absolute_error
9   2020-05-12 18:13:18.966245  ...     mean_squared_error
10  2020-05-12 18:13:18.972555  ...  median_absolute_error
11  2020-05-12 18:13:18.977438  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3792896/9912422 [00:00<00:00, 35517866.29it/s]9920512it [00:00, 35438460.52it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:01<?, ?it/s]32768it [00:01, 18072.49it/s]            
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 3374299.33it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 17928.66it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b865e8cf8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b38fa1eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b385cf0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b38fa1eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b385270b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b35d534e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b35d4d710> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b38fa1eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b384e56d8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b35d534e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b865aaf28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc975a39240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d327ccef7eacbb30c9f3c9ee97a160637ed03edcffc58ff1fa04a0f08e383093
  Stored in directory: /tmp/pip-ephem-wheel-cache-3elims78/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc90d831828> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2408448/17464789 [===>..........................] - ETA: 0s
 7471104/17464789 [===========>..................] - ETA: 0s
12353536/17464789 [====================>.........] - ETA: 0s
17104896/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 18:14:45.563005: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 18:14:45.566649: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 18:14:45.566779: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55608a071be0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 18:14:45.566794: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7778 - accuracy: 0.4927
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7152 - accuracy: 0.4968
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7214 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6252 - accuracy: 0.5027
11000/25000 [============>.................] - ETA: 4s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 3s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6631 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6781 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6905 - accuracy: 0.4984
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6866 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 9s 358us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 18:15:01.010742
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 18:15:01.010742  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:07:41, 11.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:01:30, 15.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:34:16, 22.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:24:29, 32.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:10:21, 46.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.18M/862M [00:01<3:36:12, 65.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:30:41, 94.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<1:45:02, 134kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:13:16, 191kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<51:17, 273kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:01<35:45, 388kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:01<25:02, 552kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.3M/862M [00:02<17:30, 785kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<12:19, 1.11MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:02<08:39, 1.57MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:02<06:09, 2.20MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<05:05, 2.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<05:27, 2.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<05:38, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<04:24, 3.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<05:37, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<05:26, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:11, 3.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<05:44, 2.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:14, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<06:05, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:09<04:27, 2.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<11:02, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<09:20, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<06:52, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<07:28, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<08:25, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<06:42, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:13<04:52, 2.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<11:27, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<09:36, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<07:03, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<07:36, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<08:29, 1.53MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<06:37, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<04:49, 2.68MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<07:44, 1.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<06:59, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<05:16, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<06:19, 2.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<07:34, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:21<04:20, 2.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<09:26, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<08:09, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:50, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:53, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<06:18, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:35, 2.76MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:17, 1.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<09:26, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:56, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:24, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:15, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<06:32, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:45, 2.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<11:19, 1.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<09:24, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<06:54, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:23, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:23, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:38, 2.67MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:55, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:04, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:11, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:29, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:53, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:18, 2.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:22, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:39, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<05:01, 2.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:00, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:10, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:07, 2.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:47, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:55, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:09, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:03, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:42, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:20, 2.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:44, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:27, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:57, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:13, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:33, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:51, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:42, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:00, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:22, 2.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:55, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:02, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:30, 2.60MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<03:21, 3.49MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:21, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:23, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:11, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<05:13, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<07:06, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:24, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:47, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<03:29, 3.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<28:58, 399kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<21:37, 534kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<15:27, 746kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<13:10, 872kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<12:00, 956kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<09:00, 1.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:34, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:12, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:24, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:48, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:43, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:47, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:55, 2.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:05, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:38, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:16, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:17, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:26, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<03:47, 2.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<09:00, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<07:39, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:39, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<04:04, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<2:15:23, 81.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<1:37:30, 113kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<1:08:52, 161kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<48:11, 229kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<40:28, 272kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<29:40, 371kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<21:01, 522kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<14:47, 739kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<1:09:39, 157kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<50:04, 218kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<35:18, 309kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<26:50, 405kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<21:12, 512kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<15:22, 705kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<10:53, 992kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<11:36, 930kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:09, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:45, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<04:51, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<13:11, 813kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<11:51, 904kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<08:52, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:29, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:52, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:08, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:35, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:24<03:22, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<09:07, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<07:38, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<05:35, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<04:04, 2.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:36, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<09:16, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:01, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:15, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:44, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:17, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:56, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<02:56, 3.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<08:41, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<08:37, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:35, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:57, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:05, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:49, 2.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:50, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:43, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:26, 2.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:53, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:28, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:29, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<03:59, 2.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<09:03, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:49, 1.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:40, 1.51MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:00, 2.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:29, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:51, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:52, 2.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<02:49, 3.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:18, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:15, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:17, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:44, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:15, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:47, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:35, 2.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<02:38, 3.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:28, 729kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<11:42, 839kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<08:37, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:40, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:52, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:23, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:02, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:44, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:32, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:18, 2.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<13:58, 688kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<10:58, 875kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:56, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:29, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:35, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:49, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:12, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:24, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:01, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:11, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:32, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:58, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:20, 2.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:41, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:30, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:49, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:56, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:39, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:23, 2.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:40, 2.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<04:26, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:20, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:17, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:06, 2.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<07:49, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:34, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:50, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:50, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:38, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:21, 2.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<08:02, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:43, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:57, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:16, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:50, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:18, 2.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:16, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:26, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:17, 2.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:13, 2.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:00, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:02, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:53, 2.98MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:49, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:06, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:07, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:06, 2.72MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:47, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:48, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:45, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:30, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:37, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:14, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:02, 2.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:48, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:39, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:45, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:44, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:07, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:58, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:24, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:43, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<05:23, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:11, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:02, 2.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:05, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:32, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:24, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:00, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:46, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:52, 2.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:37, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:16, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:25, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:29, 3.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<11:12, 697kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<08:47, 888kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<06:20, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:00, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:05, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<04:44, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:24, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:16, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:01, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:26, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:05, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:57, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:52, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:39, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:06, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:41, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:24, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:35, 2.88MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:11, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:23, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:27, 2.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:56, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:02, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:44, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:05, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:38, 1.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:37, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:38, 2.73MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:21, 1.65MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:55, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:57, 2.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:30, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<04:16, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:21, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:26, 2.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:43, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:09, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:06, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:35, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<04:14, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:19, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:27, 2.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:15, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:27, 2.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:55, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:05, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:17, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:31, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:17, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:30, 2.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:06, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:49, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:01, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:13, 2.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:13, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:27, 2.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:03, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:57, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:53, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:49, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<02:07, 3.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:48, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:52, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:04, 3.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:52, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:10, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:04, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<02:13, 2.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<1:16:45, 81.8kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<54:21, 115kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<38:04, 164kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<27:51, 223kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<20:09, 308kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<14:13, 435kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<11:14, 546kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<09:21, 656kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:51, 893kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<04:52, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:37, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:41, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:27, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:36, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:59, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:07, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:15, 2.64MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:00, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:31, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:36, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<01:53, 3.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<36:30, 161kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<26:58, 217kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<19:09, 305kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<13:28, 432kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<10:51, 534kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:16, 699kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:55, 973kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:18, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:24, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:13, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:18, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<17:03, 332kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<13:22, 423kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<09:41, 582kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<06:48, 823kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<08:10, 684kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:23, 873kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<04:36, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:20, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:22, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:24, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<02:26, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<05:11, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:16, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:07, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:16, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:36, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:01, 2.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:18, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:39, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:42, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:57, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:21, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:37, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:53, 2.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:12, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:52, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:09, 2.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:32, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:05, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:26, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:46, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:48, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:34, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:56, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:21, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<02:16, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:43, 2.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:12, 2.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:09, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:34, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:08, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:47, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:05, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:25, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:50, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:16, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:39, 2.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:09, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:27, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:32, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:01, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:21, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:42, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:58, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:15, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:24, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:35, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:54, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:15, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:38, 2.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:50, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:13, 1.37MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:22, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:32, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:52, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:14, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:36, 2.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:47, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:09, 1.35MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:19, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:28, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:45, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:08, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:32, 2.68MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:41, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:22, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:46, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:03, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:55, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:26, 2.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:49, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:14, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:47, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:17, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:35, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:16, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:42, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:58, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:18, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:50, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:19, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:59, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:33, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:53, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:04, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:21, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:50, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:19, 2.77MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:27, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:07, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:33, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:52, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:10, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:14, 2.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:36, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:15, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:39, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:51, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:08, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:39, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:12, 2.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:50, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:41, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:16, 2.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:33, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:29, 2.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:08, 2.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:24, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:03, 3.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:22, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:44, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:24, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:01, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:40, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:14, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:45, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:58, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:33, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:07, 2.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:38, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:12, 1.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:37, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.68MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:54, 1.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:30, 1.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:04, 2.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:32, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:07, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:32, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:37, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:50, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:27, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:02, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:25, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:01, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:29, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:23, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:02, 2.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:14, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:52, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:06, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:23, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:47, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:30, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:20, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:59, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:09, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:06, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:47, 2.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:59, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:40, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:13, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:17, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:09, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<00:51, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:11, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:55, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:39, 3.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:40, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:25, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:02, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:07, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:59, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:44, 2.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:54, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:05, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:37, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:14, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:03, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:47, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:34, 3.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:26, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:13, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:53, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:06, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:51, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:36, 2.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:23, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:08, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:50, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:54, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:01, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:48, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:34, 2.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:20, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:07, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:49, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:57, 1.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:32, 2.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:27, 2.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:20, 3.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:56, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:58, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:31, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:44, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:33, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:42, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:33, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:23, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:57, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:46, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:33, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:31, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:23, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:27, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:18, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:45, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:38, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:27, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:31, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:24, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:16, 2.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:39, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:32, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:23, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:25, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:13, 2.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:50, 710kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:38, 912kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:26, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:24, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:18, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:11, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:39, 689kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:30, 878kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:21, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:13, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<11:43, 32.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<08:14, 46.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<05:37, 65.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:38, 94.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<02:26, 129kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<01:43, 180kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<01:07, 255kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:43, 340kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:31, 458kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:20, 642kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:13, 771kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:10, 972kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.01MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 877/400000 [00:00<00:45, 8765.41it/s]  0%|          | 1870/400000 [00:00<00:43, 9084.79it/s]  1%|          | 2864/400000 [00:00<00:42, 9323.12it/s]  1%|          | 3830/400000 [00:00<00:42, 9421.13it/s]  1%|          | 4688/400000 [00:00<00:43, 9151.45it/s]  1%|▏         | 5511/400000 [00:00<00:44, 8852.00it/s]  2%|▏         | 6454/400000 [00:00<00:43, 9017.13it/s]  2%|▏         | 7375/400000 [00:00<00:43, 9071.71it/s]  2%|▏         | 8338/400000 [00:00<00:42, 9231.33it/s]  2%|▏         | 9319/400000 [00:01<00:41, 9395.87it/s]  3%|▎         | 10237/400000 [00:01<00:42, 9194.20it/s]  3%|▎         | 11195/400000 [00:01<00:41, 9303.69it/s]  3%|▎         | 12125/400000 [00:01<00:41, 9302.21it/s]  3%|▎         | 13049/400000 [00:01<00:41, 9249.81it/s]  3%|▎         | 13970/400000 [00:01<00:42, 9140.99it/s]  4%|▎         | 14897/400000 [00:01<00:41, 9176.90it/s]  4%|▍         | 15879/400000 [00:01<00:41, 9359.10it/s]  4%|▍         | 16830/400000 [00:01<00:40, 9402.69it/s]  4%|▍         | 17792/400000 [00:01<00:40, 9466.61it/s]  5%|▍         | 18739/400000 [00:02<00:40, 9417.01it/s]  5%|▍         | 19681/400000 [00:02<00:41, 9245.57it/s]  5%|▌         | 20607/400000 [00:02<00:41, 9178.44it/s]  5%|▌         | 21574/400000 [00:02<00:40, 9320.33it/s]  6%|▌         | 22529/400000 [00:02<00:40, 9387.03it/s]  6%|▌         | 23469/400000 [00:02<00:40, 9276.77it/s]  6%|▌         | 24405/400000 [00:02<00:40, 9300.81it/s]  6%|▋         | 25373/400000 [00:02<00:39, 9411.38it/s]  7%|▋         | 26325/400000 [00:02<00:39, 9440.86it/s]  7%|▋         | 27277/400000 [00:02<00:39, 9461.31it/s]  7%|▋         | 28224/400000 [00:03<00:39, 9339.54it/s]  7%|▋         | 29159/400000 [00:03<00:40, 9193.88it/s]  8%|▊         | 30135/400000 [00:03<00:39, 9356.27it/s]  8%|▊         | 31101/400000 [00:03<00:39, 9445.15it/s]  8%|▊         | 32056/400000 [00:03<00:38, 9476.15it/s]  8%|▊         | 33005/400000 [00:03<00:40, 8972.81it/s]  8%|▊         | 33909/400000 [00:03<00:41, 8896.27it/s]  9%|▊         | 34842/400000 [00:03<00:40, 9021.20it/s]  9%|▉         | 35748/400000 [00:03<00:40, 8958.47it/s]  9%|▉         | 36697/400000 [00:03<00:39, 9109.56it/s]  9%|▉         | 37652/400000 [00:04<00:39, 9236.62it/s] 10%|▉         | 38578/400000 [00:04<00:39, 9078.27it/s] 10%|▉         | 39488/400000 [00:04<00:40, 8973.15it/s] 10%|█         | 40388/400000 [00:04<00:40, 8877.08it/s] 10%|█         | 41278/400000 [00:04<00:40, 8815.90it/s] 11%|█         | 42161/400000 [00:04<00:40, 8767.72it/s] 11%|█         | 43078/400000 [00:04<00:40, 8884.43it/s] 11%|█         | 43968/400000 [00:04<00:41, 8554.05it/s] 11%|█         | 44827/400000 [00:04<00:42, 8347.09it/s] 11%|█▏        | 45666/400000 [00:05<00:43, 8225.86it/s] 12%|█▏        | 46492/400000 [00:05<00:43, 8103.77it/s] 12%|█▏        | 47326/400000 [00:05<00:43, 8172.13it/s] 12%|█▏        | 48146/400000 [00:05<00:43, 8159.49it/s] 12%|█▏        | 49046/400000 [00:05<00:41, 8391.55it/s] 12%|█▏        | 49888/400000 [00:05<00:41, 8351.04it/s] 13%|█▎        | 50725/400000 [00:05<00:42, 8228.20it/s] 13%|█▎        | 51647/400000 [00:05<00:40, 8500.57it/s] 13%|█▎        | 52596/400000 [00:05<00:39, 8773.84it/s] 13%|█▎        | 53511/400000 [00:05<00:39, 8883.39it/s] 14%|█▎        | 54403/400000 [00:06<00:39, 8655.37it/s] 14%|█▍        | 55273/400000 [00:06<00:40, 8473.42it/s] 14%|█▍        | 56148/400000 [00:06<00:40, 8551.74it/s] 14%|█▍        | 57133/400000 [00:06<00:38, 8902.52it/s] 15%|█▍        | 58036/400000 [00:06<00:38, 8939.81it/s] 15%|█▍        | 58974/400000 [00:06<00:37, 9064.93it/s] 15%|█▍        | 59884/400000 [00:06<00:37, 9052.37it/s] 15%|█▌        | 60847/400000 [00:06<00:36, 9217.29it/s] 15%|█▌        | 61809/400000 [00:06<00:36, 9332.57it/s] 16%|█▌        | 62750/400000 [00:06<00:36, 9352.89it/s] 16%|█▌        | 63705/400000 [00:07<00:35, 9410.30it/s] 16%|█▌        | 64648/400000 [00:07<00:36, 9170.25it/s] 16%|█▋        | 65568/400000 [00:07<00:36, 9065.99it/s] 17%|█▋        | 66556/400000 [00:07<00:35, 9295.44it/s] 17%|█▋        | 67528/400000 [00:07<00:35, 9418.39it/s] 17%|█▋        | 68508/400000 [00:07<00:34, 9528.04it/s] 17%|█▋        | 69463/400000 [00:07<00:36, 9170.75it/s] 18%|█▊        | 70424/400000 [00:07<00:35, 9296.17it/s] 18%|█▊        | 71405/400000 [00:07<00:34, 9443.33it/s] 18%|█▊        | 72374/400000 [00:07<00:34, 9515.42it/s] 18%|█▊        | 73336/400000 [00:08<00:34, 9546.48it/s] 19%|█▊        | 74293/400000 [00:08<00:34, 9402.84it/s] 19%|█▉        | 75262/400000 [00:08<00:34, 9485.05it/s] 19%|█▉        | 76250/400000 [00:08<00:33, 9598.59it/s] 19%|█▉        | 77232/400000 [00:08<00:33, 9663.90it/s] 20%|█▉        | 78200/400000 [00:08<00:33, 9649.11it/s] 20%|█▉        | 79166/400000 [00:08<00:33, 9522.78it/s] 20%|██        | 80136/400000 [00:08<00:33, 9573.69it/s] 20%|██        | 81095/400000 [00:08<00:34, 9227.45it/s] 21%|██        | 82037/400000 [00:08<00:34, 9282.24it/s] 21%|██        | 83000/400000 [00:09<00:33, 9381.02it/s] 21%|██        | 83940/400000 [00:09<00:33, 9325.19it/s] 21%|██        | 84912/400000 [00:09<00:33, 9438.93it/s] 21%|██▏       | 85858/400000 [00:09<00:34, 9238.59it/s] 22%|██▏       | 86815/400000 [00:09<00:33, 9334.02it/s] 22%|██▏       | 87797/400000 [00:09<00:32, 9472.77it/s] 22%|██▏       | 88746/400000 [00:09<00:33, 9274.17it/s] 22%|██▏       | 89701/400000 [00:09<00:33, 9352.78it/s] 23%|██▎       | 90686/400000 [00:09<00:32, 9495.37it/s] 23%|██▎       | 91638/400000 [00:10<00:32, 9485.05it/s] 23%|██▎       | 92592/400000 [00:10<00:32, 9499.61it/s] 23%|██▎       | 93543/400000 [00:10<00:32, 9447.76it/s] 24%|██▎       | 94544/400000 [00:10<00:31, 9607.10it/s] 24%|██▍       | 95506/400000 [00:10<00:31, 9560.54it/s] 24%|██▍       | 96463/400000 [00:10<00:32, 9339.63it/s] 24%|██▍       | 97399/400000 [00:10<00:32, 9257.74it/s] 25%|██▍       | 98327/400000 [00:10<00:33, 8968.49it/s] 25%|██▍       | 99294/400000 [00:10<00:32, 9167.02it/s] 25%|██▌       | 100221/400000 [00:10<00:32, 9195.44it/s] 25%|██▌       | 101181/400000 [00:11<00:32, 9312.24it/s] 26%|██▌       | 102148/400000 [00:11<00:31, 9416.47it/s] 26%|██▌       | 103092/400000 [00:11<00:32, 9112.40it/s] 26%|██▌       | 104007/400000 [00:11<00:32, 9039.96it/s] 26%|██▌       | 104914/400000 [00:11<00:33, 8931.51it/s] 26%|██▋       | 105867/400000 [00:11<00:32, 9102.23it/s] 27%|██▋       | 106780/400000 [00:11<00:32, 9065.94it/s] 27%|██▋       | 107689/400000 [00:11<00:33, 8666.06it/s] 27%|██▋       | 108621/400000 [00:11<00:32, 8852.18it/s] 27%|██▋       | 109511/400000 [00:11<00:33, 8698.68it/s] 28%|██▊       | 110453/400000 [00:12<00:32, 8903.09it/s] 28%|██▊       | 111348/400000 [00:12<00:32, 8850.86it/s] 28%|██▊       | 112236/400000 [00:12<00:32, 8740.28it/s] 28%|██▊       | 113200/400000 [00:12<00:31, 8990.90it/s] 29%|██▊       | 114105/400000 [00:12<00:31, 9007.77it/s] 29%|██▉       | 115015/400000 [00:12<00:31, 9034.54it/s] 29%|██▉       | 115995/400000 [00:12<00:30, 9250.95it/s] 29%|██▉       | 116923/400000 [00:12<00:31, 9107.68it/s] 29%|██▉       | 117886/400000 [00:12<00:30, 9257.57it/s] 30%|██▉       | 118822/400000 [00:12<00:30, 9287.56it/s] 30%|██▉       | 119789/400000 [00:13<00:29, 9397.42it/s] 30%|███       | 120731/400000 [00:13<00:29, 9310.11it/s] 30%|███       | 121664/400000 [00:13<00:30, 9212.21it/s] 31%|███       | 122610/400000 [00:13<00:29, 9283.05it/s] 31%|███       | 123590/400000 [00:13<00:29, 9430.66it/s] 31%|███       | 124549/400000 [00:13<00:29, 9476.51it/s] 31%|███▏      | 125533/400000 [00:13<00:28, 9580.32it/s] 32%|███▏      | 126492/400000 [00:13<00:29, 9263.08it/s] 32%|███▏      | 127422/400000 [00:13<00:29, 9243.84it/s] 32%|███▏      | 128386/400000 [00:14<00:29, 9357.49it/s] 32%|███▏      | 129381/400000 [00:14<00:28, 9525.55it/s] 33%|███▎      | 130336/400000 [00:14<00:29, 9239.60it/s] 33%|███▎      | 131269/400000 [00:14<00:29, 9265.36it/s] 33%|███▎      | 132255/400000 [00:14<00:28, 9434.87it/s] 33%|███▎      | 133221/400000 [00:14<00:28, 9500.69it/s] 34%|███▎      | 134185/400000 [00:14<00:27, 9541.38it/s] 34%|███▍      | 135178/400000 [00:14<00:27, 9652.82it/s] 34%|███▍      | 136145/400000 [00:14<00:28, 9308.88it/s] 34%|███▍      | 137080/400000 [00:14<00:28, 9217.92it/s] 35%|███▍      | 138005/400000 [00:15<00:28, 9112.75it/s] 35%|███▍      | 138982/400000 [00:15<00:28, 9299.33it/s] 35%|███▍      | 139918/400000 [00:15<00:27, 9314.39it/s] 35%|███▌      | 140855/400000 [00:15<00:27, 9329.86it/s] 35%|███▌      | 141803/400000 [00:15<00:27, 9374.27it/s] 36%|███▌      | 142742/400000 [00:15<00:28, 9119.78it/s] 36%|███▌      | 143657/400000 [00:15<00:28, 9094.68it/s] 36%|███▌      | 144568/400000 [00:15<00:29, 8725.87it/s] 36%|███▋      | 145461/400000 [00:15<00:28, 8785.01it/s] 37%|███▋      | 146346/400000 [00:15<00:28, 8801.80it/s] 37%|███▋      | 147230/400000 [00:16<00:28, 8812.45it/s] 37%|███▋      | 148113/400000 [00:16<00:28, 8695.29it/s] 37%|███▋      | 148984/400000 [00:16<00:29, 8502.33it/s] 37%|███▋      | 149844/400000 [00:16<00:29, 8528.52it/s] 38%|███▊      | 150742/400000 [00:16<00:28, 8656.66it/s] 38%|███▊      | 151701/400000 [00:16<00:27, 8916.17it/s] 38%|███▊      | 152643/400000 [00:16<00:27, 9059.76it/s] 38%|███▊      | 153552/400000 [00:16<00:27, 9062.57it/s] 39%|███▊      | 154461/400000 [00:16<00:27, 9047.56it/s] 39%|███▉      | 155398/400000 [00:16<00:26, 9140.20it/s] 39%|███▉      | 156357/400000 [00:17<00:26, 9269.99it/s] 39%|███▉      | 157286/400000 [00:17<00:26, 9141.03it/s] 40%|███▉      | 158210/400000 [00:17<00:26, 9167.65it/s] 40%|███▉      | 159183/400000 [00:17<00:25, 9327.77it/s] 40%|████      | 160118/400000 [00:17<00:25, 9254.66it/s] 40%|████      | 161056/400000 [00:17<00:25, 9291.37it/s] 40%|████      | 161986/400000 [00:17<00:25, 9261.24it/s] 41%|████      | 162913/400000 [00:17<00:26, 9059.65it/s] 41%|████      | 163821/400000 [00:17<00:26, 9011.15it/s] 41%|████      | 164784/400000 [00:17<00:25, 9187.16it/s] 41%|████▏     | 165723/400000 [00:18<00:25, 9246.52it/s] 42%|████▏     | 166700/400000 [00:18<00:24, 9397.16it/s] 42%|████▏     | 167684/400000 [00:18<00:24, 9523.60it/s] 42%|████▏     | 168638/400000 [00:18<00:25, 9229.27it/s] 42%|████▏     | 169574/400000 [00:18<00:24, 9267.59it/s] 43%|████▎     | 170579/400000 [00:18<00:24, 9488.05it/s] 43%|████▎     | 171531/400000 [00:18<00:24, 9407.36it/s] 43%|████▎     | 172474/400000 [00:18<00:25, 9098.41it/s] 43%|████▎     | 173388/400000 [00:18<00:25, 9051.52it/s] 44%|████▎     | 174339/400000 [00:19<00:24, 9183.13it/s] 44%|████▍     | 175261/400000 [00:19<00:24, 9190.92it/s] 44%|████▍     | 176219/400000 [00:19<00:24, 9302.18it/s] 44%|████▍     | 177156/400000 [00:19<00:23, 9321.01it/s] 45%|████▍     | 178109/400000 [00:19<00:23, 9381.68it/s] 45%|████▍     | 179068/400000 [00:19<00:23, 9442.94it/s] 45%|████▌     | 180064/400000 [00:19<00:22, 9589.63it/s] 45%|████▌     | 181024/400000 [00:19<00:22, 9580.74it/s] 45%|████▌     | 181983/400000 [00:19<00:22, 9540.55it/s] 46%|████▌     | 182942/400000 [00:19<00:22, 9554.19it/s] 46%|████▌     | 183898/400000 [00:20<00:22, 9539.98it/s] 46%|████▌     | 184882/400000 [00:20<00:22, 9627.62it/s] 46%|████▋     | 185846/400000 [00:20<00:22, 9626.35it/s] 47%|████▋     | 186809/400000 [00:20<00:22, 9566.22it/s] 47%|████▋     | 187766/400000 [00:20<00:22, 9378.78it/s] 47%|████▋     | 188705/400000 [00:20<00:23, 9105.01it/s] 47%|████▋     | 189656/400000 [00:20<00:22, 9222.49it/s] 48%|████▊     | 190651/400000 [00:20<00:22, 9426.35it/s] 48%|████▊     | 191618/400000 [00:20<00:21, 9495.49it/s] 48%|████▊     | 192570/400000 [00:20<00:21, 9499.78it/s] 48%|████▊     | 193561/400000 [00:21<00:21, 9618.21it/s] 49%|████▊     | 194551/400000 [00:21<00:21, 9699.97it/s] 49%|████▉     | 195523/400000 [00:21<00:21, 9538.66it/s] 49%|████▉     | 196479/400000 [00:21<00:21, 9413.93it/s] 49%|████▉     | 197427/400000 [00:21<00:21, 9431.66it/s] 50%|████▉     | 198372/400000 [00:21<00:21, 9376.54it/s] 50%|████▉     | 199325/400000 [00:21<00:21, 9421.10it/s] 50%|█████     | 200268/400000 [00:21<00:21, 9420.93it/s] 50%|█████     | 201211/400000 [00:21<00:21, 9255.73it/s] 51%|█████     | 202138/400000 [00:21<00:22, 8896.94it/s] 51%|█████     | 203032/400000 [00:22<00:22, 8754.61it/s] 51%|█████     | 203911/400000 [00:22<00:23, 8462.98it/s] 51%|█████     | 204762/400000 [00:22<00:24, 8132.14it/s] 51%|█████▏    | 205581/400000 [00:22<00:24, 7947.76it/s] 52%|█████▏    | 206381/400000 [00:22<00:24, 7961.22it/s] 52%|█████▏    | 207254/400000 [00:22<00:23, 8175.30it/s] 52%|█████▏    | 208195/400000 [00:22<00:22, 8509.49it/s] 52%|█████▏    | 209137/400000 [00:22<00:21, 8763.41it/s] 53%|█████▎    | 210046/400000 [00:22<00:21, 8858.17it/s] 53%|█████▎    | 210937/400000 [00:23<00:21, 8841.97it/s] 53%|█████▎    | 211894/400000 [00:23<00:20, 9048.11it/s] 53%|█████▎    | 212855/400000 [00:23<00:20, 9209.00it/s] 53%|█████▎    | 213819/400000 [00:23<00:19, 9333.22it/s] 54%|█████▎    | 214755/400000 [00:23<00:19, 9270.71it/s] 54%|█████▍    | 215717/400000 [00:23<00:19, 9371.81it/s] 54%|█████▍    | 216660/400000 [00:23<00:19, 9388.95it/s] 54%|█████▍    | 217601/400000 [00:23<00:19, 9321.98it/s] 55%|█████▍    | 218535/400000 [00:23<00:20, 8824.01it/s] 55%|█████▍    | 219424/400000 [00:23<00:20, 8634.01it/s] 55%|█████▌    | 220340/400000 [00:24<00:20, 8783.67it/s] 55%|█████▌    | 221223/400000 [00:24<00:20, 8791.64it/s] 56%|█████▌    | 222106/400000 [00:24<00:20, 8609.26it/s] 56%|█████▌    | 223006/400000 [00:24<00:20, 8721.71it/s] 56%|█████▌    | 223925/400000 [00:24<00:19, 8856.50it/s] 56%|█████▌    | 224921/400000 [00:24<00:19, 9158.01it/s] 56%|█████▋    | 225921/400000 [00:24<00:18, 9393.15it/s] 57%|█████▋    | 226928/400000 [00:24<00:18, 9585.12it/s] 57%|█████▋    | 227930/400000 [00:24<00:17, 9710.74it/s] 57%|█████▋    | 228905/400000 [00:24<00:18, 9504.29it/s] 57%|█████▋    | 229860/400000 [00:25<00:17, 9496.00it/s] 58%|█████▊    | 230812/400000 [00:25<00:18, 9324.85it/s] 58%|█████▊    | 231747/400000 [00:25<00:18, 9033.41it/s] 58%|█████▊    | 232660/400000 [00:25<00:18, 9060.97it/s] 58%|█████▊    | 233569/400000 [00:25<00:18, 8918.64it/s] 59%|█████▊    | 234488/400000 [00:25<00:18, 8994.19it/s] 59%|█████▉    | 235471/400000 [00:25<00:17, 9228.93it/s] 59%|█████▉    | 236439/400000 [00:25<00:17, 9357.88it/s] 59%|█████▉    | 237425/400000 [00:25<00:17, 9502.26it/s] 60%|█████▉    | 238378/400000 [00:25<00:17, 9445.23it/s] 60%|█████▉    | 239346/400000 [00:26<00:16, 9512.35it/s] 60%|██████    | 240318/400000 [00:26<00:16, 9571.22it/s] 60%|██████    | 241277/400000 [00:26<00:16, 9571.26it/s] 61%|██████    | 242235/400000 [00:26<00:16, 9343.92it/s] 61%|██████    | 243172/400000 [00:26<00:17, 8956.46it/s] 61%|██████    | 244073/400000 [00:26<00:17, 8792.64it/s] 61%|██████    | 244957/400000 [00:26<00:17, 8684.72it/s] 61%|██████▏   | 245829/400000 [00:26<00:17, 8619.91it/s] 62%|██████▏   | 246775/400000 [00:26<00:17, 8853.94it/s] 62%|██████▏   | 247676/400000 [00:27<00:17, 8900.06it/s] 62%|██████▏   | 248569/400000 [00:27<00:17, 8832.23it/s] 62%|██████▏   | 249454/400000 [00:27<00:17, 8471.26it/s] 63%|██████▎   | 250347/400000 [00:27<00:17, 8602.96it/s] 63%|██████▎   | 251267/400000 [00:27<00:16, 8773.03it/s] 63%|██████▎   | 252183/400000 [00:27<00:16, 8884.79it/s] 63%|██████▎   | 253075/400000 [00:27<00:16, 8889.72it/s] 63%|██████▎   | 253966/400000 [00:27<00:17, 8504.94it/s] 64%|██████▎   | 254842/400000 [00:27<00:16, 8578.41it/s] 64%|██████▍   | 255704/400000 [00:27<00:16, 8560.34it/s] 64%|██████▍   | 256563/400000 [00:28<00:16, 8450.00it/s] 64%|██████▍   | 257411/400000 [00:28<00:16, 8405.24it/s] 65%|██████▍   | 258382/400000 [00:28<00:16, 8756.22it/s] 65%|██████▍   | 259374/400000 [00:28<00:15, 9071.98it/s] 65%|██████▌   | 260362/400000 [00:28<00:15, 9299.09it/s] 65%|██████▌   | 261298/400000 [00:28<00:15, 9163.81it/s] 66%|██████▌   | 262289/400000 [00:28<00:14, 9375.15it/s] 66%|██████▌   | 263248/400000 [00:28<00:14, 9438.58it/s] 66%|██████▌   | 264196/400000 [00:28<00:14, 9271.97it/s] 66%|██████▋   | 265127/400000 [00:28<00:14, 9226.00it/s] 67%|██████▋   | 266052/400000 [00:29<00:14, 9225.93it/s] 67%|██████▋   | 267022/400000 [00:29<00:14, 9361.58it/s] 67%|██████▋   | 267968/400000 [00:29<00:14, 9390.61it/s] 67%|██████▋   | 268934/400000 [00:29<00:13, 9467.84it/s] 67%|██████▋   | 269916/400000 [00:29<00:13, 9570.23it/s] 68%|██████▊   | 270874/400000 [00:29<00:13, 9505.49it/s] 68%|██████▊   | 271826/400000 [00:29<00:13, 9481.69it/s] 68%|██████▊   | 272818/400000 [00:29<00:13, 9608.70it/s] 68%|██████▊   | 273811/400000 [00:29<00:13, 9702.64it/s] 69%|██████▊   | 274783/400000 [00:30<00:13, 9543.10it/s] 69%|██████▉   | 275739/400000 [00:30<00:13, 9335.10it/s] 69%|██████▉   | 276694/400000 [00:30<00:13, 9397.43it/s] 69%|██████▉   | 277636/400000 [00:30<00:13, 9355.68it/s] 70%|██████▉   | 278573/400000 [00:30<00:13, 9183.86it/s] 70%|██████▉   | 279493/400000 [00:30<00:13, 8901.94it/s] 70%|███████   | 280387/400000 [00:30<00:13, 8809.84it/s] 70%|███████   | 281376/400000 [00:30<00:13, 9106.33it/s] 71%|███████   | 282375/400000 [00:30<00:12, 9354.31it/s] 71%|███████   | 283345/400000 [00:30<00:12, 9454.90it/s] 71%|███████   | 284338/400000 [00:31<00:12, 9589.88it/s] 71%|███████▏  | 285300/400000 [00:31<00:12, 9268.95it/s] 72%|███████▏  | 286232/400000 [00:31<00:12, 9260.09it/s] 72%|███████▏  | 287226/400000 [00:31<00:11, 9453.99it/s] 72%|███████▏  | 288205/400000 [00:31<00:11, 9550.26it/s] 72%|███████▏  | 289163/400000 [00:31<00:11, 9407.82it/s] 73%|███████▎  | 290106/400000 [00:31<00:11, 9338.48it/s] 73%|███████▎  | 291096/400000 [00:31<00:11, 9493.94it/s] 73%|███████▎  | 292091/400000 [00:31<00:11, 9624.79it/s] 73%|███████▎  | 293079/400000 [00:31<00:11, 9697.38it/s] 74%|███████▎  | 294051/400000 [00:32<00:11, 9631.45it/s] 74%|███████▍  | 295016/400000 [00:32<00:11, 9176.88it/s] 74%|███████▍  | 295939/400000 [00:32<00:11, 9115.40it/s] 74%|███████▍  | 296855/400000 [00:32<00:11, 8898.35it/s] 74%|███████▍  | 297749/400000 [00:32<00:11, 8840.10it/s] 75%|███████▍  | 298718/400000 [00:32<00:11, 9076.55it/s] 75%|███████▍  | 299654/400000 [00:32<00:10, 9158.23it/s] 75%|███████▌  | 300655/400000 [00:32<00:10, 9397.61it/s] 75%|███████▌  | 301649/400000 [00:32<00:10, 9553.95it/s] 76%|███████▌  | 302625/400000 [00:32<00:10, 9612.28it/s] 76%|███████▌  | 303592/400000 [00:33<00:10, 9627.78it/s] 76%|███████▌  | 304557/400000 [00:33<00:10, 9534.77it/s] 76%|███████▋  | 305562/400000 [00:33<00:09, 9681.20it/s] 77%|███████▋  | 306549/400000 [00:33<00:09, 9737.00it/s] 77%|███████▋  | 307542/400000 [00:33<00:09, 9793.61it/s] 77%|███████▋  | 308523/400000 [00:33<00:09, 9764.89it/s] 77%|███████▋  | 309501/400000 [00:33<00:09, 9457.09it/s] 78%|███████▊  | 310450/400000 [00:33<00:09, 9154.69it/s] 78%|███████▊  | 311408/400000 [00:33<00:09, 9274.69it/s] 78%|███████▊  | 312401/400000 [00:34<00:09, 9460.72it/s] 78%|███████▊  | 313368/400000 [00:34<00:09, 9521.39it/s] 79%|███████▊  | 314323/400000 [00:34<00:09, 9286.14it/s] 79%|███████▉  | 315255/400000 [00:34<00:09, 9089.35it/s] 79%|███████▉  | 316203/400000 [00:34<00:09, 9202.25it/s] 79%|███████▉  | 317203/400000 [00:34<00:08, 9425.92it/s] 80%|███████▉  | 318199/400000 [00:34<00:08, 9577.46it/s] 80%|███████▉  | 319160/400000 [00:34<00:08, 9321.31it/s] 80%|████████  | 320120/400000 [00:34<00:08, 9402.18it/s] 80%|████████  | 321097/400000 [00:34<00:08, 9508.05it/s] 81%|████████  | 322093/400000 [00:35<00:08, 9636.77it/s] 81%|████████  | 323082/400000 [00:35<00:07, 9708.89it/s] 81%|████████  | 324055/400000 [00:35<00:07, 9525.66it/s] 81%|████████▏ | 325033/400000 [00:35<00:07, 9600.22it/s] 81%|████████▏ | 325995/400000 [00:35<00:07, 9356.43it/s] 82%|████████▏ | 326933/400000 [00:35<00:07, 9304.27it/s] 82%|████████▏ | 327925/400000 [00:35<00:07, 9477.93it/s] 82%|████████▏ | 328875/400000 [00:35<00:07, 9348.05it/s] 82%|████████▏ | 329875/400000 [00:35<00:07, 9533.18it/s] 83%|████████▎ | 330851/400000 [00:35<00:07, 9598.58it/s] 83%|████████▎ | 331848/400000 [00:36<00:07, 9705.08it/s] 83%|████████▎ | 332846/400000 [00:36<00:06, 9784.59it/s] 83%|████████▎ | 333826/400000 [00:36<00:06, 9521.64it/s] 84%|████████▎ | 334781/400000 [00:36<00:07, 9217.79it/s] 84%|████████▍ | 335756/400000 [00:36<00:06, 9368.99it/s] 84%|████████▍ | 336737/400000 [00:36<00:06, 9495.01it/s] 84%|████████▍ | 337703/400000 [00:36<00:06, 9541.67it/s] 85%|████████▍ | 338660/400000 [00:36<00:06, 9402.58it/s] 85%|████████▍ | 339653/400000 [00:36<00:06, 9552.98it/s] 85%|████████▌ | 340641/400000 [00:36<00:06, 9646.96it/s] 85%|████████▌ | 341608/400000 [00:37<00:06, 9497.26it/s] 86%|████████▌ | 342560/400000 [00:37<00:06, 9200.04it/s] 86%|████████▌ | 343522/400000 [00:37<00:06, 9321.90it/s] 86%|████████▌ | 344471/400000 [00:37<00:05, 9371.56it/s] 86%|████████▋ | 345411/400000 [00:37<00:05, 9216.22it/s] 87%|████████▋ | 346335/400000 [00:37<00:05, 9153.36it/s] 87%|████████▋ | 347262/400000 [00:37<00:05, 9186.31it/s] 87%|████████▋ | 348182/400000 [00:37<00:05, 8968.56it/s] 87%|████████▋ | 349148/400000 [00:37<00:05, 9163.58it/s] 88%|████████▊ | 350129/400000 [00:38<00:05, 9347.06it/s] 88%|████████▊ | 351119/400000 [00:38<00:05, 9504.06it/s] 88%|████████▊ | 352074/400000 [00:38<00:05, 9515.91it/s] 88%|████████▊ | 353052/400000 [00:38<00:04, 9593.42it/s] 89%|████████▊ | 354037/400000 [00:38<00:04, 9666.04it/s] 89%|████████▉ | 355030/400000 [00:38<00:04, 9741.22it/s] 89%|████████▉ | 356026/400000 [00:38<00:04, 9803.98it/s] 89%|████████▉ | 357008/400000 [00:38<00:04, 9564.37it/s] 89%|████████▉ | 357967/400000 [00:38<00:04, 9278.39it/s] 90%|████████▉ | 358905/400000 [00:38<00:04, 9308.41it/s] 90%|████████▉ | 359892/400000 [00:39<00:04, 9467.96it/s] 90%|█████████ | 360871/400000 [00:39<00:04, 9560.12it/s] 90%|█████████ | 361829/400000 [00:39<00:04, 9523.11it/s] 91%|█████████ | 362783/400000 [00:39<00:03, 9512.34it/s] 91%|█████████ | 363759/400000 [00:39<00:03, 9584.43it/s] 91%|█████████ | 364758/400000 [00:39<00:03, 9700.71it/s] 91%|█████████▏| 365743/400000 [00:39<00:03, 9743.01it/s] 92%|█████████▏| 366718/400000 [00:39<00:03, 9490.44it/s] 92%|█████████▏| 367676/400000 [00:39<00:03, 9514.62it/s] 92%|█████████▏| 368644/400000 [00:39<00:03, 9561.67it/s] 92%|█████████▏| 369602/400000 [00:40<00:03, 9523.17it/s] 93%|█████████▎| 370556/400000 [00:40<00:03, 9417.15it/s] 93%|█████████▎| 371516/400000 [00:40<00:03, 9470.97it/s] 93%|█████████▎| 372464/400000 [00:40<00:02, 9260.82it/s] 93%|█████████▎| 373392/400000 [00:40<00:02, 9036.09it/s] 94%|█████████▎| 374321/400000 [00:40<00:02, 9110.20it/s] 94%|█████████▍| 375261/400000 [00:40<00:02, 9193.54it/s] 94%|█████████▍| 376198/400000 [00:40<00:02, 9243.14it/s] 94%|█████████▍| 377157/400000 [00:40<00:02, 9344.24it/s] 95%|█████████▍| 378114/400000 [00:40<00:02, 9409.23it/s] 95%|█████████▍| 379103/400000 [00:41<00:02, 9546.10it/s] 95%|█████████▌| 380083/400000 [00:41<00:02, 9620.66it/s] 95%|█████████▌| 381051/400000 [00:41<00:01, 9636.90it/s] 96%|█████████▌| 382016/400000 [00:41<00:01, 9602.34it/s] 96%|█████████▌| 382977/400000 [00:41<00:01, 9551.33it/s] 96%|█████████▌| 383975/400000 [00:41<00:01, 9674.81it/s] 96%|█████████▌| 384983/400000 [00:41<00:01, 9792.63it/s] 96%|█████████▋| 385964/400000 [00:41<00:01, 9752.80it/s] 97%|█████████▋| 386940/400000 [00:41<00:01, 9707.33it/s] 97%|█████████▋| 387912/400000 [00:41<00:01, 9622.51it/s] 97%|█████████▋| 388875/400000 [00:42<00:01, 9345.70it/s] 97%|█████████▋| 389812/400000 [00:42<00:01, 9274.04it/s] 98%|█████████▊| 390741/400000 [00:42<00:01, 9101.94it/s] 98%|█████████▊| 391731/400000 [00:42<00:00, 9326.42it/s] 98%|█████████▊| 392703/400000 [00:42<00:00, 9440.82it/s] 98%|█████████▊| 393650/400000 [00:42<00:00, 9425.76it/s] 99%|█████████▊| 394595/400000 [00:42<00:00, 9318.58it/s] 99%|█████████▉| 395529/400000 [00:42<00:00, 9020.84it/s] 99%|█████████▉| 396466/400000 [00:42<00:00, 9121.06it/s] 99%|█████████▉| 397394/400000 [00:43<00:00, 9166.36it/s]100%|█████████▉| 398360/400000 [00:43<00:00, 9308.92it/s]100%|█████████▉| 399345/400000 [00:43<00:00, 9463.76it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9239.39it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0eeb982be0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011682053151219358 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011237791948095214 	 Accuracy: 51

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
2020-05-12 18:23:58.509669: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 18:23:58.513398: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 18:23:58.513584: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e04ef682a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 18:23:58.513632: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0e9eec0128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9196 - accuracy: 0.4835 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6615 - accuracy: 0.5003
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6316 - accuracy: 0.5023
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6620 - accuracy: 0.5003
11000/25000 [============>.................] - ETA: 4s - loss: 7.6764 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 4s - loss: 7.6564 - accuracy: 0.5007
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 3s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6684 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0e4c06d4e0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0e4c06dda0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 937ms/step - loss: 1.6089 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.5328 - val_crf_viterbi_accuracy: 0.1333

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

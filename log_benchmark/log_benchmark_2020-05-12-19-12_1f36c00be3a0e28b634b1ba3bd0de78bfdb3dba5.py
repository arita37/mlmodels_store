
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa878c88fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 19:12:20.596176
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 19:12:20.602067
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 19:12:20.606817
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 19:12:20.610599
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa884a52470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355244.3438
Epoch 2/10

1/1 [==============================] - 0s 115ms/step - loss: 270389.0938
Epoch 3/10

1/1 [==============================] - 0s 101ms/step - loss: 179415.8750
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 110193.1094
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 65908.1953
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 40539.6680
Epoch 7/10

1/1 [==============================] - 0s 108ms/step - loss: 26397.0117
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 18059.1426
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 12913.5771
Epoch 10/10

1/1 [==============================] - 0s 112ms/step - loss: 9614.1318

  #### Inference Need return ypred, ytrue ######################### 
[[-1.1180248e+00 -1.8523285e+00 -6.2343514e-01 -1.0642970e+00
  -6.2570417e-01 -9.4714582e-02  4.6365184e-01  1.2657497e+00
  -4.1078001e-01  1.1859230e+00 -5.6070983e-01 -6.8757796e-01
  -1.2767169e+00 -1.4009036e+00 -1.6708818e-01  2.2436279e-01
  -4.2936254e-01 -8.8479590e-01  1.0975803e+00  9.6678376e-01
   2.2182162e-01 -8.6290962e-01 -1.1084647e+00 -3.1505701e-01
   4.7017384e-01 -1.3465326e+00 -9.2442667e-01  1.3638735e-02
   6.0109362e-02  4.6101809e-01 -6.6985708e-01  1.4320750e+00
  -9.3567586e-01  7.7105701e-02  3.6923632e-01  1.9463389e-01
   3.9207244e-01  2.6722926e-01  1.2971749e+00  1.4460980e+00
   1.5553420e+00  1.0062163e+00  1.8694668e-01 -3.6919606e-01
  -6.4944065e-01  7.2485918e-01  3.2970583e-01  1.5505168e-01
  -5.0542790e-01 -4.2294735e-01 -3.6908293e-01 -1.9869974e+00
   1.3839047e+00 -1.3625942e+00 -1.2378182e+00 -1.5186051e+00
   9.9857301e-01 -8.8814145e-01  7.8925776e-01  4.2641315e-01
   2.6501870e-01  1.6064190e+00 -6.1346799e-01 -1.7229235e-01
   1.6255127e+00  1.6708219e+00  1.2762645e-01  6.8859053e-01
   4.1591543e-01 -5.7034099e-01 -7.6332070e-02  1.6104572e+00
   3.0034971e-01 -1.3164423e+00 -1.2094467e+00 -8.0741084e-01
  -4.2787588e-01 -2.5289431e-01 -2.3639679e-02  5.4377812e-01
  -3.6515215e-01 -4.8832637e-01  4.6349168e-02  5.0975472e-01
  -2.9534876e-01 -1.2194290e+00 -2.8864190e-01  1.6661242e-01
  -9.0091944e-01 -3.0690351e-01 -3.3743829e-02  8.8392854e-01
  -1.3315556e+00 -3.4506738e-02 -9.7344762e-01  1.2469435e+00
  -1.0078301e+00  6.3943046e-01 -1.5543455e-01 -1.3924541e+00
  -6.2179267e-01  5.1909316e-01  3.5437092e-02 -1.0141850e+00
  -6.1621740e-02  6.0682315e-01  4.4597995e-01 -8.9343417e-01
  -7.2653830e-02  1.8589222e-01  8.7225074e-01 -3.3019191e-01
   9.1429895e-01  3.5775650e-01 -2.4440461e-01 -9.8979890e-02
   4.7051001e-01  1.8300660e+00 -4.6234381e-01 -8.7253451e-03
   1.4270505e-01  6.6615663e+00  5.9434657e+00  6.3768015e+00
   8.1422710e+00  6.4487872e+00  6.1602793e+00  5.4310794e+00
   5.6288652e+00  6.1895747e+00  6.4152594e+00  6.4108310e+00
   6.3871412e+00  7.3266420e+00  6.0981531e+00  5.9840345e+00
   6.1501517e+00  7.7678232e+00  7.0787120e+00  5.9798503e+00
   6.4063926e+00  7.1528440e+00  7.5235457e+00  7.2183728e+00
   5.9559040e+00  6.8492413e+00  6.0156393e+00  7.0230436e+00
   6.8266296e+00  6.8086519e+00  6.2770739e+00  7.1630483e+00
   7.2584667e+00  6.3991547e+00  6.7135601e+00  6.2531023e+00
   6.4191151e+00  8.4539957e+00  7.7322683e+00  5.3983350e+00
   4.4963632e+00  6.5640273e+00  7.4075031e+00  6.6590309e+00
   6.9549313e+00  6.7458186e+00  7.1200137e+00  7.2145324e+00
   4.9518361e+00  5.2256546e+00  7.5482435e+00  5.5786767e+00
   5.7263503e+00  7.1746025e+00  6.9691777e+00  6.2807622e+00
   7.0317373e+00  6.2263150e+00  5.9063911e+00  6.0164680e+00
   2.5456924e+00  3.4605038e-01  5.2395195e-01  5.8803695e-01
   1.4320915e+00  2.2004580e-01  7.5965846e-01  1.0173867e+00
   2.6214933e-01  1.2077562e+00  4.2599559e-01  6.2961268e-01
   1.2725674e+00  1.5226800e+00  1.5689142e+00  9.3132770e-01
   1.5166104e-01  1.9879042e+00  8.5451597e-01  3.5389107e-01
   8.1327689e-01  9.2494088e-01  6.4514667e-01  1.5581410e+00
   5.5034006e-01  2.2640061e-01  7.3974580e-01  1.3688688e+00
   2.0186300e+00  1.1779908e+00  1.8887048e+00  2.5674963e+00
   9.0513587e-01  1.1892688e+00  1.2162817e+00  2.5748253e-01
   1.4912051e-01  1.2244544e+00  1.6228476e+00  1.2157130e+00
   3.9244562e-01  6.2531608e-01  5.8549821e-01  6.5158087e-01
   2.0025315e+00  1.2339960e+00  2.4010601e+00  4.1537762e-01
   1.6072705e+00  1.3441646e-01  1.8154128e+00  3.5690624e-01
   7.3987234e-01  8.9122880e-01  5.9846151e-01  2.9219615e-01
   4.8391563e-01  3.0907029e-01  3.4194815e-01  1.3790956e+00
   5.7446563e-01  2.4911880e-01  9.4340509e-01  8.9136380e-01
   6.0423082e-01  6.8918920e-01  2.5936718e+00  6.2115782e-01
   2.9405797e-01  5.1560354e-01  2.0688109e+00  1.7229788e+00
   2.3341098e+00  1.4903070e+00  2.0023527e+00  1.2138796e-01
   9.8354882e-01  1.4956053e+00  1.5184896e+00  1.2460179e+00
   5.2148014e-01  1.2678425e+00  4.8543227e-01  5.2750480e-01
   5.9948134e-01  5.1654446e-01  5.2285326e-01  7.2379929e-01
   7.5727087e-01  8.0829185e-01  2.2183900e+00  7.9808784e-01
   1.7783242e+00  7.4333155e-01  2.5467378e-01  6.9282913e-01
   1.9331261e+00  1.0057473e+00  1.7221688e+00  3.5378242e-01
   6.7897153e-01  8.6912298e-01  1.8788544e+00  1.3276513e+00
   2.0059240e+00  4.8660159e-01  1.0555161e+00  6.9719225e-01
   1.9587352e+00  5.2171224e-01  5.0165695e-01  1.5579653e-01
   1.0170327e+00  1.2507609e+00  4.5148063e-01  1.1649272e+00
   1.6339073e+00  5.7809567e-01  9.2265236e-01  5.4517972e-01
   1.4621031e-01  6.3429365e+00  7.8104768e+00  7.4056640e+00
   8.3115158e+00  7.6073685e+00  6.1510887e+00  6.5842834e+00
   7.9528332e+00  6.7271438e+00  6.5084200e+00  6.4089060e+00
   7.8432369e+00  8.5665731e+00  7.5468922e+00  7.1418157e+00
   6.8213377e+00  6.9559255e+00  5.5069408e+00  6.8306561e+00
   8.1571655e+00  6.4306016e+00  7.2264776e+00  7.3829174e+00
   8.8869028e+00  7.6239004e+00  5.9008026e+00  5.2134614e+00
   5.9215703e+00  6.5387383e+00  7.4011602e+00  6.4611006e+00
   7.4930153e+00  7.7634602e+00  5.8646464e+00  7.0072618e+00
   8.5031891e+00  6.8601990e+00  6.1090498e+00  7.2319503e+00
   6.4900126e+00  6.9558096e+00  7.3871813e+00  5.7434788e+00
   6.2601933e+00  6.9057851e+00  6.2918591e+00  5.5729332e+00
   7.5356407e+00  7.1605926e+00  7.2258563e+00  8.0538635e+00
   5.9813433e+00  8.3254948e+00  7.0276537e+00  7.3043170e+00
   7.0324650e+00  7.5219598e+00  6.2528648e+00  7.2459040e+00
  -2.8313143e+00 -6.7412577e+00  5.0366440e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 19:12:29.372393
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.3158
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 19:12:29.376791
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9106.57
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 19:12:29.380266
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.9169
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 19:12:29.383459
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -814.558
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140361198547408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140359988695728
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140359988696232
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140359988696736
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140359988697240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140359988697744

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa8723bf588> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.472147
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.443071
grad_step = 000002, loss = 0.425058
grad_step = 000003, loss = 0.408170
grad_step = 000004, loss = 0.390341
grad_step = 000005, loss = 0.373746
grad_step = 000006, loss = 0.364702
grad_step = 000007, loss = 0.362929
grad_step = 000008, loss = 0.353573
grad_step = 000009, loss = 0.341177
grad_step = 000010, loss = 0.331742
grad_step = 000011, loss = 0.325220
grad_step = 000012, loss = 0.319316
grad_step = 000013, loss = 0.312769
grad_step = 000014, loss = 0.305064
grad_step = 000015, loss = 0.296392
grad_step = 000016, loss = 0.287534
grad_step = 000017, loss = 0.279419
grad_step = 000018, loss = 0.272565
grad_step = 000019, loss = 0.266327
grad_step = 000020, loss = 0.259587
grad_step = 000021, loss = 0.251973
grad_step = 000022, loss = 0.244309
grad_step = 000023, loss = 0.237358
grad_step = 000024, loss = 0.230976
grad_step = 000025, loss = 0.224619
grad_step = 000026, loss = 0.217962
grad_step = 000027, loss = 0.211097
grad_step = 000028, loss = 0.204353
grad_step = 000029, loss = 0.197988
grad_step = 000030, loss = 0.191930
grad_step = 000031, loss = 0.185899
grad_step = 000032, loss = 0.179704
grad_step = 000033, loss = 0.173478
grad_step = 000034, loss = 0.167264
grad_step = 000035, loss = 0.161390
grad_step = 000036, loss = 0.156043
grad_step = 000037, loss = 0.150244
grad_step = 000038, loss = 0.144408
grad_step = 000039, loss = 0.138948
grad_step = 000040, loss = 0.133759
grad_step = 000041, loss = 0.128460
grad_step = 000042, loss = 0.123155
grad_step = 000043, loss = 0.118002
grad_step = 000044, loss = 0.113048
grad_step = 000045, loss = 0.108192
grad_step = 000046, loss = 0.103383
grad_step = 000047, loss = 0.098666
grad_step = 000048, loss = 0.094066
grad_step = 000049, loss = 0.089640
grad_step = 000050, loss = 0.085283
grad_step = 000051, loss = 0.080997
grad_step = 000052, loss = 0.076869
grad_step = 000053, loss = 0.072923
grad_step = 000054, loss = 0.069021
grad_step = 000055, loss = 0.065218
grad_step = 000056, loss = 0.061605
grad_step = 000057, loss = 0.058131
grad_step = 000058, loss = 0.054724
grad_step = 000059, loss = 0.051462
grad_step = 000060, loss = 0.048366
grad_step = 000061, loss = 0.045368
grad_step = 000062, loss = 0.042492
grad_step = 000063, loss = 0.039773
grad_step = 000064, loss = 0.037176
grad_step = 000065, loss = 0.034687
grad_step = 000066, loss = 0.032335
grad_step = 000067, loss = 0.030127
grad_step = 000068, loss = 0.028024
grad_step = 000069, loss = 0.026030
grad_step = 000070, loss = 0.024176
grad_step = 000071, loss = 0.022438
grad_step = 000072, loss = 0.020796
grad_step = 000073, loss = 0.019265
grad_step = 000074, loss = 0.017838
grad_step = 000075, loss = 0.016511
grad_step = 000076, loss = 0.015279
grad_step = 000077, loss = 0.014135
grad_step = 000078, loss = 0.013078
grad_step = 000079, loss = 0.012101
grad_step = 000080, loss = 0.011203
grad_step = 000081, loss = 0.010387
grad_step = 000082, loss = 0.009652
grad_step = 000083, loss = 0.008992
grad_step = 000084, loss = 0.008355
grad_step = 000085, loss = 0.007732
grad_step = 000086, loss = 0.007162
grad_step = 000087, loss = 0.006686
grad_step = 000088, loss = 0.006271
grad_step = 000089, loss = 0.005843
grad_step = 000090, loss = 0.005451
grad_step = 000091, loss = 0.005154
grad_step = 000092, loss = 0.004848
grad_step = 000093, loss = 0.004539
grad_step = 000094, loss = 0.004319
grad_step = 000095, loss = 0.004095
grad_step = 000096, loss = 0.003864
grad_step = 000097, loss = 0.003700
grad_step = 000098, loss = 0.003529
grad_step = 000099, loss = 0.003360
grad_step = 000100, loss = 0.003243
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003112
grad_step = 000102, loss = 0.002985
grad_step = 000103, loss = 0.002903
grad_step = 000104, loss = 0.002808
grad_step = 000105, loss = 0.002711
grad_step = 000106, loss = 0.002651
grad_step = 000107, loss = 0.002585
grad_step = 000108, loss = 0.002513
grad_step = 000109, loss = 0.002469
grad_step = 000110, loss = 0.002424
grad_step = 000111, loss = 0.002369
grad_step = 000112, loss = 0.002335
grad_step = 000113, loss = 0.002309
grad_step = 000114, loss = 0.002269
grad_step = 000115, loss = 0.002239
grad_step = 000116, loss = 0.002221
grad_step = 000117, loss = 0.002197
grad_step = 000118, loss = 0.002171
grad_step = 000119, loss = 0.002154
grad_step = 000120, loss = 0.002142
grad_step = 000121, loss = 0.002126
grad_step = 000122, loss = 0.002108
grad_step = 000123, loss = 0.002099
grad_step = 000124, loss = 0.002095
grad_step = 000125, loss = 0.002096
grad_step = 000126, loss = 0.002110
grad_step = 000127, loss = 0.002145
grad_step = 000128, loss = 0.002121
grad_step = 000129, loss = 0.002066
grad_step = 000130, loss = 0.002045
grad_step = 000131, loss = 0.002049
grad_step = 000132, loss = 0.002036
grad_step = 000133, loss = 0.002037
grad_step = 000134, loss = 0.002046
grad_step = 000135, loss = 0.002015
grad_step = 000136, loss = 0.001989
grad_step = 000137, loss = 0.002003
grad_step = 000138, loss = 0.002005
grad_step = 000139, loss = 0.001987
grad_step = 000140, loss = 0.001986
grad_step = 000141, loss = 0.001979
grad_step = 000142, loss = 0.001956
grad_step = 000143, loss = 0.001957
grad_step = 000144, loss = 0.001966
grad_step = 000145, loss = 0.001954
grad_step = 000146, loss = 0.001949
grad_step = 000147, loss = 0.001954
grad_step = 000148, loss = 0.001942
grad_step = 000149, loss = 0.001928
grad_step = 000150, loss = 0.001924
grad_step = 000151, loss = 0.001919
grad_step = 000152, loss = 0.001907
grad_step = 000153, loss = 0.001899
grad_step = 000154, loss = 0.001898
grad_step = 000155, loss = 0.001895
grad_step = 000156, loss = 0.001889
grad_step = 000157, loss = 0.001893
grad_step = 000158, loss = 0.001919
grad_step = 000159, loss = 0.001982
grad_step = 000160, loss = 0.002118
grad_step = 000161, loss = 0.002268
grad_step = 000162, loss = 0.002144
grad_step = 000163, loss = 0.001887
grad_step = 000164, loss = 0.001893
grad_step = 000165, loss = 0.002050
grad_step = 000166, loss = 0.001994
grad_step = 000167, loss = 0.001840
grad_step = 000168, loss = 0.001911
grad_step = 000169, loss = 0.001994
grad_step = 000170, loss = 0.001875
grad_step = 000171, loss = 0.001829
grad_step = 000172, loss = 0.001918
grad_step = 000173, loss = 0.001888
grad_step = 000174, loss = 0.001808
grad_step = 000175, loss = 0.001845
grad_step = 000176, loss = 0.001874
grad_step = 000177, loss = 0.001821
grad_step = 000178, loss = 0.001795
grad_step = 000179, loss = 0.001836
grad_step = 000180, loss = 0.001830
grad_step = 000181, loss = 0.001782
grad_step = 000182, loss = 0.001791
grad_step = 000183, loss = 0.001813
grad_step = 000184, loss = 0.001795
grad_step = 000185, loss = 0.001766
grad_step = 000186, loss = 0.001773
grad_step = 000187, loss = 0.001790
grad_step = 000188, loss = 0.001773
grad_step = 000189, loss = 0.001751
grad_step = 000190, loss = 0.001750
grad_step = 000191, loss = 0.001760
grad_step = 000192, loss = 0.001759
grad_step = 000193, loss = 0.001742
grad_step = 000194, loss = 0.001730
grad_step = 000195, loss = 0.001731
grad_step = 000196, loss = 0.001736
grad_step = 000197, loss = 0.001738
grad_step = 000198, loss = 0.001729
grad_step = 000199, loss = 0.001718
grad_step = 000200, loss = 0.001709
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001703
grad_step = 000202, loss = 0.001701
grad_step = 000203, loss = 0.001701
grad_step = 000204, loss = 0.001704
grad_step = 000205, loss = 0.001712
grad_step = 000206, loss = 0.001729
grad_step = 000207, loss = 0.001760
grad_step = 000208, loss = 0.001828
grad_step = 000209, loss = 0.001916
grad_step = 000210, loss = 0.002019
grad_step = 000211, loss = 0.001986
grad_step = 000212, loss = 0.001834
grad_step = 000213, loss = 0.001681
grad_step = 000214, loss = 0.001701
grad_step = 000215, loss = 0.001820
grad_step = 000216, loss = 0.001841
grad_step = 000217, loss = 0.001730
grad_step = 000218, loss = 0.001656
grad_step = 000219, loss = 0.001709
grad_step = 000220, loss = 0.001779
grad_step = 000221, loss = 0.001752
grad_step = 000222, loss = 0.001674
grad_step = 000223, loss = 0.001649
grad_step = 000224, loss = 0.001690
grad_step = 000225, loss = 0.001729
grad_step = 000226, loss = 0.001684
grad_step = 000227, loss = 0.001636
grad_step = 000228, loss = 0.001640
grad_step = 000229, loss = 0.001667
grad_step = 000230, loss = 0.001675
grad_step = 000231, loss = 0.001657
grad_step = 000232, loss = 0.001628
grad_step = 000233, loss = 0.001614
grad_step = 000234, loss = 0.001626
grad_step = 000235, loss = 0.001642
grad_step = 000236, loss = 0.001642
grad_step = 000237, loss = 0.001630
grad_step = 000238, loss = 0.001618
grad_step = 000239, loss = 0.001608
grad_step = 000240, loss = 0.001597
grad_step = 000241, loss = 0.001592
grad_step = 000242, loss = 0.001596
grad_step = 000243, loss = 0.001603
grad_step = 000244, loss = 0.001610
grad_step = 000245, loss = 0.001621
grad_step = 000246, loss = 0.001644
grad_step = 000247, loss = 0.001686
grad_step = 000248, loss = 0.001744
grad_step = 000249, loss = 0.001819
grad_step = 000250, loss = 0.001868
grad_step = 000251, loss = 0.001824
grad_step = 000252, loss = 0.001706
grad_step = 000253, loss = 0.001584
grad_step = 000254, loss = 0.001581
grad_step = 000255, loss = 0.001663
grad_step = 000256, loss = 0.001704
grad_step = 000257, loss = 0.001658
grad_step = 000258, loss = 0.001585
grad_step = 000259, loss = 0.001559
grad_step = 000260, loss = 0.001585
grad_step = 000261, loss = 0.001628
grad_step = 000262, loss = 0.001641
grad_step = 000263, loss = 0.001608
grad_step = 000264, loss = 0.001558
grad_step = 000265, loss = 0.001540
grad_step = 000266, loss = 0.001558
grad_step = 000267, loss = 0.001583
grad_step = 000268, loss = 0.001590
grad_step = 000269, loss = 0.001576
grad_step = 000270, loss = 0.001550
grad_step = 000271, loss = 0.001530
grad_step = 000272, loss = 0.001525
grad_step = 000273, loss = 0.001537
grad_step = 000274, loss = 0.001551
grad_step = 000275, loss = 0.001557
grad_step = 000276, loss = 0.001554
grad_step = 000277, loss = 0.001545
grad_step = 000278, loss = 0.001532
grad_step = 000279, loss = 0.001518
grad_step = 000280, loss = 0.001509
grad_step = 000281, loss = 0.001504
grad_step = 000282, loss = 0.001503
grad_step = 000283, loss = 0.001504
grad_step = 000284, loss = 0.001507
grad_step = 000285, loss = 0.001513
grad_step = 000286, loss = 0.001524
grad_step = 000287, loss = 0.001543
grad_step = 000288, loss = 0.001574
grad_step = 000289, loss = 0.001622
grad_step = 000290, loss = 0.001689
grad_step = 000291, loss = 0.001748
grad_step = 000292, loss = 0.001745
grad_step = 000293, loss = 0.001665
grad_step = 000294, loss = 0.001534
grad_step = 000295, loss = 0.001482
grad_step = 000296, loss = 0.001533
grad_step = 000297, loss = 0.001595
grad_step = 000298, loss = 0.001590
grad_step = 000299, loss = 0.001519
grad_step = 000300, loss = 0.001472
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001482
grad_step = 000302, loss = 0.001525
grad_step = 000303, loss = 0.001555
grad_step = 000304, loss = 0.001549
grad_step = 000305, loss = 0.001508
grad_step = 000306, loss = 0.001468
grad_step = 000307, loss = 0.001454
grad_step = 000308, loss = 0.001468
grad_step = 000309, loss = 0.001491
grad_step = 000310, loss = 0.001499
grad_step = 000311, loss = 0.001486
grad_step = 000312, loss = 0.001464
grad_step = 000313, loss = 0.001447
grad_step = 000314, loss = 0.001443
grad_step = 000315, loss = 0.001447
grad_step = 000316, loss = 0.001454
grad_step = 000317, loss = 0.001460
grad_step = 000318, loss = 0.001462
grad_step = 000319, loss = 0.001460
grad_step = 000320, loss = 0.001449
grad_step = 000321, loss = 0.001438
grad_step = 000322, loss = 0.001428
grad_step = 000323, loss = 0.001423
grad_step = 000324, loss = 0.001420
grad_step = 000325, loss = 0.001419
grad_step = 000326, loss = 0.001417
grad_step = 000327, loss = 0.001416
grad_step = 000328, loss = 0.001418
grad_step = 000329, loss = 0.001422
grad_step = 000330, loss = 0.001429
grad_step = 000331, loss = 0.001440
grad_step = 000332, loss = 0.001459
grad_step = 000333, loss = 0.001498
grad_step = 000334, loss = 0.001568
grad_step = 000335, loss = 0.001666
grad_step = 000336, loss = 0.001795
grad_step = 000337, loss = 0.001788
grad_step = 000338, loss = 0.001667
grad_step = 000339, loss = 0.001465
grad_step = 000340, loss = 0.001411
grad_step = 000341, loss = 0.001502
grad_step = 000342, loss = 0.001556
grad_step = 000343, loss = 0.001508
grad_step = 000344, loss = 0.001437
grad_step = 000345, loss = 0.001417
grad_step = 000346, loss = 0.001443
grad_step = 000347, loss = 0.001473
grad_step = 000348, loss = 0.001464
grad_step = 000349, loss = 0.001423
grad_step = 000350, loss = 0.001391
grad_step = 000351, loss = 0.001405
grad_step = 000352, loss = 0.001431
grad_step = 000353, loss = 0.001420
grad_step = 000354, loss = 0.001387
grad_step = 000355, loss = 0.001371
grad_step = 000356, loss = 0.001387
grad_step = 000357, loss = 0.001404
grad_step = 000358, loss = 0.001390
grad_step = 000359, loss = 0.001361
grad_step = 000360, loss = 0.001352
grad_step = 000361, loss = 0.001366
grad_step = 000362, loss = 0.001377
grad_step = 000363, loss = 0.001372
grad_step = 000364, loss = 0.001358
grad_step = 000365, loss = 0.001346
grad_step = 000366, loss = 0.001342
grad_step = 000367, loss = 0.001346
grad_step = 000368, loss = 0.001347
grad_step = 000369, loss = 0.001342
grad_step = 000370, loss = 0.001335
grad_step = 000371, loss = 0.001335
grad_step = 000372, loss = 0.001340
grad_step = 000373, loss = 0.001341
grad_step = 000374, loss = 0.001339
grad_step = 000375, loss = 0.001334
grad_step = 000376, loss = 0.001331
grad_step = 000377, loss = 0.001332
grad_step = 000378, loss = 0.001338
grad_step = 000379, loss = 0.001347
grad_step = 000380, loss = 0.001358
grad_step = 000381, loss = 0.001372
grad_step = 000382, loss = 0.001397
grad_step = 000383, loss = 0.001423
grad_step = 000384, loss = 0.001463
grad_step = 000385, loss = 0.001454
grad_step = 000386, loss = 0.001430
grad_step = 000387, loss = 0.001368
grad_step = 000388, loss = 0.001324
grad_step = 000389, loss = 0.001312
grad_step = 000390, loss = 0.001316
grad_step = 000391, loss = 0.001322
grad_step = 000392, loss = 0.001323
grad_step = 000393, loss = 0.001332
grad_step = 000394, loss = 0.001346
grad_step = 000395, loss = 0.001338
grad_step = 000396, loss = 0.001314
grad_step = 000397, loss = 0.001287
grad_step = 000398, loss = 0.001278
grad_step = 000399, loss = 0.001287
grad_step = 000400, loss = 0.001296
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001298
grad_step = 000402, loss = 0.001290
grad_step = 000403, loss = 0.001284
grad_step = 000404, loss = 0.001284
grad_step = 000405, loss = 0.001286
grad_step = 000406, loss = 0.001290
grad_step = 000407, loss = 0.001287
grad_step = 000408, loss = 0.001282
grad_step = 000409, loss = 0.001273
grad_step = 000410, loss = 0.001266
grad_step = 000411, loss = 0.001262
grad_step = 000412, loss = 0.001262
grad_step = 000413, loss = 0.001263
grad_step = 000414, loss = 0.001262
grad_step = 000415, loss = 0.001259
grad_step = 000416, loss = 0.001254
grad_step = 000417, loss = 0.001250
grad_step = 000418, loss = 0.001247
grad_step = 000419, loss = 0.001248
grad_step = 000420, loss = 0.001255
grad_step = 000421, loss = 0.001267
grad_step = 000422, loss = 0.001294
grad_step = 000423, loss = 0.001331
grad_step = 000424, loss = 0.001407
grad_step = 000425, loss = 0.001465
grad_step = 000426, loss = 0.001553
grad_step = 000427, loss = 0.001576
grad_step = 000428, loss = 0.001540
grad_step = 000429, loss = 0.001454
grad_step = 000430, loss = 0.001324
grad_step = 000431, loss = 0.001283
grad_step = 000432, loss = 0.001313
grad_step = 000433, loss = 0.001348
grad_step = 000434, loss = 0.001345
grad_step = 000435, loss = 0.001300
grad_step = 000436, loss = 0.001249
grad_step = 000437, loss = 0.001236
grad_step = 000438, loss = 0.001291
grad_step = 000439, loss = 0.001328
grad_step = 000440, loss = 0.001267
grad_step = 000441, loss = 0.001206
grad_step = 000442, loss = 0.001216
grad_step = 000443, loss = 0.001248
grad_step = 000444, loss = 0.001252
grad_step = 000445, loss = 0.001225
grad_step = 000446, loss = 0.001210
grad_step = 000447, loss = 0.001205
grad_step = 000448, loss = 0.001209
grad_step = 000449, loss = 0.001218
grad_step = 000450, loss = 0.001211
grad_step = 000451, loss = 0.001190
grad_step = 000452, loss = 0.001178
grad_step = 000453, loss = 0.001189
grad_step = 000454, loss = 0.001202
grad_step = 000455, loss = 0.001195
grad_step = 000456, loss = 0.001180
grad_step = 000457, loss = 0.001174
grad_step = 000458, loss = 0.001180
grad_step = 000459, loss = 0.001188
grad_step = 000460, loss = 0.001190
grad_step = 000461, loss = 0.001195
grad_step = 000462, loss = 0.001210
grad_step = 000463, loss = 0.001230
grad_step = 000464, loss = 0.001265
grad_step = 000465, loss = 0.001279
grad_step = 000466, loss = 0.001284
grad_step = 000467, loss = 0.001228
grad_step = 000468, loss = 0.001175
grad_step = 000469, loss = 0.001156
grad_step = 000470, loss = 0.001178
grad_step = 000471, loss = 0.001200
grad_step = 000472, loss = 0.001178
grad_step = 000473, loss = 0.001149
grad_step = 000474, loss = 0.001150
grad_step = 000475, loss = 0.001170
grad_step = 000476, loss = 0.001170
grad_step = 000477, loss = 0.001144
grad_step = 000478, loss = 0.001133
grad_step = 000479, loss = 0.001143
grad_step = 000480, loss = 0.001150
grad_step = 000481, loss = 0.001148
grad_step = 000482, loss = 0.001142
grad_step = 000483, loss = 0.001137
grad_step = 000484, loss = 0.001128
grad_step = 000485, loss = 0.001120
grad_step = 000486, loss = 0.001117
grad_step = 000487, loss = 0.001120
grad_step = 000488, loss = 0.001124
grad_step = 000489, loss = 0.001125
grad_step = 000490, loss = 0.001126
grad_step = 000491, loss = 0.001127
grad_step = 000492, loss = 0.001131
grad_step = 000493, loss = 0.001134
grad_step = 000494, loss = 0.001137
grad_step = 000495, loss = 0.001135
grad_step = 000496, loss = 0.001136
grad_step = 000497, loss = 0.001134
grad_step = 000498, loss = 0.001134
grad_step = 000499, loss = 0.001127
grad_step = 000500, loss = 0.001119
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001108
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

  date_run                              2020-05-12 19:12:52.382525
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.228076
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 19:12:52.389103
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140296
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 19:12:52.397097
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.124405
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 19:12:52.403175
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.13184
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
0   2020-05-12 19:12:20.596176  ...    mean_absolute_error
1   2020-05-12 19:12:20.602067  ...     mean_squared_error
2   2020-05-12 19:12:20.606817  ...  median_absolute_error
3   2020-05-12 19:12:20.610599  ...               r2_score
4   2020-05-12 19:12:29.372393  ...    mean_absolute_error
5   2020-05-12 19:12:29.376791  ...     mean_squared_error
6   2020-05-12 19:12:29.380266  ...  median_absolute_error
7   2020-05-12 19:12:29.383459  ...               r2_score
8   2020-05-12 19:12:52.382525  ...    mean_absolute_error
9   2020-05-12 19:12:52.389103  ...     mean_squared_error
10  2020-05-12 19:12:52.397097  ...  median_absolute_error
11  2020-05-12 19:12:52.403175  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144749.07it/s] 60%|█████▉    | 5906432/9912422 [00:00<00:19, 206552.36it/s]9920512it [00:00, 37117398.64it/s]                           
0it [00:00, ?it/s]32768it [00:00, 561596.51it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163383.62it/s]1654784it [00:00, 11392617.10it/s]                         
0it [00:00, ?it/s]8192it [00:00, 139641.94it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b6c32bbe0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1ece6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1e3120f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1ece6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b6c336ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1ba964e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b6c336ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1ece6eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b6c336ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b1ba964e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6b6c2edf28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f18d6e0b240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=12080fe93f0733adbba63a9c0c9b36a61f1b9161cf746258542ac842b78cb61c
  Stored in directory: /tmp/pip-ephem-wheel-cache-xemehs90/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f186ec047b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2965504/17464789 [====>.........................] - ETA: 0s
11149312/17464789 [==================>...........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 19:14:20.343377: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 19:14:20.347634: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 19:14:20.347876: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558b248c1a70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 19:14:20.347891: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.1113 - accuracy: 0.4710
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6935 - accuracy: 0.4983
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7218 - accuracy: 0.4964
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6225 - accuracy: 0.5029
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6240 - accuracy: 0.5028
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6038 - accuracy: 0.5041
11000/25000 [============>.................] - ETA: 4s - loss: 7.5649 - accuracy: 0.5066
12000/25000 [=============>................] - ETA: 4s - loss: 7.5772 - accuracy: 0.5058
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5805 - accuracy: 0.5056
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5593 - accuracy: 0.5070
15000/25000 [=================>............] - ETA: 3s - loss: 7.5746 - accuracy: 0.5060
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5813 - accuracy: 0.5056
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5918 - accuracy: 0.5049
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6138 - accuracy: 0.5034
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6222 - accuracy: 0.5029
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6168 - accuracy: 0.5033
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6111 - accuracy: 0.5036
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6241 - accuracy: 0.5028
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 9s 375us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 19:14:36.486251
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 19:14:36.486251  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<51:07:09, 4.68kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<36:01:19, 6.65kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<25:15:56, 9.48kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:41:09, 13.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<12:20:39, 19.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.40M/862M [00:02<8:35:05, 27.6kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:02<5:58:22, 39.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:02<4:10:04, 56.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:02<2:53:58, 80.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:02<2:01:28, 115kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:02<1:24:34, 164kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:03<59:06, 233kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.6M/862M [00:03<41:14, 332kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:03<28:50, 473kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:03<20:08, 673kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:03<14:12, 951kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.8M/862M [00:04<11:50, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:04<08:39, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:04<06:11, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:06<28:53, 464kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:06<23:20, 574kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:06<17:01, 786kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<12:04, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:08<14:30, 919kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:08<11:31, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<08:20, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:10<08:56, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:10<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<05:38, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:12<07:04, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:12<07:38, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:12<05:56, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<04:17, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:14<18:11, 721kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:14<13:52, 946kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<10:03, 1.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<07:10, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:16<57:36, 227kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:16<41:38, 313kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<29:22, 443kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:18<23:36, 550kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:18<19:09, 678kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:18<13:57, 929kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<09:57, 1.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:20<11:39, 1.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:20<09:31, 1.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<05:01, 2.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:22<48:25, 265kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:22<35:12, 365kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<24:55, 515kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:24<20:25, 626kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:24<16:56, 754kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:24<12:24, 1.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:26<10:08, 1.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:26<08:24, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:10, 2.05MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<07:16, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<07:40, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<05:57, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:17, 2.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<35:09, 358kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:30<25:54, 485kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<18:22, 683kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<15:46, 793kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<13:36, 919kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<10:04, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:17, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<08:49, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<07:27, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:31, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<06:46, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<05:58, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<04:29, 2.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<06:02, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<04:13, 2.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:07, 3.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<20:07, 608kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<16:55, 723kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<12:26, 982kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<08:58, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:42<09:18, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<07:56, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:53, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<06:39, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<07:29, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<05:56, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<04:18, 2.80MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<12:24, 969kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<11:31, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<08:47, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:18, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<09:09, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<07:49, 1.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<05:50, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:17, 2.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<08:01, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<07:19, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:58, 2.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<16:36, 712kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<12:50, 920kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<09:41, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:55, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<10:14, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<09:54, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<07:33, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:24, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<12:37, 925kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<09:59, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<07:19, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<07:34, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<08:02, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<06:11, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:36, 2.51MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<05:29, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<04:10, 2.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<05:20, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<06:26, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<05:07, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:43, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<08:40, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<07:11, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:16, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<03:52, 2.93MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<10:46, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<10:11, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:32, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<07:59, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:42, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<04:58, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:39, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<09:01, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<09:02, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:01, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:10, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<06:21, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<04:43, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<05:37, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:34, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:15, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:49, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<10:46, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<08:49, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<06:27, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<07:19, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<05:46, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:10, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:19<11:55, 910kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:37, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<07:01, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<07:11, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:32, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:48, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:17, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<05:46, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:17, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<03:59, 2.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<05:01, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:01, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:50, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<03:29, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<10:21, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<08:29, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:12, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:27, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<30:24, 345kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<22:28, 467kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<15:57, 656kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<13:22, 780kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<11:48, 884kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<08:52, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:20, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<13:01, 796kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<10:18, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:28, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<07:23, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<07:37, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<05:50, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:20, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:23, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:45, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<03:49, 2.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<02:47, 3.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<09:01, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<07:30, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:29, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:57, 2.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<36:28, 276kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<27:53, 361kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<20:05, 501kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<14:08, 709kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<18:19, 547kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<13:47, 726kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:56, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:06, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<09:27, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<08:57, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<06:45, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:55, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<06:04, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:25, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:04, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<04:54, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<05:46, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:36, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:20, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<07:01, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<06:02, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:28, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:15, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<12:37, 766kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<09:53, 977kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:10, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<07:08, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:06, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:32, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<05:10, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<05:51, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:33, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:24, 2.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:41, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<03:20, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:18, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<05:00, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<02:54, 3.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<08:11, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<06:49, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:00, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<05:25, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<05:47, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:15, 2.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<07:03, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:00, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:25, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<05:00, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:34, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:28, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:19, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:04, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:06, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:03, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:45, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:46, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<02:45, 3.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<05:54, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<05:11, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:53, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<04:33, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:09, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<03:08, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:08, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:56, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:54, 2.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:50, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<05:38, 1.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:40, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<04:22, 1.97MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<05:05, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:59, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:56, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<04:30, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<04:05, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:04, 2.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:02, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:48, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<03:51, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<02:48, 3.00MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<08:56, 941kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<07:14, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:18, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<05:27, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<05:47, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<04:31, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:15, 2.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<08:31, 972kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<06:55, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:02, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<05:14, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<05:30, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:13, 1.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:03, 2.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<05:32, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<03:37, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<04:13, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<02:57, 2.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<03:45, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:21, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:28, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:31, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<13:55, 570kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<10:35, 748kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:34, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<07:01, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<06:45, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<05:06, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:41, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<05:18, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<04:37, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:25, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<04:00, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<04:32, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:36, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:36, 2.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<13:51, 553kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<10:34, 723kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:34, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<06:52, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<06:29, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:53, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:35, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:22, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:53, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:53, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<04:17, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:24, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:28, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<05:21, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<04:33, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:21, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:00, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:32, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:34, 2.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:34, 2.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:57, 1.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:01, 1.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:42, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:04, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:37, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:38, 2.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<07:43, 919kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<06:15, 1.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:34, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<04:39, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:53, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:00, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:11, 3.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<10:57, 636kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<08:24, 827kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<06:17, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:28, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<06:17, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<05:15, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<03:52, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:07, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<03:42, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:10, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:02, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:37, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<02:55, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:07, 3.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<04:06, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:19<03:35, 1.86MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:52, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:05, 3.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:54, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<04:14, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:07, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:15, 2.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<09:46, 670kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:23<07:36, 859kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<05:30, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:11, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<05:11, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<04:00, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:52, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:42, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<04:45, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:30, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:46, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<04:06, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:14, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:20, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<06:45, 927kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<05:24, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:56, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:07, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<04:12, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<03:13, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:19, 2.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:45, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:59, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:22, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:56, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:07, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<09:23, 638kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<07:15, 823kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<05:13, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:52, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:03, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:57, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:22, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:41, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<02:53, 2.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:04, 2.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<07:28, 774kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:53, 979kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<04:16, 1.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:10, 1.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:21, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:23, 2.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<06:20, 890kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<05:02, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:38, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<03:45, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<03:55, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:04, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:11, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<05:17, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<04:20, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:09, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:16, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<42:18, 128kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<30:52, 176kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<21:50, 248kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<15:15, 353kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<13:01, 412kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<09:41, 553kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:52, 775kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<05:57, 889kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<05:24, 979kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:03, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:53, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:34, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:42, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:47, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:00, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<05:04, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:44, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:37, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:34, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<05:38, 902kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:26, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:14, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:04<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<04:54, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<04:36, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:28, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<05:39, 875kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:31, 1.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:16, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:18, 1.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:27, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:42, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:56, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<05:23, 890kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<04:14, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:07, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:14, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<04:37, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<03:48, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:47, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:54, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<03:07, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:27, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:46, 2.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<05:03, 910kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:04, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:58, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:00, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<03:06, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:25, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:44, 2.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<08:13, 542kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<06:14, 714kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:26, 996kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<04:03, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<03:51, 1.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:05, 2.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<03:11, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:42, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:19, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:04, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:33, 2.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<02:01, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:53, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:25, 2.92MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:51, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:15, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:48, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:18, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:19, 934kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:27, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:30, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:37, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:41, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:29, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<06:55, 564kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<05:15, 740kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:46, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:27, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:49, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:04, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:16, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:28, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:54, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:22, 2.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:39, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:17, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:42, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:56, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:07, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:41, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:12, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<05:36, 635kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<04:20, 818kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:06, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:53, 1.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:49, 1.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:09, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:32, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<02:39, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:52<02:15, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:40, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:51, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<02:02, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:36, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:09, 2.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<05:57, 550kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<04:31, 724kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<03:13, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:56, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:46, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:05, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:29, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<02:09, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:52, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:36, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:46, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:24, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:00, 2.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<05:17, 566kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<04:03, 739kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:54, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:36, 1.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:30, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:55, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:21, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:19, 859kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:39, 1.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:55, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:55, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:57, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:31, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:04, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<04:28, 608kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:22, 805kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:24, 1.12MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:42, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<04:09, 639kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<03:11, 831kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:16, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<02:08, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<02:06, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:35, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:08, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<01:49, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:33, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:49, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<15:14, 160kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<11:13, 217kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<07:56, 306kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<05:33, 433kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<04:27, 534kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<03:23, 700kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<02:24, 972kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<02:08, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:45, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:17, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:21, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:28, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:07, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:48, 2.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:18, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:10, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:51, 2.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:02, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:12, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:30<00:58, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:41, 2.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<02:10, 930kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:45, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:16, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<01:17, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<01:22, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<01:03, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:45, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<01:06, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<01:00, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:44, 2.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<00:53, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<01:02, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:35, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<01:51, 939kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:30, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:05, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:05, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:08, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<00:52, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:37, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:11, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:00, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:44, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:50, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:57, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:44, 2.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:31, 2.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:57, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:37, 2.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:42, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:39, 2.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:29, 2.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:36, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:44, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:35, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:25, 3.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<01:20, 944kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<01:04, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:46, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:46, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:48, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:37, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:26, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<01:06, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:53, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:40, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:42, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:23, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:48, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:40, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:20, 2.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:32, 592kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<01:17, 707kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:56, 961kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:39, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:39, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:33, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:23, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:25, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:28, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:15, 2.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:45, 925kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:36, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:26, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:25, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:26, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:20, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:22, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:19, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:14, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:15, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:17, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:09, 2.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:14, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:13, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:09, 2.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:19<00:10, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:12, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:09, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:17, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:13, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:09, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:08, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:07, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 2.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 2.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 3.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.91MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 775/400000 [00:00<00:51, 7745.52it/s]  0%|          | 1558/400000 [00:00<00:51, 7769.23it/s]  1%|          | 2415/400000 [00:00<00:49, 7992.86it/s]  1%|          | 3237/400000 [00:00<00:49, 8059.34it/s]  1%|          | 4032/400000 [00:00<00:49, 8023.67it/s]  1%|          | 4790/400000 [00:00<00:50, 7882.73it/s]  1%|▏         | 5581/400000 [00:00<00:49, 7890.12it/s]  2%|▏         | 6378/400000 [00:00<00:49, 7913.52it/s]  2%|▏         | 7147/400000 [00:00<00:50, 7843.83it/s]  2%|▏         | 7954/400000 [00:01<00:49, 7908.24it/s]  2%|▏         | 8739/400000 [00:01<00:49, 7888.86it/s]  2%|▏         | 9523/400000 [00:01<00:49, 7871.72it/s]  3%|▎         | 10386/400000 [00:01<00:48, 8083.77it/s]  3%|▎         | 11276/400000 [00:01<00:46, 8310.69it/s]  3%|▎         | 12105/400000 [00:01<00:46, 8277.49it/s]  3%|▎         | 12958/400000 [00:01<00:46, 8351.63it/s]  3%|▎         | 13872/400000 [00:01<00:45, 8572.18it/s]  4%|▎         | 14753/400000 [00:01<00:44, 8641.20it/s]  4%|▍         | 15618/400000 [00:01<00:44, 8619.74it/s]  4%|▍         | 16481/400000 [00:02<00:45, 8485.24it/s]  4%|▍         | 17331/400000 [00:02<00:45, 8478.55it/s]  5%|▍         | 18180/400000 [00:02<00:45, 8436.62it/s]  5%|▍         | 19066/400000 [00:02<00:44, 8558.71it/s]  5%|▍         | 19923/400000 [00:02<00:44, 8556.68it/s]  5%|▌         | 20801/400000 [00:02<00:43, 8620.50it/s]  5%|▌         | 21690/400000 [00:02<00:43, 8697.46it/s]  6%|▌         | 22579/400000 [00:02<00:43, 8753.93it/s]  6%|▌         | 23455/400000 [00:02<00:43, 8675.06it/s]  6%|▌         | 24352/400000 [00:02<00:42, 8759.46it/s]  6%|▋         | 25229/400000 [00:03<00:42, 8720.88it/s]  7%|▋         | 26115/400000 [00:03<00:42, 8759.70it/s]  7%|▋         | 27052/400000 [00:03<00:41, 8932.46it/s]  7%|▋         | 27947/400000 [00:03<00:41, 8885.54it/s]  7%|▋         | 28837/400000 [00:03<00:42, 8812.74it/s]  7%|▋         | 29719/400000 [00:03<00:43, 8522.40it/s]  8%|▊         | 30584/400000 [00:03<00:43, 8516.29it/s]  8%|▊         | 31438/400000 [00:03<00:44, 8308.76it/s]  8%|▊         | 32272/400000 [00:03<00:45, 8061.30it/s]  8%|▊         | 33196/400000 [00:03<00:43, 8380.76it/s]  9%|▊         | 34073/400000 [00:04<00:43, 8493.31it/s]  9%|▊         | 34966/400000 [00:04<00:42, 8617.22it/s]  9%|▉         | 35908/400000 [00:04<00:41, 8841.40it/s]  9%|▉         | 36796/400000 [00:04<00:41, 8655.19it/s]  9%|▉         | 37666/400000 [00:04<00:43, 8365.66it/s] 10%|▉         | 38508/400000 [00:04<00:43, 8338.93it/s] 10%|▉         | 39416/400000 [00:04<00:42, 8548.20it/s] 10%|█         | 40298/400000 [00:04<00:41, 8626.72it/s] 10%|█         | 41177/400000 [00:04<00:41, 8673.51it/s] 11%|█         | 42050/400000 [00:04<00:41, 8689.73it/s] 11%|█         | 42938/400000 [00:05<00:40, 8744.64it/s] 11%|█         | 43890/400000 [00:05<00:39, 8963.41it/s] 11%|█         | 44825/400000 [00:05<00:39, 9073.65it/s] 11%|█▏        | 45765/400000 [00:05<00:38, 9167.05it/s] 12%|█▏        | 46684/400000 [00:05<00:39, 8916.20it/s] 12%|█▏        | 47579/400000 [00:05<00:41, 8552.57it/s] 12%|█▏        | 48512/400000 [00:05<00:40, 8770.35it/s] 12%|█▏        | 49435/400000 [00:05<00:39, 8902.04it/s] 13%|█▎        | 50330/400000 [00:05<00:39, 8915.87it/s] 13%|█▎        | 51242/400000 [00:05<00:38, 8975.37it/s] 13%|█▎        | 52142/400000 [00:06<00:39, 8858.45it/s] 13%|█▎        | 53034/400000 [00:06<00:39, 8875.84it/s] 13%|█▎        | 53923/400000 [00:06<00:39, 8866.03it/s] 14%|█▎        | 54811/400000 [00:06<00:38, 8858.54it/s] 14%|█▍        | 55729/400000 [00:06<00:38, 8950.45it/s] 14%|█▍        | 56625/400000 [00:06<00:39, 8793.97it/s] 14%|█▍        | 57506/400000 [00:06<00:41, 8302.23it/s] 15%|█▍        | 58422/400000 [00:06<00:39, 8540.58it/s] 15%|█▍        | 59283/400000 [00:06<00:40, 8437.45it/s] 15%|█▌        | 60132/400000 [00:07<00:42, 8040.89it/s] 15%|█▌        | 60944/400000 [00:07<00:42, 7971.10it/s] 15%|█▌        | 61753/400000 [00:07<00:42, 8005.07it/s] 16%|█▌        | 62653/400000 [00:07<00:40, 8277.99it/s] 16%|█▌        | 63569/400000 [00:07<00:39, 8522.65it/s] 16%|█▌        | 64475/400000 [00:07<00:38, 8675.02it/s] 16%|█▋        | 65374/400000 [00:07<00:38, 8764.86it/s] 17%|█▋        | 66327/400000 [00:07<00:37, 8980.21it/s] 17%|█▋        | 67238/400000 [00:07<00:36, 9018.47it/s] 17%|█▋        | 68143/400000 [00:07<00:37, 8933.86it/s] 17%|█▋        | 69045/400000 [00:08<00:36, 8959.40it/s] 17%|█▋        | 69943/400000 [00:08<00:37, 8698.67it/s] 18%|█▊        | 70816/400000 [00:08<00:38, 8614.72it/s] 18%|█▊        | 71680/400000 [00:08<00:38, 8568.41it/s] 18%|█▊        | 72539/400000 [00:08<00:38, 8536.25it/s] 18%|█▊        | 73423/400000 [00:08<00:37, 8623.85it/s] 19%|█▊        | 74287/400000 [00:08<00:38, 8359.89it/s] 19%|█▉        | 75126/400000 [00:08<00:39, 8282.25it/s] 19%|█▉        | 76005/400000 [00:08<00:38, 8426.38it/s] 19%|█▉        | 76850/400000 [00:08<00:38, 8388.78it/s] 19%|█▉        | 77746/400000 [00:09<00:37, 8551.18it/s] 20%|█▉        | 78603/400000 [00:09<00:37, 8548.72it/s] 20%|█▉        | 79502/400000 [00:09<00:36, 8674.75it/s] 20%|██        | 80435/400000 [00:09<00:36, 8860.55it/s] 20%|██        | 81323/400000 [00:09<00:36, 8830.41it/s] 21%|██        | 82208/400000 [00:09<00:36, 8800.17it/s] 21%|██        | 83089/400000 [00:09<00:36, 8751.82it/s] 21%|██        | 83965/400000 [00:09<00:36, 8648.74it/s] 21%|██        | 84900/400000 [00:09<00:35, 8845.49it/s] 21%|██▏       | 85788/400000 [00:10<00:35, 8853.99it/s] 22%|██▏       | 86675/400000 [00:10<00:35, 8814.29it/s] 22%|██▏       | 87558/400000 [00:10<00:35, 8769.58it/s] 22%|██▏       | 88436/400000 [00:10<00:36, 8502.51it/s] 22%|██▏       | 89289/400000 [00:10<00:37, 8247.25it/s] 23%|██▎       | 90117/400000 [00:10<00:38, 8005.51it/s] 23%|██▎       | 90922/400000 [00:10<00:39, 7864.15it/s] 23%|██▎       | 91712/400000 [00:10<00:39, 7830.67it/s] 23%|██▎       | 92498/400000 [00:10<00:39, 7761.52it/s] 23%|██▎       | 93278/400000 [00:10<00:39, 7772.96it/s] 24%|██▎       | 94057/400000 [00:11<00:39, 7777.88it/s] 24%|██▎       | 94836/400000 [00:11<00:39, 7696.06it/s] 24%|██▍       | 95653/400000 [00:11<00:38, 7829.35it/s] 24%|██▍       | 96540/400000 [00:11<00:37, 8113.36it/s] 24%|██▍       | 97415/400000 [00:11<00:36, 8294.09it/s] 25%|██▍       | 98286/400000 [00:11<00:35, 8413.45it/s] 25%|██▍       | 99131/400000 [00:11<00:35, 8386.56it/s] 25%|██▍       | 99972/400000 [00:11<00:35, 8370.75it/s] 25%|██▌       | 100811/400000 [00:11<00:36, 8104.57it/s] 25%|██▌       | 101625/400000 [00:11<00:37, 7979.87it/s] 26%|██▌       | 102448/400000 [00:12<00:36, 8050.58it/s] 26%|██▌       | 103255/400000 [00:12<00:37, 7973.94it/s] 26%|██▌       | 104123/400000 [00:12<00:36, 8171.34it/s] 26%|██▋       | 105000/400000 [00:12<00:35, 8341.77it/s] 26%|██▋       | 105837/400000 [00:12<00:35, 8176.43it/s] 27%|██▋       | 106658/400000 [00:12<00:36, 7988.50it/s] 27%|██▋       | 107460/400000 [00:12<00:37, 7902.59it/s] 27%|██▋       | 108281/400000 [00:12<00:36, 7990.36it/s] 27%|██▋       | 109082/400000 [00:12<00:36, 7982.31it/s] 27%|██▋       | 109897/400000 [00:12<00:36, 8030.85it/s] 28%|██▊       | 110703/400000 [00:13<00:35, 8038.11it/s] 28%|██▊       | 111508/400000 [00:13<00:36, 7969.32it/s] 28%|██▊       | 112306/400000 [00:13<00:36, 7864.79it/s] 28%|██▊       | 113094/400000 [00:13<00:36, 7810.35it/s] 28%|██▊       | 113876/400000 [00:13<00:36, 7780.63it/s] 29%|██▊       | 114655/400000 [00:13<00:36, 7763.48it/s] 29%|██▉       | 115432/400000 [00:13<00:36, 7732.27it/s] 29%|██▉       | 116206/400000 [00:13<00:36, 7693.73it/s] 29%|██▉       | 116976/400000 [00:13<00:36, 7676.61it/s] 29%|██▉       | 117747/400000 [00:14<00:36, 7685.39it/s] 30%|██▉       | 118521/400000 [00:14<00:36, 7698.91it/s] 30%|██▉       | 119291/400000 [00:14<00:36, 7639.37it/s] 30%|███       | 120061/400000 [00:14<00:36, 7656.88it/s] 30%|███       | 120831/400000 [00:14<00:36, 7667.05it/s] 30%|███       | 121598/400000 [00:14<00:36, 7659.31it/s] 31%|███       | 122365/400000 [00:14<00:36, 7660.12it/s] 31%|███       | 123132/400000 [00:14<00:36, 7574.03it/s] 31%|███       | 123903/400000 [00:14<00:36, 7614.00it/s] 31%|███       | 124670/400000 [00:14<00:36, 7628.24it/s] 31%|███▏      | 125433/400000 [00:15<00:36, 7607.86it/s] 32%|███▏      | 126194/400000 [00:15<00:36, 7585.00it/s] 32%|███▏      | 126957/400000 [00:15<00:35, 7595.59it/s] 32%|███▏      | 127730/400000 [00:15<00:35, 7634.54it/s] 32%|███▏      | 128501/400000 [00:15<00:35, 7655.60it/s] 32%|███▏      | 129272/400000 [00:15<00:35, 7670.10it/s] 33%|███▎      | 130040/400000 [00:15<00:35, 7669.42it/s] 33%|███▎      | 130807/400000 [00:15<00:35, 7620.40it/s] 33%|███▎      | 131570/400000 [00:15<00:35, 7621.98it/s] 33%|███▎      | 132342/400000 [00:15<00:34, 7649.74it/s] 33%|███▎      | 133113/400000 [00:16<00:34, 7665.66it/s] 33%|███▎      | 133918/400000 [00:16<00:34, 7776.49it/s] 34%|███▎      | 134697/400000 [00:16<00:34, 7660.65it/s] 34%|███▍      | 135502/400000 [00:16<00:34, 7771.37it/s] 34%|███▍      | 136301/400000 [00:16<00:33, 7835.39it/s] 34%|███▍      | 137141/400000 [00:16<00:32, 7994.26it/s] 34%|███▍      | 137948/400000 [00:16<00:32, 8014.40it/s] 35%|███▍      | 138755/400000 [00:16<00:32, 8028.65it/s] 35%|███▍      | 139614/400000 [00:16<00:31, 8189.16it/s] 35%|███▌      | 140446/400000 [00:16<00:31, 8225.82it/s] 35%|███▌      | 141270/400000 [00:17<00:31, 8096.72it/s] 36%|███▌      | 142081/400000 [00:17<00:31, 8097.78it/s] 36%|███▌      | 142892/400000 [00:17<00:31, 8057.47it/s] 36%|███▌      | 143770/400000 [00:17<00:31, 8259.27it/s] 36%|███▌      | 144598/400000 [00:17<00:31, 8236.38it/s] 36%|███▋      | 145423/400000 [00:17<00:30, 8224.20it/s] 37%|███▋      | 146270/400000 [00:17<00:30, 8296.08it/s] 37%|███▋      | 147112/400000 [00:17<00:30, 8330.76it/s] 37%|███▋      | 147946/400000 [00:17<00:30, 8331.67it/s] 37%|███▋      | 148780/400000 [00:17<00:30, 8170.21it/s] 37%|███▋      | 149634/400000 [00:18<00:30, 8276.25it/s] 38%|███▊      | 150531/400000 [00:18<00:29, 8472.06it/s] 38%|███▊      | 151381/400000 [00:18<00:29, 8356.72it/s] 38%|███▊      | 152219/400000 [00:18<00:30, 8157.76it/s] 38%|███▊      | 153037/400000 [00:18<00:30, 8032.20it/s] 38%|███▊      | 153843/400000 [00:18<00:30, 7955.06it/s] 39%|███▊      | 154641/400000 [00:18<00:31, 7886.74it/s] 39%|███▉      | 155431/400000 [00:18<00:31, 7800.27it/s] 39%|███▉      | 156213/400000 [00:18<00:31, 7794.83it/s] 39%|███▉      | 156994/400000 [00:18<00:31, 7796.37it/s] 39%|███▉      | 157775/400000 [00:19<00:31, 7778.08it/s] 40%|███▉      | 158557/400000 [00:19<00:30, 7790.09it/s] 40%|███▉      | 159337/400000 [00:19<00:31, 7759.15it/s] 40%|████      | 160114/400000 [00:19<00:30, 7760.54it/s] 40%|████      | 160897/400000 [00:19<00:30, 7779.03it/s] 40%|████      | 161676/400000 [00:19<00:30, 7780.54it/s] 41%|████      | 162455/400000 [00:19<00:30, 7777.97it/s] 41%|████      | 163233/400000 [00:19<00:30, 7760.72it/s] 41%|████      | 164016/400000 [00:19<00:30, 7780.73it/s] 41%|████      | 164797/400000 [00:19<00:30, 7788.04it/s] 41%|████▏     | 165576/400000 [00:20<00:31, 7540.35it/s] 42%|████▏     | 166332/400000 [00:20<00:31, 7443.47it/s] 42%|████▏     | 167105/400000 [00:20<00:30, 7524.89it/s] 42%|████▏     | 167888/400000 [00:20<00:30, 7611.88it/s] 42%|████▏     | 168670/400000 [00:20<00:30, 7670.88it/s] 42%|████▏     | 169442/400000 [00:20<00:30, 7683.64it/s] 43%|████▎     | 170219/400000 [00:20<00:29, 7709.21it/s] 43%|████▎     | 170991/400000 [00:20<00:29, 7673.96it/s] 43%|████▎     | 171766/400000 [00:20<00:29, 7695.34it/s] 43%|████▎     | 172536/400000 [00:21<00:29, 7642.66it/s] 43%|████▎     | 173352/400000 [00:21<00:29, 7789.84it/s] 44%|████▎     | 174165/400000 [00:21<00:28, 7887.06it/s] 44%|████▎     | 174976/400000 [00:21<00:28, 7942.24it/s] 44%|████▍     | 175797/400000 [00:21<00:27, 8017.18it/s] 44%|████▍     | 176632/400000 [00:21<00:27, 8112.69it/s] 44%|████▍     | 177507/400000 [00:21<00:26, 8290.96it/s] 45%|████▍     | 178356/400000 [00:21<00:26, 8349.44it/s] 45%|████▍     | 179193/400000 [00:21<00:27, 8108.04it/s] 45%|████▌     | 180007/400000 [00:21<00:27, 8050.29it/s] 45%|████▌     | 180814/400000 [00:22<00:27, 7964.45it/s] 45%|████▌     | 181612/400000 [00:22<00:27, 7867.64it/s] 46%|████▌     | 182401/400000 [00:22<00:28, 7670.98it/s] 46%|████▌     | 183171/400000 [00:22<00:28, 7589.87it/s] 46%|████▌     | 183975/400000 [00:22<00:27, 7718.72it/s] 46%|████▌     | 184773/400000 [00:22<00:27, 7793.30it/s] 46%|████▋     | 185601/400000 [00:22<00:27, 7932.08it/s] 47%|████▋     | 186435/400000 [00:22<00:26, 8048.89it/s] 47%|████▋     | 187244/400000 [00:22<00:26, 8059.02it/s] 47%|████▋     | 188072/400000 [00:22<00:26, 8122.82it/s] 47%|████▋     | 188913/400000 [00:23<00:25, 8205.35it/s] 47%|████▋     | 189757/400000 [00:23<00:25, 8273.26it/s] 48%|████▊     | 190652/400000 [00:23<00:24, 8464.67it/s] 48%|████▊     | 191505/400000 [00:23<00:24, 8483.57it/s] 48%|████▊     | 192414/400000 [00:23<00:23, 8654.83it/s] 48%|████▊     | 193282/400000 [00:23<00:23, 8649.21it/s] 49%|████▊     | 194197/400000 [00:23<00:23, 8791.93it/s] 49%|████▉     | 195078/400000 [00:23<00:23, 8728.16it/s] 49%|████▉     | 195952/400000 [00:23<00:23, 8588.08it/s] 49%|████▉     | 196813/400000 [00:23<00:23, 8548.99it/s] 49%|████▉     | 197669/400000 [00:24<00:23, 8501.44it/s] 50%|████▉     | 198543/400000 [00:24<00:23, 8569.29it/s] 50%|████▉     | 199401/400000 [00:24<00:23, 8460.79it/s] 50%|█████     | 200261/400000 [00:24<00:23, 8499.84it/s] 50%|█████     | 201215/400000 [00:24<00:22, 8787.16it/s] 51%|█████     | 202127/400000 [00:24<00:22, 8882.10it/s] 51%|█████     | 203018/400000 [00:24<00:22, 8844.39it/s] 51%|█████     | 203942/400000 [00:24<00:21, 8956.64it/s] 51%|█████     | 204840/400000 [00:24<00:22, 8748.67it/s] 51%|█████▏    | 205806/400000 [00:24<00:21, 9001.58it/s] 52%|█████▏    | 206771/400000 [00:25<00:21, 9186.33it/s] 52%|█████▏    | 207705/400000 [00:25<00:20, 9231.66it/s] 52%|█████▏    | 208681/400000 [00:25<00:20, 9380.96it/s] 52%|█████▏    | 209622/400000 [00:25<00:20, 9297.64it/s] 53%|█████▎    | 210607/400000 [00:25<00:20, 9455.92it/s] 53%|█████▎    | 211555/400000 [00:25<00:20, 9229.04it/s] 53%|█████▎    | 212486/400000 [00:25<00:20, 9251.30it/s] 53%|█████▎    | 213446/400000 [00:25<00:19, 9351.37it/s] 54%|█████▎    | 214383/400000 [00:25<00:20, 9009.40it/s] 54%|█████▍    | 215288/400000 [00:25<00:20, 9010.35it/s] 54%|█████▍    | 216192/400000 [00:26<00:20, 8988.70it/s] 54%|█████▍    | 217116/400000 [00:26<00:20, 9056.76it/s] 55%|█████▍    | 218024/400000 [00:26<00:20, 8998.90it/s] 55%|█████▍    | 218925/400000 [00:26<00:20, 8779.63it/s] 55%|█████▍    | 219828/400000 [00:26<00:20, 8852.09it/s] 55%|█████▌    | 220761/400000 [00:26<00:19, 8988.54it/s] 55%|█████▌    | 221726/400000 [00:26<00:19, 9175.63it/s] 56%|█████▌    | 222718/400000 [00:26<00:18, 9386.36it/s] 56%|█████▌    | 223660/400000 [00:26<00:19, 9161.16it/s] 56%|█████▌    | 224590/400000 [00:27<00:19, 9199.97it/s] 56%|█████▋    | 225538/400000 [00:27<00:18, 9279.98it/s] 57%|█████▋    | 226468/400000 [00:27<00:19, 9077.66it/s] 57%|█████▋    | 227378/400000 [00:27<00:19, 9001.62it/s] 57%|█████▋    | 228280/400000 [00:27<00:19, 8885.14it/s] 57%|█████▋    | 229186/400000 [00:27<00:19, 8936.22it/s] 58%|█████▊    | 230114/400000 [00:27<00:18, 9035.73it/s] 58%|█████▊    | 231097/400000 [00:27<00:18, 9259.69it/s] 58%|█████▊    | 232038/400000 [00:27<00:18, 9303.97it/s] 58%|█████▊    | 232970/400000 [00:27<00:18, 9179.53it/s] 58%|█████▊    | 233922/400000 [00:28<00:17, 9276.48it/s] 59%|█████▊    | 234855/400000 [00:28<00:17, 9291.26it/s] 59%|█████▉    | 235802/400000 [00:28<00:17, 9342.02it/s] 59%|█████▉    | 236775/400000 [00:28<00:17, 9452.40it/s] 59%|█████▉    | 237722/400000 [00:28<00:17, 9371.07it/s] 60%|█████▉    | 238698/400000 [00:28<00:17, 9482.02it/s] 60%|█████▉    | 239648/400000 [00:28<00:16, 9487.35it/s] 60%|██████    | 240648/400000 [00:28<00:16, 9634.44it/s] 60%|██████    | 241613/400000 [00:28<00:16, 9464.56it/s] 61%|██████    | 242561/400000 [00:28<00:17, 8946.72it/s] 61%|██████    | 243476/400000 [00:29<00:17, 9006.41it/s] 61%|██████    | 244459/400000 [00:29<00:16, 9235.88it/s] 61%|██████▏   | 245415/400000 [00:29<00:16, 9329.84it/s] 62%|██████▏   | 246373/400000 [00:29<00:16, 9399.92it/s] 62%|██████▏   | 247316/400000 [00:29<00:16, 9382.86it/s] 62%|██████▏   | 248307/400000 [00:29<00:15, 9533.77it/s] 62%|██████▏   | 249263/400000 [00:29<00:15, 9452.46it/s] 63%|██████▎   | 250228/400000 [00:29<00:15, 9507.00it/s] 63%|██████▎   | 251180/400000 [00:29<00:15, 9493.55it/s] 63%|██████▎   | 252131/400000 [00:29<00:15, 9374.12it/s] 63%|██████▎   | 253070/400000 [00:30<00:15, 9349.06it/s] 64%|██████▎   | 254006/400000 [00:30<00:15, 9251.37it/s] 64%|██████▎   | 254964/400000 [00:30<00:15, 9345.10it/s] 64%|██████▍   | 255900/400000 [00:30<00:15, 9228.20it/s] 64%|██████▍   | 256824/400000 [00:30<00:15, 9042.34it/s] 64%|██████▍   | 257730/400000 [00:30<00:16, 8741.32it/s] 65%|██████▍   | 258656/400000 [00:30<00:15, 8889.93it/s] 65%|██████▍   | 259548/400000 [00:30<00:15, 8861.94it/s] 65%|██████▌   | 260437/400000 [00:30<00:16, 8691.34it/s] 65%|██████▌   | 261309/400000 [00:31<00:16, 8572.99it/s] 66%|██████▌   | 262190/400000 [00:31<00:15, 8640.19it/s] 66%|██████▌   | 263056/400000 [00:31<00:16, 8531.28it/s] 66%|██████▌   | 263913/400000 [00:31<00:15, 8541.15it/s] 66%|██████▌   | 264769/400000 [00:31<00:16, 8329.36it/s] 66%|██████▋   | 265604/400000 [00:31<00:16, 8241.45it/s] 67%|██████▋   | 266477/400000 [00:31<00:15, 8382.13it/s] 67%|██████▋   | 267331/400000 [00:31<00:15, 8426.79it/s] 67%|██████▋   | 268175/400000 [00:31<00:15, 8422.56it/s] 67%|██████▋   | 269019/400000 [00:31<00:15, 8225.15it/s] 67%|██████▋   | 269844/400000 [00:32<00:16, 8042.42it/s] 68%|██████▊   | 270651/400000 [00:32<00:16, 7949.70it/s] 68%|██████▊   | 271448/400000 [00:32<00:16, 7804.87it/s] 68%|██████▊   | 272231/400000 [00:32<00:16, 7656.68it/s] 68%|██████▊   | 272999/400000 [00:32<00:16, 7639.95it/s] 68%|██████▊   | 273765/400000 [00:32<00:16, 7629.15it/s] 69%|██████▊   | 274557/400000 [00:32<00:16, 7713.60it/s] 69%|██████▉   | 275374/400000 [00:32<00:15, 7843.44it/s] 69%|██████▉   | 276203/400000 [00:32<00:15, 7971.41it/s] 69%|██████▉   | 277008/400000 [00:32<00:15, 7994.18it/s] 69%|██████▉   | 277872/400000 [00:33<00:14, 8176.92it/s] 70%|██████▉   | 278707/400000 [00:33<00:14, 8227.98it/s] 70%|██████▉   | 279584/400000 [00:33<00:14, 8381.46it/s] 70%|███████   | 280424/400000 [00:33<00:14, 8214.30it/s] 70%|███████   | 281248/400000 [00:33<00:14, 8040.30it/s] 71%|███████   | 282055/400000 [00:33<00:14, 7964.71it/s] 71%|███████   | 282854/400000 [00:33<00:14, 7898.36it/s] 71%|███████   | 283646/400000 [00:33<00:14, 7845.72it/s] 71%|███████   | 284432/400000 [00:33<00:14, 7828.26it/s] 71%|███████▏  | 285216/400000 [00:33<00:14, 7746.42it/s] 71%|███████▏  | 285993/400000 [00:34<00:14, 7751.81it/s] 72%|███████▏  | 286769/400000 [00:34<00:14, 7718.08it/s] 72%|███████▏  | 287542/400000 [00:34<00:14, 7517.96it/s] 72%|███████▏  | 288308/400000 [00:34<00:14, 7559.37it/s] 72%|███████▏  | 289065/400000 [00:34<00:14, 7546.27it/s] 72%|███████▏  | 289826/400000 [00:34<00:14, 7563.01it/s] 73%|███████▎  | 290596/400000 [00:34<00:14, 7601.77it/s] 73%|███████▎  | 291368/400000 [00:34<00:14, 7636.27it/s] 73%|███████▎  | 292147/400000 [00:34<00:14, 7681.57it/s] 73%|███████▎  | 292916/400000 [00:35<00:13, 7675.75it/s] 73%|███████▎  | 293693/400000 [00:35<00:13, 7702.80it/s] 74%|███████▎  | 294464/400000 [00:35<00:13, 7679.92it/s] 74%|███████▍  | 295245/400000 [00:35<00:13, 7717.59it/s] 74%|███████▍  | 296018/400000 [00:35<00:13, 7720.49it/s] 74%|███████▍  | 296793/400000 [00:35<00:13, 7728.40it/s] 74%|███████▍  | 297570/400000 [00:35<00:13, 7738.90it/s] 75%|███████▍  | 298344/400000 [00:35<00:13, 7640.67it/s] 75%|███████▍  | 299109/400000 [00:35<00:13, 7617.62it/s] 75%|███████▍  | 299872/400000 [00:35<00:13, 7577.38it/s] 75%|███████▌  | 300643/400000 [00:36<00:13, 7615.98it/s] 75%|███████▌  | 301426/400000 [00:36<00:12, 7678.06it/s] 76%|███████▌  | 302195/400000 [00:36<00:12, 7635.72it/s] 76%|███████▌  | 303063/400000 [00:36<00:12, 7921.10it/s] 76%|███████▌  | 303859/400000 [00:36<00:12, 7826.30it/s] 76%|███████▌  | 304675/400000 [00:36<00:12, 7923.43it/s] 76%|███████▋  | 305506/400000 [00:36<00:11, 8033.14it/s] 77%|███████▋  | 306322/400000 [00:36<00:11, 8069.95it/s] 77%|███████▋  | 307158/400000 [00:36<00:11, 8154.52it/s] 77%|███████▋  | 307986/400000 [00:36<00:11, 8191.36it/s] 77%|███████▋  | 308806/400000 [00:37<00:11, 8098.29it/s] 77%|███████▋  | 309629/400000 [00:37<00:11, 8137.19it/s] 78%|███████▊  | 310444/400000 [00:37<00:11, 8032.39it/s] 78%|███████▊  | 311248/400000 [00:37<00:11, 7902.86it/s] 78%|███████▊  | 312040/400000 [00:37<00:11, 7595.40it/s] 78%|███████▊  | 312823/400000 [00:37<00:11, 7663.30it/s] 78%|███████▊  | 313605/400000 [00:37<00:11, 7709.61it/s] 79%|███████▊  | 314398/400000 [00:37<00:11, 7774.25it/s] 79%|███████▉  | 315177/400000 [00:37<00:10, 7736.45it/s] 79%|███████▉  | 315980/400000 [00:37<00:10, 7821.35it/s] 79%|███████▉  | 316777/400000 [00:38<00:10, 7863.33it/s] 79%|███████▉  | 317627/400000 [00:38<00:10, 8041.81it/s] 80%|███████▉  | 318499/400000 [00:38<00:09, 8232.65it/s] 80%|███████▉  | 319366/400000 [00:38<00:09, 8357.34it/s] 80%|████████  | 320204/400000 [00:38<00:09, 8134.84it/s] 80%|████████  | 321029/400000 [00:38<00:09, 8167.40it/s] 80%|████████  | 321857/400000 [00:38<00:09, 8200.40it/s] 81%|████████  | 322679/400000 [00:38<00:09, 8152.41it/s] 81%|████████  | 323532/400000 [00:38<00:09, 8262.01it/s] 81%|████████  | 324360/400000 [00:38<00:09, 8095.13it/s] 81%|████████▏ | 325172/400000 [00:39<00:09, 7968.63it/s] 81%|████████▏ | 325971/400000 [00:39<00:09, 7885.63it/s] 82%|████████▏ | 326761/400000 [00:39<00:09, 7846.18it/s] 82%|████████▏ | 327547/400000 [00:39<00:09, 7816.30it/s] 82%|████████▏ | 328330/400000 [00:39<00:09, 7811.60it/s] 82%|████████▏ | 329112/400000 [00:39<00:09, 7791.16it/s] 82%|████████▏ | 329894/400000 [00:39<00:08, 7797.70it/s] 83%|████████▎ | 330675/400000 [00:39<00:08, 7792.95it/s] 83%|████████▎ | 331455/400000 [00:39<00:08, 7781.10it/s] 83%|████████▎ | 332234/400000 [00:39<00:08, 7750.64it/s] 83%|████████▎ | 333010/400000 [00:40<00:08, 7736.51it/s] 83%|████████▎ | 333784/400000 [00:40<00:08, 7693.32it/s] 84%|████████▎ | 334567/400000 [00:40<00:08, 7731.60it/s] 84%|████████▍ | 335341/400000 [00:40<00:08, 7731.20it/s] 84%|████████▍ | 336119/400000 [00:40<00:08, 7745.61it/s] 84%|████████▍ | 336894/400000 [00:40<00:08, 7730.32it/s] 84%|████████▍ | 337671/400000 [00:40<00:08, 7740.15it/s] 85%|████████▍ | 338449/400000 [00:40<00:07, 7749.52it/s] 85%|████████▍ | 339225/400000 [00:40<00:07, 7751.56it/s] 85%|████████▌ | 340005/400000 [00:40<00:07, 7764.72it/s] 85%|████████▌ | 340782/400000 [00:41<00:07, 7734.02it/s] 85%|████████▌ | 341562/400000 [00:41<00:07, 7752.90it/s] 86%|████████▌ | 342342/400000 [00:41<00:07, 7766.35it/s] 86%|████████▌ | 343124/400000 [00:41<00:07, 7781.95it/s] 86%|████████▌ | 343906/400000 [00:41<00:07, 7792.94it/s] 86%|████████▌ | 344686/400000 [00:41<00:07, 7751.47it/s] 86%|████████▋ | 345477/400000 [00:41<00:06, 7796.11it/s] 87%|████████▋ | 346257/400000 [00:41<00:06, 7777.08it/s] 87%|████████▋ | 347044/400000 [00:41<00:06, 7803.74it/s] 87%|████████▋ | 347833/400000 [00:41<00:06, 7827.11it/s] 87%|████████▋ | 348629/400000 [00:42<00:06, 7864.23it/s] 87%|████████▋ | 349496/400000 [00:42<00:06, 8089.18it/s] 88%|████████▊ | 350406/400000 [00:42<00:05, 8366.20it/s] 88%|████████▊ | 351247/400000 [00:42<00:05, 8158.33it/s] 88%|████████▊ | 352067/400000 [00:42<00:06, 7918.26it/s] 88%|████████▊ | 352863/400000 [00:42<00:06, 7796.67it/s] 88%|████████▊ | 353646/400000 [00:42<00:05, 7758.73it/s] 89%|████████▊ | 354425/400000 [00:42<00:05, 7743.32it/s] 89%|████████▉ | 355201/400000 [00:42<00:05, 7638.21it/s] 89%|████████▉ | 355967/400000 [00:43<00:05, 7614.61it/s] 89%|████████▉ | 356730/400000 [00:43<00:05, 7565.25it/s] 89%|████████▉ | 357488/400000 [00:43<00:05, 7557.70it/s] 90%|████████▉ | 358250/400000 [00:43<00:05, 7575.98it/s] 90%|████████▉ | 359067/400000 [00:43<00:05, 7743.33it/s] 90%|████████▉ | 359888/400000 [00:43<00:05, 7876.70it/s] 90%|█████████ | 360698/400000 [00:43<00:04, 7939.42it/s] 90%|█████████ | 361642/400000 [00:43<00:04, 8335.29it/s] 91%|█████████ | 362537/400000 [00:43<00:04, 8509.74it/s] 91%|█████████ | 363394/400000 [00:43<00:04, 8428.59it/s] 91%|█████████ | 364241/400000 [00:44<00:04, 8408.48it/s] 91%|█████████▏| 365085/400000 [00:44<00:04, 8050.92it/s] 91%|█████████▏| 365917/400000 [00:44<00:04, 8128.57it/s] 92%|█████████▏| 366759/400000 [00:44<00:04, 8211.80it/s] 92%|█████████▏| 367583/400000 [00:44<00:04, 8080.41it/s] 92%|█████████▏| 368394/400000 [00:44<00:03, 7996.54it/s] 92%|█████████▏| 369196/400000 [00:44<00:03, 7891.43it/s] 92%|█████████▏| 369987/400000 [00:44<00:03, 7861.28it/s] 93%|█████████▎| 370775/400000 [00:44<00:03, 7837.18it/s] 93%|█████████▎| 371560/400000 [00:44<00:03, 7819.57it/s] 93%|█████████▎| 372343/400000 [00:45<00:03, 7811.44it/s] 93%|█████████▎| 373125/400000 [00:45<00:03, 7778.61it/s] 93%|█████████▎| 373904/400000 [00:45<00:03, 7780.92it/s] 94%|█████████▎| 374683/400000 [00:45<00:03, 7763.12it/s] 94%|█████████▍| 375460/400000 [00:45<00:03, 7746.28it/s] 94%|█████████▍| 376239/400000 [00:45<00:03, 7758.93it/s] 94%|█████████▍| 377015/400000 [00:45<00:02, 7702.34it/s] 94%|█████████▍| 377786/400000 [00:45<00:02, 7692.55it/s] 95%|█████████▍| 378556/400000 [00:45<00:02, 7694.37it/s] 95%|█████████▍| 379326/400000 [00:45<00:02, 7656.27it/s] 95%|█████████▌| 380093/400000 [00:46<00:02, 7659.20it/s] 95%|█████████▌| 380861/400000 [00:46<00:02, 7664.46it/s] 95%|█████████▌| 381632/400000 [00:46<00:02, 7676.87it/s] 96%|█████████▌| 382402/400000 [00:46<00:02, 7681.59it/s] 96%|█████████▌| 383178/400000 [00:46<00:02, 7702.13it/s] 96%|█████████▌| 383953/400000 [00:46<00:02, 7714.76it/s] 96%|█████████▌| 384725/400000 [00:46<00:01, 7700.79it/s] 96%|█████████▋| 385502/400000 [00:46<00:01, 7720.31it/s] 97%|█████████▋| 386280/400000 [00:46<00:01, 7735.94it/s] 97%|█████████▋| 387057/400000 [00:46<00:01, 7743.81it/s] 97%|█████████▋| 387832/400000 [00:47<00:01, 7723.94it/s] 97%|█████████▋| 388616/400000 [00:47<00:01, 7758.38it/s] 97%|█████████▋| 389507/400000 [00:47<00:01, 8070.12it/s] 98%|█████████▊| 390359/400000 [00:47<00:01, 8199.03it/s] 98%|█████████▊| 391190/400000 [00:47<00:01, 8230.20it/s] 98%|█████████▊| 392033/400000 [00:47<00:00, 8286.05it/s] 98%|█████████▊| 392889/400000 [00:47<00:00, 8364.24it/s] 98%|█████████▊| 393813/400000 [00:47<00:00, 8607.11it/s] 99%|█████████▊| 394677/400000 [00:47<00:00, 8609.29it/s] 99%|█████████▉| 395576/400000 [00:47<00:00, 8718.92it/s] 99%|█████████▉| 396478/400000 [00:48<00:00, 8805.52it/s] 99%|█████████▉| 397360/400000 [00:48<00:00, 8625.72it/s]100%|█████████▉| 398279/400000 [00:48<00:00, 8785.32it/s]100%|█████████▉| 399160/400000 [00:48<00:00, 8764.53it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8249.16it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe24e604d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011018183524687062 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011271272215954834 	 Accuracy: 50

  model saves at 50% accuracy 

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
2020-05-12 19:23:44.272134: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 19:23:44.276192: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 19:23:44.276368: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558c520411f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 19:23:44.276385: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe201b43198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.2526 - accuracy: 0.5270
 2000/25000 [=>............................] - ETA: 10s - loss: 7.3293 - accuracy: 0.5220
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4417 - accuracy: 0.5147 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5363 - accuracy: 0.5085
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.4704 - accuracy: 0.5128
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5414 - accuracy: 0.5082
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5516 - accuracy: 0.5075
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5508 - accuracy: 0.5076
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5716 - accuracy: 0.5062
11000/25000 [============>.................] - ETA: 4s - loss: 7.5872 - accuracy: 0.5052
12000/25000 [=============>................] - ETA: 4s - loss: 7.6091 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6100 - accuracy: 0.5037
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6342 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6529 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6505 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6450 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 10s 382us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fe1b24ea710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fe1b564fe48> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 950ms/step - loss: 1.5241 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4651 - val_crf_viterbi_accuracy: 0.0133

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

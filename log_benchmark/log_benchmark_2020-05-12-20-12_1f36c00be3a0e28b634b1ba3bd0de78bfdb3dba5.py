
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fe146e74f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 20:13:10.099873
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 20:13:10.103883
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 20:13:10.107016
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 20:13:10.110176
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fe152c3e400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354258.0625
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 227355.2031
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 115587.1562
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 54869.8750
Epoch 5/10

1/1 [==============================] - 0s 115ms/step - loss: 28749.5371
Epoch 6/10

1/1 [==============================] - 0s 105ms/step - loss: 17083.5332
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 11238.0977
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 7967.2729
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 6003.4644
Epoch 10/10

1/1 [==============================] - 0s 100ms/step - loss: 4764.5723

  #### Inference Need return ypred, ytrue ######################### 
[[ -0.15528047   0.4546255   -0.5133266   -1.354348    -1.5166847
   -1.690542     0.64332736   1.0123665   -1.2127504    0.96883315
    1.7423023    0.11278772  -0.12537724   1.6358345   -0.16911
   -0.08714283   1.528193     0.15428239  -1.1486104   -2.5972192
    1.1388893   -0.5051328    0.5586263    0.583199    -3.3349996
   -0.7855629    0.8012527    0.78776205   0.20897257  -0.50059944
    0.969253    -0.07416546  -0.23101139   2.1763182   -0.6871727
   -0.03587055  -0.23201054   0.39235163  -1.0727743   -2.0520067
   -0.3257028    1.7505102    0.23656857   0.70048857  -0.49082893
    1.4240769   -0.1556533    1.2520998   -0.36573875   1.3806142
    1.7569233    0.01651073  -0.86126995  -1.8100846   -1.7868938
   -0.04558134   0.4181184    1.4396821    0.8748507   -0.83972484
   -0.31972468  10.734104    10.05018     10.888568    12.50458
   10.216146    12.478028     6.723877    10.255269     9.239946
   10.716033     9.666829    11.692679     9.419496     8.5019045
   11.498955     9.951879     8.049392    12.015432    10.483395
   11.845788    10.043597    12.068027    11.17259     11.444706
    9.049167     9.470979    11.670424     9.892439    11.481848
    8.8790655   10.856517     9.385976     9.529444     9.71164
   10.864907     8.31401      8.555724    10.150407    10.266446
    8.715332     8.771774     8.1838875    9.655439    11.073008
   10.055219     9.609761     8.819208     8.220617    12.2709055
    7.747553     8.523215    10.331599    11.890826     9.476957
   10.718728     8.824672    10.861634    12.333063    10.677646
    1.0561678    0.12101769   0.71496946  -2.3721266   -1.8342712
   -0.05834574  -1.3332559    1.0788693   -0.56187165   0.45169008
   -1.4268357    0.06766826  -2.3051453   -1.500279    -0.4904046
   -0.93932205   0.7553602   -1.1099088   -2.057348    -0.6422758
    1.687637     1.0113575   -1.4249039   -0.25182796  -1.8734165
   -0.01612693  -2.164382    -1.3321486   -0.92638046   0.38772523
    0.10273424  -1.313221    -0.10427999  -0.17400125  -0.69212234
   -0.6409581   -1.4735816   -0.23489556   0.68449914   0.40495658
   -0.69555205   1.1949842   -0.35689813   0.7880423    0.76396275
    0.8437127   -0.9945317   -0.9608702   -0.6332091   -1.143515
    2.3097086    0.8049821    1.7495729    2.7784438   -0.32756114
    0.36991584  -1.2707148    0.14693658   0.3508579    0.3805168
    1.5473952    0.30828798   0.69265455   0.3646134    2.3361769
    0.09139395   0.53561676   0.81655425   2.6322346    2.5637648
    0.18787229   0.20165837   0.4277271    0.05925792   0.94344926
    0.4663458    2.7762818    0.12444186   0.38882685   0.29540306
    1.3510091    1.9931688    0.09848249   0.16364849   3.1178193
    0.08714211   0.56585747   3.700739     0.11330628   0.23298162
    0.4207592    2.564899     1.7870196    0.20014387   0.7138955
    0.9516228    0.3344133    0.61154103   1.8708397    0.2148416
    0.89388835   1.3512254    1.7475734    1.1609861    1.1049421
    1.1055288    0.1663695    0.19861507   3.7873664    1.0384638
    1.754306     1.1475439    1.8758421    1.7938553    0.65148085
    2.6688433    2.3132858    1.3241434    1.7579298    4.347518
    0.77699655   7.9862247   11.496445     8.472391    10.309525
    8.297485    10.773024     8.875556    10.034969    11.653503
   11.650161     9.050905    12.022965    10.458893     9.705562
    8.390272     9.4112625    9.073064     8.410173    11.24897
    9.406374    11.76845      9.404479     9.470201     9.302159
    8.469224     9.730977    12.503291    12.102907    12.256018
    8.298659     8.417874     9.030114    10.307108     9.945802
    9.8418665    8.053611    10.224387    10.656519    10.506439
   12.126338    11.012453    12.677846    10.435838     9.565705
    7.676367    11.030879     9.865099     7.3642683   12.684059
    8.8175335    8.675909     8.877832    10.449333    10.425414
    9.161192    10.547429    10.30544     10.165135     9.427797
    0.3144223    0.5455146    1.4439745    0.07955605   2.564065
    0.82630867   1.6502849    1.9879113    0.44635248   0.41315293
    0.16265726   2.949313     2.0539823    0.53177965   0.7948189
    0.06939864   2.7859468    0.65849596   1.6239352    2.5120492
    3.452608     1.0323768    1.9952124    1.2421174    0.6969634
    3.0558538    1.4334257    1.387167     1.8743093    0.49029887
    1.0554845    1.5023067    1.2976015    0.46422887   0.62279236
    0.4514327    0.17427152   0.80858696   0.74733365   3.1822863
    1.7433937    1.4744971    1.8948848    0.11768842   1.5171609
    0.89681864   1.9053475    0.8039149    3.0728917    2.6925507
    0.07347149   0.09784001   2.3390992    0.85555637   0.605862
    2.8916554    0.15533066   0.7860291    0.53443885   2.0486042
  -10.455838     2.9614701   -8.412972  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 20:13:19.558401
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.8371
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 20:13:19.563440
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8458.13
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 20:13:19.567202
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.6967
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 20:13:19.570859
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -756.485
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140605192013304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140604233249344
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140604233249848
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140604233250352
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140604233250856
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140604233251360

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fe146b3d400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.519818
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.484825
grad_step = 000002, loss = 0.451496
grad_step = 000003, loss = 0.415469
grad_step = 000004, loss = 0.376826
grad_step = 000005, loss = 0.337299
grad_step = 000006, loss = 0.310486
grad_step = 000007, loss = 0.310663
grad_step = 000008, loss = 0.296140
grad_step = 000009, loss = 0.271708
grad_step = 000010, loss = 0.254948
grad_step = 000011, loss = 0.246031
grad_step = 000012, loss = 0.238649
grad_step = 000013, loss = 0.229333
grad_step = 000014, loss = 0.217667
grad_step = 000015, loss = 0.204673
grad_step = 000016, loss = 0.192174
grad_step = 000017, loss = 0.181727
grad_step = 000018, loss = 0.172970
grad_step = 000019, loss = 0.164176
grad_step = 000020, loss = 0.155099
grad_step = 000021, loss = 0.146158
grad_step = 000022, loss = 0.137506
grad_step = 000023, loss = 0.129455
grad_step = 000024, loss = 0.122010
grad_step = 000025, loss = 0.114864
grad_step = 000026, loss = 0.107699
grad_step = 000027, loss = 0.100595
grad_step = 000028, loss = 0.094058
grad_step = 000029, loss = 0.088337
grad_step = 000030, loss = 0.082960
grad_step = 000031, loss = 0.077373
grad_step = 000032, loss = 0.071729
grad_step = 000033, loss = 0.066540
grad_step = 000034, loss = 0.061940
grad_step = 000035, loss = 0.057673
grad_step = 000036, loss = 0.053498
grad_step = 000037, loss = 0.049417
grad_step = 000038, loss = 0.045606
grad_step = 000039, loss = 0.042156
grad_step = 000040, loss = 0.038999
grad_step = 000041, loss = 0.036040
grad_step = 000042, loss = 0.033235
grad_step = 000043, loss = 0.030561
grad_step = 000044, loss = 0.028016
grad_step = 000045, loss = 0.025665
grad_step = 000046, loss = 0.023608
grad_step = 000047, loss = 0.021791
grad_step = 000048, loss = 0.019979
grad_step = 000049, loss = 0.018204
grad_step = 000050, loss = 0.016693
grad_step = 000051, loss = 0.015418
grad_step = 000052, loss = 0.014183
grad_step = 000053, loss = 0.012946
grad_step = 000054, loss = 0.011834
grad_step = 000055, loss = 0.010888
grad_step = 000056, loss = 0.010035
grad_step = 000057, loss = 0.009255
grad_step = 000058, loss = 0.008519
grad_step = 000059, loss = 0.007816
grad_step = 000060, loss = 0.007209
grad_step = 000061, loss = 0.006705
grad_step = 000062, loss = 0.006218
grad_step = 000063, loss = 0.005747
grad_step = 000064, loss = 0.005340
grad_step = 000065, loss = 0.004977
grad_step = 000066, loss = 0.004657
grad_step = 000067, loss = 0.004381
grad_step = 000068, loss = 0.004106
grad_step = 000069, loss = 0.003845
grad_step = 000070, loss = 0.003647
grad_step = 000071, loss = 0.003474
grad_step = 000072, loss = 0.003296
grad_step = 000073, loss = 0.003141
grad_step = 000074, loss = 0.003012
grad_step = 000075, loss = 0.002902
grad_step = 000076, loss = 0.002809
grad_step = 000077, loss = 0.002720
grad_step = 000078, loss = 0.002639
grad_step = 000079, loss = 0.002577
grad_step = 000080, loss = 0.002522
grad_step = 000081, loss = 0.002476
grad_step = 000082, loss = 0.002435
grad_step = 000083, loss = 0.002395
grad_step = 000084, loss = 0.002369
grad_step = 000085, loss = 0.002346
grad_step = 000086, loss = 0.002322
grad_step = 000087, loss = 0.002301
grad_step = 000088, loss = 0.002285
grad_step = 000089, loss = 0.002273
grad_step = 000090, loss = 0.002260
grad_step = 000091, loss = 0.002248
grad_step = 000092, loss = 0.002238
grad_step = 000093, loss = 0.002228
grad_step = 000094, loss = 0.002221
grad_step = 000095, loss = 0.002212
grad_step = 000096, loss = 0.002204
grad_step = 000097, loss = 0.002197
grad_step = 000098, loss = 0.002191
grad_step = 000099, loss = 0.002183
grad_step = 000100, loss = 0.002176
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002170
grad_step = 000102, loss = 0.002164
grad_step = 000103, loss = 0.002158
grad_step = 000104, loss = 0.002152
grad_step = 000105, loss = 0.002146
grad_step = 000106, loss = 0.002141
grad_step = 000107, loss = 0.002135
grad_step = 000108, loss = 0.002129
grad_step = 000109, loss = 0.002124
grad_step = 000110, loss = 0.002119
grad_step = 000111, loss = 0.002113
grad_step = 000112, loss = 0.002108
grad_step = 000113, loss = 0.002103
grad_step = 000114, loss = 0.002098
grad_step = 000115, loss = 0.002092
grad_step = 000116, loss = 0.002087
grad_step = 000117, loss = 0.002082
grad_step = 000118, loss = 0.002077
grad_step = 000119, loss = 0.002072
grad_step = 000120, loss = 0.002068
grad_step = 000121, loss = 0.002064
grad_step = 000122, loss = 0.002064
grad_step = 000123, loss = 0.002067
grad_step = 000124, loss = 0.002064
grad_step = 000125, loss = 0.002053
grad_step = 000126, loss = 0.002040
grad_step = 000127, loss = 0.002041
grad_step = 000128, loss = 0.002044
grad_step = 000129, loss = 0.002037
grad_step = 000130, loss = 0.002026
grad_step = 000131, loss = 0.002021
grad_step = 000132, loss = 0.002023
grad_step = 000133, loss = 0.002021
grad_step = 000134, loss = 0.002014
grad_step = 000135, loss = 0.002005
grad_step = 000136, loss = 0.002002
grad_step = 000137, loss = 0.002002
grad_step = 000138, loss = 0.002000
grad_step = 000139, loss = 0.001995
grad_step = 000140, loss = 0.001988
grad_step = 000141, loss = 0.001983
grad_step = 000142, loss = 0.001981
grad_step = 000143, loss = 0.001980
grad_step = 000144, loss = 0.001978
grad_step = 000145, loss = 0.001974
grad_step = 000146, loss = 0.001969
grad_step = 000147, loss = 0.001964
grad_step = 000148, loss = 0.001959
grad_step = 000149, loss = 0.001955
grad_step = 000150, loss = 0.001952
grad_step = 000151, loss = 0.001950
grad_step = 000152, loss = 0.001948
grad_step = 000153, loss = 0.001947
grad_step = 000154, loss = 0.001948
grad_step = 000155, loss = 0.001949
grad_step = 000156, loss = 0.001952
grad_step = 000157, loss = 0.001948
grad_step = 000158, loss = 0.001940
grad_step = 000159, loss = 0.001926
grad_step = 000160, loss = 0.001917
grad_step = 000161, loss = 0.001916
grad_step = 000162, loss = 0.001919
grad_step = 000163, loss = 0.001921
grad_step = 000164, loss = 0.001917
grad_step = 000165, loss = 0.001909
grad_step = 000166, loss = 0.001900
grad_step = 000167, loss = 0.001893
grad_step = 000168, loss = 0.001889
grad_step = 000169, loss = 0.001888
grad_step = 000170, loss = 0.001889
grad_step = 000171, loss = 0.001890
grad_step = 000172, loss = 0.001892
grad_step = 000173, loss = 0.001892
grad_step = 000174, loss = 0.001894
grad_step = 000175, loss = 0.001890
grad_step = 000176, loss = 0.001882
grad_step = 000177, loss = 0.001869
grad_step = 000178, loss = 0.001858
grad_step = 000179, loss = 0.001851
grad_step = 000180, loss = 0.001849
grad_step = 000181, loss = 0.001849
grad_step = 000182, loss = 0.001853
grad_step = 000183, loss = 0.001867
grad_step = 000184, loss = 0.001891
grad_step = 000185, loss = 0.001937
grad_step = 000186, loss = 0.001935
grad_step = 000187, loss = 0.001898
grad_step = 000188, loss = 0.001833
grad_step = 000189, loss = 0.001838
grad_step = 000190, loss = 0.001881
grad_step = 000191, loss = 0.001862
grad_step = 000192, loss = 0.001820
grad_step = 000193, loss = 0.001821
grad_step = 000194, loss = 0.001846
grad_step = 000195, loss = 0.001845
grad_step = 000196, loss = 0.001811
grad_step = 000197, loss = 0.001805
grad_step = 000198, loss = 0.001824
grad_step = 000199, loss = 0.001825
grad_step = 000200, loss = 0.001806
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001790
grad_step = 000202, loss = 0.001795
grad_step = 000203, loss = 0.001805
grad_step = 000204, loss = 0.001801
grad_step = 000205, loss = 0.001786
grad_step = 000206, loss = 0.001777
grad_step = 000207, loss = 0.001779
grad_step = 000208, loss = 0.001785
grad_step = 000209, loss = 0.001782
grad_step = 000210, loss = 0.001773
grad_step = 000211, loss = 0.001764
grad_step = 000212, loss = 0.001762
grad_step = 000213, loss = 0.001765
grad_step = 000214, loss = 0.001766
grad_step = 000215, loss = 0.001762
grad_step = 000216, loss = 0.001755
grad_step = 000217, loss = 0.001749
grad_step = 000218, loss = 0.001746
grad_step = 000219, loss = 0.001746
grad_step = 000220, loss = 0.001746
grad_step = 000221, loss = 0.001746
grad_step = 000222, loss = 0.001743
grad_step = 000223, loss = 0.001739
grad_step = 000224, loss = 0.001735
grad_step = 000225, loss = 0.001731
grad_step = 000226, loss = 0.001728
grad_step = 000227, loss = 0.001725
grad_step = 000228, loss = 0.001724
grad_step = 000229, loss = 0.001723
grad_step = 000230, loss = 0.001722
grad_step = 000231, loss = 0.001722
grad_step = 000232, loss = 0.001721
grad_step = 000233, loss = 0.001720
grad_step = 000234, loss = 0.001719
grad_step = 000235, loss = 0.001717
grad_step = 000236, loss = 0.001716
grad_step = 000237, loss = 0.001715
grad_step = 000238, loss = 0.001713
grad_step = 000239, loss = 0.001711
grad_step = 000240, loss = 0.001709
grad_step = 000241, loss = 0.001706
grad_step = 000242, loss = 0.001703
grad_step = 000243, loss = 0.001700
grad_step = 000244, loss = 0.001697
grad_step = 000245, loss = 0.001694
grad_step = 000246, loss = 0.001691
grad_step = 000247, loss = 0.001689
grad_step = 000248, loss = 0.001687
grad_step = 000249, loss = 0.001685
grad_step = 000250, loss = 0.001683
grad_step = 000251, loss = 0.001681
grad_step = 000252, loss = 0.001680
grad_step = 000253, loss = 0.001678
grad_step = 000254, loss = 0.001676
grad_step = 000255, loss = 0.001675
grad_step = 000256, loss = 0.001673
grad_step = 000257, loss = 0.001671
grad_step = 000258, loss = 0.001670
grad_step = 000259, loss = 0.001668
grad_step = 000260, loss = 0.001667
grad_step = 000261, loss = 0.001666
grad_step = 000262, loss = 0.001666
grad_step = 000263, loss = 0.001669
grad_step = 000264, loss = 0.001679
grad_step = 000265, loss = 0.001704
grad_step = 000266, loss = 0.001749
grad_step = 000267, loss = 0.001807
grad_step = 000268, loss = 0.001803
grad_step = 000269, loss = 0.001733
grad_step = 000270, loss = 0.001659
grad_step = 000271, loss = 0.001673
grad_step = 000272, loss = 0.001716
grad_step = 000273, loss = 0.001689
grad_step = 000274, loss = 0.001651
grad_step = 000275, loss = 0.001672
grad_step = 000276, loss = 0.001676
grad_step = 000277, loss = 0.001650
grad_step = 000278, loss = 0.001651
grad_step = 000279, loss = 0.001662
grad_step = 000280, loss = 0.001648
grad_step = 000281, loss = 0.001639
grad_step = 000282, loss = 0.001646
grad_step = 000283, loss = 0.001647
grad_step = 000284, loss = 0.001634
grad_step = 000285, loss = 0.001629
grad_step = 000286, loss = 0.001638
grad_step = 000287, loss = 0.001635
grad_step = 000288, loss = 0.001622
grad_step = 000289, loss = 0.001621
grad_step = 000290, loss = 0.001628
grad_step = 000291, loss = 0.001623
grad_step = 000292, loss = 0.001615
grad_step = 000293, loss = 0.001613
grad_step = 000294, loss = 0.001614
grad_step = 000295, loss = 0.001613
grad_step = 000296, loss = 0.001610
grad_step = 000297, loss = 0.001606
grad_step = 000298, loss = 0.001602
grad_step = 000299, loss = 0.001602
grad_step = 000300, loss = 0.001603
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001599
grad_step = 000302, loss = 0.001595
grad_step = 000303, loss = 0.001593
grad_step = 000304, loss = 0.001592
grad_step = 000305, loss = 0.001590
grad_step = 000306, loss = 0.001589
grad_step = 000307, loss = 0.001586
grad_step = 000308, loss = 0.001583
grad_step = 000309, loss = 0.001580
grad_step = 000310, loss = 0.001578
grad_step = 000311, loss = 0.001577
grad_step = 000312, loss = 0.001575
grad_step = 000313, loss = 0.001572
grad_step = 000314, loss = 0.001570
grad_step = 000315, loss = 0.001567
grad_step = 000316, loss = 0.001564
grad_step = 000317, loss = 0.001563
grad_step = 000318, loss = 0.001562
grad_step = 000319, loss = 0.001559
grad_step = 000320, loss = 0.001554
grad_step = 000321, loss = 0.001552
grad_step = 000322, loss = 0.001551
grad_step = 000323, loss = 0.001550
grad_step = 000324, loss = 0.001548
grad_step = 000325, loss = 0.001546
grad_step = 000326, loss = 0.001544
grad_step = 000327, loss = 0.001543
grad_step = 000328, loss = 0.001544
grad_step = 000329, loss = 0.001548
grad_step = 000330, loss = 0.001560
grad_step = 000331, loss = 0.001579
grad_step = 000332, loss = 0.001597
grad_step = 000333, loss = 0.001601
grad_step = 000334, loss = 0.001567
grad_step = 000335, loss = 0.001535
grad_step = 000336, loss = 0.001527
grad_step = 000337, loss = 0.001536
grad_step = 000338, loss = 0.001548
grad_step = 000339, loss = 0.001538
grad_step = 000340, loss = 0.001527
grad_step = 000341, loss = 0.001520
grad_step = 000342, loss = 0.001518
grad_step = 000343, loss = 0.001523
grad_step = 000344, loss = 0.001522
grad_step = 000345, loss = 0.001516
grad_step = 000346, loss = 0.001506
grad_step = 000347, loss = 0.001498
grad_step = 000348, loss = 0.001498
grad_step = 000349, loss = 0.001506
grad_step = 000350, loss = 0.001516
grad_step = 000351, loss = 0.001521
grad_step = 000352, loss = 0.001524
grad_step = 000353, loss = 0.001514
grad_step = 000354, loss = 0.001506
grad_step = 000355, loss = 0.001492
grad_step = 000356, loss = 0.001483
grad_step = 000357, loss = 0.001478
grad_step = 000358, loss = 0.001478
grad_step = 000359, loss = 0.001483
grad_step = 000360, loss = 0.001491
grad_step = 000361, loss = 0.001506
grad_step = 000362, loss = 0.001513
grad_step = 000363, loss = 0.001530
grad_step = 000364, loss = 0.001498
grad_step = 000365, loss = 0.001474
grad_step = 000366, loss = 0.001462
grad_step = 000367, loss = 0.001471
grad_step = 000368, loss = 0.001489
grad_step = 000369, loss = 0.001486
grad_step = 000370, loss = 0.001484
grad_step = 000371, loss = 0.001469
grad_step = 000372, loss = 0.001459
grad_step = 000373, loss = 0.001455
grad_step = 000374, loss = 0.001457
grad_step = 000375, loss = 0.001462
grad_step = 000376, loss = 0.001467
grad_step = 000377, loss = 0.001479
grad_step = 000378, loss = 0.001481
grad_step = 000379, loss = 0.001490
grad_step = 000380, loss = 0.001471
grad_step = 000381, loss = 0.001460
grad_step = 000382, loss = 0.001457
grad_step = 000383, loss = 0.001457
grad_step = 000384, loss = 0.001459
grad_step = 000385, loss = 0.001454
grad_step = 000386, loss = 0.001449
grad_step = 000387, loss = 0.001443
grad_step = 000388, loss = 0.001445
grad_step = 000389, loss = 0.001443
grad_step = 000390, loss = 0.001438
grad_step = 000391, loss = 0.001437
grad_step = 000392, loss = 0.001433
grad_step = 000393, loss = 0.001429
grad_step = 000394, loss = 0.001425
grad_step = 000395, loss = 0.001430
grad_step = 000396, loss = 0.001450
grad_step = 000397, loss = 0.001465
grad_step = 000398, loss = 0.001509
grad_step = 000399, loss = 0.001483
grad_step = 000400, loss = 0.001445
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001413
grad_step = 000402, loss = 0.001430
grad_step = 000403, loss = 0.001465
grad_step = 000404, loss = 0.001452
grad_step = 000405, loss = 0.001429
grad_step = 000406, loss = 0.001404
grad_step = 000407, loss = 0.001400
grad_step = 000408, loss = 0.001405
grad_step = 000409, loss = 0.001418
grad_step = 000410, loss = 0.001441
grad_step = 000411, loss = 0.001444
grad_step = 000412, loss = 0.001448
grad_step = 000413, loss = 0.001422
grad_step = 000414, loss = 0.001406
grad_step = 000415, loss = 0.001403
grad_step = 000416, loss = 0.001414
grad_step = 000417, loss = 0.001434
grad_step = 000418, loss = 0.001437
grad_step = 000419, loss = 0.001421
grad_step = 000420, loss = 0.001396
grad_step = 000421, loss = 0.001381
grad_step = 000422, loss = 0.001380
grad_step = 000423, loss = 0.001388
grad_step = 000424, loss = 0.001400
grad_step = 000425, loss = 0.001398
grad_step = 000426, loss = 0.001390
grad_step = 000427, loss = 0.001376
grad_step = 000428, loss = 0.001375
grad_step = 000429, loss = 0.001386
grad_step = 000430, loss = 0.001414
grad_step = 000431, loss = 0.001468
grad_step = 000432, loss = 0.001570
grad_step = 000433, loss = 0.001525
grad_step = 000434, loss = 0.001445
grad_step = 000435, loss = 0.001372
grad_step = 000436, loss = 0.001407
grad_step = 000437, loss = 0.001466
grad_step = 000438, loss = 0.001426
grad_step = 000439, loss = 0.001375
grad_step = 000440, loss = 0.001391
grad_step = 000441, loss = 0.001421
grad_step = 000442, loss = 0.001434
grad_step = 000443, loss = 0.001392
grad_step = 000444, loss = 0.001367
grad_step = 000445, loss = 0.001363
grad_step = 000446, loss = 0.001378
grad_step = 000447, loss = 0.001391
grad_step = 000448, loss = 0.001377
grad_step = 000449, loss = 0.001366
grad_step = 000450, loss = 0.001367
grad_step = 000451, loss = 0.001363
grad_step = 000452, loss = 0.001363
grad_step = 000453, loss = 0.001361
grad_step = 000454, loss = 0.001356
grad_step = 000455, loss = 0.001354
grad_step = 000456, loss = 0.001359
grad_step = 000457, loss = 0.001355
grad_step = 000458, loss = 0.001347
grad_step = 000459, loss = 0.001337
grad_step = 000460, loss = 0.001327
grad_step = 000461, loss = 0.001326
grad_step = 000462, loss = 0.001333
grad_step = 000463, loss = 0.001339
grad_step = 000464, loss = 0.001347
grad_step = 000465, loss = 0.001359
grad_step = 000466, loss = 0.001371
grad_step = 000467, loss = 0.001391
grad_step = 000468, loss = 0.001378
grad_step = 000469, loss = 0.001359
grad_step = 000470, loss = 0.001326
grad_step = 000471, loss = 0.001313
grad_step = 000472, loss = 0.001323
grad_step = 000473, loss = 0.001340
grad_step = 000474, loss = 0.001355
grad_step = 000475, loss = 0.001342
grad_step = 000476, loss = 0.001328
grad_step = 000477, loss = 0.001312
grad_step = 000478, loss = 0.001306
grad_step = 000479, loss = 0.001307
grad_step = 000480, loss = 0.001312
grad_step = 000481, loss = 0.001319
grad_step = 000482, loss = 0.001323
grad_step = 000483, loss = 0.001332
grad_step = 000484, loss = 0.001337
grad_step = 000485, loss = 0.001351
grad_step = 000486, loss = 0.001357
grad_step = 000487, loss = 0.001361
grad_step = 000488, loss = 0.001355
grad_step = 000489, loss = 0.001333
grad_step = 000490, loss = 0.001308
grad_step = 000491, loss = 0.001289
grad_step = 000492, loss = 0.001292
grad_step = 000493, loss = 0.001310
grad_step = 000494, loss = 0.001326
grad_step = 000495, loss = 0.001331
grad_step = 000496, loss = 0.001328
grad_step = 000497, loss = 0.001320
grad_step = 000498, loss = 0.001318
grad_step = 000499, loss = 0.001305
grad_step = 000500, loss = 0.001296
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001279
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

  date_run                              2020-05-12 20:13:40.390137
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.199451
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 20:13:40.396119
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0886329
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 20:13:40.404293
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.129929
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 20:13:40.409770
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.346809
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
0   2020-05-12 20:13:10.099873  ...    mean_absolute_error
1   2020-05-12 20:13:10.103883  ...     mean_squared_error
2   2020-05-12 20:13:10.107016  ...  median_absolute_error
3   2020-05-12 20:13:10.110176  ...               r2_score
4   2020-05-12 20:13:19.558401  ...    mean_absolute_error
5   2020-05-12 20:13:19.563440  ...     mean_squared_error
6   2020-05-12 20:13:19.567202  ...  median_absolute_error
7   2020-05-12 20:13:19.570859  ...               r2_score
8   2020-05-12 20:13:40.390137  ...    mean_absolute_error
9   2020-05-12 20:13:40.396119  ...     mean_squared_error
10  2020-05-12 20:13:40.404293  ...  median_absolute_error
11  2020-05-12 20:13:40.409770  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2727936/9912422 [00:00<00:00, 27202984.43it/s] 99%|█████████▉| 9822208/9912422 [00:00<00:00, 33370705.71it/s]9920512it [00:00, 29767525.98it/s]                             
0it [00:00, ?it/s]32768it [00:00, 572481.02it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 450286.85it/s]1654784it [00:00, 11165701.67it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 12605.19it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0041bfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0c4b04a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab244a0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab2e1eeb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab23a4080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0c4b04a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2aafbc8ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab2e1eeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab23616a0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2b0c4b04a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2ab221b6a0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd66913e240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=ac54c6480fb5231ddaeba43c03b40970e76142fd3d1b6cd2b28c79b6410e45ae
  Stored in directory: /tmp/pip-ephem-wheel-cache-x3rs52qx/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd600f396d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3874816/17464789 [=====>........................] - ETA: 0s
10797056/17464789 [=================>............] - ETA: 0s
15622144/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 20:15:07.862330: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 20:15:07.866329: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 20:15:07.867491: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cc9eae7680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 20:15:07.867518: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6053 - accuracy: 0.5040
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5555 - accuracy: 0.5073
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6181 - accuracy: 0.5032
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5612 - accuracy: 0.5069
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5712 - accuracy: 0.5062
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5792 - accuracy: 0.5057
11000/25000 [============>.................] - ETA: 3s - loss: 7.5969 - accuracy: 0.5045
12000/25000 [=============>................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6018 - accuracy: 0.5042
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6108 - accuracy: 0.5036
15000/25000 [=================>............] - ETA: 2s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6379 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6549 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6841 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6896 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 7s 287us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 20:15:21.834420
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 20:15:21.834420  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:43:07, 9.69kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:32:06, 13.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:19:41, 19.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<8:38:06, 27.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.25M/862M [00:01<6:02:25, 39.5kB/s].vector_cache/glove.6B.zip:   1%|          | 5.89M/862M [00:01<4:12:44, 56.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.74M/862M [00:01<2:56:14, 80.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.2M/862M [00:01<2:02:48, 115kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:25:39, 164kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<59:56, 234kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:01<41:49, 333kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:02<29:12, 474kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:02<20:28, 674kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<14:20, 957kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:02<10:05, 1.35MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<07:06, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<05:19, 2.54MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<05:37, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<08:20, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<05:06, 2.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<07:10, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<06:22, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<04:47, 2.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<06:28, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<07:14, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<05:39, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:09<04:09, 3.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<08:03, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<06:59, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<05:10, 2.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<06:42, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<07:21, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<06:11, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<05:41, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:14<04:18, 3.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<06:04, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<06:54, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<05:30, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<04:00, 3.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<1:35:25, 136kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<1:08:04, 190kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<47:53, 270kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<36:27, 353kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<26:49, 480kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<19:04, 674kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<16:20, 784kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<14:03, 912kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<10:23, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:22<07:25, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<12:30, 1.02MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<09:51, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<07:18, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:15, 2.42MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<27:38, 459kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<21:58, 577kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<15:56, 795kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<11:15, 1.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<21:38, 583kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:25, 767kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<11:47, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:11, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:06, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:40, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:37, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:35, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:55, 2.53MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:22, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:58, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:30, 2.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:05, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:18, 2.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:14, 3.80MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:08, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:19, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:37, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<03:51, 3.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<21:48, 556kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:29, 736kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<11:47, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:04, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:06, 1.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:40, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:18, 1.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:40, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:56, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:18, 2.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:42, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:21, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:23, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<06:33, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:59, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:57, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<11:24:14, 17.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<7:59:53, 24.6kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<5:35:27, 35.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<3:56:51, 49.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<2:46:54, 70.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<1:56:49, 100kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:24:16, 138kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<1:01:20, 190kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<43:28, 268kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<30:26, 381kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<11:27:14, 16.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<8:01:58, 24.0kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<5:36:55, 34.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<3:57:49, 48.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<2:48:46, 68.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<1:58:36, 96.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<1:22:48, 138kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<2:19:46, 81.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<1:38:46, 116kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<1:09:15, 165kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<48:29, 235kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:41:41, 112kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:13:30, 155kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<51:57, 219kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<36:20, 311kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<3:09:57, 59.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<2:14:01, 84.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<1:33:52, 120kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:08:11, 165kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<50:01, 225kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<35:28, 316kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<24:53, 449kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<24:29, 456kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<18:17, 610kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<13:03, 853kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:43, 947kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:19, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<06:46, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:20, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:24, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:44, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:47, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:24, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:57, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:39, 2.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:59, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:20, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:58, 2.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:18, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:55, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:23, 3.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<37:08, 290kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<27:04, 397kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<19:10, 560kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<13:30, 791kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<51:51, 206kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<38:28, 278kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<27:21, 390kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<19:25, 549kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<15:57, 666kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<12:15, 866kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<08:47, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<06:16, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<46:13, 228kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<33:22, 316kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<23:35, 446kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<18:55, 554kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<15:23, 681kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:18, 926kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<09:33, 1.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:35, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:44, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:06, 2.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<29:57, 345kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<22:00, 470kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<15:35, 662kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<13:19, 771kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<11:26, 898kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:31, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<06:02, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<35:34, 287kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<25:56, 393kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<18:20, 555kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<15:10, 668kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<11:38, 870kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<08:23, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<08:13, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:46, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:59, 2.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:50, 1.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:08, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:48, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:27, 2.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<9:35:16, 17.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<6:43:29, 24.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<4:41:57, 35.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<3:18:57, 49.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<2:20:11, 70.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<1:38:05, 100kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<1:10:43, 139kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<51:30, 190kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<36:30, 268kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<26:59, 360kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<19:52, 489kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<14:07, 686kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<12:05, 798kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<10:26, 924kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:44, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:31, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:47, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:09, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:14, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:53, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<06:04, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:39, 2.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:27, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:10, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:35, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:25, 2.75MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:35, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:07, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:04, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<02:57, 3.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<1:18:06, 119kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<55:35, 168kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<39:00, 238kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<29:22, 315kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<22:26, 412kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<16:09, 572kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<12:44, 721kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<09:51, 930kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<07:07, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<07:06, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:51, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:15, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:07, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:22, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:17, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:10<02:24, 3.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<37:46, 238kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<27:19, 328kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<19:17, 464kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<15:33, 572kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<11:47, 755kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<08:27, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<07:59, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<06:29, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:44, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:21, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:31, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:16, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<03:03, 2.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<58:53, 148kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<42:05, 207kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<29:36, 293kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<22:39, 381kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<17:40, 488kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<12:47, 674kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<10:17, 832kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:04, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:51, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:07, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:45, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:37, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:58, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:04, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:36, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:45, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:02, 4.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<21:59, 377kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<17:05, 485kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<12:21, 669kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<08:40, 947kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<59:25, 138kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<42:24, 194kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<29:48, 275kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<22:41, 359kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<17:32, 465kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<12:37, 645kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<08:57, 906kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<08:52, 912kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<07:02, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:07, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:27, 1.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:37, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:25, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:16, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:48, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:49, 2.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:51, 2.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:18, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:21, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:26, 3.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:16, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:30, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:08, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:41, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:46, 2.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:45, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:11, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:17, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:26, 3.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:56, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:32, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:40, 2.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:37, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:17, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:27, 3.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:29, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:12, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:25, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:54, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:06, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:14, 3.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<28:19, 259kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<20:32, 356kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<14:29, 503kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<11:48, 615kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<09:47, 741kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:13, 1.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<08:24, 856kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:37, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<04:48, 1.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:01, 1.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:15, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:08, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:51, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:33, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:26, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:06, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:20, 2.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:16, 2.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:41, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:55, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:08, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:46, 2.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:05, 3.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<01:34, 4.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<32:12, 211kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<23:55, 283kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<17:03, 397kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<12:56, 519kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<09:44, 688kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<06:57, 959kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:24, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:02, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:41, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:08, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:33, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:38, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:23, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:41, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:54, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:06, 3.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:44, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:41, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:26, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:52, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:04, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:12, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:42, 723kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<06:37, 949kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<27:29, 227kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<19:52, 313kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<14:00, 443kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<11:12, 550kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<09:05, 678kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:40, 922kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<04:42, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<5:52:48, 17.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<4:07:20, 24.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<2:52:33, 35.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<2:01:29, 49.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<1:25:33, 70.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<59:46, 100kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<43:00, 139kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<31:18, 190kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<22:07, 269kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<15:29, 381kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<13:09, 448kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<09:48, 600kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:59, 838kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<06:13, 934kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<05:32, 1.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:08, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:56, 1.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<07:42, 747kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<05:53, 976kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<04:15, 1.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:02, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<35:58, 158kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<25:44, 221kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<18:04, 313kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<13:54, 404kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<10:52, 516kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<07:52, 711kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:21, 872kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:01, 1.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:38, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:49, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:47, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:55, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:54, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:34, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:56, 2.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:36, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:54, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:18, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<01:40, 3.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<44:17, 119kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<31:30, 167kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<22:03, 238kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<16:33, 314kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<12:38, 411kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<09:05, 571kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<06:21, 807kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<4:57:56, 17.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<3:28:49, 24.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<2:25:34, 35.1kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<1:42:22, 49.5kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<1:12:38, 69.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<50:58, 99.1kB/s]  .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<35:28, 141kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<27:37, 181kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<19:44, 253kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<14:01, 355kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<09:46, 504kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<27:47, 177kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<19:55, 247kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<14:00, 350kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<10:52, 447kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<08:35, 565kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<06:12, 780kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<04:25, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:33, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:40, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:40, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:58, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:33, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:54, 2.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:24, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:38, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:02, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:29, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:39, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:20, 1.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:44, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:16, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:03, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:31, 2.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:07, 3.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<33:53, 131kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<24:09, 184kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<16:55, 261kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<12:47, 343kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<09:49, 445kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<07:05, 616kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:36, 770kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:21, 986kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:08, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:10, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:39, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:56, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:19, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:23, 2.93MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<30:18, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:35, 190kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<15:07, 269kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<11:26, 353kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:24, 480kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<05:56, 674kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:03, 784kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:56, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:50, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:53, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:49, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:08, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:32, 2.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:46, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:49, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:22, 2.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:49, 2.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:39, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:14, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:43, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:56, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:32, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:38, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:30, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:08, 3.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:36, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:50, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:26, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:04, 3.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:48, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:36, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:12, 2.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:37, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:48, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:24, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:01, 3.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:48, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:10, 2.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:33, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:43, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:22, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:58, 3.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:10, 595kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:55, 782kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:48, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:38, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:09, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:34, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:46, 1.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:50, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:26, 2.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:01, 2.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:47, 496kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<04:19, 662kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:03, 927kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<02:46, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:10, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:34, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:07, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<21:10, 129kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<15:22, 178kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<10:49, 251kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<07:32, 357kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<06:11, 431kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<04:35, 579kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:15, 810kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:51, 910kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:31, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:53, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:42, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:27, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:03, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:18, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:09, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:52, 2.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:10, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:03, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:47, 3.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:05, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:59, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:44, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:11, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:55, 2.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:41, 3.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:03, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:55, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:45, 2.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:32, 3.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<08:53, 238kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:25, 329kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<04:29, 465kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:34, 573kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:55, 700kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:07, 959kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:29, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:49, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:28, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:04, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:10, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:13, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:40, 2.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:10, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:00, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:44, 2.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:55, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:00, 1.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:33, 3.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:31, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:14, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:53, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:59, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:38, 2.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:48, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:32, 2.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:47, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<00:26, 3.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<1:23:07, 17.2kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<58:04, 24.6kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<39:56, 35.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<27:33, 49.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<19:20, 70.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<13:18, 100kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<09:22, 138kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<06:39, 194kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<04:35, 275kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<03:24, 359kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<02:38, 464kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:52, 643kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:17, 907kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:30, 765kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:10, 981kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:49, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:47, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:36, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:22, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:28, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:18, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:25, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:28, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:21, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:15, 3.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:34, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:19, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:13, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:45, 890kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:35, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:25, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:18, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:11, 2.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 3.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 3.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:03, 3.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.68MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 879/400000 [00:00<00:45, 8786.45it/s]  0%|          | 1748/400000 [00:00<00:45, 8755.51it/s]  1%|          | 2618/400000 [00:00<00:45, 8736.50it/s]  1%|          | 3433/400000 [00:00<00:46, 8549.83it/s]  1%|          | 4305/400000 [00:00<00:46, 8599.46it/s]  1%|▏         | 5177/400000 [00:00<00:45, 8634.30it/s]  2%|▏         | 6053/400000 [00:00<00:45, 8671.24it/s]  2%|▏         | 6911/400000 [00:00<00:45, 8643.48it/s]  2%|▏         | 7725/400000 [00:00<00:46, 8378.44it/s]  2%|▏         | 8592/400000 [00:01<00:46, 8463.48it/s]  2%|▏         | 9449/400000 [00:01<00:45, 8494.03it/s]  3%|▎         | 10331/400000 [00:01<00:45, 8588.64it/s]  3%|▎         | 11204/400000 [00:01<00:45, 8628.08it/s]  3%|▎         | 12089/400000 [00:01<00:44, 8692.52it/s]  3%|▎         | 12954/400000 [00:01<00:44, 8677.12it/s]  3%|▎         | 13819/400000 [00:01<00:44, 8658.94it/s]  4%|▎         | 14689/400000 [00:01<00:44, 8669.10it/s]  4%|▍         | 15555/400000 [00:01<00:45, 8500.65it/s]  4%|▍         | 16405/400000 [00:01<00:45, 8366.08it/s]  4%|▍         | 17242/400000 [00:02<00:46, 8272.28it/s]  5%|▍         | 18130/400000 [00:02<00:45, 8444.11it/s]  5%|▍         | 18990/400000 [00:02<00:44, 8489.40it/s]  5%|▍         | 19869/400000 [00:02<00:44, 8577.37it/s]  5%|▌         | 20730/400000 [00:02<00:44, 8585.62it/s]  5%|▌         | 21625/400000 [00:02<00:43, 8689.95it/s]  6%|▌         | 22495/400000 [00:02<00:43, 8603.89it/s]  6%|▌         | 23357/400000 [00:02<00:43, 8602.62it/s]  6%|▌         | 24218/400000 [00:02<00:44, 8521.70it/s]  6%|▋         | 25071/400000 [00:02<00:44, 8449.36it/s]  6%|▋         | 25958/400000 [00:03<00:43, 8571.06it/s]  7%|▋         | 26816/400000 [00:03<00:43, 8570.44it/s]  7%|▋         | 27676/400000 [00:03<00:43, 8578.90it/s]  7%|▋         | 28535/400000 [00:03<00:43, 8549.74it/s]  7%|▋         | 29391/400000 [00:03<00:43, 8514.46it/s]  8%|▊         | 30247/400000 [00:03<00:43, 8524.39it/s]  8%|▊         | 31100/400000 [00:03<00:43, 8486.61it/s]  8%|▊         | 31974/400000 [00:03<00:42, 8559.36it/s]  8%|▊         | 32838/400000 [00:03<00:42, 8583.31it/s]  8%|▊         | 33702/400000 [00:03<00:42, 8600.08it/s]  9%|▊         | 34585/400000 [00:04<00:42, 8667.45it/s]  9%|▉         | 35455/400000 [00:04<00:42, 8676.56it/s]  9%|▉         | 36323/400000 [00:04<00:42, 8546.37it/s]  9%|▉         | 37179/400000 [00:04<00:42, 8517.92it/s] 10%|▉         | 38060/400000 [00:04<00:42, 8601.12it/s] 10%|▉         | 38932/400000 [00:04<00:41, 8635.84it/s] 10%|▉         | 39796/400000 [00:04<00:42, 8562.94it/s] 10%|█         | 40661/400000 [00:04<00:41, 8588.70it/s] 10%|█         | 41521/400000 [00:04<00:42, 8534.26it/s] 11%|█         | 42388/400000 [00:04<00:41, 8573.37it/s] 11%|█         | 43246/400000 [00:05<00:42, 8460.79it/s] 11%|█         | 44148/400000 [00:05<00:41, 8618.32it/s] 11%|█▏        | 45029/400000 [00:05<00:40, 8674.40it/s] 11%|█▏        | 45898/400000 [00:05<00:40, 8668.42it/s] 12%|█▏        | 46766/400000 [00:05<00:41, 8463.15it/s] 12%|█▏        | 47632/400000 [00:05<00:41, 8521.02it/s] 12%|█▏        | 48486/400000 [00:05<00:41, 8518.19it/s] 12%|█▏        | 49360/400000 [00:05<00:40, 8581.27it/s] 13%|█▎        | 50219/400000 [00:05<00:40, 8548.76it/s] 13%|█▎        | 51092/400000 [00:05<00:40, 8600.63it/s] 13%|█▎        | 51968/400000 [00:06<00:40, 8645.10it/s] 13%|█▎        | 52859/400000 [00:06<00:39, 8721.66it/s] 13%|█▎        | 53732/400000 [00:06<00:39, 8660.12it/s] 14%|█▎        | 54599/400000 [00:06<00:40, 8617.59it/s] 14%|█▍        | 55477/400000 [00:06<00:39, 8663.84it/s] 14%|█▍        | 56344/400000 [00:06<00:39, 8653.68it/s] 14%|█▍        | 57210/400000 [00:06<00:39, 8605.91it/s] 15%|█▍        | 58073/400000 [00:06<00:39, 8611.41it/s] 15%|█▍        | 58935/400000 [00:06<00:39, 8582.25it/s] 15%|█▍        | 59794/400000 [00:06<00:40, 8501.07it/s] 15%|█▌        | 60661/400000 [00:07<00:39, 8549.23it/s] 15%|█▌        | 61517/400000 [00:07<00:39, 8502.56it/s] 16%|█▌        | 62397/400000 [00:07<00:39, 8589.29it/s] 16%|█▌        | 63257/400000 [00:07<00:39, 8575.02it/s] 16%|█▌        | 64115/400000 [00:07<00:40, 8340.10it/s] 16%|█▌        | 64951/400000 [00:07<00:40, 8296.71it/s] 16%|█▋        | 65782/400000 [00:07<00:40, 8276.00it/s] 17%|█▋        | 66656/400000 [00:07<00:39, 8409.70it/s] 17%|█▋        | 67499/400000 [00:07<00:39, 8405.02it/s] 17%|█▋        | 68356/400000 [00:07<00:39, 8452.30it/s] 17%|█▋        | 69221/400000 [00:08<00:38, 8509.35it/s] 18%|█▊        | 70101/400000 [00:08<00:38, 8592.31it/s] 18%|█▊        | 70961/400000 [00:08<00:38, 8573.90it/s] 18%|█▊        | 71819/400000 [00:08<00:39, 8401.61it/s] 18%|█▊        | 72680/400000 [00:08<00:38, 8463.01it/s] 18%|█▊        | 73548/400000 [00:08<00:38, 8526.54it/s] 19%|█▊        | 74429/400000 [00:08<00:37, 8606.53it/s] 19%|█▉        | 75291/400000 [00:08<00:38, 8514.53it/s] 19%|█▉        | 76144/400000 [00:08<00:38, 8362.80it/s] 19%|█▉        | 77021/400000 [00:09<00:38, 8480.93it/s] 19%|█▉        | 77879/400000 [00:09<00:37, 8498.56it/s] 20%|█▉        | 78742/400000 [00:09<00:37, 8535.78it/s] 20%|█▉        | 79597/400000 [00:09<00:38, 8355.24it/s] 20%|██        | 80453/400000 [00:09<00:37, 8411.96it/s] 20%|██        | 81296/400000 [00:09<00:38, 8359.82it/s] 21%|██        | 82158/400000 [00:09<00:37, 8434.85it/s] 21%|██        | 83026/400000 [00:09<00:37, 8504.83it/s] 21%|██        | 83884/400000 [00:09<00:37, 8524.87it/s] 21%|██        | 84737/400000 [00:09<00:37, 8473.63it/s] 21%|██▏       | 85615/400000 [00:10<00:36, 8562.09it/s] 22%|██▏       | 86490/400000 [00:10<00:36, 8617.21it/s] 22%|██▏       | 87370/400000 [00:10<00:36, 8670.76it/s] 22%|██▏       | 88238/400000 [00:10<00:36, 8624.76it/s] 22%|██▏       | 89125/400000 [00:10<00:35, 8696.03it/s] 22%|██▏       | 89995/400000 [00:10<00:35, 8635.40it/s] 23%|██▎       | 90890/400000 [00:10<00:35, 8724.64it/s] 23%|██▎       | 91782/400000 [00:10<00:35, 8780.27it/s] 23%|██▎       | 92661/400000 [00:10<00:35, 8668.94it/s] 23%|██▎       | 93552/400000 [00:10<00:35, 8737.98it/s] 24%|██▎       | 94427/400000 [00:11<00:35, 8668.79it/s] 24%|██▍       | 95295/400000 [00:11<00:35, 8542.23it/s] 24%|██▍       | 96151/400000 [00:11<00:35, 8518.96it/s] 24%|██▍       | 97029/400000 [00:11<00:35, 8567.18it/s] 24%|██▍       | 97900/400000 [00:11<00:35, 8606.47it/s] 25%|██▍       | 98762/400000 [00:11<00:35, 8586.28it/s] 25%|██▍       | 99639/400000 [00:11<00:34, 8639.67it/s] 25%|██▌       | 100505/400000 [00:11<00:34, 8643.04it/s] 25%|██▌       | 101370/400000 [00:11<00:34, 8552.38it/s] 26%|██▌       | 102235/400000 [00:11<00:34, 8580.76it/s] 26%|██▌       | 103094/400000 [00:12<00:34, 8567.68it/s] 26%|██▌       | 103951/400000 [00:12<00:34, 8526.21it/s] 26%|██▌       | 104824/400000 [00:12<00:34, 8586.34it/s] 26%|██▋       | 105683/400000 [00:12<00:34, 8582.96it/s] 27%|██▋       | 106567/400000 [00:12<00:33, 8656.73it/s] 27%|██▋       | 107433/400000 [00:12<00:34, 8603.58it/s] 27%|██▋       | 108298/400000 [00:12<00:33, 8614.97it/s] 27%|██▋       | 109168/400000 [00:12<00:33, 8640.09it/s] 28%|██▊       | 110053/400000 [00:12<00:33, 8701.53it/s] 28%|██▊       | 110924/400000 [00:12<00:33, 8661.36it/s] 28%|██▊       | 111791/400000 [00:13<00:33, 8657.54it/s] 28%|██▊       | 112680/400000 [00:13<00:32, 8722.51it/s] 28%|██▊       | 113553/400000 [00:13<00:33, 8674.20it/s] 29%|██▊       | 114422/400000 [00:13<00:32, 8674.88it/s] 29%|██▉       | 115294/400000 [00:13<00:32, 8686.56it/s] 29%|██▉       | 116163/400000 [00:13<00:33, 8557.25it/s] 29%|██▉       | 117020/400000 [00:13<00:33, 8548.21it/s] 29%|██▉       | 117907/400000 [00:13<00:32, 8641.60it/s] 30%|██▉       | 118772/400000 [00:13<00:32, 8642.65it/s] 30%|██▉       | 119659/400000 [00:13<00:32, 8708.98it/s] 30%|███       | 120536/400000 [00:14<00:32, 8726.09it/s] 30%|███       | 121410/400000 [00:14<00:31, 8728.43it/s] 31%|███       | 122303/400000 [00:14<00:31, 8784.63it/s] 31%|███       | 123204/400000 [00:14<00:31, 8848.39it/s] 31%|███       | 124090/400000 [00:14<00:31, 8675.25it/s] 31%|███       | 124959/400000 [00:14<00:33, 8270.41it/s] 31%|███▏      | 125791/400000 [00:14<00:33, 8201.34it/s] 32%|███▏      | 126665/400000 [00:14<00:32, 8353.56it/s] 32%|███▏      | 127543/400000 [00:14<00:32, 8474.47it/s] 32%|███▏      | 128438/400000 [00:14<00:31, 8609.72it/s] 32%|███▏      | 129306/400000 [00:15<00:31, 8629.07it/s] 33%|███▎      | 130174/400000 [00:15<00:31, 8643.97it/s] 33%|███▎      | 131040/400000 [00:15<00:31, 8570.03it/s] 33%|███▎      | 131898/400000 [00:15<00:31, 8545.56it/s] 33%|███▎      | 132754/400000 [00:15<00:31, 8475.71it/s] 33%|███▎      | 133612/400000 [00:15<00:31, 8506.45it/s] 34%|███▎      | 134471/400000 [00:15<00:31, 8528.71it/s] 34%|███▍      | 135339/400000 [00:15<00:30, 8572.82it/s] 34%|███▍      | 136203/400000 [00:15<00:30, 8591.72it/s] 34%|███▍      | 137084/400000 [00:15<00:30, 8655.64it/s] 34%|███▍      | 137951/400000 [00:16<00:30, 8659.16it/s] 35%|███▍      | 138826/400000 [00:16<00:30, 8683.91it/s] 35%|███▍      | 139695/400000 [00:16<00:30, 8663.90it/s] 35%|███▌      | 140579/400000 [00:16<00:29, 8713.47it/s] 35%|███▌      | 141461/400000 [00:16<00:29, 8742.74it/s] 36%|███▌      | 142338/400000 [00:16<00:29, 8750.45it/s] 36%|███▌      | 143225/400000 [00:16<00:29, 8785.56it/s] 36%|███▌      | 144114/400000 [00:16<00:29, 8816.16it/s] 36%|███▋      | 145007/400000 [00:16<00:28, 8849.06it/s] 36%|███▋      | 145892/400000 [00:16<00:28, 8844.91it/s] 37%|███▋      | 146777/400000 [00:17<00:28, 8785.54it/s] 37%|███▋      | 147666/400000 [00:17<00:28, 8815.36it/s] 37%|███▋      | 148548/400000 [00:17<00:28, 8769.54it/s] 37%|███▋      | 149444/400000 [00:17<00:28, 8825.60it/s] 38%|███▊      | 150341/400000 [00:17<00:28, 8866.19it/s] 38%|███▊      | 151228/400000 [00:17<00:28, 8756.00it/s] 38%|███▊      | 152114/400000 [00:17<00:28, 8784.00it/s] 38%|███▊      | 152993/400000 [00:17<00:28, 8780.12it/s] 38%|███▊      | 153876/400000 [00:17<00:27, 8794.23it/s] 39%|███▊      | 154756/400000 [00:18<00:27, 8779.92it/s] 39%|███▉      | 155635/400000 [00:18<00:27, 8733.64it/s] 39%|███▉      | 156525/400000 [00:18<00:27, 8781.41it/s] 39%|███▉      | 157404/400000 [00:18<00:27, 8717.94it/s] 40%|███▉      | 158289/400000 [00:18<00:27, 8757.13it/s] 40%|███▉      | 159169/400000 [00:18<00:27, 8768.32it/s] 40%|████      | 160046/400000 [00:18<00:27, 8737.45it/s] 40%|████      | 160932/400000 [00:18<00:27, 8773.73it/s] 40%|████      | 161810/400000 [00:18<00:27, 8749.80it/s] 41%|████      | 162686/400000 [00:18<00:28, 8326.52it/s] 41%|████      | 163569/400000 [00:19<00:27, 8471.17it/s] 41%|████      | 164440/400000 [00:19<00:27, 8541.06it/s] 41%|████▏     | 165297/400000 [00:19<00:27, 8539.59it/s] 42%|████▏     | 166179/400000 [00:19<00:27, 8621.22it/s] 42%|████▏     | 167066/400000 [00:19<00:26, 8691.67it/s] 42%|████▏     | 167937/400000 [00:19<00:27, 8542.83it/s] 42%|████▏     | 168826/400000 [00:19<00:26, 8643.94it/s] 42%|████▏     | 169727/400000 [00:19<00:26, 8748.42it/s] 43%|████▎     | 170611/400000 [00:19<00:26, 8774.68it/s] 43%|████▎     | 171490/400000 [00:19<00:26, 8746.39it/s] 43%|████▎     | 172376/400000 [00:20<00:25, 8779.39it/s] 43%|████▎     | 173255/400000 [00:20<00:26, 8683.74it/s] 44%|████▎     | 174124/400000 [00:20<00:26, 8644.61it/s] 44%|████▎     | 174996/400000 [00:20<00:25, 8664.30it/s] 44%|████▍     | 175863/400000 [00:20<00:26, 8585.05it/s] 44%|████▍     | 176727/400000 [00:20<00:25, 8600.90it/s] 44%|████▍     | 177593/400000 [00:20<00:25, 8616.71it/s] 45%|████▍     | 178455/400000 [00:20<00:25, 8596.59it/s] 45%|████▍     | 179340/400000 [00:20<00:25, 8669.34it/s] 45%|████▌     | 180208/400000 [00:20<00:25, 8659.35it/s] 45%|████▌     | 181075/400000 [00:21<00:25, 8656.14it/s] 45%|████▌     | 181941/400000 [00:21<00:25, 8555.58it/s] 46%|████▌     | 182818/400000 [00:21<00:25, 8616.36it/s] 46%|████▌     | 183680/400000 [00:21<00:25, 8601.44it/s] 46%|████▌     | 184559/400000 [00:21<00:24, 8655.03it/s] 46%|████▋     | 185425/400000 [00:21<00:24, 8598.25it/s] 47%|████▋     | 186286/400000 [00:21<00:25, 8497.00it/s] 47%|████▋     | 187161/400000 [00:21<00:24, 8569.98it/s] 47%|████▋     | 188027/400000 [00:21<00:24, 8594.14it/s] 47%|████▋     | 188906/400000 [00:21<00:24, 8648.97it/s] 47%|████▋     | 189786/400000 [00:22<00:24, 8692.28it/s] 48%|████▊     | 190656/400000 [00:22<00:24, 8651.98it/s] 48%|████▊     | 191522/400000 [00:22<00:24, 8647.48it/s] 48%|████▊     | 192424/400000 [00:22<00:23, 8753.45it/s] 48%|████▊     | 193319/400000 [00:22<00:23, 8810.21it/s] 49%|████▊     | 194204/400000 [00:22<00:23, 8821.39it/s] 49%|████▉     | 195087/400000 [00:22<00:23, 8765.26it/s] 49%|████▉     | 195964/400000 [00:22<00:23, 8765.73it/s] 49%|████▉     | 196845/400000 [00:22<00:23, 8778.05it/s] 49%|████▉     | 197723/400000 [00:22<00:23, 8701.75it/s] 50%|████▉     | 198594/400000 [00:23<00:23, 8672.54it/s] 50%|████▉     | 199462/400000 [00:23<00:23, 8666.79it/s] 50%|█████     | 200360/400000 [00:23<00:22, 8756.22it/s] 50%|█████     | 201255/400000 [00:23<00:22, 8811.77it/s] 51%|█████     | 202137/400000 [00:23<00:22, 8811.80it/s] 51%|█████     | 203019/400000 [00:23<00:22, 8758.22it/s] 51%|█████     | 203896/400000 [00:23<00:22, 8743.40it/s] 51%|█████     | 204771/400000 [00:23<00:22, 8743.91it/s] 51%|█████▏    | 205673/400000 [00:23<00:22, 8822.74it/s] 52%|█████▏    | 206557/400000 [00:23<00:21, 8826.78it/s] 52%|█████▏    | 207440/400000 [00:24<00:21, 8761.89it/s] 52%|█████▏    | 208317/400000 [00:24<00:22, 8695.35it/s] 52%|█████▏    | 209195/400000 [00:24<00:21, 8719.20it/s] 53%|█████▎    | 210068/400000 [00:24<00:22, 8605.77it/s] 53%|█████▎    | 210930/400000 [00:24<00:22, 8528.46it/s] 53%|█████▎    | 211798/400000 [00:24<00:21, 8571.86it/s] 53%|█████▎    | 212665/400000 [00:24<00:21, 8598.66it/s] 53%|█████▎    | 213567/400000 [00:24<00:21, 8720.08it/s] 54%|█████▎    | 214455/400000 [00:24<00:21, 8767.23it/s] 54%|█████▍    | 215333/400000 [00:24<00:21, 8712.19it/s] 54%|█████▍    | 216216/400000 [00:25<00:21, 8746.07it/s] 54%|█████▍    | 217093/400000 [00:25<00:20, 8752.54it/s] 54%|█████▍    | 217969/400000 [00:25<00:21, 8605.29it/s] 55%|█████▍    | 218831/400000 [00:25<00:21, 8581.99it/s] 55%|█████▍    | 219708/400000 [00:25<00:20, 8636.55it/s] 55%|█████▌    | 220573/400000 [00:25<00:20, 8629.18it/s] 55%|█████▌    | 221437/400000 [00:25<00:20, 8575.80it/s] 56%|█████▌    | 222312/400000 [00:25<00:20, 8625.18it/s] 56%|█████▌    | 223182/400000 [00:25<00:20, 8645.21it/s] 56%|█████▌    | 224062/400000 [00:25<00:20, 8688.57it/s] 56%|█████▌    | 224944/400000 [00:26<00:20, 8725.91it/s] 56%|█████▋    | 225831/400000 [00:26<00:19, 8765.83it/s] 57%|█████▋    | 226708/400000 [00:26<00:19, 8739.15it/s] 57%|█████▋    | 227587/400000 [00:26<00:19, 8752.85it/s] 57%|█████▋    | 228476/400000 [00:26<00:19, 8790.73it/s] 57%|█████▋    | 229366/400000 [00:26<00:19, 8823.03it/s] 58%|█████▊    | 230249/400000 [00:26<00:19, 8795.50it/s] 58%|█████▊    | 231129/400000 [00:26<00:19, 8690.27it/s] 58%|█████▊    | 232021/400000 [00:26<00:19, 8755.35it/s] 58%|█████▊    | 232897/400000 [00:27<00:19, 8690.12it/s] 58%|█████▊    | 233767/400000 [00:27<00:19, 8381.59it/s] 59%|█████▊    | 234669/400000 [00:27<00:19, 8562.32it/s] 59%|█████▉    | 235537/400000 [00:27<00:19, 8595.24it/s] 59%|█████▉    | 236403/400000 [00:27<00:18, 8612.10it/s] 59%|█████▉    | 237266/400000 [00:27<00:18, 8599.10it/s] 60%|█████▉    | 238152/400000 [00:27<00:18, 8673.08it/s] 60%|█████▉    | 239027/400000 [00:27<00:18, 8694.01it/s] 60%|█████▉    | 239897/400000 [00:27<00:18, 8684.66it/s] 60%|██████    | 240796/400000 [00:27<00:18, 8773.62it/s] 60%|██████    | 241693/400000 [00:28<00:17, 8829.87it/s] 61%|██████    | 242577/400000 [00:28<00:17, 8818.13it/s] 61%|██████    | 243460/400000 [00:28<00:17, 8781.35it/s] 61%|██████    | 244339/400000 [00:28<00:17, 8746.43it/s] 61%|██████▏   | 245214/400000 [00:28<00:17, 8743.27it/s] 62%|██████▏   | 246104/400000 [00:28<00:17, 8788.65it/s] 62%|██████▏   | 247015/400000 [00:28<00:17, 8880.00it/s] 62%|██████▏   | 247904/400000 [00:28<00:17, 8775.56it/s] 62%|██████▏   | 248783/400000 [00:28<00:17, 8656.81it/s] 62%|██████▏   | 249678/400000 [00:28<00:17, 8742.58it/s] 63%|██████▎   | 250563/400000 [00:29<00:17, 8771.75it/s] 63%|██████▎   | 251441/400000 [00:29<00:17, 8687.53it/s] 63%|██████▎   | 252340/400000 [00:29<00:16, 8774.00it/s] 63%|██████▎   | 253218/400000 [00:29<00:16, 8738.71it/s] 64%|██████▎   | 254098/400000 [00:29<00:16, 8752.23it/s] 64%|██████▎   | 254974/400000 [00:29<00:16, 8664.04it/s] 64%|██████▍   | 255856/400000 [00:29<00:16, 8708.60it/s] 64%|██████▍   | 256736/400000 [00:29<00:16, 8735.07it/s] 64%|██████▍   | 257610/400000 [00:29<00:16, 8719.14it/s] 65%|██████▍   | 258502/400000 [00:29<00:16, 8775.87it/s] 65%|██████▍   | 259400/400000 [00:30<00:15, 8834.17it/s] 65%|██████▌   | 260303/400000 [00:30<00:15, 8891.86it/s] 65%|██████▌   | 261193/400000 [00:30<00:15, 8790.85it/s] 66%|██████▌   | 262073/400000 [00:30<00:15, 8780.94it/s] 66%|██████▌   | 262952/400000 [00:30<00:15, 8779.13it/s] 66%|██████▌   | 263831/400000 [00:30<00:15, 8727.66it/s] 66%|██████▌   | 264704/400000 [00:30<00:15, 8698.50it/s] 66%|██████▋   | 265575/400000 [00:30<00:15, 8652.04it/s] 67%|██████▋   | 266441/400000 [00:30<00:15, 8639.58it/s] 67%|██████▋   | 267319/400000 [00:30<00:15, 8680.76it/s] 67%|██████▋   | 268190/400000 [00:31<00:15, 8686.78it/s] 67%|██████▋   | 269059/400000 [00:31<00:15, 8681.89it/s] 67%|██████▋   | 269934/400000 [00:31<00:14, 8701.89it/s] 68%|██████▊   | 270805/400000 [00:31<00:14, 8682.23it/s] 68%|██████▊   | 271674/400000 [00:31<00:15, 8541.83it/s] 68%|██████▊   | 272549/400000 [00:31<00:14, 8601.42it/s] 68%|██████▊   | 273417/400000 [00:31<00:14, 8622.11it/s] 69%|██████▊   | 274289/400000 [00:31<00:14, 8650.78it/s] 69%|██████▉   | 275155/400000 [00:31<00:14, 8587.33it/s] 69%|██████▉   | 276023/400000 [00:31<00:14, 8614.16it/s] 69%|██████▉   | 276905/400000 [00:32<00:14, 8673.91it/s] 69%|██████▉   | 277781/400000 [00:32<00:14, 8698.83it/s] 70%|██████▉   | 278663/400000 [00:32<00:13, 8732.63it/s] 70%|██████▉   | 279537/400000 [00:32<00:13, 8723.31it/s] 70%|███████   | 280428/400000 [00:32<00:13, 8775.98it/s] 70%|███████   | 281306/400000 [00:32<00:13, 8769.08it/s] 71%|███████   | 282193/400000 [00:32<00:13, 8797.50it/s] 71%|███████   | 283082/400000 [00:32<00:13, 8825.01it/s] 71%|███████   | 283965/400000 [00:32<00:13, 8801.63it/s] 71%|███████   | 284846/400000 [00:32<00:13, 8776.78it/s] 71%|███████▏  | 285724/400000 [00:33<00:13, 8762.39it/s] 72%|███████▏  | 286622/400000 [00:33<00:12, 8824.70it/s] 72%|███████▏  | 287506/400000 [00:33<00:12, 8827.45it/s] 72%|███████▏  | 288389/400000 [00:33<00:12, 8683.87it/s] 72%|███████▏  | 289258/400000 [00:33<00:13, 8329.97it/s] 73%|███████▎  | 290095/400000 [00:33<00:13, 8338.28it/s] 73%|███████▎  | 290932/400000 [00:33<00:13, 8313.38it/s] 73%|███████▎  | 291810/400000 [00:33<00:12, 8447.48it/s] 73%|███████▎  | 292678/400000 [00:33<00:12, 8513.49it/s] 73%|███████▎  | 293573/400000 [00:33<00:12, 8639.68it/s] 74%|███████▎  | 294439/400000 [00:34<00:12, 8594.21it/s] 74%|███████▍  | 295310/400000 [00:34<00:12, 8628.13it/s] 74%|███████▍  | 296180/400000 [00:34<00:12, 8646.81it/s] 74%|███████▍  | 297074/400000 [00:34<00:11, 8730.81it/s] 74%|███████▍  | 297948/400000 [00:34<00:11, 8718.34it/s] 75%|███████▍  | 298828/400000 [00:34<00:11, 8740.51it/s] 75%|███████▍  | 299714/400000 [00:34<00:11, 8773.95it/s] 75%|███████▌  | 300598/400000 [00:34<00:11, 8792.99it/s] 75%|███████▌  | 301478/400000 [00:34<00:11, 8771.76it/s] 76%|███████▌  | 302356/400000 [00:34<00:11, 8772.54it/s] 76%|███████▌  | 303261/400000 [00:35<00:10, 8853.87it/s] 76%|███████▌  | 304147/400000 [00:35<00:10, 8805.07it/s] 76%|███████▋  | 305028/400000 [00:35<00:10, 8747.45it/s] 76%|███████▋  | 305927/400000 [00:35<00:10, 8816.94it/s] 77%|███████▋  | 306810/400000 [00:35<00:10, 8775.35it/s] 77%|███████▋  | 307698/400000 [00:35<00:10, 8805.15it/s] 77%|███████▋  | 308579/400000 [00:35<00:10, 8799.83it/s] 77%|███████▋  | 309460/400000 [00:35<00:10, 8747.64it/s] 78%|███████▊  | 310355/400000 [00:35<00:10, 8807.28it/s] 78%|███████▊  | 311248/400000 [00:35<00:10, 8843.50it/s] 78%|███████▊  | 312133/400000 [00:36<00:09, 8830.72it/s] 78%|███████▊  | 313017/400000 [00:36<00:09, 8816.28it/s] 78%|███████▊  | 313904/400000 [00:36<00:09, 8830.54it/s] 79%|███████▊  | 314788/400000 [00:36<00:09, 8787.36it/s] 79%|███████▉  | 315667/400000 [00:36<00:09, 8708.01it/s] 79%|███████▉  | 316544/400000 [00:36<00:09, 8725.12it/s] 79%|███████▉  | 317417/400000 [00:36<00:09, 8602.46it/s] 80%|███████▉  | 318278/400000 [00:36<00:09, 8565.70it/s] 80%|███████▉  | 319135/400000 [00:36<00:09, 8488.72it/s] 80%|████████  | 320014/400000 [00:37<00:09, 8574.37it/s] 80%|████████  | 320890/400000 [00:37<00:09, 8628.83it/s] 80%|████████  | 321766/400000 [00:37<00:09, 8665.12it/s] 81%|████████  | 322640/400000 [00:37<00:08, 8685.34it/s] 81%|████████  | 323509/400000 [00:37<00:08, 8677.61it/s] 81%|████████  | 324386/400000 [00:37<00:08, 8703.28it/s] 81%|████████▏ | 325284/400000 [00:37<00:08, 8781.89it/s] 82%|████████▏ | 326179/400000 [00:37<00:08, 8831.40it/s] 82%|████████▏ | 327065/400000 [00:37<00:08, 8838.07it/s] 82%|████████▏ | 327949/400000 [00:37<00:08, 8696.39it/s] 82%|████████▏ | 328841/400000 [00:38<00:08, 8760.87it/s] 82%|████████▏ | 329730/400000 [00:38<00:07, 8799.18it/s] 83%|████████▎ | 330611/400000 [00:38<00:07, 8780.99it/s] 83%|████████▎ | 331490/400000 [00:38<00:07, 8774.04it/s] 83%|████████▎ | 332368/400000 [00:38<00:07, 8698.50it/s] 83%|████████▎ | 333271/400000 [00:38<00:07, 8794.19it/s] 84%|████████▎ | 334151/400000 [00:38<00:07, 8771.43it/s] 84%|████████▍ | 335029/400000 [00:38<00:07, 8705.13it/s] 84%|████████▍ | 335900/400000 [00:38<00:07, 8604.19it/s] 84%|████████▍ | 336761/400000 [00:38<00:07, 8566.36it/s] 84%|████████▍ | 337628/400000 [00:39<00:07, 8595.89it/s] 85%|████████▍ | 338488/400000 [00:39<00:07, 8508.34it/s] 85%|████████▍ | 339373/400000 [00:39<00:07, 8607.21it/s] 85%|████████▌ | 340257/400000 [00:39<00:06, 8675.13it/s] 85%|████████▌ | 341142/400000 [00:39<00:06, 8726.84it/s] 86%|████████▌ | 342016/400000 [00:39<00:06, 8725.09it/s] 86%|████████▌ | 342889/400000 [00:39<00:06, 8431.14it/s] 86%|████████▌ | 343735/400000 [00:39<00:06, 8338.38it/s] 86%|████████▌ | 344571/400000 [00:39<00:06, 8261.55it/s] 86%|████████▋ | 345399/400000 [00:39<00:06, 8073.62it/s] 87%|████████▋ | 346232/400000 [00:40<00:06, 8147.70it/s] 87%|████████▋ | 347115/400000 [00:40<00:06, 8338.96it/s] 87%|████████▋ | 347983/400000 [00:40<00:06, 8437.26it/s] 87%|████████▋ | 348858/400000 [00:40<00:05, 8528.23it/s] 87%|████████▋ | 349731/400000 [00:40<00:05, 8586.96it/s] 88%|████████▊ | 350606/400000 [00:40<00:05, 8634.68it/s] 88%|████████▊ | 351488/400000 [00:40<00:05, 8688.62it/s] 88%|████████▊ | 352358/400000 [00:40<00:05, 8663.43it/s] 88%|████████▊ | 353238/400000 [00:40<00:05, 8703.54it/s] 89%|████████▊ | 354109/400000 [00:40<00:05, 8694.18it/s] 89%|████████▊ | 354979/400000 [00:41<00:05, 8331.50it/s] 89%|████████▉ | 355827/400000 [00:41<00:05, 8374.57it/s] 89%|████████▉ | 356705/400000 [00:41<00:05, 8491.59it/s] 89%|████████▉ | 357591/400000 [00:41<00:04, 8597.36it/s] 90%|████████▉ | 358453/400000 [00:41<00:04, 8561.58it/s] 90%|████████▉ | 359311/400000 [00:41<00:04, 8548.04it/s] 90%|█████████ | 360189/400000 [00:41<00:04, 8614.83it/s] 90%|█████████ | 361052/400000 [00:41<00:04, 8559.32it/s] 90%|█████████ | 361909/400000 [00:41<00:04, 8462.17it/s] 91%|█████████ | 362756/400000 [00:41<00:04, 8353.97it/s] 91%|█████████ | 363622/400000 [00:42<00:04, 8442.51it/s] 91%|█████████ | 364468/400000 [00:42<00:04, 8284.40it/s] 91%|█████████▏| 365342/400000 [00:42<00:04, 8413.60it/s] 92%|█████████▏| 366236/400000 [00:42<00:03, 8562.27it/s] 92%|█████████▏| 367094/400000 [00:42<00:03, 8404.54it/s] 92%|█████████▏| 367945/400000 [00:42<00:03, 8433.53it/s] 92%|█████████▏| 368812/400000 [00:42<00:03, 8500.38it/s] 92%|█████████▏| 369683/400000 [00:42<00:03, 8562.14it/s] 93%|█████████▎| 370555/400000 [00:42<00:03, 8606.73it/s] 93%|█████████▎| 371419/400000 [00:43<00:03, 8614.58it/s] 93%|█████████▎| 372283/400000 [00:43<00:03, 8621.56it/s] 93%|█████████▎| 373146/400000 [00:43<00:03, 8603.35it/s] 94%|█████████▎| 374017/400000 [00:43<00:03, 8633.31it/s] 94%|█████████▎| 374881/400000 [00:43<00:02, 8593.67it/s] 94%|█████████▍| 375741/400000 [00:43<00:02, 8583.80it/s] 94%|█████████▍| 376632/400000 [00:43<00:02, 8677.57it/s] 94%|█████████▍| 377536/400000 [00:43<00:02, 8780.52it/s] 95%|█████████▍| 378419/400000 [00:43<00:02, 8793.31it/s] 95%|█████████▍| 379303/400000 [00:43<00:02, 8804.40it/s] 95%|█████████▌| 380184/400000 [00:44<00:02, 8786.91it/s] 95%|█████████▌| 381071/400000 [00:44<00:02, 8810.24it/s] 95%|█████████▌| 381953/400000 [00:44<00:02, 8800.75it/s] 96%|█████████▌| 382834/400000 [00:44<00:01, 8702.98it/s] 96%|█████████▌| 383716/400000 [00:44<00:01, 8735.38it/s] 96%|█████████▌| 384590/400000 [00:44<00:01, 8575.70it/s] 96%|█████████▋| 385459/400000 [00:44<00:01, 8609.49it/s] 97%|█████████▋| 386388/400000 [00:44<00:01, 8802.46it/s] 97%|█████████▋| 387310/400000 [00:44<00:01, 8921.47it/s] 97%|█████████▋| 388256/400000 [00:44<00:01, 9076.30it/s] 97%|█████████▋| 389166/400000 [00:45<00:01, 8986.05it/s] 98%|█████████▊| 390075/400000 [00:45<00:01, 9014.23it/s] 98%|█████████▊| 390978/400000 [00:45<00:01, 8919.01it/s] 98%|█████████▊| 391871/400000 [00:45<00:00, 8878.38it/s] 98%|█████████▊| 392777/400000 [00:45<00:00, 8931.96it/s] 98%|█████████▊| 393671/400000 [00:45<00:00, 8800.51it/s] 99%|█████████▊| 394552/400000 [00:45<00:00, 8726.89it/s] 99%|█████████▉| 395426/400000 [00:45<00:00, 8565.32it/s] 99%|█████████▉| 396285/400000 [00:45<00:00, 8571.07it/s] 99%|█████████▉| 397158/400000 [00:45<00:00, 8615.87it/s]100%|█████████▉| 398021/400000 [00:46<00:00, 8615.53it/s]100%|█████████▉| 398900/400000 [00:46<00:00, 8666.71it/s]100%|█████████▉| 399768/400000 [00:46<00:00, 8643.50it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8643.89it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f69c46b8c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.0113185759266047 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.01158103755485254 	 Accuracy: 48

  model saves at 48% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15927 out of table with 15924 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15927 out of table with 15924 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-12 20:24:21.676961: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 20:24:21.681639: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 20:24:21.681767: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f88b5a2710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 20:24:21.681781: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f696a16fba8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7433 - accuracy: 0.4950
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7004 - accuracy: 0.4978
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7305 - accuracy: 0.4958
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7367 - accuracy: 0.4954
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7065 - accuracy: 0.4974
11000/25000 [============>.................] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6926 - accuracy: 0.4983
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6787 - accuracy: 0.4992
15000/25000 [=================>............] - ETA: 2s - loss: 7.6748 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6317 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6158 - accuracy: 0.5033
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6291 - accuracy: 0.5024
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6330 - accuracy: 0.5022
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6450 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f693513e9e8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f69cbe16160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.1438 - crf_viterbi_accuracy: 0.2133 - val_loss: 2.0264 - val_crf_viterbi_accuracy: 0.2400

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

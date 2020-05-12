
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa914466fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 21:12:25.365939
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 21:12:25.370170
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 21:12:25.373372
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 21:12:25.377308
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa9202304e0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356593.8125
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 289715.3125
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 175640.8594
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 89177.4219
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 40074.2578
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 19833.1191
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 11287.6367
Epoch 8/10

1/1 [==============================] - 0s 116ms/step - loss: 7245.4956
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 5199.1577
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 3986.8562

  #### Inference Need return ypred, ytrue ######################### 
[[-4.18131143e-01  2.01093569e-01 -1.94586396e-01  1.47064400e+00
   3.42258394e-01 -5.37093818e-01  7.93849707e-01  1.42059064e+00
   1.47774148e+00 -9.86916423e-01 -3.94752204e-01  1.11218011e+00
   6.57138884e-01 -1.53247833e-01 -3.97836447e-01  2.48863637e-01
   3.11791927e-01  1.60917675e+00 -3.41200918e-01 -1.39371443e+00
   2.51277924e-01  4.82900500e-01  3.31173204e-02 -1.18065953e+00
  -7.54869878e-01 -1.51201272e+00 -2.38695860e-01 -9.24710274e-01
  -3.44037950e-01 -1.43835545e-01 -1.44722313e-01  1.26502454e-01
   3.05350929e-01 -7.69651175e-01 -2.34515381e+00  1.39870584e+00
  -4.79435086e-01 -4.86877620e-01  5.29091001e-01  8.80096793e-01
  -3.43656451e-01 -1.71550527e-01  6.23989880e-01 -1.76735312e-01
   9.98244405e-01 -6.73229277e-01 -7.68035650e-03  1.73882556e+00
   1.37656963e+00  6.14146590e-02  6.95928216e-01  1.20422697e+00
   1.32427716e+00  9.26221728e-01 -1.46308041e+00  2.83602655e-01
   2.57914960e-01 -4.55204904e-01  1.76792502e-01 -1.80810004e-01
   2.92986900e-01  7.45513380e-01 -1.43037766e-01 -1.03709280e+00
   6.17684066e-01 -2.37104201e+00 -9.47936893e-01  7.97269583e-01
  -1.67570353e+00  1.20330799e+00 -1.25711608e+00 -1.86230838e+00
  -1.99290186e-01 -1.97695589e+00 -1.13548505e+00  1.71087909e+00
  -1.66806236e-01 -1.28862381e+00  2.58508742e-01  4.23806190e-01
  -9.80469286e-02 -1.97803915e-01 -1.67915034e+00 -7.75317550e-02
  -9.07013297e-01 -1.60001457e-01 -1.32054162e+00 -4.82752264e-01
  -5.04555702e-01  1.46665215e+00 -7.23458290e-01 -1.00148582e+00
  -5.82246423e-01 -5.39978921e-01 -4.36169058e-02  5.50051033e-01
   7.82005489e-01  6.05936766e-01  8.60956907e-01 -1.02889800e+00
   2.79680043e-01 -1.35433733e+00 -1.04731649e-01  7.14846373e-01
  -3.30105007e-01 -2.57291555e+00 -9.71285760e-01  1.54166296e-02
  -5.38893938e-01  1.82336879e+00  8.84532928e-02 -1.32148892e-01
  -1.02611244e+00 -3.13710690e-01 -5.34780502e-01 -9.05481637e-01
  -7.56390989e-01  9.04654801e-01  1.76303267e-01  6.10849798e-01
  -8.73883218e-02  9.79995632e+00  1.27878723e+01  1.13833818e+01
   1.12306175e+01  1.08933372e+01  1.16224518e+01  1.05268459e+01
   1.09704828e+01  1.08405533e+01  8.36312008e+00  1.08345051e+01
   1.16987972e+01  1.12687311e+01  1.12136660e+01  1.20284719e+01
   1.11832714e+01  1.20453377e+01  1.28827114e+01  1.18850794e+01
   1.20906019e+01  9.72020435e+00  1.27365284e+01  1.08242874e+01
   1.10766125e+01  1.14564476e+01  9.37177372e+00  1.08366871e+01
   1.17950516e+01  1.05420256e+01  1.25947781e+01  1.22916679e+01
   1.22111712e+01  1.21687994e+01  1.06289282e+01  1.14933844e+01
   8.93055058e+00  1.18751993e+01  1.11739845e+01  9.42303753e+00
   1.41534548e+01  1.11185532e+01  1.14129276e+01  1.23983173e+01
   1.05047569e+01  1.20087414e+01  1.13792591e+01  1.13472471e+01
   1.11235819e+01  1.16187906e+01  1.15401974e+01  1.15891895e+01
   1.11698780e+01  1.07028284e+01  1.13501225e+01  1.02021275e+01
   1.12068148e+01  1.00636702e+01  1.24016151e+01  1.16555519e+01
   2.25376987e+00  1.73988712e+00  2.55528736e+00  4.20785189e-01
   1.75816381e+00  4.25929844e-01  7.89919734e-01  2.65529990e-01
   5.55503011e-01  3.40104938e-01  1.83421636e+00  1.62956691e+00
   8.95146430e-01  9.85820711e-01  3.81040525e+00  8.26104879e-02
   1.01427412e+00  1.25539660e-01  3.60808253e-01  2.03523874e+00
   7.02794552e-01  1.61570787e+00  7.01611161e-01  2.38116550e+00
   3.01884532e-01  3.26587379e-01  1.36948061e+00  1.23752141e+00
   2.21278906e+00  3.05061483e+00  6.61397576e-01  1.00724626e+00
   2.05778599e+00  1.50095880e-01  2.51591969e+00  5.51979721e-01
   2.37301683e+00  2.77170992e+00  1.47694099e+00  5.69480658e-01
   2.17910290e+00  2.01339817e+00  1.13427138e+00  1.20076692e+00
   8.74520361e-01  1.44425809e+00  1.21716714e+00  1.10808671e-01
   2.04798174e+00  2.29288054e+00  1.06837213e+00  2.66722918e-01
   1.36297870e+00  1.53821528e+00  1.91714704e-01  3.86340976e-01
   5.30454993e-01  2.06890106e-01  1.35873604e+00  5.18035769e-01
   1.24368322e+00  1.57677007e+00  9.38052773e-01  7.58010685e-01
   3.75026560e+00  1.25571549e+00  1.66335607e+00  5.63757241e-01
   1.80845368e+00  3.64248228e+00  4.19771135e-01  4.39314306e-01
   3.52612376e-01  3.29386771e-01  3.03976631e+00  1.55090392e-01
   6.14756882e-01  1.20861459e+00  2.38419950e-01  1.96357799e+00
   2.68796742e-01  9.92981970e-01  1.29020333e+00  8.08363020e-01
   2.64603257e-01  1.71679986e+00  1.91121817e+00  3.91429543e-01
   1.51954865e+00  2.36899912e-01  5.68087637e-01  1.11302948e+00
   8.22973847e-01  3.72920752e-01  8.92686427e-01  8.21117163e-01
   1.28744769e+00  1.35753846e+00  1.46745276e+00  3.36667836e-01
   1.85609102e-01  2.12672138e+00  2.93962538e-01  6.34232759e-02
   4.84328389e-01  5.12846410e-01  6.94851398e-01  3.67438257e-01
   5.86134911e-01  1.57870090e+00  1.08621061e-01  4.32162702e-01
   6.44054353e-01  6.22455657e-01  2.57398558e+00  5.86459875e-01
   1.68996024e+00  2.30388784e+00  1.96229768e+00  3.06394696e-01
   1.07201993e-01  1.14625883e+01  1.05311537e+01  1.11717329e+01
   1.18975601e+01  1.01028137e+01  1.07427330e+01  1.00518723e+01
   9.79890728e+00  1.14202213e+01  9.96828270e+00  1.04634676e+01
   1.16472406e+01  1.08405895e+01  1.17010612e+01  1.03147287e+01
   1.05998268e+01  1.00549088e+01  1.04039440e+01  1.16192484e+01
   1.19505424e+01  1.09748030e+01  1.10704813e+01  1.15769205e+01
   9.38426208e+00  1.16862020e+01  1.30636082e+01  1.29790201e+01
   1.10106297e+01  1.03381672e+01  1.17178946e+01  1.13309650e+01
   1.08780069e+01  1.00690451e+01  1.03618364e+01  1.14049635e+01
   1.20289154e+01  1.02656746e+01  1.29378548e+01  1.12638721e+01
   1.03620377e+01  1.07331495e+01  1.09955435e+01  1.10926704e+01
   1.02064953e+01  1.12016487e+01  1.08804264e+01  1.13912926e+01
   1.08158808e+01  1.13085203e+01  1.14151449e+01  1.02850990e+01
   1.16325417e+01  1.11940842e+01  1.24052286e+01  1.23150988e+01
   1.03631601e+01  1.08480215e+01  1.11335917e+01  1.14440012e+01
  -1.00045204e+01 -5.04002857e+00  1.24967079e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 21:12:34.139200
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.2251
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 21:12:34.143466
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8349.92
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 21:12:34.146982
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.5046
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 21:12:34.151175
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -746.794
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140363807229544
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140361294656232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140361294656736
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140361294657240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140361294657744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140361294658248

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa914466240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.612731
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.579343
grad_step = 000002, loss = 0.553276
grad_step = 000003, loss = 0.523316
grad_step = 000004, loss = 0.488990
grad_step = 000005, loss = 0.452599
grad_step = 000006, loss = 0.422125
grad_step = 000007, loss = 0.412007
grad_step = 000008, loss = 0.398274
grad_step = 000009, loss = 0.383611
grad_step = 000010, loss = 0.368445
grad_step = 000011, loss = 0.355343
grad_step = 000012, loss = 0.340876
grad_step = 000013, loss = 0.326284
grad_step = 000014, loss = 0.312625
grad_step = 000015, loss = 0.299945
grad_step = 000016, loss = 0.287859
grad_step = 000017, loss = 0.275467
grad_step = 000018, loss = 0.261896
grad_step = 000019, loss = 0.247778
grad_step = 000020, loss = 0.234977
grad_step = 000021, loss = 0.224220
grad_step = 000022, loss = 0.214122
grad_step = 000023, loss = 0.203677
grad_step = 000024, loss = 0.192626
grad_step = 000025, loss = 0.181412
grad_step = 000026, loss = 0.170784
grad_step = 000027, loss = 0.160847
grad_step = 000028, loss = 0.151008
grad_step = 000029, loss = 0.141188
grad_step = 000030, loss = 0.131590
grad_step = 000031, loss = 0.122406
grad_step = 000032, loss = 0.114099
grad_step = 000033, loss = 0.106593
grad_step = 000034, loss = 0.099043
grad_step = 000035, loss = 0.091324
grad_step = 000036, loss = 0.084124
grad_step = 000037, loss = 0.077731
grad_step = 000038, loss = 0.071621
grad_step = 000039, loss = 0.065652
grad_step = 000040, loss = 0.060181
grad_step = 000041, loss = 0.055102
grad_step = 000042, loss = 0.050345
grad_step = 000043, loss = 0.046005
grad_step = 000044, loss = 0.041954
grad_step = 000045, loss = 0.038150
grad_step = 000046, loss = 0.034622
grad_step = 000047, loss = 0.031452
grad_step = 000048, loss = 0.028608
grad_step = 000049, loss = 0.025978
grad_step = 000050, loss = 0.023465
grad_step = 000051, loss = 0.021075
grad_step = 000052, loss = 0.018919
grad_step = 000053, loss = 0.017079
grad_step = 000054, loss = 0.015371
grad_step = 000055, loss = 0.013767
grad_step = 000056, loss = 0.012334
grad_step = 000057, loss = 0.011069
grad_step = 000058, loss = 0.009940
grad_step = 000059, loss = 0.008908
grad_step = 000060, loss = 0.007965
grad_step = 000061, loss = 0.007156
grad_step = 000062, loss = 0.006460
grad_step = 000063, loss = 0.005858
grad_step = 000064, loss = 0.005326
grad_step = 000065, loss = 0.004859
grad_step = 000066, loss = 0.004446
grad_step = 000067, loss = 0.004112
grad_step = 000068, loss = 0.003837
grad_step = 000069, loss = 0.003585
grad_step = 000070, loss = 0.003384
grad_step = 000071, loss = 0.003219
grad_step = 000072, loss = 0.003079
grad_step = 000073, loss = 0.002958
grad_step = 000074, loss = 0.002858
grad_step = 000075, loss = 0.002784
grad_step = 000076, loss = 0.002721
grad_step = 000077, loss = 0.002665
grad_step = 000078, loss = 0.002620
grad_step = 000079, loss = 0.002579
grad_step = 000080, loss = 0.002545
grad_step = 000081, loss = 0.002520
grad_step = 000082, loss = 0.002493
grad_step = 000083, loss = 0.002470
grad_step = 000084, loss = 0.002449
grad_step = 000085, loss = 0.002430
grad_step = 000086, loss = 0.002411
grad_step = 000087, loss = 0.002391
grad_step = 000088, loss = 0.002373
grad_step = 000089, loss = 0.002354
grad_step = 000090, loss = 0.002336
grad_step = 000091, loss = 0.002318
grad_step = 000092, loss = 0.002301
grad_step = 000093, loss = 0.002284
grad_step = 000094, loss = 0.002268
grad_step = 000095, loss = 0.002252
grad_step = 000096, loss = 0.002237
grad_step = 000097, loss = 0.002222
grad_step = 000098, loss = 0.002210
grad_step = 000099, loss = 0.002198
grad_step = 000100, loss = 0.002187
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002177
grad_step = 000102, loss = 0.002168
grad_step = 000103, loss = 0.002159
grad_step = 000104, loss = 0.002152
grad_step = 000105, loss = 0.002145
grad_step = 000106, loss = 0.002139
grad_step = 000107, loss = 0.002133
grad_step = 000108, loss = 0.002128
grad_step = 000109, loss = 0.002123
grad_step = 000110, loss = 0.002118
grad_step = 000111, loss = 0.002113
grad_step = 000112, loss = 0.002108
grad_step = 000113, loss = 0.002103
grad_step = 000114, loss = 0.002098
grad_step = 000115, loss = 0.002094
grad_step = 000116, loss = 0.002089
grad_step = 000117, loss = 0.002084
grad_step = 000118, loss = 0.002079
grad_step = 000119, loss = 0.002074
grad_step = 000120, loss = 0.002069
grad_step = 000121, loss = 0.002064
grad_step = 000122, loss = 0.002059
grad_step = 000123, loss = 0.002054
grad_step = 000124, loss = 0.002049
grad_step = 000125, loss = 0.002044
grad_step = 000126, loss = 0.002039
grad_step = 000127, loss = 0.002034
grad_step = 000128, loss = 0.002029
grad_step = 000129, loss = 0.002024
grad_step = 000130, loss = 0.002020
grad_step = 000131, loss = 0.002015
grad_step = 000132, loss = 0.002010
grad_step = 000133, loss = 0.002006
grad_step = 000134, loss = 0.002004
grad_step = 000135, loss = 0.002008
grad_step = 000136, loss = 0.002019
grad_step = 000137, loss = 0.002033
grad_step = 000138, loss = 0.002049
grad_step = 000139, loss = 0.002026
grad_step = 000140, loss = 0.001992
grad_step = 000141, loss = 0.001971
grad_step = 000142, loss = 0.001979
grad_step = 000143, loss = 0.001997
grad_step = 000144, loss = 0.001992
grad_step = 000145, loss = 0.001970
grad_step = 000146, loss = 0.001952
grad_step = 000147, loss = 0.001955
grad_step = 000148, loss = 0.001966
grad_step = 000149, loss = 0.001963
grad_step = 000150, loss = 0.001948
grad_step = 000151, loss = 0.001934
grad_step = 000152, loss = 0.001932
grad_step = 000153, loss = 0.001938
grad_step = 000154, loss = 0.001939
grad_step = 000155, loss = 0.001933
grad_step = 000156, loss = 0.001921
grad_step = 000157, loss = 0.001912
grad_step = 000158, loss = 0.001909
grad_step = 000159, loss = 0.001910
grad_step = 000160, loss = 0.001912
grad_step = 000161, loss = 0.001910
grad_step = 000162, loss = 0.001906
grad_step = 000163, loss = 0.001899
grad_step = 000164, loss = 0.001892
grad_step = 000165, loss = 0.001886
grad_step = 000166, loss = 0.001881
grad_step = 000167, loss = 0.001878
grad_step = 000168, loss = 0.001873
grad_step = 000169, loss = 0.001869
grad_step = 000170, loss = 0.001865
grad_step = 000171, loss = 0.001865
grad_step = 000172, loss = 0.001867
grad_step = 000173, loss = 0.001877
grad_step = 000174, loss = 0.001899
grad_step = 000175, loss = 0.001951
grad_step = 000176, loss = 0.002026
grad_step = 000177, loss = 0.002095
grad_step = 000178, loss = 0.002015
grad_step = 000179, loss = 0.001879
grad_step = 000180, loss = 0.001847
grad_step = 000181, loss = 0.001937
grad_step = 000182, loss = 0.001978
grad_step = 000183, loss = 0.001893
grad_step = 000184, loss = 0.001839
grad_step = 000185, loss = 0.001888
grad_step = 000186, loss = 0.001909
grad_step = 000187, loss = 0.001848
grad_step = 000188, loss = 0.001815
grad_step = 000189, loss = 0.001861
grad_step = 000190, loss = 0.001879
grad_step = 000191, loss = 0.001835
grad_step = 000192, loss = 0.001823
grad_step = 000193, loss = 0.001851
grad_step = 000194, loss = 0.001843
grad_step = 000195, loss = 0.001804
grad_step = 000196, loss = 0.001802
grad_step = 000197, loss = 0.001822
grad_step = 000198, loss = 0.001815
grad_step = 000199, loss = 0.001793
grad_step = 000200, loss = 0.001799
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001818
grad_step = 000202, loss = 0.001822
grad_step = 000203, loss = 0.001816
grad_step = 000204, loss = 0.001829
grad_step = 000205, loss = 0.001832
grad_step = 000206, loss = 0.001811
grad_step = 000207, loss = 0.001779
grad_step = 000208, loss = 0.001772
grad_step = 000209, loss = 0.001790
grad_step = 000210, loss = 0.001800
grad_step = 000211, loss = 0.001790
grad_step = 000212, loss = 0.001768
grad_step = 000213, loss = 0.001760
grad_step = 000214, loss = 0.001767
grad_step = 000215, loss = 0.001776
grad_step = 000216, loss = 0.001777
grad_step = 000217, loss = 0.001769
grad_step = 000218, loss = 0.001760
grad_step = 000219, loss = 0.001753
grad_step = 000220, loss = 0.001750
grad_step = 000221, loss = 0.001750
grad_step = 000222, loss = 0.001751
grad_step = 000223, loss = 0.001751
grad_step = 000224, loss = 0.001750
grad_step = 000225, loss = 0.001748
grad_step = 000226, loss = 0.001742
grad_step = 000227, loss = 0.001737
grad_step = 000228, loss = 0.001733
grad_step = 000229, loss = 0.001730
grad_step = 000230, loss = 0.001730
grad_step = 000231, loss = 0.001731
grad_step = 000232, loss = 0.001735
grad_step = 000233, loss = 0.001736
grad_step = 000234, loss = 0.001736
grad_step = 000235, loss = 0.001731
grad_step = 000236, loss = 0.001724
grad_step = 000237, loss = 0.001716
grad_step = 000238, loss = 0.001711
grad_step = 000239, loss = 0.001710
grad_step = 000240, loss = 0.001712
grad_step = 000241, loss = 0.001715
grad_step = 000242, loss = 0.001718
grad_step = 000243, loss = 0.001724
grad_step = 000244, loss = 0.001737
grad_step = 000245, loss = 0.001767
grad_step = 000246, loss = 0.001837
grad_step = 000247, loss = 0.001967
grad_step = 000248, loss = 0.002155
grad_step = 000249, loss = 0.002249
grad_step = 000250, loss = 0.002086
grad_step = 000251, loss = 0.001775
grad_step = 000252, loss = 0.001731
grad_step = 000253, loss = 0.001926
grad_step = 000254, loss = 0.001936
grad_step = 000255, loss = 0.001729
grad_step = 000256, loss = 0.001706
grad_step = 000257, loss = 0.001845
grad_step = 000258, loss = 0.001797
grad_step = 000259, loss = 0.001673
grad_step = 000260, loss = 0.001741
grad_step = 000261, loss = 0.001786
grad_step = 000262, loss = 0.001686
grad_step = 000263, loss = 0.001683
grad_step = 000264, loss = 0.001747
grad_step = 000265, loss = 0.001696
grad_step = 000266, loss = 0.001658
grad_step = 000267, loss = 0.001706
grad_step = 000268, loss = 0.001695
grad_step = 000269, loss = 0.001648
grad_step = 000270, loss = 0.001671
grad_step = 000271, loss = 0.001683
grad_step = 000272, loss = 0.001648
grad_step = 000273, loss = 0.001646
grad_step = 000274, loss = 0.001667
grad_step = 000275, loss = 0.001647
grad_step = 000276, loss = 0.001631
grad_step = 000277, loss = 0.001646
grad_step = 000278, loss = 0.001642
grad_step = 000279, loss = 0.001623
grad_step = 000280, loss = 0.001627
grad_step = 000281, loss = 0.001633
grad_step = 000282, loss = 0.001620
grad_step = 000283, loss = 0.001612
grad_step = 000284, loss = 0.001618
grad_step = 000285, loss = 0.001614
grad_step = 000286, loss = 0.001603
grad_step = 000287, loss = 0.001601
grad_step = 000288, loss = 0.001604
grad_step = 000289, loss = 0.001599
grad_step = 000290, loss = 0.001592
grad_step = 000291, loss = 0.001590
grad_step = 000292, loss = 0.001593
grad_step = 000293, loss = 0.001592
grad_step = 000294, loss = 0.001593
grad_step = 000295, loss = 0.001607
grad_step = 000296, loss = 0.001645
grad_step = 000297, loss = 0.001712
grad_step = 000298, loss = 0.001810
grad_step = 000299, loss = 0.001896
grad_step = 000300, loss = 0.001909
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001755
grad_step = 000302, loss = 0.001607
grad_step = 000303, loss = 0.001582
grad_step = 000304, loss = 0.001633
grad_step = 000305, loss = 0.001705
grad_step = 000306, loss = 0.001729
grad_step = 000307, loss = 0.001635
grad_step = 000308, loss = 0.001551
grad_step = 000309, loss = 0.001556
grad_step = 000310, loss = 0.001600
grad_step = 000311, loss = 0.001632
grad_step = 000312, loss = 0.001627
grad_step = 000313, loss = 0.001582
grad_step = 000314, loss = 0.001530
grad_step = 000315, loss = 0.001527
grad_step = 000316, loss = 0.001554
grad_step = 000317, loss = 0.001571
grad_step = 000318, loss = 0.001579
grad_step = 000319, loss = 0.001565
grad_step = 000320, loss = 0.001535
grad_step = 000321, loss = 0.001509
grad_step = 000322, loss = 0.001506
grad_step = 000323, loss = 0.001510
grad_step = 000324, loss = 0.001515
grad_step = 000325, loss = 0.001531
grad_step = 000326, loss = 0.001543
grad_step = 000327, loss = 0.001550
grad_step = 000328, loss = 0.001549
grad_step = 000329, loss = 0.001552
grad_step = 000330, loss = 0.001541
grad_step = 000331, loss = 0.001529
grad_step = 000332, loss = 0.001510
grad_step = 000333, loss = 0.001497
grad_step = 000334, loss = 0.001483
grad_step = 000335, loss = 0.001473
grad_step = 000336, loss = 0.001468
grad_step = 000337, loss = 0.001465
grad_step = 000338, loss = 0.001461
grad_step = 000339, loss = 0.001457
grad_step = 000340, loss = 0.001455
grad_step = 000341, loss = 0.001454
grad_step = 000342, loss = 0.001451
grad_step = 000343, loss = 0.001449
grad_step = 000344, loss = 0.001451
grad_step = 000345, loss = 0.001463
grad_step = 000346, loss = 0.001508
grad_step = 000347, loss = 0.001627
grad_step = 000348, loss = 0.001955
grad_step = 000349, loss = 0.002146
grad_step = 000350, loss = 0.002510
grad_step = 000351, loss = 0.002293
grad_step = 000352, loss = 0.001745
grad_step = 000353, loss = 0.001504
grad_step = 000354, loss = 0.001842
grad_step = 000355, loss = 0.002002
grad_step = 000356, loss = 0.001646
grad_step = 000357, loss = 0.001472
grad_step = 000358, loss = 0.001658
grad_step = 000359, loss = 0.001757
grad_step = 000360, loss = 0.001826
grad_step = 000361, loss = 0.001555
grad_step = 000362, loss = 0.001477
grad_step = 000363, loss = 0.001620
grad_step = 000364, loss = 0.001709
grad_step = 000365, loss = 0.001568
grad_step = 000366, loss = 0.001442
grad_step = 000367, loss = 0.001543
grad_step = 000368, loss = 0.001581
grad_step = 000369, loss = 0.001549
grad_step = 000370, loss = 0.001432
grad_step = 000371, loss = 0.001456
grad_step = 000372, loss = 0.001527
grad_step = 000373, loss = 0.001488
grad_step = 000374, loss = 0.001421
grad_step = 000375, loss = 0.001423
grad_step = 000376, loss = 0.001476
grad_step = 000377, loss = 0.001446
grad_step = 000378, loss = 0.001412
grad_step = 000379, loss = 0.001402
grad_step = 000380, loss = 0.001430
grad_step = 000381, loss = 0.001433
grad_step = 000382, loss = 0.001394
grad_step = 000383, loss = 0.001387
grad_step = 000384, loss = 0.001405
grad_step = 000385, loss = 0.001410
grad_step = 000386, loss = 0.001383
grad_step = 000387, loss = 0.001375
grad_step = 000388, loss = 0.001380
grad_step = 000389, loss = 0.001385
grad_step = 000390, loss = 0.001381
grad_step = 000391, loss = 0.001363
grad_step = 000392, loss = 0.001360
grad_step = 000393, loss = 0.001364
grad_step = 000394, loss = 0.001367
grad_step = 000395, loss = 0.001359
grad_step = 000396, loss = 0.001349
grad_step = 000397, loss = 0.001347
grad_step = 000398, loss = 0.001346
grad_step = 000399, loss = 0.001348
grad_step = 000400, loss = 0.001346
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001339
grad_step = 000402, loss = 0.001333
grad_step = 000403, loss = 0.001329
grad_step = 000404, loss = 0.001329
grad_step = 000405, loss = 0.001328
grad_step = 000406, loss = 0.001327
grad_step = 000407, loss = 0.001325
grad_step = 000408, loss = 0.001321
grad_step = 000409, loss = 0.001315
grad_step = 000410, loss = 0.001311
grad_step = 000411, loss = 0.001308
grad_step = 000412, loss = 0.001306
grad_step = 000413, loss = 0.001302
grad_step = 000414, loss = 0.001297
grad_step = 000415, loss = 0.001294
grad_step = 000416, loss = 0.001293
grad_step = 000417, loss = 0.001290
grad_step = 000418, loss = 0.001286
grad_step = 000419, loss = 0.001284
grad_step = 000420, loss = 0.001285
grad_step = 000421, loss = 0.001293
grad_step = 000422, loss = 0.001314
grad_step = 000423, loss = 0.001375
grad_step = 000424, loss = 0.001497
grad_step = 000425, loss = 0.001768
grad_step = 000426, loss = 0.001954
grad_step = 000427, loss = 0.002109
grad_step = 000428, loss = 0.001719
grad_step = 000429, loss = 0.001336
grad_step = 000430, loss = 0.001323
grad_step = 000431, loss = 0.001557
grad_step = 000432, loss = 0.001610
grad_step = 000433, loss = 0.001384
grad_step = 000434, loss = 0.001271
grad_step = 000435, loss = 0.001398
grad_step = 000436, loss = 0.001476
grad_step = 000437, loss = 0.001359
grad_step = 000438, loss = 0.001247
grad_step = 000439, loss = 0.001320
grad_step = 000440, loss = 0.001393
grad_step = 000441, loss = 0.001313
grad_step = 000442, loss = 0.001240
grad_step = 000443, loss = 0.001286
grad_step = 000444, loss = 0.001329
grad_step = 000445, loss = 0.001284
grad_step = 000446, loss = 0.001234
grad_step = 000447, loss = 0.001248
grad_step = 000448, loss = 0.001286
grad_step = 000449, loss = 0.001270
grad_step = 000450, loss = 0.001226
grad_step = 000451, loss = 0.001226
grad_step = 000452, loss = 0.001254
grad_step = 000453, loss = 0.001242
grad_step = 000454, loss = 0.001215
grad_step = 000455, loss = 0.001214
grad_step = 000456, loss = 0.001217
grad_step = 000457, loss = 0.001219
grad_step = 000458, loss = 0.001221
grad_step = 000459, loss = 0.001206
grad_step = 000460, loss = 0.001199
grad_step = 000461, loss = 0.001202
grad_step = 000462, loss = 0.001198
grad_step = 000463, loss = 0.001199
grad_step = 000464, loss = 0.001200
grad_step = 000465, loss = 0.001191
grad_step = 000466, loss = 0.001186
grad_step = 000467, loss = 0.001182
grad_step = 000468, loss = 0.001178
grad_step = 000469, loss = 0.001180
grad_step = 000470, loss = 0.001182
grad_step = 000471, loss = 0.001178
grad_step = 000472, loss = 0.001177
grad_step = 000473, loss = 0.001176
grad_step = 000474, loss = 0.001173
grad_step = 000475, loss = 0.001172
grad_step = 000476, loss = 0.001170
grad_step = 000477, loss = 0.001168
grad_step = 000478, loss = 0.001169
grad_step = 000479, loss = 0.001172
grad_step = 000480, loss = 0.001178
grad_step = 000481, loss = 0.001189
grad_step = 000482, loss = 0.001216
grad_step = 000483, loss = 0.001252
grad_step = 000484, loss = 0.001323
grad_step = 000485, loss = 0.001374
grad_step = 000486, loss = 0.001429
grad_step = 000487, loss = 0.001365
grad_step = 000488, loss = 0.001270
grad_step = 000489, loss = 0.001170
grad_step = 000490, loss = 0.001144
grad_step = 000491, loss = 0.001191
grad_step = 000492, loss = 0.001242
grad_step = 000493, loss = 0.001249
grad_step = 000494, loss = 0.001196
grad_step = 000495, loss = 0.001144
grad_step = 000496, loss = 0.001131
grad_step = 000497, loss = 0.001155
grad_step = 000498, loss = 0.001183
grad_step = 000499, loss = 0.001184
grad_step = 000500, loss = 0.001169
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001144
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

  date_run                              2020-05-12 21:12:57.639677
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.218071
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 21:12:57.645644
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.124696
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 21:12:57.653495
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130574
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 21:12:57.659198
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.894795
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
0   2020-05-12 21:12:25.365939  ...    mean_absolute_error
1   2020-05-12 21:12:25.370170  ...     mean_squared_error
2   2020-05-12 21:12:25.373372  ...  median_absolute_error
3   2020-05-12 21:12:25.377308  ...               r2_score
4   2020-05-12 21:12:34.139200  ...    mean_absolute_error
5   2020-05-12 21:12:34.143466  ...     mean_squared_error
6   2020-05-12 21:12:34.146982  ...  median_absolute_error
7   2020-05-12 21:12:34.151175  ...               r2_score
8   2020-05-12 21:12:57.639677  ...    mean_absolute_error
9   2020-05-12 21:12:57.645644  ...     mean_squared_error
10  2020-05-12 21:12:57.653495  ...  median_absolute_error
11  2020-05-12 21:12:57.659198  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 141437.68it/s] 78%|███████▊  | 7766016/9912422 [00:00<00:10, 201895.90it/s]9920512it [00:00, 42671567.80it/s]                           
0it [00:00, ?it/s]32768it [00:00, 580673.18it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 127883.50it/s]1654784it [00:00, 9019311.88it/s]                          
0it [00:00, ?it/s]8192it [00:00, 121158.20it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39f55dbcf8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a7f94eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a75c50f0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a7f94eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a751a0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a4d464e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a4d3fc18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a7f94eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a74d86d8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39a4d464e0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f39f559eef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa35ec8a208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0a4439638e8660d609027965a3c370aa0e00b81abe82d70e22bb60c531cac95c
  Stored in directory: /tmp/pip-ephem-wheel-cache-q10ao2v0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa2f6a82828> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1638400/17464789 [=>............................] - ETA: 0s
 7847936/17464789 [============>.................] - ETA: 0s
14434304/17464789 [=======================>......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 21:14:24.191696: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 21:14:24.196101: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 21:14:24.196246: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55de373f6110 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 21:14:24.196261: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0500 - accuracy: 0.4750
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7637 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7152 - accuracy: 0.4968
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7030 - accuracy: 0.4976
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 4s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 4s - loss: 7.6986 - accuracy: 0.4979
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7185 - accuracy: 0.4966
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
15000/25000 [=================>............] - ETA: 3s - loss: 7.6983 - accuracy: 0.4979
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7136 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6922 - accuracy: 0.4983
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6866 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6988 - accuracy: 0.4979
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 10s 385us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 21:14:40.922889
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 21:14:40.922889  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:00:21, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:56:37, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:30:52, 22.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:22:07, 32.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<5:08:41, 46.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.95M/862M [00:01<3:35:05, 66.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:29:46, 94.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.2M/862M [00:01<1:44:33, 135kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<1:12:52, 192kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.8M/862M [00:01<50:53, 274kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.1M/862M [00:01<35:32, 391kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:01<24:51, 556kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<17:23, 790kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<12:13, 1.12MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<08:35, 1.58MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:02<06:06, 2.22MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<05:10, 2.61MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<05:31, 2.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:55, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:04<04:38, 2.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:41, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<05:38, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<04:21, 3.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<05:45, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:52, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<05:31, 2.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:09<04:01, 3.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<34:21, 385kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<25:11, 525kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<18:22, 719kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:10<12:57, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<26:02, 505kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<19:36, 671kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:12<14:01, 936kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<12:52, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<11:42, 1.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<08:46, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:14<06:19, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<09:34, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<08:03, 1.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<05:57, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<04:55, 2.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<9:54:53, 21.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<6:56:42, 31.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:18<4:50:40, 44.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<3:48:41, 56.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<2:42:42, 79.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<1:54:25, 112kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<1:21:51, 157kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<58:36, 219kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<41:16, 310kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<31:48, 401kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<24:52, 512kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<18:00, 707kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<12:42, 999kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<24:46, 512kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<18:38, 680kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<13:18, 951kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<12:14, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<11:09, 1.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:24, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<05:59, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<25:52, 485kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<19:23, 647kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:52, 902kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:35, 990kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:22, 1.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:32, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<06:06, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:03, 950kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:26, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<07:37, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:12, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:17, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:25, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:28, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:47, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:21, 2.81MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:54, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:21, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:03, 3.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:42, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:14, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:58, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:37, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<03:52, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:06, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:52, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:05, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<03:51, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:04, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:47, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:27, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:01, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<03:48, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:26, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:00, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<03:47, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:28, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:08, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:47, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:34, 3.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:55, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:21, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:02, 2.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:29, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:12, 1.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:50, 2.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<03:35, 3.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:12, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:32, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<04:10, 2.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:32, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:19, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:01, 2.26MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<03:36, 3.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<23:07, 489kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<17:21, 651kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<12:23, 910kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<11:16, 997kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:02, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<06:33, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<05:14, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<8:46:53, 21.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<6:08:58, 30.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<4:17:11, 43.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<3:28:53, 53.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<2:28:28, 74.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:44:22, 106kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:14:30, 148kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<53:18, 207kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<37:30, 293kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<28:43, 382kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<21:12, 517kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<15:05, 725kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<13:06, 832kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<11:26, 953kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:28, 1.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<06:08, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:34, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:25, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:43, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:50, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:17, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:52, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:33, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:23, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:16, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:39, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:44, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:15, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:55, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:33, 2.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<13:28, 783kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<10:31, 1.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:37, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<07:45, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:31, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:47, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:47, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:48, 2.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:07, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:45, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:33, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:50, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:26, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:22, 3.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:18, 3.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:20, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:14, 3.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<02:24, 4.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<1:16:30, 132kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<54:33, 185kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<38:20, 262kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<29:18, 341kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<22:33, 443kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<16:13, 616kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:30, 866kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:15, 882kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:57, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:30, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:40, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<13:44, 719kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<10:43, 919kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:46, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:32, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:17, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:33, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:04, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:43, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:00, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:45, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:52, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:24, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:19, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:35, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:12, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:11, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:28, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:05, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:02, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<02:56, 3.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:47, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:15, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:44, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<03:25, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<09:34, 980kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:39, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<05:35, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:04, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:50, 2.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:51, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:17, 2.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:03, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:02, 3.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:17, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:09, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:51, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:55, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:08, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:45, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:43, 2.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:43, 3.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:28, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:27, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:00, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:51, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:19, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:14, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:20, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:56, 2.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:56, 2.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:07, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:47, 2.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:52, 3.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:35, 2.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:38, 3.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:27, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:43, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:30, 2.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:25, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:55, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:48, 3.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<32:23, 260kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<23:33, 358kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<16:39, 504kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<13:34, 616kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<10:20, 807kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:24, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<07:08, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:50, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:16, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:55, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:17, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:12, 2.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:09, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:45, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:49, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:53, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:33, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:39, 3.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:47, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:21, 2.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<02:42, 2.95MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<01:59, 4.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<08:08, 976kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:31, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:45, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:10, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:25, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:17, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:09, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:31, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:34, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:35, 3.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:52, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:29, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:56, 2.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<16:59, 452kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<12:40, 605kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<09:01, 847kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:08, 934kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:16, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:26, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:54, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:46, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:50, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:14, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:43, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:47, 2.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:41, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:21, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:31, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:29, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:11, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:24, 3.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:24, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:07, 2.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:20, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:27, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:53, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:05, 2.32MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:14, 2.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:00, 2.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:17, 3.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:13, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:42, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:56, 2.39MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:11, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:57, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:14, 3.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:11, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:56, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:12, 3.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:55, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:12, 3.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:08, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<02:47, 2.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:04, 3.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<01:32, 4.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<26:25, 254kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<19:51, 338kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<14:13, 471kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<09:56, 669kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<3:22:33, 32.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<2:22:20, 46.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<1:39:23, 66.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<1:10:45, 93.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<50:09, 131kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<35:08, 186kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<26:00, 250kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<19:31, 333kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<13:55, 466kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<09:52, 656kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<08:31, 756kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<06:38, 969kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:47, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:49, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:41, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:36, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:31, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:19, 2.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:04, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:47, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:06, 2.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:55, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:01, 3.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:51, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:37, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<01:59, 3.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:48, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:57, 3.07MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<01:55, 3.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:44, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:52, 3.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:41, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:28, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:52, 3.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:40, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:03, 1.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:23, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:43, 3.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<08:38, 658kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<06:37, 857kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<04:45, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:37, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:49, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:15, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:51, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:06, 2.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:46, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:04, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:23, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:42, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<11:09, 485kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<08:22, 646kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<05:58, 901kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:24, 989kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:53, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:41, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:24, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:56, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:43, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:20, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:27, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:42, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:22, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:30, 3.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<37:20, 134kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<26:38, 187kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<18:39, 266kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<14:07, 349kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<10:18, 478kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<07:22, 666kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<05:10, 942kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<23:12, 210kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<16:43, 290kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<11:46, 410kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<09:18, 515kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:27, 643kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:25, 880kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:56, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:59, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:01, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:04, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:12, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:10, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:22, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:39, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:55, 2.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:54, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:52, 1.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:56, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:47, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:00, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:20, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:41, 2.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:47, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:56, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:16, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:38, 2.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<03:39, 1.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:32, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:41, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:55, 2.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:34, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:35, 1.20MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:45, 1.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<04:04, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:49, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:52, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:03, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:06, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:10, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:46, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:01, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<03:46, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:51, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:07, 1.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:16, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:35, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:03, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:33, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:52, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:15, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:46, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:19, 2.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:53, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:14, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:47, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:18, 2.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:36, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:18, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:30, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<09:48, 383kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<07:43, 486kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:35, 668kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:55, 941kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<05:31, 668kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<04:43, 780kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:30, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<02:29, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:20, 833kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<03:55, 922kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:54, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:06, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:20, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:23, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:50, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:19, 2.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<05:44, 605kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:55, 706kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<03:37, 954kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:34, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:48, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:45, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:05, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:30, 2.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:08, 2.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<21:17, 157kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<15:55, 210kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<11:20, 293kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<07:56, 416kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<05:34, 587kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<17:47, 184kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<13:31, 241kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<09:42, 336kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<06:46, 476kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<05:44, 557kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<05:00, 638kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:44, 853kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:38, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:53, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:57, 1.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:18, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:39, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:10, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:26, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:53, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:24, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:35, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:56, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:32, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:09, 2.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:23, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:46, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:27, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:03, 2.72MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:45, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:00, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:34, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:08, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:37, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:50, 977kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:23, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:46, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:41, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:56, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:31, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:06, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:25, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:47, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:26, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:05, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:46, 3.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:56, 522kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:10, 617kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<03:03, 838kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:10, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:04, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:08, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:40, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:11, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:45, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:54, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:28, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:03, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:28, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:41, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:18, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<00:58, 2.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:09, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:25, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:07, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:48, 2.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<01:27, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:50, 2.59MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:03, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:19, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:03, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:46, 2.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:37, 3.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:16, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<02:10, 959kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:49, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:20, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:59, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:20, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:04, 974kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:45, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:17, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:56, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:21, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:29, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:10, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:51, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:38, 2.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:38, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:10, 864kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:47, 1.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:18, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:56, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:22, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:26, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:07, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:36, 2.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:51, 610kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:57, 589kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:17, 758kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:38, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:10, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:51, 1.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<05:02, 332kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<03:57, 422kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:52, 580kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<02:01, 810kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:25, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<04:23, 366kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<03:56, 407kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:57, 540kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<02:06, 752kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:29, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:42, 902kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:35, 964kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:12, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:52, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:37, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:01, 726kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:13, 662kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:44, 842kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:15, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:53, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:14, 1.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:34, 893kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:16, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:55, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:40, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:04, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:04, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:36, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:26, 2.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<03:48, 332kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<03:21, 376kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<02:30, 501kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:46, 699kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<01:14, 980kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:53, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<03:48, 314kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<03:18, 360kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:27, 481kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:44, 670kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<01:13, 939kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:52, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<05:41, 198kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<04:35, 245kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<03:19, 336kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<02:19, 472kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:38, 661kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<01:08, 926kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<03:46, 280kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<03:12, 328kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<02:22, 442kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:39, 619kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:10, 862kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:49, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<04:43, 209kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<03:32, 278kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:30, 387kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:44, 547kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<01:12, 765kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:27, 632kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:28, 621kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:08, 798kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:48, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:34, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:25, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:02, 820kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:19, 640kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:03, 798kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:45, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:24, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:18, 2.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<07:16, 107kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<05:19, 146kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<03:45, 205kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<02:34, 290kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:47, 409kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<01:14, 574kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:41, 421kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:24, 502kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:02, 678kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:43, 939kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:30, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:39, 982kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:39, 979kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:30, 1.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:47, 726kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:44, 774kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:33, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:23, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:27, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:37, 800kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:30, 971kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:22, 1.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:22, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:23, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:17, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:06, 3.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:21, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:28, 767kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:19<00:22, 956kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:16, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<01:51, 159kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<01:23, 213kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:57, 298kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:37, 420kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:25, 588kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:16, 824kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:48, 279kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:38, 355kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:26, 488kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:17, 682kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:10, 948kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:12, 771kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:11, 820kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:08, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:09, 590kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:10, 528kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:07, 676kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 927kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 507kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 592kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 794kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 693/400000 [00:00<00:57, 6919.02it/s]  0%|          | 1446/400000 [00:00<00:56, 7089.74it/s]  1%|          | 2192/400000 [00:00<00:55, 7195.41it/s]  1%|          | 2939/400000 [00:00<00:54, 7272.97it/s]  1%|          | 3682/400000 [00:00<00:54, 7316.83it/s]  1%|          | 4396/400000 [00:00<00:54, 7262.71it/s]  1%|▏         | 5122/400000 [00:00<00:54, 7260.11it/s]  1%|▏         | 5866/400000 [00:00<00:53, 7312.37it/s]  2%|▏         | 6611/400000 [00:00<00:53, 7351.04it/s]  2%|▏         | 7383/400000 [00:01<00:52, 7456.14it/s]  2%|▏         | 8157/400000 [00:01<00:51, 7537.48it/s]  2%|▏         | 8912/400000 [00:01<00:51, 7540.95it/s]  2%|▏         | 9658/400000 [00:01<00:51, 7514.87it/s]  3%|▎         | 10439/400000 [00:01<00:51, 7600.19it/s]  3%|▎         | 11195/400000 [00:01<00:51, 7491.31it/s]  3%|▎         | 11978/400000 [00:01<00:51, 7588.24it/s]  3%|▎         | 12740/400000 [00:01<00:50, 7595.19it/s]  3%|▎         | 13524/400000 [00:01<00:50, 7664.19it/s]  4%|▎         | 14290/400000 [00:01<00:50, 7634.67it/s]  4%|▍         | 15054/400000 [00:02<00:52, 7304.09it/s]  4%|▍         | 15811/400000 [00:02<00:52, 7378.49it/s]  4%|▍         | 16551/400000 [00:02<00:53, 7230.89it/s]  4%|▍         | 17305/400000 [00:02<00:52, 7320.34it/s]  5%|▍         | 18115/400000 [00:02<00:50, 7536.12it/s]  5%|▍         | 18877/400000 [00:02<00:50, 7554.53it/s]  5%|▍         | 19644/400000 [00:02<00:50, 7584.97it/s]  5%|▌         | 20460/400000 [00:02<00:48, 7745.88it/s]  5%|▌         | 21237/400000 [00:02<00:49, 7640.40it/s]  6%|▌         | 22085/400000 [00:02<00:47, 7873.53it/s]  6%|▌         | 22879/400000 [00:03<00:47, 7891.32it/s]  6%|▌         | 23671/400000 [00:03<00:47, 7886.69it/s]  6%|▌         | 24462/400000 [00:03<00:48, 7783.07it/s]  6%|▋         | 25268/400000 [00:03<00:47, 7861.64it/s]  7%|▋         | 26056/400000 [00:03<00:47, 7794.75it/s]  7%|▋         | 26899/400000 [00:03<00:46, 7974.92it/s]  7%|▋         | 27699/400000 [00:03<00:47, 7839.47it/s]  7%|▋         | 28485/400000 [00:03<00:47, 7845.43it/s]  7%|▋         | 29271/400000 [00:03<00:47, 7770.74it/s]  8%|▊         | 30050/400000 [00:03<00:49, 7412.43it/s]  8%|▊         | 30812/400000 [00:04<00:49, 7472.01it/s]  8%|▊         | 31616/400000 [00:04<00:48, 7630.82it/s]  8%|▊         | 32383/400000 [00:04<00:48, 7564.90it/s]  8%|▊         | 33142/400000 [00:04<00:49, 7389.38it/s]  8%|▊         | 33884/400000 [00:04<00:49, 7398.40it/s]  9%|▊         | 34626/400000 [00:04<00:49, 7390.19it/s]  9%|▉         | 35379/400000 [00:04<00:49, 7428.22it/s]  9%|▉         | 36123/400000 [00:04<00:49, 7332.39it/s]  9%|▉         | 36982/400000 [00:04<00:47, 7667.17it/s]  9%|▉         | 37846/400000 [00:04<00:45, 7935.20it/s] 10%|▉         | 38646/400000 [00:05<00:47, 7629.86it/s] 10%|▉         | 39416/400000 [00:05<00:47, 7543.31it/s] 10%|█         | 40266/400000 [00:05<00:46, 7804.98it/s] 10%|█         | 41053/400000 [00:05<00:46, 7748.09it/s] 10%|█         | 41832/400000 [00:05<00:47, 7573.01it/s] 11%|█         | 42593/400000 [00:05<00:48, 7439.87it/s] 11%|█         | 43341/400000 [00:05<00:48, 7303.49it/s] 11%|█         | 44075/400000 [00:05<00:49, 7255.62it/s] 11%|█         | 44804/400000 [00:05<00:48, 7265.50it/s] 11%|█▏        | 45566/400000 [00:06<00:48, 7367.52it/s] 12%|█▏        | 46366/400000 [00:06<00:46, 7545.02it/s] 12%|█▏        | 47123/400000 [00:06<00:47, 7466.20it/s] 12%|█▏        | 47969/400000 [00:06<00:45, 7737.01it/s] 12%|█▏        | 48747/400000 [00:06<00:46, 7599.74it/s] 12%|█▏        | 49511/400000 [00:06<00:47, 7407.90it/s] 13%|█▎        | 50256/400000 [00:06<00:47, 7383.63it/s] 13%|█▎        | 51013/400000 [00:06<00:46, 7436.60it/s] 13%|█▎        | 51808/400000 [00:06<00:45, 7580.52it/s] 13%|█▎        | 52583/400000 [00:06<00:45, 7628.77it/s] 13%|█▎        | 53372/400000 [00:07<00:45, 7702.16it/s] 14%|█▎        | 54144/400000 [00:07<00:45, 7595.19it/s] 14%|█▎        | 54905/400000 [00:07<00:46, 7494.35it/s] 14%|█▍        | 55715/400000 [00:07<00:44, 7664.19it/s] 14%|█▍        | 56498/400000 [00:07<00:44, 7712.72it/s] 14%|█▍        | 57287/400000 [00:07<00:44, 7763.07it/s] 15%|█▍        | 58115/400000 [00:07<00:43, 7908.71it/s] 15%|█▍        | 58908/400000 [00:07<00:44, 7653.81it/s] 15%|█▍        | 59763/400000 [00:07<00:43, 7902.11it/s] 15%|█▌        | 60558/400000 [00:07<00:43, 7845.58it/s] 15%|█▌        | 61391/400000 [00:08<00:42, 7983.60it/s] 16%|█▌        | 62258/400000 [00:08<00:41, 8175.02it/s] 16%|█▌        | 63079/400000 [00:08<00:41, 8143.93it/s] 16%|█▌        | 63929/400000 [00:08<00:40, 8245.85it/s] 16%|█▌        | 64756/400000 [00:08<00:40, 8203.42it/s] 16%|█▋        | 65578/400000 [00:08<00:41, 8066.44it/s] 17%|█▋        | 66387/400000 [00:08<00:43, 7751.09it/s] 17%|█▋        | 67166/400000 [00:08<00:43, 7651.43it/s] 17%|█▋        | 67977/400000 [00:08<00:42, 7781.93it/s] 17%|█▋        | 68818/400000 [00:09<00:41, 7960.24it/s] 17%|█▋        | 69617/400000 [00:09<00:41, 7889.20it/s] 18%|█▊        | 70409/400000 [00:09<00:41, 7866.29it/s] 18%|█▊        | 71219/400000 [00:09<00:41, 7934.96it/s] 18%|█▊        | 72067/400000 [00:09<00:40, 8089.38it/s] 18%|█▊        | 72889/400000 [00:09<00:40, 8125.91it/s] 18%|█▊        | 73735/400000 [00:09<00:39, 8222.07it/s] 19%|█▊        | 74559/400000 [00:09<00:39, 8215.63it/s] 19%|█▉        | 75382/400000 [00:09<00:40, 7945.23it/s] 19%|█▉        | 76183/400000 [00:09<00:40, 7963.04it/s] 19%|█▉        | 76982/400000 [00:10<00:41, 7819.04it/s] 19%|█▉        | 77783/400000 [00:10<00:40, 7871.76it/s] 20%|█▉        | 78574/400000 [00:10<00:40, 7881.73it/s] 20%|█▉        | 79364/400000 [00:10<00:41, 7786.43it/s] 20%|██        | 80150/400000 [00:10<00:40, 7806.91it/s] 20%|██        | 80932/400000 [00:10<00:41, 7738.83it/s] 20%|██        | 81707/400000 [00:10<00:42, 7497.23it/s] 21%|██        | 82480/400000 [00:10<00:41, 7564.73it/s] 21%|██        | 83239/400000 [00:10<00:42, 7430.86it/s] 21%|██        | 84018/400000 [00:10<00:41, 7533.15it/s] 21%|██        | 84773/400000 [00:11<00:41, 7524.76it/s] 21%|██▏       | 85527/400000 [00:11<00:42, 7482.67it/s] 22%|██▏       | 86277/400000 [00:11<00:42, 7436.19it/s] 22%|██▏       | 87022/400000 [00:11<00:42, 7299.64it/s] 22%|██▏       | 87753/400000 [00:11<00:42, 7268.05it/s] 22%|██▏       | 88481/400000 [00:11<00:43, 7111.20it/s] 22%|██▏       | 89196/400000 [00:11<00:43, 7121.66it/s] 22%|██▏       | 89910/400000 [00:11<00:43, 7099.23it/s] 23%|██▎       | 90621/400000 [00:11<00:43, 7071.08it/s] 23%|██▎       | 91329/400000 [00:11<00:44, 7015.20it/s] 23%|██▎       | 92063/400000 [00:12<00:43, 7108.46it/s] 23%|██▎       | 92775/400000 [00:12<00:43, 7107.84it/s] 23%|██▎       | 93487/400000 [00:12<00:43, 7042.10it/s] 24%|██▎       | 94192/400000 [00:12<00:43, 7025.70it/s] 24%|██▎       | 94906/400000 [00:12<00:43, 7056.42it/s] 24%|██▍       | 95612/400000 [00:12<00:44, 6805.71it/s] 24%|██▍       | 96324/400000 [00:12<00:44, 6892.82it/s] 24%|██▍       | 97039/400000 [00:12<00:43, 6967.48it/s] 24%|██▍       | 97765/400000 [00:12<00:42, 7052.33it/s] 25%|██▍       | 98472/400000 [00:13<00:44, 6825.51it/s] 25%|██▍       | 99197/400000 [00:13<00:43, 6947.45it/s] 25%|██▍       | 99985/400000 [00:13<00:41, 7202.07it/s] 25%|██▌       | 100756/400000 [00:13<00:40, 7345.13it/s] 25%|██▌       | 101574/400000 [00:13<00:39, 7575.04it/s] 26%|██▌       | 102336/400000 [00:13<00:40, 7423.84it/s] 26%|██▌       | 103083/400000 [00:13<00:40, 7367.31it/s] 26%|██▌       | 103823/400000 [00:13<00:41, 7091.50it/s] 26%|██▌       | 104577/400000 [00:13<00:40, 7218.51it/s] 26%|██▋       | 105338/400000 [00:13<00:40, 7329.47it/s] 27%|██▋       | 106128/400000 [00:14<00:39, 7491.37it/s] 27%|██▋       | 106927/400000 [00:14<00:38, 7632.79it/s] 27%|██▋       | 107724/400000 [00:14<00:37, 7729.07it/s] 27%|██▋       | 108500/400000 [00:14<00:37, 7696.52it/s] 27%|██▋       | 109277/400000 [00:14<00:37, 7716.66it/s] 28%|██▊       | 110094/400000 [00:14<00:36, 7847.11it/s] 28%|██▊       | 110881/400000 [00:14<00:36, 7845.04it/s] 28%|██▊       | 111698/400000 [00:14<00:36, 7938.35it/s] 28%|██▊       | 112493/400000 [00:14<00:36, 7840.30it/s] 28%|██▊       | 113278/400000 [00:14<00:37, 7719.29it/s] 29%|██▊       | 114051/400000 [00:15<00:37, 7588.57it/s] 29%|██▊       | 114812/400000 [00:15<00:38, 7467.56it/s] 29%|██▉       | 115561/400000 [00:15<00:38, 7437.61it/s] 29%|██▉       | 116329/400000 [00:15<00:37, 7506.68it/s] 29%|██▉       | 117081/400000 [00:15<00:38, 7381.41it/s] 29%|██▉       | 117850/400000 [00:15<00:37, 7470.42it/s] 30%|██▉       | 118599/400000 [00:15<00:38, 7339.84it/s] 30%|██▉       | 119335/400000 [00:15<00:38, 7250.72it/s] 30%|███       | 120062/400000 [00:15<00:39, 7138.72it/s] 30%|███       | 120799/400000 [00:15<00:38, 7204.43it/s] 30%|███       | 121521/400000 [00:16<00:38, 7188.49it/s] 31%|███       | 122241/400000 [00:16<00:38, 7132.70it/s] 31%|███       | 122960/400000 [00:16<00:38, 7149.70it/s] 31%|███       | 123676/400000 [00:16<00:38, 7102.55it/s] 31%|███       | 124387/400000 [00:16<00:39, 6994.85it/s] 31%|███▏      | 125090/400000 [00:16<00:39, 7002.81it/s] 31%|███▏      | 125791/400000 [00:16<00:39, 6934.15it/s] 32%|███▏      | 126485/400000 [00:16<00:39, 6919.67it/s] 32%|███▏      | 127193/400000 [00:16<00:39, 6964.68it/s] 32%|███▏      | 127938/400000 [00:17<00:38, 7101.84it/s] 32%|███▏      | 128710/400000 [00:17<00:37, 7274.80it/s] 32%|███▏      | 129452/400000 [00:17<00:37, 7305.90it/s] 33%|███▎      | 130184/400000 [00:17<00:37, 7268.47it/s] 33%|███▎      | 130919/400000 [00:17<00:36, 7290.32it/s] 33%|███▎      | 131685/400000 [00:17<00:36, 7395.30it/s] 33%|███▎      | 132473/400000 [00:17<00:35, 7533.97it/s] 33%|███▎      | 133236/400000 [00:17<00:35, 7561.34it/s] 34%|███▎      | 134023/400000 [00:17<00:34, 7651.14it/s] 34%|███▎      | 134789/400000 [00:17<00:35, 7434.02it/s] 34%|███▍      | 135535/400000 [00:18<00:36, 7275.07it/s] 34%|███▍      | 136265/400000 [00:18<00:37, 7100.88it/s] 34%|███▍      | 136986/400000 [00:18<00:36, 7132.12it/s] 34%|███▍      | 137701/400000 [00:18<00:36, 7129.47it/s] 35%|███▍      | 138421/400000 [00:18<00:36, 7149.23it/s] 35%|███▍      | 139189/400000 [00:18<00:35, 7300.59it/s] 35%|███▍      | 139924/400000 [00:18<00:35, 7314.11it/s] 35%|███▌      | 140747/400000 [00:18<00:34, 7565.35it/s] 35%|███▌      | 141566/400000 [00:18<00:33, 7740.98it/s] 36%|███▌      | 142392/400000 [00:18<00:32, 7889.53it/s] 36%|███▌      | 143228/400000 [00:19<00:31, 8024.80it/s] 36%|███▌      | 144072/400000 [00:19<00:31, 8144.76it/s] 36%|███▌      | 144889/400000 [00:19<00:31, 8082.60it/s] 36%|███▋      | 145720/400000 [00:19<00:31, 8146.54it/s] 37%|███▋      | 146536/400000 [00:19<00:32, 7892.65it/s] 37%|███▋      | 147328/400000 [00:19<00:32, 7692.84it/s] 37%|███▋      | 148142/400000 [00:19<00:32, 7821.62it/s] 37%|███▋      | 148927/400000 [00:19<00:32, 7613.47it/s] 37%|███▋      | 149699/400000 [00:19<00:32, 7643.04it/s] 38%|███▊      | 150466/400000 [00:19<00:32, 7579.71it/s] 38%|███▊      | 151226/400000 [00:20<00:32, 7568.85it/s] 38%|███▊      | 152041/400000 [00:20<00:32, 7732.50it/s] 38%|███▊      | 152816/400000 [00:20<00:32, 7715.15it/s] 38%|███▊      | 153589/400000 [00:20<00:32, 7643.17it/s] 39%|███▊      | 154355/400000 [00:20<00:33, 7402.79it/s] 39%|███▉      | 155098/400000 [00:20<00:34, 7015.21it/s] 39%|███▉      | 155821/400000 [00:20<00:34, 7077.14it/s] 39%|███▉      | 156604/400000 [00:20<00:33, 7286.71it/s] 39%|███▉      | 157425/400000 [00:20<00:32, 7539.43it/s] 40%|███▉      | 158201/400000 [00:21<00:31, 7601.92it/s] 40%|███▉      | 158966/400000 [00:21<00:31, 7539.38it/s] 40%|███▉      | 159752/400000 [00:21<00:31, 7629.63it/s] 40%|████      | 160572/400000 [00:21<00:30, 7791.62it/s] 40%|████      | 161424/400000 [00:21<00:29, 7996.44it/s] 41%|████      | 162237/400000 [00:21<00:29, 8035.06it/s] 41%|████      | 163043/400000 [00:21<00:30, 7798.08it/s] 41%|████      | 163826/400000 [00:21<00:30, 7749.46it/s] 41%|████      | 164697/400000 [00:21<00:29, 8011.58it/s] 41%|████▏     | 165555/400000 [00:21<00:28, 8171.65it/s] 42%|████▏     | 166376/400000 [00:22<00:29, 7990.73it/s] 42%|████▏     | 167179/400000 [00:22<00:29, 7977.32it/s] 42%|████▏     | 167983/400000 [00:22<00:29, 7995.43it/s] 42%|████▏     | 168819/400000 [00:22<00:28, 8101.32it/s] 42%|████▏     | 169644/400000 [00:22<00:28, 8143.10it/s] 43%|████▎     | 170460/400000 [00:22<00:28, 8092.28it/s] 43%|████▎     | 171271/400000 [00:22<00:28, 7962.23it/s] 43%|████▎     | 172153/400000 [00:22<00:27, 8201.21it/s] 43%|████▎     | 172976/400000 [00:22<00:28, 8107.26it/s] 43%|████▎     | 173789/400000 [00:22<00:28, 7908.83it/s] 44%|████▎     | 174583/400000 [00:23<00:28, 7886.67it/s] 44%|████▍     | 175374/400000 [00:23<00:28, 7890.21it/s] 44%|████▍     | 176165/400000 [00:23<00:28, 7895.06it/s] 44%|████▍     | 176991/400000 [00:23<00:27, 7999.49it/s] 44%|████▍     | 177848/400000 [00:23<00:27, 8160.14it/s] 45%|████▍     | 178666/400000 [00:23<00:27, 8122.22it/s] 45%|████▍     | 179480/400000 [00:23<00:27, 8124.13it/s] 45%|████▌     | 180294/400000 [00:23<00:28, 7832.45it/s] 45%|████▌     | 181081/400000 [00:23<00:27, 7831.93it/s] 45%|████▌     | 181867/400000 [00:23<00:28, 7773.58it/s] 46%|████▌     | 182671/400000 [00:24<00:27, 7851.52it/s] 46%|████▌     | 183458/400000 [00:24<00:27, 7837.80it/s] 46%|████▌     | 184282/400000 [00:24<00:27, 7953.27it/s] 46%|████▋     | 185173/400000 [00:24<00:26, 8215.45it/s] 46%|████▋     | 185998/400000 [00:24<00:26, 8101.24it/s] 47%|████▋     | 186811/400000 [00:24<00:26, 8007.01it/s] 47%|████▋     | 187614/400000 [00:24<00:26, 7967.96it/s] 47%|████▋     | 188468/400000 [00:24<00:26, 8128.32it/s] 47%|████▋     | 189327/400000 [00:24<00:25, 8260.91it/s] 48%|████▊     | 190155/400000 [00:24<00:25, 8142.20it/s] 48%|████▊     | 190971/400000 [00:25<00:25, 8045.75it/s] 48%|████▊     | 191777/400000 [00:25<00:26, 7945.01it/s] 48%|████▊     | 192604/400000 [00:25<00:25, 8038.71it/s] 48%|████▊     | 193410/400000 [00:25<00:26, 7832.12it/s] 49%|████▊     | 194234/400000 [00:25<00:25, 7948.99it/s] 49%|████▉     | 195049/400000 [00:25<00:25, 8007.65it/s] 49%|████▉     | 195858/400000 [00:25<00:25, 8030.98it/s] 49%|████▉     | 196674/400000 [00:25<00:25, 8067.77it/s] 49%|████▉     | 197482/400000 [00:25<00:25, 7824.76it/s] 50%|████▉     | 198267/400000 [00:26<00:25, 7777.87it/s] 50%|████▉     | 199047/400000 [00:26<00:25, 7755.07it/s] 50%|████▉     | 199824/400000 [00:26<00:25, 7756.09it/s] 50%|█████     | 200601/400000 [00:26<00:25, 7754.32it/s] 50%|█████     | 201377/400000 [00:26<00:25, 7731.33it/s] 51%|█████     | 202151/400000 [00:26<00:25, 7703.58it/s] 51%|█████     | 202922/400000 [00:26<00:25, 7638.78it/s] 51%|█████     | 203775/400000 [00:26<00:24, 7885.44it/s] 51%|█████     | 204598/400000 [00:26<00:24, 7985.63it/s] 51%|█████▏    | 205411/400000 [00:26<00:24, 8027.49it/s] 52%|█████▏    | 206216/400000 [00:27<00:24, 7835.35it/s] 52%|█████▏    | 207002/400000 [00:27<00:24, 7728.01it/s] 52%|█████▏    | 207777/400000 [00:27<00:24, 7725.20it/s] 52%|█████▏    | 208589/400000 [00:27<00:24, 7837.96it/s] 52%|█████▏    | 209413/400000 [00:27<00:23, 7953.92it/s] 53%|█████▎    | 210210/400000 [00:27<00:24, 7878.45it/s] 53%|█████▎    | 210999/400000 [00:27<00:24, 7750.21it/s] 53%|█████▎    | 211787/400000 [00:27<00:24, 7787.51it/s] 53%|█████▎    | 212630/400000 [00:27<00:23, 7967.91it/s] 53%|█████▎    | 213429/400000 [00:27<00:23, 7927.82it/s] 54%|█████▎    | 214223/400000 [00:28<00:23, 7752.71it/s] 54%|█████▍    | 215000/400000 [00:28<00:23, 7756.48it/s] 54%|█████▍    | 215842/400000 [00:28<00:23, 7942.97it/s] 54%|█████▍    | 216676/400000 [00:28<00:22, 8056.55it/s] 54%|█████▍    | 217485/400000 [00:28<00:22, 8065.15it/s] 55%|█████▍    | 218293/400000 [00:28<00:22, 8057.88it/s] 55%|█████▍    | 219100/400000 [00:28<00:22, 8045.68it/s] 55%|█████▍    | 219935/400000 [00:28<00:22, 8132.41it/s] 55%|█████▌    | 220821/400000 [00:28<00:21, 8333.42it/s] 55%|█████▌    | 221666/400000 [00:28<00:21, 8365.39it/s] 56%|█████▌    | 222504/400000 [00:29<00:21, 8362.29it/s] 56%|█████▌    | 223342/400000 [00:29<00:21, 8185.77it/s] 56%|█████▌    | 224162/400000 [00:29<00:21, 8044.21it/s] 56%|█████▌    | 224968/400000 [00:29<00:21, 8007.61it/s] 56%|█████▋    | 225770/400000 [00:29<00:21, 7939.27it/s] 57%|█████▋    | 226566/400000 [00:29<00:21, 7945.41it/s] 57%|█████▋    | 227362/400000 [00:29<00:21, 7878.62it/s] 57%|█████▋    | 228151/400000 [00:29<00:21, 7821.03it/s] 57%|█████▋    | 228952/400000 [00:29<00:21, 7876.51it/s] 57%|█████▋    | 229741/400000 [00:29<00:21, 7816.93it/s] 58%|█████▊    | 230530/400000 [00:30<00:21, 7837.92it/s] 58%|█████▊    | 231315/400000 [00:30<00:21, 7781.80it/s] 58%|█████▊    | 232094/400000 [00:30<00:21, 7698.34it/s] 58%|█████▊    | 232935/400000 [00:30<00:21, 7896.62it/s] 58%|█████▊    | 233727/400000 [00:30<00:21, 7851.70it/s] 59%|█████▊    | 234541/400000 [00:30<00:20, 7933.13it/s] 59%|█████▉    | 235336/400000 [00:30<00:20, 7846.86it/s] 59%|█████▉    | 236122/400000 [00:30<00:21, 7735.02it/s] 59%|█████▉    | 236939/400000 [00:30<00:20, 7859.73it/s] 59%|█████▉    | 237727/400000 [00:31<00:21, 7688.66it/s] 60%|█████▉    | 238498/400000 [00:31<00:21, 7629.06it/s] 60%|█████▉    | 239263/400000 [00:31<00:22, 7100.51it/s] 60%|█████▉    | 239982/400000 [00:31<00:22, 7056.79it/s] 60%|██████    | 240694/400000 [00:31<00:22, 7066.64it/s] 60%|██████    | 241440/400000 [00:31<00:22, 7178.99it/s] 61%|██████    | 242189/400000 [00:31<00:21, 7269.01it/s] 61%|██████    | 242977/400000 [00:31<00:21, 7441.98it/s] 61%|██████    | 243787/400000 [00:31<00:20, 7626.45it/s] 61%|██████    | 244562/400000 [00:31<00:20, 7661.85it/s] 61%|██████▏   | 245331/400000 [00:32<00:20, 7595.32it/s] 62%|██████▏   | 246093/400000 [00:32<00:20, 7555.30it/s] 62%|██████▏   | 246850/400000 [00:32<00:20, 7514.57it/s] 62%|██████▏   | 247603/400000 [00:32<00:20, 7419.39it/s] 62%|██████▏   | 248346/400000 [00:32<00:20, 7380.95it/s] 62%|██████▏   | 249127/400000 [00:32<00:20, 7503.55it/s] 62%|██████▏   | 249896/400000 [00:32<00:19, 7556.01it/s] 63%|██████▎   | 250665/400000 [00:32<00:19, 7594.15it/s] 63%|██████▎   | 251528/400000 [00:32<00:18, 7875.96it/s] 63%|██████▎   | 252319/400000 [00:32<00:18, 7787.15it/s] 63%|██████▎   | 253137/400000 [00:33<00:18, 7899.38it/s] 63%|██████▎   | 253929/400000 [00:33<00:18, 7842.57it/s] 64%|██████▎   | 254715/400000 [00:33<00:18, 7673.63it/s] 64%|██████▍   | 255584/400000 [00:33<00:18, 7951.22it/s] 64%|██████▍   | 256407/400000 [00:33<00:17, 8031.71it/s] 64%|██████▍   | 257214/400000 [00:33<00:17, 8022.43it/s] 65%|██████▍   | 258019/400000 [00:33<00:17, 7986.52it/s] 65%|██████▍   | 258820/400000 [00:33<00:17, 7964.69it/s] 65%|██████▍   | 259621/400000 [00:33<00:17, 7977.59it/s] 65%|██████▌   | 260429/400000 [00:33<00:17, 8007.72it/s] 65%|██████▌   | 261261/400000 [00:34<00:17, 8098.06it/s] 66%|██████▌   | 262102/400000 [00:34<00:16, 8189.00it/s] 66%|██████▌   | 262922/400000 [00:34<00:16, 8066.63it/s] 66%|██████▌   | 263790/400000 [00:34<00:16, 8239.78it/s] 66%|██████▌   | 264616/400000 [00:34<00:17, 7904.75it/s] 66%|██████▋   | 265411/400000 [00:34<00:17, 7773.64it/s] 67%|██████▋   | 266226/400000 [00:34<00:16, 7882.14it/s] 67%|██████▋   | 267019/400000 [00:34<00:16, 7895.76it/s] 67%|██████▋   | 267811/400000 [00:34<00:16, 7859.00it/s] 67%|██████▋   | 268599/400000 [00:35<00:17, 7576.77it/s] 67%|██████▋   | 269393/400000 [00:35<00:17, 7680.52it/s] 68%|██████▊   | 270164/400000 [00:35<00:17, 7599.12it/s] 68%|██████▊   | 270959/400000 [00:35<00:16, 7700.87it/s] 68%|██████▊   | 271808/400000 [00:35<00:16, 7921.47it/s] 68%|██████▊   | 272603/400000 [00:35<00:16, 7907.49it/s] 68%|██████▊   | 273396/400000 [00:35<00:16, 7822.61it/s] 69%|██████▊   | 274180/400000 [00:35<00:16, 7788.87it/s] 69%|██████▊   | 274964/400000 [00:35<00:16, 7802.41it/s] 69%|██████▉   | 275746/400000 [00:35<00:15, 7798.62it/s] 69%|██████▉   | 276530/400000 [00:36<00:15, 7809.28it/s] 69%|██████▉   | 277312/400000 [00:36<00:15, 7730.19it/s] 70%|██████▉   | 278091/400000 [00:36<00:15, 7746.18it/s] 70%|██████▉   | 278868/400000 [00:36<00:15, 7752.35it/s] 70%|██████▉   | 279649/400000 [00:36<00:15, 7767.06it/s] 70%|███████   | 280426/400000 [00:36<00:15, 7694.87it/s] 70%|███████   | 281196/400000 [00:36<00:15, 7516.47it/s] 70%|███████   | 281949/400000 [00:36<00:15, 7456.23it/s] 71%|███████   | 282733/400000 [00:36<00:15, 7566.35it/s] 71%|███████   | 283491/400000 [00:36<00:15, 7499.42it/s] 71%|███████   | 284242/400000 [00:37<00:15, 7490.38it/s] 71%|███████   | 284992/400000 [00:37<00:15, 7350.43it/s] 71%|███████▏  | 285729/400000 [00:37<00:15, 7353.84it/s] 72%|███████▏  | 286490/400000 [00:37<00:15, 7428.30it/s] 72%|███████▏  | 287290/400000 [00:37<00:14, 7590.65it/s] 72%|███████▏  | 288051/400000 [00:37<00:14, 7577.07it/s] 72%|███████▏  | 288810/400000 [00:37<00:15, 7269.24it/s] 72%|███████▏  | 289599/400000 [00:37<00:14, 7443.16it/s] 73%|███████▎  | 290408/400000 [00:37<00:14, 7625.32it/s] 73%|███████▎  | 291204/400000 [00:37<00:14, 7721.32it/s] 73%|███████▎  | 292069/400000 [00:38<00:13, 7976.41it/s] 73%|███████▎  | 292931/400000 [00:38<00:13, 8158.90it/s] 73%|███████▎  | 293765/400000 [00:38<00:12, 8210.19it/s] 74%|███████▎  | 294589/400000 [00:38<00:13, 7892.99it/s] 74%|███████▍  | 295383/400000 [00:38<00:13, 7899.79it/s] 74%|███████▍  | 296182/400000 [00:38<00:13, 7924.96it/s] 74%|███████▍  | 296977/400000 [00:38<00:13, 7894.15it/s] 74%|███████▍  | 297769/400000 [00:38<00:13, 7836.65it/s] 75%|███████▍  | 298554/400000 [00:38<00:13, 7779.53it/s] 75%|███████▍  | 299333/400000 [00:38<00:13, 7655.37it/s] 75%|███████▌  | 300108/400000 [00:39<00:13, 7683.18it/s] 75%|███████▌  | 300945/400000 [00:39<00:12, 7876.79it/s] 75%|███████▌  | 301767/400000 [00:39<00:12, 7976.06it/s] 76%|███████▌  | 302567/400000 [00:39<00:12, 7942.91it/s] 76%|███████▌  | 303439/400000 [00:39<00:11, 8159.56it/s] 76%|███████▌  | 304258/400000 [00:39<00:11, 8165.62it/s] 76%|███████▋  | 305077/400000 [00:39<00:11, 8138.93it/s] 76%|███████▋  | 305948/400000 [00:39<00:11, 8301.94it/s] 77%|███████▋  | 306780/400000 [00:39<00:11, 8267.78it/s] 77%|███████▋  | 307608/400000 [00:40<00:11, 8102.61it/s] 77%|███████▋  | 308420/400000 [00:40<00:11, 8051.04it/s] 77%|███████▋  | 309227/400000 [00:40<00:11, 8019.47it/s] 78%|███████▊  | 310040/400000 [00:40<00:11, 8050.16it/s] 78%|███████▊  | 310846/400000 [00:40<00:11, 8043.65it/s] 78%|███████▊  | 311655/400000 [00:40<00:10, 8055.84it/s] 78%|███████▊  | 312461/400000 [00:40<00:10, 7984.44it/s] 78%|███████▊  | 313260/400000 [00:40<00:10, 7977.79it/s] 79%|███████▊  | 314059/400000 [00:40<00:10, 7972.89it/s] 79%|███████▊  | 314857/400000 [00:40<00:10, 7858.19it/s] 79%|███████▉  | 315644/400000 [00:41<00:11, 7645.92it/s] 79%|███████▉  | 316423/400000 [00:41<00:10, 7686.83it/s] 79%|███████▉  | 317193/400000 [00:41<00:10, 7608.88it/s] 79%|███████▉  | 317968/400000 [00:41<00:10, 7648.28it/s] 80%|███████▉  | 318734/400000 [00:41<00:10, 7650.81it/s] 80%|███████▉  | 319516/400000 [00:41<00:10, 7698.42it/s] 80%|████████  | 320296/400000 [00:41<00:10, 7727.62it/s] 80%|████████  | 321112/400000 [00:41<00:10, 7850.78it/s] 80%|████████  | 321933/400000 [00:41<00:09, 7954.42it/s] 81%|████████  | 322730/400000 [00:41<00:10, 7708.29it/s] 81%|████████  | 323504/400000 [00:42<00:09, 7658.68it/s] 81%|████████  | 324319/400000 [00:42<00:09, 7797.49it/s] 81%|████████▏ | 325153/400000 [00:42<00:09, 7951.60it/s] 81%|████████▏ | 325986/400000 [00:42<00:09, 8059.00it/s] 82%|████████▏ | 326794/400000 [00:42<00:09, 7962.23it/s] 82%|████████▏ | 327598/400000 [00:42<00:09, 7983.36it/s] 82%|████████▏ | 328398/400000 [00:42<00:09, 7911.76it/s] 82%|████████▏ | 329241/400000 [00:42<00:08, 8059.05it/s] 83%|████████▎ | 330049/400000 [00:42<00:08, 8032.34it/s] 83%|████████▎ | 330854/400000 [00:42<00:08, 7877.20it/s] 83%|████████▎ | 331644/400000 [00:43<00:08, 7788.61it/s] 83%|████████▎ | 332425/400000 [00:43<00:08, 7636.51it/s] 83%|████████▎ | 333191/400000 [00:43<00:08, 7499.56it/s] 83%|████████▎ | 333996/400000 [00:43<00:08, 7656.45it/s] 84%|████████▎ | 334786/400000 [00:43<00:08, 7726.77it/s] 84%|████████▍ | 335602/400000 [00:43<00:08, 7851.48it/s] 84%|████████▍ | 336394/400000 [00:43<00:08, 7870.15it/s] 84%|████████▍ | 337226/400000 [00:43<00:07, 7997.33it/s] 85%|████████▍ | 338107/400000 [00:43<00:07, 8223.02it/s] 85%|████████▍ | 338932/400000 [00:43<00:07, 8104.17it/s] 85%|████████▍ | 339757/400000 [00:44<00:07, 8143.17it/s] 85%|████████▌ | 340618/400000 [00:44<00:07, 8277.36it/s] 85%|████████▌ | 341448/400000 [00:44<00:07, 8231.50it/s] 86%|████████▌ | 342273/400000 [00:44<00:07, 8197.73it/s] 86%|████████▌ | 343094/400000 [00:44<00:07, 8033.14it/s] 86%|████████▌ | 343899/400000 [00:44<00:07, 7959.11it/s] 86%|████████▌ | 344732/400000 [00:44<00:06, 8065.66it/s] 86%|████████▋ | 345581/400000 [00:44<00:06, 8187.39it/s] 87%|████████▋ | 346401/400000 [00:44<00:06, 8039.12it/s] 87%|████████▋ | 347207/400000 [00:45<00:06, 7930.04it/s] 87%|████████▋ | 348002/400000 [00:45<00:06, 7810.51it/s] 87%|████████▋ | 348785/400000 [00:45<00:06, 7382.25it/s] 87%|████████▋ | 349530/400000 [00:45<00:06, 7280.15it/s] 88%|████████▊ | 350263/400000 [00:45<00:06, 7292.90it/s] 88%|████████▊ | 351051/400000 [00:45<00:06, 7456.99it/s] 88%|████████▊ | 351849/400000 [00:45<00:06, 7604.87it/s] 88%|████████▊ | 352630/400000 [00:45<00:06, 7663.00it/s] 88%|████████▊ | 353482/400000 [00:45<00:05, 7898.76it/s] 89%|████████▊ | 354276/400000 [00:45<00:05, 7893.80it/s] 89%|████████▉ | 355112/400000 [00:46<00:05, 8027.79it/s] 89%|████████▉ | 355917/400000 [00:46<00:05, 7979.33it/s] 89%|████████▉ | 356731/400000 [00:46<00:05, 8024.34it/s] 89%|████████▉ | 357537/400000 [00:46<00:05, 8032.46it/s] 90%|████████▉ | 358388/400000 [00:46<00:05, 8168.63it/s] 90%|████████▉ | 359215/400000 [00:46<00:04, 8197.81it/s] 90%|█████████ | 360036/400000 [00:46<00:04, 8118.53it/s] 90%|█████████ | 360884/400000 [00:46<00:04, 8222.49it/s] 90%|█████████ | 361708/400000 [00:46<00:04, 8221.45it/s] 91%|█████████ | 362531/400000 [00:46<00:04, 8130.41it/s] 91%|█████████ | 363345/400000 [00:47<00:04, 7853.40it/s] 91%|█████████ | 364133/400000 [00:47<00:04, 7677.83it/s] 91%|█████████ | 364923/400000 [00:47<00:04, 7742.02it/s] 91%|█████████▏| 365700/400000 [00:47<00:04, 7747.06it/s] 92%|█████████▏| 366477/400000 [00:47<00:04, 7579.33it/s] 92%|█████████▏| 367275/400000 [00:47<00:04, 7693.54it/s] 92%|█████████▏| 368047/400000 [00:47<00:04, 7530.81it/s] 92%|█████████▏| 368832/400000 [00:47<00:04, 7622.54it/s] 92%|█████████▏| 369664/400000 [00:47<00:03, 7818.85it/s] 93%|█████████▎| 370449/400000 [00:47<00:03, 7745.05it/s] 93%|█████████▎| 371255/400000 [00:48<00:03, 7835.42it/s] 93%|█████████▎| 372052/400000 [00:48<00:03, 7874.82it/s] 93%|█████████▎| 372866/400000 [00:48<00:03, 7951.47it/s] 93%|█████████▎| 373688/400000 [00:48<00:03, 8028.20it/s] 94%|█████████▎| 374492/400000 [00:48<00:03, 7991.21it/s] 94%|█████████▍| 375300/400000 [00:48<00:03, 8017.43it/s] 94%|█████████▍| 376168/400000 [00:48<00:02, 8204.92it/s] 94%|█████████▍| 376990/400000 [00:48<00:02, 8095.35it/s] 94%|█████████▍| 377801/400000 [00:48<00:02, 8086.10it/s] 95%|█████████▍| 378611/400000 [00:49<00:02, 7985.44it/s] 95%|█████████▍| 379461/400000 [00:49<00:02, 8132.78it/s] 95%|█████████▌| 380276/400000 [00:49<00:02, 7935.45it/s] 95%|█████████▌| 381124/400000 [00:49<00:02, 8089.07it/s] 96%|█████████▌| 382000/400000 [00:49<00:02, 8273.97it/s] 96%|█████████▌| 382830/400000 [00:49<00:02, 8269.59it/s] 96%|█████████▌| 383659/400000 [00:49<00:02, 7921.22it/s] 96%|█████████▌| 384503/400000 [00:49<00:01, 8069.87it/s] 96%|█████████▋| 385314/400000 [00:49<00:01, 7992.06it/s] 97%|█████████▋| 386131/400000 [00:49<00:01, 8042.98it/s] 97%|█████████▋| 386970/400000 [00:50<00:01, 8141.62it/s] 97%|█████████▋| 387786/400000 [00:50<00:01, 8032.16it/s] 97%|█████████▋| 388591/400000 [00:50<00:01, 8010.59it/s] 97%|█████████▋| 389410/400000 [00:50<00:01, 8063.18it/s] 98%|█████████▊| 390218/400000 [00:50<00:01, 7979.53it/s] 98%|█████████▊| 391017/400000 [00:50<00:01, 7856.84it/s] 98%|█████████▊| 391804/400000 [00:50<00:01, 7721.91it/s] 98%|█████████▊| 392582/400000 [00:50<00:00, 7736.53it/s] 98%|█████████▊| 393357/400000 [00:50<00:00, 7643.96it/s] 99%|█████████▊| 394123/400000 [00:50<00:00, 7599.45it/s] 99%|█████████▊| 394884/400000 [00:51<00:00, 7532.54it/s] 99%|█████████▉| 395638/400000 [00:51<00:00, 7454.19it/s] 99%|█████████▉| 396385/400000 [00:51<00:00, 7383.49it/s] 99%|█████████▉| 397184/400000 [00:51<00:00, 7555.08it/s] 99%|█████████▉| 397975/400000 [00:51<00:00, 7657.89it/s]100%|█████████▉| 398794/400000 [00:51<00:00, 7807.51it/s]100%|█████████▉| 399577/400000 [00:51<00:00, 7722.91it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7730.30it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8278115d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010965317018724477 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010912840581657894 	 Accuracy: 63

  model saves at 63% accuracy 

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
2020-05-12 21:23:47.664854: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 21:23:47.669493: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 21:23:47.669642: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad5c402970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 21:23:47.669677: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f822151a1d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5164 - accuracy: 0.5098
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5363 - accuracy: 0.5085
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5571 - accuracy: 0.5071
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6340 - accuracy: 0.5021
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6007 - accuracy: 0.5043
11000/25000 [============>.................] - ETA: 4s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 4s - loss: 7.6794 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6619 - accuracy: 0.5003
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6480 - accuracy: 0.5012
15000/25000 [=================>............] - ETA: 3s - loss: 7.6707 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6837 - accuracy: 0.4989
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 10s 383us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f81d7f30550> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f81d7f1b588> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 920ms/step - loss: 1.7783 - crf_viterbi_accuracy: 0.2400 - val_loss: 1.6753 - val_crf_viterbi_accuracy: 0.4267

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

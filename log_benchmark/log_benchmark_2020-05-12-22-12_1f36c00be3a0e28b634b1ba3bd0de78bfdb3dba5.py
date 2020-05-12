
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff00729ff60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 22:12:36.839897
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 22:12:36.843187
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 22:12:36.845590
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 22:12:36.848461
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff013069438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353509.6562
Epoch 2/10

1/1 [==============================] - 0s 88ms/step - loss: 257240.6875
Epoch 3/10

1/1 [==============================] - 0s 87ms/step - loss: 163502.2344
Epoch 4/10

1/1 [==============================] - 0s 90ms/step - loss: 96400.2188
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 56534.3398
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 34630.0352
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 22352.5508
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 15300.0059
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 11073.5469
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 8406.9541

  #### Inference Need return ypred, ytrue ######################### 
[[-0.14820203  6.5560207   7.616928    5.9641037   6.417455    7.42127
   7.34372     7.719402    7.666386    6.7548966   8.115058    5.634235
   6.800407    5.823252    6.816253    6.678488    6.257852    6.364865
   7.1126075   6.6324472   7.435202    5.9166737   8.058292    7.6113377
   6.555685    8.17271     7.999414    5.9616246   6.2477055   5.992982
   6.488097    7.705196    5.61872     7.6870112   7.330826    6.5955176
   5.4122148   7.391216    5.676224    7.441731    7.2981124   7.2270346
   7.5656004   4.930769    7.016369    7.3649473   5.3737097   7.65519
   7.646004    7.6310267   5.972217    6.9410458   6.734211    6.3197584
   6.186367    6.404965    8.466086    7.000558    5.3767085   7.0786247
  -0.7659239   1.1786815   1.1706043   0.6942993   0.7709085   1.6889347
   0.84589326 -0.44947723 -0.9807789   0.2829975   0.6344005  -0.22655255
  -0.12431917  0.37952393 -0.45843345  0.18217081  0.3752517   0.32940513
   0.55117303  1.602185    0.6210736  -0.98950505 -0.83728516 -0.7806231
  -1.4130881   0.32320467  0.8428039  -0.68786156  1.3998898  -0.3882399
  -0.36170578  0.7221607  -0.570154   -1.5257518  -1.0752753   0.3119859
  -1.565497   -1.9033      0.09922081 -0.9905336   1.1428449   0.31823283
  -1.1890426  -0.28206414  0.3923167   1.7624627  -1.3172144  -0.97026134
   0.2776984  -0.70830226 -1.8701457   1.507899    0.06948957  0.67098784
  -0.9726744  -0.70711726  0.39957267  1.0179608  -0.7934231   0.39506456
   0.26995167 -1.7724664   0.3044452   0.45522934  0.6919006  -0.12651904
   0.6363793  -1.2496345   1.7540233  -0.17119594  0.24619782  0.0137012
  -0.30577803 -1.2638327   1.2576532   0.33986595  0.8081285  -1.1391857
   0.4843737  -0.07425851  1.1400757   0.7677879  -0.6743799   0.7013671
  -1.0476161  -1.2242417  -0.3639085  -1.3189492  -0.9577146   0.4876283
   0.34533864 -0.7922742  -1.7532266  -0.8051321   0.6637709  -0.42508644
  -1.7700311   0.73761535 -0.33994722  0.2961876  -0.7170207   0.85172236
   0.5853346   0.25474724 -0.1675567  -0.35761565  1.7526616   0.55545914
  -0.5290555  -0.31103683 -0.13282943 -0.14408007  0.33576924  1.4651521
   0.68051416  2.1509595   0.9570205  -0.24621186 -0.9013358   0.4315136
   0.05897993  5.9373536   8.121624    7.878829    7.2208395   7.041693
   7.548411    8.098699    6.8644266   8.22513     7.156585    7.764862
   7.9743733   7.4787827   7.8381953   7.949237    5.872278    6.807485
   8.6932      6.9201317   6.1525326   7.960146    8.444953    7.48385
   6.898654    6.8651795   8.781896    7.4298077   7.684136    6.782421
   8.268012    6.9289427   7.040347    7.8497252   8.2611685   7.116475
   6.769674    8.423144    7.023948    7.8178587   7.6470146   7.8879995
   7.434152    8.002434    6.511111    6.3992558   7.699627    8.045111
   8.236615    7.330826    7.6166215   7.0220394   6.9788327   6.128159
   7.487327    7.76973     7.1824236   7.796547    7.0837955   6.8994174
   2.2220457   1.8752313   2.703886    2.0462031   2.1426678   1.0216455
   1.3707573   0.2721231   1.0326692   1.0110859   0.6348735   0.83359355
   1.6809307   0.28279984  2.927753    0.43268895  0.4066354   2.5649953
   0.49690068  0.48762333  2.2526088   1.3956676   0.7362509   0.19557852
   1.7075157   1.662866    0.807477    0.5701536   0.8569333   0.7779764
   2.337222    1.3490261   0.8897041   1.8736968   0.1767397   1.5903081
   0.45025194  2.2869148   2.1425943   1.4261508   0.4482515   1.7648491
   1.6869882   1.1637118   0.7685819   1.9697682   1.1321071   2.7198687
   1.2717872   1.3352027   0.5965926   1.5376213   0.11778784  0.69220406
   0.9745235   1.1516706   1.1498275   1.3199329   2.0062912   0.24821067
   1.1800082   1.8711255   1.8931882   0.8481776   1.0333924   1.4409565
   0.35148656  1.2524158   2.0025086   0.8805193   1.3800273   1.1215663
   1.0535125   1.1681614   0.12959331  0.9716647   0.2508477   0.48612845
   0.7883467   1.681315    0.49802387  1.5234401   0.7343308   1.3080449
   1.5060431   1.3392491   0.5530919   1.9831922   2.0475197   1.4164069
   0.26807725  1.2746832   2.6633744   1.5984682   0.75513387  0.8962535
   0.9480407   1.5631573   0.7688933   2.5793672   1.0028002   0.62608933
   1.4389583   1.084053    0.6174378   0.4849316   0.39351356  1.1130594
   1.6796763   0.28633046  1.0795721   0.31162494  2.2876272   1.0154996
   1.464822    1.3076248   1.2964084   0.21590602  1.1279857   1.1053706
   4.434943   -4.0898204  -6.654429  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 22:12:44.924552
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.9369
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 22:12:44.927666
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9036.46
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 22:12:44.930271
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1348
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 22:12:44.933299
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -808.279
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140668547150288
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140667588448776
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140667588449280
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140667588449784
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140667588450288
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140667588450792

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff01b016358> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.576853
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.547050
grad_step = 000002, loss = 0.524059
grad_step = 000003, loss = 0.497728
grad_step = 000004, loss = 0.466268
grad_step = 000005, loss = 0.434371
grad_step = 000006, loss = 0.415730
grad_step = 000007, loss = 0.413264
grad_step = 000008, loss = 0.401332
grad_step = 000009, loss = 0.381648
grad_step = 000010, loss = 0.365891
grad_step = 000011, loss = 0.356127
grad_step = 000012, loss = 0.349005
grad_step = 000013, loss = 0.341037
grad_step = 000014, loss = 0.330893
grad_step = 000015, loss = 0.318939
grad_step = 000016, loss = 0.306546
grad_step = 000017, loss = 0.295468
grad_step = 000018, loss = 0.286245
grad_step = 000019, loss = 0.277200
grad_step = 000020, loss = 0.266888
grad_step = 000021, loss = 0.256098
grad_step = 000022, loss = 0.246177
grad_step = 000023, loss = 0.237556
grad_step = 000024, loss = 0.229608
grad_step = 000025, loss = 0.221466
grad_step = 000026, loss = 0.212760
grad_step = 000027, loss = 0.203783
grad_step = 000028, loss = 0.195197
grad_step = 000029, loss = 0.187427
grad_step = 000030, loss = 0.180086
grad_step = 000031, loss = 0.172389
grad_step = 000032, loss = 0.164607
grad_step = 000033, loss = 0.157484
grad_step = 000034, loss = 0.151036
grad_step = 000035, loss = 0.144734
grad_step = 000036, loss = 0.138255
grad_step = 000037, loss = 0.131784
grad_step = 000038, loss = 0.125682
grad_step = 000039, loss = 0.119963
grad_step = 000040, loss = 0.114372
grad_step = 000041, loss = 0.108867
grad_step = 000042, loss = 0.103583
grad_step = 000043, loss = 0.098582
grad_step = 000044, loss = 0.093813
grad_step = 000045, loss = 0.089216
grad_step = 000046, loss = 0.084744
grad_step = 000047, loss = 0.080399
grad_step = 000048, loss = 0.076233
grad_step = 000049, loss = 0.072274
grad_step = 000050, loss = 0.068490
grad_step = 000051, loss = 0.064837
grad_step = 000052, loss = 0.061343
grad_step = 000053, loss = 0.058063
grad_step = 000054, loss = 0.054939
grad_step = 000055, loss = 0.051903
grad_step = 000056, loss = 0.049003
grad_step = 000057, loss = 0.046264
grad_step = 000058, loss = 0.043652
grad_step = 000059, loss = 0.041161
grad_step = 000060, loss = 0.038804
grad_step = 000061, loss = 0.036582
grad_step = 000062, loss = 0.034485
grad_step = 000063, loss = 0.032491
grad_step = 000064, loss = 0.030588
grad_step = 000065, loss = 0.028788
grad_step = 000066, loss = 0.027102
grad_step = 000067, loss = 0.025509
grad_step = 000068, loss = 0.023996
grad_step = 000069, loss = 0.022580
grad_step = 000070, loss = 0.021256
grad_step = 000071, loss = 0.020006
grad_step = 000072, loss = 0.018819
grad_step = 000073, loss = 0.017699
grad_step = 000074, loss = 0.016654
grad_step = 000075, loss = 0.015674
grad_step = 000076, loss = 0.014749
grad_step = 000077, loss = 0.013881
grad_step = 000078, loss = 0.013071
grad_step = 000079, loss = 0.012310
grad_step = 000080, loss = 0.011591
grad_step = 000081, loss = 0.010912
grad_step = 000082, loss = 0.010283
grad_step = 000083, loss = 0.009693
grad_step = 000084, loss = 0.009138
grad_step = 000085, loss = 0.008618
grad_step = 000086, loss = 0.008131
grad_step = 000087, loss = 0.007675
grad_step = 000088, loss = 0.007247
grad_step = 000089, loss = 0.006847
grad_step = 000090, loss = 0.006475
grad_step = 000091, loss = 0.006127
grad_step = 000092, loss = 0.005801
grad_step = 000093, loss = 0.005498
grad_step = 000094, loss = 0.005216
grad_step = 000095, loss = 0.004954
grad_step = 000096, loss = 0.004709
grad_step = 000097, loss = 0.004484
grad_step = 000098, loss = 0.004274
grad_step = 000099, loss = 0.004080
grad_step = 000100, loss = 0.003901
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003735
grad_step = 000102, loss = 0.003583
grad_step = 000103, loss = 0.003443
grad_step = 000104, loss = 0.003315
grad_step = 000105, loss = 0.003197
grad_step = 000106, loss = 0.003089
grad_step = 000107, loss = 0.002990
grad_step = 000108, loss = 0.002900
grad_step = 000109, loss = 0.002818
grad_step = 000110, loss = 0.002744
grad_step = 000111, loss = 0.002676
grad_step = 000112, loss = 0.002615
grad_step = 000113, loss = 0.002559
grad_step = 000114, loss = 0.002508
grad_step = 000115, loss = 0.002462
grad_step = 000116, loss = 0.002421
grad_step = 000117, loss = 0.002384
grad_step = 000118, loss = 0.002351
grad_step = 000119, loss = 0.002322
grad_step = 000120, loss = 0.002294
grad_step = 000121, loss = 0.002268
grad_step = 000122, loss = 0.002245
grad_step = 000123, loss = 0.002227
grad_step = 000124, loss = 0.002208
grad_step = 000125, loss = 0.002190
grad_step = 000126, loss = 0.002174
grad_step = 000127, loss = 0.002161
grad_step = 000128, loss = 0.002149
grad_step = 000129, loss = 0.002137
grad_step = 000130, loss = 0.002124
grad_step = 000131, loss = 0.002113
grad_step = 000132, loss = 0.002103
grad_step = 000133, loss = 0.002094
grad_step = 000134, loss = 0.002086
grad_step = 000135, loss = 0.002079
grad_step = 000136, loss = 0.002071
grad_step = 000137, loss = 0.002063
grad_step = 000138, loss = 0.002054
grad_step = 000139, loss = 0.002045
grad_step = 000140, loss = 0.002036
grad_step = 000141, loss = 0.002027
grad_step = 000142, loss = 0.002020
grad_step = 000143, loss = 0.002012
grad_step = 000144, loss = 0.002005
grad_step = 000145, loss = 0.001999
grad_step = 000146, loss = 0.001996
grad_step = 000147, loss = 0.002003
grad_step = 000148, loss = 0.002027
grad_step = 000149, loss = 0.002054
grad_step = 000150, loss = 0.002047
grad_step = 000151, loss = 0.001988
grad_step = 000152, loss = 0.001947
grad_step = 000153, loss = 0.001960
grad_step = 000154, loss = 0.001989
grad_step = 000155, loss = 0.001983
grad_step = 000156, loss = 0.001941
grad_step = 000157, loss = 0.001913
grad_step = 000158, loss = 0.001921
grad_step = 000159, loss = 0.001938
grad_step = 000160, loss = 0.001937
grad_step = 000161, loss = 0.001911
grad_step = 000162, loss = 0.001883
grad_step = 000163, loss = 0.001872
grad_step = 000164, loss = 0.001878
grad_step = 000165, loss = 0.001886
grad_step = 000166, loss = 0.001894
grad_step = 000167, loss = 0.001876
grad_step = 000168, loss = 0.001856
grad_step = 000169, loss = 0.001835
grad_step = 000170, loss = 0.001822
grad_step = 000171, loss = 0.001822
grad_step = 000172, loss = 0.001821
grad_step = 000173, loss = 0.001827
grad_step = 000174, loss = 0.001828
grad_step = 000175, loss = 0.001830
grad_step = 000176, loss = 0.001832
grad_step = 000177, loss = 0.001832
grad_step = 000178, loss = 0.001828
grad_step = 000179, loss = 0.001808
grad_step = 000180, loss = 0.001782
grad_step = 000181, loss = 0.001754
grad_step = 000182, loss = 0.001733
grad_step = 000183, loss = 0.001720
grad_step = 000184, loss = 0.001714
grad_step = 000185, loss = 0.001714
grad_step = 000186, loss = 0.001719
grad_step = 000187, loss = 0.001733
grad_step = 000188, loss = 0.001773
grad_step = 000189, loss = 0.001885
grad_step = 000190, loss = 0.001929
grad_step = 000191, loss = 0.001919
grad_step = 000192, loss = 0.001828
grad_step = 000193, loss = 0.001714
grad_step = 000194, loss = 0.001672
grad_step = 000195, loss = 0.001755
grad_step = 000196, loss = 0.001815
grad_step = 000197, loss = 0.001738
grad_step = 000198, loss = 0.001643
grad_step = 000199, loss = 0.001649
grad_step = 000200, loss = 0.001712
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001719
grad_step = 000202, loss = 0.001657
grad_step = 000203, loss = 0.001630
grad_step = 000204, loss = 0.001692
grad_step = 000205, loss = 0.001727
grad_step = 000206, loss = 0.001683
grad_step = 000207, loss = 0.001610
grad_step = 000208, loss = 0.001633
grad_step = 000209, loss = 0.001680
grad_step = 000210, loss = 0.001639
grad_step = 000211, loss = 0.001602
grad_step = 000212, loss = 0.001625
grad_step = 000213, loss = 0.001639
grad_step = 000214, loss = 0.001617
grad_step = 000215, loss = 0.001598
grad_step = 000216, loss = 0.001611
grad_step = 000217, loss = 0.001630
grad_step = 000218, loss = 0.001613
grad_step = 000219, loss = 0.001594
grad_step = 000220, loss = 0.001594
grad_step = 000221, loss = 0.001606
grad_step = 000222, loss = 0.001611
grad_step = 000223, loss = 0.001596
grad_step = 000224, loss = 0.001587
grad_step = 000225, loss = 0.001591
grad_step = 000226, loss = 0.001597
grad_step = 000227, loss = 0.001597
grad_step = 000228, loss = 0.001588
grad_step = 000229, loss = 0.001583
grad_step = 000230, loss = 0.001584
grad_step = 000231, loss = 0.001588
grad_step = 000232, loss = 0.001590
grad_step = 000233, loss = 0.001586
grad_step = 000234, loss = 0.001580
grad_step = 000235, loss = 0.001577
grad_step = 000236, loss = 0.001577
grad_step = 000237, loss = 0.001580
grad_step = 000238, loss = 0.001581
grad_step = 000239, loss = 0.001580
grad_step = 000240, loss = 0.001576
grad_step = 000241, loss = 0.001572
grad_step = 000242, loss = 0.001570
grad_step = 000243, loss = 0.001570
grad_step = 000244, loss = 0.001571
grad_step = 000245, loss = 0.001571
grad_step = 000246, loss = 0.001572
grad_step = 000247, loss = 0.001571
grad_step = 000248, loss = 0.001571
grad_step = 000249, loss = 0.001569
grad_step = 000250, loss = 0.001567
grad_step = 000251, loss = 0.001565
grad_step = 000252, loss = 0.001564
grad_step = 000253, loss = 0.001562
grad_step = 000254, loss = 0.001561
grad_step = 000255, loss = 0.001561
grad_step = 000256, loss = 0.001561
grad_step = 000257, loss = 0.001563
grad_step = 000258, loss = 0.001567
grad_step = 000259, loss = 0.001576
grad_step = 000260, loss = 0.001596
grad_step = 000261, loss = 0.001634
grad_step = 000262, loss = 0.001709
grad_step = 000263, loss = 0.001816
grad_step = 000264, loss = 0.001955
grad_step = 000265, loss = 0.001933
grad_step = 000266, loss = 0.001777
grad_step = 000267, loss = 0.001593
grad_step = 000268, loss = 0.001607
grad_step = 000269, loss = 0.001725
grad_step = 000270, loss = 0.001690
grad_step = 000271, loss = 0.001618
grad_step = 000272, loss = 0.001641
grad_step = 000273, loss = 0.001632
grad_step = 000274, loss = 0.001588
grad_step = 000275, loss = 0.001610
grad_step = 000276, loss = 0.001645
grad_step = 000277, loss = 0.001591
grad_step = 000278, loss = 0.001551
grad_step = 000279, loss = 0.001586
grad_step = 000280, loss = 0.001598
grad_step = 000281, loss = 0.001569
grad_step = 000282, loss = 0.001567
grad_step = 000283, loss = 0.001582
grad_step = 000284, loss = 0.001565
grad_step = 000285, loss = 0.001543
grad_step = 000286, loss = 0.001562
grad_step = 000287, loss = 0.001579
grad_step = 000288, loss = 0.001562
grad_step = 000289, loss = 0.001548
grad_step = 000290, loss = 0.001553
grad_step = 000291, loss = 0.001553
grad_step = 000292, loss = 0.001541
grad_step = 000293, loss = 0.001540
grad_step = 000294, loss = 0.001550
grad_step = 000295, loss = 0.001554
grad_step = 000296, loss = 0.001547
grad_step = 000297, loss = 0.001539
grad_step = 000298, loss = 0.001539
grad_step = 000299, loss = 0.001541
grad_step = 000300, loss = 0.001537
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001531
grad_step = 000302, loss = 0.001529
grad_step = 000303, loss = 0.001532
grad_step = 000304, loss = 0.001535
grad_step = 000305, loss = 0.001534
grad_step = 000306, loss = 0.001532
grad_step = 000307, loss = 0.001532
grad_step = 000308, loss = 0.001535
grad_step = 000309, loss = 0.001539
grad_step = 000310, loss = 0.001542
grad_step = 000311, loss = 0.001545
grad_step = 000312, loss = 0.001550
grad_step = 000313, loss = 0.001558
grad_step = 000314, loss = 0.001575
grad_step = 000315, loss = 0.001598
grad_step = 000316, loss = 0.001632
grad_step = 000317, loss = 0.001663
grad_step = 000318, loss = 0.001704
grad_step = 000319, loss = 0.001715
grad_step = 000320, loss = 0.001709
grad_step = 000321, loss = 0.001650
grad_step = 000322, loss = 0.001580
grad_step = 000323, loss = 0.001526
grad_step = 000324, loss = 0.001517
grad_step = 000325, loss = 0.001546
grad_step = 000326, loss = 0.001580
grad_step = 000327, loss = 0.001591
grad_step = 000328, loss = 0.001566
grad_step = 000329, loss = 0.001530
grad_step = 000330, loss = 0.001512
grad_step = 000331, loss = 0.001520
grad_step = 000332, loss = 0.001540
grad_step = 000333, loss = 0.001547
grad_step = 000334, loss = 0.001535
grad_step = 000335, loss = 0.001516
grad_step = 000336, loss = 0.001508
grad_step = 000337, loss = 0.001514
grad_step = 000338, loss = 0.001523
grad_step = 000339, loss = 0.001526
grad_step = 000340, loss = 0.001519
grad_step = 000341, loss = 0.001509
grad_step = 000342, loss = 0.001504
grad_step = 000343, loss = 0.001507
grad_step = 000344, loss = 0.001512
grad_step = 000345, loss = 0.001516
grad_step = 000346, loss = 0.001514
grad_step = 000347, loss = 0.001509
grad_step = 000348, loss = 0.001506
grad_step = 000349, loss = 0.001505
grad_step = 000350, loss = 0.001509
grad_step = 000351, loss = 0.001517
grad_step = 000352, loss = 0.001526
grad_step = 000353, loss = 0.001535
grad_step = 000354, loss = 0.001542
grad_step = 000355, loss = 0.001547
grad_step = 000356, loss = 0.001543
grad_step = 000357, loss = 0.001533
grad_step = 000358, loss = 0.001517
grad_step = 000359, loss = 0.001503
grad_step = 000360, loss = 0.001496
grad_step = 000361, loss = 0.001494
grad_step = 000362, loss = 0.001495
grad_step = 000363, loss = 0.001496
grad_step = 000364, loss = 0.001498
grad_step = 000365, loss = 0.001500
grad_step = 000366, loss = 0.001503
grad_step = 000367, loss = 0.001506
grad_step = 000368, loss = 0.001510
grad_step = 000369, loss = 0.001513
grad_step = 000370, loss = 0.001516
grad_step = 000371, loss = 0.001517
grad_step = 000372, loss = 0.001519
grad_step = 000373, loss = 0.001522
grad_step = 000374, loss = 0.001527
grad_step = 000375, loss = 0.001532
grad_step = 000376, loss = 0.001536
grad_step = 000377, loss = 0.001533
grad_step = 000378, loss = 0.001526
grad_step = 000379, loss = 0.001513
grad_step = 000380, loss = 0.001502
grad_step = 000381, loss = 0.001494
grad_step = 000382, loss = 0.001491
grad_step = 000383, loss = 0.001489
grad_step = 000384, loss = 0.001486
grad_step = 000385, loss = 0.001482
grad_step = 000386, loss = 0.001478
grad_step = 000387, loss = 0.001476
grad_step = 000388, loss = 0.001477
grad_step = 000389, loss = 0.001481
grad_step = 000390, loss = 0.001485
grad_step = 000391, loss = 0.001489
grad_step = 000392, loss = 0.001491
grad_step = 000393, loss = 0.001492
grad_step = 000394, loss = 0.001494
grad_step = 000395, loss = 0.001500
grad_step = 000396, loss = 0.001512
grad_step = 000397, loss = 0.001537
grad_step = 000398, loss = 0.001579
grad_step = 000399, loss = 0.001657
grad_step = 000400, loss = 0.001733
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001797
grad_step = 000402, loss = 0.001775
grad_step = 000403, loss = 0.001737
grad_step = 000404, loss = 0.001657
grad_step = 000405, loss = 0.001582
grad_step = 000406, loss = 0.001515
grad_step = 000407, loss = 0.001484
grad_step = 000408, loss = 0.001512
grad_step = 000409, loss = 0.001572
grad_step = 000410, loss = 0.001586
grad_step = 000411, loss = 0.001530
grad_step = 000412, loss = 0.001472
grad_step = 000413, loss = 0.001469
grad_step = 000414, loss = 0.001500
grad_step = 000415, loss = 0.001518
grad_step = 000416, loss = 0.001507
grad_step = 000417, loss = 0.001485
grad_step = 000418, loss = 0.001469
grad_step = 000419, loss = 0.001466
grad_step = 000420, loss = 0.001473
grad_step = 000421, loss = 0.001479
grad_step = 000422, loss = 0.001479
grad_step = 000423, loss = 0.001470
grad_step = 000424, loss = 0.001462
grad_step = 000425, loss = 0.001457
grad_step = 000426, loss = 0.001457
grad_step = 000427, loss = 0.001458
grad_step = 000428, loss = 0.001459
grad_step = 000429, loss = 0.001458
grad_step = 000430, loss = 0.001457
grad_step = 000431, loss = 0.001454
grad_step = 000432, loss = 0.001451
grad_step = 000433, loss = 0.001446
grad_step = 000434, loss = 0.001443
grad_step = 000435, loss = 0.001443
grad_step = 000436, loss = 0.001445
grad_step = 000437, loss = 0.001447
grad_step = 000438, loss = 0.001445
grad_step = 000439, loss = 0.001441
grad_step = 000440, loss = 0.001437
grad_step = 000441, loss = 0.001434
grad_step = 000442, loss = 0.001434
grad_step = 000443, loss = 0.001435
grad_step = 000444, loss = 0.001436
grad_step = 000445, loss = 0.001435
grad_step = 000446, loss = 0.001434
grad_step = 000447, loss = 0.001432
grad_step = 000448, loss = 0.001431
grad_step = 000449, loss = 0.001431
grad_step = 000450, loss = 0.001430
grad_step = 000451, loss = 0.001429
grad_step = 000452, loss = 0.001428
grad_step = 000453, loss = 0.001426
grad_step = 000454, loss = 0.001425
grad_step = 000455, loss = 0.001424
grad_step = 000456, loss = 0.001423
grad_step = 000457, loss = 0.001422
grad_step = 000458, loss = 0.001421
grad_step = 000459, loss = 0.001421
grad_step = 000460, loss = 0.001420
grad_step = 000461, loss = 0.001420
grad_step = 000462, loss = 0.001421
grad_step = 000463, loss = 0.001423
grad_step = 000464, loss = 0.001428
grad_step = 000465, loss = 0.001437
grad_step = 000466, loss = 0.001456
grad_step = 000467, loss = 0.001495
grad_step = 000468, loss = 0.001556
grad_step = 000469, loss = 0.001656
grad_step = 000470, loss = 0.001749
grad_step = 000471, loss = 0.001847
grad_step = 000472, loss = 0.001845
grad_step = 000473, loss = 0.001795
grad_step = 000474, loss = 0.001632
grad_step = 000475, loss = 0.001494
grad_step = 000476, loss = 0.001447
grad_step = 000477, loss = 0.001496
grad_step = 000478, loss = 0.001578
grad_step = 000479, loss = 0.001581
grad_step = 000480, loss = 0.001493
grad_step = 000481, loss = 0.001417
grad_step = 000482, loss = 0.001439
grad_step = 000483, loss = 0.001507
grad_step = 000484, loss = 0.001512
grad_step = 000485, loss = 0.001445
grad_step = 000486, loss = 0.001403
grad_step = 000487, loss = 0.001428
grad_step = 000488, loss = 0.001465
grad_step = 000489, loss = 0.001457
grad_step = 000490, loss = 0.001417
grad_step = 000491, loss = 0.001399
grad_step = 000492, loss = 0.001416
grad_step = 000493, loss = 0.001433
grad_step = 000494, loss = 0.001425
grad_step = 000495, loss = 0.001403
grad_step = 000496, loss = 0.001396
grad_step = 000497, loss = 0.001406
grad_step = 000498, loss = 0.001413
grad_step = 000499, loss = 0.001407
grad_step = 000500, loss = 0.001394
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001389
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

  date_run                              2020-05-12 22:13:03.480258
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.263808
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 22:13:03.486437
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.150079
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 22:13:03.493715
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.164691
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 22:13:03.499112
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.28051
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
0   2020-05-12 22:12:36.839897  ...    mean_absolute_error
1   2020-05-12 22:12:36.843187  ...     mean_squared_error
2   2020-05-12 22:12:36.845590  ...  median_absolute_error
3   2020-05-12 22:12:36.848461  ...               r2_score
4   2020-05-12 22:12:44.924552  ...    mean_absolute_error
5   2020-05-12 22:12:44.927666  ...     mean_squared_error
6   2020-05-12 22:12:44.930271  ...  median_absolute_error
7   2020-05-12 22:12:44.933299  ...               r2_score
8   2020-05-12 22:13:03.480258  ...    mean_absolute_error
9   2020-05-12 22:13:03.486437  ...     mean_squared_error
10  2020-05-12 22:13:03.493715  ...  median_absolute_error
11  2020-05-12 22:13:03.499112  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 316830.49it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 409431.84it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 566793.59it/s] 29%|██▉       | 2867200/9912422 [00:00<00:08, 797791.99it/s] 53%|█████▎    | 5267456/9912422 [00:00<00:04, 1120273.32it/s] 80%|███████▉  | 7905280/9912422 [00:00<00:01, 1565495.12it/s]9920512it [00:01, 9454985.76it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 135618.55it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 312198.06it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 404276.88it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 559566.98it/s]1654784it [00:00, 2803078.87it/s]                           
0it [00:00, ?it/s]8192it [00:00, 90714.95it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fff07cfd0> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb1a7ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb10ad0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb1a7ee80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb1005080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fae82f4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fae829be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb1a7ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fb0fc26a0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fae82f4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4fff086ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7d3fa18208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f01235a4e0412bb2a301b60f74f97c02cca1fc5b1613357775cf1f2a8c9ffdb3
  Stored in directory: /tmp/pip-ephem-wheel-cache-web_lr89/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7cd8702b00> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 1s
   24576/17464789 [..............................] - ETA: 50s
   57344/17464789 [..............................] - ETA: 43s
   90112/17464789 [..............................] - ETA: 41s
  180224/17464789 [..............................] - ETA: 27s
  368640/17464789 [..............................] - ETA: 16s
  770048/17464789 [>.............................] - ETA: 9s 
 1531904/17464789 [=>............................] - ETA: 5s
 3055616/17464789 [====>.........................] - ETA: 2s
 6119424/17464789 [=========>....................] - ETA: 1s
 9166848/17464789 [==============>...............] - ETA: 0s
12247040/17464789 [====================>.........] - ETA: 0s
15360000/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 22:14:32.439416: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 22:14:32.443789: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 22:14:32.443928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd5b407700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 22:14:32.443940: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2680 - accuracy: 0.5260
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4520 - accuracy: 0.5140 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6544 - accuracy: 0.5008
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5527 - accuracy: 0.5074
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5919 - accuracy: 0.5049
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5409 - accuracy: 0.5082
11000/25000 [============>.................] - ETA: 3s - loss: 7.5579 - accuracy: 0.5071
12000/25000 [=============>................] - ETA: 3s - loss: 7.5567 - accuracy: 0.5072
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5628 - accuracy: 0.5068
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5681 - accuracy: 0.5064
15000/25000 [=================>............] - ETA: 2s - loss: 7.5808 - accuracy: 0.5056
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5880 - accuracy: 0.5051
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6089 - accuracy: 0.5038
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6295 - accuracy: 0.5024
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6298 - accuracy: 0.5024
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6338 - accuracy: 0.5021
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6422 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6347 - accuracy: 0.5021
25000/25000 [==============================] - 7s 267us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 22:14:45.315608
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 22:14:45.315608  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<12:22:35, 19.4kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<8:40:42, 27.6kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.15M/862M [00:00<6:02:09, 39.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.2M/862M [00:00<4:11:28, 56.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:00<2:54:30, 80.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:00<2:00:57, 115kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:01<1:23:51, 164kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:01<58:18, 234kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:01<40:33, 333kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.4M/862M [00:02<29:52, 451kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:02<21:02, 639kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:02<15:22, 872kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<12:33:58, 17.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:04<8:48:15, 25.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:04<6:08:26, 36.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<17:05:05, 13.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:06<11:57:15, 18.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<8:22:17, 26.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:08<5:52:22, 37.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<4:07:30, 53.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:10<2:54:01, 75.8kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:10<2:01:25, 108kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<1:46:12, 124kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:12<1:15:17, 174kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<54:26, 240kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<39:10, 333kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<29:14, 444kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<21:27, 605kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<16:54, 764kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:18<12:50, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<10:53, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<08:35, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<07:55, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<06:32, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<06:28, 1.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<06:08, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:24<04:23, 2.89MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:24<04:20, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<13:34:46, 15.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<9:30:56, 22.1kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<6:39:39, 31.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<4:41:15, 44.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:28<3:15:59, 63.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<3:11:11, 65.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<2:14:50, 92.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:30<1:34:01, 132kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<2:07:42, 97.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<1:30:25, 137kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<1:04:48, 191kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<46:06, 268kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:34<32:14, 382kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<57:11, 215kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<40:57, 300kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<30:22, 403kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<22:09, 552kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<17:15, 705kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<13:01, 933kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<10:52, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<08:20, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<07:40, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:17, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<06:11, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:14, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<05:27, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<04:42, 2.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<05:04, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<04:28, 2.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<04:53, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<04:17, 2.73MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<04:46, 2.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<04:14, 2.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:23, 3.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<7:09:06, 27.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<4:59:41, 38.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<3:32:09, 54.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<2:29:49, 77.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:57<1:44:24, 110kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<2:15:01, 85.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<1:35:39, 120kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<1:08:15, 167kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<48:35, 235kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<35:32, 319kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<25:44, 441kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<19:37, 575kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<14:23, 784kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<11:47, 951kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<09:05, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<08:00, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<07:00, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<06:28, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:20, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<05:23, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:38, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:53, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:14, 2.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<04:34, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<03:55, 2.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<04:22, 2.47MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<03:53, 2.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<04:18, 2.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<03:43, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<04:13, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<03:47, 2.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:02, 3.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<6:30:53, 27.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<4:32:57, 38.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:24<3:10:47, 55.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<12:39:25, 13.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<8:52:43, 19.8kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<6:12:07, 28.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<4:21:37, 40.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<3:03:30, 56.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<2:09:45, 80.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<1:31:45, 113kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<1:05:30, 158kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:32<45:45, 225kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<37:19, 275kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<27:11, 378kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:34<19:00, 537kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<34:34, 295kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<25:13, 404kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<17:41, 574kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<17:15, 587kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<12:49, 789kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:38<09:01, 1.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<22:23, 449kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<16:27, 611kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:40<11:32, 866kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<21:40, 461kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<16:07, 619kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<11:21, 876kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<12:06, 820kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<09:19, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<06:35, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<16:03, 614kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<12:07, 813kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:46<08:33, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<09:58, 982kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<07:43, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<05:29, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<09:09, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<07:12, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:50<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<19:23, 498kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<14:22, 671kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:52<10:05, 950kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<21:18, 450kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<15:41, 610kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<11:18, 843kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<5:54:27, 26.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<4:07:45, 38.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<2:54:29, 54.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<2:02:49, 77.0kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:57<1:25:35, 110kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<1:09:24, 135kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<49:19, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [01:59<34:23, 271kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<49:58, 187kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<35:42, 261kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:01<24:55, 372kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<42:06, 220kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<30:12, 306kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<22:23, 410kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<16:20, 562kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:05<11:29, 795kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<12:54, 707kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<09:47, 931kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:07<06:52, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<1:41:09, 89.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<1:11:31, 126kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<51:03, 176kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<36:27, 246kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<26:40, 334kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<19:25, 459kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:13<13:36, 651kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<16:09, 547kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<12:03, 733kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<09:41, 905kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<07:30, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<06:31, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<05:16, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:19<03:44, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<4:38:46, 31.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<3:16:32, 43.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<2:17:16, 62.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<1:37:43, 87.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<1:09:24, 123kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:22<48:29, 176kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<35:21, 241kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<5:22:21, 26.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<3:45:28, 37.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<2:38:23, 53.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<1:52:04, 75.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<1:18:23, 107kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<56:27, 148kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<40:44, 206kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<28:33, 292kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<22:00, 378kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<16:40, 498kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<11:44, 704kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<10:34, 780kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<08:04, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:32<05:40, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<1:09:16, 118kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<49:08, 166kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:34<34:12, 237kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<56:18, 144kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<40:02, 202kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:36<27:53, 288kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<1:10:54, 113kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<50:16, 160kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<35:05, 228kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<27:55, 285kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<20:10, 395kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:40<14:06, 561kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<19:45, 400kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<14:29, 545kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<10:13, 769kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:42<07:26, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<8:19:41, 15.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<5:50:07, 22.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:44<4:03:33, 31.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<2:56:46, 43.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<2:04:19, 62.3kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:46<1:26:37, 88.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<1:03:59, 120kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<45:23, 169kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:48<31:38, 241kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<29:29, 258kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<21:44, 350kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<15:13, 497kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<13:26, 561kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<10:02, 750kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:52<07:03, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<10:04, 742kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<07:39, 975kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:54<05:23, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<11:47, 628kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<08:50, 836kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:56<06:12, 1.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<13:02, 563kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<09:44, 752kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<06:50, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<10:41, 679kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<07:59, 908kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<05:38, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<07:17, 987kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<05:43, 1.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:02<04:02, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<09:26, 755kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<07:12, 989kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:04<05:04, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<12:13, 577kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<09:08, 771kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:06<06:25, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<12:36, 554kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<09:48, 712kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<06:54, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<07:40, 902kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<05:58, 1.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<04:12, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<10:30, 653kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<07:54, 866kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:12<05:33, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<12:03, 563kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<09:01, 751kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<06:31, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<4:11:28, 26.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<2:55:47, 38.2kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<2:03:17, 54.1kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<1:26:46, 76.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:17<1:00:28, 109kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<44:39, 148kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<32:12, 205kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<22:30, 291kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<17:51, 365kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<13:02, 500kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<09:08, 709kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<09:07, 707kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<06:56, 930kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:23<04:52, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<08:31, 750kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<06:41, 955kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<04:44, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<05:26, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<04:18, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<03:05, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:18, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<03:33, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<02:32, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<04:13, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:28, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:31<02:28, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<07:28, 818kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<05:44, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:33<04:02, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<07:19, 825kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<05:37, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:35<03:57, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<08:18, 720kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<06:17, 949kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<04:27, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<05:02, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:58, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:51, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<03:40, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<03:11, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:18, 2.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<03:25, 1.69MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<03:10, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:18, 2.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<03:05, 1.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:58, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:44<02:09, 2.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:04, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<3:25:22, 27.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<2:23:39, 39.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:46<1:39:54, 56.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<1:12:54, 76.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<51:31, 108kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:48<35:55, 154kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<26:38, 207kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<19:04, 289kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<13:18, 410kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<11:49, 460kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<08:44, 623kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<06:08, 880kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<06:31, 824kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<05:05, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<03:36, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<04:23, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:31, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<02:31, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<03:18, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:45, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:58<01:58, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:35, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:57, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<02:06, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<03:48, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<03:03, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<02:11, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<03:36, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<02:08, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<03:01, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:27, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:46, 2.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<03:18, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:08<01:56, 2.49MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<03:54, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<03:08, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<02:13, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<03:45, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<03:15, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<02:19, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<02:57, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<02:23, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:13<01:43, 2.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<02:41, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<2:38:12, 29.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<1:50:17, 41.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<1:17:23, 59.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<54:25, 83.8kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:17<37:52, 120kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<27:50, 162kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<19:49, 227kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:19<13:46, 323kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<12:28, 355kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<09:05, 487kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:21<06:20, 690kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<06:43, 649kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<05:04, 859kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:23<03:33, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<04:57, 866kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<04:00, 1.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<02:50, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<03:30, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:47, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:27<01:59, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<03:11, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:29<01:50, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<03:19, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:53, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<02:03, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<02:44, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:15, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<01:36, 2.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<03:34, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<02:50, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<02:00, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<02:57, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<02:20, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:40, 2.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<02:32, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:38<02:10, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<01:33, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<01:51, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:19, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<02:36, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<02:07, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:42<01:30, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<02:15, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<2:13:10, 27.2kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<1:32:56, 38.8kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:44<1:04:21, 55.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<47:28, 74.9kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<33:29, 106kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<23:17, 151kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<17:08, 203kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<12:24, 281kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<08:39, 398kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<06:54, 494kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<05:04, 672kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<03:34, 946kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<03:27, 968kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:41, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<01:53, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:59, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:19, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<01:39, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:25, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:58, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<01:23, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:21, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:54, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:22, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<01:50, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:10, 2.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:42, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:26, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<01:01, 2.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<02:21, 1.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:58, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:24, 2.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:49, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:30, 1.90MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:04, 2.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<02:22, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<01:52, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<01:20, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:47, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<01:34, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:07, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:11<01:40, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<01:23, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:11<00:59, 2.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<01:37, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:30:20, 28.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:02:57, 41.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:13<43:27, 58.9kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<31:39, 80.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<22:21, 113kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<15:31, 162kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<11:20, 218kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<08:14, 300kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<05:44, 425kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<04:35, 523kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<03:26, 698kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<02:24, 981kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:19, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<01:16, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:52, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:34, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<01:07, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:22, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:17, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:55, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:11, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:08, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<00:48, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<01:11, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:02, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<00:44, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:09, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:05, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:46, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<01:04, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<00:44, 2.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:00, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:54, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:39, 2.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:59, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<00:35, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<01:21, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:08, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:48, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<01:04, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:53, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:37, 2.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<01:05, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:58, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:41, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:55, 1.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:51, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:36, 2.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:34, 2.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:48<00:32, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:33, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<49:11, 26.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<34:07, 38.2kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:50<23:01, 54.6kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<17:25, 71.7kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<12:16, 101kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:52<08:23, 144kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<06:02, 195kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<04:18, 273kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<02:56, 388kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<02:21, 472kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<01:43, 638kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<01:10, 901kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<01:14, 844kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:59, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<00:40, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:46, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:37, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:00<00:25, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:41, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:33, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:02<00:22, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:43, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:34, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<00:23, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:31, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:25, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:06<00:17, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:36, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:28, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:19, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:30, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:25, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:16, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:19, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:12<00:12, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:25, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:20, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:13, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:18, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:14, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:15, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:12, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<00:07, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:18, 939kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:14, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:19<00:08, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:09, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<08:16, 28.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<05:29, 40.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:21<02:59, 57.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<02:11, 75.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<01:29, 106kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:23<00:43, 151kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:30, 189kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:20, 264kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:08, 376kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:03, 445kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 604kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 836/400000 [00:00<00:47, 8353.24it/s]  0%|          | 1791/400000 [00:00<00:45, 8678.09it/s]  1%|          | 2719/400000 [00:00<00:44, 8847.66it/s]  1%|          | 3561/400000 [00:00<00:45, 8713.88it/s]  1%|          | 4458/400000 [00:00<00:45, 8788.66it/s]  1%|▏         | 5411/400000 [00:00<00:43, 8997.73it/s]  2%|▏         | 6316/400000 [00:00<00:43, 9011.20it/s]  2%|▏         | 7272/400000 [00:00<00:42, 9166.95it/s]  2%|▏         | 8250/400000 [00:00<00:41, 9341.14it/s]  2%|▏         | 9200/400000 [00:01<00:41, 9388.21it/s]  3%|▎         | 10162/400000 [00:01<00:41, 9454.80it/s]  3%|▎         | 11093/400000 [00:01<00:41, 9267.80it/s]  3%|▎         | 12053/400000 [00:01<00:41, 9362.06it/s]  3%|▎         | 13019/400000 [00:01<00:40, 9447.60it/s]  3%|▎         | 13976/400000 [00:01<00:40, 9482.61it/s]  4%|▎         | 14965/400000 [00:01<00:40, 9599.08it/s]  4%|▍         | 15927/400000 [00:01<00:39, 9605.05it/s]  4%|▍         | 16925/400000 [00:01<00:39, 9714.33it/s]  4%|▍         | 17914/400000 [00:01<00:39, 9764.34it/s]  5%|▍         | 18895/400000 [00:02<00:38, 9775.77it/s]  5%|▍         | 19873/400000 [00:02<00:39, 9609.35it/s]  5%|▌         | 20851/400000 [00:02<00:39, 9658.04it/s]  5%|▌         | 21855/400000 [00:02<00:38, 9767.29it/s]  6%|▌         | 22833/400000 [00:02<00:39, 9631.04it/s]  6%|▌         | 23797/400000 [00:02<00:39, 9517.93it/s]  6%|▌         | 24750/400000 [00:02<00:39, 9464.65it/s]  6%|▋         | 25732/400000 [00:02<00:39, 9568.05it/s]  7%|▋         | 26752/400000 [00:02<00:38, 9748.52it/s]  7%|▋         | 27735/400000 [00:02<00:38, 9771.51it/s]  7%|▋         | 28714/400000 [00:03<00:38, 9688.36it/s]  7%|▋         | 29744/400000 [00:03<00:37, 9863.14it/s]  8%|▊         | 30732/400000 [00:03<00:38, 9650.75it/s]  8%|▊         | 31700/400000 [00:03<00:38, 9626.21it/s]  8%|▊         | 32665/400000 [00:03<00:38, 9539.13it/s]  8%|▊         | 33641/400000 [00:03<00:38, 9601.28it/s]  9%|▊         | 34608/400000 [00:03<00:37, 9619.55it/s]  9%|▉         | 35571/400000 [00:03<00:38, 9456.87it/s]  9%|▉         | 36552/400000 [00:03<00:38, 9559.64it/s]  9%|▉         | 37509/400000 [00:03<00:38, 9359.45it/s] 10%|▉         | 38447/400000 [00:04<00:39, 9217.83it/s] 10%|▉         | 39378/400000 [00:04<00:39, 9242.93it/s] 10%|█         | 40373/400000 [00:04<00:38, 9443.39it/s] 10%|█         | 41410/400000 [00:04<00:36, 9702.18it/s] 11%|█         | 42416/400000 [00:04<00:36, 9806.30it/s] 11%|█         | 43423/400000 [00:04<00:36, 9882.75it/s] 11%|█         | 44414/400000 [00:04<00:36, 9718.53it/s] 11%|█▏        | 45388/400000 [00:04<00:36, 9656.56it/s] 12%|█▏        | 46357/400000 [00:04<00:36, 9666.09it/s] 12%|█▏        | 47355/400000 [00:04<00:36, 9756.82it/s] 12%|█▏        | 48332/400000 [00:05<00:36, 9758.99it/s] 12%|█▏        | 49309/400000 [00:05<00:37, 9296.31it/s] 13%|█▎        | 50244/400000 [00:05<00:37, 9300.42it/s] 13%|█▎        | 51178/400000 [00:05<00:37, 9266.40it/s] 13%|█▎        | 52134/400000 [00:05<00:37, 9351.84it/s] 13%|█▎        | 53097/400000 [00:05<00:36, 9431.79it/s] 14%|█▎        | 54042/400000 [00:05<00:36, 9435.61it/s] 14%|█▎        | 54987/400000 [00:05<00:36, 9359.22it/s] 14%|█▍        | 55995/400000 [00:05<00:35, 9562.97it/s] 14%|█▍        | 56992/400000 [00:05<00:35, 9678.78it/s] 14%|█▍        | 57996/400000 [00:06<00:34, 9783.17it/s] 15%|█▍        | 58982/400000 [00:06<00:34, 9805.61it/s] 15%|█▍        | 59964/400000 [00:06<00:35, 9668.07it/s] 15%|█▌        | 60932/400000 [00:06<00:35, 9654.61it/s] 15%|█▌        | 61899/400000 [00:06<00:35, 9615.13it/s] 16%|█▌        | 62862/400000 [00:06<00:35, 9535.21it/s] 16%|█▌        | 63827/400000 [00:06<00:35, 9567.84it/s] 16%|█▌        | 64785/400000 [00:06<00:35, 9540.54it/s] 16%|█▋        | 65773/400000 [00:06<00:34, 9638.30it/s] 17%|█▋        | 66738/400000 [00:07<00:35, 9516.06it/s] 17%|█▋        | 67691/400000 [00:07<00:36, 9027.06it/s] 17%|█▋        | 68661/400000 [00:07<00:35, 9217.66it/s] 17%|█▋        | 69632/400000 [00:07<00:35, 9358.70it/s] 18%|█▊        | 70632/400000 [00:07<00:34, 9540.89it/s] 18%|█▊        | 71590/400000 [00:07<00:34, 9548.93it/s] 18%|█▊        | 72548/400000 [00:07<00:34, 9412.69it/s] 18%|█▊        | 73492/400000 [00:07<00:34, 9408.68it/s] 19%|█▊        | 74435/400000 [00:07<00:34, 9400.69it/s] 19%|█▉        | 75418/400000 [00:07<00:34, 9524.21it/s] 19%|█▉        | 76428/400000 [00:08<00:33, 9688.39it/s] 19%|█▉        | 77399/400000 [00:08<00:33, 9625.00it/s] 20%|█▉        | 78363/400000 [00:08<00:33, 9540.77it/s] 20%|█▉        | 79319/400000 [00:08<00:33, 9444.09it/s] 20%|██        | 80265/400000 [00:08<00:33, 9438.56it/s] 20%|██        | 81254/400000 [00:08<00:33, 9568.43it/s] 21%|██        | 82279/400000 [00:08<00:32, 9762.27it/s] 21%|██        | 83270/400000 [00:08<00:32, 9805.85it/s] 21%|██        | 84252/400000 [00:08<00:32, 9803.41it/s] 21%|██▏       | 85234/400000 [00:08<00:32, 9737.20it/s] 22%|██▏       | 86233/400000 [00:09<00:31, 9809.42it/s] 22%|██▏       | 87224/400000 [00:09<00:31, 9837.71it/s] 22%|██▏       | 88209/400000 [00:09<00:31, 9808.66it/s] 22%|██▏       | 89198/400000 [00:09<00:31, 9830.64it/s] 23%|██▎       | 90182/400000 [00:09<00:31, 9779.28it/s] 23%|██▎       | 91162/400000 [00:09<00:31, 9784.52it/s] 23%|██▎       | 92142/400000 [00:09<00:31, 9787.34it/s] 23%|██▎       | 93121/400000 [00:09<00:31, 9657.47it/s] 24%|██▎       | 94088/400000 [00:09<00:32, 9365.87it/s] 24%|██▍       | 95027/400000 [00:09<00:33, 9221.03it/s] 24%|██▍       | 95969/400000 [00:10<00:32, 9279.08it/s] 24%|██▍       | 96918/400000 [00:10<00:32, 9339.61it/s] 24%|██▍       | 97911/400000 [00:10<00:31, 9508.91it/s] 25%|██▍       | 98879/400000 [00:10<00:31, 9559.23it/s] 25%|██▍       | 99862/400000 [00:10<00:31, 9637.96it/s] 25%|██▌       | 100827/400000 [00:10<00:31, 9514.50it/s] 25%|██▌       | 101844/400000 [00:10<00:30, 9699.77it/s] 26%|██▌       | 102842/400000 [00:10<00:30, 9781.11it/s] 26%|██▌       | 103822/400000 [00:10<00:30, 9555.44it/s] 26%|██▌       | 104780/400000 [00:10<00:30, 9543.66it/s] 26%|██▋       | 105736/400000 [00:11<00:30, 9543.89it/s] 27%|██▋       | 106692/400000 [00:11<00:30, 9544.98it/s] 27%|██▋       | 107663/400000 [00:11<00:30, 9592.34it/s] 27%|██▋       | 108644/400000 [00:11<00:30, 9654.40it/s] 27%|██▋       | 109610/400000 [00:11<00:30, 9632.50it/s] 28%|██▊       | 110574/400000 [00:11<00:30, 9613.37it/s] 28%|██▊       | 111536/400000 [00:11<00:30, 9451.55it/s] 28%|██▊       | 112495/400000 [00:11<00:30, 9491.91it/s] 28%|██▊       | 113445/400000 [00:11<00:30, 9484.51it/s] 29%|██▊       | 114422/400000 [00:11<00:29, 9566.65it/s] 29%|██▉       | 115391/400000 [00:12<00:29, 9598.34it/s] 29%|██▉       | 116359/400000 [00:12<00:29, 9621.90it/s] 29%|██▉       | 117399/400000 [00:12<00:28, 9841.33it/s] 30%|██▉       | 118385/400000 [00:12<00:28, 9745.15it/s] 30%|██▉       | 119361/400000 [00:12<00:28, 9715.69it/s] 30%|███       | 120334/400000 [00:12<00:29, 9591.97it/s] 30%|███       | 121297/400000 [00:12<00:29, 9597.12it/s] 31%|███       | 122277/400000 [00:12<00:28, 9653.89it/s] 31%|███       | 123243/400000 [00:12<00:28, 9645.32it/s] 31%|███       | 124223/400000 [00:12<00:28, 9691.01it/s] 31%|███▏      | 125193/400000 [00:13<00:30, 9103.93it/s] 32%|███▏      | 126130/400000 [00:13<00:29, 9181.86it/s] 32%|███▏      | 127083/400000 [00:13<00:29, 9282.28it/s] 32%|███▏      | 128016/400000 [00:13<00:29, 9250.68it/s] 32%|███▏      | 129003/400000 [00:13<00:28, 9427.10it/s] 32%|███▏      | 129949/400000 [00:13<00:28, 9421.67it/s] 33%|███▎      | 130951/400000 [00:13<00:28, 9592.09it/s] 33%|███▎      | 131971/400000 [00:13<00:27, 9765.89it/s] 33%|███▎      | 132970/400000 [00:13<00:27, 9830.65it/s] 33%|███▎      | 133955/400000 [00:14<00:27, 9658.39it/s] 34%|███▎      | 134923/400000 [00:14<00:27, 9625.17it/s] 34%|███▍      | 135890/400000 [00:14<00:27, 9638.31it/s] 34%|███▍      | 136922/400000 [00:14<00:26, 9830.62it/s] 34%|███▍      | 137907/400000 [00:14<00:26, 9781.61it/s] 35%|███▍      | 138899/400000 [00:14<00:26, 9821.01it/s] 35%|███▍      | 139882/400000 [00:14<00:26, 9746.73it/s] 35%|███▌      | 140858/400000 [00:14<00:26, 9646.55it/s] 35%|███▌      | 141824/400000 [00:14<00:26, 9578.37it/s] 36%|███▌      | 142783/400000 [00:14<00:26, 9575.72it/s] 36%|███▌      | 143746/400000 [00:15<00:26, 9589.27it/s] 36%|███▌      | 144706/400000 [00:15<00:26, 9531.63it/s] 36%|███▋      | 145745/400000 [00:15<00:26, 9773.09it/s] 37%|███▋      | 146725/400000 [00:15<00:25, 9775.73it/s] 37%|███▋      | 147704/400000 [00:15<00:25, 9727.69it/s] 37%|███▋      | 148678/400000 [00:15<00:25, 9701.65it/s] 37%|███▋      | 149684/400000 [00:15<00:25, 9805.77it/s] 38%|███▊      | 150666/400000 [00:15<00:26, 9554.02it/s] 38%|███▊      | 151642/400000 [00:15<00:25, 9613.71it/s] 38%|███▊      | 152605/400000 [00:15<00:26, 9420.46it/s] 38%|███▊      | 153549/400000 [00:16<00:26, 9373.50it/s] 39%|███▊      | 154488/400000 [00:16<00:26, 9216.33it/s] 39%|███▉      | 155460/400000 [00:16<00:26, 9361.57it/s] 39%|███▉      | 156486/400000 [00:16<00:25, 9612.05it/s] 39%|███▉      | 157469/400000 [00:16<00:25, 9674.64it/s] 40%|███▉      | 158439/400000 [00:16<00:25, 9642.80it/s] 40%|███▉      | 159405/400000 [00:16<00:24, 9637.87it/s] 40%|████      | 160374/400000 [00:16<00:24, 9651.79it/s] 40%|████      | 161387/400000 [00:16<00:24, 9790.36it/s] 41%|████      | 162368/400000 [00:16<00:24, 9676.48it/s] 41%|████      | 163376/400000 [00:17<00:24, 9791.32it/s] 41%|████      | 164357/400000 [00:17<00:24, 9688.56it/s] 41%|████▏     | 165327/400000 [00:17<00:24, 9689.37it/s] 42%|████▏     | 166345/400000 [00:17<00:23, 9830.62it/s] 42%|████▏     | 167330/400000 [00:17<00:23, 9731.77it/s] 42%|████▏     | 168305/400000 [00:17<00:24, 9527.29it/s] 42%|████▏     | 169305/400000 [00:17<00:23, 9663.26it/s] 43%|████▎     | 170286/400000 [00:17<00:23, 9703.91it/s] 43%|████▎     | 171258/400000 [00:17<00:23, 9594.61it/s] 43%|████▎     | 172219/400000 [00:17<00:24, 9431.46it/s] 43%|████▎     | 173207/400000 [00:18<00:23, 9559.18it/s] 44%|████▎     | 174179/400000 [00:18<00:23, 9606.21it/s] 44%|████▍     | 175162/400000 [00:18<00:23, 9670.99it/s] 44%|████▍     | 176130/400000 [00:18<00:23, 9572.94it/s] 44%|████▍     | 177091/400000 [00:18<00:23, 9583.96it/s] 45%|████▍     | 178072/400000 [00:18<00:22, 9650.56it/s] 45%|████▍     | 179063/400000 [00:18<00:22, 9725.49it/s] 45%|████▌     | 180037/400000 [00:18<00:22, 9706.78it/s] 45%|████▌     | 181053/400000 [00:18<00:22, 9836.69it/s] 46%|████▌     | 182038/400000 [00:19<00:22, 9764.25it/s] 46%|████▌     | 183018/400000 [00:19<00:22, 9774.78it/s] 46%|████▌     | 183996/400000 [00:19<00:23, 9341.74it/s] 46%|████▌     | 184935/400000 [00:19<00:23, 9251.15it/s] 46%|████▋     | 185935/400000 [00:19<00:22, 9462.31it/s] 47%|████▋     | 186885/400000 [00:19<00:22, 9448.11it/s] 47%|████▋     | 187871/400000 [00:19<00:22, 9567.65it/s] 47%|████▋     | 188830/400000 [00:19<00:22, 9478.95it/s] 47%|████▋     | 189824/400000 [00:19<00:21, 9611.21it/s] 48%|████▊     | 190787/400000 [00:19<00:21, 9523.46it/s] 48%|████▊     | 191741/400000 [00:20<00:22, 9317.17it/s] 48%|████▊     | 192675/400000 [00:20<00:22, 9261.89it/s] 48%|████▊     | 193631/400000 [00:20<00:22, 9339.68it/s] 49%|████▊     | 194626/400000 [00:20<00:21, 9514.54it/s] 49%|████▉     | 195580/400000 [00:20<00:21, 9495.89it/s] 49%|████▉     | 196531/400000 [00:20<00:21, 9484.42it/s] 49%|████▉     | 197481/400000 [00:20<00:21, 9443.70it/s] 50%|████▉     | 198426/400000 [00:20<00:21, 9435.76it/s] 50%|████▉     | 199408/400000 [00:20<00:21, 9545.69it/s] 50%|█████     | 200371/400000 [00:20<00:20, 9569.86it/s] 50%|█████     | 201329/400000 [00:21<00:21, 9286.20it/s] 51%|█████     | 202323/400000 [00:21<00:20, 9471.20it/s] 51%|█████     | 203281/400000 [00:21<00:20, 9502.08it/s] 51%|█████     | 204262/400000 [00:21<00:20, 9591.33it/s] 51%|█████▏    | 205223/400000 [00:21<00:20, 9594.18it/s] 52%|█████▏    | 206184/400000 [00:21<00:20, 9385.65it/s] 52%|█████▏    | 207180/400000 [00:21<00:20, 9549.79it/s] 52%|█████▏    | 208137/400000 [00:21<00:20, 9530.62it/s] 52%|█████▏    | 209092/400000 [00:21<00:20, 9465.79it/s] 53%|█████▎    | 210045/400000 [00:21<00:20, 9484.48it/s] 53%|█████▎    | 211027/400000 [00:22<00:19, 9581.57it/s] 53%|█████▎    | 211986/400000 [00:22<00:19, 9429.82it/s] 53%|█████▎    | 212931/400000 [00:22<00:19, 9405.96it/s] 53%|█████▎    | 213878/400000 [00:22<00:19, 9422.68it/s] 54%|█████▎    | 214864/400000 [00:22<00:19, 9548.97it/s] 54%|█████▍    | 215834/400000 [00:22<00:19, 9592.73it/s] 54%|█████▍    | 216794/400000 [00:22<00:19, 9312.49it/s] 54%|█████▍    | 217738/400000 [00:22<00:19, 9349.45it/s] 55%|█████▍    | 218722/400000 [00:22<00:19, 9491.23it/s] 55%|█████▍    | 219683/400000 [00:22<00:18, 9526.19it/s] 55%|█████▌    | 220637/400000 [00:23<00:18, 9475.67it/s] 55%|█████▌    | 221586/400000 [00:23<00:18, 9466.68it/s] 56%|█████▌    | 222544/400000 [00:23<00:18, 9497.85it/s] 56%|█████▌    | 223530/400000 [00:23<00:18, 9601.21it/s] 56%|█████▌    | 224516/400000 [00:23<00:18, 9674.25it/s] 56%|█████▋    | 225484/400000 [00:23<00:18, 9466.85it/s] 57%|█████▋    | 226533/400000 [00:23<00:17, 9750.44it/s] 57%|█████▋    | 227530/400000 [00:23<00:17, 9814.40it/s] 57%|█████▋    | 228514/400000 [00:23<00:17, 9766.33it/s] 57%|█████▋    | 229493/400000 [00:23<00:17, 9690.27it/s] 58%|█████▊    | 230487/400000 [00:24<00:17, 9762.14it/s] 58%|█████▊    | 231473/400000 [00:24<00:17, 9789.48it/s] 58%|█████▊    | 232453/400000 [00:24<00:17, 9650.79it/s] 58%|█████▊    | 233426/400000 [00:24<00:17, 9674.08it/s] 59%|█████▊    | 234395/400000 [00:24<00:17, 9603.07it/s] 59%|█████▉    | 235356/400000 [00:24<00:17, 9543.60it/s] 59%|█████▉    | 236331/400000 [00:24<00:17, 9603.80it/s] 59%|█████▉    | 237326/400000 [00:24<00:16, 9704.58it/s] 60%|█████▉    | 238298/400000 [00:24<00:16, 9652.79it/s] 60%|█████▉    | 239264/400000 [00:25<00:17, 9423.64it/s] 60%|██████    | 240208/400000 [00:25<00:17, 9369.62it/s] 60%|██████    | 241147/400000 [00:25<00:17, 9296.15it/s] 61%|██████    | 242080/400000 [00:25<00:16, 9305.87it/s] 61%|██████    | 243057/400000 [00:25<00:16, 9439.55it/s] 61%|██████    | 244002/400000 [00:25<00:16, 9383.34it/s] 61%|██████    | 244942/400000 [00:25<00:16, 9343.30it/s] 61%|██████▏   | 245877/400000 [00:25<00:16, 9327.27it/s] 62%|██████▏   | 246830/400000 [00:25<00:16, 9385.98it/s] 62%|██████▏   | 247806/400000 [00:25<00:16, 9493.72it/s] 62%|██████▏   | 248773/400000 [00:26<00:15, 9544.06it/s] 62%|██████▏   | 249728/400000 [00:26<00:16, 9375.92it/s] 63%|██████▎   | 250738/400000 [00:26<00:15, 9579.83it/s] 63%|██████▎   | 251698/400000 [00:26<00:15, 9580.42it/s] 63%|██████▎   | 252687/400000 [00:26<00:15, 9670.21it/s] 63%|██████▎   | 253656/400000 [00:26<00:15, 9652.03it/s] 64%|██████▎   | 254622/400000 [00:26<00:15, 9507.45it/s] 64%|██████▍   | 255706/400000 [00:26<00:14, 9869.92it/s] 64%|██████▍   | 256698/400000 [00:26<00:14, 9848.03it/s] 64%|██████▍   | 257697/400000 [00:26<00:14, 9887.75it/s] 65%|██████▍   | 258698/400000 [00:27<00:14, 9922.60it/s] 65%|██████▍   | 259692/400000 [00:27<00:14, 9665.67it/s] 65%|██████▌   | 260671/400000 [00:27<00:14, 9702.60it/s] 65%|██████▌   | 261675/400000 [00:27<00:14, 9799.57it/s] 66%|██████▌   | 262670/400000 [00:27<00:13, 9843.15it/s] 66%|██████▌   | 263656/400000 [00:27<00:13, 9823.22it/s] 66%|██████▌   | 264640/400000 [00:27<00:14, 9596.99it/s] 66%|██████▋   | 265622/400000 [00:27<00:13, 9661.70it/s] 67%|██████▋   | 266590/400000 [00:27<00:13, 9646.56it/s] 67%|██████▋   | 267586/400000 [00:27<00:13, 9737.51it/s] 67%|██████▋   | 268561/400000 [00:28<00:13, 9726.37it/s] 67%|██████▋   | 269535/400000 [00:28<00:13, 9363.01it/s] 68%|██████▊   | 270475/400000 [00:28<00:13, 9356.38it/s] 68%|██████▊   | 271431/400000 [00:28<00:13, 9414.20it/s] 68%|██████▊   | 272406/400000 [00:28<00:13, 9510.09it/s] 68%|██████▊   | 273366/400000 [00:28<00:13, 9534.37it/s] 69%|██████▊   | 274346/400000 [00:28<00:13, 9610.85it/s] 69%|██████▉   | 275321/400000 [00:28<00:12, 9651.79it/s] 69%|██████▉   | 276287/400000 [00:28<00:12, 9615.06it/s] 69%|██████▉   | 277249/400000 [00:28<00:12, 9483.21it/s] 70%|██████▉   | 278199/400000 [00:29<00:13, 9329.91it/s] 70%|██████▉   | 279134/400000 [00:29<00:13, 9163.41it/s] 70%|███████   | 280052/400000 [00:29<00:13, 9089.63it/s] 70%|███████   | 280979/400000 [00:29<00:13, 9141.55it/s] 70%|███████   | 281945/400000 [00:29<00:12, 9290.76it/s] 71%|███████   | 282966/400000 [00:29<00:12, 9547.37it/s] 71%|███████   | 283971/400000 [00:29<00:11, 9692.02it/s] 71%|███████   | 284958/400000 [00:29<00:11, 9744.18it/s] 71%|███████▏  | 285935/400000 [00:29<00:11, 9749.10it/s] 72%|███████▏  | 286912/400000 [00:29<00:11, 9748.46it/s] 72%|███████▏  | 287888/400000 [00:30<00:11, 9673.55it/s] 72%|███████▏  | 288857/400000 [00:30<00:11, 9424.44it/s] 72%|███████▏  | 289802/400000 [00:30<00:11, 9399.05it/s] 73%|███████▎  | 290769/400000 [00:30<00:11, 9477.32it/s] 73%|███████▎  | 291801/400000 [00:30<00:11, 9623.92it/s] 73%|███████▎  | 292765/400000 [00:30<00:11, 9519.34it/s] 73%|███████▎  | 293719/400000 [00:30<00:11, 9474.47it/s] 74%|███████▎  | 294720/400000 [00:30<00:10, 9628.50it/s] 74%|███████▍  | 295685/400000 [00:30<00:10, 9588.31it/s] 74%|███████▍  | 296645/400000 [00:31<00:10, 9538.93it/s] 74%|███████▍  | 297600/400000 [00:31<00:10, 9421.15it/s] 75%|███████▍  | 298543/400000 [00:31<00:10, 9352.62it/s] 75%|███████▍  | 299547/400000 [00:31<00:10, 9546.36it/s] 75%|███████▌  | 300518/400000 [00:31<00:10, 9590.55it/s] 75%|███████▌  | 301485/400000 [00:31<00:10, 9612.41it/s] 76%|███████▌  | 302458/400000 [00:31<00:10, 9644.89it/s] 76%|███████▌  | 303424/400000 [00:31<00:10, 9643.15it/s] 76%|███████▌  | 304405/400000 [00:31<00:09, 9691.28it/s] 76%|███████▋  | 305375/400000 [00:31<00:09, 9621.20it/s] 77%|███████▋  | 306367/400000 [00:32<00:09, 9706.28it/s] 77%|███████▋  | 307339/400000 [00:32<00:09, 9684.03it/s] 77%|███████▋  | 308308/400000 [00:32<00:09, 9607.72it/s] 77%|███████▋  | 309283/400000 [00:32<00:09, 9648.18it/s] 78%|███████▊  | 310288/400000 [00:32<00:09, 9765.09it/s] 78%|███████▊  | 311286/400000 [00:32<00:09, 9826.25it/s] 78%|███████▊  | 312308/400000 [00:32<00:08, 9938.67it/s] 78%|███████▊  | 313303/400000 [00:32<00:08, 9689.83it/s] 79%|███████▊  | 314274/400000 [00:32<00:08, 9530.75it/s] 79%|███████▉  | 315229/400000 [00:32<00:08, 9527.58it/s] 79%|███████▉  | 316212/400000 [00:33<00:08, 9614.20it/s] 79%|███████▉  | 317185/400000 [00:33<00:08, 9647.06it/s] 80%|███████▉  | 318170/400000 [00:33<00:08, 9704.24it/s] 80%|███████▉  | 319161/400000 [00:33<00:08, 9763.86it/s] 80%|████████  | 320138/400000 [00:33<00:08, 9654.12it/s] 80%|████████  | 321105/400000 [00:33<00:08, 9567.44it/s] 81%|████████  | 322107/400000 [00:33<00:08, 9698.75it/s] 81%|████████  | 323078/400000 [00:33<00:08, 9547.08it/s] 81%|████████  | 324063/400000 [00:33<00:07, 9632.68it/s] 81%|████████▏ | 325028/400000 [00:33<00:07, 9615.35it/s] 81%|████████▏ | 325992/400000 [00:34<00:07, 9621.56it/s] 82%|████████▏ | 326955/400000 [00:34<00:07, 9500.92it/s] 82%|████████▏ | 327906/400000 [00:34<00:07, 9227.83it/s] 82%|████████▏ | 328859/400000 [00:34<00:07, 9314.10it/s] 82%|████████▏ | 329818/400000 [00:34<00:07, 9394.76it/s] 83%|████████▎ | 330794/400000 [00:34<00:07, 9499.24it/s] 83%|████████▎ | 331760/400000 [00:34<00:07, 9546.49it/s] 83%|████████▎ | 332716/400000 [00:34<00:07, 9400.72it/s] 83%|████████▎ | 333719/400000 [00:34<00:06, 9578.62it/s] 84%|████████▎ | 334679/400000 [00:34<00:06, 9526.81it/s] 84%|████████▍ | 335652/400000 [00:35<00:06, 9584.96it/s] 84%|████████▍ | 336642/400000 [00:35<00:06, 9674.61it/s] 84%|████████▍ | 337611/400000 [00:35<00:06, 9666.53it/s] 85%|████████▍ | 338631/400000 [00:35<00:06, 9818.84it/s] 85%|████████▍ | 339614/400000 [00:35<00:06, 9747.21it/s] 85%|████████▌ | 340590/400000 [00:35<00:06, 9658.05it/s] 85%|████████▌ | 341610/400000 [00:35<00:05, 9813.96it/s] 86%|████████▌ | 342593/400000 [00:35<00:05, 9808.96it/s] 86%|████████▌ | 343595/400000 [00:35<00:05, 9868.40it/s] 86%|████████▌ | 344593/400000 [00:35<00:05, 9897.67it/s] 86%|████████▋ | 345590/400000 [00:36<00:05, 9916.65it/s] 87%|████████▋ | 346583/400000 [00:36<00:05, 9898.83it/s] 87%|████████▋ | 347574/400000 [00:36<00:05, 9791.74it/s] 87%|████████▋ | 348570/400000 [00:36<00:05, 9840.63it/s] 87%|████████▋ | 349555/400000 [00:36<00:05, 9788.40it/s] 88%|████████▊ | 350535/400000 [00:36<00:05, 9700.27it/s] 88%|████████▊ | 351529/400000 [00:36<00:04, 9768.04it/s] 88%|████████▊ | 352507/400000 [00:36<00:04, 9743.81it/s] 88%|████████▊ | 353482/400000 [00:36<00:04, 9678.49it/s] 89%|████████▊ | 354451/400000 [00:37<00:04, 9565.22it/s] 89%|████████▉ | 355449/400000 [00:37<00:04, 9683.54it/s] 89%|████████▉ | 356419/400000 [00:37<00:04, 9490.57it/s] 89%|████████▉ | 357370/400000 [00:37<00:04, 8913.49it/s] 90%|████████▉ | 358297/400000 [00:37<00:04, 9016.78it/s] 90%|████████▉ | 359223/400000 [00:37<00:04, 9086.30it/s] 90%|█████████ | 360136/400000 [00:37<00:04, 9077.61it/s] 90%|█████████ | 361065/400000 [00:37<00:04, 9139.78it/s] 90%|█████████ | 361982/400000 [00:37<00:04, 8854.89it/s] 91%|█████████ | 362890/400000 [00:37<00:04, 8920.34it/s] 91%|█████████ | 363799/400000 [00:38<00:04, 8970.48it/s] 91%|█████████ | 364702/400000 [00:38<00:03, 8987.72it/s] 91%|█████████▏| 365605/400000 [00:38<00:03, 8997.78it/s] 92%|█████████▏| 366522/400000 [00:38<00:03, 9047.03it/s] 92%|█████████▏| 367493/400000 [00:38<00:03, 9235.47it/s] 92%|█████████▏| 368419/400000 [00:38<00:03, 9200.96it/s] 92%|█████████▏| 369409/400000 [00:38<00:03, 9398.16it/s] 93%|█████████▎| 370457/400000 [00:38<00:03, 9698.11it/s] 93%|█████████▎| 371431/400000 [00:38<00:02, 9606.12it/s] 93%|█████████▎| 372395/400000 [00:38<00:02, 9568.93it/s] 93%|█████████▎| 373354/400000 [00:39<00:02, 9563.61it/s] 94%|█████████▎| 374319/400000 [00:39<00:02, 9586.21it/s] 94%|█████████▍| 375300/400000 [00:39<00:02, 9651.86it/s] 94%|█████████▍| 376266/400000 [00:39<00:02, 9452.44it/s] 94%|█████████▍| 377226/400000 [00:39<00:02, 9495.34it/s] 95%|█████████▍| 378177/400000 [00:39<00:02, 9380.98it/s] 95%|█████████▍| 379117/400000 [00:39<00:02, 9341.16it/s] 95%|█████████▌| 380068/400000 [00:39<00:02, 9390.78it/s] 95%|█████████▌| 381019/400000 [00:39<00:02, 9425.68it/s] 96%|█████████▌| 382021/400000 [00:39<00:01, 9594.72it/s] 96%|█████████▌| 382982/400000 [00:40<00:01, 9496.37it/s] 96%|█████████▌| 383933/400000 [00:40<00:01, 9446.65it/s] 96%|█████████▌| 384879/400000 [00:40<00:01, 9189.63it/s] 96%|█████████▋| 385801/400000 [00:40<00:01, 9127.80it/s] 97%|█████████▋| 386767/400000 [00:40<00:01, 9278.54it/s] 97%|█████████▋| 387697/400000 [00:40<00:01, 9199.93it/s] 97%|█████████▋| 388619/400000 [00:40<00:01, 9204.30it/s] 97%|█████████▋| 389637/400000 [00:40<00:01, 9474.36it/s] 98%|█████████▊| 390630/400000 [00:40<00:00, 9606.41it/s] 98%|█████████▊| 391613/400000 [00:40<00:00, 9669.64it/s] 98%|█████████▊| 392582/400000 [00:41<00:00, 9540.75it/s] 98%|█████████▊| 393538/400000 [00:41<00:00, 9418.15it/s] 99%|█████████▊| 394482/400000 [00:41<00:00, 9272.71it/s] 99%|█████████▉| 395411/400000 [00:41<00:00, 9181.77it/s] 99%|█████████▉| 396434/400000 [00:41<00:00, 9471.76it/s] 99%|█████████▉| 397385/400000 [00:41<00:00, 9470.60it/s]100%|█████████▉| 398375/400000 [00:41<00:00, 9593.14it/s]100%|█████████▉| 399353/400000 [00:41<00:00, 9646.48it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9550.72it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1fa827aa58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011291256022555634 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011891767731478382 	 Accuracy: 43

  model saves at 43% accuracy 

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
2020-05-12 22:23:43.324053: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 22:23:43.328420: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 22:23:43.328569: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557b24f41880 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 22:23:43.328583: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1f4dcfc978> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7688 - accuracy: 0.4933
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8238 - accuracy: 0.4897
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7985 - accuracy: 0.4914
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7499 - accuracy: 0.4946
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7382 - accuracy: 0.4953
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
12000/25000 [=============>................] - ETA: 3s - loss: 7.7637 - accuracy: 0.4937
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7551 - accuracy: 0.4942
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7619 - accuracy: 0.4938
15000/25000 [=================>............] - ETA: 2s - loss: 7.7453 - accuracy: 0.4949
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7356 - accuracy: 0.4955
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7289 - accuracy: 0.4959
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7228 - accuracy: 0.4963
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7150 - accuracy: 0.4968
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7103 - accuracy: 0.4972
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1f0cea0eb8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1f0b150e10> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3221 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.2631 - val_crf_viterbi_accuracy: 0.6533

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb1dc29ffd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 23:11:57.973913
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-12 23:11:57.977866
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-12 23:11:57.980860
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-12 23:11:57.984167
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb1e8069438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352046.9062
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 249207.5938
Epoch 3/10

1/1 [==============================] - 0s 90ms/step - loss: 154102.7656
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 87973.4141
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 50704.1836
Epoch 6/10

1/1 [==============================] - 0s 92ms/step - loss: 31582.0605
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 21028.6152
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 14844.7021
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 11032.1982
Epoch 10/10

1/1 [==============================] - 0s 106ms/step - loss: 8583.4463

  #### Inference Need return ypred, ytrue ######################### 
[[-2.4465926e-02  6.0272584e+00  6.8874998e+00  7.3459806e+00
   5.9709282e+00  6.7075105e+00  6.0365257e+00  6.5883956e+00
   5.7405157e+00  6.6013322e+00  7.3974829e+00  7.0564933e+00
   7.1760640e+00  6.2808123e+00  6.5981569e+00  6.9590249e+00
   6.5390058e+00  7.7160101e+00  7.6881180e+00  6.0559249e+00
   6.3604832e+00  6.4949589e+00  5.8460207e+00  6.2869177e+00
   6.5347614e+00  6.5651660e+00  6.4397535e+00  7.5755477e+00
   6.7556748e+00  7.1318836e+00  5.8667068e+00  4.9512353e+00
   5.5560365e+00  6.2910838e+00  6.6844640e+00  7.8352880e+00
   6.5208359e+00  7.0973744e+00  7.6369033e+00  5.5178924e+00
   7.7807436e+00  6.9997878e+00  6.7930121e+00  5.9512758e+00
   6.6725001e+00  6.5393801e+00  6.7709827e+00  7.5391669e+00
   6.9035711e+00  6.8719215e+00  6.6227055e+00  7.0266385e+00
   5.4106646e+00  5.6362429e+00  6.2591853e+00  6.2709856e+00
   5.9714618e+00  5.9913592e+00  4.7722654e+00  7.2275567e+00
   8.7591058e-01 -8.6996156e-01  7.7409410e-01  1.4822507e-01
  -9.9955553e-01  1.9776511e-01 -3.9498404e-01  3.5574991e-01
  -1.3367936e-02  6.3290101e-01  4.0166205e-01 -9.7239292e-01
  -1.4414332e+00  3.2983428e-01  3.6945263e-01  6.0734844e-01
   4.1350657e-01 -1.0075750e+00 -1.1016597e+00 -2.1673764e-01
   1.0527815e+00  6.4657956e-01  3.0994511e-01 -5.7743937e-01
   1.1052356e+00  5.4212850e-01 -4.1664243e-01 -9.2573559e-01
   3.3579260e-01 -1.4237380e-01 -8.2922310e-02 -8.2290125e-01
   1.8752016e+00  3.1376854e-03 -6.3821614e-01  6.3598973e-01
  -1.2809587e+00  1.4673270e+00  2.8242400e-01 -9.3911922e-01
   1.5074139e+00  7.7576768e-01  6.6813290e-01 -6.4528847e-01
   1.5669413e+00 -6.2195235e-01 -2.9931444e-01  2.1416950e-01
  -1.3680311e+00 -1.5149313e-01 -1.1178626e+00 -4.4420540e-01
  -2.3858690e-01 -3.5034364e-01  6.1767735e-02 -1.4374533e+00
   1.3051556e+00  2.1960733e+00 -2.6838434e-01 -2.0062728e-01
  -1.3890558e-01  1.1242636e+00  1.2181017e+00  1.1502297e+00
   2.0838307e-01 -1.7161369e-01 -1.2678573e+00  4.1488349e-02
   3.7184975e-01  1.0256699e+00  9.2253196e-01 -5.9531879e-01
   1.8738601e-01  9.4707721e-01  9.7654951e-01 -5.8533305e-01
   7.8474772e-01  1.5646176e+00  2.0251478e-01 -6.0325110e-01
   8.5807991e-01  1.1487136e+00  7.0701279e-02 -7.3328614e-02
  -6.8158865e-01  1.1187266e+00 -9.9482715e-01 -1.2139351e+00
  -9.5650494e-01 -3.4314516e-01 -1.6823310e+00  8.9773607e-01
   3.2470644e-02 -3.3088851e-01 -9.3738168e-01 -1.5206873e-02
   9.9784631e-01 -4.5099682e-01  7.7690774e-01 -1.4500706e+00
   8.3864015e-01  6.6067564e-01 -1.0662687e+00 -2.7399114e-01
   4.2625487e-01 -1.3760839e+00 -5.8044237e-01 -5.8068061e-01
  -1.5843759e+00  4.3355149e-01  6.4928514e-01 -7.1149099e-01
  -1.6574576e-01 -6.2649041e-01  4.4993544e-01  1.7108793e+00
  -1.0878819e+00  7.2342241e-01 -1.0103929e+00 -1.4837811e+00
   1.2831920e-01  8.4046669e+00  7.8927126e+00  7.8115993e+00
   8.2521420e+00  7.0303431e+00  8.1502857e+00  7.3336291e+00
   6.9364061e+00  7.8173580e+00  6.3914180e+00  7.7582622e+00
   7.8298192e+00  7.0742679e+00  7.8726444e+00  7.7817616e+00
   8.2504845e+00  7.5623326e+00  7.8120089e+00  5.4401145e+00
   7.1480546e+00  6.1535511e+00  6.3308601e+00  8.1182747e+00
   7.3237882e+00  6.8569703e+00  6.6751781e+00  6.6566787e+00
   8.6197491e+00  6.2780104e+00  7.1926374e+00  7.2526507e+00
   7.7000566e+00  6.8555093e+00  7.7408171e+00  6.8288560e+00
   7.8834977e+00  8.1736336e+00  6.9772468e+00  5.8990412e+00
   7.6751995e+00  7.1272855e+00  6.2829280e+00  8.8700056e+00
   6.8451719e+00  6.4048686e+00  5.8770027e+00  6.9074831e+00
   7.3360806e+00  7.3779416e+00  6.8384562e+00  7.0727367e+00
   7.3240390e+00  8.3879747e+00  6.4406586e+00  6.9951053e+00
   6.6810961e+00  9.1584549e+00  7.6193981e+00  7.4229412e+00
   3.1937861e-01  3.2170308e-01  2.5980716e+00  2.2515774e+00
   5.4294449e-01  9.8575115e-01  2.2962344e-01  4.2295700e-01
   2.0714450e+00  1.3299589e+00  1.1632560e+00  1.3026267e+00
   1.0378714e+00  1.4611920e+00  1.9058747e+00  2.5398102e+00
   7.8569233e-01  2.3499868e+00  3.8699543e-01  1.3708186e-01
   1.7403525e+00  1.3746209e+00  1.2223722e+00  1.0252396e+00
   7.9277587e-01  1.4324520e+00  1.8912506e+00  5.4693645e-01
   1.0257579e+00  5.7706171e-01  6.2639344e-01  1.5159764e+00
   1.4471095e+00  3.9496142e-01  9.1716039e-01  1.7239529e+00
   4.3390757e-01  9.2855322e-01  1.8051331e+00  5.2385938e-01
   1.2280021e+00  1.6615763e+00  1.4196134e+00  7.7757072e-01
   7.3702371e-01  9.8485303e-01  4.5456541e-01  5.9482944e-01
   7.6475227e-01  8.5256779e-01  3.8541812e-01  2.3077607e+00
   8.0076408e-01  2.8740573e-01  1.0373998e+00  2.6660492e+00
   1.2234555e+00  1.5603406e+00  1.3646593e+00  1.1821984e+00
   1.1918198e+00  2.2267038e-01  8.0821514e-01  6.1002570e-01
   1.8490994e+00  1.3606880e+00  1.0785108e+00  2.6657581e-01
   9.8859292e-01  6.5081620e-01  2.2813947e+00  1.7599157e+00
   7.1929240e-01  1.1413195e+00  6.8230844e-01  7.4862218e-01
   2.0716720e+00  2.4997339e+00  2.9062188e-01  1.2964053e+00
   4.1555429e-01  1.0759076e+00  8.2399142e-01  7.5457025e-01
   5.8135480e-01  2.1000783e+00  7.3053503e-01  6.1317325e-01
   2.0340824e-01  1.1260099e+00  2.0473347e+00  1.8750243e+00
   3.6511970e-01  8.8742870e-01  1.2237382e+00  6.0149592e-01
   1.0664194e+00  6.7073238e-01  4.6960866e-01  4.8798978e-01
   1.9939615e+00  1.3510222e+00  1.5745392e+00  1.8774232e+00
   1.7028915e+00  1.9906086e+00  8.5658485e-01  4.6293569e-01
   3.3627611e-01  1.3110975e+00  1.0234027e+00  6.2312502e-01
   8.4698617e-01  1.8537354e-01  5.6868768e-01  3.5448515e-01
   7.0007581e-01  4.5701694e-01  5.9739006e-01  4.1682911e-01
   3.8032663e+00 -8.8581209e+00 -6.3337965e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 23:12:06.464396
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.7946
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-12 23:12:06.468056
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9202.18
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-12 23:12:06.471259
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.5975
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-12 23:12:06.474263
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -823.121
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140401520580200
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140399008150192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140399008150696
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140399008151200
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140399008151704
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140399008152208

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb1ddf3ffd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.484897
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.462076
grad_step = 000002, loss = 0.447732
grad_step = 000003, loss = 0.432811
grad_step = 000004, loss = 0.415583
grad_step = 000005, loss = 0.399195
grad_step = 000006, loss = 0.383492
grad_step = 000007, loss = 0.367842
grad_step = 000008, loss = 0.352854
grad_step = 000009, loss = 0.337896
grad_step = 000010, loss = 0.322963
grad_step = 000011, loss = 0.310456
grad_step = 000012, loss = 0.300331
grad_step = 000013, loss = 0.291049
grad_step = 000014, loss = 0.281794
grad_step = 000015, loss = 0.272944
grad_step = 000016, loss = 0.264894
grad_step = 000017, loss = 0.256568
grad_step = 000018, loss = 0.247900
grad_step = 000019, loss = 0.239083
grad_step = 000020, loss = 0.230539
grad_step = 000021, loss = 0.222528
grad_step = 000022, loss = 0.214650
grad_step = 000023, loss = 0.206813
grad_step = 000024, loss = 0.199293
grad_step = 000025, loss = 0.192085
grad_step = 000026, loss = 0.184951
grad_step = 000027, loss = 0.178036
grad_step = 000028, loss = 0.171421
grad_step = 000029, loss = 0.164864
grad_step = 000030, loss = 0.158307
grad_step = 000031, loss = 0.151950
grad_step = 000032, loss = 0.145963
grad_step = 000033, loss = 0.140242
grad_step = 000034, loss = 0.134559
grad_step = 000035, loss = 0.129138
grad_step = 000036, loss = 0.124031
grad_step = 000037, loss = 0.118980
grad_step = 000038, loss = 0.113960
grad_step = 000039, loss = 0.109077
grad_step = 000040, loss = 0.104423
grad_step = 000041, loss = 0.099902
grad_step = 000042, loss = 0.095569
grad_step = 000043, loss = 0.091479
grad_step = 000044, loss = 0.087528
grad_step = 000045, loss = 0.083710
grad_step = 000046, loss = 0.080060
grad_step = 000047, loss = 0.076397
grad_step = 000048, loss = 0.072883
grad_step = 000049, loss = 0.069493
grad_step = 000050, loss = 0.066201
grad_step = 000051, loss = 0.063025
grad_step = 000052, loss = 0.059953
grad_step = 000053, loss = 0.056966
grad_step = 000054, loss = 0.053735
grad_step = 000055, loss = 0.050600
grad_step = 000056, loss = 0.047719
grad_step = 000057, loss = 0.045092
grad_step = 000058, loss = 0.042593
grad_step = 000059, loss = 0.040145
grad_step = 000060, loss = 0.037699
grad_step = 000061, loss = 0.035287
grad_step = 000062, loss = 0.032985
grad_step = 000063, loss = 0.030798
grad_step = 000064, loss = 0.028726
grad_step = 000065, loss = 0.026757
grad_step = 000066, loss = 0.024866
grad_step = 000067, loss = 0.023065
grad_step = 000068, loss = 0.021358
grad_step = 000069, loss = 0.019739
grad_step = 000070, loss = 0.018203
grad_step = 000071, loss = 0.016740
grad_step = 000072, loss = 0.015351
grad_step = 000073, loss = 0.014051
grad_step = 000074, loss = 0.012841
grad_step = 000075, loss = 0.011739
grad_step = 000076, loss = 0.010728
grad_step = 000077, loss = 0.009799
grad_step = 000078, loss = 0.008940
grad_step = 000079, loss = 0.008154
grad_step = 000080, loss = 0.007439
grad_step = 000081, loss = 0.006799
grad_step = 000082, loss = 0.006228
grad_step = 000083, loss = 0.005718
grad_step = 000084, loss = 0.005265
grad_step = 000085, loss = 0.004891
grad_step = 000086, loss = 0.004628
grad_step = 000087, loss = 0.004486
grad_step = 000088, loss = 0.004141
grad_step = 000089, loss = 0.003657
grad_step = 000090, loss = 0.003431
grad_step = 000091, loss = 0.003414
grad_step = 000092, loss = 0.003215
grad_step = 000093, loss = 0.002915
grad_step = 000094, loss = 0.002894
grad_step = 000095, loss = 0.002876
grad_step = 000096, loss = 0.002644
grad_step = 000097, loss = 0.002561
grad_step = 000098, loss = 0.002598
grad_step = 000099, loss = 0.002490
grad_step = 000100, loss = 0.002367
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002394
grad_step = 000102, loss = 0.002368
grad_step = 000103, loss = 0.002266
grad_step = 000104, loss = 0.002236
grad_step = 000105, loss = 0.002250
grad_step = 000106, loss = 0.002207
grad_step = 000107, loss = 0.002134
grad_step = 000108, loss = 0.002131
grad_step = 000109, loss = 0.002137
grad_step = 000110, loss = 0.002088
grad_step = 000111, loss = 0.002038
grad_step = 000112, loss = 0.002030
grad_step = 000113, loss = 0.002034
grad_step = 000114, loss = 0.002004
grad_step = 000115, loss = 0.001963
grad_step = 000116, loss = 0.001944
grad_step = 000117, loss = 0.001945
grad_step = 000118, loss = 0.001940
grad_step = 000119, loss = 0.001919
grad_step = 000120, loss = 0.001890
grad_step = 000121, loss = 0.001874
grad_step = 000122, loss = 0.001869
grad_step = 000123, loss = 0.001871
grad_step = 000124, loss = 0.001870
grad_step = 000125, loss = 0.001862
grad_step = 000126, loss = 0.001852
grad_step = 000127, loss = 0.001837
grad_step = 000128, loss = 0.001824
grad_step = 000129, loss = 0.001813
grad_step = 000130, loss = 0.001803
grad_step = 000131, loss = 0.001795
grad_step = 000132, loss = 0.001789
grad_step = 000133, loss = 0.001783
grad_step = 000134, loss = 0.001780
grad_step = 000135, loss = 0.001782
grad_step = 000136, loss = 0.001802
grad_step = 000137, loss = 0.001870
grad_step = 000138, loss = 0.002083
grad_step = 000139, loss = 0.002418
grad_step = 000140, loss = 0.002736
grad_step = 000141, loss = 0.002116
grad_step = 000142, loss = 0.001752
grad_step = 000143, loss = 0.002167
grad_step = 000144, loss = 0.002150
grad_step = 000145, loss = 0.001751
grad_step = 000146, loss = 0.001922
grad_step = 000147, loss = 0.002031
grad_step = 000148, loss = 0.001769
grad_step = 000149, loss = 0.001798
grad_step = 000150, loss = 0.001953
grad_step = 000151, loss = 0.001756
grad_step = 000152, loss = 0.001735
grad_step = 000153, loss = 0.001859
grad_step = 000154, loss = 0.001743
grad_step = 000155, loss = 0.001695
grad_step = 000156, loss = 0.001792
grad_step = 000157, loss = 0.001721
grad_step = 000158, loss = 0.001668
grad_step = 000159, loss = 0.001736
grad_step = 000160, loss = 0.001702
grad_step = 000161, loss = 0.001646
grad_step = 000162, loss = 0.001691
grad_step = 000163, loss = 0.001686
grad_step = 000164, loss = 0.001629
grad_step = 000165, loss = 0.001654
grad_step = 000166, loss = 0.001660
grad_step = 000167, loss = 0.001622
grad_step = 000168, loss = 0.001623
grad_step = 000169, loss = 0.001634
grad_step = 000170, loss = 0.001607
grad_step = 000171, loss = 0.001606
grad_step = 000172, loss = 0.001623
grad_step = 000173, loss = 0.001597
grad_step = 000174, loss = 0.001574
grad_step = 000175, loss = 0.001580
grad_step = 000176, loss = 0.001583
grad_step = 000177, loss = 0.001573
grad_step = 000178, loss = 0.001584
grad_step = 000179, loss = 0.001614
grad_step = 000180, loss = 0.001579
grad_step = 000181, loss = 0.001556
grad_step = 000182, loss = 0.001552
grad_step = 000183, loss = 0.001552
grad_step = 000184, loss = 0.001544
grad_step = 000185, loss = 0.001546
grad_step = 000186, loss = 0.001567
grad_step = 000187, loss = 0.001579
grad_step = 000188, loss = 0.001592
grad_step = 000189, loss = 0.001564
grad_step = 000190, loss = 0.001543
grad_step = 000191, loss = 0.001519
grad_step = 000192, loss = 0.001520
grad_step = 000193, loss = 0.001533
grad_step = 000194, loss = 0.001532
grad_step = 000195, loss = 0.001534
grad_step = 000196, loss = 0.001527
grad_step = 000197, loss = 0.001513
grad_step = 000198, loss = 0.001505
grad_step = 000199, loss = 0.001506
grad_step = 000200, loss = 0.001504
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001498
grad_step = 000202, loss = 0.001494
grad_step = 000203, loss = 0.001496
grad_step = 000204, loss = 0.001499
grad_step = 000205, loss = 0.001502
grad_step = 000206, loss = 0.001519
grad_step = 000207, loss = 0.001573
grad_step = 000208, loss = 0.001584
grad_step = 000209, loss = 0.001644
grad_step = 000210, loss = 0.001655
grad_step = 000211, loss = 0.001636
grad_step = 000212, loss = 0.001543
grad_step = 000213, loss = 0.001485
grad_step = 000214, loss = 0.001492
grad_step = 000215, loss = 0.001538
grad_step = 000216, loss = 0.001582
grad_step = 000217, loss = 0.001550
grad_step = 000218, loss = 0.001509
grad_step = 000219, loss = 0.001469
grad_step = 000220, loss = 0.001468
grad_step = 000221, loss = 0.001491
grad_step = 000222, loss = 0.001508
grad_step = 000223, loss = 0.001517
grad_step = 000224, loss = 0.001485
grad_step = 000225, loss = 0.001462
grad_step = 000226, loss = 0.001452
grad_step = 000227, loss = 0.001457
grad_step = 000228, loss = 0.001471
grad_step = 000229, loss = 0.001477
grad_step = 000230, loss = 0.001476
grad_step = 000231, loss = 0.001464
grad_step = 000232, loss = 0.001455
grad_step = 000233, loss = 0.001446
grad_step = 000234, loss = 0.001438
grad_step = 000235, loss = 0.001432
grad_step = 000236, loss = 0.001429
grad_step = 000237, loss = 0.001428
grad_step = 000238, loss = 0.001428
grad_step = 000239, loss = 0.001429
grad_step = 000240, loss = 0.001435
grad_step = 000241, loss = 0.001450
grad_step = 000242, loss = 0.001480
grad_step = 000243, loss = 0.001557
grad_step = 000244, loss = 0.001639
grad_step = 000245, loss = 0.001824
grad_step = 000246, loss = 0.001840
grad_step = 000247, loss = 0.001807
grad_step = 000248, loss = 0.001558
grad_step = 000249, loss = 0.001416
grad_step = 000250, loss = 0.001492
grad_step = 000251, loss = 0.001609
grad_step = 000252, loss = 0.001576
grad_step = 000253, loss = 0.001431
grad_step = 000254, loss = 0.001420
grad_step = 000255, loss = 0.001510
grad_step = 000256, loss = 0.001516
grad_step = 000257, loss = 0.001455
grad_step = 000258, loss = 0.001408
grad_step = 000259, loss = 0.001430
grad_step = 000260, loss = 0.001455
grad_step = 000261, loss = 0.001448
grad_step = 000262, loss = 0.001438
grad_step = 000263, loss = 0.001390
grad_step = 000264, loss = 0.001404
grad_step = 000265, loss = 0.001444
grad_step = 000266, loss = 0.001411
grad_step = 000267, loss = 0.001384
grad_step = 000268, loss = 0.001387
grad_step = 000269, loss = 0.001399
grad_step = 000270, loss = 0.001394
grad_step = 000271, loss = 0.001381
grad_step = 000272, loss = 0.001378
grad_step = 000273, loss = 0.001364
grad_step = 000274, loss = 0.001368
grad_step = 000275, loss = 0.001381
grad_step = 000276, loss = 0.001365
grad_step = 000277, loss = 0.001356
grad_step = 000278, loss = 0.001354
grad_step = 000279, loss = 0.001350
grad_step = 000280, loss = 0.001350
grad_step = 000281, loss = 0.001345
grad_step = 000282, loss = 0.001342
grad_step = 000283, loss = 0.001348
grad_step = 000284, loss = 0.001347
grad_step = 000285, loss = 0.001342
grad_step = 000286, loss = 0.001346
grad_step = 000287, loss = 0.001347
grad_step = 000288, loss = 0.001353
grad_step = 000289, loss = 0.001360
grad_step = 000290, loss = 0.001370
grad_step = 000291, loss = 0.001382
grad_step = 000292, loss = 0.001407
grad_step = 000293, loss = 0.001413
grad_step = 000294, loss = 0.001425
grad_step = 000295, loss = 0.001401
grad_step = 000296, loss = 0.001376
grad_step = 000297, loss = 0.001340
grad_step = 000298, loss = 0.001307
grad_step = 000299, loss = 0.001292
grad_step = 000300, loss = 0.001298
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001314
grad_step = 000302, loss = 0.001328
grad_step = 000303, loss = 0.001329
grad_step = 000304, loss = 0.001315
grad_step = 000305, loss = 0.001298
grad_step = 000306, loss = 0.001281
grad_step = 000307, loss = 0.001271
grad_step = 000308, loss = 0.001268
grad_step = 000309, loss = 0.001266
grad_step = 000310, loss = 0.001263
grad_step = 000311, loss = 0.001259
grad_step = 000312, loss = 0.001254
grad_step = 000313, loss = 0.001249
grad_step = 000314, loss = 0.001244
grad_step = 000315, loss = 0.001240
grad_step = 000316, loss = 0.001238
grad_step = 000317, loss = 0.001237
grad_step = 000318, loss = 0.001242
grad_step = 000319, loss = 0.001264
grad_step = 000320, loss = 0.001326
grad_step = 000321, loss = 0.001518
grad_step = 000322, loss = 0.001774
grad_step = 000323, loss = 0.002184
grad_step = 000324, loss = 0.001859
grad_step = 000325, loss = 0.001644
grad_step = 000326, loss = 0.001488
grad_step = 000327, loss = 0.001342
grad_step = 000328, loss = 0.001571
grad_step = 000329, loss = 0.001629
grad_step = 000330, loss = 0.001278
grad_step = 000331, loss = 0.001403
grad_step = 000332, loss = 0.001490
grad_step = 000333, loss = 0.001361
grad_step = 000334, loss = 0.001366
grad_step = 000335, loss = 0.001306
grad_step = 000336, loss = 0.001333
grad_step = 000337, loss = 0.001381
grad_step = 000338, loss = 0.001291
grad_step = 000339, loss = 0.001235
grad_step = 000340, loss = 0.001344
grad_step = 000341, loss = 0.001303
grad_step = 000342, loss = 0.001254
grad_step = 000343, loss = 0.001272
grad_step = 000344, loss = 0.001218
grad_step = 000345, loss = 0.001296
grad_step = 000346, loss = 0.001229
grad_step = 000347, loss = 0.001224
grad_step = 000348, loss = 0.001217
grad_step = 000349, loss = 0.001216
grad_step = 000350, loss = 0.001243
grad_step = 000351, loss = 0.001204
grad_step = 000352, loss = 0.001199
grad_step = 000353, loss = 0.001185
grad_step = 000354, loss = 0.001198
grad_step = 000355, loss = 0.001196
grad_step = 000356, loss = 0.001192
grad_step = 000357, loss = 0.001177
grad_step = 000358, loss = 0.001170
grad_step = 000359, loss = 0.001173
grad_step = 000360, loss = 0.001164
grad_step = 000361, loss = 0.001179
grad_step = 000362, loss = 0.001169
grad_step = 000363, loss = 0.001168
grad_step = 000364, loss = 0.001165
grad_step = 000365, loss = 0.001154
grad_step = 000366, loss = 0.001153
grad_step = 000367, loss = 0.001148
grad_step = 000368, loss = 0.001147
grad_step = 000369, loss = 0.001144
grad_step = 000370, loss = 0.001146
grad_step = 000371, loss = 0.001144
grad_step = 000372, loss = 0.001142
grad_step = 000373, loss = 0.001143
grad_step = 000374, loss = 0.001139
grad_step = 000375, loss = 0.001138
grad_step = 000376, loss = 0.001136
grad_step = 000377, loss = 0.001133
grad_step = 000378, loss = 0.001131
grad_step = 000379, loss = 0.001129
grad_step = 000380, loss = 0.001128
grad_step = 000381, loss = 0.001124
grad_step = 000382, loss = 0.001123
grad_step = 000383, loss = 0.001121
grad_step = 000384, loss = 0.001120
grad_step = 000385, loss = 0.001119
grad_step = 000386, loss = 0.001120
grad_step = 000387, loss = 0.001122
grad_step = 000388, loss = 0.001127
grad_step = 000389, loss = 0.001135
grad_step = 000390, loss = 0.001153
grad_step = 000391, loss = 0.001177
grad_step = 000392, loss = 0.001221
grad_step = 000393, loss = 0.001270
grad_step = 000394, loss = 0.001340
grad_step = 000395, loss = 0.001356
grad_step = 000396, loss = 0.001346
grad_step = 000397, loss = 0.001243
grad_step = 000398, loss = 0.001139
grad_step = 000399, loss = 0.001091
grad_step = 000400, loss = 0.001127
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001186
grad_step = 000402, loss = 0.001185
grad_step = 000403, loss = 0.001133
grad_step = 000404, loss = 0.001086
grad_step = 000405, loss = 0.001098
grad_step = 000406, loss = 0.001138
grad_step = 000407, loss = 0.001148
grad_step = 000408, loss = 0.001119
grad_step = 000409, loss = 0.001082
grad_step = 000410, loss = 0.001076
grad_step = 000411, loss = 0.001095
grad_step = 000412, loss = 0.001116
grad_step = 000413, loss = 0.001116
grad_step = 000414, loss = 0.001094
grad_step = 000415, loss = 0.001072
grad_step = 000416, loss = 0.001061
grad_step = 000417, loss = 0.001065
grad_step = 000418, loss = 0.001076
grad_step = 000419, loss = 0.001086
grad_step = 000420, loss = 0.001094
grad_step = 000421, loss = 0.001094
grad_step = 000422, loss = 0.001091
grad_step = 000423, loss = 0.001081
grad_step = 000424, loss = 0.001070
grad_step = 000425, loss = 0.001060
grad_step = 000426, loss = 0.001052
grad_step = 000427, loss = 0.001046
grad_step = 000428, loss = 0.001042
grad_step = 000429, loss = 0.001039
grad_step = 000430, loss = 0.001039
grad_step = 000431, loss = 0.001039
grad_step = 000432, loss = 0.001041
grad_step = 000433, loss = 0.001044
grad_step = 000434, loss = 0.001049
grad_step = 000435, loss = 0.001057
grad_step = 000436, loss = 0.001073
grad_step = 000437, loss = 0.001101
grad_step = 000438, loss = 0.001144
grad_step = 000439, loss = 0.001211
grad_step = 000440, loss = 0.001264
grad_step = 000441, loss = 0.001347
grad_step = 000442, loss = 0.001325
grad_step = 000443, loss = 0.001255
grad_step = 000444, loss = 0.001100
grad_step = 000445, loss = 0.001023
grad_step = 000446, loss = 0.001058
grad_step = 000447, loss = 0.001129
grad_step = 000448, loss = 0.001140
grad_step = 000449, loss = 0.001066
grad_step = 000450, loss = 0.001016
grad_step = 000451, loss = 0.001035
grad_step = 000452, loss = 0.001079
grad_step = 000453, loss = 0.001082
grad_step = 000454, loss = 0.001037
grad_step = 000455, loss = 0.001007
grad_step = 000456, loss = 0.001016
grad_step = 000457, loss = 0.001037
grad_step = 000458, loss = 0.001042
grad_step = 000459, loss = 0.001025
grad_step = 000460, loss = 0.001012
grad_step = 000461, loss = 0.001015
grad_step = 000462, loss = 0.001010
grad_step = 000463, loss = 0.001002
grad_step = 000464, loss = 0.001001
grad_step = 000465, loss = 0.001011
grad_step = 000466, loss = 0.001017
grad_step = 000467, loss = 0.001003
grad_step = 000468, loss = 0.000987
grad_step = 000469, loss = 0.000990
grad_step = 000470, loss = 0.000996
grad_step = 000471, loss = 0.000994
grad_step = 000472, loss = 0.000988
grad_step = 000473, loss = 0.000989
grad_step = 000474, loss = 0.001000
grad_step = 000475, loss = 0.000998
grad_step = 000476, loss = 0.000991
grad_step = 000477, loss = 0.000990
grad_step = 000478, loss = 0.000993
grad_step = 000479, loss = 0.000999
grad_step = 000480, loss = 0.000999
grad_step = 000481, loss = 0.000996
grad_step = 000482, loss = 0.001004
grad_step = 000483, loss = 0.001010
grad_step = 000484, loss = 0.001018
grad_step = 000485, loss = 0.001017
grad_step = 000486, loss = 0.001020
grad_step = 000487, loss = 0.001019
grad_step = 000488, loss = 0.001015
grad_step = 000489, loss = 0.000998
grad_step = 000490, loss = 0.000981
grad_step = 000491, loss = 0.000967
grad_step = 000492, loss = 0.000956
grad_step = 000493, loss = 0.000949
grad_step = 000494, loss = 0.000947
grad_step = 000495, loss = 0.000951
grad_step = 000496, loss = 0.000958
grad_step = 000497, loss = 0.000964
grad_step = 000498, loss = 0.000965
grad_step = 000499, loss = 0.000968
grad_step = 000500, loss = 0.000971
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000977
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

  date_run                              2020-05-12 23:12:24.940484
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.251109
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-12 23:12:24.946671
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.178413
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-12 23:12:24.952798
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.125744
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-12 23:12:24.958723
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.71104
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
0   2020-05-12 23:11:57.973913  ...    mean_absolute_error
1   2020-05-12 23:11:57.977866  ...     mean_squared_error
2   2020-05-12 23:11:57.980860  ...  median_absolute_error
3   2020-05-12 23:11:57.984167  ...               r2_score
4   2020-05-12 23:12:06.464396  ...    mean_absolute_error
5   2020-05-12 23:12:06.468056  ...     mean_squared_error
6   2020-05-12 23:12:06.471259  ...  median_absolute_error
7   2020-05-12 23:12:06.474263  ...               r2_score
8   2020-05-12 23:12:24.940484  ...    mean_absolute_error
9   2020-05-12 23:12:24.946671  ...     mean_squared_error
10  2020-05-12 23:12:24.952798  ...  median_absolute_error
11  2020-05-12 23:12:24.958723  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|███▉      | 3956736/9912422 [00:00<00:00, 38040795.74it/s]9920512it [00:00, 35226526.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 597371.07it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160207.29it/s]1654784it [00:00, 11578141.02it/s]                         
0it [00:00, ?it/s]8192it [00:00, 214460.18it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9df441cc88> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da6dd6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da64060b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da6dd6e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da635d048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da3b87470> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da3b816a0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da6dd6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da631a668> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9da6406080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9df43dfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f53b3098208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=87d27aaa2fe0aaba651f05bd9ff64c5fc9e2462b1163d7f5681a3bceba9a49ed
  Stored in directory: /tmp/pip-ephem-wheel-cache-el3kr1d3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f534ae93710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4022272/17464789 [=====>........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 23:13:51.307993: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 23:13:51.312252: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 23:13:51.312375: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fac6ce5f40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 23:13:51.312386: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4750 - accuracy: 0.5125 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6685 - accuracy: 0.4999
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6896 - accuracy: 0.4985
11000/25000 [============>.................] - ETA: 3s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 3s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6715 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6636 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6425 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6464 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-12 23:14:04.343941
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-12 23:14:04.343941  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<26:45:34, 8.95kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:57:57, 12.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<13:19:46, 18.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<9:20:15, 25.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.48M/862M [00:01<6:31:13, 36.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.58M/862M [00:01<4:32:00, 52.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<3:09:14, 74.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<2:11:40, 106kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<1:31:39, 152kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:02<1:03:50, 217kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<44:37, 309kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:02<31:08, 440kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<21:48, 625kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.6M/862M [00:02<15:15, 888kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<11:21, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<09:49, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<09:15, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<06:58, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:05<05:02, 2.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<13:17, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<10:58, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<08:02, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<08:16, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<07:15, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<05:24, 2.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:37, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<07:26, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<05:50, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:11<04:13, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<16:30, 796kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<14:21, 916kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<10:43, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<07:37, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<1:29:01, 147kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<1:03:38, 205kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<44:45, 291kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<34:17, 379kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<25:19, 514kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<18:01, 720kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<15:38, 827kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<13:34, 953kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<10:09, 1.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<09:10, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<07:44, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<05:44, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<07:00, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:23<06:13, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<04:40, 2.74MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<06:16, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:25<05:41, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:25<04:15, 2.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:00, 2.11MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:46, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:20, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<03:53, 3.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<1:33:29, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<1:06:43, 189kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<46:56, 268kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<35:42, 351kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<27:41, 452kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<20:00, 625kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<14:05, 885kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<54:56, 227kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<39:44, 313kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<28:04, 443kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<22:30, 551kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<18:17, 678kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<13:25, 922kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:21, 1.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<09:11, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<06:42, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:34, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:31, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:51, 2.51MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:16, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:53, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:22, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:53, 3.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:02, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:57, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:34, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:25, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:25, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<04:47, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:09, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:30, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:08, 2.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:43, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:25, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:06, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:28, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:49, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:46, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:58, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:46, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:08, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:48, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:32, 3.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:38, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:58, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:30, 2.56MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:35, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:35, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:09, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:50, 2.99MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:21, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:04, 2.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:16, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:12, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<03:38, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:41, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:17, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:57, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:10, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:13, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:53, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<03:36, 3.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:06, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:32, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:09, 2.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<04:57, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<03:44, 2.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<04:58, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:44, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:37, 3.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<04:52, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:50, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:42, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:25, 3.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:35, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:14, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:22, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:03, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:39, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:17, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:48, 2.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<12:51, 835kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<10:13, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<07:26, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:28, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:46, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:57, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:24, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:39, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:12, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:56, 2.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:59, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:53, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:38, 2.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:23, 3.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:00, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:05, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:32, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:22, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:06, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:51, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:31, 2.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<11:56, 864kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<09:31, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:54, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<1:04:57, 158kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<47:51, 214kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<34:00, 301kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<23:52, 427kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<20:45, 490kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<15:41, 648kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<11:12, 906kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<07:58, 1.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<18:29, 547kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<15:12, 665kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<11:12, 901kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<07:57, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<15:27, 650kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<11:56, 840kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<08:37, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<08:09, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:50, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:02, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:37, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:56, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:34, 2.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<12:10, 807kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<09:38, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:00, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<06:59, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:03, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:00, 2.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<05:07, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:42, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<03:31, 2.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<04:30, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:17, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<04:15, 2.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:05, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<10:53, 878kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<08:40, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<06:18, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<06:27, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<06:37, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:09, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:43, 2.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<14:00, 672kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:00<10:50, 868kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:48, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<07:28, 1.25MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<07:18, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:33, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:01, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<05:59, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:12, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:53, 2.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:44, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:16, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:07, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:00, 3.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<05:43, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<05:02, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:47, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<04:34, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<05:13, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:06, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:58, 3.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<06:24, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<05:30, 1.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:03, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<04:45, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<05:18, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:02, 2.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<05:57, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:06, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:35, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:17, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:07, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:03, 2.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:26, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:05, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<03:04, 2.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:00, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:43, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:42, 3.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:15, 1.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<05:11, 1.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:52, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:31, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:04, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:04, 2.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:02, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:42, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:46, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:45, 3.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<10:09, 825kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<08:03, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:49, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:50, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<05:55, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:19, 2.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:53, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:18, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:13, 2.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:35, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:36, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:53, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:42, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:11, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:41, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:08, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:03, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:56, 2.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:55, 801kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:49, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:38, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:40, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:23, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:11, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:05, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:19, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:58, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<02:41, 2.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:37, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:24, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:29, 3.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<09:13, 825kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:19, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:18, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:19, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:31, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:02, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:35, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:36, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:14, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<05:12, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:50, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:15, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:48, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:50, 2.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:32, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:07, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:17, 2.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:23, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<08:45, 822kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:53, 1.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:59, 1.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:04, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:18, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:11, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:49, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:33, 2.75MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:21, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:03, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:12, 3.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<22:18, 310kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<16:24, 421kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<11:38, 592kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:34, 716kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:25, 923kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:19, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:14, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:14, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:00, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:51, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<07:44, 867kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:10, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<04:29, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:33, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:39, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:36, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:35, 2.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<07:56, 828kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:17, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:34, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:34, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:53, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:53, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:27, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:51, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:00, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:11, 2.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:14, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:42, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:44, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:15, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:55, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:07, 2.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:09, 870kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:42, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:08, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:12, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:22, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:24, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:26, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<07:16, 835kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:47, 1.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:11, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:12, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:13, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:14, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:20, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<12:19, 483kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<09:16, 641kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:37, 893kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:54, 995kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:24, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:05, 1.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<02:54, 2.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<18:46, 309kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<13:45, 422kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<09:44, 593kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<08:03, 713kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<06:50, 837kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:05, 1.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<03:35, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<14:37, 388kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<10:52, 521kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:44, 729kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<06:34, 851kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<05:54, 948kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:24, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:08, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<04:35, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:49, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:48, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:10, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:04, 2.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:39, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:07, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:28, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:47, 2.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<05:58, 892kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:44, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:26, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:34, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:39, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:51, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:03, 2.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<06:32, 792kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<05:10, 1.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:45, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:42, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:40, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:48, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:01, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<04:22, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:38, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:39, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:54, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:08, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:26, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:45, 2.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:32, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:44, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:55, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:08, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:28, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:46, 2.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<05:56, 803kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:41, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:24, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:22, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:28, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:54, 2.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:34, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:00, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:44, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:07, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<02:35, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:18, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:42, 2.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:09, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:29, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:57, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:25, 3.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:09, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:36, 2.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:01, 2.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:24, 1.77MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:53, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:22, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:35, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:17, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:42, 2.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:04, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:24, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<01:53, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:23, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:06, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:56, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:27, 2.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:42, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:14, 3.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<12:40, 310kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<09:18, 422kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<06:35, 593kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:23, 717kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:40, 828kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:27, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:28, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:51, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:25, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:02, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:12, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:43, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:14, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:53, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:25, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:46, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:01, 1.77MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:49, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:21, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:42, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:35, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:12, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:34, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:54, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:30, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:05, 3.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:32, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:08, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:34, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:50, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:01, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:35, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:08, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:59, 812kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:08, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:15, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:16, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:19, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:47, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:16, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:31, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:05, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:32, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:44, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:53, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:03, 2.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:08, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:21, 2.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:33, 1.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:21, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:57, 2.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<06:29, 435kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:50, 582kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<03:26, 814kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:59, 924kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:40, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:59, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:25, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:08, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:47, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:18, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:28, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:38, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:16, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:55, 2.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:58, 857kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:20, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:41, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:43, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:45, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:21, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:57, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:04, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:43, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:15, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:22, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:28, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:10, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:49, 2.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:49, 806kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:12, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:35, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:35, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:37, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:14, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:52, 2.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:48, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:30, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:05, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:11, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:16, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:58, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:42, 2.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:07, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:44, 2.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:56, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:05, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:37, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:01, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:56, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:41, 2.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:51, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:58, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:32, 3.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:14, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:03, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:46, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:53, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:59, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:46, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:33, 2.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:48, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:44, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:33, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<00:42, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:48, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:28, 3.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:39, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:27, 3.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:36, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:43, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:33, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:24, 3.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:40, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:36, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:27, 2.83MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:34, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:40, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:31, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:22, 3.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:59, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:49, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:38, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:42, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:23, 2.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:22, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:28, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:32, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:17, 3.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:51, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:41, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:29, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:30, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:17, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:56, 798kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:43, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:30, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:20, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:46, 883kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:40, 993kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:29, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:19, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:07, 542kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:50, 713kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:35, 991kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:30, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:28, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:18, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:12, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 3.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:03, 3.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:16, 701kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:12, 904kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.73MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.89MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 862/400000 [00:00<00:46, 8611.94it/s]  0%|          | 1743/400000 [00:00<00:45, 8670.10it/s]  1%|          | 2655/400000 [00:00<00:45, 8797.69it/s]  1%|          | 3614/400000 [00:00<00:43, 9018.42it/s]  1%|          | 4519/400000 [00:00<00:43, 9027.52it/s]  1%|▏         | 5448/400000 [00:00<00:43, 9104.07it/s]  2%|▏         | 6431/400000 [00:00<00:42, 9309.03it/s]  2%|▏         | 7340/400000 [00:00<00:42, 9230.07it/s]  2%|▏         | 8227/400000 [00:00<00:42, 9117.99it/s]  2%|▏         | 9129/400000 [00:01<00:43, 9086.31it/s]  3%|▎         | 10015/400000 [00:01<00:43, 8906.12it/s]  3%|▎         | 10891/400000 [00:01<00:43, 8859.82it/s]  3%|▎         | 11767/400000 [00:01<00:45, 8586.80it/s]  3%|▎         | 12631/400000 [00:01<00:45, 8600.14it/s]  3%|▎         | 13514/400000 [00:01<00:44, 8665.74it/s]  4%|▎         | 14394/400000 [00:01<00:44, 8705.33it/s]  4%|▍         | 15284/400000 [00:01<00:43, 8762.07it/s]  4%|▍         | 16175/400000 [00:01<00:43, 8803.44it/s]  4%|▍         | 17055/400000 [00:01<00:43, 8746.08it/s]  4%|▍         | 17949/400000 [00:02<00:43, 8801.02it/s]  5%|▍         | 18829/400000 [00:02<00:44, 8654.19it/s]  5%|▍         | 19722/400000 [00:02<00:43, 8733.09it/s]  5%|▌         | 20596/400000 [00:02<00:43, 8626.87it/s]  5%|▌         | 21535/400000 [00:02<00:42, 8841.30it/s]  6%|▌         | 22444/400000 [00:02<00:42, 8914.26it/s]  6%|▌         | 23337/400000 [00:02<00:42, 8886.81it/s]  6%|▌         | 24264/400000 [00:02<00:41, 8997.26it/s]  6%|▋         | 25191/400000 [00:02<00:41, 9075.82it/s]  7%|▋         | 26100/400000 [00:02<00:41, 9052.79it/s]  7%|▋         | 27006/400000 [00:03<00:41, 9029.76it/s]  7%|▋         | 27910/400000 [00:03<00:42, 8806.51it/s]  7%|▋         | 28793/400000 [00:03<00:42, 8720.88it/s]  7%|▋         | 29674/400000 [00:03<00:42, 8746.21it/s]  8%|▊         | 30556/400000 [00:03<00:42, 8766.73it/s]  8%|▊         | 31452/400000 [00:03<00:41, 8823.26it/s]  8%|▊         | 32339/400000 [00:03<00:41, 8836.28it/s]  8%|▊         | 33229/400000 [00:03<00:41, 8854.70it/s]  9%|▊         | 34123/400000 [00:03<00:41, 8879.71it/s]  9%|▉         | 35028/400000 [00:03<00:40, 8928.27it/s]  9%|▉         | 35922/400000 [00:04<00:40, 8923.43it/s]  9%|▉         | 36815/400000 [00:04<00:40, 8913.33it/s]  9%|▉         | 37707/400000 [00:04<00:40, 8864.34it/s] 10%|▉         | 38594/400000 [00:04<00:41, 8769.84it/s] 10%|▉         | 39473/400000 [00:04<00:41, 8773.49it/s] 10%|█         | 40358/400000 [00:04<00:40, 8795.04it/s] 10%|█         | 41272/400000 [00:04<00:40, 8894.20it/s] 11%|█         | 42162/400000 [00:04<00:40, 8865.05it/s] 11%|█         | 43049/400000 [00:04<00:40, 8864.01it/s] 11%|█         | 43940/400000 [00:04<00:40, 8876.87it/s] 11%|█         | 44849/400000 [00:05<00:39, 8937.30it/s] 11%|█▏        | 45768/400000 [00:05<00:39, 9007.78it/s] 12%|█▏        | 46730/400000 [00:05<00:38, 9178.71it/s] 12%|█▏        | 47649/400000 [00:05<00:39, 8959.77it/s] 12%|█▏        | 48572/400000 [00:05<00:38, 9037.43it/s] 12%|█▏        | 49512/400000 [00:05<00:38, 9142.59it/s] 13%|█▎        | 50469/400000 [00:05<00:37, 9264.78it/s] 13%|█▎        | 51397/400000 [00:05<00:38, 9136.27it/s] 13%|█▎        | 52312/400000 [00:05<00:38, 9028.68it/s] 13%|█▎        | 53217/400000 [00:05<00:38, 9005.62it/s] 14%|█▎        | 54119/400000 [00:06<00:38, 8939.97it/s] 14%|█▍        | 55037/400000 [00:06<00:38, 9009.94it/s] 14%|█▍        | 55940/400000 [00:06<00:38, 9015.20it/s] 14%|█▍        | 56842/400000 [00:06<00:38, 8977.79it/s] 14%|█▍        | 57741/400000 [00:06<00:39, 8761.94it/s] 15%|█▍        | 58619/400000 [00:06<00:39, 8704.78it/s] 15%|█▍        | 59503/400000 [00:06<00:38, 8744.25it/s] 15%|█▌        | 60391/400000 [00:06<00:38, 8783.74it/s] 15%|█▌        | 61270/400000 [00:06<00:38, 8767.04it/s] 16%|█▌        | 62148/400000 [00:06<00:38, 8758.72it/s] 16%|█▌        | 63025/400000 [00:07<00:39, 8493.57it/s] 16%|█▌        | 63880/400000 [00:07<00:39, 8507.88it/s] 16%|█▌        | 64733/400000 [00:07<00:39, 8433.54it/s] 16%|█▋        | 65622/400000 [00:07<00:39, 8563.25it/s] 17%|█▋        | 66510/400000 [00:07<00:38, 8653.64it/s] 17%|█▋        | 67385/400000 [00:07<00:38, 8681.48it/s] 17%|█▋        | 68254/400000 [00:07<00:38, 8663.91it/s] 17%|█▋        | 69162/400000 [00:07<00:37, 8782.12it/s] 18%|█▊        | 70080/400000 [00:07<00:37, 8896.26it/s] 18%|█▊        | 70971/400000 [00:08<00:37, 8860.31it/s] 18%|█▊        | 71858/400000 [00:08<00:37, 8831.39it/s] 18%|█▊        | 72742/400000 [00:08<00:37, 8707.87it/s] 18%|█▊        | 73614/400000 [00:08<00:38, 8547.45it/s] 19%|█▊        | 74488/400000 [00:08<00:37, 8602.44it/s] 19%|█▉        | 75372/400000 [00:08<00:37, 8670.86it/s] 19%|█▉        | 76252/400000 [00:08<00:37, 8707.09it/s] 19%|█▉        | 77124/400000 [00:08<00:37, 8708.21it/s] 19%|█▉        | 77996/400000 [00:08<00:37, 8658.81it/s] 20%|█▉        | 78883/400000 [00:08<00:36, 8720.80it/s] 20%|█▉        | 79769/400000 [00:09<00:36, 8761.11it/s] 20%|██        | 80646/400000 [00:09<00:36, 8725.00it/s] 20%|██        | 81522/400000 [00:09<00:36, 8734.92it/s] 21%|██        | 82396/400000 [00:09<00:36, 8703.22it/s] 21%|██        | 83281/400000 [00:09<00:36, 8744.57it/s] 21%|██        | 84176/400000 [00:09<00:35, 8803.21it/s] 21%|██▏       | 85061/400000 [00:09<00:35, 8815.83it/s] 21%|██▏       | 85943/400000 [00:09<00:35, 8797.43it/s] 22%|██▏       | 86823/400000 [00:09<00:35, 8709.51it/s] 22%|██▏       | 87695/400000 [00:09<00:35, 8698.14it/s] 22%|██▏       | 88579/400000 [00:10<00:35, 8738.94it/s] 22%|██▏       | 89467/400000 [00:10<00:35, 8778.08it/s] 23%|██▎       | 90358/400000 [00:10<00:35, 8816.78it/s] 23%|██▎       | 91240/400000 [00:10<00:35, 8707.12it/s] 23%|██▎       | 92129/400000 [00:10<00:35, 8760.59it/s] 23%|██▎       | 93011/400000 [00:10<00:34, 8777.80it/s] 23%|██▎       | 93890/400000 [00:10<00:34, 8779.95it/s] 24%|██▎       | 94769/400000 [00:10<00:34, 8779.51it/s] 24%|██▍       | 95648/400000 [00:10<00:34, 8699.22it/s] 24%|██▍       | 96519/400000 [00:10<00:34, 8678.08it/s] 24%|██▍       | 97388/400000 [00:11<00:35, 8506.43it/s] 25%|██▍       | 98258/400000 [00:11<00:35, 8562.99it/s] 25%|██▍       | 99138/400000 [00:11<00:34, 8629.96it/s] 25%|██▌       | 100002/400000 [00:11<00:35, 8532.19it/s] 25%|██▌       | 100870/400000 [00:11<00:34, 8573.29it/s] 25%|██▌       | 101739/400000 [00:11<00:34, 8605.59it/s] 26%|██▌       | 102600/400000 [00:11<00:34, 8565.19it/s] 26%|██▌       | 103481/400000 [00:11<00:34, 8635.68it/s] 26%|██▌       | 104381/400000 [00:11<00:33, 8740.35it/s] 26%|██▋       | 105256/400000 [00:11<00:34, 8667.30it/s] 27%|██▋       | 106124/400000 [00:12<00:34, 8549.88it/s] 27%|██▋       | 106989/400000 [00:12<00:34, 8579.46it/s] 27%|██▋       | 107855/400000 [00:12<00:33, 8602.30it/s] 27%|██▋       | 108716/400000 [00:12<00:34, 8555.75it/s] 27%|██▋       | 109572/400000 [00:12<00:34, 8536.56it/s] 28%|██▊       | 110427/400000 [00:12<00:33, 8539.08it/s] 28%|██▊       | 111282/400000 [00:12<00:34, 8431.78it/s] 28%|██▊       | 112165/400000 [00:12<00:33, 8544.87it/s] 28%|██▊       | 113080/400000 [00:12<00:32, 8716.31it/s] 29%|██▊       | 114026/400000 [00:12<00:32, 8924.19it/s] 29%|██▊       | 114924/400000 [00:13<00:31, 8940.55it/s] 29%|██▉       | 115848/400000 [00:13<00:31, 9027.47it/s] 29%|██▉       | 116753/400000 [00:13<00:31, 8878.80it/s] 29%|██▉       | 117643/400000 [00:13<00:32, 8740.76it/s] 30%|██▉       | 118524/400000 [00:13<00:32, 8759.59it/s] 30%|██▉       | 119402/400000 [00:13<00:32, 8757.53it/s] 30%|███       | 120331/400000 [00:13<00:31, 8909.55it/s] 30%|███       | 121247/400000 [00:13<00:31, 8981.02it/s] 31%|███       | 122175/400000 [00:13<00:30, 9065.68it/s] 31%|███       | 123109/400000 [00:13<00:30, 9143.54it/s] 31%|███       | 124025/400000 [00:14<00:30, 9104.24it/s] 31%|███       | 124936/400000 [00:14<00:30, 8942.53it/s] 31%|███▏      | 125832/400000 [00:14<00:32, 8546.51it/s] 32%|███▏      | 126692/400000 [00:14<00:32, 8499.96it/s] 32%|███▏      | 127546/400000 [00:14<00:32, 8500.63it/s] 32%|███▏      | 128399/400000 [00:14<00:32, 8423.88it/s] 32%|███▏      | 129246/400000 [00:14<00:32, 8435.96it/s] 33%|███▎      | 130091/400000 [00:14<00:32, 8432.71it/s] 33%|███▎      | 130936/400000 [00:14<00:31, 8422.41it/s] 33%|███▎      | 131786/400000 [00:15<00:31, 8444.46it/s] 33%|███▎      | 132661/400000 [00:15<00:31, 8531.85it/s] 33%|███▎      | 133539/400000 [00:15<00:30, 8602.32it/s] 34%|███▎      | 134400/400000 [00:15<00:31, 8442.20it/s] 34%|███▍      | 135246/400000 [00:15<00:31, 8375.12it/s] 34%|███▍      | 136085/400000 [00:15<00:31, 8288.08it/s] 34%|███▍      | 136948/400000 [00:15<00:31, 8385.40it/s] 34%|███▍      | 137862/400000 [00:15<00:30, 8596.40it/s] 35%|███▍      | 138724/400000 [00:15<00:30, 8601.04it/s] 35%|███▍      | 139598/400000 [00:15<00:30, 8642.00it/s] 35%|███▌      | 140498/400000 [00:16<00:29, 8745.74it/s] 35%|███▌      | 141397/400000 [00:16<00:29, 8816.08it/s] 36%|███▌      | 142280/400000 [00:16<00:29, 8741.18it/s] 36%|███▌      | 143155/400000 [00:16<00:30, 8516.39it/s] 36%|███▌      | 144009/400000 [00:16<00:30, 8439.17it/s] 36%|███▌      | 144869/400000 [00:16<00:30, 8485.31it/s] 36%|███▋      | 145746/400000 [00:16<00:29, 8567.05it/s] 37%|███▋      | 146604/400000 [00:16<00:29, 8568.49it/s] 37%|███▋      | 147494/400000 [00:16<00:29, 8662.60it/s] 37%|███▋      | 148383/400000 [00:16<00:28, 8729.10it/s] 37%|███▋      | 149286/400000 [00:17<00:28, 8815.23it/s] 38%|███▊      | 150203/400000 [00:17<00:28, 8916.96it/s] 38%|███▊      | 151096/400000 [00:17<00:27, 8897.79it/s] 38%|███▊      | 151987/400000 [00:17<00:28, 8814.28it/s] 38%|███▊      | 152869/400000 [00:17<00:28, 8764.19it/s] 38%|███▊      | 153758/400000 [00:17<00:27, 8801.52it/s] 39%|███▊      | 154640/400000 [00:17<00:27, 8805.90it/s] 39%|███▉      | 155580/400000 [00:17<00:27, 8974.82it/s] 39%|███▉      | 156522/400000 [00:17<00:26, 9101.82it/s] 39%|███▉      | 157434/400000 [00:17<00:26, 9050.11it/s] 40%|███▉      | 158340/400000 [00:18<00:27, 8907.67it/s] 40%|███▉      | 159269/400000 [00:18<00:26, 9017.80it/s] 40%|████      | 160172/400000 [00:18<00:27, 8774.05it/s] 40%|████      | 161052/400000 [00:18<00:27, 8761.17it/s] 40%|████      | 161930/400000 [00:18<00:27, 8610.63it/s] 41%|████      | 162793/400000 [00:18<00:27, 8611.99it/s] 41%|████      | 163663/400000 [00:18<00:27, 8636.99it/s] 41%|████      | 164557/400000 [00:18<00:26, 8725.67it/s] 41%|████▏     | 165462/400000 [00:18<00:26, 8817.96it/s] 42%|████▏     | 166345/400000 [00:18<00:26, 8796.29it/s] 42%|████▏     | 167235/400000 [00:19<00:26, 8826.23it/s] 42%|████▏     | 168119/400000 [00:19<00:26, 8806.02it/s] 42%|████▏     | 169000/400000 [00:19<00:26, 8752.93it/s] 42%|████▏     | 169876/400000 [00:19<00:26, 8739.26it/s] 43%|████▎     | 170751/400000 [00:19<00:26, 8699.48it/s] 43%|████▎     | 171622/400000 [00:19<00:26, 8636.86it/s] 43%|████▎     | 172494/400000 [00:19<00:26, 8661.36it/s] 43%|████▎     | 173383/400000 [00:19<00:25, 8727.17it/s] 44%|████▎     | 174256/400000 [00:19<00:25, 8691.85it/s] 44%|████▍     | 175126/400000 [00:19<00:26, 8462.19it/s] 44%|████▍     | 176029/400000 [00:20<00:25, 8622.99it/s] 44%|████▍     | 176949/400000 [00:20<00:25, 8786.87it/s] 44%|████▍     | 177830/400000 [00:20<00:25, 8671.89it/s] 45%|████▍     | 178718/400000 [00:20<00:25, 8731.17it/s] 45%|████▍     | 179598/400000 [00:20<00:25, 8751.13it/s] 45%|████▌     | 180475/400000 [00:20<00:25, 8668.04it/s] 45%|████▌     | 181473/400000 [00:20<00:24, 9023.62it/s] 46%|████▌     | 182382/400000 [00:20<00:24, 9043.14it/s] 46%|████▌     | 183293/400000 [00:20<00:23, 9061.80it/s] 46%|████▌     | 184243/400000 [00:20<00:23, 9188.47it/s] 46%|████▋     | 185170/400000 [00:21<00:23, 9210.42it/s] 47%|████▋     | 186113/400000 [00:21<00:23, 9274.31it/s] 47%|████▋     | 187042/400000 [00:21<00:23, 9130.43it/s] 47%|████▋     | 187957/400000 [00:21<00:23, 9102.42it/s] 47%|████▋     | 188869/400000 [00:21<00:23, 8978.35it/s] 47%|████▋     | 189768/400000 [00:21<00:23, 8769.68it/s] 48%|████▊     | 190676/400000 [00:21<00:23, 8859.66it/s] 48%|████▊     | 191583/400000 [00:21<00:23, 8919.10it/s] 48%|████▊     | 192499/400000 [00:21<00:23, 8989.75it/s] 48%|████▊     | 193400/400000 [00:22<00:22, 8995.58it/s] 49%|████▊     | 194301/400000 [00:22<00:22, 8972.78it/s] 49%|████▉     | 195199/400000 [00:22<00:22, 8941.35it/s] 49%|████▉     | 196094/400000 [00:22<00:23, 8796.33it/s] 49%|████▉     | 196990/400000 [00:22<00:22, 8842.77it/s] 49%|████▉     | 197896/400000 [00:22<00:22, 8905.16it/s] 50%|████▉     | 198806/400000 [00:22<00:22, 8961.56it/s] 50%|████▉     | 199748/400000 [00:22<00:22, 9091.50it/s] 50%|█████     | 200668/400000 [00:22<00:21, 9121.14it/s] 50%|█████     | 201581/400000 [00:22<00:21, 9066.71it/s] 51%|█████     | 202489/400000 [00:23<00:21, 9064.02it/s] 51%|█████     | 203396/400000 [00:23<00:22, 8893.74it/s] 51%|█████     | 204314/400000 [00:23<00:21, 8976.69it/s] 51%|█████▏    | 205213/400000 [00:23<00:21, 8945.90it/s] 52%|█████▏    | 206109/400000 [00:23<00:21, 8884.37it/s] 52%|█████▏    | 206998/400000 [00:23<00:21, 8847.77it/s] 52%|█████▏    | 207892/400000 [00:23<00:21, 8874.64it/s] 52%|█████▏    | 208795/400000 [00:23<00:21, 8918.69it/s] 52%|█████▏    | 209689/400000 [00:23<00:21, 8922.59it/s] 53%|█████▎    | 210615/400000 [00:23<00:20, 9018.37it/s] 53%|█████▎    | 211545/400000 [00:24<00:20, 9099.36it/s] 53%|█████▎    | 212467/400000 [00:24<00:20, 9134.22it/s] 53%|█████▎    | 213406/400000 [00:24<00:20, 9207.41it/s] 54%|█████▎    | 214328/400000 [00:24<00:20, 8954.69it/s] 54%|█████▍    | 215249/400000 [00:24<00:20, 9027.37it/s] 54%|█████▍    | 216201/400000 [00:24<00:20, 9167.91it/s] 54%|█████▍    | 217128/400000 [00:24<00:19, 9197.28it/s] 55%|█████▍    | 218049/400000 [00:24<00:19, 9163.83it/s] 55%|█████▍    | 218967/400000 [00:24<00:20, 8887.65it/s] 55%|█████▍    | 219888/400000 [00:24<00:20, 8981.78it/s] 55%|█████▌    | 220798/400000 [00:25<00:19, 9016.18it/s] 55%|█████▌    | 221701/400000 [00:25<00:19, 8987.64it/s] 56%|█████▌    | 222621/400000 [00:25<00:19, 9048.98it/s] 56%|█████▌    | 223527/400000 [00:25<00:19, 8979.19it/s] 56%|█████▌    | 224433/400000 [00:25<00:19, 9002.94it/s] 56%|█████▋    | 225334/400000 [00:25<00:19, 8945.22it/s] 57%|█████▋    | 226229/400000 [00:25<00:20, 8585.76it/s] 57%|█████▋    | 227135/400000 [00:25<00:19, 8721.83it/s] 57%|█████▋    | 228038/400000 [00:25<00:19, 8811.10it/s] 57%|█████▋    | 228953/400000 [00:25<00:19, 8907.57it/s] 57%|█████▋    | 229958/400000 [00:26<00:18, 9220.47it/s] 58%|█████▊    | 230898/400000 [00:26<00:18, 9273.12it/s] 58%|█████▊    | 231870/400000 [00:26<00:17, 9402.43it/s] 58%|█████▊    | 232813/400000 [00:26<00:17, 9375.35it/s] 58%|█████▊    | 233753/400000 [00:26<00:17, 9372.35it/s] 59%|█████▊    | 234692/400000 [00:26<00:18, 9172.63it/s] 59%|█████▉    | 235612/400000 [00:26<00:18, 9116.86it/s] 59%|█████▉    | 236529/400000 [00:26<00:17, 9130.58it/s] 59%|█████▉    | 237444/400000 [00:26<00:17, 9049.11it/s] 60%|█████▉    | 238350/400000 [00:26<00:17, 8982.71it/s] 60%|█████▉    | 239249/400000 [00:27<00:18, 8916.72it/s] 60%|██████    | 240142/400000 [00:27<00:18, 8868.15it/s] 60%|██████    | 241030/400000 [00:27<00:18, 8797.32it/s] 60%|██████    | 241922/400000 [00:27<00:17, 8832.14it/s] 61%|██████    | 242806/400000 [00:27<00:17, 8807.94it/s] 61%|██████    | 243688/400000 [00:27<00:17, 8803.49it/s] 61%|██████    | 244569/400000 [00:27<00:18, 8530.85it/s] 61%|██████▏   | 245434/400000 [00:27<00:18, 8564.26it/s] 62%|██████▏   | 246344/400000 [00:27<00:17, 8716.97it/s] 62%|██████▏   | 247218/400000 [00:28<00:17, 8657.11it/s] 62%|██████▏   | 248097/400000 [00:28<00:17, 8695.50it/s] 62%|██████▏   | 248976/400000 [00:28<00:17, 8722.45it/s] 62%|██████▏   | 249849/400000 [00:28<00:17, 8684.64it/s] 63%|██████▎   | 250724/400000 [00:28<00:17, 8701.78it/s] 63%|██████▎   | 251650/400000 [00:28<00:16, 8861.73it/s] 63%|██████▎   | 252548/400000 [00:28<00:16, 8895.37it/s] 63%|██████▎   | 253439/400000 [00:28<00:16, 8823.22it/s] 64%|██████▎   | 254384/400000 [00:28<00:16, 9001.40it/s] 64%|██████▍   | 255348/400000 [00:28<00:15, 9182.44it/s] 64%|██████▍   | 256300/400000 [00:29<00:15, 9278.30it/s] 64%|██████▍   | 257230/400000 [00:29<00:15, 9083.51it/s] 65%|██████▍   | 258141/400000 [00:29<00:15, 8988.32it/s] 65%|██████▍   | 259042/400000 [00:29<00:15, 8945.27it/s] 65%|██████▍   | 259938/400000 [00:29<00:16, 8730.31it/s] 65%|██████▌   | 260851/400000 [00:29<00:15, 8846.36it/s] 65%|██████▌   | 261738/400000 [00:29<00:15, 8848.17it/s] 66%|██████▌   | 262628/400000 [00:29<00:15, 8863.02it/s] 66%|██████▌   | 263552/400000 [00:29<00:15, 8972.43it/s] 66%|██████▌   | 264461/400000 [00:29<00:15, 9004.69it/s] 66%|██████▋   | 265396/400000 [00:30<00:14, 9104.59it/s] 67%|██████▋   | 266314/400000 [00:30<00:14, 9126.70it/s] 67%|██████▋   | 267228/400000 [00:30<00:14, 9106.21it/s] 67%|██████▋   | 268140/400000 [00:30<00:14, 9107.79it/s] 67%|██████▋   | 269064/400000 [00:30<00:14, 9145.16it/s] 67%|██████▋   | 269981/400000 [00:30<00:14, 9152.35it/s] 68%|██████▊   | 270901/400000 [00:30<00:14, 9164.58it/s] 68%|██████▊   | 271818/400000 [00:30<00:14, 8936.48it/s] 68%|██████▊   | 272713/400000 [00:30<00:14, 8771.27it/s] 68%|██████▊   | 273627/400000 [00:30<00:14, 8876.02it/s] 69%|██████▊   | 274559/400000 [00:31<00:13, 9003.75it/s] 69%|██████▉   | 275463/400000 [00:31<00:13, 9013.92it/s] 69%|██████▉   | 276366/400000 [00:31<00:13, 8985.96it/s] 69%|██████▉   | 277275/400000 [00:31<00:13, 9016.37it/s] 70%|██████▉   | 278178/400000 [00:31<00:13, 8978.08it/s] 70%|██████▉   | 279077/400000 [00:31<00:13, 8927.28it/s] 70%|██████▉   | 279971/400000 [00:31<00:13, 8748.39it/s] 70%|███████   | 280847/400000 [00:31<00:13, 8751.18it/s] 70%|███████   | 281729/400000 [00:31<00:13, 8769.66it/s] 71%|███████   | 282646/400000 [00:31<00:13, 8885.55it/s] 71%|███████   | 283536/400000 [00:32<00:13, 8888.02it/s] 71%|███████   | 284426/400000 [00:32<00:13, 8877.27it/s] 71%|███████▏  | 285325/400000 [00:32<00:12, 8909.15it/s] 72%|███████▏  | 286272/400000 [00:32<00:12, 9067.13it/s] 72%|███████▏  | 287202/400000 [00:32<00:12, 9133.33it/s] 72%|███████▏  | 288121/400000 [00:32<00:12, 9148.74it/s] 72%|███████▏  | 289043/400000 [00:32<00:12, 9167.90it/s] 72%|███████▏  | 289961/400000 [00:32<00:12, 9062.08it/s] 73%|███████▎  | 290946/400000 [00:32<00:11, 9282.52it/s] 73%|███████▎  | 291947/400000 [00:32<00:11, 9487.91it/s] 73%|███████▎  | 292915/400000 [00:33<00:11, 9542.78it/s] 73%|███████▎  | 293871/400000 [00:33<00:11, 9345.68it/s] 74%|███████▎  | 294808/400000 [00:33<00:11, 9135.83it/s] 74%|███████▍  | 295725/400000 [00:33<00:11, 9118.40it/s] 74%|███████▍  | 296639/400000 [00:33<00:11, 9110.00it/s] 74%|███████▍  | 297584/400000 [00:33<00:11, 9208.76it/s] 75%|███████▍  | 298506/400000 [00:33<00:11, 9182.30it/s] 75%|███████▍  | 299426/400000 [00:33<00:11, 9049.74it/s] 75%|███████▌  | 300332/400000 [00:33<00:11, 9000.08it/s] 75%|███████▌  | 301247/400000 [00:34<00:10, 9041.35it/s] 76%|███████▌  | 302188/400000 [00:34<00:10, 9148.59it/s] 76%|███████▌  | 303120/400000 [00:34<00:10, 9198.43it/s] 76%|███████▌  | 304041/400000 [00:34<00:10, 9126.46it/s] 76%|███████▋  | 305032/400000 [00:34<00:10, 9345.91it/s] 76%|███████▋  | 305969/400000 [00:34<00:10, 9272.04it/s] 77%|███████▋  | 306898/400000 [00:34<00:10, 9246.24it/s] 77%|███████▋  | 307845/400000 [00:34<00:09, 9309.95it/s] 77%|███████▋  | 308777/400000 [00:34<00:09, 9291.00it/s] 77%|███████▋  | 309708/400000 [00:34<00:09, 9296.51it/s] 78%|███████▊  | 310639/400000 [00:35<00:09, 9179.69it/s] 78%|███████▊  | 311558/400000 [00:35<00:09, 9032.35it/s] 78%|███████▊  | 312463/400000 [00:35<00:09, 9003.42it/s] 78%|███████▊  | 313365/400000 [00:35<00:09, 8878.81it/s] 79%|███████▊  | 314284/400000 [00:35<00:09, 8968.02it/s] 79%|███████▉  | 315210/400000 [00:35<00:09, 9052.76it/s] 79%|███████▉  | 316118/400000 [00:35<00:09, 9058.29it/s] 79%|███████▉  | 317040/400000 [00:35<00:09, 9103.87it/s] 79%|███████▉  | 317951/400000 [00:35<00:09, 8883.66it/s] 80%|███████▉  | 318841/400000 [00:35<00:09, 8713.45it/s] 80%|███████▉  | 319741/400000 [00:36<00:09, 8795.00it/s] 80%|████████  | 320666/400000 [00:36<00:08, 8924.49it/s] 80%|████████  | 321566/400000 [00:36<00:08, 8945.32it/s] 81%|████████  | 322462/400000 [00:36<00:08, 8859.14it/s] 81%|████████  | 323351/400000 [00:36<00:08, 8866.42it/s] 81%|████████  | 324270/400000 [00:36<00:08, 8959.14it/s] 81%|████████▏ | 325167/400000 [00:36<00:08, 8764.52it/s] 82%|████████▏ | 326051/400000 [00:36<00:08, 8784.11it/s] 82%|████████▏ | 326931/400000 [00:36<00:08, 8766.70it/s] 82%|████████▏ | 327825/400000 [00:36<00:08, 8816.35it/s] 82%|████████▏ | 328713/400000 [00:37<00:08, 8833.42it/s] 82%|████████▏ | 329614/400000 [00:37<00:07, 8884.84it/s] 83%|████████▎ | 330503/400000 [00:37<00:07, 8803.70it/s] 83%|████████▎ | 331384/400000 [00:37<00:08, 8532.78it/s] 83%|████████▎ | 332289/400000 [00:37<00:07, 8680.80it/s] 83%|████████▎ | 333227/400000 [00:37<00:07, 8877.10it/s] 84%|████████▎ | 334142/400000 [00:37<00:07, 8957.06it/s] 84%|████████▍ | 335040/400000 [00:37<00:07, 8873.61it/s] 84%|████████▍ | 335929/400000 [00:37<00:07, 8832.65it/s] 84%|████████▍ | 336872/400000 [00:37<00:07, 9002.20it/s] 84%|████████▍ | 337803/400000 [00:38<00:06, 9090.33it/s] 85%|████████▍ | 338723/400000 [00:38<00:06, 9121.55it/s] 85%|████████▍ | 339644/400000 [00:38<00:06, 9145.17it/s] 85%|████████▌ | 340560/400000 [00:38<00:06, 8992.54it/s] 85%|████████▌ | 341461/400000 [00:38<00:06, 8809.54it/s] 86%|████████▌ | 342347/400000 [00:38<00:06, 8824.02it/s] 86%|████████▌ | 343246/400000 [00:38<00:06, 8871.19it/s] 86%|████████▌ | 344195/400000 [00:38<00:06, 9047.23it/s] 86%|████████▋ | 345102/400000 [00:38<00:06, 8988.48it/s] 87%|████████▋ | 346038/400000 [00:38<00:05, 9095.87it/s] 87%|████████▋ | 346986/400000 [00:39<00:05, 9207.77it/s] 87%|████████▋ | 347912/400000 [00:39<00:05, 9220.96it/s] 87%|████████▋ | 348835/400000 [00:39<00:05, 9185.52it/s] 87%|████████▋ | 349755/400000 [00:39<00:05, 9102.07it/s] 88%|████████▊ | 350666/400000 [00:39<00:05, 8989.56it/s] 88%|████████▊ | 351566/400000 [00:39<00:05, 8971.64it/s] 88%|████████▊ | 352464/400000 [00:39<00:05, 8941.66it/s] 88%|████████▊ | 353359/400000 [00:39<00:05, 8924.85it/s] 89%|████████▊ | 354252/400000 [00:39<00:05, 8860.25it/s] 89%|████████▉ | 355139/400000 [00:40<00:05, 8653.43it/s] 89%|████████▉ | 356027/400000 [00:40<00:05, 8719.31it/s] 89%|████████▉ | 356920/400000 [00:40<00:04, 8779.56it/s] 89%|████████▉ | 357808/400000 [00:40<00:04, 8808.78it/s] 90%|████████▉ | 358690/400000 [00:40<00:04, 8465.64it/s] 90%|████████▉ | 359540/400000 [00:40<00:04, 8465.84it/s] 90%|█████████ | 360429/400000 [00:40<00:04, 8588.10it/s] 90%|█████████ | 361290/400000 [00:40<00:04, 8517.57it/s] 91%|█████████ | 362201/400000 [00:40<00:04, 8685.53it/s] 91%|█████████ | 363072/400000 [00:40<00:04, 8538.19it/s] 91%|█████████ | 363928/400000 [00:41<00:04, 8468.54it/s] 91%|█████████ | 364793/400000 [00:41<00:04, 8520.56it/s] 91%|█████████▏| 365705/400000 [00:41<00:03, 8691.20it/s] 92%|█████████▏| 366638/400000 [00:41<00:03, 8871.85it/s] 92%|█████████▏| 367549/400000 [00:41<00:03, 8939.70it/s] 92%|█████████▏| 368508/400000 [00:41<00:03, 9122.72it/s] 92%|█████████▏| 369437/400000 [00:41<00:03, 9170.46it/s] 93%|█████████▎| 370365/400000 [00:41<00:03, 9200.44it/s] 93%|█████████▎| 371288/400000 [00:41<00:03, 9206.69it/s] 93%|█████████▎| 372210/400000 [00:41<00:03, 9154.18it/s] 93%|█████████▎| 373127/400000 [00:42<00:02, 9083.12it/s] 94%|█████████▎| 374072/400000 [00:42<00:02, 9187.55it/s] 94%|█████████▍| 375002/400000 [00:42<00:02, 9219.11it/s] 94%|█████████▍| 375927/400000 [00:42<00:02, 9226.04it/s] 94%|█████████▍| 376850/400000 [00:42<00:02, 9147.45it/s] 94%|█████████▍| 377766/400000 [00:42<00:02, 9114.82it/s] 95%|█████████▍| 378713/400000 [00:42<00:02, 9218.34it/s] 95%|█████████▍| 379678/400000 [00:42<00:02, 9343.03it/s] 95%|█████████▌| 380614/400000 [00:42<00:02, 9285.02it/s] 95%|█████████▌| 381546/400000 [00:42<00:01, 9294.98it/s] 96%|█████████▌| 382476/400000 [00:43<00:01, 9091.35it/s] 96%|█████████▌| 383387/400000 [00:43<00:01, 8934.52it/s] 96%|█████████▌| 384282/400000 [00:43<00:01, 8790.19it/s] 96%|█████████▋| 385182/400000 [00:43<00:01, 8850.07it/s] 97%|█████████▋| 386087/400000 [00:43<00:01, 8906.41it/s] 97%|█████████▋| 386995/400000 [00:43<00:01, 8957.01it/s] 97%|█████████▋| 387892/400000 [00:43<00:01, 8871.16it/s] 97%|█████████▋| 388802/400000 [00:43<00:01, 8935.76it/s] 97%|█████████▋| 389716/400000 [00:43<00:01, 8993.90it/s] 98%|█████████▊| 390617/400000 [00:43<00:01, 8997.18it/s] 98%|█████████▊| 391518/400000 [00:44<00:00, 8955.21it/s] 98%|█████████▊| 392437/400000 [00:44<00:00, 9023.10it/s] 98%|█████████▊| 393340/400000 [00:44<00:00, 9023.14it/s] 99%|█████████▊| 394251/400000 [00:44<00:00, 9047.04it/s] 99%|█████████▉| 395164/400000 [00:44<00:00, 9068.65it/s] 99%|█████████▉| 396071/400000 [00:44<00:00, 9003.53it/s] 99%|█████████▉| 396992/400000 [00:44<00:00, 9063.85it/s] 99%|█████████▉| 397906/400000 [00:44<00:00, 9084.07it/s]100%|█████████▉| 398850/400000 [00:44<00:00, 9187.18it/s]100%|█████████▉| 399770/400000 [00:44<00:00, 9177.66it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8887.67it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f02ea23dd68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01125273658482302 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011206286208685426 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15911 out of table with 15804 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15911 out of table with 15804 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-12 23:23:01.298231: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 23:23:01.301710: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-12 23:23:01.302435: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b8de1bb7e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 23:23:01.302451: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f02937bba20> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3983 - accuracy: 0.5175 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4775 - accuracy: 0.5123
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5171 - accuracy: 0.5098
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5072 - accuracy: 0.5104
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5107 - accuracy: 0.5102
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5001 - accuracy: 0.5109
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5823 - accuracy: 0.5055
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6498 - accuracy: 0.5011
11000/25000 [============>.................] - ETA: 3s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 3s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7032 - accuracy: 0.4976
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7148 - accuracy: 0.4969
15000/25000 [=================>............] - ETA: 2s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6915 - accuracy: 0.4984
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7000 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7007 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 7s 265us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f024a7b98d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0257dcc160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7061 - crf_viterbi_accuracy: 0.1733 - val_loss: 1.5776 - val_crf_viterbi_accuracy: 0.1733

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
